import torch
import einops
import numpy as np
import torch.nn as nn
from .base import AbstractSSL
from accelerate import Accelerator
from typing import Tuple, Dict, Optional
from ..transformer_encoder import TransformerEncoder, TransformerEncoderConfig
from ..ema import EMA


# https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py#L239
def off_diag(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def off_diag_cov_loss(x: torch.Tensor) -> torch.Tensor:
    cov = torch.cov(einops.rearrange(x, "... E -> E (...)"))
    return off_diag(cov).square().mean()


accelerator = Accelerator()


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init._no_grad_trunc_normal_(m.weight, mean=0, std=0.02, a=-2, b=2)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):
    def __init__(
        self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        assert student_output.size() == teacher_output.size()
        assert student_output.dim() == 2
        student_out = student_output / self.student_temp

        # teacher centering and sharpening
        teacher_out = torch.nn.functional.softmax(
            (teacher_output - self.center) / self.teacher_temp, dim=-1
        )
        teacher_out = teacher_out.detach()
        loss = torch.sum(
            -teacher_out * torch.nn.functional.log_softmax(student_out, dim=-1), dim=-1
        ).mean()
        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = accelerator.reduce(batch_center, reduction="sum")
        batch_center = batch_center / (len(teacher_output) * accelerator.num_processes)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


class MultiviewDynamicsSSL(AbstractSSL):
    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
        window_size: int,
        feature_dim: int,
        projection_dim: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float = 0.0,
        variance_reg_coef: float = 0.0,
        covariance_reg_coef: float = 0.04,
        dynamics_loss_coef: float = 1.0,
        projection_consistency_coef: float = 0.0,
        ema_beta: Optional[float] = None,  # None for SimSiam; float for EMA encoder
        beta_scheduling: bool = False,
        projector_use_ema: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        clip_grad_norm: Optional[float] = None,
        separate_single_views: bool = False,
        separate_covariance_reg_loss: bool = False,
        dynamics_loss_type: str = "cosine",  # cosine or dino
        dino_head_cfg: Optional[Dict] = None,
        mask_frames: Optional[int] = None,
    ):
        self.encoder = encoder
        self.projector = projector
        forward_dynamics_cfg = TransformerEncoderConfig(
            block_size=window_size,
            input_dim=feature_dim + projection_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            output_dim=feature_dim,
        )
        self.forward_dynamics = TransformerEncoder(forward_dynamics_cfg)
        self.forward_dynamics_optimizer = self.forward_dynamics.configure_optimizers(
            weight_decay=weight_decay,
            lr=lr,
            betas=betas,
        )
        self.mask_frames = mask_frames
        if self.mask_frames is not None:
            # assume at least 1 frame is observed
            assert (
                self.mask_frames < window_size
            ), "mask_frames should be less than window_size"
            self.placeholder_latent = torch.nn.Parameter(torch.randn(1, feature_dim))
        self.forward_dynamics, self.forward_dynamics_optimizer = accelerator.prepare(
            self.forward_dynamics,
            self.forward_dynamics_optimizer,
        )
        self.clip_grad_norm = clip_grad_norm
        self.variance_reg_coef = variance_reg_coef
        self.covariance_reg_coef = covariance_reg_coef
        self.dynamics_loss_coef = dynamics_loss_coef
        self.projection_consistency_coef = projection_consistency_coef
        self.ema_beta = ema_beta
        self.beta_scheduling = beta_scheduling
        self.projector_use_ema = projector_use_ema
        if self.ema_beta is not None:
            self.ema_encoder = EMA(self.encoder, self.ema_beta)
            if self.projector_use_ema:
                self.ema_projector = EMA(self.projector, self.ema_beta)
        self.separate_single_views = separate_single_views  # inverse/forward dynamics on each view separately for dynamics loss
        self.separate_covariance_reg_loss = (
            separate_covariance_reg_loss  # separate covariance reg loss for each view
        )
        self.dynamics_loss_type = dynamics_loss_type
        if self.dynamics_loss_type == "dino":
            self.student_head = DINOHead(in_dim=feature_dim, **dino_head_cfg)
            if self.ema_beta is not None:
                self.teacher_head = DINOHead(in_dim=feature_dim, **dino_head_cfg)
                # TODO: should we copy all params to teacher?
                self.teacher_head = EMA(self.teacher_head, self.ema_beta, copy=False)
                self.student_head, self.teacher_head = accelerator.prepare(
                    self.student_head, self.teacher_head
                )
            else:
                self.student_head = accelerator.prepare(self.student_head)
                self.teacher_head = self.student_head
            self.dino_loss = DINOLoss(out_dim=dino_head_cfg["out_dim"])
            self.dino_loss = accelerator.prepare(self.dino_loss)

        if self.separate_single_views and self.projection_consistency_coef > 0:
            raise ValueError(
                "Projection consistency loss requires different views, but separate_single_views is True."
            )

    def forward(
        self,
        obs: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor],]:
        obs_enc = self.encoder(obs)
        if self.ema_beta is not None:
            obs_target = self.ema_encoder(obs)  # use EMA encoder as target
            if self.projector_use_ema:
                obs_proj = self.ema_projector(obs_enc)
            else:
                obs_proj = self.projector(obs_enc)
        else:
            obs_target = obs_enc  # use SimSiam target
            obs_proj = self.projector(obs_enc)

        variance_loss = self._variance_reg_loss(obs_enc)
        covariance_loss = self._covariance_reg_loss(
            obs_enc, self.separate_covariance_reg_loss
        )
        dynamics_loss, dynamics_loss_components = self._forward_dyn_loss(
            obs_enc, obs_proj, obs_target, self.separate_single_views
        )
        (
            proj_consistency_loss,
            proj_consistency_loss_components,
        ) = self._proj_consistency_loss(obs_proj)
        total_loss = (
            dynamics_loss + variance_loss + covariance_loss + proj_consistency_loss
        )
        loss_components = {
            "total_loss": total_loss,
            **dynamics_loss_components,
            "variance_loss": variance_loss,
            "covariance_loss": covariance_loss,
            **proj_consistency_loss_components,
        }
        return obs_enc, obs_proj, total_loss, loss_components

    def _forward_dyn_loss(
        self,
        obs_enc: torch.Tensor,
        obs_proj: torch.Tensor,
        obs_target: torch.Tensor,
        separate_single_views: bool = False,
    ):
        V = obs_proj.shape[2]  # number of views
        total = torch.zeros(1, device=obs_enc.device)
        loss_components = {}
        if separate_single_views:
            for i in range(V):
                loss = self._forward_dyn_loss_one_pair(
                    obs_enc, obs_proj, obs_target, i, i
                )
                loss *= self.dynamics_loss_coef / V
                total += loss
                loss_components[f"dynamics_loss_{i}_{i}"] = loss
        else:
            total_view_pairs = V * (V - 1)  # w/ order
            for i in range(V):
                for j in range(V):
                    if i == j:
                        continue
                    loss = self._forward_dyn_loss_one_pair(
                        obs_enc, obs_proj, obs_target, i, j
                    )
                    loss *= self.dynamics_loss_coef / total_view_pairs
                    total += loss
                    loss_components[f"dynamics_loss_{i}_{j}"] = loss
        loss_components["dynamics_loss_total"] = total
        if self.ema_beta:
            loss_components["ema_beta"] = torch.Tensor([self.ema_encoder.beta]).to(
                obs_enc.device
            )
        return total, loss_components

    def _forward_dyn_loss_one_pair(
        self,
        obs_enc: torch.Tensor,
        obs_proj: torch.Tensor,
        obs_target: torch.Tensor,
        i: int,
        j: int,
    ):
        obs_enc_masked = obs_enc[:, :-1, j].clone()
        if self.mask_frames is not None:
            obs_enc_masked[:, -self.mask_frames :] = self.placeholder_latent.repeat(
                obs_enc_masked.shape[0], self.mask_frames, 1
            )
        forward_dyn_input = torch.cat([obs_enc_masked, obs_proj[:, 1:, i]], dim=-1)
        obs_enc_pred = self.forward_dynamics(forward_dyn_input)  # (N, T-1, E)
        if self.dynamics_loss_type == "cosine":
            loss = (
                1
                - torch.nn.functional.cosine_similarity(
                    obs_enc_pred, obs_target[:, 1:, j].detach(), dim=-1
                ).mean()
            )
        elif self.dynamics_loss_type == "dino":
            student_output = self.student_head(obs_enc_pred)  # (N, T-1, C)
            teacher_output = self.teacher_head(
                obs_target[:, 1:, j].detach()
            )  # (N, T-1, C); head grad on teacher will be detached in DINOLoss
            student_output = einops.rearrange(student_output, "N T C -> (N T) C")
            teacher_output = einops.rearrange(teacher_output, "N T C -> (N T) C")
            loss = self.dino_loss(student_output, teacher_output)
        return loss

    def _variance_reg_loss(self, obs_enc: torch.Tensor):
        obs_enc_std = obs_enc.std(dim=[0, 1, 2]).mean()
        return torch.nn.functional.relu(1 - obs_enc_std) * self.variance_reg_coef

    def _covariance_reg_loss(self, obs_enc: torch.Tensor, separate: bool = False):
        if separate:
            total_loss = 0
            for v in range(obs_enc.size(2)):  # obs_enc has shape (N, T, V, E)
                obs_enc_view = obs_enc[:, :, v, :]
                view_loss = off_diag_cov_loss(obs_enc_view)
                total_loss += view_loss
            total_loss = total_loss / obs_enc.shape[2]
        else:
            total_loss = off_diag_cov_loss(obs_enc)
        return total_loss * self.covariance_reg_coef

    def _proj_consistency_loss(self, obs_proj: torch.Tensor):
        total = torch.zeros(1, device=obs_proj.device)
        loss_components = {}
        V = obs_proj.shape[2]  # number of views
        total_view_pairs = V * (V - 1) / 2  # w/o order
        for i in range(V - 1):
            for j in range(i + 1, V):
                loss = (
                    torch.nn.functional.mse_loss(obs_proj[:, :, i], obs_proj[:, :, j])
                    * self.projection_consistency_coef
                    / total_view_pairs
                )
                total += loss
                loss_components[f"projection_consistency_loss_{i}_{j}"] = loss
        loss_components["projection_consistency_loss_total"] = total
        return total, loss_components

    def adjust_beta(self, epoch: int, max_epoch: int):
        if (self.ema_beta is None) or not self.beta_scheduling:
            return
        self.ema_encoder.beta = 1.0 - 0.5 * (
            1.0 + np.cos(np.pi * epoch / max_epoch)
        ) * (1.0 - self.ema_beta)
        if self.projector_use_ema:
            self.ema_projector.beta = 1.0 - 0.5 * (
                1.0 + np.cos(np.pi * epoch / max_epoch)
            ) * (1.0 - self.ema_beta)

    def step(self):
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.forward_dynamics.parameters(), self.clip_grad_norm
            )
        self.forward_dynamics_optimizer.step()
        self.forward_dynamics_optimizer.zero_grad()
        if self.ema_beta is not None:
            self.ema_encoder.step(self.encoder)
            if self.projector_use_ema:
                self.ema_projector.step(self.projector)
