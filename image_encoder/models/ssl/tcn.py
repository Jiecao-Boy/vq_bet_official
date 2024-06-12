import torch
import einops
import torch.nn as nn
from .base import AbstractSSL
from typing import Tuple, Dict


class TCNSSL(AbstractSSL):
    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
        contrastive_loss_margin: float = 0.2,
        triplet_loss_type: str = "mse",  # or "cosine"
        single_view: bool = False,
    ):
        assert triplet_loss_type in ["mse", "cosine"]
        self.encoder = encoder
        self.projector = projector
        self.contrastive_loss_margin = contrastive_loss_margin
        self.triplet_loss_type = triplet_loss_type
        self.single_view = single_view

    def project_each_view(self, obs_enc: torch.Tensor):
        N, T, V = obs_enc.shape[:3]
        obs_proj = []
        for i in range(V):
            this_view_proj = self.projector(obs_enc[:, :, i])  # (N, T, Z)
            obs_proj.append(this_view_proj)
        return torch.stack(obs_proj, dim=2)  # (N, T, V, Z)

    def forward(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor],]:
        # must call encoder on cat(obs, goal)
        # so that batchnorm update doesn't break backprop
        N_obs, T_obs, V_obs = obs.shape[:3]
        x = torch.cat([obs, goal], dim=1)
        enc = self.encoder(x)

        obs_enc = enc[:, :T_obs]
        goal_enc = enc[:, T_obs:]

        obs_proj = self.projector(obs_enc)
        if self.single_view:
            loss, loss_components = self._single_view_loss(obs_enc, goal_enc)
        else:
            loss, loss_components = self._multiview_loss(obs_enc, goal_enc)
        return obs_enc, obs_proj, loss, loss_components

    def _triplet_loss(self, anchor, positives, negatives, margin):
        if self.triplet_loss_type == "mse":
            positive_loss = torch.nn.functional.mse_loss(anchor, positives)
            negative_loss = torch.nn.functional.mse_loss(anchor, negatives)
        else:
            positive_loss = -torch.nn.functional.cosine_similarity(
                anchor, positives, dim=-1
            ).mean()
            negative_loss = -torch.nn.functional.cosine_similarity(
                anchor, negatives, dim=-1
            ).mean()
        loss = torch.relu(margin + positive_loss - negative_loss)
        return loss, {
            "positive_loss": positive_loss,
            "negative_loss": negative_loss,
            "loss": loss,
        }

    def _multiview_loss(self, obs_enc: torch.Tensor, goal_enc: torch.Tensor):
        V = obs_enc.shape[2]  # number of views
        total = torch.zeros(1, device=obs_enc.device, requires_grad=True)
        loss_components = {}
        total_view_pairs = V * (V - 1)  # w/ order
        for i in range(V):
            for j in range(V):
                if i == j:
                    continue

                if torch.rand(1) > 0.5:
                    obs_enc, goal_enc = goal_enc, obs_enc
                anchor = obs_enc[:, :, i, :].clone()
                positives = obs_enc[:, :, j, :].clone()
                negatives = goal_enc[:, :, i, :].clone()

                loss, this_view_loss_components = self._triplet_loss(
                    anchor, positives, negatives, self.contrastive_loss_margin
                )
                this_view_loss_components = {
                    f"{k}_{i}_{j}": v / total_view_pairs
                    for k, v in loss_components.items()
                }
                loss_components.update(this_view_loss_components)
                total = total + loss / total_view_pairs
        loss_components["total_loss"] = total
        return total, loss_components

    def _single_view_loss(self, obs_enc: torch.Tensor, goal_enc: torch.Tensor):
        N, T, V, E = obs_enc.shape
        T_goal = goal_enc.shape[1]
        assert T_goal == T / 2
        total = torch.zeros(1, device=obs_enc.device, requires_grad=True)
        loss_components = {}
        for i in range(V):
            cur_obs_enc = obs_enc[:, :, i, :]
            cur_goal_enc = goal_enc[:, :, i, :]

            all_idx = torch.randperm(T)
            anchor_idx = all_idx[:T_goal]
            positive_idx = all_idx[T_goal:]
            negative_idx = torch.randperm(T_goal)
            anchor = cur_obs_enc[:, anchor_idx, :]
            positives = cur_obs_enc[:, positive_idx, :]
            negatives = cur_goal_enc[:, negative_idx, :]

            loss, this_view_loss_components = self._triplet_loss(
                anchor, positives, negatives, self.contrastive_loss_margin
            )
            this_view_loss_components = {
                f"{k}_{i}": v / V for k, v in loss_components.items()
            }
            loss_components.update(this_view_loss_components)
            total = total + loss / V
        loss_components["total_loss"] = total
        return total, loss_components

    def step(self):
        pass

    def adjust_beta(self, epoch: int, main_epoch: int):
        pass
