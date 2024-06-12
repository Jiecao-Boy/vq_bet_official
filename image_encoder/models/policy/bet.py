import torch
import einops
from .base import AbstractPolicy
import torch.nn as nn
from typing import Any, Tuple, Optional, Dict, Union, List
import torch.nn.functional as F
import tqdm
from ..transformer_encoder import TransformerEncoder, TransformerEncoderConfig
import logging

import accelerate

GENERATOR_SEED_FIXED = 123456789


class BehaviorTransformer(AbstractPolicy):
    def __init__(
        self,
        obs_dim: int,
        views: int,
        act_dim: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float = 0.0,
        bias: bool = True,
        num_extra_predicted_actions: Optional[int] = None,
        goal_dim: int = 0,
        trainable_obs_padding: bool = False,
        n_clusters: int = 32,
        kmeans_fit_steps: int = 500,
        kmeans_iters: int = 50,
        offset_loss_multiplier: float = 1.0e3,
        offset_distance_metric: str = "L2",
        gamma: float = 2.0,
    ):
        super().__init__()
        self._obs_dim = views * obs_dim
        self._act_dim = act_dim
        self._goal_dim = goal_dim
        self._num_extra_predicted_actions = num_extra_predicted_actions or 0
        # Gradient-free, all zeros if we don't want to train this.
        self._obs_padding = nn.Parameter(
            trainable_obs_padding * torch.randn(obs_dim),
            requires_grad=trainable_obs_padding,
        )
        self.GOAL_SPEC = ["concat", "stack", "unconditional"]

        if goal_dim <= 0:
            self._cbet_method = "unconditional"
        elif obs_dim == goal_dim:
            self._cbet_method = "concat"
        else:
            self._cbet_method = "stack"
        config = TransformerEncoderConfig(
            block_size=block_size + self._num_extra_predicted_actions,
            input_dim=self._obs_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            output_dim=n_embd,
            dropout=dropout,
            bias=bias,
        )
        self._gpt_model = TransformerEncoder(config=config)
        # For now, we assume the number of clusters is given.
        assert n_clusters > 0 and kmeans_fit_steps > 0
        self._K = n_clusters
        self._kmeans_fit_steps = kmeans_fit_steps
        self._clustering_algo = KMeansDiscretizer(
            num_bins=n_clusters, kmeans_iters=kmeans_iters
        )
        self._current_steps = 0
        self._map_to_cbet_preds = nn.Linear(
            self._gpt_model.config.output_dim,
            (act_dim + 1) * n_clusters,
        )
        self._collected_actions = []
        self._have_fit_kmeans = False
        self._offset_loss_multiplier = offset_loss_multiplier
        # Placeholder for the cluster centers.
        generator = torch.Generator()
        generator.manual_seed(GENERATOR_SEED_FIXED)
        self.register_buffer(
            "_cluster_centers",
            torch.randn(
                (n_clusters, act_dim), generator=generator, dtype=torch.float32
            ),
        )
        self._criterion = FocalLoss(gamma=gamma, reduction="none")
        self._offset_criterion = (
            nn.MSELoss(reduction="none")
            if offset_distance_metric == "L2"
            else nn.L1Loss(reduction="none")
        )
        self._accelerator = accelerate.Accelerator()

    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: Optional[torch.Tensor] = None,
        goal_seq: Optional[torch.Tensor] = None,
        padding_seq: Optional[torch.Tensor] = None,
        predict_with_offset: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if action_seq is not None:
            action_seq = action_seq.float()
            # NOTE: this assumes there's no action padding
        else:
            # TODO: padding_seq None causes bugs in fit_kmeans
            padding_seq = None
        if self._current_steps == 0:
            self._cluster_centers = self._cluster_centers.to(obs_seq.device)
        if self._current_steps < self._kmeans_fit_steps and action_seq is not None:
            self._current_steps += 1
            self._fit_kmeans(obs_seq, action_seq, goal_seq, padding_seq)
        return self._predict(
            obs_seq,
            action_seq,
            goal_seq,
            padding_seq,
            predict_with_offset=predict_with_offset,
        )

    def _fit_kmeans(
        self,
        obs_seq: torch.Tensor,
        action_seq: Optional[torch.Tensor],
        goal_seq: Optional[torch.Tensor],
        padding_seq: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert self._current_steps <= self._kmeans_fit_steps
        if self._current_steps == 1:
            self._cluster_centers = self._cluster_centers.to(action_seq.device)

        all_action_seq = self._accelerator.gather(action_seq)
        all_padding_seq = self._accelerator.gather(padding_seq)
        self._collected_actions.append(
            all_action_seq[torch.logical_not(all_padding_seq)]
        )
        if self._current_steps == self._kmeans_fit_steps:
            logging.info("Fitting KMeans")
            self._clustering_algo.fit(
                torch.cat(self._collected_actions, dim=0).view(-1, self._act_dim)
            )
            self._have_fit_kmeans = True
            self._cluster_centers = self._clustering_algo.bin_centers.float().to(
                action_seq.device
            )

    def _predict(
        self,
        obs_seq: torch.Tensor,
        action_seq: Optional[torch.Tensor],
        goal_seq: Optional[torch.Tensor],
        is_padded_action_seq: Optional[torch.Tensor],
        predict_with_offset: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        batch_size, obs_T, num_view, _ = obs_seq.shape
        _, action_T, _ = (
            action_seq.shape if action_seq is not None else (None, None, None)
        )
        # Take the one that is not None.
        actions_to_predict = action_T or obs_T
        # Now, figure out if we should pad the obs seq.
        if obs_T < actions_to_predict:
            # We need to pad the obs seq.
            pad_size = actions_to_predict - obs_T
            # Here we assume that there are 4 views
            padded_obs_seq = torch.cat(
                [
                    obs_seq,
                    einops.repeat(
                        self._obs_padding,
                        "D -> N T V D",
                        N=batch_size,
                        T=pad_size,
                        V=num_view,
                    ),
                ],
                dim=1,
            )
        else:
            padded_obs_seq = obs_seq

        # Assume dimensions are N T D for N sequences of T timesteps with dimension D.
        if self._cbet_method == "unconditional":
            gpt_input = padded_obs_seq
        elif self._cbet_method == "concat":
            gpt_input = torch.cat([goal_seq, padded_obs_seq], dim=1)
        elif self._cbet_method == "stack":
            gpt_input = torch.cat([goal_seq, padded_obs_seq], dim=-1)
        else:
            raise NotImplementedError
        gpt_input = einops.rearrange(gpt_input, "N T V E -> N T (V E)")
        gpt_output = self._gpt_model(gpt_input)
        if self._cbet_method == "concat":
            # Chop off the goal encodings.
            gpt_output = gpt_output[:, goal_seq.size(1) :, :]
        cbet_preds = self._map_to_cbet_preds(gpt_output)
        cbet_logits, cbet_offsets = torch.split(
            cbet_preds, [self._K, self._K * self._act_dim], dim=-1
        )
        cbet_offsets = einops.rearrange(cbet_offsets, "N T (K A) -> N T K A", K=self._K)

        cbet_probs = torch.softmax(cbet_logits, dim=-1)
        N, T, choices = cbet_probs.shape
        # Sample from the multinomial distribution, one per row.
        sampled_centers = einops.rearrange(
            torch.multinomial(cbet_probs.view(-1, choices), num_samples=1),
            "(N T) 1 -> N T 1",
            N=N,
        )
        flattened_cbet_offsets = einops.rearrange(cbet_offsets, "N T K A -> (N T) K A")
        sampled_offsets = flattened_cbet_offsets[
            torch.arange(flattened_cbet_offsets.shape[0]), sampled_centers.flatten()
        ].view(N, T, self._act_dim)
        centers = self._cluster_centers[sampled_centers.flatten()].view(
            N, T, self._act_dim
        )
        a_hat = centers
        if predict_with_offset:
            a_hat += sampled_offsets
        if action_seq is None:
            return a_hat, None, {}
        # We are in training, so figure out the loss for the actions.
        # First, we need to find the closest cluster center for each action.
        action_bins = self._find_closest_cluster(action_seq)
        true_offsets = action_seq - self._cluster_centers[action_bins]
        predicted_offsets = flattened_cbet_offsets[
            torch.arange(flattened_cbet_offsets.shape[0]), action_bins.flatten()
        ].view(N, T, self._act_dim)
        # Now we can compute the loss.
        offset_loss = self._offset_criterion(predicted_offsets, true_offsets)
        cbet_loss = self._criterion(
            einops.rearrange(cbet_logits, "N T D -> (N T) D"),
            einops.rearrange(action_bins, "N T -> (N T)"),
        )
        # Now, use the padding mask to mask out the loss.
        if is_padded_action_seq is not None:
            cbet_loss *= ~is_padded_action_seq.view(-1)
            offset_loss *= ~is_padded_action_seq.unsqueeze(-1)
        cbet_loss, offset_loss = cbet_loss.mean(), offset_loss.mean()
        loss = cbet_loss + self._offset_loss_multiplier * offset_loss
        action_mse = F.mse_loss(a_hat, action_seq, reduction="none")
        action_l1 = F.l1_loss(a_hat, action_seq, reduction="none")
        norm = torch.norm(action_seq, p=2, dim=-1, keepdim=True) + 1e-9
        normalized_mse = (action_mse / norm).mean()
        if self._current_steps < self._kmeans_fit_steps:
            loss = loss.detach() + (loss * 0.0)

        loss_dict = {
            "classification_loss": cbet_loss,
            "offset_loss": offset_loss,
            "loss": loss,
            "L2_loss": action_mse.mean(),
            "L2_loss_normalized": normalized_mse.mean(),
            "L1_loss": action_l1.mean(),
        }
        return a_hat, loss, loss_dict

    def _find_closest_cluster(self, action_seq: torch.Tensor) -> torch.Tensor:
        N, T, _ = action_seq.shape
        flattened_actions = einops.rearrange(action_seq, "N T A -> (N T) A")
        cluster_center_distance = torch.sum(
            (flattened_actions[:, None, :] - self._cluster_centers[None, :, :]) ** 2,
            dim=2,
        )  # (N T) K A -> (N T) K
        closest_cluster_center = torch.argmin(cluster_center_distance, dim=1)  # (N T)
        discretized_action = einops.rearrange(
            closest_cluster_center, "(N T) -> N T", N=N, T=T
        )
        return discretized_action

    def get_optimizer(self, weight_decay, lr, betas):
        optimizer = self._gpt_model.configure_optimizers(
            weight_decay=weight_decay,
            lr=lr,
            betas=betas,
            device_type=next(self._gpt_model.parameters()).device,
        )
        optimizer.add_param_group({"params": self._map_to_cbet_preds.parameters()})
        return optimizer


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if reduction not in ("mean", "sum", "none"):
            raise NotImplementedError
        self.reduction = reduction

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class KMeansDiscretizer:
    """
    Simplified and modified version of KMeans algorithm from sklearn.
    We initialize this with a fixed seed to ensure that on each GPU we come up with the same
    clusters.
    """

    def __init__(
        self,
        num_bins: int = 100,
        kmeans_iters: int = 50,
    ):
        super().__init__()
        self.n_bins = num_bins
        self.kmeans_iters = kmeans_iters

    def fit(self, input_actions: torch.Tensor) -> None:
        self.bin_centers = KMeansDiscretizer._kmeans(
            input_actions, ncluster=self.n_bins, niter=self.kmeans_iters
        )

    @classmethod
    def _kmeans(cls, x: torch.Tensor, ncluster: int = 512, niter: int = 50):
        """
        Simple k-means clustering algorithm adapted from Karpathy's minGPT libary
        https://github.com/karpathy/minGPT/blob/master/play_image.ipynb
        """
        N, D = x.size()
        generator = torch.Generator()
        generator.manual_seed(GENERATOR_SEED_FIXED)

        c = x[
            torch.randperm(N, generator=generator)[:ncluster]
        ]  # init clusters at random, with a fixed seed

        pbar = tqdm.trange(niter, ncols=80)
        pbar.set_description("K-means clustering")
        for i in pbar:
            # assign all pixels to the closest codebook element
            a = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
            # move each codebook element to be the mean of the pixels that assigned to it
            c = torch.stack([x[a == k].mean(0) for k in range(ncluster)])
            # re-assign any poorly positioned codebook elements
            nanix = torch.any(torch.isnan(c), dim=1)
            ndead = nanix.sum().item()
            if ndead:
                tqdm.tqdm.write(
                    "done step %d/%d, re-initialized %d dead clusters"
                    % (i + 1, niter, ndead)
                )
            c[nanix] = x[
                torch.randperm(N, generator=generator)[:ndead]
            ]  # re-init dead clusters
        return c
