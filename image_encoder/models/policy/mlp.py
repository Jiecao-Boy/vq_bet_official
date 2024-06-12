import torch
import einops
from .base import AbstractPolicy
import torch.nn as nn
from typing import Tuple, Union, Optional, Dict, List
import torch.nn.functional as F
from ..mlp import MLP


class MLPPolicy(AbstractPolicy):
    def __init__(
        self,
        rep_dim: int,
        views: int,
        hidden_dim: int,
        output_dim: int,
        hidden_depth: int,
    ):
        super(MLPPolicy, self).__init__()

        self.fc = MLP(
            views * rep_dim,
            hidden_dim,
            output_dim,
            hidden_depth,
            batchnorm=True,
        )

    def forward(
        self,
        obs_seq: torch.Tensor,
        target_action_seq: Optional[torch.Tensor] = None,
        padding_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        # obs_seq: N T V E
        obs_seq = obs_seq.float()
        N, T, V, E = obs_seq.shape
        obs_seq = einops.rearrange(obs_seq, "N T V E -> (N T) (V E)")
        predicted_action_seq = self.fc(obs_seq).view(N, T, -1)

        loss = None
        loss_components = None
        if target_action_seq is not None:
            target_action_seq = target_action_seq.float()
            loss = F.mse_loss(predicted_action_seq, target_action_seq)
            loss_components = {"loss": loss}
        return predicted_action_seq, loss, loss_components

    def get_optimizer(
        self, lr: float, weight_decay: float, betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )
        return optimizer
