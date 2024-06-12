import torch
import einops
from ..mlp import MLP
from .base import AbstractPolicy
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class MLPClassifier(AbstractPolicy):
    def __init__(
        self,
        rep_dim: int,
        views: int,
        hidden_dim: int,
        output_dim: int,
        hidden_depth: int,
    ):
        super(MLPClassifier, self).__init__()

        self.fc = MLP(
            views * rep_dim,
            hidden_dim,
            output_dim,
            hidden_depth,
            batchnorm=True,  # TODO: do we need syncbatchnorm for this?
        )

    def forward(
        self,
        obs_seq: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        padding_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        # obs_seq: N T V E
        obs_seq = obs_seq.float()
        N, T, V, E = obs_seq.shape
        obs_seq = einops.rearrange(obs_seq, "N T V E -> (N T) (V E)")
        logits = self.fc(obs_seq).view(N, T, -1)

        loss = None
        loss_components = None
        if target is not None:
            logits_flat = einops.rearrange(logits, "N T C -> (N T) C")
            target_flat = einops.rearrange(target, "N T -> (N T)")
            loss = F.cross_entropy(logits_flat, target_flat)
            loss_components = {"loss": loss}
        return logits, loss, loss_components

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
