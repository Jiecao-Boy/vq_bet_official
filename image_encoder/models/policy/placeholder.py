import torch
from .base import AbstractPolicy
from typing import Tuple, Optional, Dict


class PlaceholderPolicy(AbstractPolicy):
    def __init__(
        self,
        output_dim: int,
    ):
        super(PlaceholderPolicy, self).__init__()
        # placeholder parameter for optimizer
        self.param = torch.nn.Parameter(torch.tensor(0.0))
        self.output_dim = output_dim

    def forward(
        self,
        obs_seq: torch.Tensor,
        target_action_seq: Optional[torch.Tensor] = None,
        padding_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        # obs_seq: N T V E
        N, T, V, E = obs_seq.shape
        placeholder_loss = torch.tensor(0.0).to(obs_seq.device)
        placeholder_loss.requires_grad = True
        action = torch.zeros(N, T, self.output_dim).to(obs_seq.device)
        return action, placeholder_loss, {"loss": placeholder_loss}

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
