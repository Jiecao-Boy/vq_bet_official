import torch
import torch.nn as nn


class NullProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.placeholder_param = nn.Parameter(torch.zeros(1))
        assert output_dim == 0

    def forward(self, obs_enc: torch.Tensor):
        N, T, V = obs_enc.shape[:3]
        return torch.zeros(N, T, V, 0, device=obs_enc.device)

    def configure_optimizers(self, weight_decay, lr, betas):
        return torch.optim.AdamW(
            self.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
