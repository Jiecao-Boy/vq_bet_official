import torch
import einops
import logging
from typing import Optional
from .base import AbstractPolicy
from accelerate import Accelerator


class VINN(AbstractPolicy):
    def __init__(
        self,
        act_dim: int,
        lwr_param: int = 5,
        weighted: bool = True,
        reset_on_train: bool = False,
        only_take_first_frame: bool = False,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.obses_train = []
        self.actions_train = []
        self._accelerator = Accelerator()
        self.obses = torch.Tensor()
        self.actions = torch.Tensor()
        self.k = lwr_param
        self.weighted = weighted
        self.placeholder_param = torch.nn.Parameter(torch.Tensor(1, 1))
        self.reset_on_train = reset_on_train
        self.should_reset = False
        self.only_take_first_frame = only_take_first_frame

    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: Optional[torch.Tensor] = None,
        padding_seq: Optional[torch.Tensor] = None,
    ):
        # this placeholder loss is just so that torch garbage collects upstream
        placeholder_loss = torch.mean(obs_seq) * self.placeholder_param
        obs_seq = obs_seq.to(self._accelerator.device)

        if self.should_reset and (action_seq is not None):
            self._reset()

        if self.training:
            obs_seq = self._accelerator.gather(obs_seq)
            if self.only_take_first_frame:
                obs = obs_seq[:, 0, :, :].clone()
                obs = einops.rearrange(obs, "N V E -> N (V E)")
            else:
                obs = einops.rearrange(obs_seq, "N T V E -> (N T) (V E)")

            assert action_seq is not None, "action_seq must be provided during training"
            action_seq = action_seq.to(self._accelerator.device)
            if padding_seq is not None:
                padding_seq = padding_seq.to(self._accelerator.device)
                # padding seq is bool shaped N T; use it to mask out invalid actions
                act = []
                for idx in range(len(action_seq)):
                    act.append(action_seq[idx, ~padding_seq[idx]])
                action_seq = torch.stack(act, dim=0)
            action_seq = self._accelerator.gather(action_seq)
            if self.only_take_first_frame:
                act = action_seq[:, 0, :].clone()
            else:
                act = einops.rearrange(act, "N T A -> (N T) A").clone()

            if self.obses.numel() == 0:
                self.obses = einops.rearrange(obs, "N E -> 1 N E")
                self.actions = act
            else:
                self.obses_train.append(obs)
                self.actions_train.append(act)
        else:
            if len(self.obses_train) > 0:
                self.obses_train = torch.cat(self.obses_train, dim=0)
                self.obses_train = einops.rearrange(self.obses_train, "N E -> 1 N E")
                self.obses = torch.cat([self.obses, self.obses_train], dim=1)
                self.obses_train = []
            if len(self.actions_train) > 0:
                self.actions_train = torch.cat(self.actions_train, dim=0)
                self.actions = torch.cat([self.actions, self.actions_train], dim=0)
                self.actions_train = []

        # predict one action for each obs
        pred_act = self._predict(obs_seq)
        if action_seq is not None:
            mse = torch.mean((pred_act - action_seq) ** 2)
        else:
            mse = torch.Tensor([0])

        return pred_act, placeholder_loss, {"mse": mse}

    def train(self, mode: bool = True):
        if mode and self.reset_on_train:
            self.should_reset = True
        return super().train(mode)

    def _reset(self):
        del self.obses
        del self.actions
        del self.obses_train
        del self.actions_train
        self.obses = torch.Tensor().to(self._accelerator.device)
        self.actions = torch.Tensor().to(self._accelerator.device)
        self.obses_train = []
        self.actions_train = []
        self.should_reset = False

    def _predict(self, obs_seq: torch.Tensor):
        if self.actions.numel() == 0:
            # N x T x act_dim
            return torch.zeros(*obs_seq.shape[:2], self.act_dim, device=obs_seq.device)
        obs = einops.rearrange(obs_seq, "N T V E -> (N T) 1 (V E)")
        dist = torch.sum((obs - self.obses) ** 2, dim=-1)
        nearest, idx = torch.sort(dist, dim=-1)
        k_nearest_idx = idx[:, : self.k]
        k_nearest_idx = idx[:, : self.k]
        k_nearest_act = self.actions[torch.flatten(k_nearest_idx)]
        k_nearest_act = einops.rearrange(k_nearest_act, "(N K) A -> N K A", K=self.k)
        k_nearest_dist = nearest[:, : self.k]
        k_nearest_dist = nearest[:, : self.k]
        if not self.weighted:
            k_nearest_dist = torch.ones_like(k_nearest_dist)
        weighted_act = self._get_weighted_actions(k_nearest_act, k_nearest_dist)
        actions = einops.rearrange(weighted_act, "(N T) A -> N T A", N=obs_seq.shape[0])
        return actions

    def _get_weighted_actions(self, actions: torch.Tensor, dist: torch.Tensor):
        # actions: N K A; dist: N K
        weights = torch.exp(-dist) + 1e-9
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True))
        weights = einops.rearrange(weights, "N K -> N K 1")
        result = einops.reduce(actions * weights, "N K A -> N A", "sum")
        return result

    def get_optimizer(self, weight_decay, lr, betas):
        # placeholder
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-3,
            weight_decay=1e-5,
        )
        return optimizer
