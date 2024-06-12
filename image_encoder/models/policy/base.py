import abc

import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, List, Dict, Any
from utils import SaveModule, TrainWithLogger


class AbstractPolicy(nn.Module, TrainWithLogger):
    @abc.abstractmethod
    def forward(
        self,
        obs_seq: torch.Tensor,
        target_action_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, torch.Tensor]:
        """
        Given a sequence of observations (N T V E):
            N: batch size
            T: sequence length
            V: number of views
            E: embedding size
        and an optional sequence of actions (N T A):
            N: batch size
            T: sequence length
            A: action dimension
        return the predicted action, and if target_action_seq is not None, a loss for training, and a dictionary for each loss component.
        If target_action_seq is None, return None for loss and loss_components.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer(
        self,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        raise NotImplementedError
