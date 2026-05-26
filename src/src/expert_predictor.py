from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
import torch.nn as nn


class ExpertPredictor(ABC):
    """专家预测器抽象基类"""

    @abstractmethod
    def predict(
        self, hidden_states: torch.Tensor, model: nn.Module, next_layer_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError


class GatePredictor(ExpertPredictor):
    """使用下一层 gate 网络预测下一层活跃专家"""

    def predict(
        self, hidden_states: torch.Tensor, model: nn.Module, next_layer_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if next_layer_idx >= len(model.layers):
            return None
        next_layer = model.layers[next_layer_idx]
        predicted_experts, routing_weights = next_layer.mlp.gate(hidden_states)
        return predicted_experts, routing_weights
