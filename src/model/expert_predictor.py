from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn

from model.expert_types import ExpertDemand, build_future_demands


class ExpertPredictor(ABC):
    """专家预测器抽象基类"""

    @abstractmethod
    def predict(
        self,
        hidden_states: torch.Tensor,
        model: nn.Module,
        current_layer: int,
        lookahead: int = 1,
    ) -> List[ExpertDemand]:
        raise NotImplementedError


class GatePredictor(ExpertPredictor):
    """使用下一层 gate 网络预测下一层活跃专家"""

    def predict(
        self,
        hidden_states: torch.Tensor,
        model: nn.Module,
        current_layer: int,
        lookahead: int = 1,
    ) -> List[ExpertDemand]:
        next_layer_idx = current_layer + lookahead
        if next_layer_idx >= len(model.layers):
            return []
        next_layer = model.layers[next_layer_idx]
        predicted_experts, routing_weights = next_layer.mlp.gate(hidden_states)
        return build_future_demands(next_layer_idx, predicted_experts, routing_weights)
