from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import torch


@dataclass(frozen=True)
class ExpertKey:
    """专家唯一标识，由层号和专家编号组成"""
    layer: int
    expert_id: int


@dataclass
class ExpertDemand:
    """专家需求，描述某个专家需要处理的 token 数量和重要性得分"""
    key: ExpertKey
    token_count: int
    score: float = 1.0
    source: str = "current"


@dataclass
class ExpertAssignment:
    """专家分配，记录某个专家对应的 token 索引和路由权重"""
    expert_id: int
    token_indices: torch.Tensor
    routing_weights: torch.Tensor


@dataclass
class PlacementSnapshot:
    """专家驻留状态快照，供调度器只读决策"""
    gpu_resident: Set[Tuple[int, int]] = field(default_factory=set)
    placeholder_resident: Set[Tuple[int, int]] = field(default_factory=set)
    loading: Set[Tuple[int, int]] = field(default_factory=set)
    cpu_resident: Set[Tuple[int, int]] = field(default_factory=set)
    ssd_resident: Set[Tuple[int, int]] = field(default_factory=set)
    free_placeholders: int = 0

    def is_on_gpu(self, layer: int, expert_id: int) -> bool:
        """判断专家是否在 GPU 上（静态驻留或 placeholder）"""
        key = (layer, expert_id)
        return key in self.gpu_resident or key in self.placeholder_resident


@dataclass
class ExpertLayerRequest:
    """调度器输入，包含当前层信息、阶段、当前/未来需求和 token 分配"""
    layer: int
    phase: str
    current: List[ExpertDemand]
    future: List[ExpertDemand]
    assignments: Dict[int, ExpertAssignment]


@dataclass
class ExpertSchedule:
    """调度器输出，包含 CPU/GPU/preload/evict 决策"""
    cpu: List[ExpertDemand] = field(default_factory=list)
    gpu: List[ExpertDemand] = field(default_factory=list)
    preload: List[ExpertDemand] = field(default_factory=list)
    evict: List[ExpertDemand] = field(default_factory=list)
    reason: str = ""

    @property
    def cpu_expert_ids(self) -> List[int]:
        """获取 CPU 执行的专家 ID 列表"""
        return [d.key.expert_id for d in self.cpu]

    @property
    def gpu_expert_ids(self) -> List[int]:
        """获取 GPU 执行的专家 ID 列表"""
        return [d.key.expert_id for d in self.gpu]

    @property
    def preload_expert_ids(self) -> List[int]:
        """获取预加载的专家 ID 列表"""
        return [d.key.expert_id for d in self.preload]


@dataclass
class ExpertLayerContext:
    """执行器输入，包含当前层专家模块、隐藏状态和专家分配"""
    layer: int
    experts: object
    inps_flat: torch.Tensor
    hidden_dim: int
    assignments: Dict[int, ExpertAssignment]


def build_current_demands(
    layer: int,
    active_experts: List[int],
    token_indices_by_expert: Dict[int, torch.Tensor],
) -> List[ExpertDemand]:
    """从当前活跃专家生成当前层需求"""
    demands = []
    for expert_id in active_experts:
        token_count = int(token_indices_by_expert[expert_id].shape[0])
        demands.append(
            ExpertDemand(
                key=ExpertKey(layer=layer, expert_id=expert_id),
                token_count=token_count,
                score=float(token_count),
                source="current",
            )
        )
    return demands


def build_assignments(
    raw_assignments: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
) -> Dict[int, ExpertAssignment]:
    """把原始 (token_indices, routing_weights) 转成结构化的 ExpertAssignment"""
    return {
        expert_id: ExpertAssignment(
            expert_id=expert_id,
            token_indices=token_indices,
            routing_weights=routing_weights,
        )
        for expert_id, (token_indices, routing_weights) in raw_assignments.items()
    }


def build_future_demands(
    layer: int,
    predicted_experts,
    predicted_weights=None,
    source: str = "predicted",
) -> List[ExpertDemand]:
    """从预测的专家张量生成未来专家需求"""
    if predicted_experts is None:
        return []
    if isinstance(predicted_experts, list):
        return predicted_experts

    counts: Dict[int, int] = {}
    scores: Dict[int, float] = {}
    flat_experts = predicted_experts.reshape(-1, predicted_experts.shape[-1])
    flat_weights = None
    if predicted_weights is not None:
        flat_weights = predicted_weights.reshape(-1, predicted_weights.shape[-1])

    for token_idx in range(flat_experts.shape[0]):
        for slot_idx in range(flat_experts.shape[1]):
            expert_id = int(flat_experts[token_idx, slot_idx].item())
            counts[expert_id] = counts.get(expert_id, 0) + 1
            if flat_weights is not None:
                scores[expert_id] = scores.get(expert_id, 0.0) + float(flat_weights[token_idx, slot_idx].item())
            else:
                scores[expert_id] = scores.get(expert_id, 0.0) + 1.0

    return [
        ExpertDemand(
            key=ExpertKey(layer=layer, expert_id=expert_id),
            token_count=counts[expert_id],
            score=scores[expert_id],
            source=source,
        )
        for expert_id in counts
    ]


def unique_demands(demands: List[ExpertDemand]) -> List[ExpertDemand]:
    """去重未来需求，保留每个专家的首次出现"""
    seen = set()
    result = []
    for demand in demands:
        key = (demand.key.layer, demand.key.expert_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(demand)
    return result
