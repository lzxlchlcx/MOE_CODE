from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.expert_latency import ExpertLatencyModel
from model.expert_types import (
    ExpertDemand,
    ExpertKey,
    ExpertLayerRequest,
    ExpertSchedule,
    PlacementSnapshot,
    build_assignments,
    build_current_demands,
    build_future_demands,
    unique_demands,
)


def _build_expert_mask(selected_experts: torch.Tensor, n_expert: int) -> torch.Tensor:
    """构建专家掩码张量，将 selected_experts 转换为 one-hot 编码并调整维度顺序

    Args:
        selected_experts: 形状 [batch_size, seq_len, 2]，每个 token 选择的 top-2 专家索引
        n_expert: 总专家数

    Returns:
        形状 [n_expert, 2, batch_size*seq_len] 的 one-hot 掩码张量
    """
    return torch.nn.functional.one_hot(selected_experts, num_classes=n_expert).permute(2, 1, 0)


def _collect_active_experts(expert_mask: torch.Tensor, n_expert: int) -> Tuple[List[int], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """收集有 token 分配的活跃专家及其对应的 token 索引
    
    Args:
        expert_mask: 形状 [n_expert, 2, batch_size*seq_len] 的专家掩码
        n_expert: 总专家数
    
    Returns:
        active_experts: 活跃专家索引列表
        idxs: 字典 {专家索引: 该专家在 top-2 中的位置索引}
        top_2s: 字典 {专家索引: 分配给该专家的 token 索引}
    """
    idxs = {}
    top_2s = {}
    active_experts = []
    for i_expert in range(n_expert):
        idx, top_2 = torch.where(expert_mask[i_expert])
        if top_2.shape[0] > 0:
            idxs[i_expert] = idx
            top_2s[i_expert] = top_2
            active_experts.append(i_expert)
    return active_experts, idxs, top_2s


def _organize_token_assignments(expert_mask: torch.Tensor, routing_weights: torch.Tensor, active_experts: List[int]) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """为每个活跃专家组织 token 分配信息和对应的路由权重
    
    Args:
        expert_mask: 形状 [n_expert, 2, batch_size*seq_len] 的专家掩码
        routing_weights: 形状 [batch_size, seq_len, 2] 的路由权重
        active_experts: 活跃专家索引列表
    
    Returns:
        expert_assignments: 字典 {专家索引: (token索引张量, 对应的路由权重张量)}
    """
    expert_assignments = {}
    for i_expert in active_experts:
        idx, top_2 = torch.where(expert_mask[i_expert])
        routing_weight_subset = routing_weights[top_2, idx, None]
        expert_assignments[i_expert] = (top_2, routing_weight_subset)
    return expert_assignments


class ExpertSchedulingStrategy:
    """专家调度策略基类，定义策略接口"""
    def __init__(self, dev, is_expert_in_gpu):
        self.dev = dev
        self.is_expert_in_gpu = is_expert_in_gpu

    def decide_and_prepare(
        self,
        i_layer: int,
        experts: nn.ModuleList,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        n_expert: int,
        **kwargs,
    ) -> Tuple[List[int], List[int], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """策略决策与预处理接口
        
        Args:
            i_layer: 当前层索引
            experts: 专家模块列表
            selected_experts: 每个 token 选择的 top-2 专家
            routing_weights: 每个 token 对所选专家的路由权重
            n_expert: 总专家数
        
        Returns:
            cpu_experts: 在 CPU 上执行的专家索引列表
            gpu_experts: 在 GPU 上执行的专家索引列表
            expert_assignments: 每个专家的 token 分配信息
        """
        raise NotImplementedError


class GPUOnlyStrategy(ExpertSchedulingStrategy):
    """纯 GPU 调度策略：所有专家都在 GPU 上执行"""
    def decide_and_prepare(
        self,
        i_layer: int,
        experts: nn.ModuleList,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        n_expert: int,
        **kwargs,
    ) -> Tuple[List[int], List[int], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """决策：所有活跃专家都在 GPU 上执行"""
        expert_mask = _build_expert_mask(selected_experts, n_expert)
        active_experts, _, _ = _collect_active_experts(expert_mask, n_expert)
        expert_assignments = _organize_token_assignments(expert_mask, routing_weights, active_experts)
        return [], active_experts, expert_assignments


class FiddlerStrategy(ExpertSchedulingStrategy):
    """CPU-GPU 混合调度策略：通过代价优化决定每个专家在 CPU 还是 GPU 上执行
    
    职责：
    - 构建每个活跃专家的 CPU/GPU 执行代价表
    - 穷举 2^n_active 种配置找到最小代价的分配方案
    - 统计 GPU 缓存命中率
    """
    def __init__(self, dev, is_expert_in_gpu, latency_cpu, latency_gpu):
        super().__init__(dev, is_expert_in_gpu)
        self.latency_cpu = latency_cpu  # CPU 上每个 token 的延迟
        self.latency_gpu = latency_gpu  # 将专家权重从 CPU 拷贝到 GPU 的固定延迟
        self.cnt_expert_hit = 0  # GPU 缓存命中的 token 数
        self.cnt_expert_all = 0  # 总 token 数

    def decide_and_prepare(
        self,
        i_layer: int,
        experts: nn.ModuleList,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        n_expert: int,
    ) -> Tuple[List[int], List[int], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """决策：通过穷举 2^n_active 种配置找到代价最小的 CPU/GPU 分配方案
        
        Args:
            i_layer: 当前层索引
            experts: 专家模块列表
            selected_experts: 每个 token 选择的 top-2 专家
            routing_weights: 每个 token 对所选专家的路由权重
            n_expert: 总专家数
        
        Returns:
            cpu_experts: 在 CPU 上执行的专家索引列表
            gpu_experts: 在 GPU 上执行的专家索引列表
            expert_assignments: 每个专家的 token 分配信息
        """
        expert_mask = _build_expert_mask(selected_experts, n_expert)
        active_experts, idxs, top_2s = _collect_active_experts(expert_mask, n_expert)
        expert_assignments = _organize_token_assignments(expert_mask, routing_weights, active_experts)
        
        n_active = len(active_experts)
        cost_cpu = np.zeros(n_active, dtype=float)
        cost_gpu = np.zeros(n_active, dtype=float)
        for bit, i_expert in enumerate(active_experts):
            token_count = top_2s[i_expert].shape[0]
            cost_cpu[bit] = token_count * self.latency_cpu
            cost_gpu[bit] = self.latency_gpu
            if self.is_expert_in_gpu(i_layer, i_expert):
                cost_gpu[bit] = 0
                self.cnt_expert_hit += token_count
            self.cnt_expert_all += token_count
        
        # 穷举搜索最优配置
        best_config = -1
        best_cost = float("inf")
        for config in range(1 << n_active):
            sum_cost = 0
            for bit in range(n_active):
                if (config >> bit) & 1:
                    sum_cost += cost_cpu[bit]
                else:
                    sum_cost += cost_gpu[bit]
            if sum_cost < best_cost:
                best_cost = sum_cost
                best_config = config
        
        # 解码最优配置
        cpu_experts = []
        gpu_experts = []
        for bit, i_expert in enumerate(active_experts):
            if (best_config >> bit) & 1:
                cpu_experts.append(i_expert)
            else:
                gpu_experts.append(i_expert)
        
        return cpu_experts, gpu_experts, expert_assignments


def _build_latency_lookup(entries: list) -> Dict[int, float]:
    """从 benchmark JSON 数组构建 {token_count: avg_time_ms} 查找表

    Args:
        entries: benchmark 数据列表，每项包含 token_count 和 avg_time_ms

    Returns:
        {token_count: avg_time_ms} 字典
    """
    table = {}
    for entry in entries:
        tc = entry["token_count"]
        table[tc] = entry["avg_time_ms"]
    return table


def _lookup_latency(table: Dict[int, float], token_count: int) -> float:
    """从查找表获取延迟值

    查找策略：
    - 精确匹配：直接返回
    - 超出最大值：使用最大 token_count 的条目
    - 其他：使用最接近的 token_count 条目

    Args:
        table: {token_count: avg_time_ms} 查找表
        token_count: 要查询的 token 数量

    Returns:
        对应的延迟值（毫秒）
    """
    if token_count in table:
        return table[token_count]
    max_tc = max(table.keys())
    if token_count >= max_tc:
        return table[max_tc]
    closest = min(table.keys(), key=lambda k: abs(k - token_count))
    return table[closest]


class ExpertScheduler(ABC):
    """阶段感知专家调度器抽象接口

    职责：
    - 接收结构化的调度请求（ExpertLayerRequest）和驻留快照（PlacementSnapshot）
    - 结合延迟模型做出调度决策
    - 输出 ExpertSchedule，包含 CPU/GPU/preload/evict 四类专家列表
    - 只负责"决定"，不负责"执行"
    """

    @abstractmethod
    def schedule(
        self,
        request: ExpertLayerRequest,
        placement: PlacementSnapshot,
        latency: ExpertLatencyModel,
    ) -> ExpertSchedule:
        """根据请求、驻留状态和延迟模型生成调度计划"""
        raise NotImplementedError


class PDScopeScheduler(ExpertScheduler):
    """PDScope 论文风格的阶段感知调度器

    职责：
    - 区分 prefill 和 decode 阶段，分别执行不同的调度策略
    - prefill：三步调度法（全局队列 → 当前按需 → 预加载）
    - decode：ABCD 模式决策（当前/下一层驻留是否充足）
    - 结合 I/O 气泡窗口和命中率决定预加载数量

    参数：
    - alpha: GPU 启动开销（秒）
    - t_attn: 注意力计算时间窗口，用于计算预加载可用时间
    - r_hit: 预期的 placeholder 命中率
    """

    def __init__(self, alpha: float = 0.1, t_attn: float = 0.6, r_hit: float = 0.8):
        self.alpha = alpha
        self.t_attn = t_attn
        self.r_hit = r_hit

    def schedule(
        self,
        request: ExpertLayerRequest,
        placement: PlacementSnapshot,
        latency: ExpertLatencyModel,
    ) -> ExpertSchedule:
        """根据阶段分发到 prefill 或 decode 调度逻辑"""
        if request.phase == "prefill":
            return self.schedule_prefill(request, placement, latency)
        return self.schedule_decode(request, placement, latency)

    def schedule_prefill(
        self,
        request: ExpertLayerRequest,
        placement: PlacementSnapshot,
        latency: ExpertLatencyModel,
    ) -> ExpertSchedule:
        """Prefill 阶段调度：三步调度法

        步骤：
        1. 分离当前驻留/非驻留专家和未来非驻留专家
        2. 合并排序后构造全局候选队列
        3. 从全局队列中选出当前层值得按需加载到 GPU 的专家
        4. 根据 I/O 气泡窗口选择未来专家预加载
        """
        current = request.current
        current_resident = [d for d in current if placement.is_on_gpu(d.key.layer, d.key.expert_id)]
        current_non_resident = [d for d in current if not placement.is_on_gpu(d.key.layer, d.key.expert_id)]
        future_non_resident = [
            d for d in request.future
            if not placement.is_on_gpu(d.key.layer, d.key.expert_id)
            and (d.key.layer, d.key.expert_id) not in placement.loading
        ]

        combined = current_non_resident + future_non_resident
        combined.sort(key=lambda d: (d.token_count, d.score))

        global_queue = self._select_global_queue(combined, latency)
        current_global = [d for d in global_queue if d.source == "current"]
        ondemand, t_gpu, t_cpu = self._select_current_ondemand(current_global, latency)

        gpu_keys = {(d.key.layer, d.key.expert_id) for d in current_resident + ondemand}
        gpu = [d for d in current if (d.key.layer, d.key.expert_id) in gpu_keys]
        cpu = [d for d in current if (d.key.layer, d.key.expert_id) not in gpu_keys]
        preload = self._select_preload(global_queue, t_gpu, t_cpu, placement, latency)

        return ExpertSchedule(cpu=cpu, gpu=gpu, preload=preload, evict=[], reason="prefill")

    def schedule_decode(
        self,
        request: ExpertLayerRequest,
        placement: PlacementSnapshot,
        latency: ExpertLatencyModel,
    ) -> ExpertSchedule:
        """Decode 阶段调度：ABCD 模式决策

        步骤：
        1. 计算理想 GPU 专家数 n_g_rho（最小化 max(GPU时间, CPU时间)）
        2. 评估当前层和下一层的驻留是否充足
        3. 根据四种情况决策：
           - 当前不足 + 下一层不足 → fallback 到 prefill 策略
           - 当前不足 + 下一层充足 → 补当前层 GPU 专家（mode-a）
           - 当前充足 + 下一层不足 → 预加载下一层专家（mode-b）
           - 当前充足 + 下一层充足 → 不额外操作（mode-c）
        """
        current = request.current
        if not current:
            return ExpertSchedule(reason="decode-empty")

        k = len(current)
        t_c = latency.cpu(1)
        t_g = latency.gpu_compute(1)
        n_g_rho = min(
            range(k + 1),
            key=lambda n_g: max(n_g * t_g, (k - n_g) * t_c),
        )

        current_resident = [d for d in current if placement.is_on_gpu(d.key.layer, d.key.expert_id)]
        current_non_resident = [d for d in current if not placement.is_on_gpu(d.key.layer, d.key.expert_id)]
        future_non_resident = [
            d for d in sorted(request.future, key=lambda x: (x.score, x.token_count), reverse=True)
            if not placement.is_on_gpu(d.key.layer, d.key.expert_id)
            and (d.key.layer, d.key.expert_id) not in placement.loading
        ]
        next_resident_count = sum(
            1 for d in request.future
            if placement.is_on_gpu(d.key.layer, d.key.expert_id)
        )

        cur_below = len(current_resident) < n_g_rho
        next_below = next_resident_count < n_g_rho

        if cur_below and next_below:
            schedule = self.schedule_prefill(request, placement, latency)
            schedule.reason = "decode-fallback-prefill"
            return schedule
        if cur_below and not next_below:
            need = min(n_g_rho - len(current_resident), len(current_non_resident))
            ondemand = current_non_resident[:need]
            gpu_keys = {(d.key.layer, d.key.expert_id) for d in current_resident + ondemand}
            gpu = [d for d in current if (d.key.layer, d.key.expert_id) in gpu_keys]
            cpu = [d for d in current if (d.key.layer, d.key.expert_id) not in gpu_keys]
            return ExpertSchedule(cpu=cpu, gpu=gpu, preload=[], evict=[], reason="decode-mode-a")
        if not cur_below and next_below:
            need_next = max(0, min(n_g_rho - next_resident_count, placement.free_placeholders))
            preload = future_non_resident[:need_next]
            gpu = current_resident
            gpu_keys = {(d.key.layer, d.key.expert_id) for d in gpu}
            cpu = [d for d in current if (d.key.layer, d.key.expert_id) not in gpu_keys]
            return ExpertSchedule(cpu=cpu, gpu=gpu, preload=preload, evict=[], reason="decode-mode-b")

        gpu = current_resident
        gpu_keys = {(d.key.layer, d.key.expert_id) for d in gpu}
        cpu = [d for d in current if (d.key.layer, d.key.expert_id) not in gpu_keys]
        return ExpertSchedule(cpu=cpu, gpu=gpu, preload=[], evict=[], reason="decode-mode-c")

    def _select_global_queue(
        self,
        demands: List[ExpertDemand],
        latency: ExpertLatencyModel,
    ) -> List[ExpertDemand]:
        """构造全局候选队列：从合并需求中选出值得 GPU 化的专家子集

        从右向左扫描，找到第一个满足 T_gpu < T_cpu 的分割点，
        分割点右侧的专家放入 GPU 候选队列。
        """
        if not demands:
            return []
        for i in range(len(demands)):
            gpu_side = demands[i:]
            cpu_side = demands[:i]
            t_gpu = self.alpha + len(gpu_side) * latency.t_io + latency.gpu_compute(1)
            t_cpu = sum(latency.cpu(d.token_count) for d in cpu_side) + self.t_attn
            if t_gpu < t_cpu:
                return gpu_side
        return []

    def _select_current_ondemand(
        self,
        current_global: List[ExpertDemand],
        latency: ExpertLatencyModel,
    ) -> Tuple[List[ExpertDemand], float, float]:
        """从当前层全局候选中选出值得按需加载到 GPU 的专家

        遍历所有分割点，找到第一个 GPU 时间 < CPU 时间的位置，
        该位置右侧的当前层专家值得按需加载到 GPU 执行。

        Returns:
            ondemand: 按需加载到 GPU 的当前层专家列表
            t_gpu: GPU 侧执行时间
            t_cpu: CPU 侧执行时间
        """
        if not current_global:
            return [], 0.0, 0.0
        last_t_gpu = 0.0
        last_t_cpu = sum(latency.cpu(d.token_count) for d in current_global)
        for i in range(len(current_global) + 1):
            gpu_side = current_global[i:]
            cpu_side = current_global[:i]
            t_compute = sum(latency.gpu_compute(d.token_count) for d in gpu_side)
            t_io = len(gpu_side) * latency.t_io
            t_gpu = max(t_compute, self.alpha + t_io) + latency.gpu_compute(1) if gpu_side else 0.0
            t_cpu = sum(latency.cpu(d.token_count) for d in cpu_side)
            last_t_gpu = t_gpu
            last_t_cpu = t_cpu
            if gpu_side and t_gpu < t_cpu:
                return gpu_side, t_gpu, t_cpu
        return [], last_t_gpu, last_t_cpu

    def _select_preload(
        self,
        global_queue: List[ExpertDemand],
        t_gpu: float,
        t_cpu: float,
        placement: PlacementSnapshot,
        latency: ExpertLatencyModel,
    ) -> List[ExpertDemand]:
        """根据 I/O 气泡窗口和 placeholder 容量选择可预加载的未来专家

        计算逻辑：
        1. t_gap = max(0, t_cpu - t_gpu) 为可用气泡时间
        2. capacity = min(空闲placeholder数, 气泡内可传输的专家数)
        3. 从未来预测专家中按 score 降序选取最多 capacity 个
        """
        t_gap = max(0.0, t_cpu - t_gpu)
        capacity = min(placement.free_placeholders, math.floor((t_gap + self.t_attn) / max(latency.t_io, 1e-9)))
        if capacity <= 0:
            return []
        xi = (2 * self.r_hit - 1) * latency.t_io
        if xi <= 0:
            return []
        future = [d for d in global_queue if d.source == "predicted"]
        future.sort(key=lambda d: (d.score, d.token_count), reverse=True)
        return future[:capacity]


class PrefetchHybridStrategy(ExpertSchedulingStrategy):
    """PDScope AdaptSched 调度策略：区分 Prefill 三步调度法和 Decode ABC 策略

    职责：
    - 作为 deepseek.py 与 PDScopeScheduler 之间的适配层
    - 从 gate 输出构造当前专家需求
    - 接收预测出的未来专家需求
    - 接收 placeholder_manager 的驻留快照
    - 构造 ExpertLayerRequest 并调用 PDScopeScheduler
    - 把 ExpertSchedule 转回 deepseek.py 使用的返回格式

    参数：
    - t_io: 专家权重传输延迟（秒）
    - latency_cpu_table: CPU 延迟查找表 {token_count: time_ms}
    - latency_gpu_table: GPU 延迟查找表 {token_count: time_ms}
    """

    def __init__(self, dev, is_expert_in_gpu, t_io: float,
                 latency_cpu_table: Dict[int, float],
                 latency_gpu_table: Dict[int, float]):
        super().__init__(dev, is_expert_in_gpu)
        self.t_io = t_io
        self.latency_cpu_table = latency_cpu_table
        self.latency_gpu_table = latency_gpu_table
        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0
        self.latency_model = ExpertLatencyModel(t_io, latency_cpu_table, latency_gpu_table)
        self.scheduler = PDScopeScheduler()

    def decide_and_prepare(
        self,
        i_layer: int,
        experts: nn.ModuleList,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        n_expert: int,
        predicted_next_experts: Optional[torch.Tensor] = None,
        predicted_next_weights: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        future_demands: Optional[List[ExpertDemand]] = None,
        placement: Optional[PlacementSnapshot] = None,
    ) -> Tuple[List[int], List[int], List[int], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """策略决策入口：构造请求并调用 PDScopeScheduler

        流程：
        1. 从 selected_experts 构建 expert mask
        2. 收集活跃专家并组织 token 分配
        3. 构造 current_demands 和 future_demands
        4. 更新 GPU 命中统计
        5. 构造 ExpertLayerRequest 并调用 scheduler.schedule()
        6. 从 ExpertSchedule 提取 CPU/GPU/preload 专家 id 列表

        Returns:
            cpu_expert_ids: CPU 执行的专家 id 列表
            gpu_expert_ids: GPU 执行的专家 id 列表
            preload_expert_ids: 预加载的专家 id 列表
            raw_assignments: 原始 (token_indices, routing_weights) 字典
        """
        expert_mask = _build_expert_mask(selected_experts, n_expert)
        active_experts, _, token_indices_by_expert = _collect_active_experts(expert_mask, n_expert)
        raw_assignments = _organize_token_assignments(expert_mask, routing_weights, active_experts)
        current_demands = build_current_demands(i_layer, active_experts, token_indices_by_expert)
        future = future_demands
        if future is None:
            future = build_future_demands(i_layer + 1, predicted_next_experts, predicted_next_weights)
        future = unique_demands(future)
        if placement is None:
            gpu_resident = set()
            for demand in current_demands + future:
                if self.is_expert_in_gpu(demand.key.layer, demand.key.expert_id):
                    gpu_resident.add((demand.key.layer, demand.key.expert_id))
            placement = PlacementSnapshot(gpu_resident=gpu_resident)

        for demand in current_demands:
            if placement.is_on_gpu(demand.key.layer, demand.key.expert_id):
                self.cnt_expert_hit += demand.token_count
            self.cnt_expert_all += demand.token_count

        request = ExpertLayerRequest(
            layer=i_layer,
            phase="prefill" if is_prefill else "decode",
            current=current_demands,
            future=future,
            assignments=build_assignments(raw_assignments),
        )
        schedule = self.scheduler.schedule(request, placement, self.latency_model)
        return schedule.cpu_expert_ids, schedule.gpu_expert_ids, schedule.preload_expert_ids, raw_assignments

    def _prefill_schedule(
        self, i_layer, cpu_experts, gpu_experts, token_counts,
        predicted_next_experts, predicted_next_weights,
    ) -> List[int]:
        """旧版 prefill 调度逻辑（已被 PDScopeScheduler.schedule_prefill 替代）

        保留用于兼容和对比测试。内部使用原始 tuple 格式而非 ExpertDemand。
        """
        t_attn = 0.6

        cur_non_resident = []
        for eid in cpu_experts:
            tc = token_counts.get(eid, 1)
            cur_non_resident.append(("cur", eid, tc))
        for eid in gpu_experts:
            if not self.is_expert_in_gpu(i_layer, eid):
                tc = token_counts.get(eid, 1)
                cur_non_resident.append(("cur", eid, tc))

        next_token_counts = {}
        for batch_idx in range(predicted_next_experts.shape[0]):
            for expert in predicted_next_experts[batch_idx]:
                eid = expert.item()
                next_token_counts[eid] = next_token_counts.get(eid, 0) + 1

        next_non_resident = []
        for eid, tc in next_token_counts.items():
            if not self.is_expert_in_gpu(i_layer + 1, eid):
                next_non_resident.append(("next", eid, tc))

        E_all = cur_non_resident + next_non_resident
        E_all.sort(key=lambda x: x[2])

        n = len(cur_non_resident)
        n_prime = len(next_non_resident)
        total = len(E_all)

        alpha = 0.1
        L_global = []
        for i in range(total):
            n_transfer = total - i
            T_all_G = alpha + n_transfer * self.t_io + _lookup_latency(self.latency_gpu_table, 1)
            T_all_C = sum(
                _lookup_latency(self.latency_cpu_table, E_all[j][2])
                for j in range(i)
            ) + t_attn
            if T_all_G < T_all_C:
                L_global = E_all[i:]
                break

        cur_in_global = [e for e in L_global if e[0] == "cur"]
        n_cur = len(cur_in_global)
        L_on = []
        T_G = 0.0
        T_C = 0.0
        for i_prime in range(n_cur + 1):
            n_g = sum(1 for e in cur_in_global[i_prime:])
            T_G_compute = n_g * _lookup_latency(self.latency_gpu_table, 1)
            T_G_io = alpha + (n_cur - i_prime) * self.t_io if i_prime < n_cur else 0
            T_G = max(T_G_compute, T_G_io) + _lookup_latency(self.latency_gpu_table, 1)
            T_C = sum(
                _lookup_latency(self.latency_cpu_table, cur_in_global[j][2])
                for j in range(i_prime)
            )
            if T_G < T_C and i_prime > 0:
                L_on = cur_in_global[i_prime:]
                break

        T_gap = max(0, T_C - T_G)
        f = math.floor((T_gap + t_attn) / self.t_io)
        R_hit = 0.8
        f_abs = abs(f)
        if f_abs > 0:
            xi = R_hit * (f - f_abs + 1) * self.t_io - (1 - R_hit) * (f_abs - f) * self.t_io
        else:
            xi = 0

        prefetch = []
        if xi > 0 and f_abs > 0:
            next_in_global = [e for e in L_global if e[0] == "next"]
            for e in next_in_global[:f_abs]:
                prefetch.append(e[1])

        return prefetch

    def _decode_schedule(
        self, i_layer, cpu_experts, gpu_experts, token_counts,
        predicted_next_experts, predicted_next_weights,
    ) -> List[int]:
        """旧版 decode 调度逻辑（已被 PDScopeScheduler.schedule_decode 替代）

        保留用于兼容和对比测试。内部使用原始 tuple 格式而非 ExpertDemand。
        """
        k = len(cpu_experts) + len(gpu_experts)
        if k == 0:
            return []

        t_c_1 = _lookup_latency(self.latency_cpu_table, 1)
        t_g_1 = _lookup_latency(self.latency_gpu_table, 1)

        n_g_rho = 1
        best_max = float("inf")
        for n_g in range(k + 1):
            val = max(n_g * t_g_1, (k - n_g) * t_c_1)
            if val < best_max:
                best_max = val
                n_g_rho = n_g

        cur_gpu_resident = sum(1 for e in gpu_experts if self.is_expert_in_gpu(i_layer, e))

        next_token_counts = {}
        for batch_idx in range(predicted_next_experts.shape[0]):
            for expert in predicted_next_experts[batch_idx]:
                eid = expert.item()
                next_token_counts[eid] = next_token_counts.get(eid, 0) + 1

        next_gpu_resident = 0
        for eid in next_token_counts:
            if self.is_expert_in_gpu(i_layer + 1, eid):
                next_gpu_resident += 1

        cur_below = cur_gpu_resident < n_g_rho
        next_below = next_gpu_resident < n_g_rho

        if cur_below and next_below:
            return self._prefill_schedule(
                i_layer, cpu_experts, gpu_experts, token_counts,
                predicted_next_experts, predicted_next_weights,
            )
        elif cur_below and not next_below:
            return []
        elif not cur_below and next_below:
            prefetch = []
            for eid in next_token_counts:
                if not self.is_expert_in_gpu(i_layer + 1, eid):
                    prefetch.append(eid)
            return prefetch
        else:
            return []
