from typing import List, Dict, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    ) -> Tuple[List[int], List[int], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """决策：所有活跃专家都在 GPU 上执行"""
        expert_mask = _build_expert_mask(selected_experts, n_expert)
        active_experts, _, _ = _collect_active_experts(expert_mask, n_expert)
        expert_assignments = _organize_token_assignments(expert_mask, routing_weights, active_experts)
        return [], active_experts, expert_assignments


class HybridCPUGPUStrategy(ExpertSchedulingStrategy):
    """CPU-GPU 混合调度策略：通过代价优化决定每个专家在 CPU 还是 GPU 上执行"""
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
        
        代价计算：
        - CPU 代价：token_count * latency_cpu
        - GPU 代价：如果专家已在 GPU 上则为 0，否则为 latency_gpu
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
    """从 benchmark JSON 数组构建 {token_count: avg_time_ms} 查找表"""
    table = {}
    for entry in entries:
        tc = entry["token_count"]
        table[tc] = entry["avg_time_ms"]
    return table


def _lookup_latency(table: Dict[int, float], token_count: int) -> float:
    """从查找表获取延迟值，超出范围使用最大 token_count 的条目"""
    if token_count in table:
        return table[token_count]
    max_tc = max(table.keys())
    if token_count >= max_tc:
        return table[max_tc]
    closest = min(table.keys(), key=lambda k: abs(k - token_count))
    return table[closest]


class PrefetchHybridStrategy(ExpertSchedulingStrategy):
    """PDScope AdaptSched 调度策略：区分 Prefill 三步调度法和 Decode ABC 策略"""

    def __init__(self, dev, is_expert_in_gpu, t_io: float,
                 latency_cpu_table: Dict[int, float],
                 latency_gpu_table: Dict[int, float]):
        super().__init__(dev, is_expert_in_gpu)
        self.t_io = t_io
        self.latency_cpu_table = latency_cpu_table
        self.latency_gpu_table = latency_gpu_table
        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

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
    ) -> Tuple[List[int], List[int], List[int], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        expert_mask = _build_expert_mask(selected_experts, n_expert)
        active_experts, idxs, top_2s = _collect_active_experts(expert_mask, n_expert)
        expert_assignments = _organize_token_assignments(expert_mask, routing_weights, active_experts)

        n_active = len(active_experts)
        cost_cpu = np.zeros(n_active, dtype=float)
        cost_gpu = np.zeros(n_active, dtype=float)
        token_counts = {}
        for bit, i_expert in enumerate(active_experts):
            tc = top_2s[i_expert].shape[0]
            token_counts[i_expert] = tc
            cost_cpu[bit] = tc * _lookup_latency(self.latency_cpu_table, tc)
            cost_gpu[bit] = _lookup_latency(self.latency_gpu_table, tc)
            if self.is_expert_in_gpu(i_layer, i_expert):
                cost_gpu[bit] = 0
                self.cnt_expert_hit += tc
            self.cnt_expert_all += tc

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

        cpu_experts = []
        gpu_experts = []
        for bit, i_expert in enumerate(active_experts):
            if (best_config >> bit) & 1:
                cpu_experts.append(i_expert)
            else:
                gpu_experts.append(i_expert)

        prefetch_experts = []
        if predicted_next_experts is not None and i_layer + 1 < 100:
            if is_prefill:
                prefetch_experts = self._prefill_schedule(
                    i_layer, cpu_experts, gpu_experts, token_counts,
                    predicted_next_experts, predicted_next_weights,
                )
            else:
                prefetch_experts = self._decode_schedule(
                    i_layer, cpu_experts, gpu_experts, token_counts,
                    predicted_next_experts, predicted_next_weights,
                )

        return cpu_experts, gpu_experts, prefetch_experts, expert_assignments

    def _prefill_schedule(
        self, i_layer, cpu_experts, gpu_experts, token_counts,
        predicted_next_experts, predicted_next_weights,
    ) -> List[int]:
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
