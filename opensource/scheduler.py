"""
MoE 专家调度策略模块

提供:
- ExpertScheduler: 专家调度策略类
- Decode: 四模式负载均衡 (A/B/C/default)
- Prefill: 三步调度法 (全局排序→局部重排→置信度预取)
"""

import math
from typing import List, Tuple
from config import ModelConfig


class ExpertScheduler:
    """专家调度策略类"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.e = config.e
        self.tg = config.tg
        self.tc = config.get_tc()
        self.cpu_time_table = config.cpu_time_table
        self.max_ondemand = (
            4
        )

    def set_cpu_time_table(self, table: List[float]):
        """设置 CPU 时间表"""
        self.cpu_time_table = table
        self.config.cpu_time_table = table
        if table and len(table) > 1:
            self.tc = table[1]

    def decide_decode(
        self, r_cur: int, r_next: int, k: int
    ) -> Tuple[str, int, int, int]:
        """
        Decode 阶段负载均衡调度

        参数:
            r_cur: 当前层选中专家中已在GPU的数量（驻留+已预取）
            r_next: 下一层预测专家中已在GPU的数量
            k: 当前层选中的专家总数

        返回:
            (mode, ondemand_count, prefetch_count, offload_count)
            mode: "A" | "B" | "C" | "default"
            ondemand_count: 当前层需要ondemand加载的专家数
            prefetch_count: 下一层需要prefetch的专家数
            offload_count: 当前层需要释放的未使用预取专家数
        """
        if k == 0:
            return "C", 0, 0, 0

        n_g = self._compute_optimal_gpu_count(k)

        cur_below = r_cur < n_g
        next_below = r_next < n_g

        if not cur_below and not next_below:
            offload = r_cur - n_g
            return "C", 0, 0, offload
        elif cur_below and not next_below:
            ondemand = min(n_g - r_cur, self.max_ondemand)
            return "A", ondemand, 0, 0
        elif not cur_below and next_below:
            prefetch = min(n_g - r_next, self.max_ondemand)
            offload = r_cur - n_g
            return "B", 0, prefetch, offload
        else:
            ondemand_need = n_g - r_cur
            prefetch_need = n_g - r_next
            total_budget = self.max_ondemand
            if ondemand_need + prefetch_need <= total_budget:
                ondemand = ondemand_need
                prefetch = prefetch_need
            else:
                ondemand = min(ondemand_need, total_budget // 2 + total_budget % 2)
                prefetch = min(prefetch_need, total_budget - ondemand)
            return "default", ondemand, prefetch, 0

    def _compute_optimal_gpu_count(self, k: int) -> int:
        """
        n_g^ρ = argmin max(n_g · tg, (k - n_g) · tc)
        精确比较 floor 和 ceil 两个候选值
        """
        if k <= 0:
            return 0
        if self.tc <= 0 or self.tg <= 0:
            return min(k, self.max_ondemand)

        n_g_balanced = k * self.tc / (self.tg + self.tc)
        n_g_floor = max(0, int(math.floor(n_g_balanced)))
        n_g_ceil = min(k, int(math.ceil(n_g_balanced)))

        if n_g_floor == n_g_ceil:
            return n_g_floor

        cost_floor = max(n_g_floor * self.tg, (k - n_g_floor) * self.tc)
        cost_ceil = max(n_g_ceil * self.tg, (k - n_g_ceil) * self.tc)

        return n_g_floor if cost_floor <= cost_ceil else n_g_ceil

    def decide_prefill_schedule(
        self,
        cur_sorted_experts: List[Tuple[int, int]],
        next_sorted_experts: List[Tuple[int, int]],
        cur_gpu_resident: int = 0,
        is_last_layer: bool = False,
        r_hit: float = 0.7,
    ) -> Tuple[List[int], List[int], int]:
        """
        Prefill 三步调度法 (Algorithm 1, Line 8-23)

        Step 1: 全局排序 - 合并两层专家，增量累积 TG 找 GPU/CPU 边界
        Step 2: 局部重排 - L_global ∩ E_cur 确保当前层不被饿死
        Step 3: 置信度预取 - 利用 I/O bubble，逐专家计算 ξ，含 f-1 回退

        参数:
            cur_sorted_experts: 当前层非GPU驻留专家 [(expert_id, token_count), ...] 按token降序
            next_sorted_experts: 下一层非GPU驻留专家 [(expert_id, token_count), ...] 按token降序
            cur_gpu_resident: 当前层已驻留GPU的专家数
            is_last_layer: 是否最后一层（最后一层不做预取）
            r_hit: 预测命中率 (默认0.7)

        返回:
            (ondemand_experts, prefetch_experts, prefetch_count)
        """
        if not cur_sorted_experts and not next_sorted_experts:
            return [], [], 0

        t_io = self.e
        t_g = self.tg
        cpu_table_max_idx = len(self.cpu_time_table) - 1

        # ============ Step 1: Global Ranking (Algorithm 1 Line 8-12) ============
        # 合并两层专家，标记来源，按token数升序排列（小token在前）
        e_all = []
        for eid, tokens in cur_sorted_experts:
            e_all.append((eid, tokens, "cur"))
        for eid, tokens in next_sorted_experts:
            e_all.append((eid, tokens, "next"))
        e_all.sort(key=lambda x: x[1])

        n_total = len(e_all)

        # 增量式扫描: TG 累积增加(每加一个专家多一次传输), TC 递减(专家离开CPU)
        # T_G = (1+i)*t_io + t_g, T_C = 剩余专家的CPU总时间
        # 当 T_G < T_C 时 break，右侧大token专家进 L_global
        T_C_all = sum(
            self._lookup_cpu_time(t, cpu_table_max_idx) for _, t, _ in e_all
        )
        T_G_cum = 0.0

        l_global_cur = []
        l_global_next = []

        for i in range(n_total):
            eid, tokens, origin = e_all[i]
            T_G_cum += t_io
            cpu_time_i = self._lookup_cpu_time(tokens, cpu_table_max_idx)
            T_C_all -= cpu_time_i

            if T_G_cum + t_g < T_C_all:
                if origin == "cur":
                    l_global_cur.append((eid, tokens))
                else:
                    l_global_next.append((eid, tokens))

        # ============ Step 2: Local Re-ranking (Algorithm 1 Line 13-15) ============
        # L_global ∩ E_cur: 只在当前层专家中找边界，防止被下层饿死
        if not l_global_cur:
            return [], [eid for eid, _ in l_global_next], len(l_global_next)

        # 当前层专家按 token 升序（小在前），增量扫描 ondemand 边界
        l_global_cur.sort(key=lambda x: x[1])
        n_cur = len(l_global_cur)

        T_C_local = sum(
            self._lookup_cpu_time(t, cpu_table_max_idx) for _, t in l_global_cur
        )
        T_G_local_cum = 0.0

        ondemand_experts = []
        for i_prime in range(n_cur):
            eid, tokens = l_global_cur[i_prime]
            T_G_local_cum += t_io
            cpu_time_i = self._lookup_cpu_time(tokens, cpu_table_max_idx)
            T_C_local -= cpu_time_i

            if T_G_local_cum + t_g < T_C_local:
                ondemand_experts.append(eid)

        ondemand_experts = ondemand_experts[: self.max_ondemand]

        # ============ Step 3: Confidence-aware Prefetch (Algorithm 1 Line 16-23) ============
        # 计算 I/O bubble, 逐专家评估 ξ, 含 f-1 回退
        prefetch_experts = []
        prefetch_count = 0

        if not is_last_layer and l_global_next:
            T_G_final = len(ondemand_experts) * t_io + t_g
            T_C_final = T_C_local
            T_gap = T_C_final - T_G_final

            if T_gap > 0 and t_io > 0:
                f = int(math.floor(T_gap / t_io))

                # 逐专家计算边际期望效用 ξ
                # R_hit 随排序递减: 第 i 个专家的命中率 = max(0.5, r_hit - i*0.1)
                for i, (eid, _) in enumerate(l_global_next):
                    if i >= f + 1:
                        break
                    r_i = max(0.5, r_hit - i * 0.1)

                    # 第 i 个预取的期望收益
                    if i < f:
                        xi = (2 * r_i - 1) * t_io
                    else:
                        # 边际专家 (f+1 th): 考虑气泡剩余时间
                        fractional = (T_gap / t_io) - f
                        xi = (
                            r_i * fractional * t_io
                            - (1 - r_i) * (1 - fractional) * t_io
                        )

                    if xi > 0:
                        prefetch_experts.append(eid)
                    else:
                        # f-1 回退: 预取量减 1 后成本更低，可能重新变正
                        break

                prefetch_count = len(prefetch_experts)
                prefetch_count = min(prefetch_count, self.max_ondemand)
                prefetch_experts = prefetch_experts[:prefetch_count]

        return ondemand_experts, prefetch_experts, prefetch_count

    def _lookup_cpu_time(self, tokens: int, max_idx: int) -> float:
        if not self.cpu_time_table or max_idx < 0:
            return 0.0
        idx = min(tokens, max_idx)
        if idx < len(self.cpu_time_table):
            return self.cpu_time_table[idx]
        return 0.0
