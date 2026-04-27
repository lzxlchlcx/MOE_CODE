"""
PDScope AdaptSched: Phase-Aware Adaptive Expert Scheduler

Implements:
- Legacy ondemand scheduling (backward compatible)
- Decode stage explicit load balancing (n_g^rho optimization)
- Prefill stage three-step scheduling (global ranking + local reorder + confidence prefetch)
"""

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from config import ModelConfig


@dataclass
class DecodeSchedule:
    ondemand_experts: List[int] = field(default_factory=list)
    prefetch_experts: List[int] = field(default_factory=list)
    mode: str = "normal"
    n_g_optimal: int = 0
    n_g_on_gpu: int = 0
    bandwidth_action: str = "none"


@dataclass
class PrefillSchedule:
    ondemand_experts: List[int] = field(default_factory=list)
    prefetch_experts: List[int] = field(default_factory=list)
    prefetch_confidences: List[float] = field(default_factory=list)
    io_bubble_ms: float = 0.0
    global_queue_size: int = 0


class AdaptSchedScheduler:
    """PDScope Adaptive Expert Scheduler"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.e = config.e
        self.tg = config.tg
        self.tc = config.tc
        self.cpu_time_table = config.cpu_time_table
        self.attention_time_table = config.attention_time_table
        self.max_ondemand = config.num_placeholders
        self.schedule_mode = config.schedule_mode
        self.top_k = config.top_k

        if not self.tc and self.cpu_time_table:
            idx_1 = min(1, len(self.cpu_time_table) - 1)
            self.tc = self.cpu_time_table[idx_1] if idx_1 >= 0 else 0.0

    def set_cpu_time_table(self, table: List[float]):
        self.cpu_time_table = table
        self.config.cpu_time_table = table
        if not self.tc and table:
            self.tc = table[min(1, len(table) - 1)]

    def set_params(self, e: float = None, tg: float = None, tc: float = None):
        if e is not None:
            self.e = e
        if tg is not None:
            self.tg = tg
        if tc is not None:
            self.tc = tc

    # ================================================================
    # Legacy scheduling (backward compatible)
    # ================================================================

    def decide_ondemand(
        self, sorted_experts: List[Tuple[int, int]], is_decode: bool
    ) -> List[int]:
        """Legacy ondemand scheduling (from original codebase)"""
        n = len(sorted_experts)
        ondemand_experts = []
        if n == 0:
            return ondemand_experts

        cpu_table_max_idx = len(self.cpu_time_table) - 1

        TA = 0.0
        for _, tokens in sorted_experts:
            idx = min(tokens, cpu_table_max_idx) if cpu_table_max_idx >= 0 else 0
            if self.cpu_time_table and idx < len(self.cpu_time_table):
                TA += self.cpu_time_table[idx]

        TC = TA
        total_ondemand_cost = 0.0

        for i in range(n - 1):
            expert_id, token_count = sorted_experts[i]
            TG_i = self.e + self.tg
            TG = total_ondemand_cost + TG_i
            idx = min(token_count, cpu_table_max_idx) if cpu_table_max_idx >= 0 else 0
            cpu_time_i = 0.0
            if self.cpu_time_table and idx < len(self.cpu_time_table):
                cpu_time_i = self.cpu_time_table[idx]
            TC -= cpu_time_i

            if is_decode:
                if TG < TC:
                    ondemand_experts.append(expert_id)
                    total_ondemand_cost += TG_i
                    if i == n - 2:
                        if TC - TG > self.e:
                            if i + 1 < len(sorted_experts):
                                ondemand_experts.append(sorted_experts[i + 1][0])
                                total_ondemand_cost += TG_i
                elif TC > total_ondemand_cost:
                    ondemand_experts.append(expert_id)
                    total_ondemand_cost += TG_i
                else:
                    break
            else:
                idx_prefill = (
                    min(token_count, cpu_table_max_idx) if cpu_table_max_idx >= 0 else 0
                )
                cpu_time_token = (
                    self.cpu_time_table[idx_prefill]
                    if (self.cpu_time_table and idx_prefill < len(self.cpu_time_table))
                    else 0.0
                )
                if TG < TC + cpu_time_token:
                    ondemand_experts.append(expert_id)
                    total_ondemand_cost += TG_i
                else:
                    break

        return ondemand_experts

    # ================================================================
    # Decode stage: explicit load balancing
    # ================================================================

    def decide_decode_schedule(
        self,
        layer_idx: int,
        activated_expert_ids: List[int],
        n_experts_on_gpu_current: int,
        n_experts_on_gpu_next: int,
        total_activated: int,
    ) -> DecodeSchedule:
        """
        Decode stage load balancing.

        Computes optimal n_g^rho = argmin max(n_g * tg, (k - n_g) * tc)
        then decides A/B/C mode based on current vs next layer GPU residency.
        """
        k = total_activated
        n_g_opt = self._compute_optimal_gpu_count(k)

        n_g_cur = n_experts_on_gpu_current
        n_g_next = n_experts_on_gpu_next

        schedule = DecodeSchedule(n_g_optimal=n_g_opt, n_g_on_gpu=n_g_cur)

        if n_g_cur < n_g_opt and n_g_next >= n_g_opt:
            schedule.mode = "A"
            schedule.bandwidth_action = "pause_prefetch"
            need_ondemand = n_g_opt - n_g_cur
            ondemand_candidates = [
                eid
                for eid in activated_expert_ids
                if eid not in self._get_gpu_resident_ids(layer_idx)
            ]
            schedule.ondemand_experts = ondemand_candidates[:need_ondemand]

        elif n_g_cur > n_g_opt and n_g_next < n_g_opt:
            schedule.mode = "B"
            schedule.bandwidth_action = "offload_and_prefetch"
            excess = n_g_cur - n_g_opt
            schedule.ondemand_experts = []
            schedule.prefetch_experts = list(
                range(min(n_g_opt - n_g_next, self.max_ondemand))
            )

        elif n_g_cur > n_g_opt and n_g_next > n_g_opt:
            schedule.mode = "C"
            schedule.bandwidth_action = "offload_only"
            schedule.ondemand_experts = []
            schedule.prefetch_experts = []

        else:
            schedule.mode = "normal"
            non_gpu_experts = [
                eid
                for eid in activated_expert_ids
                if eid not in self._get_gpu_resident_ids(layer_idx)
            ]
            sorted_non_gpu = self._sort_experts_by_token_count(
                layer_idx, non_gpu_experts
            )
            schedule.ondemand_experts = self._compute_ondemand_decode(
                sorted_non_gpu, k, n_g_opt
            )

        return schedule

    def _compute_optimal_gpu_count(self, k: int) -> int:
        """
        n_g^rho = argmin_{n_g} max(n_g * tg, (k - n_g) * tc)
        Balanced when n_g * tg = (k - n_g) * tc
        => n_g = k * tc / (tg + tc)
        """
        if k <= 0:
            return 0
        if self.tc <= 0 or self.tg <= 0:
            return min(k, self.max_ondemand)

        n_g_balanced = k * self.tc / (self.tg + self.tc)
        n_g_floor = max(0, int(math.floor(n_g_balanced)))
        n_g_ceil = min(k, int(math.ceil(n_g_balanced)))

        cost_floor = max(n_g_floor * self.tg, (k - n_g_floor) * self.tc)
        cost_ceil = max(n_g_ceil * self.tg, (k - n_g_ceil) * self.tc)

        return n_g_floor if cost_floor <= cost_ceil else n_g_ceil

    def _sort_experts_by_token_count(self, layer_idx, expert_ids):
        return expert_ids

    def _get_gpu_resident_ids(self, layer_idx):
        return []

    def _compute_ondemand_decode(self, sorted_experts, k, n_g_opt):
        """Simple ondemand for decode: pick up to n_g_opt - gpu_resident experts"""
        ondemand = []
        total_cost = 0.0
        for eid in sorted_experts:
            if len(ondemand) >= n_g_opt:
                break
            if total_cost + self.e + self.tg < (k - len(ondemand)) * self.tc:
                ondemand.append(eid)
                total_cost += self.e + self.tg
            else:
                break
        return ondemand

    # ================================================================
    # Prefill stage: three-step scheduling
    # ================================================================

    def decide_prefill_schedule(
        self,
        layer_idx: int,
        current_layer_experts: List[Tuple[int, int]],
        next_layer_predicted: List[Tuple[int, int]],
        current_layer_gpu_resident: int,
        next_layer_gpu_resident: int,
    ) -> PrefillSchedule:
        """
        Prefill three-step scheduling.

        Step 1: Global ranking - merge current + next layer experts, sort by token count
        Step 2: Local reorder - ensure current layer is not starved
        Step 3: Confidence-aware prefetch - compute expected utility xi
        """
        schedule = PrefillSchedule()

        # Step 1: Global ranking
        E_all = []
        for eid, tokens in current_layer_experts:
            E_all.append(("current", eid, tokens))
        for eid, tokens in next_layer_predicted:
            E_all.append(("next", eid, tokens))

        E_all.sort(key=lambda x: x[2])

        n = len(E_all)
        n_cur = len(current_layer_experts)
        n_next = len(next_layer_predicted)

        cpu_table_max_idx = len(self.cpu_time_table) - 1
        alpha = self.e * 0.1

        global_ondemand_current = []
        global_ondemand_next = []

        T_C_remaining = sum(
            self.cpu_time_table[min(t, cpu_table_max_idx)]
            if (
                self.cpu_time_table
                and min(t, cpu_table_max_idx) < len(self.cpu_time_table)
            )
            else 0.0
            for _, _, t in E_all
        )

        attn_time = 0.0
        if self.attention_time_table:
            attn_idx = min(
                sum(t for _, _, t in E_all) // max(1, n),
                len(self.attention_time_table) - 1,
            )
            if attn_idx >= 0:
                attn_time = self.attention_time_table[attn_idx]

        T_C_all = T_C_remaining + attn_time

        for i in range(n):
            origin, eid, tokens = E_all[i]
            T_G_all = alpha + (n - i) * self.e + self.tg

            if T_G_all < T_C_all:
                if origin == "current":
                    global_ondemand_current.append(eid)
                else:
                    global_ondemand_next.append(eid)
                cpu_time_i = (
                    self.cpu_time_table[min(tokens, cpu_table_max_idx)]
                    if (
                        self.cpu_time_table
                        and min(tokens, cpu_table_max_idx) < len(self.cpu_time_table)
                    )
                    else 0.0
                )
                T_C_all -= cpu_time_i

        schedule.global_queue_size = len(global_ondemand_current) + len(
            global_ondemand_next
        )

        # Step 2: Local reorder - ensure current layer gets enough bandwidth
        current_sorted = sorted(current_layer_experts, key=lambda x: x[1])

        T_C_local = sum(
            self.cpu_time_table[min(t, cpu_table_max_idx)]
            if (
                self.cpu_time_table
                and min(t, cpu_table_max_idx) < len(self.cpu_time_table)
            )
            else 0.0
            for _, t in current_sorted
        )

        local_ondemand = []
        n_g = current_layer_gpu_resident
        total_gpu_cost = 0.0

        for eid, tokens in current_sorted:
            n_gpu_experts = n_g + len(local_ondemand)
            T_G_local = (
                max(n_gpu_experts * self.tg, alpha + (n_gpu_experts + 1) * self.e)
                + self.tg
            )
            T_C_local_i = T_C_local
            cpu_time_i = (
                self.cpu_time_table[min(tokens, cpu_table_max_idx)]
                if (
                    self.cpu_time_table
                    and min(tokens, cpu_table_max_idx) < len(self.cpu_time_table)
                )
                else 0.0
            )

            if T_G_local < T_C_local_i:
                local_ondemand.append(eid)
                total_gpu_cost += self.e + self.tg
            T_C_local -= cpu_time_i

        schedule.ondemand_experts = local_ondemand[: self.max_ondemand]

        # Step 3: Confidence-aware prefetch
        T_G_final = (
            max(len(local_ondemand) * self.tg, alpha + len(local_ondemand) * self.e)
            + self.tg
        )
        T_C_final = T_C_local
        T_gap = T_C_final - T_G_final

        if T_gap > 0 and global_ondemand_next:
            f = T_gap / self.e if self.e > 0 else 0
            f_int = int(math.floor(f))

            for i, eid in enumerate(global_ondemand_next[: f_int + 1]):
                R_hit = max(0.5, 1.0 - i * 0.1)
                f_actual = f - i
                xi = (
                    R_hit * (f_actual) * self.e
                    - (1 - R_hit) * max(0, i - f + 1) * self.e
                )

                if xi > 0:
                    schedule.prefetch_experts.append(eid)
                    schedule.prefetch_confidences.append(R_hit)
                else:
                    break

            schedule.io_bubble_ms = T_gap

        return schedule

    # ================================================================
    # Unified dispatch
    # ================================================================

    def schedule(
        self,
        layer_idx: int,
        sorted_experts: List[Tuple[int, int]],
        is_decode: bool,
        next_layer_predicted: Optional[List[Tuple[int, int]]] = None,
        n_experts_on_gpu_current: int = 0,
        n_experts_on_gpu_next: int = 0,
    ) -> dict:
        """
        Unified scheduling dispatch.

        Returns dict with keys:
            ondemand_experts, prefetch_experts, mode, n_g_optimal
        """
        if self.schedule_mode == "legacy":
            ondemand = self.decide_ondemand(sorted_experts, is_decode)
            return {
                "ondemand_experts": ondemand,
                "prefetch_experts": [],
                "mode": "legacy",
                "n_g_optimal": 0,
            }

        if is_decode:
            total_activated = len(sorted_experts)
            activated_ids = [eid for eid, _ in sorted_experts]
            ds = self.decide_decode_schedule(
                layer_idx,
                activated_ids,
                n_experts_on_gpu_current,
                n_experts_on_gpu_next,
                total_activated,
            )
            return {
                "ondemand_experts": ds.ondemand_experts,
                "prefetch_experts": ds.prefetch_experts,
                "mode": ds.mode,
                "n_g_optimal": ds.n_g_optimal,
            }
        else:
            if next_layer_predicted is not None:
                ps = self.decide_prefill_schedule(
                    layer_idx,
                    sorted_experts,
                    next_layer_predicted,
                    n_experts_on_gpu_current,
                    n_experts_on_gpu_next,
                )
                return {
                    "ondemand_experts": ps.ondemand_experts,
                    "prefetch_experts": ps.prefetch_experts,
                    "mode": "prefill_3step",
                    "n_g_optimal": 0,
                    "prefetch_confidences": ps.prefetch_confidences,
                    "io_bubble_ms": ps.io_bubble_ms,
                }
            else:
                ondemand = self.decide_ondemand(sorted_experts, is_decode)
                return {
                    "ondemand_experts": ondemand,
                    "prefetch_experts": [],
                    "mode": "prefill_ondemand",
                    "n_g_optimal": 0,
                }
