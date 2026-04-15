"""
MoE 专家调度策略模块

提供:
- ExpertScheduler: 专家调度策略基类
- ondemand 决策算法（从各模型文件提取）
- 预取策略开关
"""

from typing import List, Tuple
from config import ModelConfig


class ExpertScheduler:
    """专家调度策略类"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.e = config.e
        self.tg = config.tg
        self.cpu_time_table = config.cpu_time_table
        self.max_ondemand = (
            4  # decode阶段最多ondemand搬运4个专家（与placeholder数量匹配）
        )

    def decide_ondemand(
        self, sorted_experts: List[Tuple[int, int]], is_decode: bool
    ) -> List[int]:
        """
        根据 e/tg 和 CPU 时间表决定哪些专家做 ondemand 加载

        参数:
            sorted_experts: 按token数降序排序的专家列表 [(expert_id, token_count), ...]
            is_decode: 是否为decode阶段

        返回:
            ondemand_experts: 需要做ondemand加载的专家ID列表
        """
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
                        elif TC - TG > self.e / 2:
                            self._set_prefil_pre(True)
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

    def should_prefetch(self, layer_id: int, expert_id: int) -> bool:
        """
        是否预取该专家

        返回:
            bool: 是否应该预取
        """
        return self.config.prefetch_enabled

    def _set_prefil_pre(self, value: bool):
        """设置 prefil_pre 标志（供模型类使用）"""
        pass

    def set_cpu_time_table(self, table: List[float]):
        """设置 CPU 时间表"""
        self.cpu_time_table = table
        self.config.cpu_time_table = table
