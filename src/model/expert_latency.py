from typing import Dict


def _lookup_latency(table: Dict[int, float], token_count: int) -> float:
    if token_count in table:
        return table[token_count]
    max_tc = max(table.keys())
    if token_count >= max_tc:
        return table[max_tc]
    closest = min(table.keys(), key=lambda k: abs(k - token_count))
    return table[closest]


class ExpertLatencyModel:
    def __init__(
        self,
        t_io: float,
        latency_cpu_table: Dict[int, float],
        latency_gpu_table: Dict[int, float],
    ):
        self.t_io = t_io
        self.latency_cpu_table = latency_cpu_table
        self.latency_gpu_table = latency_gpu_table

    def cpu(self, token_count: int) -> float:
        return _lookup_latency(self.latency_cpu_table, max(1, int(token_count)))

    def gpu_compute(self, token_count: int) -> float:
        return _lookup_latency(self.latency_gpu_table, max(1, int(token_count)))

    def transfer(self, layer: int, expert_id: int) -> float:
        return self.t_io

    def gpu_total(self, token_count: int, resident: bool, layer: int, expert_id: int) -> float:
        cost = self.gpu_compute(token_count)
        if not resident:
            cost += self.transfer(layer, expert_id)
        return cost
