"""
MoE 框架统一日志系统

提供:
- RunMetrics: 运行时指标收集器
- BenchmarkLogger: 日志记录器
- BenchmarkReport: 运行报告 + 自动分析
"""

import json
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

from config import ModelConfig


@dataclass
class RunMetrics:
    """运行时指标收集器"""

    # 专家命中统计
    expert_hit_count: int = 0
    expert_total_count: int = 0

    # 层时间统计
    layer_gpu_times: Dict[int, List[float]] = field(default_factory=dict)
    layer_cpu_times: Dict[int, List[float]] = field(default_factory=dict)
    layer_parallel_times: Dict[int, List[float]] = field(default_factory=dict)

    # Token时间统计
    token_decode_times: List[float] = field(default_factory=list)

    # 专家统计
    expert_stats: List[Dict] = field(default_factory=list)

    # 性能各阶段统计
    perf_stats: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "token_embedding": [],
            "self_attention": [],
            "moe_gating": [],
            "expert_compute": [],
            "expert_compute-cpu": [],
        }
    )

    # 总体时间
    prefill_time: float = 0.0
    decode_time: float = 0.0

    # 批处理和输出配置
    batch_size: int = 1
    output_token: int = 32
    n_sample: int = 5

    # 专家分类统计 (运行时收集)
    expert_classification: Dict[str, int] = field(
        default_factory=lambda: {
            "gpu": 0,
            "cpu": 0,
            "ondemand": 0,
            "prefetch": 0,
            "placeholder": 0,
            "loading": 0,
        }
    )


class BenchmarkLogger:
    """基准测试日志记录器"""

    def __init__(self, config: ModelConfig, log_dir: Optional[str] = None):
        self.config = config
        self.log_dir = log_dir or config.log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.metrics = RunMetrics(batch_size=config.batch_size)

        self._start_time = time.time()
        self._timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def log_expert_hit(self, is_hit: bool):
        """记录专家命中"""
        self.metrics.expert_total_count += 1
        if is_hit:
            self.metrics.expert_hit_count += 1

    def log_layer_stats(
        self, layer_id: int, gpu_time: float, cpu_time: float, parallel_time: float
    ):
        """记录每层时间统计"""
        if layer_id not in self.metrics.layer_gpu_times:
            self.metrics.layer_gpu_times[layer_id] = []
            self.metrics.layer_cpu_times[layer_id] = []
            self.metrics.layer_parallel_times[layer_id] = []

        self.metrics.layer_gpu_times[layer_id].append(gpu_time)
        self.metrics.layer_cpu_times[layer_id].append(cpu_time)
        self.metrics.layer_parallel_times[layer_id].append(parallel_time)

    def log_expert_stats(
        self,
        layer_id: int,
        expert_id: int,
        device: str,
        token_count: int,
        time_ms: float,
        status: str = "normal",
    ):
        """记录专家统计"""
        self.metrics.expert_stats.append(
            {
                "layer_id": layer_id,
                "expert_id": expert_id,
                "device": device,
                "token_count": token_count,
                "time_ms": time_ms,
                "status": status,
            }
        )

        if device in self.metrics.expert_classification:
            self.metrics.expert_classification[device] += 1

    def log_expert_class(self, expert_class: str):
        """记录专家分类（用于统计各类型专家数量）"""
        if expert_class in self.metrics.expert_classification:
            self.metrics.expert_classification[expert_class] += 1

    def log_token_decode(self, token_idx: int, time_ms: float):
        """记录每个token的decode时间"""
        self.metrics.token_decode_times.append(time_ms)

    def log_perf_stat(self, stat_name: str, time_sec: float):
        """记录性能各阶段统计"""
        if stat_name in self.metrics.perf_stats:
            self.metrics.perf_stats[stat_name].append(time_sec)

    def set_prefill_decode_times(self, prefill_time: float, decode_time: float):
        """设置prefill和decode总时间"""
        self.metrics.prefill_time = prefill_time
        self.metrics.decode_time = decode_time

    def set_output_config(self, output_token: int, n_sample: int):
        """设置输出配置"""
        self.metrics.output_token = output_token
        self.metrics.n_sample = n_sample

    def finalize(self) -> "BenchmarkReport":
        """运行结束，生成报告"""
        total_time = self.metrics.prefill_time + self.metrics.decode_time

        report = BenchmarkReport(
            config=self.config,
            metrics=self.metrics,
            timestamp=self._timestamp,
            total_runtime=time.time() - self._start_time,
        )

        return report


class BenchmarkReport:
    """基准测试报告 + 自动分析"""

    def __init__(
        self,
        config: ModelConfig,
        metrics: RunMetrics,
        timestamp: str,
        total_runtime: float,
    ):
        self.config = config
        self.metrics = metrics
        self.timestamp = timestamp
        self.total_runtime = total_runtime

        # 计算衍生指标
        self._calculate_derived_metrics()

    def _calculate_derived_metrics(self):
        """计算衍生指标"""
        # 命中率
        self.hit_rate = (
            self.metrics.expert_hit_count / self.metrics.expert_total_count
            if self.metrics.expert_total_count > 0
            else 0.0
        )

        # 平均并行度
        self._calculate_parallel_degree()

        # 吞吐率
        total_tokens = (
            self.metrics.output_token * self.metrics.batch_size * self.metrics.n_sample
        )
        total_time = (
            self.metrics.prefill_time + self.metrics.decode_time
        ) * self.metrics.n_sample
        self.throughput = total_tokens / total_time if total_time > 0 else 0.0

        # 平均decode时间
        if self.metrics.token_decode_times:
            self.avg_decode_time_ms = np.mean(self.metrics.token_decode_times) * 1000
        else:
            self.avg_decode_time_ms = 0.0

        # 层时间分布
        self._calculate_layer_distribution()

    def _calculate_parallel_degree(self):
        """计算平均并行度"""
        all_gpu_times = []
        all_cpu_times = []
        all_parallel_times = []

        for layer_id in self.metrics.layer_gpu_times:
            all_gpu_times.extend(self.metrics.layer_gpu_times[layer_id])
            all_cpu_times.extend(self.metrics.layer_cpu_times[layer_id])
            all_parallel_times.extend(self.metrics.layer_parallel_times[layer_id])

        if all_parallel_times and len(all_parallel_times) > 0:
            total_gpu_time = sum(all_gpu_times)
            total_cpu_time = sum(all_cpu_times)
            total_parallel_time = sum(all_parallel_times)

            if total_parallel_time > 0:
                self.avg_parallel_degree = (
                    total_gpu_time + total_cpu_time
                ) / total_parallel_time
            else:
                self.avg_parallel_degree = 1.0
        else:
            self.avg_parallel_degree = 1.0

    def _calculate_layer_distribution(self):
        """计算层时间分布"""
        self.layer_avg_times = {}
        self.layer_time_variance = 0.0

        all_layer_times = []

        for layer_id in self.metrics.layer_parallel_times:
            times = self.metrics.layer_parallel_times[layer_id]
            if times:
                avg_time = np.mean(times)
                self.layer_avg_times[layer_id] = avg_time
                all_layer_times.append(avg_time)

        if len(all_layer_times) > 1:
            self.layer_time_variance = np.var(all_layer_times)

    def analyze(self) -> List[str]:
        """自动检测调优空间，返回建议列表"""
        suggestions = []

        # 1. 专家命中率检查
        if self.hit_rate < 0.30:
            suggestions.append(
                f"专家命中率较低 ({self.hit_rate * 100:.1f}%)，建议：增加GPU常驻专家数或优化热点策略"
            )
        elif self.hit_rate < 0.50:
            suggestions.append(
                f"专家命中率适中 ({self.hit_rate * 100:.1f}%)，可考虑优化热点专家选择"
            )

        # 2. 并行度检查
        if self.avg_parallel_degree < 1.3:
            suggestions.append(
                f"并行度较低 ({self.avg_parallel_degree:.2f}x)，建议：调整e/tg参数或减少ondemand专家数量"
            )

        # 3. CPU专家占比检查
        total_experts = sum(self.metrics.expert_classification.values())
        if total_experts > 0:
            cpu_ratio = self.metrics.expert_classification.get("cpu", 0) / total_experts
            if cpu_ratio > 0.50:
                suggestions.append(
                    f"CPU专家占比较高 ({cpu_ratio * 100:.1f}%)，建议：增加热点专家或增大batch_size"
                )

        # 4. 层时间方差检查
        if self.layer_time_variance > 0.001:
            bottleneck_layers = sorted(
                self.layer_avg_times.items(), key=lambda x: x[1], reverse=True
            )[:3]

            layer_str = ", ".join(
                [f"层{lid}({t * 1000:.1f}ms)" for lid, t in bottleneck_layers]
            )
            suggestions.append(f"各层时间方差较大，潜在瓶颈层: {layer_str}")

        # 5. 吞吐率检查（根据模型和batch_size给出参考）
        if self.config.batch_size == 1 and self.throughput < 5:
            suggestions.append(
                f"吞吐率较低 ({self.throughput:.2f} token/s)，建议尝试增大batch_size"
            )

        return suggestions

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp,
            "total_runtime": self.total_runtime,
            "config": self.config.to_dict(),
            "metrics": asdict(self.metrics),
            "derived": {
                "hit_rate": self.hit_rate,
                "avg_parallel_degree": self.avg_parallel_degree,
                "throughput": self.throughput,
                "avg_decode_time_ms": self.avg_decode_time_ms,
                "layer_avg_times": {str(k): v for k, v in self.layer_avg_times.items()},
                "layer_time_variance": self.layer_time_variance,
            },
        }

    def save(self, dir_path: Optional[str] = None):
        """保存报告（JSON + TXT）"""
        dir_path = dir_path or self.config.log_dir
        os.makedirs(dir_path, exist_ok=True)

        base_path = os.path.join(dir_path, f"run_{self.timestamp}")

        # 保存 JSON
        json_path = f"{base_path}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

        # 保存 TXT（人类可读）
        txt_path = f"{base_path}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(self._generate_text_report())

        return json_path, txt_path

    def _generate_text_report(self) -> str:
        """生成人类可读的文本报告"""
        lines = []
        lines.append("=" * 80)
        lines.append("MoE 基准测试报告")
        lines.append(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")

        # 配置摘要
        lines.append("【配置摘要】")
        lines.append(f"  模型: {self.config.model_name}")
        lines.append(f"  模型路径: {self.config.model_path}")
        lines.append(f"  Batch Size: {self.config.batch_size}")
        lines.append(f"  CPU Offload: {self.config.cpu_offload}")
        lines.append(
            f"  预取策略: {'启用' if self.config.prefetch_enabled else '禁用'}"
        )
        lines.append(f"  e={self.config.e:.2f}ms, tg={self.config.tg:.2f}ms")
        lines.append("")

        # 性能摘要
        lines.append("【性能摘要】")
        lines.append(f"  Prefill时间: {self.metrics.prefill_time:.2f}s")
        lines.append(f"  Decode时间: {self.metrics.decode_time:.2f}s")
        lines.append(
            f"  总时间: {self.metrics.prefill_time + self.metrics.decode_time:.2f}s"
        )
        lines.append(f"  吞吐率: {self.throughput:.2f} token/s")
        lines.append(f"  平均Decode时间: {self.avg_decode_time_ms:.2f}ms/token")
        lines.append("")

        # 专家统计
        lines.append("【专家统计】")
        lines.append(
            f"  专家命中率: {self.hit_rate * 100:.1f}% ({self.metrics.expert_hit_count}/{self.metrics.expert_total_count})"
        )
        lines.append(f"  平均并行度: {self.avg_parallel_degree:.2f}x")
        lines.append("")

        # 专家分类
        lines.append("【专家分类】")
        total = sum(self.metrics.expert_classification.values())
        for cls, count in self.metrics.expert_classification.items():
            if total > 0:
                pct = count / total * 100
                lines.append(f"  {cls}: {count} ({pct:.1f}%)")
            else:
                lines.append(f"  {cls}: {count}")
        lines.append("")

        # 层时间Top 5
        if self.layer_avg_times:
            lines.append("【层时间Top 5 (最慢)】")
            sorted_layers = sorted(
                self.layer_avg_times.items(), key=lambda x: x[1], reverse=True
            )[:5]
            for lid, avg_time in sorted_layers:
                lines.append(f"  层 {lid}: {avg_time * 1000:.2f}ms")
            lines.append("")

        # 调优建议
        suggestions = self.analyze()
        if suggestions:
            lines.append("【调优建议】")
            for i, s in enumerate(suggestions, 1):
                lines.append(f"  {i}. {s}")
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    def print_summary(self):
        """打印摘要"""
        SEP = "=" * 60

        print(f"\n{SEP}")
        print("  MoE 基准测试结果摘要")
        print(SEP)

        print(f"  模型:       {self.config.model_name}")
        print(f"  Batch Size: {self.config.batch_size}")
        print(f"  预取策略:   {'启用' if self.config.prefetch_enabled else '禁用'}")
        print(f"  e/tg:       {self.config.e:.2f}ms / {self.config.tg:.2f}ms")
        print(f"{SEP}")

        hit_bar_len = 30
        hit_filled = int(self.hit_rate * hit_bar_len)
        hit_bar = "█" * hit_filled + "░" * (hit_bar_len - hit_filled)
        print(f"  命中率:     [{hit_bar}] {self.hit_rate * 100:.1f}%")

        par_bar_filled = min(
            int((self.avg_parallel_degree / 2.0) * hit_bar_len), hit_bar_len
        )
        par_bar = "█" * par_bar_filled + "░" * (hit_bar_len - par_bar_filled)
        print(f"  并行度:     [{par_bar}] {self.avg_parallel_degree:.2f}x")

        print(f"  吞吐率:     {self.throughput:.2f} token/s")
        print(f"  平均Decode: {self.avg_decode_time_ms:.2f}ms/token")
        print(f"{SEP}")

        suggestions = self.analyze()
        if suggestions:
            print("\n  【调优建议】")
            for i, s in enumerate(suggestions, 1):
                print(f"  {i}. {s}")
            print()
