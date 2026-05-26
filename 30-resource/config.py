"""
MoE 框架统一配置管理模块

提供:
- ModelConfig: 模型运行时配置
- SystemConfig: 微基准测试生成的系统配置
- JSON 配置文件的加载/保存
- 默认配置获取
- 从 microbench 结果自动生成配置
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Optional
import torch


@dataclass
class ModelConfig:
    """MoE 模型运行配置"""

    # 模型基本信息
    model_name: str = "deepseek"
    model_path: str = "/mnt/g/Models/DeepSeek-v2-lite-chat"
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_layer: int = 26
    hidden_size: int = 2048

    # 专家权重名称 (不同模型的差异)
    expert_param_names: List[str] = field(
        default_factory=lambda: ["gate_proj", "up_proj", "down_proj"]
    )

    # 性能参数 (来自 microbench)
    e: float = 1.11
    tg: float = 0.20
    tc: float = 0.0
    cpu_time_table: List[float] = field(default_factory=list)
    cpu_time_table_file: str = ""

    def get_tc(self) -> float:
        if self.tc > 0:
            return self.tc
        if self.cpu_time_table and len(self.cpu_time_table) > 1:
            return self.cpu_time_table[1]
        return 5.0

    # 运行配置
    batch_size: int = 1
    cache: int = 2
    cpu_offload: int = 1
    beam_width: int = 1
    prefetch_enabled: bool = False
    num_placeholders: int = 6
    hot_expert_file: str = "./hot/deep.txt"
    cpu_threads: int = 16

    # 日志配置
    log_dir: str = "./log"

    def to_dict(self):
        return asdict(self)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_args(cls, args) -> "ModelConfig":
        """从命令行参数构建配置"""
        config = cls(
            model_path=args.model,
            batch_size=args.batch_size,
            cache=args.cache,
            cpu_offload=args.cpu_offload,
            beam_width=args.beam_width,
        )

        # 从路径推断模型名
        config._infer_model_name()

        # 应用预取设置
        if hasattr(args, "prefetch"):
            config.prefetch_enabled = args.prefetch == "enabled"

        # 应用日志目录
        if hasattr(args, "log_dir"):
            config.log_dir = args.log_dir

        # 加载默认值
        config._apply_default_values()

        return config

    def _infer_model_name(self):
        """从模型路径推断模型名称"""
        model_path_lower = self.model_path.lower()
        if "deepseek" in model_path_lower:
            self.model_name = "deepseek"
        elif "qwen" in model_path_lower:
            self.model_name = "qwen"
        elif "moonlight" in model_path_lower or "moon" in model_path_lower:
            self.model_name = "moon"
        elif "mixtral" in model_path_lower:
            self.model_name = "mixtral"

    def _apply_default_values(self):
        """根据模型名应用默认值"""
        defaults = get_default_config_dict(self.model_name)
        model_specific_keys = [
            "n_routed_experts",
            "n_shared_experts",
            "n_layer",
            "hidden_size",
            "expert_param_names",
            "e",
            "tg",
            "hot_expert_file",
            "cpu_time_table_file",
        ]
        for key in model_specific_keys:
            if key in defaults:
                setattr(self, key, defaults[key])
        for key, value in defaults.items():
            if key not in model_specific_keys and getattr(self, key, None) in [
                0,
                "",
                [],
            ]:
                setattr(self, key, value)


@dataclass
class SystemConfig:
    """微基准测试生成的系统配置"""

    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    transfer_time_ms: float = 0.0
    gpu_compute_time_ms: float = 0.0
    cpu_time_table: List[float] = field(default_factory=list)
    attention_time_table: List[float] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "SystemConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def get_default_config_dict(model_name: str) -> dict:
    """获取指定模型的默认配置字典"""

    defaults = {
        "deepseek": {
            "model_name": "deepseek",
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "n_layer": 26,
            "hidden_size": 2048,
            "expert_param_names": ["gate_proj", "up_proj", "down_proj"],
            "e": 1.11,
            "tg": 0.20,
            "hot_expert_file": "./hot/deep.txt",
            "cpu_time_table_file": "microdeepseek.txt",
        },
        "qwen": {
            "model_name": "qwen",
            "n_routed_experts": 128,
            "n_shared_experts": 0,
            "n_layer": 48,
            "hidden_size": 2048,
            "expert_param_names": ["gate_proj", "up_proj", "down_proj"],
            "e": 0.55,
            "tg": 0.85,
            "hot_expert_file": "./hot/qwen.txt",
            "cpu_time_table_file": "microqwen.txt",
        },
        "moon": {
            "model_name": "moon",
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "n_layer": 26,
            "hidden_size": 2048,
            "expert_param_names": ["gate_proj", "up_proj", "down_proj"],
            "e": 1.39,
            "tg": 0.95,
            "hot_expert_file": "./hot/moon.txt",
            "cpu_time_table_file": "micromoon.txt",
        },
        "mixtral": {
            "model_name": "mixtral",
            "n_routed_experts": 8,
            "n_shared_experts": 0,
            "n_layer": 32,
            "hidden_size": 4096,
            "expert_param_names": ["w1", "w2", "w3"],
            "e": 27.0,
            "tg": 4.22,
            "hot_expert_file": "./hot/mix.txt",
            "cpu_time_table_file": "micromixtral.txt",
        },
    }

    return defaults.get(model_name, defaults["deepseek"])


def get_default_config(model_name: str) -> ModelConfig:
    """获取指定模型的默认配置对象"""
    return ModelConfig(**get_default_config_dict(model_name))


def merge_system_to_model(
    system_config: SystemConfig, model_config: ModelConfig
) -> ModelConfig:
    """将系统配置合并到模型配置"""
    model_config.e = system_config.transfer_time_ms
    model_config.tg = system_config.gpu_compute_time_ms
    model_config.cpu_time_table = system_config.cpu_time_table
    return model_config


def load_cpu_time_table_from_file(file_path: str) -> List[float]:
    """从 microbench 输出文件加载 CPU 时间表

    支持两种格式:
    1. 旧格式 (micro*.txt): 每行一个浮点数，直接就是 CPU 时间
    2. 新格式 (micro*_New.txt): CSV 格式，第4列为 CPU 时间（已废弃，统一用 micro*.txt）
    """
    if not os.path.exists(file_path):
        # 尝试 _New 后缀
        base, ext = os.path.splitext(file_path)
        alt_path = f"{base}_New{ext}"
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            return []

    table = []
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        if not lines:
            return []

        first_line = lines[0].strip()
        if "," in first_line:
            # 新格式: CSV with header
            for line in lines[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 4:
                    cpu_time = parts[3]
                    table.append(float(cpu_time) if cpu_time != "NaN" else 0.0)
        else:
            # 旧格式: 每行一个浮点数
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        table.append(float(line))
                    except ValueError:
                        continue
    except Exception:
        pass
    return table


def auto_detect_gpu_info() -> dict:
    """自动检测GPU信息"""
    if torch.cuda.is_available():
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory
            / (1024**3),
        }
    return {"gpu_name": "Unknown", "gpu_memory_gb": 0.0}
