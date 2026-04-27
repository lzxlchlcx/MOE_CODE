"""
MoE 框架统一配置管理模块 (PDScope 版本)

新增:
- schedule_mode: pdscope / legacy 调度模式
- tc: CPU 单专家单 token 计算时间 (Decode 负载均衡需要)
- attention_time_table: Attention 计算时间表
- num_transfer_streams: 并发传输 CUDA Stream 数量
- schedule_thread_core: 后台调度线程绑定的 CPU 核心
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Optional
import torch


@dataclass
class ModelConfig:
    model_name: str = "deepseek"
    model_path: str = "/mnt/g/Models/DeepSeek-v2-lite-chat"
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_layer: int = 26
    hidden_size: int = 2048
    top_k: int = 6

    expert_param_names: List[str] = field(
        default_factory=lambda: ["gate_proj", "up_proj", "down_proj"]
    )

    e: float = 1.11
    tg: float = 0.20
    tc: float = 0.0
    cpu_time_table: List[float] = field(default_factory=list)
    cpu_time_table_file: str = ""
    attention_time_table: List[float] = field(default_factory=list)

    batch_size: int = 1
    cache: int = 4
    cpu_offload: int = 1
    beam_width: int = 1
    prefetch_enabled: bool = False
    num_placeholders: int = 6
    hot_expert_file: str = "./hot/deep.txt"
    cpu_threads: int = 16

    schedule_mode: str = "pdscope"
    num_transfer_streams: int = 3
    schedule_thread_core: int = -1

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
        config = cls(
            model_path=args.model,
            batch_size=args.batch_size,
            cache=args.cache,
            cpu_offload=args.cpu_offload,
            beam_width=args.beam_width,
        )
        config._infer_model_name()
        if hasattr(args, "prefetch"):
            config.prefetch_enabled = args.prefetch == "enabled"
        if hasattr(args, "log_dir"):
            config.log_dir = args.log_dir
        if hasattr(args, "schedule"):
            config.schedule_mode = args.schedule
        config._apply_default_values()
        return config

    def _infer_model_name(self):
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
        defaults = get_default_config_dict(self.model_name)
        model_specific_keys = [
            "n_routed_experts",
            "n_shared_experts",
            "n_layer",
            "hidden_size",
            "expert_param_names",
            "e",
            "tg",
            "tc",
            "hot_expert_file",
            "cpu_time_table_file",
            "top_k",
            "num_placeholders",
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
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    transfer_time_ms: float = 0.0
    gpu_compute_time_ms: float = 0.0
    single_token_cpu_time_ms: float = 0.0
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
    defaults = {
        "deepseek": {
            "model_name": "deepseek",
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "n_layer": 26,
            "hidden_size": 2048,
            "top_k": 6,
            "expert_param_names": ["gate_proj", "up_proj", "down_proj"],
            "e": 1.11,
            "tg": 0.20,
            "tc": 0.0,
            "hot_expert_file": "./hot/deep.txt",
            "cpu_time_table_file": "microdeepseek.txt",
        },
        "qwen": {
            "model_name": "qwen",
            "n_routed_experts": 128,
            "n_shared_experts": 0,
            "n_layer": 48,
            "hidden_size": 2048,
            "top_k": 8,
            "expert_param_names": ["gate_proj", "up_proj", "down_proj"],
            "e": 0.55,
            "tg": 0.85,
            "tc": 0.0,
            "hot_expert_file": "./hot/qwen.txt",
            "cpu_time_table_file": "microqwen.txt",
        },
        "moon": {
            "model_name": "moon",
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "n_layer": 26,
            "hidden_size": 2048,
            "top_k": 6,
            "expert_param_names": ["gate_proj", "up_proj", "down_proj"],
            "e": 1.39,
            "tg": 0.95,
            "tc": 0.0,
            "hot_expert_file": "./hot/moon.txt",
            "cpu_time_table_file": "micromoon.txt",
        },
        "mixtral": {
            "model_name": "mixtral",
            "n_routed_experts": 8,
            "n_shared_experts": 0,
            "n_layer": 32,
            "hidden_size": 4096,
            "top_k": 2,
            "expert_param_names": ["w1", "w2", "w3"],
            "e": 27.0,
            "tg": 4.22,
            "tc": 0.0,
            "hot_expert_file": "./hot/mix.txt",
            "cpu_time_table_file": "micromixtral.txt",
        },
    }
    return defaults.get(model_name, defaults["deepseek"])


def get_default_config(model_name: str) -> ModelConfig:
    return ModelConfig(**get_default_config_dict(model_name))


def merge_system_to_model(
    system_config: SystemConfig, model_config: ModelConfig
) -> ModelConfig:
    model_config.e = system_config.transfer_time_ms
    model_config.tg = system_config.gpu_compute_time_ms
    model_config.tc = system_config.single_token_cpu_time_ms
    model_config.cpu_time_table = system_config.cpu_time_table
    model_config.attention_time_table = system_config.attention_time_table
    return model_config


def load_cpu_time_table_from_file(file_path: str) -> List[float]:
    if not os.path.exists(file_path):
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
            for line in lines[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 4:
                    cpu_time = parts[3]
                    table.append(float(cpu_time) if cpu_time != "NaN" else 0.0)
        else:
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
    if torch.cuda.is_available():
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory
            / (1024**3),
        }
    return {"gpu_name": "Unknown", "gpu_memory_gb": 0.0}
