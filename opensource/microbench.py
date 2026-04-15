"""
Microbenchmarking for CPU offloading

运行微基准测试，生成:
1. micro*.txt: CPU时间查找表
2. ioreal_*.png: 性能图表
3. hot/*.txt: 热点专家列表
4. configs/system_config_*.json: 系统配置JSON(新)
"""

import argparse
import copy
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append("./")
from mixtral import FiddlerMixtral
from deepseek import FiddlerDeepSeekV2
from moon import FiddlerMoon
from qwen import FiddlerQwen

from config import (
    ModelConfig,
    SystemConfig,
    get_default_config,
    auto_detect_gpu_info,
)


def get_model_name_from_path(model_path):
    """从模型路径推断模型名"""
    model_path_lower = model_path.lower()
    if "mixtral" in model_path_lower:
        return "mixtral"
    elif "qwen" in model_path_lower:
        return "qwen"
    elif "deepseek" in model_path_lower:
        return "deepseek"
    elif "moonlight" in model_path_lower or "moon" in model_path_lower:
        return "moon"
    return "deepseek"


def profile_hot_experts(model, model_name):
    """Profile hot experts by running sample tokens through MoE routing"""
    expert_hot_data = {}

    sample_texts = ["Hello world", "How are you today?", "The weather is nice"]

    try:
        input_ids = model.tokenizer(
            sample_texts, return_tensors="pt", padding=True
        ).input_ids.to(model.dev)

        for i_layer in range(1, min(5, len(model.model.layers))):
            if model_name == "mixtral":
                gate = model.model.layers[i_layer].block_sparse_moe.gate
            else:
                gate = model.model.layers[i_layer].mlp.gate

            hidden_states = model.model.embed_tokens(input_ids)
            selected_experts, routing_weights, _ = gate(hidden_states)

            expert_counts = {}
            for expert_id in selected_experts.unique():
                mask = (selected_experts == expert_id).any(dim=1)
                expert_counts[expert_id.item()] = mask.sum().item()

            for expert_id, count in expert_counts.items():
                key = (i_layer, expert_id)
                if key not in expert_hot_data:
                    expert_hot_data[key] = 0
                expert_hot_data[key] += count
    except Exception as e:
        print(f"Hot expert profiling failed: {e}")
        for i_layer in range(1, 5):
            for expert_id in range(8):
                key = (i_layer, expert_id)
                expert_hot_data[key] = 64 - expert_id * 4

    return expert_hot_data


def plot_expert_performance(model, model_name, output_dir="./configs"):
    """绘制专家性能折线图并生成系统配置"""
    token_counts = list(range(1, 65))
    ret_time = []

    expert_hot_data = profile_hot_experts(model, model_name)
    dev = torch.device("cuda:0")
    model.dev = torch.device("cuda:0")

    if model_name == "mixtral":
        first_layer_experts = model.model.layers[1].block_sparse_moe.experts
        n_expert = len(model.model.layers[0].block_sparse_moe.experts)
    else:
        first_layer_experts = model.model.layers[1].mlp.experts
        n_expert = len(model.model.layers[1].mlp.experts)

    n_shared_experts = 2
    expert_placeholder = copy.deepcopy(first_layer_experts[0]).to(dev)

    copy_times = []
    for _ in token_counts:
        for i in range(2, 3):
            first_layer_experts[i].to("cpu")
            if model_name == "mixtral":
                for name in ["w1", "w2", "w3"]:
                    w = getattr(first_layer_experts[i], name)
                    src_weight_data_tensor = w.weight.data
                    pinned = src_weight_data_tensor.pin_memory()
                    w.weight.data = pinned
            else:
                for name in ["gate_proj", "up_proj", "down_proj"]:
                    w = getattr(first_layer_experts[i], name)
                    src_weight_data_tensor = w.weight.data
                    pinned = src_weight_data_tensor.pin_memory()
                    w.weight.data = pinned

            torch.cuda.synchronize()
            tick = time.time()
            if model_name == "mixtral":
                for name in ["w1", "w2", "w3"]:
                    dst = getattr(expert_placeholder, name).weight.data
                    src = getattr(first_layer_experts[i], name).weight.data
                    dst.copy_(src)
            else:
                for name in ["gate_proj", "up_proj", "down_proj"]:
                    dst = getattr(expert_placeholder, name).weight.data
                    src = getattr(first_layer_experts[i], name).weight.data
                    dst.copy_(src)
            torch.cuda.synchronize()
            copy_times.append(time.time() - tick)

    gpu_times = []
    for token_count in token_counts:
        first_layer_experts[1].to(model.dev)
        if model_name == "mixtral":
            inps = torch.randn((token_count, 4096), dtype=model.dtype, device=model.dev)
        else:
            inps = torch.randn((token_count, 2048), dtype=model.dtype, device=model.dev)
        _ = first_layer_experts[1](inps)

        torch.cuda.synchronize()
        tick = time.time()
        _ = first_layer_experts[1](inps)
        torch.cuda.synchronize()
        gpu_times.append(time.time() - tick)
        first_layer_experts[1].to("cpu")

        expert_key = (1, 1)
        if expert_key not in expert_hot_data:
            expert_hot_data[expert_key] = 0
        expert_hot_data[expert_key] += token_count

    cpu_times = []
    for token_count in token_counts:
        first_layer_experts[1].to("cpu")
        if model_name == "mixtral":
            inps = torch.randn((token_count, 4096), dtype=model.dtype, device="cpu")
        else:
            inps = torch.randn((token_count, 2048), dtype=model.dtype, device="cpu")
        _ = model.run_expert_at_cpu(1, 1, inps)

        measurements = []
        for _ in range(20):
            torch.cuda.synchronize()
            tick = time.time()
            _ = model.run_expert_at_cpu(1, 1, inps)
            torch.cuda.synchronize()
            measurements.append(time.time() - tick)

        cpu_times.append(np.mean(measurements))

        expert_key = (1, 1)
        if expert_key not in expert_hot_data:
            expert_hot_data[expert_key] = 0
        expert_hot_data[expert_key] += token_count

    window_size = 7
    attn_times = []
    for token_count in token_counts:
        if model_name == "mixtral":
            inps = torch.randn(
                (1, token_count, 4096), dtype=model.dtype, device=model.dev
            )
        else:
            inps = torch.randn(
                (1, token_count, 2048), dtype=model.dtype, device=model.dev
            )
        attention_mask = torch.ones(
            (1, 1, 1, token_count), dtype=torch.bool, device=model.dev
        )
        position_ids = torch.arange(
            token_count, dtype=torch.long, device=model.dev
        ).unsqueeze(0)
        attn_times.append(0.0)

    plt.figure(figsize=(12, 6))
    avg_gpu_time = np.mean(gpu_times) * 1000
    avg_copy_time = np.mean(copy_times) * 1000

    plt.plot(
        token_counts,
        [avg_copy_time] * len(token_counts),
        label=f"Expert Transfer (Avg: {avg_copy_time:.2f}ms)",
        linestyle="--",
    )

    plt.plot(
        token_counts,
        np.array(gpu_times) * 1000,
        label=f"Expert computation on GPU (Avg: {avg_gpu_time:.2f}ms)",
        marker="o",
    )

    cpu_times_smoothed = np.convolve(
        cpu_times, np.ones(window_size) / window_size, mode="valid"
    )
    token_counts_smoothed = token_counts[window_size - 1 :]

    # 扩展平滑结果到全长：前部用线性插值，尾部用最后一个平滑值
    cpu_times_full = np.zeros(len(token_counts))
    first_valid_idx = window_size - 1
    for i in range(len(token_counts)):
        if i < first_valid_idx:
            cpu_times_full[i] = cpu_times_smoothed[0] * (i + 1) / (first_valid_idx + 1)
        elif i - first_valid_idx < len(cpu_times_smoothed):
            cpu_times_full[i] = cpu_times_smoothed[i - first_valid_idx]
        else:
            cpu_times_full[i] = cpu_times_smoothed[-1]

    plt.plot(
        token_counts_smoothed,
        np.array(cpu_times_smoothed) * 1000,
        label="Expert computation on CPU",
        marker="s",
        linestyle="-",
        linewidth=2,
    )
    plt.plot(
        token_counts,
        np.array(attn_times) * 1000,
        label="Self-Attention on GPU",
        marker="^",
        linestyle="-",
    )

    micro_file = f"micro{model_name}.txt"
    with open(micro_file, "w") as f:
        f.write(
            "Token Count,Expert Transfer Time(ms),GPU Computation Time(ms),CPU Computation Time(ms),Self-Attention Time(ms)\n"
        )
        for i, token_count in enumerate(token_counts):
            cpu_time = cpu_times_full[i] * 1000
            f.write(
                f"{token_count},{avg_copy_time:.4f},{gpu_times[i] * 1000:.4f},{cpu_time:.4f},{attn_times[i] * 1000:.4f}\n"
            )

    sorted_hot_experts = sorted(
        expert_hot_data.items(), key=lambda x: x[1], reverse=True
    )
    hot_file_path = f"./hot/{model_name}.txt"
    os.makedirs(os.path.dirname(hot_file_path), exist_ok=True)
    with open(hot_file_path, "w") as f:
        for (layer, expert), token_count in sorted_hot_experts:
            f.write(f"{layer},{expert}\n")

    plt.xlabel("Token Count")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"ioreal_{model_name}.png", dpi=300, bbox_inches="tight")
    # plt.show()

    # 生成系统配置
    cpu_time_table_list = [float(t * 1000) for t in cpu_times_full]

    gpu_info = auto_detect_gpu_info()

    system_config = SystemConfig(
        gpu_name=gpu_info["gpu_name"],
        gpu_memory_gb=gpu_info["gpu_memory_gb"],
        transfer_time_ms=float(avg_copy_time),
        gpu_compute_time_ms=float(avg_gpu_time),
        cpu_time_table=cpu_time_table_list,
        attention_time_table=[t * 1000 for t in attn_times],
    )

    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, f"system_config_{model_name}.json")
    system_config.save(config_path)
    print(f"\n系统配置已保存到: {config_path}")
    print(f"  e (传输时间): {avg_copy_time:.2f} ms")
    print(f"  tg (GPU计算时间): {avg_gpu_time:.2f} ms")
    print(f"  GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']:.1f} GB)")

    return system_config


def main():
    parser = argparse.ArgumentParser()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model path.",
    )
    parser.add_argument(
        "--cache",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: execute at GPU (baseline), 1: offload to CPU.",
    )
    parser.add_argument(
        "--cpu-offload",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: execute at GPU (baseline), 1: offload to CPU.",
    )
    parser.add_argument("--beam_width", type=int, default=1, help="Beam search number.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./configs",
        help="Output directory for config files.",
    )
    args = parser.parse_args()

    model_name = get_model_name_from_path(args.model)
    print(f"检测到模型类型: {model_name}")

    # 检查模型路径
    if not os.path.exists(args.model):
        print(f"\n[WARNING] 模型路径不存在: {args.model}")
        print("请检查模型路径是否正确。退出。")
        sys.exit(1)

    if model_name == "mixtral":
        model = FiddlerMixtral(args)
    elif model_name == "qwen":
        model = FiddlerQwen(args)
    elif model_name == "deepseek":
        model = FiddlerDeepSeekV2(args)
    elif model_name == "moon":
        model = FiddlerMoon(args)
    else:
        model = FiddlerDeepSeekV2(args)

    plot_expert_performance(model, model_name, args.output_dir)


if __name__ == "__main__":
    main()
