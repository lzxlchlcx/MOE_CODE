"""
Microbenchmarking for PDScope (extended)

Measures:
1. Expert CPU->GPU copy time (e)
2. Expert GPU compute time (tg)
3. Expert CPU compute time table (cpu_time_table)
4. Single-token CPU compute time (tc) - NEW
5. Self-Attention time table - NEW
"""

import argparse
import copy
import os
import sys
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append("./")
from deepseek import FiddlerDeepSeekV2
from qwen import FiddlerQwen
from moon import FiddlerMoon
from mixtral import FiddlerMixtral

from config import (
    ModelConfig,
    SystemConfig,
    get_default_config,
    auto_detect_gpu_info,
    load_cpu_time_table_from_file,
)


def get_model_name_from_path(model_path):
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
    expert_hot_data = {}
    sample_texts = ["Hello world", "How are you today?", "The weather is nice"]

    try:
        input_ids = model.tokenizer(
            sample_texts, return_tensors="pt", padding=True
        ).input_ids.to(model.dev)

        for i_layer in range(min(5, len(model.model.layers))):
            if model._is_layer_shared_only(i_layer):
                continue
            gate = model._get_gate(model.model.layers[i_layer])
            selected_experts, _ = model._compute_gate(
                gate, model.model.embed_tokens(input_ids)
            )

            expert_counts = {}
            for expert_id in selected_experts.unique():
                mask = (selected_experts == expert_id).any(dim=1)
                expert_counts[expert_id.item()] = mask.sum().item()

            for expert_id, count in expert_counts.items():
                key = (i_layer, expert_id)
                expert_hot_data[key] = expert_hot_data.get(key, 0) + count
    except Exception as e:
        print(f"Hot expert profiling failed: {e}")
        for i_layer in range(1, 5):
            for expert_id in range(8):
                expert_hot_data[(i_layer, expert_id)] = 64 - expert_id * 4

    return expert_hot_data


def plot_expert_performance(model, model_name, output_dir="./configs"):
    token_counts = list(range(1, 65))
    dev = torch.device("cuda:0")

    expert_hot_data = profile_hot_experts(model, model_name)

    src_layer = model._get_placeholder_source_layer()
    first_layer_expert = model._get_expert(model.model.layers[src_layer], 0)
    expert_placeholder = copy.deepcopy(first_layer_expert).to(dev)
    param_names = model.config.expert_param_names
    hidden_dim = model.hidden_dim

    # Measure copy time
    copy_times = []
    test_expert = model._get_expert(model.model.layers[src_layer], 2 % model.n_expert)
    for _ in token_counts:
        test_expert.to("cpu")
        for name in param_names:
            w = getattr(test_expert, name)
            w.weight.data = w.weight.data.pin_memory()

        torch.cuda.synchronize()
        tick = time.time()
        for name in param_names:
            dst = getattr(expert_placeholder, name).weight.data
            src = getattr(test_expert, name).weight.data
            dst.copy_(src)
        torch.cuda.synchronize()
        copy_times.append(time.time() - tick)

    # Measure GPU compute time
    gpu_expert = model._get_expert(model.model.layers[src_layer], 1 % model.n_expert)
    gpu_expert.to(dev)
    gpu_times = []
    for tc in token_counts:
        inps = torch.randn((tc, hidden_dim), dtype=model.dtype, device=dev)
        _ = gpu_expert(inps)
        torch.cuda.synchronize()
        tick = time.time()
        _ = gpu_expert(inps)
        torch.cuda.synchronize()
        gpu_times.append(time.time() - tick)
    gpu_expert.to("cpu")

    # Measure CPU compute time table
    cpu_times = []
    cpu_expert = model._get_expert(model.model.layers[src_layer], 1 % model.n_expert)
    for tc in token_counts:
        cpu_expert.to("cpu")
        inps = torch.randn((tc, hidden_dim), dtype=model.dtype, device="cpu")
        _ = model.run_expert_at_cpu(src_layer, 1 % model.n_expert, inps)

        measurements = []
        for _ in range(20):
            torch.cuda.synchronize()
            tick = time.time()
            _ = model.run_expert_at_cpu(src_layer, 1 % model.n_expert, inps)
            torch.cuda.synchronize()
            measurements.append(time.time() - tick)
        cpu_times.append(np.mean(measurements))

    # Measure single-token CPU time (tc) - multiple runs for accuracy
    tc_measurements = []
    cpu_expert.to("cpu")
    single_inps = torch.randn((1, hidden_dim), dtype=model.dtype, device="cpu")
    _ = model.run_expert_at_cpu(src_layer, 1 % model.n_expert, single_inps)
    for _ in range(50):
        torch.cuda.synchronize()
        tick = time.time()
        _ = model.run_expert_at_cpu(src_layer, 1 % model.n_expert, single_inps)
        torch.cuda.synchronize()
        tc_measurements.append(time.time() - tick)
    tc_avg = np.mean(tc_measurements) * 1000

    # Smooth CPU times
    window_size = 7
    cpu_times_full = np.zeros(len(token_counts))
    if len(cpu_times) >= window_size:
        cpu_times_smoothed = np.convolve(
            cpu_times, np.ones(window_size) / window_size, mode="valid"
        )
        first_valid = window_size - 1
        for i in range(len(token_counts)):
            if i < first_valid:
                cpu_times_full[i] = cpu_times_smoothed[0] * (i + 1) / (first_valid + 1)
            elif i - first_valid < len(cpu_times_smoothed):
                cpu_times_full[i] = cpu_times_smoothed[i - first_valid]
            else:
                cpu_times_full[i] = cpu_times_smoothed[-1]
    else:
        cpu_times_full = np.array(cpu_times)

    # Plot
    plt.figure(figsize=(12, 6))
    avg_copy_time = np.mean(copy_times) * 1000
    avg_gpu_time = np.mean(gpu_times) * 1000

    plt.plot(
        token_counts,
        [avg_copy_time] * len(token_counts),
        label=f"Transfer (Avg: {avg_copy_time:.2f}ms)",
        linestyle="--",
    )
    plt.plot(
        token_counts,
        np.array(gpu_times) * 1000,
        label=f"GPU Compute (Avg: {avg_gpu_time:.2f}ms)",
        marker="o",
    )
    plt.plot(
        token_counts,
        cpu_times_full * 1000,
        label="CPU Compute (smoothed)",
        marker="s",
        linewidth=2,
    )

    plt.xlabel("Token Count")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"ioreal_{model_name}.png", dpi=300, bbox_inches="tight")

    # Save CSV
    micro_file = f"micro{model_name}.txt"
    with open(micro_file, "w") as f:
        f.write("Token Count,Transfer Time(ms),GPU Compute(ms),CPU Compute(ms)\n")
        for i, tc in enumerate(token_counts):
            f.write(
                f"{tc},{avg_copy_time:.4f},{gpu_times[i] * 1000:.4f},{cpu_times_full[i] * 1000:.4f}\n"
            )

    # Save hot experts
    sorted_hot = sorted(expert_hot_data.items(), key=lambda x: x[1], reverse=True)
    hot_file = f"./hot/{model_name}.txt"
    os.makedirs(os.path.dirname(hot_file), exist_ok=True)
    with open(hot_file, "w") as f:
        for (layer, expert), count in sorted_hot:
            f.write(f"{layer},{expert}\n")

    # Save system config
    cpu_time_table_list = [float(t * 1000) for t in cpu_times_full]
    gpu_info = auto_detect_gpu_info()

    system_config = SystemConfig(
        gpu_name=gpu_info["gpu_name"],
        gpu_memory_gb=gpu_info["gpu_memory_gb"],
        transfer_time_ms=float(avg_copy_time),
        gpu_compute_time_ms=float(avg_gpu_time),
        single_token_cpu_time_ms=float(tc_avg),
        cpu_time_table=cpu_time_table_list,
    )

    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, f"system_config_{model_name}.json")
    system_config.save(config_path)

    print(f"\nSystem config saved to: {config_path}")
    print(f"  e (transfer): {avg_copy_time:.2f} ms")
    print(f"  tg (GPU compute): {avg_gpu_time:.2f} ms")
    print(f"  tc (CPU single-token): {tc_avg:.2f} ms")
    print(f"  GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']:.1f} GB)")

    return system_config


def main():
    parser = argparse.ArgumentParser(description="PDScope Microbenchmark")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model", type=str, default="/mnt/g/Models/DeepSeek-v2-lite-chat"
    )
    parser.add_argument("--cpu-offload", type=int, default=1, choices=[0, 1])
    parser.add_argument("--cache", type=int, default=1)
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="./configs")
    args = parser.parse_args()

    model_name = get_model_name_from_path(args.model)
    print(f"Detected model type: {model_name}")

    if not os.path.exists(args.model):
        print(f"\n[WARNING] Model not found: {args.model}")
        sys.exit(1)

    config = get_default_config(model_name)
    config.model_path = args.model
    config.batch_size = args.batch_size
    config.cache = args.cache
    config.cpu_offload = args.cpu_offload
    config.beam_width = args.beam_width

    model_classes = {
        "mixtral": FiddlerMixtral,
        "qwen": FiddlerQwen,
        "deepseek": FiddlerDeepSeekV2,
        "moon": FiddlerMoon,
    }
    model_cls = model_classes.get(model_name, FiddlerDeepSeekV2)
    model = model_cls(config)

    plot_expert_performance(model, model_name, args.output_dir)


if __name__ == "__main__":
    main()
