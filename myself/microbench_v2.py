"""
Microbenchmarking v2 — 独立复现四个关键参数的测量

测量:
1. e  — 专家权重 CPU→GPU 传输时间 (ms)
2. tg — 专家在 GPU 上的前向计算时间 (ms)
3. cpu_time_table — 不同 token 数(1-64)下 CPU 前向计算时间 (ms)
4. attention_time_table — 不同 token 数(1-64)下 Attention GPU 前向计算时间 (ms)

用法:
    python myself/microbench_v2.py --model /mnt/g/Models/DeepSeek-v2-lite-chat
    python myself/microbench_v2.py --model /mnt/g/Models/Qwen3-30B-A3B
"""

import argparse
import copy
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers


TOKEN_COUNTS = list(range(1, 65))
WINDOW_SIZE = 7
CPU_MEASURE_REPEATS = 20

WEIGHT_NAMES_MAP = {
    "mixtral": ["w1", "w2", "w3"],
    "deepseek": ["gate_proj", "up_proj", "down_proj"],
    "qwen": ["gate_proj", "up_proj", "down_proj"],
    "moon": ["gate_proj", "up_proj", "down_proj"],
}


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


def auto_detect_gpu_info():
    if torch.cuda.is_available():
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory
            / (1024**3),
        }
    return {"gpu_name": "Unknown", "gpu_memory_gb": 0.0}


def discover_expert_path(model, model_name):
    first_layer = model.model.layers[1]
    if model_name == "mixtral":
        experts = first_layer.block_sparse_moe.experts
    else:
        experts = first_layer.mlp.experts
    hidden_size = model.config.hidden_size
    return experts, hidden_size


def load_model(model_path):
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        dtype_str = cfg.get("torch_dtype", "bfloat16")
        dtype = getattr(torch, dtype_str)
    else:
        dtype = torch.bfloat16

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="cpu", trust_remote_code=True
    )
    hf_model.eval()
    print(f"模型加载完成, dtype={dtype}")
    return hf_model, dtype


def measure_transfer_time(experts, placeholder, weight_names, n_measurements=64):
    src_expert = experts[2]
    src_expert.to("cpu")

    for name in weight_names:
        w = getattr(src_expert, name)
        w.weight.data = w.weight.data.pin_memory()

    copy_times = []
    for _ in range(n_measurements):
        torch.cuda.synchronize()
        tick = time.time()

        for name in weight_names:
            dst = getattr(placeholder, name).weight.data
            src = getattr(src_expert, name).weight.data
            dst.copy_(src)

        torch.cuda.synchronize()
        copy_times.append(time.time() - tick)

    avg_ms = np.mean(copy_times) * 1000
    print(f"  e (传输时间): {avg_ms:.2f} ms ({n_measurements} 次测量)")
    return avg_ms


def measure_gpu_compute(expert, hidden_size, dtype, dev, token_counts):
    expert_gpu = copy.deepcopy(expert).to(dev)
    expert_gpu.eval()

    gpu_times_ms = []
    for token_count in token_counts:
        inps = torch.randn(token_count, hidden_size, dtype=dtype, device=dev)

        with torch.no_grad():
            _ = expert_gpu(inps)

        torch.cuda.synchronize()
        tick = time.time()
        with torch.no_grad():
            _ = expert_gpu(inps)
        torch.cuda.synchronize()
        gpu_times_ms.append((time.time() - tick) * 1000)

    avg_ms = np.mean(gpu_times_ms)
    print(f"  tg (GPU计算时间): {avg_ms:.2f} ms (avg over {len(token_counts)} tokens)")
    return gpu_times_ms


def measure_attention_compute(
    model, model_name, hidden_size, dtype, dev, token_counts, repeats=20
):
    use_position_ids = model_name in ("deepseek", "moon")
    attn = copy.deepcopy(model.model.layers[1].self_attn).to(dev)
    attn.eval()

    attn_times_ms = []
    for token_count in token_counts:
        hidden_states = torch.randn(
            1, token_count, hidden_size, dtype=dtype, device=dev
        )
        position_ids = torch.arange(
            token_count, dtype=torch.long, device=dev
        ).unsqueeze(0)
        attention_mask = torch.tril(
            torch.ones(1, 1, token_count, token_count, dtype=torch.bool, device=dev)
        )

        fwd_kwargs = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if use_position_ids:
            fwd_kwargs["position_ids"] = position_ids
        else:
            fwd_kwargs["position_embeddings"] = model.rotary_emb(
                hidden_states, position_ids
            )

        with torch.no_grad():
            _ = attn(**fwd_kwargs)

        measurements = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            tick = time.time()
            with torch.no_grad():
                _ = attn(**fwd_kwargs)
            torch.cuda.synchronize()
            measurements.append((time.time() - tick) * 1000)

        attn_times_ms.append(np.mean(measurements))

    avg_ms = np.mean(attn_times_ms)
    print(
        f"  attention (GPU计算时间): avg {avg_ms:.2f} ms over {len(token_counts)} token counts"
    )
    return attn_times_ms


def measure_cpu_compute(
    expert, hidden_size, dtype, token_counts, repeats=CPU_MEASURE_REPEATS
):
    expert_cpu = copy.deepcopy(expert).to("cpu")
    expert_cpu.eval()

    cpu_times_ms = []
    for token_count in token_counts:
        inps = torch.randn(token_count, hidden_size, dtype=dtype, device="cpu")

        with torch.no_grad():
            _ = expert_cpu(inps)

        measurements = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            tick = time.time()
            with torch.no_grad():
                _ = expert_cpu(inps)
            torch.cuda.synchronize()
            measurements.append((time.time() - tick) * 1000)

        cpu_times_ms.append(np.mean(measurements))

    print(
        f"  cpu_time_table: {len(cpu_times_ms)} entries, "
        f"range [{min(cpu_times_ms):.2f}, {max(cpu_times_ms):.2f}] ms"
    )
    return cpu_times_ms


def smooth_cpu_times(cpu_times, window_size=WINDOW_SIZE):
    cpu_times_arr = np.array(cpu_times)

    smoothed = np.convolve(
        cpu_times_arr, np.ones(window_size) / window_size, mode="valid"
    )

    first_valid_idx = window_size - 1
    cpu_times_full = np.zeros(len(cpu_times))
    for i in range(len(cpu_times)):
        if i < first_valid_idx:
            cpu_times_full[i] = smoothed[0] * (i + 1) / (first_valid_idx + 1)
        elif i - first_valid_idx < len(smoothed):
            cpu_times_full[i] = smoothed[i - first_valid_idx]
        else:
            cpu_times_full[i] = smoothed[-1]

    return cpu_times_full.tolist()


def write_csv(
    path, token_counts, avg_transfer_ms, gpu_times_ms, cpu_times_ms, attn_times_ms
):
    with open(path, "w") as f:
        f.write(
            "Token Count,Expert Transfer Time(ms),GPU Computation Time(ms),CPU Computation Time(ms),Attention Time(ms)\n"
        )
        for i, token_count in enumerate(token_counts):
            f.write(
                f"{token_count},{avg_transfer_ms:.4f},{gpu_times_ms[i]:.4f},{cpu_times_ms[i]:.4f},{attn_times_ms[i]:.4f}\n"
            )
    print(f"CSV 已保存到: {path}")


def write_system_config(
    path, gpu_info, avg_transfer_ms, avg_gpu_ms, cpu_time_table_ms, attn_times_ms
):
    config = {
        "gpu_name": gpu_info["gpu_name"],
        "gpu_memory_gb": gpu_info["gpu_memory_gb"],
        "transfer_time_ms": float(avg_transfer_ms),
        "gpu_compute_time_ms": float(avg_gpu_ms),
        "cpu_time_table": [float(t) for t in cpu_time_table_ms],
        "attention_time_table": [float(t) for t in attn_times_ms],
    }
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"系统配置已保存到: {path}")
    print(f"  GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']:.1f} GB)")


def plot_results(
    path,
    token_counts,
    avg_transfer_ms,
    gpu_times_ms,
    cpu_times_smoothed_ms,
    attn_times_ms,
):
    plt.figure(figsize=(12, 6))

    plt.plot(
        token_counts,
        [avg_transfer_ms] * len(token_counts),
        label=f"Expert Transfer (Avg: {avg_transfer_ms:.2f}ms)",
        linestyle="--",
    )

    avg_gpu_ms = np.mean(gpu_times_ms)
    plt.plot(
        token_counts,
        gpu_times_ms,
        label=f"Expert GPU Compute (Avg: {avg_gpu_ms:.2f}ms)",
        marker="o",
    )

    plt.plot(
        token_counts,
        cpu_times_smoothed_ms,
        label="Expert CPU Compute (smoothed)",
        marker="s",
        linestyle="-",
        linewidth=2,
    )

    avg_attn_ms = np.mean(attn_times_ms)
    plt.plot(
        token_counts,
        attn_times_ms,
        label=f"Self-Attention GPU (Avg: {avg_attn_ms:.2f}ms)",
        marker="^",
        linestyle="-",
    )

    plt.xlabel("Token Count")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"图表已保存到: {path}")


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    model_name = get_model_name_from_path(args.model)
    print(f"检测到模型类型: {model_name}")

    if not os.path.exists(args.model):
        print(f"[WARNING] 模型路径不存在: {args.model}")
        sys.exit(1)

    gpu_info = auto_detect_gpu_info()
    print(f"GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']:.1f} GB)")

    hf_model, dtype = load_model(args.model)
    print(f"模型加载完成, dtype={dtype}")

    experts, hidden_size = discover_expert_path(hf_model, model_name)
    weight_names = WEIGHT_NAMES_MAP[model_name]
    print(f"发现 {len(experts)} 个专家, hidden_size={hidden_size}")
    print(f"权重名称: {weight_names}")

    print("\n[1/4] 测量 CPU 前向计算时间...")
    cpu_time_table_ms = measure_cpu_compute(
        experts[0], hidden_size, dtype, TOKEN_COUNTS
    )

    print("\n[2/4] 测量 GPU 前向计算时间...")
    dev = torch.device("cuda:0")
    gpu_times_ms = measure_gpu_compute(
        experts[0], hidden_size, dtype, dev, TOKEN_COUNTS
    )

    print("\n[3/4] 测量 CPU→GPU 传输时间...")
    expert_placeholder = copy.deepcopy(experts[0]).to(dev)
    avg_transfer_ms = measure_transfer_time(experts, expert_placeholder, weight_names)

    print("\n[4/4] 测量 Attention GPU 计算时间...")
    attn_times_ms = measure_attention_compute(
        hf_model, model_name, hidden_size, dtype, dev, TOKEN_COUNTS
    )

    cpu_times_smoothed_ms = cpu_time_table_ms

    avg_gpu_ms = float(np.mean(gpu_times_ms))
    avg_attn_ms = float(np.mean(attn_times_ms))

    write_csv(
        os.path.join(args.output_dir, f"micro{model_name}_v2.txt"),
        TOKEN_COUNTS,
        avg_transfer_ms,
        gpu_times_ms,
        cpu_times_smoothed_ms,
        attn_times_ms,
    )
    write_system_config(
        os.path.join(args.output_dir, f"system_config_{model_name}_v2.json"),
        gpu_info,
        avg_transfer_ms,
        avg_gpu_ms,
        cpu_times_smoothed_ms,
        attn_times_ms,
    )
    plot_results(
        os.path.join(args.output_dir, f"ioreal_{model_name}_v2.png"),
        TOKEN_COUNTS,
        avg_transfer_ms,
        gpu_times_ms,
        cpu_times_smoothed_ms,
        attn_times_ms,
    )

    print("\n===== 完成 =====")
    print(f"  e  (传输时间):     {avg_transfer_ms:.2f} ms")
    print(f"  tg (GPU计算时间):  {avg_gpu_ms:.2f} ms")
    print(f"  ta (Attention时间): {avg_attn_ms:.2f} ms")
    print(f"  cpu_time_table:    64 entries")


if __name__ == "__main__":
    main()
