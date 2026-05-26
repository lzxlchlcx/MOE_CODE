"""
MoE 命中率 + 并行度 Benchmark 脚本

用法:
    python benchmark.py --model /path/to/model [--batch_sizes 1 4] [--prefetch both] [--n_sample 3]

输出:
    1. 控制台: 各配置对比表
    2. JSON:   完整数据 (benchmark_result/benchmark_*.json)
    3. TXT:    人类可读报告 (benchmark_result/benchmark_*.txt)
"""

import argparse
import json
import os
import sys
import time
import random
import warnings
from datetime import datetime
from typing import Dict, List, Any

import numpy as np

sys.path.append("./")
from mixtral import FiddlerMixtral
from deepseek import FiddlerDeepSeekV2
from moon import FiddlerMoon
from qwen import FiddlerQwen

from config import (
    ModelConfig,
    SystemConfig,
    get_default_config,
    merge_system_to_model,
)
from latency import load_dataset, detect_model_type, get_model_class, check_model_exists, find_system_config


def run_single_benchmark(model, config, texts, n_sample, input_token, output_token):
    """运行单次 benchmark，收集命中率和并行度指标"""
    idx_text = 0

    all_hit_rates = {
        "hot_hit_rate": [],
        "prefetch_hit_rate": [],
        "gpu_available_rate": [],
    }
    all_parallel_degrees = []
    all_expert_classes = {
        "gpu": 0, "cpu": 0, "ondemand": 0,
        "prefetch": 0, "loading": 0, "placeholder": 0,
    }
    all_throughputs = []
    layer_parallel_stats = {}

    prefill_total = 0.0
    decode_total = 0.0

    for sample_idx in range(n_sample):
        batch_texts = []
        while len(batch_texts) < config.batch_size:
            if idx_text >= len(texts):
                idx_text = 0
            text = texts[idx_text]
            idx_text += 1
            if len(text.split()) >= input_token:
                batch_texts.append(text)
        batch_texts = batch_texts[:config.batch_size]

        prefill_time, decode_time, hit_rate, stats = model.generate(
            batch_texts, output_token=output_token, input_token=input_token
        )

        sample_time = prefill_time + decode_time
        throughput = (output_token * config.batch_size) / sample_time if sample_time > 0 else 0

        for key in all_hit_rates:
            all_hit_rates[key].append(hit_rate.get(key, 0))

        all_throughputs.append(throughput)
        prefill_total += prefill_time
        decode_total += decode_time

        ec = model.logger.metrics.expert_classification
        for cls_key in all_expert_classes:
            if cls_key in ec:
                all_expert_classes[cls_key] += ec[cls_key]

        gpu_t = model.logger.metrics.layer_gpu_times
        cpu_t = model.logger.metrics.layer_cpu_times
        par_t = model.logger.metrics.layer_parallel_times

        for lid in par_t:
            if lid not in layer_parallel_stats:
                layer_parallel_stats[lid] = {
                    "gpu_times": [], "cpu_times": [], "parallel_times": [],
                    "parallel_degrees": [],
                }
            for i in range(len(par_t[lid])):
                gt = gpu_t[lid][i] if i < len(gpu_t.get(lid, [])) else 0
                ct = cpu_t[lid][i] if i < len(cpu_t.get(lid, [])) else 0
                pt = par_t[lid][i]
                pd = (gt + ct) / pt if pt > 0 else 1.0
                layer_parallel_stats[lid]["gpu_times"].append(gt)
                layer_parallel_stats[lid]["cpu_times"].append(ct)
                layer_parallel_stats[lid]["parallel_times"].append(pt)
                layer_parallel_stats[lid]["parallel_degrees"].append(pd)
                all_parallel_degrees.append(pd)

        model.logger.metrics.expert_classification = {
            k: 0 for k in model.logger.metrics.expert_classification
        }
        model.logger.metrics.layer_gpu_times.clear()
        model.logger.metrics.layer_cpu_times.clear()
        model.logger.metrics.layer_parallel_times.clear()

    total_experts = sum(all_expert_classes.values()) or 1

    result = {
        "config": {
            "batch_size": config.batch_size,
            "prefetch_enabled": config.prefetch_enabled,
            "e": config.e,
            "tg": config.tg,
            "tc": config.get_tc(),
        },
        "hit_rates": {
            "hot_hit_rate": np.mean(all_hit_rates["hot_hit_rate"]),
            "prefetch_hit_rate": np.mean(all_hit_rates["prefetch_hit_rate"]),
            "gpu_available_rate": np.mean(all_hit_rates["gpu_available_rate"]),
            "hot_hit_rate_std": float(np.std(all_hit_rates["hot_hit_rate"])),
            "prefetch_hit_rate_std": float(np.std(all_hit_rates["prefetch_hit_rate"])),
            "gpu_available_rate_std": float(np.std(all_hit_rates["gpu_available_rate"])),
        },
        "parallelism": {
            "avg_parallel_degree": float(np.mean(all_parallel_degrees)) if all_parallel_degrees else 1.0,
            "max_parallel_degree": float(np.max(all_parallel_degrees)) if all_parallel_degrees else 1.0,
            "min_parallel_degree": float(np.min(all_parallel_degrees)) if all_parallel_degrees else 1.0,
            "std_parallel_degree": float(np.std(all_parallel_degrees)) if all_parallel_degrees else 0.0,
        },
        "throughput": {
            "avg": float(np.mean(all_throughputs)),
            "std": float(np.std(all_throughputs)),
            "prefill_avg": prefill_total / n_sample,
            "decode_avg": decode_total / n_sample,
        },
        "expert_classification": {
            k: v for k, v in all_expert_classes.items()
        },
        "expert_classification_pct": {
            k: v / total_experts * 100 for k, v in all_expert_classes.items()
        },
        "layer_parallel_stats": {},
    }

    for lid in sorted(layer_parallel_stats.keys()):
        s = layer_parallel_stats[lid]
        pds = s["parallel_degrees"]
        result["layer_parallel_stats"][str(lid)] = {
            "avg_parallel_degree": float(np.mean(pds)),
            "avg_gpu_time_ms": float(np.mean(s["gpu_times"]) * 1000),
            "avg_cpu_time_ms": float(np.mean(s["cpu_times"]) * 1000),
            "avg_parallel_time_ms": float(np.mean(s["parallel_times"]) * 1000),
            "n_samples": len(pds),
        }

    return result


def print_comparison_table(results: List[Dict]):
    """打印配置对比表"""
    print("\n" + "=" * 100)
    print("  Benchmark 对比表: 命中率 + 并行度")
    print("=" * 100)

    header = f"{'Config':<28} | {'Hot命中':>8} | {'预取命中':>8} | {'GPU可用':>8} | {'并行度':>8} | {'吞吐':>10} | {'GPU%':>5} | {'CPU%':>5} | {'OD%':>5} | {'PF%':>5}"
    print(header)
    print("-" * len(header))

    for r in results:
        cfg = r["config"]
        label = f"bs={cfg['batch_size']}, pf={'ON' if cfg['prefetch_enabled'] else 'OFF'}"

        hr = r["hit_rates"]
        pr = r["parallelism"]
        tp = r["throughput"]
        ec = r["expert_classification_pct"]

        row = (
            f"{label:<28} | "
            f"{hr['hot_hit_rate']:>7.2%} | "
            f"{hr['prefetch_hit_rate']:>7.2%} | "
            f"{hr['gpu_available_rate']:>7.2%} | "
            f"{pr['avg_parallel_degree']:>7.2f}x | "
            f"{tp['avg']:>8.2f} t/s | "
            f"{ec.get('gpu', 0):>4.1f}% | "
            f"{ec.get('cpu', 0):>4.1f}% | "
            f"{ec.get('ondemand', 0):>4.1f}% | "
            f"{ec.get('prefetch', 0):>4.1f}%"
        )
        print(row)

    print("=" * 100)


def print_layer_breakdown(result: Dict):
    """打印逐层并行度分解"""
    lps = result.get("layer_parallel_stats", {})
    if not lps:
        return

    print(f"\n  逐层并行度分解 (bs={result['config']['batch_size']}, "
          f"pf={'ON' if result['config']['prefetch_enabled'] else 'OFF'})")
    print("-" * 90)
    print(f"  {'Layer':>6} | {'GPU(ms)':>8} | {'CPU(ms)':>8} | {'并行(ms)':>8} | {'并行度':>8} | 标记")
    print("-" * 90)

    avg_pd = result["parallelism"]["avg_parallel_degree"]

    for lid_str in sorted(lps.keys(), key=lambda x: int(x)):
        s = lps[lid_str]
        pd = s["avg_parallel_degree"]
        marker = ""
        if pd < 1.2:
            marker = "  <<< 低并行"
        elif pd > 1.7:
            marker = "  >>> 高并行"
        print(
            f"  {int(lid_str):>6} | "
            f"{s['avg_gpu_time_ms']:>7.2f} | "
            f"{s['avg_cpu_time_ms']:>7.2f} | "
            f"{s['avg_parallel_time_ms']:>7.2f} | "
            f"{pd:>7.2f}x{marker}"
        )

    print("-" * 90)
    print(f"  {'平均':>6} | {'':>8} | {'':>8} | {'':>8} | {avg_pd:>7.2f}x")
    print()


def generate_text_report(results: List[Dict], args) -> str:
    """生成文本报告"""
    lines = []
    lines.append("=" * 100)
    lines.append("MoE Benchmark 报告 — 命中率 + 并行度")
    lines.append(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"模型: {args.model}")
    lines.append(f"input_token={args.input_token}, output_token={args.output_token}, n_sample={args.n_sample}")
    lines.append("=" * 100)
    lines.append("")

    for r in results:
        cfg = r["config"]
        label = f"batch_size={cfg['batch_size']}, prefetch={'启用' if cfg['prefetch_enabled'] else '禁用'}"
        lines.append(f"--- {label} ---")

        hr = r["hit_rates"]
        lines.append(f"  热表命中率:   {hr['hot_hit_rate']:.2%} (std={hr['hot_hit_rate_std']:.4f})")
        lines.append(f"  预取命中率:   {hr['prefetch_hit_rate']:.2%} (std={hr['prefetch_hit_rate_std']:.4f})")
        lines.append(f"  GPU可用率:    {hr['gpu_available_rate']:.2%} (std={hr['gpu_available_rate_std']:.4f})")

        pr = r["parallelism"]
        lines.append(f"  平均并行度:   {pr['avg_parallel_degree']:.2f}x (std={pr['std_parallel_degree']:.4f}, "
                      f"min={pr['min_parallel_degree']:.2f}x, max={pr['max_parallel_degree']:.2f}x)")

        tp = r["throughput"]
        lines.append(f"  吞吐率:       {tp['avg']:.2f} token/s (std={tp['std']:.2f})")
        lines.append(f"  Prefill:      {tp['prefill_avg']:.2f}s, Decode: {tp['decode_avg']:.2f}s")

        ec = r["expert_classification"]
        total = sum(ec.values()) or 1
        lines.append(f"  专家分类:     GPU={ec['gpu']}({ec['gpu']/total*100:.1f}%) "
                      f"CPU={ec['cpu']}({ec['cpu']/total*100:.1f}%) "
                      f"OD={ec['ondemand']}({ec['ondemand']/total*100:.1f}%) "
                      f"PF={ec['prefetch']}({ec['prefetch']/total*100:.1f}%)")
        lines.append("")

        lps = r.get("layer_parallel_stats", {})
        if lps:
            lines.append(f"  逐层并行度:")
            for lid_str in sorted(lps.keys(), key=lambda x: int(x)):
                s = lps[lid_str]
                lines.append(f"    Layer {int(lid_str):>2}: GPU={s['avg_gpu_time_ms']:.2f}ms "
                              f"CPU={s['avg_cpu_time_ms']:.2f}ms "
                              f"PD={s['avg_parallel_degree']:.2f}x")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="MoE 命中率 + 并行度 Benchmark")
    parser.add_argument("--model", type=str, default="/mnt/g/Models/DeepSeek-v2-lite-chat")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4],
                        help="要测试的 batch_size 列表")
    parser.add_argument("--prefetch", type=str, default="both", choices=["enabled", "disabled", "both"],
                        help="预取策略: enabled/disabled/both")
    parser.add_argument("--n_sample", type=int, default=3, help="每个配置的采样次数")
    parser.add_argument("--input_token", type=int, default=128)
    parser.add_argument("--output_token", type=int, default=32)
    parser.add_argument("--cache", type=int, default=4)
    parser.add_argument("--auto_microbench", type=int, default=1, choices=[0, 1])
    parser.add_argument("--output_dir", type=str, default="./benchmark_result")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not check_model_exists(args.model):
        print(f"[ERROR] 模型路径不存在: {args.model}")
        sys.exit(1)

    model_name = detect_model_type(args.model)

    prefetch_modes = []
    if args.prefetch == "both":
        prefetch_modes = [False, True]
    elif args.prefetch == "enabled":
        prefetch_modes = [True]
    else:
        prefetch_modes = [False]

    dataset_path = (
        "/home/lzx/program/moe_code/opensource/"
        "sharegpt_v3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json"
    )
    texts = load_dataset(dataset_path, "share")
    random.seed(42)
    random.shuffle(texts)

    results = []
    configs_to_test = []
    for bs in args.batch_sizes:
        for pf in prefetch_modes:
            configs_to_test.append((bs, pf))

    model_instance = None

    for idx, (bs, pf) in enumerate(configs_to_test):
        config = get_default_config(model_name)
        config.model_path = args.model
        config.batch_size = bs
        config.cache = args.cache
        config.cpu_offload = 1
        config.prefetch_enabled = pf

        system_config_path = find_system_config(model_name)
        if system_config_path and os.path.exists(system_config_path):
            system_config = SystemConfig.load(system_config_path)
            config = merge_system_to_model(system_config, config)

        label = f"bs={bs}, prefetch={'ON' if pf else 'OFF'}"
        print(f"\n{'='*80}")
        print(f"  [{idx+1}/{len(configs_to_test)}] 测试配置: {label}")
        print(f"{'='*80}")

        model_cls = get_model_class(model_name)
        model_instance = model_cls(config)

        result = run_single_benchmark(
            model_instance, config, texts, args.n_sample,
            args.input_token, args.output_token,
        )
        results.append(result)

        del model_instance
        import gc
        gc.collect()

    print_comparison_table(results)

    for r in results:
        print_layer_breakdown(r)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    json_path = os.path.join(args.output_dir, f"benchmark_{model_name}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nJSON 结果: {json_path}")

    report_text = generate_text_report(results, args)
    txt_path = os.path.join(args.output_dir, f"benchmark_{model_name}_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"TXT 报告: {txt_path}")


if __name__ == "__main__":
    main()
