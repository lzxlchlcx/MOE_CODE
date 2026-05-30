"""Microbenchmarking for DeepSeek-V2-Lite CPU offloading"""

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2RotaryEmbedding
from transformers.masking_utils import create_causal_mask

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.deepseek import mDeepSeek
from model.placeholder_manager import ExpertPlaceholderManager

WEIGHT_NAMES = ['gate_proj', 'up_proj', 'down_proj']


def bench_expert_cpu(model, token_counts=None, n_repeat=20):
    """测量单个 expert 在 CPU 上不同 token 数量时的计算时间

    Args:
        model: mDeepSeek 实例
        token_counts: 要测量的 token 数量列表
        n_repeat: 每个 token 数量重复测量的次数

    Returns:
        list[(token_count, avg_time_ms)]: 每个 token 数量对应的平均计算时间
    """
    if token_counts is None:
        token_counts = list(range(1, 250))
    hidden_dim = model.config.hidden_size
    expert = model.model.layers[1].mlp.experts[0]
    expert.to("cpu")

    results = []
    for tc in token_counts:
        inps = torch.randn(tc, hidden_dim, dtype=model.dtype, device="cpu")
        _ = expert(inps)

        measurements = []
        for _ in range(n_repeat):
            torch.cuda.synchronize()
            tick = time.time()
            _ = expert(inps)
            torch.cuda.synchronize()
            measurements.append(time.time() - tick)
        avg_ms = np.mean(measurements) * 1000
        results.append((tc, avg_ms))

    expert.to(model.dev)
    return results


def bench_expert_gpu(model, token_counts=None, n_repeat=10):
    """测量单个 expert 在 GPU 上不同 token 数量时的计算时间

    Args:
        model: mDeepSeek 实例
        token_counts: 要测量的 token 数量列表
        n_repeat: 每个 token 数量重复测量的次数

    Returns:
        list[(token_count, avg_time_ms)]: 每个 token 数量对应的平均计算时间
    """
    if token_counts is None:
        token_counts = list(range(1, 250))
    hidden_dim = model.config.hidden_size
    expert = model.model.layers[1].mlp.experts[1]
    expert.to(model.dev)

    results = []
    for tc in token_counts:
        inps = torch.randn(tc, hidden_dim, dtype=model.dtype, device=model.dev)
        _ = expert(inps)

        measurements = []
        for _ in range(n_repeat):
            torch.cuda.synchronize()
            tick = time.time()
            _ = expert(inps)
            torch.cuda.synchronize()
            measurements.append(time.time() - tick)
        avg_ms = np.mean(measurements) * 1000
        results.append((tc, avg_ms))

    expert.to("cpu")
    return results


def summarize_measurements(results):
    values_ms = np.array(results) * 1000
    return {
        "trials_ms": [round(t, 4) for t in values_ms],
        "avg_ms": round(float(np.mean(values_ms)), 4),
        "median_ms": round(float(np.median(values_ms)), 4),
        "p95_ms": round(float(np.percentile(values_ms, 95)), 4),
    }


def bench_expert_weight_copy(model, n_repeat=30, n_warmup=5):
    """测量 expert 权重从 CPU 搬运到 GPU 的时间

    使用 pin_memory + copy_ 方式，与实际推理一致。

    Args:
        model: mDeepSeek 实例
        n_repeat: 重复测量次数

    Returns:
        list[float]: 每次测量的搬运时间(秒)
    """
    expert_placeholder = copy.deepcopy(
        model.model.layers[1].mlp.experts[0]
    ).to(model.dev)

    src_expert = model.model.layers[1].mlp.experts[2]
    src_expert.to("cpu")
    for name in WEIGHT_NAMES:
        w = getattr(src_expert, name)
        w.weight.data = w.weight.data.pin_memory()

    results = []
    for i in range(n_warmup + n_repeat):
        torch.cuda.synchronize()
        tick = time.time()
        for name in WEIGHT_NAMES:
            dst = getattr(expert_placeholder, name).weight.data
            src = getattr(src_expert, name).weight.data
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        if i >= n_warmup:
            results.append(time.time() - tick)

    return results


def pin_expert_weights(expert):
    for param in expert.parameters():
        param.data = param.data.pin_memory()
    for buffer in expert.buffers():
        buffer.data = buffer.data.pin_memory()


def bench_expert_manager_weight_copy(model, n_repeat=30, n_warmup=5):
    """测量 placeholder manager 的真实 expert 权重加载路径。"""
    src_expert = model.model.layers[1].mlp.experts[2]
    src_expert.to("cpu")
    pin_expert_weights(src_expert)

    manager = ExpertPlaceholderManager(
        template_expert=model.model.layers[1].mlp.experts[0],
        device=model.dev,
        num_placeholders=1,
    )
    placeholder = manager.acquire_free_placeholder(1, 2)

    results = []
    for i in range(n_warmup + n_repeat):
        torch.cuda.synchronize()
        tick = time.time()
        manager.load_weights(placeholder, src_expert)
        torch.cuda.synchronize()
        if i >= n_warmup:
            results.append(time.time() - tick)

    return results


def bench_attention(model, token_counts=None, use_cache=False, n_repeat=5):
    """测量 attention 计算时间（有无 KV cache）

    Args:
        model: mDeepSeek 实例
        token_counts: 要测量的 token 数量列表
        use_cache: 是否使用 KV cache（True=decode 模式 1 token, False=prefill 全量）
        n_repeat: 重复测量次数

    Returns:
        list[(token_count, avg_time_ms)]: 每个 token 数量对应的平均计算时间
    """
    if token_counts is None:
        token_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    hidden_dim = model.config.hidden_size

    layer = model.model.layers[1]
    rotary_emb = DeepseekV2RotaryEmbedding(config=model.config, device=model.dev)

    results = []
    for tc in token_counts:
        if use_cache:
            seq_len = 1
            past_len = tc - 1
            cache = transformers.cache_utils.DynamicCache()
            fake_k = torch.randn(
                1, model.config.num_attention_heads,
                past_len,
                model.config.qk_nope_head_dim + model.config.qk_rope_head_dim,
                dtype=model.dtype, device=model.dev
            )
            fake_v = torch.randn(
                1, model.config.num_attention_heads,
                past_len,
                model.config.v_head_dim,
                dtype=model.dtype, device=model.dev
            )
            for li in range(model.n_layer):
                cache.update(fake_k, fake_v, layer_idx=li)
            cache_position = torch.arange(past_len, past_len + seq_len, device=model.dev)
        else:
            seq_len = tc
            cache = transformers.cache_utils.DynamicCache()
            cache_position = torch.arange(seq_len, device=model.dev)

        inps = torch.randn(1, seq_len, hidden_dim, dtype=model.dtype, device=model.dev)
        position_ids = torch.arange(seq_len, device=model.dev).unsqueeze(0)
        attention_mask = torch.ones(1, seq_len + (tc - 1 if use_cache else 0),
                                    dtype=torch.long, device=model.dev)
        position_embeddings = rotary_emb(inps, position_ids)

        causal_mask = create_causal_mask(
            config=model.config,
            input_embeds=inps,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=cache,
            position_ids=position_ids,
        )

        _ = layer.self_attn(
            hidden_states=inps,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        measurements = []
        for _ in range(n_repeat):
            torch.cuda.synchronize()
            tick = time.time()
            _ = layer.self_attn(
                hidden_states=inps,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            torch.cuda.synchronize()
            measurements.append(time.time() - tick)
        avg_ms = np.mean(measurements) * 1000
        results.append((tc, avg_ms))

    return results


def save_benchmark_results(model_path, cpu_results, gpu_results, copy_results,
                           manager_copy_results, attn_prefill, attn_decode, output_dir=None):
    """将所有 benchmark 结果保存为 JSON 文件

    Args:
        model_path: 模型路径，用于提取模型名作为文件名
        cpu_results: bench_expert_cpu 的返回值
        gpu_results: bench_expert_gpu 的返回值
        copy_results: bench_expert_weight_copy 的返回值
        attn_prefill: bench_attention(use_cache=False) 的返回值
        attn_decode: bench_attention(use_cache=True) 的返回值
        output_dir: 输出目录，默认为脚本所在目录
    """
    model_name = os.path.basename(model_path.rstrip('/')).lower()
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    data = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "expert_cpu": [
            {"token_count": tc, "avg_time_ms": round(t, 4)}
            for tc, t in cpu_results
        ],
        "expert_gpu": [
            {"token_count": tc, "avg_time_ms": round(t, 4)}
            for tc, t in gpu_results
        ],
        "expert_weight_copy": {
            **summarize_measurements(copy_results),
        },
        "expert_manager_weight_copy": {
            **summarize_measurements(manager_copy_results),
        },
        "attention_prefill": [
            {"token_count": tc, "avg_time_ms": round(t, 4)}
            for tc, t in attn_prefill
        ],
        "attention_decode": [
            {"past_length": tc, "avg_time_ms": round(t, 4)}
            for tc, t in attn_decode
        ],
    }

    filename = f"micro_{model_name}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nBenchmark results saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument("--model", type=str, required=True, help="Path to DeepSeek model.")
    parser.add_argument("--cpu-offload", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--copy-repeat", type=int, default=30)
    parser.add_argument("--copy-warmup", type=int, default=5)

    args = parser.parse_args()
    model = mDeepSeek(args)

    print("=" * 60)
    print("1. Expert CPU computation time (token_count, time_ms)")
    print("=" * 60)
    cpu_results = bench_expert_cpu(model, token_counts=[1, 2, 4, 8, 16, 32, 64, 128])
    for tc, t in cpu_results:
        print(f"  tokens={tc:4d}  cpu_time={t:.4f}ms")

    print()
    print("=" * 60)
    print("2. Expert GPU computation time (token_count, time_ms)")
    print("=" * 60)
    gpu_results = bench_expert_gpu(model, token_counts=[1, 2, 4, 8, 16, 32, 64, 128])
    for tc, t in gpu_results:
        print(f"  tokens={tc:4d}  gpu_time={t:.4f}ms")

    print()
    print("=" * 60)
    print("3. Expert weight copy CPU->GPU time")
    print("=" * 60)
    copy_results = bench_expert_weight_copy(model, n_repeat=args.copy_repeat, n_warmup=args.copy_warmup)
    copy_summary = summarize_measurements(copy_results)
    for i, t in enumerate(copy_results):
        print(f"  trial={i+1}  copy_time={t*1000:.4f}ms")
    print(
        f"  avg={copy_summary['avg_ms']:.4f}ms "
        f"median={copy_summary['median_ms']:.4f}ms "
        f"p95={copy_summary['p95_ms']:.4f}ms"
    )

    print()
    print("=" * 60)
    print("4. Expert manager weight copy CPU->GPU time")
    print("=" * 60)
    manager_copy_results = bench_expert_manager_weight_copy(model, n_repeat=args.copy_repeat, n_warmup=args.copy_warmup)
    manager_copy_summary = summarize_measurements(manager_copy_results)
    for i, t in enumerate(manager_copy_results):
        print(f"  trial={i+1}  manager_copy_time={t*1000:.4f}ms")
    print(
        f"  avg={manager_copy_summary['avg_ms']:.4f}ms "
        f"median={manager_copy_summary['median_ms']:.4f}ms "
        f"p95={manager_copy_summary['p95_ms']:.4f}ms"
    )

    print()
    print("=" * 60)
    print("5. Attention computation time (token_count, time_ms)")
    print("-" * 60)
    print("  Without KV cache (prefill):")
    attn_nocache = bench_attention(model, token_counts=[1, 8, 32, 64, 128, 256], use_cache=False)
    for tc, t in attn_nocache:
        print(f"    tokens={tc:4d}  attn_time={t:.4f}ms")

    print("  With KV cache (decode, 1 token):")
    attn_cache = bench_attention(model, token_counts=[1, 8, 32, 64, 128, 256], use_cache=True)
    for tc, t in attn_cache:
        print(f"    past_len={tc:4d}  attn_time={t:.4f}ms")

    save_benchmark_results(
        model_path=args.model,
        cpu_results=cpu_results,
        gpu_results=gpu_results,
        copy_results=copy_results,
        manager_copy_results=manager_copy_results,
        attn_prefill=attn_nocache,
        attn_decode=attn_cache,
    )
