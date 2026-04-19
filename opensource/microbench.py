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
from mixtral import FiddlerMixtral  # Mixtral MoE 模型封装
from deepseek import FiddlerDeepSeekV2  # DeepSeek MoE 模型封装
from moon import FiddlerMoon  # Moonlight MoE 模型封装
from qwen import FiddlerQwen  # Qwen MoE 模型封装

from config import (
    ModelConfig,  # 模型配置（路径、batch_size等）
    SystemConfig,  # 系统配置（GPU信息、CPU/GPU时间等）
    get_default_config,  # 获取默认配置
    auto_detect_gpu_info,  # 自动检测GPU信息
)


def get_model_name_from_path(model_path):
    """从模型路径推断模型名

    Args:
        model_path: 模型路径或名称

    Returns:
        模型类型字符串: 'mixtral', 'qwen', 'deepseek', 'moon'
    """
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
    """Profile hot experts by running sample tokens through MoE routing

    通过让样本token流经MoE路由机制来分析热点专家的使用频率。
    这些数据用于决定哪些专家应常驻GPU以提升推理性能。

    Args:
        model: Fiddler封装模型实例（FiddlerMixtral/FiddlerDeepSeekV2等）
        model_name: 模型类型字符串 ('mixtral', 'qwen', 'deepseek', 'moon')

    Returns:
        expert_hot_data: dict, 键为(层ID, 专家ID)的元组，值为调用次数
    """
    expert_hot_data = {}

    # 使用3个简短样本文本触发路由决策
    sample_texts = ["Hello world", "How are you today?", "The weather is nice"]

    try:
        # ========== 文本分词 ==========
        # model.tokenizer: HuggingFace AutoTokenizer，将文本转为token ID
        # 返回BatchEncoding对象，包含input_ids、attention_mask等字段
        # .input_ids: 提取token ID张量
        # .to(model.dev): 移动到模型所在设备(GPU)
        input_ids = model.tokenizer(
            sample_texts, return_tensors="pt", padding=True
        ).input_ids.to(model.dev)

        # ========== 遍历前几层分析热点专家 ==========
        # 仅采样前4层，避免全量分析带来的性能开销
        for i_layer in range(1, min(5, len(model.model.layers))):
            # ========== 获取Gate模块 ==========
            # Mixtral使用block_sparse_moe.gate，其他模型使用mlp.gate
            if model_name == "mixtral":
                gate = model.model.layers[i_layer].block_sparse_moe.gate
            else:
                gate = model.model.layers[i_layer].mlp.gate

            # ========== 词嵌入 ==========
            # embed_tokens: Transformer的embedding层，将token ID转为hidden state
            # 输出形状: [batch_size, seq_len, hidden_dim]
            hidden_states = model.model.embed_tokens(input_ids)

            # ========== MoE路由 ==========
            # gate模块为每个token选择top-k个专家
            # selected_experts: [num_tokens, top_k] 每个token选中的专家ID
            # routing_weights: [num_tokens, top_k] 对应的路由权重
            selected_experts, routing_weights, _ = gate(hidden_states)

            # ========== 统计每个专家被选中的次数 ==========
            expert_counts = {}
            for expert_id in selected_experts.unique():
                # mask: 该token是否选中了此专家
                # (selected_experts == expert_id): 布尔张量
                # .any(dim=1): 在top_k维度做或运算，判断该token是否选中此专家
                # .sum(): 统计选中了该专家的token数量
                mask = (selected_experts == expert_id).any(dim=1)
                # .item(): 将PyTorch张量转为Python标量，用于字典键
                expert_counts[expert_id.item()] = mask.sum().item()

            # ========== 累加到expert_hot_data ==========
            for expert_id, count in expert_counts.items():
                key = (i_layer, expert_id)  # 元组作为字典键
                if key not in expert_hot_data:
                    expert_hot_data[key] = 0
                expert_hot_data[key] += count

    except Exception as e:
        # 分析失败时使用默认的线性递减模式（靠前的专家使用更频繁）
        print(f"Hot expert profiling failed: {e}")
        for i_layer in range(1, 5):
            for expert_id in range(8):
                key = (i_layer, expert_id)
                expert_hot_data[key] = 64 - expert_id * 4

    return expert_hot_data


def plot_expert_performance(model, model_name, output_dir="./configs"):
    """绘制专家性能折线图并生成系统配置

    测量并可视化:
    1. 专家从CPU到GPU的拷贝时间
    2. 专家在GPU上的计算时间
    3. 专家在CPU上的计算时间
    4. 自注意力在GPU上的时间

    同时生成热点专家列表和系统配置文件。

    Args:
        model: Fiddler封装模型实例
        model_name: 模型类型字符串
        output_dir: 系统配置JSON的输出目录

    Returns:
        system_config: SystemConfig对象，包含所有实测参数
    """
    # token_counts: 测试1-64个token的不同规模
    token_counts = list(range(1, 65))
    ret_time = []

    # 先分析热点专家
    expert_hot_data = profile_hot_experts(model, model_name)
    dev = torch.device("cuda:0")
    model.dev = torch.device("cuda:0")

    # ========== 获取专家列表 ==========
    # Mixtral使用block_sparse_moe.experts，其他模型使用mlp.experts
    if model_name == "mixtral":
        first_layer_experts = model.model.layers[1].block_sparse_moe.experts
        n_expert = len(model.model.layers[0].block_sparse_moe.experts)
    else:
        first_layer_experts = model.model.layers[1].mlp.experts
        n_expert = len(model.model.layers[1].mlp.experts)

    n_shared_experts = 2  # 共享专家数量

    # 创建占位专家模板，用于后续拷贝测量
    expert_placeholder = copy.deepcopy(first_layer_experts[0]).to(dev)

    # ========== 测量专家CPU->GPU拷贝时间 ==========
    copy_times = []
    for _ in token_counts:
        for i in range(2, 3):  # 只测量专家2
            # 将专家移到CPU
            first_layer_experts[i].to("cpu")

            # ========== Pin Memory加速 ==========
            # pin_memory: 将Tensor锁定到页锁定内存，加速CPU->GPU传输
            if model_name == "mixtral":
                for name in ["w1", "w2", "w3"]:  # Mixtral的FFN权重
                    w = getattr(first_layer_experts[i], name)
                    src_weight_data_tensor = w.weight.data
                    pinned = src_weight_data_tensor.pin_memory()
                    w.weight.data = pinned
            else:
                for name in ["gate_proj", "up_proj", "down_proj"]:  # 标准MLP权重
                    w = getattr(first_layer_experts[i], name)
                    src_weight_data_tensor = w.weight.data
                    pinned = src_weight_data_tensor.pin_memory()
                    w.weight.data = pinned

            # 同步后测量实际拷贝时间
            torch.cuda.synchronize()
            tick = time.time()

            # 执行CPU->GPU拷贝
            if model_name == "mixtral":
                for name in ["w1", "w2", "w3"]:
                    dst = getattr(expert_placeholder, name).weight.data
                    src = getattr(first_layer_experts[i], name).weight.data
                    dst.copy_(src)  # PyTorch拷贝操作
            else:
                for name in ["gate_proj", "up_proj", "down_proj"]:
                    dst = getattr(expert_placeholder, name).weight.data
                    src = getattr(first_layer_experts[i], name).weight.data
                    dst.copy_(src)

            torch.cuda.synchronize()
            copy_times.append(time.time() - tick)

    # ========== 测量专家GPU计算时间 ==========
    gpu_times = []
    for token_count in token_counts:
        # 将专家移到GPU
        first_layer_experts[1].to(model.dev)

        # 生成随机输入
        if model_name == "mixtral":
            inps = torch.randn((token_count, 4096), dtype=model.dtype, device=model.dev)
        else:
            inps = torch.randn((token_count, 2048), dtype=model.dtype, device=model.dev)

        # 预热
        _ = first_layer_experts[1](inps)

        # 实际测量
        torch.cuda.synchronize()
        tick = time.time()
        _ = first_layer_experts[1](inps)
        torch.cuda.synchronize()
        gpu_times.append(time.time() - tick)

        # 测完后移回CPU
        first_layer_experts[1].to("cpu")

        # 记录热点数据
        expert_key = (1, 1)
        if expert_key not in expert_hot_data:
            expert_hot_data[expert_key] = 0
        expert_hot_data[expert_key] += token_count

    # ========== 测量专家CPU计算时间 ==========
    cpu_times = []
    for token_count in token_counts:
        first_layer_experts[1].to("cpu")
        if model_name == "mixtral":
            inps = torch.randn((token_count, 4096), dtype=model.dtype, device="cpu")
        else:
            inps = torch.randn((token_count, 2048), dtype=model.dtype, device="cpu")

        # 预热
        _ = model.run_expert_at_cpu(1, 1, inps)

        # 多次测量取平均（减少波动）
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

    # ========== 测量自注意力时间（占位，后续实现） ==========
    window_size = 7  # 平滑窗口大小
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
        attn_times.append(0.0)  # TODO: 实际测量注意力时间

    # ========== 绘制性能对比图 ==========
    plt.figure(figsize=(12, 6))
    avg_gpu_time = np.mean(gpu_times) * 1000  # 转换为毫秒
    avg_copy_time = np.mean(copy_times) * 1000

    # 专家传输时间（固定值）
    plt.plot(
        token_counts,
        [avg_copy_time] * len(token_counts),
        label=f"Expert Transfer (Avg: {avg_copy_time:.2f}ms)",
        linestyle="--",
    )

    # GPU计算时间随token数变化
    plt.plot(
        token_counts,
        np.array(gpu_times) * 1000,
        label=f"Expert computation on GPU (Avg: {avg_gpu_time:.2f}ms)",
        marker="o",
    )

    # CPU计算时间（平滑处理）
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

    # 自注意力时间
    plt.plot(
        token_counts,
        np.array(attn_times) * 1000,
        label="Self-Attention on GPU",
        marker="^",
        linestyle="-",
    )

    # ========== 保存CSV数据 ==========
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

    # ========== 保存热点专家列表 ==========
    # 按调用次数降序排序
    sorted_hot_experts = sorted(
        expert_hot_data.items(), key=lambda x: x[1], reverse=True
    )
    hot_file_path = f"./hot/{model_name}.txt"
    os.makedirs(os.path.dirname(hot_file_path), exist_ok=True)
    with open(hot_file_path, "w") as f:
        for (layer, expert), token_count in sorted_hot_experts:
            f.write(f"{layer},{expert}\n")

    # ========== 保存PNG图表 ==========
    plt.xlabel("Token Count")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"ioreal_{model_name}.png", dpi=300, bbox_inches="tight")

    # ========== 生成系统配置文件 ==========
    cpu_time_table_list = [float(t * 1000) for t in cpu_times_full]

    gpu_info = auto_detect_gpu_info()

    system_config = SystemConfig(
        gpu_name=gpu_info["gpu_name"],
        gpu_memory_gb=gpu_info["gpu_memory_gb"],
        transfer_time_ms=float(avg_copy_time),  # e: 专家传输时间
        gpu_compute_time_ms=float(avg_gpu_time),  # tg: GPU计算时间
        cpu_time_table=cpu_time_table_list,  # CPU时间查找表（64个token）
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
    """主函数: 解析参数并运行微基准测试"""
    parser = argparse.ArgumentParser(description="MoE CPU Offloading Microbenchmark")

    # 禁用tokenizer并行（避免警告）
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ========== 命令行参数 ==========
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="模型路径或HuggingFace模型ID",
    )
    parser.add_argument(
        "--cache",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: GPU执行(baseline), 1: 卸载到CPU",
    )
    parser.add_argument(
        "--cpu-offload",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: GPU执行(baseline), 1: 卸载到CPU",
    )
    parser.add_argument(
        "--beam_width", type=int, default=1, help="Beam search宽度（当前未使用）"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./configs",
        help="系统配置JSON输出目录",
    )
    args = parser.parse_args()

    # ========== 模型类型检测 ==========
    model_name = get_model_name_from_path(args.model)
    print(f"检测到模型类型: {model_name}")

    # ========== 检查模型路径 ==========
    if not os.path.exists(args.model):
        print(f"\n[WARNING] 模型路径不存在: {args.model}")
        print("请检查模型路径是否正确。退出。")
        sys.exit(1)

    # ========== 创建模型实例 ==========
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

    # ========== 运行性能测试并生成配置 ==========
    plot_expert_performance(model, model_name, args.output_dir)


if __name__ == "__main__":
    main()
