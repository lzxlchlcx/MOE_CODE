"""Microbenchmarking for CPU offloading"""

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
def plot_expert_performance(model):
    """绘制专家性能折线图"""
    token_counts = list(range(1, 65))  # 1到64个token
    ret_time = []
    # model=model.model
    dev = torch.device("cuda:0")
    model.dev = torch.device("cuda:0")
    if args.model=="/home/share/bz/model/Mixtral-8x7B-v0.1":
        model_type = 'mixtral'
    elif args.model=="/mnt/g/Models/Qwen3-30B-A3B" or args.model=="/mnt/g/Models/Qwen3-30B-A3B/":
        model_type = 'qwen'
    elif args.model=="/mnt/g/Models/DeepSeek-v2-lite-chat":
        model_type = 'deepseek'
    elif args.model=="/home/share/bz/model/Moonlight-16B-A3B-Instruct":
        model_type = 'moon'
    else:
        # 通过关键词判断
        if 'Mixtral' in args.model or 'mixtral' in args.model:
            model_type = 'mixtral'
        elif 'Qwen' in args.model or 'qwen' in args.model:
            model_type = 'qwen'
        elif 'DeepSeek' in args.model or 'deepseek' in args.model:
            model_type = 'deepseek'
        elif 'Moonlight' in args.model or 'moon' in args.model:
            model_type = 'moon'
        else:
            model_type = 'moon'
    if model_type == 'mixtral':
        first_layer_experts = model.model.layers[1].block_sparse_moe.experts
        n_expert = len(model.model.layers[0].block_sparse_moe.experts)
    else:  # qwen/moon/deepseek
        first_layer_experts = model.model.layers[1].mlp.experts
        n_expert = len(model.model.layers[1].mlp.experts)  
    n_shared_experts = 2
    expert_placeholder = copy.deepcopy(first_layer_experts[0]).to(dev)        

    # 1. 专家搬运开销 (固定值)
    copy_times = []
    for _ in token_counts:
        # 测量一次搬运时间
        # expert_placeholder = copy.deepcopy(model.model.layers[0].block_sparse_moe.experts[0]).to(model.dev)
        # model.model.layers[0].block_sparse_moe.experts[0].to("cpu")
        # for name in ['w1', 'w2', 'w3']:
        #     w = getattr(model.model.layers[0].block_sparse_moe.experts[0], name)
        #     w.weight.data = w.weight.data.pin_memory()

        for i in range(2,3):
            first_layer_experts[i].to("cpu")
            if model_type == 'mixtral': 
                for name in ['w1', 'w2', 'w3']:
                    w = getattr(first_layer_experts[i], name)
                    src_weight_data_tensor = w.weight.data 
                    pinned = src_weight_data_tensor.pin_memory()
                    w.weight.data = pinned
            else: 
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    w = getattr(first_layer_experts[i], name)
                    src_weight_data_tensor = w.weight.data 
                    pinned = src_weight_data_tensor.pin_memory()
                    w.weight.data = pinned      
            # print("111")
            torch.cuda.synchronize()
            tick = time.time()
            if model_type == 'mixtral': 
                for name in ['w1', 'w2', 'w3']:
                    dst = getattr(expert_placeholder, name).weight.data
                    src = getattr(first_layer_experts[i], name).weight.data
                    dst.copy_(src)
            else:
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    dst = getattr(expert_placeholder, name).weight.data
                    src = getattr(first_layer_experts[i], name).weight.data
                    dst.copy_(src)                
            torch.cuda.synchronize()
            copy_times.append(time.time() - tick)

        # torch.cuda.synchronize()
        # tick = time.time()
        # for name in ['w1', 'w2', 'w3']:
        #     dst = getattr(expert_placeholder, name).weight.data
        #     src = getattr(model.model.layers[0].block_sparse_moe.experts[0], name).weight.data
        #     dst.copy_(src)
        # torch.cuda.synchronize()
        # copy_times.append(time.time() - tick)
    
    # 2. GPU计算开销
    gpu_times = []
    for token_count in token_counts:
        # 预热
        first_layer_experts[1].to(model.dev)
        if model_type == 'mixtral':
            inps = torch.randn((token_count, 4096), dtype=model.dtype, device=model.dev)
        else:
            inps = torch.randn((token_count, 2048), dtype=model.dtype, device=model.dev)
        _ = first_layer_experts[1](inps)
        
        # 正式测量
        torch.cuda.synchronize()
        tick = time.time()
        _ = first_layer_experts[1](inps)
        torch.cuda.synchronize()
        gpu_times.append(time.time() - tick)
        first_layer_experts[1].to("cpu")
    print("222")    
    # 3. CPU计算开销
    cpu_times = []
    for token_count in token_counts:
        # 预热
        first_layer_experts[1].to("cpu")
        if model_type == 'mixtral':
            inps = torch.randn((token_count, 4096), dtype=model.dtype, device="cpu")
        else:
            inps = torch.randn((token_count, 2048), dtype=model.dtype, device="cpu" )    
        _ = model.run_expert_at_cpu(1, 1, inps)
        
        # 测量10次取平均值
        measurements = []
        for _ in range(20):
            torch.cuda.synchronize()
            tick = time.time()
            _ = model.run_expert_at_cpu(1, 1, inps)
            torch.cuda.synchronize()
            measurements.append(time.time() - tick)
        
        cpu_times.append(np.mean(measurements))    
    window_size = 7
    attn_times = []
    for token_count in token_counts:
        # 准备输入
        if model_type == 'mixtral':
            inps = torch.randn((1, token_count, 4096), dtype=model.dtype, device=model.dev)
        else:
            inps = torch.randn((1, token_count, 2048), dtype=model.dtype, device=model.dev )
        attention_mask = torch.ones((1, 1, 1, token_count), 
                                   dtype=torch.bool, device=model.dev)
        position_ids = torch.arange(token_count, dtype=torch.long, device=model.dev).unsqueeze(0)
        # position_embeddings = model.model.rotary_emb(inps, position_ids)        
        # 预热
        layer = model.model.layers[0]
        # _ = layer.self_attn(
        #     hidden_states=inps,
        #     attention_mask=attention_mask,
        #         position_embeddings=position_embeddings,  # 传递嵌入后的位置信息
        #     past_key_value=None,
        #     use_cache=False
        # )
        
        # 正式测量
        torch.cuda.synchronize()
        tick = time.time()
        # _ = layer.self_attn(
        #     hidden_states=inps,
        #     attention_mask=attention_mask,
        #         position_embeddings=position_embeddings,  # 传递嵌入后的位置信息
        #     past_key_value=None,
        #     use_cache=False
        # )
        # torch.cuda.synchronize()
        attn_times.append(time.time() - tick)

    plt.figure(figsize=(12, 6))
    avg_gpu_time = np.mean(gpu_times) * 1000
    # 专家搬运开销 (取平均值)
    avg_copy_time = np.mean(copy_times) * 1000    


    plt.plot(token_counts, [avg_copy_time] * len(token_counts), 
             label=f'Expert Transfer (Avg: {avg_copy_time:.2f}ms)', linestyle='--')
    
    # GPU计算开销
    plt.plot(token_counts, np.array(gpu_times) * 1000, 
             label=f'Expert computation on GPU (Avg: {avg_gpu_time:.2f}ms)', marker='o')
    
    # CPU计算开销
    # plt.errorbar(token_counts, np.array(cpu_times) * 1000, 
    #              yerr=np.array(cpu_stds) * 1000, 
    #              label='CPU Computation', marker='s', 
    #              capsize=3, elinewidth=1, markeredgewidth=1)
    window_size = 7
    cpu_times_smoothed = np.convolve(cpu_times, np.ones(window_size)/window_size, mode='valid')
    
    # 调整token_counts以匹配平滑后的数据长度
    token_counts_smoothed = token_counts[window_size-1:]
    
    # 绘制平滑后的CPU计算开销
    plt.plot(token_counts_smoothed, np.array(cpu_times_smoothed) * 1000, 
             label='Expert computation on CPU', marker='s', 
             linestyle='-', linewidth=2)
    plt.plot(token_counts, np.array(attn_times) * 1000,
             label='Self-Attention on GPU', marker='^', linestyle='-')    

    with open(f'micro{model_type}_New.txt', 'a') as f:
        # 更新表头
        f.write("Token Count,Expert Transfer Time(ms),GPU Computation Time(ms),CPU Computation Time(ms),Self-Attention Time(ms)\n")
        for i, token_count in enumerate(token_counts):
            # 前window_size-1个token没有平滑后的CPU时间
            if i < window_size-1:
                f.write(f"{token_count},{avg_copy_time:.4f},{gpu_times[i]*1000:.4f},NaN,{attn_times[i]*1000:.4f}\n")
            else:
                cpu_time = cpu_times_smoothed[i-(window_size-1)]*1000 if i-(window_size-1) < len(cpu_times_smoothed) else "NaN"
                f.write(f"{token_count},{avg_copy_time:.4f},{gpu_times[i]*1000:.4f},{cpu_time:.4f},{attn_times[i]*1000:.4f}\n")
    # with open(f'micro{model_type}.txt', 'a') as f:
    #     # 更新表头
    #     f.write("Token Count,CPU Computation Time(ms)\n")
    #     for i, token_count in enumerate(token_counts):
    #         # 前window_size-1个token没有平滑后的CPU时间
    #         if i < window_size-1:
    #             f.write(f"NaN\n")
    #         else:
    #             cpu_time = cpu_times_smoothed[i-(window_size-1)]*1000 if i-(window_size-1) < len(cpu_times_smoothed) else "NaN"
    #             f.write(f"{cpu_time:.4f}\n")
    # 绘制折线图

    # 绘制折线图
    plt.xlabel('Token Count')
    plt.ylabel('Time (ms)')
    # plt.title('Performance Comparison on V100')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    plt.savefig(f'ioreal_{model_type}.png', dpi=300, bbox_inches='tight')
    plt.show()

def weight_copy(model, from_cpu=True):
    """Time to copy weights of an expert"""
    ret_time = []

    if from_cpu:
        expert_placeholder = copy.deepcopy(
            model.model.layers[0].block_sparse_moe.experts[0]
        ).to(model.dev)
        for i in range(32):
            model.model.layers[i].block_sparse_moe.experts[0].to("cpu")
            for name in ['w1', 'w2', 'w3']:
                w = getattr(model.model.layers[i].block_sparse_moe.experts[0], name)
                src_weight_data_tensor = w.weight.data 
                pinned = src_weight_data_tensor.pin_memory()
                w.weight.data = pinned

            torch.cuda.synchronize()
            tick = time.time()
            for name in ['w1', 'w2', 'w3']:
                dst = getattr(expert_placeholder, name).weight.data
                src = getattr(model.model.layers[i].block_sparse_moe.experts[0], name).weight.data
                dst.copy_(src)
            torch.cuda.synchronize()
            # torch.cuda.synchronize()
            # tick = time.time()
            # expert_placeholder.load_state_dict(
            #     model.model.layers[i].block_sparse_moe.experts[0].state_dict()
            # )
            # torch.cuda.synchronize()
            ret_time.append(time.time() - tick)
            model.model.layers[i].block_sparse_moe.experts[0].to("cpu")
    else:
        expert_placeholder = copy.deepcopy(
            model.model.layers[0].block_sparse_moe.experts[0]
        ).to("cpu")
        for i in range(16):
            model.model.layers[i].block_sparse_moe.experts[0].to(model.dev)
            torch.cuda.synchronize()
            tick = time.time()
            expert_placeholder.load_state_dict(
                model.model.layers[i].block_sparse_moe.experts[0].state_dict()
            )
            torch.cuda.synchronize()
            ret_time.append(time.time() - tick)
    return np.array(ret_time)


def copy_activation(model, from_cpu=True):
    """Time to copy activations"""
    ret_time = []
    if from_cpu:
        for i in range(32):
            inps = torch.randn((1, 4096), dtype=model.dtype, device="cpu")
            torch.cuda.synchronize()
            tick = time.time()
            inps = inps.to(model.dev)
            torch.cuda.synchronize()
            ret_time.append(time.time() - tick)
            del inps
    else:
        for i in range(32):
            inps = torch.randn((1, 4096), dtype=model.dtype, device=model.dev)
            torch.cuda.synchronize()
            tick = time.time()
            inps = inps.to("cpu")
            torch.cuda.synchronize()
            ret_time.append(time.time() - tick)
            del inps
    return np.array(ret_time)


def expert_gpu(model, n_expert=1, batch_size=1):
    """Time to execute an expert at GPU"""
    ret_time = []

    # warm up
    model.model.layers[0].block_sparse_moe.experts[7].to(model.dev)
    inps = torch.randn((batch_size, 4096), dtype=model.dtype, device=model.dev)
    weights = torch.ones((batch_size, 1), dtype=model.dtype, device=model.dev)
    inps = model.model.layers[0].block_sparse_moe.experts[7](inps)
    model.model.layers[0].block_sparse_moe.experts[7].to("cpu")
    del inps, weights
    torch.cuda.synchronize()

    for i in range(16):
        for j in range(n_expert):
            model.model.layers[i].block_sparse_moe.experts[j].to(model.dev)
            inps = torch.randn((batch_size, 4096), dtype=model.dtype, device=model.dev)
            weights = torch.randn((batch_size, 1), dtype=model.dtype, device=model.dev)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            tick = time.time()
            inps = model.model.layers[i].block_sparse_moe.experts[j](inps)
            torch.cuda.synchronize()
            ret_time.append(time.time() - tick)
            model.model.layers[i].block_sparse_moe.experts[j].to("cpu")
            del inps, weights
    return np.array(ret_time)


def expert_cpu(model, n_expert=1, batch_size=1, multithreading=False):
    """Time to execute an expert at CPU"""
    ret_time = []
    # warm up
    model.model.layers[0].block_sparse_moe.experts[7].to("cpu")
    inps = torch.randn((batch_size, 4096), dtype=model.dtype, device="cpu")
    weights = torch.ones((batch_size, 1), dtype=model.dtype, device="cpu")
    torch.cuda.synchronize()
    tick = time.time()
    inps = model.run_expert_at_cpu(0, 7, inps)
    del inps, weights
    torch.cuda.synchronize()

    for i in range(32):
        for j in range(n_expert):
            model.model.layers[i].block_sparse_moe.experts[j].to("cpu")
            inps = torch.randn((batch_size, 4096), dtype=model.dtype, device="cpu")
            weights = torch.randn((batch_size, 1), dtype=model.dtype, device="cpu")
            torch.cuda.synchronize()
            tick = time.time()
            inps = model.run_expert_at_cpu(i, j, inps)
            torch.cuda.synchronize()
            ret_time.append(time.time() - tick)
            del inps, weights
    return np.array(ret_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model path. default `mistralai/Mixtral-8x7B-v0.1`.",
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
    parser.add_argument("--batch_size", type=int, default=1, help="Beam search number.")    
    # args = parser.parse_args()
    args = parser.parse_args()

    # model = FiddlerMixtral(args)
    if args.model=="/home/share/bz/model/Mixtral-8x7B-v0.1":
        model = FiddlerMixtral(args)
        prefix="mix"
    elif args.model=="/mnt/g/Models/Qwen3-30B-A3B" or args.model=="/mnt/g/Models/Qwen3-30B-A3B/":
        model = FiddlerQwen(args)
        prefix="qwen"
    elif args.model=="/mnt/g/Models/DeepSeek-v2-lite-chat":
        model = FiddlerDeepSeekV2(args)
        prefix="deep"
    elif args.model=="/home/share/bz/model/Moonlight-16B-A3B-Instruct":
        model = FiddlerMoon(args)
        prefix="moon"
    else:
        # 通过关键词判断
        if 'Mixtral' in args.model or 'mixtral' in args.model:
            model = FiddlerMixtral(args)
            prefix="mix"
        elif 'Qwen' in args.model or 'qwen' in args.model:
            model = FiddlerQwen(args)
            prefix="qwen"
        elif 'DeepSeek' in args.model or 'deepseek' in args.model:
            model = FiddlerDeepSeekV2(args)
            prefix="deep"
        elif 'Moonlight' in args.model or 'moon' in args.model:
            model = FiddlerMoon(args)
            prefix="moon"
        else:
            model = FiddlerMoon(args)
            prefix="moon"    
    plot_expert_performance(model)

    # def weight_copy_parallel(model, from_cpu=True, num_experts=8):
    #     """并行拷贝单层所有专家的权重"""
    #     ret_time = []

    #     if from_cpu:
    #         # 创建独立CUDA流（每个专家一个流）
    #         streams = [torch.cuda.Stream() for _ in range(num_experts)]
            
    #         # 预分配所有GPU缓冲区（在主流中预先分配）
    #         expert_buffers = []
    #         for j in range(num_experts):
    #             expert = model.model.layers[0].block_sparse_moe.experts[j].to("cpu")
    #             w1 = torch.empty_like(expert.w1.weight.data, device='cuda')
    #             w2 = torch.empty_like(expert.w2.weight.data, device='cuda')
    #             w3 = torch.empty_like(expert.w3.weight.data, device='cuda')
    #             expert_buffers.append((w1, w2, w3))
            
    #         # 准备CPU数据（固定内存）
    #         cpu_experts = []
    #         for j in range(num_experts):
    #             expert = model.model.layers[0].block_sparse_moe.experts[j].to("cpu")
    #             for name in ['w1', 'w2', 'w3']:
    #                 w = getattr(expert, name)
    #                 w.weight.data = w.weight.data.pin_memory()
    #             cpu_experts.append(expert)

    #         # 启动异步传输
    #         torch.cuda.synchronize()
    #         tick = time.time()
    #         for j in range(num_experts):
    #             with torch.cuda.stream(streams[j]):
    #                 # 并行拷贝三个权重矩阵
    #                 expert_buffers[j][0].copy_(cpu_experts[j].w1.weight.data, non_blocking=True)
    #                 expert_buffers[j][1].copy_(cpu_experts[j].w2.weight.data, non_blocking=True)
    #                 expert_buffers[j][2].copy_(cpu_experts[j].w3.weight.data, non_blocking=True)
    #         # end_event.record()
            
    #         # 等待所有流完成
    #         # torch.cuda.synchronize()
    #         torch.cuda.synchronize()
    #         ret_time.append(time.time() - tick)
    #         print(ret_time)
    #         # 清理缓冲区
    #         for buf in expert_buffers:
    #             for tensor in buf:
    #                 del tensor
    #         torch.cuda.empty_cache()

    #     return np.array(ret_time)
    # def weight_copy_parallel_optimized(model, from_cpu=True, num_experts=8):
    #     """集成流分组/矩阵交错/零拷贝的并行传输（兼容旧版本PyTorch）"""
    #     ret_time = []

    #     if from_cpu and torch.cuda.is_available():
    #         # 获取GPU DMA引擎数量（默认2，常见显卡为2-4个）
    #         num_dma_engines = 2  # 可通过nvidia-smi -q查询实际值
    #         streams = [torch.cuda.Stream() for _ in range(num_dma_engines)]
            
    #         # 预分配GPU缓冲区 --------------------------------------------------
    #         expert_buffers = []
    #         for j in range(num_experts):
    #             expert = model.model.layers[0].block_sparse_moe.experts[j].to("cpu")
    #             w1 = torch.empty_like(expert.w1.weight.data, device='cuda')
    #             w2 = torch.empty_like(expert.w2.weight.data, device='cuda')
    #             w3 = torch.empty_like(expert.w3.weight.data, device='cuda')
    #             expert_buffers.append((w1, w2, w3))
            
    #         # 准备CPU数据（固定内存+内存交错优化）---------------------------------
    #         cpu_experts = []
    #         for j in range(num_experts):
    #             expert = model.model.layers[0].block_sparse_moe.experts[j].to("cpu")
    #             # 对三个权重矩阵分别进行内存优化
    #             expert.w1.weight.data = expert.w1.weight.data.contiguous().pin_memory()
    #             expert.w2.weight.data = expert.w2.weight.data.contiguous().pin_memory()
    #             expert.w3.weight.data = expert.w3.weight.data.contiguous().pin_memory()
    #             cpu_experts.append(expert)

    #         # 启动异步传输 ------------------------------------------------------
    #         torch.cuda.synchronize()
    #         tick = time.time()
            
    #         # 策略1：流分组批处理
    #         for group_start in range(0, num_experts, num_dma_engines):
    #             group_end = min(group_start + num_dma_engines, num_experts)
                
    #             # 策略2：权重矩阵交错分配到不同流
    #             for idx_in_group, expert_id in enumerate(range(group_start, group_end)):
    #                 stream_idx = idx_in_group % num_dma_engines
                    
    #                 with torch.cuda.stream(streams[stream_idx]):
    #                     # 每个专家的三个权重使用同一个流顺序传输
    #                     expert_buffers[expert_id][0].copy_(
    #                         cpu_experts[expert_id].w1.weight.data, non_blocking=True)
    #                     expert_buffers[expert_id][1].copy_(
    #                         cpu_experts[expert_id].w2.weight.data, non_blocking=True)
    #                     expert_buffers[expert_id][2].copy_(
    #                         cpu_experts[expert_id].w3.weight.data, non_blocking=True)

    #         # if enable_unified_memory:
    #         #     for expert_id in range(num_experts):
    #         #         expert_buffers[expert_id] = (
    #         #             cpu_experts[expert_id].w1.weight.data.to('cuda', non_blocking=True),
    #         #             cpu_experts[expert_id].w2.weight.data.to('cuda', non_blocking=True),
    #         #             cpu_experts[expert_id].w3.weight.data.to('cuda', non_blocking=True)
    #         #         )

    #         # 同步计时 ----------------------------------------------------------
    #         torch.cuda.synchronize()
    #         ret_time.append(time.time() - tick)
    #         print(f"传输耗时: {ret_time[-1]:.4f}s")

    #         # 清理资源 ----------------------------------------------------------
    #         for buf in expert_buffers:
    #             for tensor in buf:
    #                 del tensor
    #         torch.cuda.empty_cache()

    #     return np.array(ret_time)
    # def format_output(array):
    #     return (
    #         f"mean: {np.mean(array) * 1000:.2f} ms, std: {np.std(array) * 1000:.2f} ms"
    #     )

    # test_names = []
    # mean_times = []
    # std_times = []


    # # 1) Weight copy, CPU -> GPU
    # data = weight_copy(model, from_cpu=True)
    # test_names.append("Weight CPU->GPU")
    # mean_times.append(np.mean(data) * 1000)
    # std_times.append(np.std(data) * 1000)
    
    # # 2) Weight copy, GPU -> CPU
    # data = weight_copy(model, from_cpu=False)
    # test_names.append("Weight GPU->CPU")
    # mean_times.append(np.mean(data) * 1000)
    # std_times.append(np.std(data) * 1000)
    
    # # 3) Activation copy, CPU -> GPU
    # data = copy_activation(model, from_cpu=True)
    # test_names.append("Activation CPU->GPU")
    # mean_times.append(np.mean(data) * 1000)
    # std_times.append(np.std(data) * 1000)
    
    # # 4) Activation copy, GPU -> CPU
    # data = copy_activation(model, from_cpu=False)
    # test_names.append("Activation GPU->CPU")
    # mean_times.append(np.mean(data) * 1000)
    # std_times.append(np.std(data) * 1000)
    
    # # 5) Execution, GPU
    # batch_sizes = [1, 2, 4, 8, 16, 32]
    # for bs in batch_sizes:
    #     data = expert_gpu(model, batch_size=bs)
    #     test_names.append(f"GPU Exec bs={bs}")
    #     mean_times.append(np.mean(data) * 1000)
    #     std_times.append(np.std(data) * 1000)
    # for bs in batch_sizes:
    #     data = expert_cpu(model, batch_size=bs)
    #     test_names.append(f"CPU Exec bs={bs}")
    #     mean_times.append(np.mean(data) * 1000)
    #     std_times.append(np.std(data) * 1000)

    # # 创建柱状图
    # plt.figure(figsize=(16, 8))
    # x = np.arange(len(test_names))
    # width = 0.35
    
    # bars = plt.bar(x, mean_times, width, yerr=std_times, capsize=5)
    
    # # 添加数值标签
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2., height,
    #             f'{height:.2f}',
    #             ha='center', va='bottom', fontsize=8)
    
    # plt.title('Microbenchmark Results (Mean Time ± Std Dev)')
    # plt.ylabel('Time (ms)')
    # plt.xticks(x, test_names, rotation=45, ha='right')
    # plt.tight_layout()
    
    # # 保存图表
    # plt.savefig('microbench_results.png', dpi=300, bbox_inches='tight')
    # plt.show()