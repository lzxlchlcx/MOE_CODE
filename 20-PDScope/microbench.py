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
    # """绘制专家性能折线图"""
    token_counts = list(range(1, 250))  
    ret_time = []
    
    dev = torch.device("cuda:0")
    model.dev = torch.device("cuda:0")
    if args.model=="/home/share/bz/model/Mixtral-8x7B-v0.1":
        model_type = 'mixtral'
    elif args.model=="/home/share/bz/model/Qwen3-30B-A3B":
        model_type = 'qwen'
    elif args.model=="/home/share/bz/model/DeepSeek-V2-Lite":
        model_type = 'deepseek'
    else:
        model_type = 'moon'
    if model_type == 'mixtral':
        first_layer_experts = model.model.layers[1].block_sparse_moe.experts
        n_expert = len(model.model.layers[0].block_sparse_moe.experts)
    else:  
        first_layer_experts = model.model.layers[1].mlp.experts
        n_expert = len(model.model.layers[1].mlp.experts)  
    n_shared_experts = 2
    expert_placeholder = copy.deepcopy(first_layer_experts[0]).to(dev)        

    
    copy_times = []
    for _ in token_counts:
        
        
        
        
        
        

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

        
        
        
        
        
        
        
        
    
    
    gpu_times = []
    for token_count in token_counts:
        
        first_layer_experts[1].to(model.dev)
        if model_type == 'mixtral':
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
    
    
    cpu_times = []
    for token_count in token_counts:
        
        first_layer_experts[1].to("cpu")
        if model_type == 'mixtral':
            inps = torch.randn((token_count, 4096), dtype=model.dtype, device="cpu")
        else:
            inps = torch.randn((token_count, 2048), dtype=model.dtype, device="cpu" )    
        _ = model.run_expert_at_cpu(1, 1, inps)
        
        
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
        
        if model_type == 'mixtral':
            inps = torch.randn((1, token_count, 4096), dtype=model.dtype, device=model.dev)
        else:
            inps = torch.randn((1, token_count, 2048), dtype=model.dtype, device=model.dev )
        attention_mask = torch.ones((1, 1, 1, token_count), 
                                   dtype=torch.bool, device=model.dev)
        position_ids = torch.arange(token_count, dtype=torch.long, device=model.dev).unsqueeze(0)
        
        
        layer = model.model.layers[0]
        
        
        
        
        
        
        
        
        
        torch.cuda.synchronize()
        tick = time.time()
        
        
        
        
        
        
        
        
        attn_times.append(time.time() - tick)

    plt.figure(figsize=(12, 6))
    avg_gpu_time = np.mean(gpu_times) * 1000
    
    avg_copy_time = np.mean(copy_times) * 1000    


    plt.plot(token_counts, [avg_copy_time] * len(token_counts), 
             label=f'Expert Transfer (Avg: {avg_copy_time:.2f}ms)', linestyle='--')
    
    
    plt.plot(token_counts, np.array(gpu_times) * 1000, 
             label=f'Expert computation on GPU (Avg: {avg_gpu_time:.2f}ms)', marker='o')
    
    
    
    
    
    
    window_size = 1
    cpu_times_smoothed = np.convolve(cpu_times, np.ones(window_size)/window_size, mode='valid')
    
    
    token_counts_smoothed = token_counts[window_size-1:]
    
    
    plt.plot(token_counts_smoothed, np.array(cpu_times_smoothed) * 1000, 
             label='Expert computation on CPU', marker='s', 
             linestyle='-', linewidth=2)
    plt.plot(token_counts, np.array(attn_times) * 1000,
             label='Self-Attention on GPU', marker='^', linestyle='-')    

    
    
    
    
    
    
    
    
    
    
    with open(f'micro{model_type}.txt', 'a') as f:
        
        f.write("Token Count,CPU Computation Time(ms)\n")
        for i, token_count in enumerate(token_counts):
            
            if i < window_size-1:
                f.write(f"NaN\n")
            else:
                cpu_time = cpu_times_smoothed[i-(window_size-1)]*1000 if i-(window_size-1) < len(cpu_times_smoothed) else "NaN"
                f.write(f"{cpu_time:.4f}\n")
    

    
    plt.xlabel('Token Count')
    plt.ylabel('Time (ms)')
    
    plt.legend()
    plt.grid(True)
    
    
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
        default=2,
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
    
    args = parser.parse_args()

    
    if args.model=="/home/share/bz/model/Mixtral-8x7B-v0.1":
        model = FiddlerMixtral(args)
        prefix="mix"
    elif args.model=="/home/share/bz/model/Qwen3-30B-A3B":
        model = FiddlerQwen(args)
        prefix="qwen"
    elif args.model=="/home/share/bz/model/DeepSeek-V2-Lite":
        model = FiddlerDeepSeekV2(args)
        prefix="deep"
    else:
        model = FiddlerMoon(args)   
        prefix="moon"     
    plot_expert_performance(model)

    
    
    

    
    
    
            
    
    
    
    
    
    
    
    
            
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
            
    
    
    
    
    
    
    
    
    
    

    
    
    
    

    
    
    
    
            
    
    
    
    
    
    
    
    
            
    
    
    
    
    
    
    
    
    

    
    
    
            
    
    
    
                
    
    
    
                    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    

    
    
    
    

    
    
    
    
    

    
    
    
    
    

    
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    