#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 清空日志文件
> ./log/linshi.txt
> ./log/expert_stats.txt

# 定义模型列表和batch_size列表
MODELS=(
    "/mnt/g/Models/DeepSeek-v2-lite-chat"
    # /mnt/g/Models/Qwen3-30B-A3B/
    # "/home/share/bz/model/Mixtral-8x7B-v0.1"
    # "/home/share/bz/model/Qwen3-30B-A3B" 
    # "/home/share/bz/model/Moonlight-16B-A3B-Instruct"
)
# BATCH_SIZES=(4 8 16 32 64)
# BATCH_SIZES=(1 4 8 16)
BATCH_SIZES=(1)

# 遍历所有模型
for model in "${MODELS[@]}"; do
    # 遍历所有batch_size
    for bs in "${BATCH_SIZES[@]}"; do
        echo "============================================"
        echo "Running benchmark for model: $model"
        echo "With batch_size: $bs"
        echo "============================================"
        echo "等待5秒..."

        # 执行测试命令
        python latency.py --model "$model" --batch_size "$bs"
        sleep 5        
        # python microbench.py --model "$model"        
        echo ""
        echo "Benchmark for $model with batch_size=$bs completed"
        echo ""
    done
done

echo "All benchmarks completed!"