#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 清空旧日志文件
mkdir -p ./log ./result ./configs

# ========== 可配置参数 ==========
AUTO_MICROBENCH=0         # 自动运行microbench: 1=启用, 0=禁用

# 定义模型列表和batch_size列表
MODELS=(
    "/mnt/g/Models/DeepSeek-v2-lite-chat"
)
BATCH_SIZES=(1)
PREFETCH_MODES=("disabled" "enabled")

echo "=========================================="
echo "MoE 预取对比测试"
echo "=========================================="
echo "模型: ${MODELS[*]}"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "预取模式: ${PREFETCH_MODES[*]}"
echo "自动microbench: $AUTO_MICROBENCH"
echo "=========================================="

# 遍历所有预取模式
for prefetch_mode in "${PREFETCH_MODES[@]}"; do
    # 遍历所有模型
    for model in "${MODELS[@]}"; do
        # 检查模型路径存在
        if [ ! -d "$model" ]; then
            echo ""
            echo "[WARNING] 模型路径不存在: $model"
            echo "跳过该模型。"
            continue
        fi

        # 遍历所有batch_size
        for bs in "${BATCH_SIZES[@]}"; do
            echo ""
            echo "============================================"
            echo "Running benchmark for model: $model"
            echo "With batch_size: $bs, prefetch: $prefetch_mode"
            echo "============================================"

            # 执行测试命令
            python latency.py \
                --model "$model" \
                --batch_size "$bs" \
                --prefetch "$prefetch_mode" \
                --auto-microbench "$AUTO_MICROBENCH"

            echo ""
            echo "Benchmark for $model with batch_size=$bs, prefetch=$prefetch_mode completed"
            sleep 2
        done
    done
done

echo ""
echo "All benchmarks completed!"
echo "结果保存在: ./result/"

# 生成对比总结
echo ""
echo "=========================================="
echo "预取对比总结"
echo "=========================================="
echo "结果文件:"
ls -lt ./result/result-deep-*.txt | head -12
echo ""
echo "请查看上述结果文件进行对比分析"
