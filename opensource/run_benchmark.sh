#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 清空旧日志文件
mkdir -p ./log ./result ./configs
> ./log/linshi.txt
> ./log/expert_stats.txt

# ========== 可配置参数 ==========
PREFETCH="enabled"        # 预取策略: disabled / enabled
AUTO_MICROBENCH=1          # 自动运行microbench: 1=启用, 0=禁用
LOG_DIR="./log"            # 日志目录

# 定义模型列表和batch_size列表
# 仅测试 deepseek 和 qwen
MODELS=(
    "/mnt/g/Models/DeepSeek-v2-lite-chat"
    # "/mnt/g/Models/Qwen3-30B-A3B/"
)
BATCH_SIZES=(1)
# BATCH_SIZES=(1 4 8 16)

# ========== 解析命令行参数 ==========
while [[ $# -gt 0 ]]; do
    case $1 in
        --prefetch)
            PREFETCH="$2"
            shift 2
            ;;
        --no-auto-microbench)
            AUTO_MICROBENCH=0
            shift
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--prefetch disabled|enabled] [--no-auto-microbench] [--log-dir DIR]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "MoE 基准测试批量运行"
echo "=========================================="
echo "预取策略: $PREFETCH"
echo "自动microbench: $AUTO_MICROBENCH"
echo "日志目录: $LOG_DIR"
echo "=========================================="

# 遍历所有模型
for model in "${MODELS[@]}"; do
    # 检查模型路径是否存在
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
        echo "With batch_size: $bs, prefetch: $PREFETCH"
        echo "============================================"

        # 执行测试命令
        python latency.py \
            --model "$model" \
            --batch_size "$bs" \
            --prefetch "$PREFETCH" \
            --auto-microbench "$AUTO_MICROBENCH" \
            --log-dir "$LOG_DIR"

        echo ""
        echo "Benchmark for $model with batch_size=$bs completed"
        sleep 2
    done
done

echo ""
echo "All benchmarks completed!"
echo "结果保存在: $LOG_DIR 和 ./result/"
