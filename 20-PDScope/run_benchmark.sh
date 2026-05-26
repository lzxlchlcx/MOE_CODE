#!/bin/bash
cd "$(dirname "$0")"
rm -f ./log/linshi.txt

MODELS=(
    "/mnt/g/Models/DeepSeek-v2-lite-chat"
)
# INPUT_SIZES=(4096 2048 1024 512 256 128 64)
BATCH_SIZES=(1)

for model in "${MODELS[@]}"; do

    for bs in "${BATCH_SIZES[@]}"; do
        echo "============================================"
        echo "Running benchmark for model: $model"
        echo "With batch_size: $bs"
        echo "============================================"
        python latency.py --model "$model" --batch_size "$bs" --cpu-offload 1

        echo ""
        echo "Benchmark for $model with batch_size=$bs completed"
        echo ""
    done
done

echo "All benchmarks completed!"
