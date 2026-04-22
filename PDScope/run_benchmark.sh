#!/bin/bash


MODELS=(
    "/home/share/model/Mixtral-8x7B-v0.1"
    "/home/share/model/Qwen3-30B-A3B" 
    "/home/share/model/DeepSeek-V2-Lite"
    "/home/share/model/Moonlight-16B-A3B-Instruct"
)
INPUT_SIZES=(4096 2048 1024 512 256 128 64)
# INPUT_SIZES=(1)

for model in "${MODELS[@]}"; do

    for bs in "${INPUT_SIZES[@]}"; do
        echo "============================================"
        echo "Running benchmark for model: $model"
        echo "With batch_size: $bs"
        echo "============================================"
        python latency.py --model "$model" --batch_size 1 --input_size "$bs"

        echo ""
        echo "Benchmark for $model with batch_size=$bs completed"
        echo ""
    done
done

echo "All benchmarks completed!"
