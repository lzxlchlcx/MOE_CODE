#!/bin/bash
cd "$(dirname "$0")"
PYTHONPATH="$(dirname "$0")" python ./scripts/analyze_hot_experts.py \
    --model /mnt/g/Models/DeepSeek-v2-lite-chat \
    --dataset /home/lzx/program/moe_code/datasets/sharegpt_v3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-samples 10 \
    --seed 0 \
    --n-token 20 \
    --output-dir ./hot \
    --output-name deep.txt \
    --cpu-offload 0 \
    --batch-size 1 \
    --beam-width 1
