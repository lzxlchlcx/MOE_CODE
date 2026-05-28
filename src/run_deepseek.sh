#!/bin/bash
cd "$(dirname "$0")"
PYTHONPATH="$(dirname "$0")" python ./scripts/infer_deepseek.py \
    --model /mnt/g/Models/DeepSeek-v2-lite-chat \
    --cpu-offload 0 \
    --batch-size 1 \
    --dataset /home/lzx/program/moe_code/datasets/sharegpt_v3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json \
    --n-token 20 \
    --beam-width 1
