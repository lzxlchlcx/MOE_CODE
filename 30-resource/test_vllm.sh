#!/bin/bash

source /home/lzx/miniconda3/etc/profile.d/conda.sh
conda activate vllm

cd /home/lzx/program/moe_code/opensource

python vllm_benchmark.py \
    --model "/mnt/g/Models/DeepSeek-v2-lite-chat" \
    --batch_size 1 \
    --cpu_offload_gb 20 \
    --max_model_len 2048 \
    --enforce_eager \
    --n_sample 1
