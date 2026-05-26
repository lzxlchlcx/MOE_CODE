#!/bin/bash
cd "$(dirname "$0")/src/"
python microbench.py \
    --model /mnt/g/Models/DeepSeek-v2-lite-chat \
    --cpu-offload 1 \
    --batch-size 1 \
    --beam-width 1
