#!/bin/bash
python ./benchmark/microbench.py \
    --model /mnt/g/Models/DeepSeek-v2-lite-chat \
    --cpu-offload 1 \
    --batch-size 1 \
    --beam-width 1
