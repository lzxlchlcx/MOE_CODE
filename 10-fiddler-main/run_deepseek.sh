#!/bin/bash
cd "$(dirname "$0")/src/fiddler"
python infer_deepseek.py \
    --model /mnt/g/Models/DeepSeek-v2-lite-chat \
    --cpu-offload 1 \
    --batch-size 1 \
    --input "Please tell me a joke." \
    --n-token 120 \
    --beam-width 1
