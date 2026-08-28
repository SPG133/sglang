#!/usr/bin/env bash
MODEL="$PWD/models/Qwen3-8B"

CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path "$MODEL" \
  --port 30001 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disable-radix-cache
