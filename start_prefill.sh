#!/usr/bin/env bash
MODEL="$PWD/models/Qwen3-8B"

CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path "$MODEL" \
  --port 30000 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --disable-radix-cache
