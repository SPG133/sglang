#!/usr/bin/env bash
MODEL="$PWD/models/Qwen3-8B"

# 启动 GPU 显存监控（与 decode 共享同一份 CSV，tag 区分）
source "$(dirname "$0")/gpu_monitor.sh"
gpu_monitor_run "gpu_mem_timeline.csv" "prefill"

CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path "$MODEL" \
  --port 30000 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --disable-radix-cache
