#!/usr/bin/env bash
# 纯分级 MLFQ 的 decode 端启动脚本
# 三级反馈队列（L0/L1/L2 严格优先级准入 + 8.1s/31s 服务计量驱逐），无弹性阈值
MODEL="$PWD/models/Qwen3-8B"

CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path "$MODEL" \
  --port 30001 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl \
  --schedule-policy mlfq \
  --disable-radix-cache