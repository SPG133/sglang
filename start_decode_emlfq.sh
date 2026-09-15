#!/usr/bin/env bash
# E-MLFQ（MLFQ + 弹性阈值）的 decode 端启动脚本
# 在纯分级 MLFQ 基础上开启弹性防饥饿：完成请求慢化比的滑动中位数为
# 自适应水位，被降级请求等待/服务超水位即晋升回 L0
MODEL="$PWD/models/Qwen3-8B"

CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path "$MODEL" \
  --port 30001 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl \
  --schedule-policy mlfq --enable-elastic-threshold \
  --disable-radix-cache