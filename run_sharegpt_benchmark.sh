#!/usr/bin/env bash
# ShareGPT 负载压测封装（逻辑同 run_lmsys_cn_benchmark.sh）
#
# 用法:
#   bash run_sharegpt_benchmark.sh                          # 默认 1000 条 @ 2 req/s
#   REQUEST_RATE=10 WORKLOAD=sharegpt_10rps.json bash run_sharegpt_benchmark.sh
#
# 说明:
#   - workload 文件不存在时自动生成（数据源 ShareGPT；--request-rate 仅在
#     生成时生效，之后 send_offset_s 烤进文件，运行严格按文件发送）
#   - ShareGPT 原始数据（442MB）首次运行自动下载进 HF 缓存，
#     国内建议先: export HF_ENDPOINT=https://hf-mirror.com
#   - 结果输出带 sharegpt_ 前缀，避免与 lmsys 数据混淆
set -e

python lmsys_cn_benchmark.py \
  --tokenizer "$PWD/models/Qwen3-8B" \
  --endpoint http://127.0.0.1:8000/generate \
  --num-requests "${NUM_REQUESTS:-1000}" \
  --request-rate "${REQUEST_RATE:-2}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-4096}" \
  --workload "${WORKLOAD:-sharegpt_workload.json}" \
  --output "data_analysis/qwen3-8B/sharegpt_$(date +%Y%m%d_%H%M%S).jsonl"