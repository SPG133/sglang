#!/usr/bin/env bash
set -e

python lmsys_cn_benchmark.py \
  --tokenizer "$PWD/models/Qwen3-8B" \
  --endpoint http://127.0.0.1:8000/generate \
  --num-requests "${NUM_REQUESTS:-1000}" \
  --request-rate "${REQUEST_RATE:-2}" \
  --workload lmsys_cn_workload.json
