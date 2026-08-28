#!/usr/bin/env bash
until curl -fsS http://127.0.0.1:30000/health >/dev/null && \
      curl -fsS http://127.0.0.1:30001/health >/dev/null; do
  echo "等待 Prefill 和 Decode..."
  sleep 2
done

python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --port 8000
