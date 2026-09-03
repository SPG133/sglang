#!/usr/bin/env bash
# gpu_monitor.sh — 后台轮询所有 GPU 的显存/利用率，写入 CSV
# 用法：
#   source ./gpu_monitor.sh
#   start_gpu_monitor "gpu_mem_timeline.csv" "prefill"   # 启动后台监控
#   # ... 跑服务器 ...
#   stop_gpu_monitor                                  # 停止后台监控
# 或一行起停：
#   gpu_monitor_run "gpu_mem_timeline.csv" "prefill"   # 自动 trap EXIT

: "${GPU_MONITOR_INTERVAL:=0.5}"   # 采样间隔（秒），可被环境变量覆盖

start_gpu_monitor() {
    local OUTFILE="${1:-gpu_mem_timeline.csv}"
    local TAG="${2:-server}"
    local INTERVAL="${3:-$GPU_MONITOR_INTERVAL}"

    if [ ! -f "$OUTFILE" ]; then
        echo "timestamp,tag,host_pid,gpu_index,memory_used_mib,memory_free_mib,utilization_gpu_pct" > "$OUTFILE"
    fi

    (
        # 子 shell 隔离变量，避免污染调用方
        while true; do
            TS=$(date +%s.%3N)
            nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
                       --format=csv,noheader,nounits 2>/dev/null | \
            awk -F', ' -v ts="$TS" -v tag="$TAG" -v pid="$$" \
                '{ printf "%.3f,%s,%s,%s,%s,%s,%s\n", ts, tag, pid, $1, $2, $3, $4 }'
            sleep "$INTERVAL"
        done
    ) >> "$OUTFILE" &
    MONITOR_PID=$!
    echo "[gpu_monitor] 启动 (tag=$TAG pid=$MONITOR_PID 输出=$OUTFILE 间隔=${INTERVAL}s)"
}

stop_gpu_monitor() {
    if [ -n "${MONITOR_PID:-}" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        kill "$MONITOR_PID" 2>/dev/null
        wait "$MONITOR_PID" 2>/dev/null
        echo "[gpu_monitor] 停止 (pid=$MONITOR_PID)"
    fi
    MONITOR_PID=""
}

# 一体化封装：启动 + 自动 trap EXIT
gpu_monitor_run() {
    local OUTFILE="${1:-gpu_mem_timeline.csv}"
    local TAG="${2:-server}"
    start_gpu_monitor "$OUTFILE" "$TAG"
    trap 'stop_gpu_monitor' EXIT INT TERM
}