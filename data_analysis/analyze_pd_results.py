#!/usr/bin/env python3
"""PD 分离 benchmark 结果（jsonl）分析脚本。

用法：
    python analyze_pd_results.py results.jsonl
    python analyze_pd_results.py results.jsonl --csv summary.csv
    python analyze_pd_results.py results.jsonl --report report.txt   # 报告直接写文件，避免管道乱码

输入：lmsys_cn_benchmark.py 产出的 jsonl，每行一个请求，含
    latency_ms / d_total_ms / decode_gpu_time_ms / decode_rounds /
    decode_round_avg_ms / decode_round_durations / d_received_from_p_ts /
    decode_completion_ts 等字段。

只依赖标准库，可在任意环境直接运行。
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path


def pct(sorted_vals, p):
    """ percentile（p 取 0-100），输入需已排序。"""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def stats(vals):
    """返回 (count, mean, p50, p90, p99, min, max)，空列表返回 None。"""
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    n = len(vals)
    return (n, sum(vals) / n, pct(vals, 50), pct(vals, 90), pct(vals, 99),
            vals[0], vals[-1])


def fmt_stats(name, s, unit="ms"):
    if s is None:
        return f"  {name:<28} 无数据"
    n, mean, p50, p90, p99, mn, mx = s
    return (f"  {name:<28} n={n:<4} mean={mean:>9.2f}{unit}  "
            f"p50={p50:>9.2f}  p90={p90:>9.2f}  p99={p99:>9.2f}  "
            f"min={mn:>9.2f}  max={mx:>9.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="benchmark 产出的 jsonl 文件")
    ap.add_argument("--csv", help="可选：把逐请求明细导出为 csv")
    ap.add_argument("--report", help="可选：报告写入该文件（utf-8），同时打印到终端")
    args = ap.parse_args()

    # 报告双写：终端 + 文件（Python 直写 utf-8，避免 PowerShell 管道转码乱码）
    report_f = open(args.report, "w", encoding="utf-8") if args.report else None

    def out(s=""):
        print(s)
        if report_f:
            report_f.write(s + "\n")

    rows = []
    for line in Path(args.input).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))

    ok = [r for r in rows if r.get("success")]
    bad = [r for r in rows if not r.get("success")]
    out(f"共 {len(rows)} 条请求，成功 {len(ok)}，失败 {len(bad)}")
    for r in bad:
        out(f"  [失败] index={r.get('index')} error={r.get('error')}")

    if not ok:
        sys.exit(0)

    # ---------- 逐请求字段提取 ----------
    for r in ok:
        lat = r.get("latency_ms")
        d_total = r.get("d_total_ms")
        gpu = r.get("decode_gpu_time_ms")
        # 衍生：P端prefill+网络 ≈ latency - D端总耗时
        r["_p_side_ms"] = (lat - d_total) if (lat is not None and d_total is not None) else None
        # 衍生：D端排队+调度 ≈ D端总耗时 - GPU decode
        r["_d_queue_ms"] = (d_total - gpu) if (d_total is not None and gpu is not None) else None
        # 衍生：D端排队占比（%）= 排队 / D端总耗时
        r["_d_queue_ratio"] = (
            r["_d_queue_ms"] / d_total * 100
            if (r["_d_queue_ms"] is not None and d_total) else None
        )
        # CSV 用：百分比形式，两位小数
        r["d_queue_pct"] = (
            round(r["_d_queue_ratio"], 2) if r["_d_queue_ratio"] is not None else None
        )

    # ---------- 汇总统计 ----------
    out("\n===== 汇总（仅成功请求） =====")
    out(fmt_stats("latency（端到端）", stats([r.get("latency_ms") for r in ok])))
    out(fmt_stats("P端prefill+网络（推算）", stats([r["_p_side_ms"] for r in ok])))
    out(fmt_stats("D端总耗时（排队+decode）", stats([r.get("d_total_ms") for r in ok])))
    out(fmt_stats("D端排队+调度（推算）", stats([r["_d_queue_ms"] for r in ok])))
    out(fmt_stats("D端排队占比", stats([r["_d_queue_ratio"] for r in ok]), unit="%"))
    out(fmt_stats("D端GPU decode（纯算）", stats([r.get("decode_gpu_time_ms") for r in ok])))
    out(fmt_stats("轮均GPU耗时", stats([r.get("decode_round_avg_ms") for r in ok])))

    # ---------- 逐轮耗时分布（跨请求合并） ----------
    all_rounds = [d for r in ok for d in (r.get("decode_round_durations") or [])]
    out("\n===== 逐轮 GPU forward 耗时分布（所有请求合并） =====")
    out(fmt_stats("每轮 forward", stats([d * 1000 for d in all_rounds])))

    # 长尾轮次 top10
    tagged = []
    for r in ok:
        for i, d in enumerate(r.get("decode_round_durations") or []):
            tagged.append((d * 1000, r.get("index"), i))
    tagged.sort(reverse=True)
    if tagged:
        out("  最慢的 10 轮：")
        for ms, rid, rnd in tagged[:10]:
            out(f"    {ms:>8.2f}ms  (请求 index={rid}, 第 {rnd} 轮)")

    # ---------- 特殊请求 ----------
    # 服务端现在返回单值 decode_gpu_time_ms；为 None 说明该请求在 D 端零 forward
    # （output_len=1 时首 token 由 P 端 prefill 直接产出）
    zero = [r for r in ok if r.get("decode_gpu_time_ms") is None]
    if zero:
        out(f"\n===== 零 decode 轮请求（共 {len(zero)} 条） =====")
        out("  说明：output_len=1 时首 token 由 P 端 prefill 直接产出，D 端无需 forward")
        for r in zero:
            out(f"    index={r.get('index')}  latency={r.get('latency_ms')}ms  "
                  f"d_total={r.get('d_total_ms')}ms")

    # ---------- 逐请求明细表 ----------
    out("\n===== 逐请求明细 =====")
    header = (f"{'idx':>4} {'latency':>9} {'P端(推算)':>10} {'D端总':>9} "
              f"{'排队':>8} {'排队%':>7} {'GPU':>9} {'轮数':>5} {'轮均':>8}")
    out(header)
    for r in ok:
        out(f"{r.get('index'):>4} "
              f"{r.get('latency_ms') or 0:>9.1f} "
              f"{r['_p_side_ms'] or 0:>10.1f} "
              f"{r.get('d_total_ms') or 0:>9.1f} "
              f"{r['_d_queue_ms'] or 0:>8.1f} "
              f"{r['_d_queue_ratio'] or 0:>6.1f}% "
              f"{r.get('decode_gpu_time_ms') or 0:>9.1f} "
              f"{r.get('decode_rounds') or 0:>5} "
              f"{(r.get('decode_round_avg_ms') or 0):>8.2f}")

    # ---------- 可选 CSV 导出 ----------
    if args.csv:
        fields = ["index", "latency_ms", "_p_side_ms", "d_total_ms",
                  "_d_queue_ms", "d_queue_pct", "decode_gpu_time_ms",
                  "decode_rounds", "decode_round_avg_ms",
                  "d_received_from_p_ts", "decode_completion_ts", "send_drift_ms"]
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(ok)
        out(f"\n明细已导出：{args.csv}")

    if report_f:
        report_f.close()


if __name__ == "__main__":
    main()
