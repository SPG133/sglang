"""GPU 显存时间线可视化

读 gpu_mem_timeline.csv，画两个子图：
  1) 显存占用 (MiB) vs 时间，prefill/decode 各一条
  2) GPU 利用率 (%) vs 时间

用法:
  python plot_gpu_memory.py [csv_path] [--out out.png]
默认读 ./gpu_mem_timeline.csv，输出 ./gpu_mem_timeline.png
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

def load(path):
    by_tag = defaultdict(list)   # tag -> [(ts, gpu, mem_used, mem_free, util)]
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                by_tag[row["tag"]].append((
                    float(row["timestamp"]),
                    int(row["gpu_index"]),
                    float(row["memory_used_mib"]),
                    float(row["memory_free_mib"]),
                    float(row["utilization_gpu_pct"]),
                ))
            except (KeyError, ValueError):
                continue
    for tag in by_tag:
        by_tag[tag].sort(key=lambda x: x[0])
    return by_tag

def plot(by_tag, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot_gpu_memory] matplotlib 不可用，只输出文字统计")
        text_summary(by_tag)
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    colors = {"prefill": "tab:blue", "decode": "tab:orange"}
    for tag, rows in by_tag.items():
        if not rows:
            continue
        t0 = rows[0][0]
        ts   = [r[0] - t0 for r in rows]
        mem  = [r[2] for r in rows]
        util = [r[4] for r in rows]
        c = colors.get(tag, None)
        ax1.plot(ts, mem, label=f"{tag} (gpu rows n={len(rows)})", color=c, alpha=0.7, linewidth=0.8)
        ax2.plot(ts, util, label=tag, color=c, alpha=0.7, linewidth=0.8)

    ax1.set_ylabel("显存占用 (MiB)")
    ax1.set_title("GPU 显存时间线")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("GPU 利用率 (%)")
    ax2.set_xlabel("相对启动时间 (秒)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"[plot_gpu_memory] 已保存 {out}")

    # 同时打印关键数字
    text_summary(by_tag)

def text_summary(by_tag):
    print("\n=== GPU 显存摘要 ===")
    for tag, rows in by_tag.items():
        if not rows:
            continue
        mem = [r[2] for r in rows]
        util = [r[4] for r in rows]
        gpu_idx = sorted({r[1] for r in rows})
        print(f"\n[{tag}]  GPU(s)={gpu_idx}  采样点={len(rows)}  时长={rows[-1][0]-rows[0][0]:.1f}s")
        print(f"  显存:  min={min(mem):.0f}  mean={sum(mem)/len(mem):.0f}  max={max(mem):.0f} MiB")
        print(f"  利用率: min={min(util):.1f}  mean={sum(util)/len(util):.1f}  max={max(util):.1f} %")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="gpu_mem_timeline.csv")
    ap.add_argument("--out", default="gpu_mem_timeline.png")
    args = ap.parse_args()
    p = Path(args.csv)
    if not p.exists():
        print(f"[plot_gpu_memory] {p} 不存在，先跑一次 start_prefill.sh + start_decode.sh")
        raise SystemExit(1)
    plot(load(p), args.out)