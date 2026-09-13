# -*- coding: utf-8 -*-
"""把 ShareGPT 数据集转成本项目统一的 workload 格式

输出与 lmsys_cn_workload.json 完全同构：
  [{prompt, reference, input_ids, output_len, send_offset_s}, ...]

用法（在服务器上、仓库根目录）:
  # 方式一：已有 ShareGPT 原始文件（ShareGPT_V3_unfiltered_cleaned_split.json）
  python make_sharegpt_workload.py --tokenizer models/Qwen3-8B \
      --sharegpt ShareGPT_V3_unfiltered_cleaned_split.json

  # 方式二：自动从 HuggingFace 下载（国内可先 export HF_ENDPOINT=https://hf-mirror.com）
  python make_sharegpt_workload.py --tokenizer models/Qwen3-8B

  # 常用参数（与 lmsys_cn_benchmark.py 语义一致）
  python make_sharegpt_workload.py --tokenizer models/Qwen3-8B \
      --num-requests 1000 --request-rate 10 --seed 1 \
      --max-input-tokens 4096 --max-output-tokens 256 \
      --output sharegpt_workload.json

  然后照常跑 benchmark：
  python lmsys_cn_benchmark.py --tokenizer models/Qwen3-8B \
      --workload sharegpt_workload.json --num-requests 1000 --max-new-tokens 4096

注意：send_offset_s 由 --request-rate 生成后固定写死在文件里，
换速率重新生成即可（benchmark 的 --request-rate 只在文件不存在时生效）。
"""
import argparse
import json
import math
import random
import urllib.request
from pathlib import Path

# vLLM/SGLang 生态默认使用的 ShareGPT 镜像地址
DEFAULT_SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon941485/sharegpt_90k_dataset"
    "/resolve/main/sharegpt_90k_dataset.json"
)


def iter_sharegpt(path: str):
    """读 ShareGPT 原始 JSON，逐条产出 (prompt, reference)。

    兼容两种主流 schema:
      {"conversations": [{"from": "human"/"gpt", "value": ...}, ...]}   # ShareGPT_V3 / 90k
      {"conversations": [{"role": "user"/"assistant", "content": ...}]} # 部分镜像
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for row in data:
        convs = row.get("conversations") or row.get("turns") or []
        prompt, reference = "", ""
        for turn in convs:
            if "from" in turn:  # ShareGPT_V3 格式
                who, val = turn["from"], turn.get("value", "")
                if who in ("human", "user") and not prompt:
                    prompt = val
                elif who in ("gpt", "chatgpt", "assistant") and prompt and not reference:
                    reference = val
            elif "role" in turn:  # role/content 格式
                who, val = turn["role"], turn.get("content", "")
                if who == "user" and not prompt:
                    prompt = val
                elif who == "assistant" and prompt and not reference:
                    reference = val
            if prompt and reference:
                break
        yield prompt, reference


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, help="本地模型路径（取 chat template）")
    p.add_argument("--sharegpt", default=None,
                   help="ShareGPT 原始 JSON 路径；缺省则自动下载")
    p.add_argument("--num-requests", type=int, default=1000)
    p.add_argument("--request-rate", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--max-output-tokens", type=int, default=256)
    p.add_argument("--output", default="sharegpt_workload.json")
    a = p.parse_args()

    # 1. 拿到 ShareGPT 原始文件
    sg_path = a.sharegpt
    if sg_path is None or not Path(sg_path).exists():
        sg_path = "sharegpt_90k_dataset.json"
        if not Path(sg_path).exists():
            print(f"本地无 ShareGPT 文件，尝试下载:\n  {DEFAULT_SHAREGPT_URL}")
            print("（国内网络建议先: export HF_ENDPOINT=https://hf-mirror.com，"
                  "或手动下载后用 --sharegpt 指定路径）")
            urllib.request.urlretrieve(DEFAULT_SHAREGPT_URL, sg_path)
            print(f"已下载: {sg_path}")

    # 2. tokenizer（chat template 必须和线上一致）
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)

    # 3. 逐条转换 + 过滤（逻辑对齐 lmsys_cn_benchmark.py 的生成路径）
    rng, data, arrival = random.Random(a.seed), [], 0.0
    n_scanned = n_dropped_len = n_dropped_empty = 0
    for prompt, reference in iter_sharegpt(sg_path):
        if len(data) == a.num_requests:
            break
        n_scanned += 1
        if not prompt or not reference:
            n_dropped_empty += 1
            continue
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tok.encode(chat)
        output_len = min(len(tok.encode(reference)), a.max_output_tokens)
        if len(ids) > a.max_input_tokens or output_len == 0:
            n_dropped_len += 1
            continue
        if data and math.isfinite(a.request_rate):
            arrival += rng.expovariate(a.request_rate)
        data.append({
            "prompt": prompt, "reference": reference, "input_ids": ids,
            "output_len": output_len, "send_offset_s": arrival,
        })

    Path(a.output).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ins = [len(x["input_ids"]) for x in data]
    outs = [x["output_len"] for x in data]
    print(f"\n完成: {a.output}")
    print(f"  条目数: {len(data)}  (扫描 {n_scanned} 条，"
          f"空对话弃 {n_dropped_empty}，超长/空输出弃 {n_dropped_len})")
    if data:
        print(f"  input 长度: min={min(ins)} p50={sorted(ins)[len(ins)//2]} max={max(ins)}")
        print(f"  output_len: min={min(outs)} p50={sorted(outs)[len(outs)//2]} max={max(outs)}")
        print(f"  到达 span: {data[-1]['send_offset_s']:.1f}s "
              f"(≈{len(data)/data[-1]['send_offset_s']:.2f} req/s)" if data[-1]['send_offset_s'] > 0
              else "  到达 span: 0s")


if __name__ == "__main__":
    main()
