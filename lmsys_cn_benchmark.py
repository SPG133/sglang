#!/usr/bin/env python3
import argparse, asyncio, json, math, random, time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from datasets import load_dataset
from transformers import AutoTokenizer


def iso():
    return datetime.now(timezone.utc).isoformat()


def load_workload(a):
    path = Path(a.workload)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"复用固定 workload：{path}")
        return data[: a.num_requests]

    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    ds = ds.shuffle(seed=a.seed, buffer_size=10_000)
    rng, data, arrival = random.Random(a.seed), [], 0.0

    for row in ds:
        if str(row.get("language", "")).lower() != "chinese":
            continue
        conv = row.get("conversation", [])
        if len(conv) < 2:
            continue
        prompt, reference = conv[0].get("content", ""), conv[1].get("content", "")
        if not prompt or not reference:
            continue
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tok.encode(chat)
        output_len = min(len(tok.encode(reference)), a.max_output_tokens)
        if len(ids) > a.max_input_tokens or output_len == 0:
            continue
        if data and math.isfinite(a.request_rate):
            arrival += rng.expovariate(a.request_rate)
        data.append({
            "prompt": prompt, "reference": reference, "input_ids": ids,
            "output_len": output_len, "send_offset_s": arrival,
        })
        if len(data) == a.num_requests:
            break

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"生成固定 workload：{path}")
    return data


async def send(session, a, req, index, begin):
    planned = req["send_offset_s"]
    await asyncio.sleep(max(0, begin + planned - time.perf_counter()))
    start, sent_at = time.perf_counter(), iso()
    answer, first_at, error = "", None, None
    meta = {}  # 流式每个 chunk 都带 meta_info，最后一个 chunk 字段最全
    payload = {
        "input_ids": req["input_ids"], "stream": True,
        "sampling_params": {
            "temperature": 0,
            # 不设上限：生成到 EOS 自然停止。服务端 init_req_max_new_tokens
            # 会 clamp 到上下文剩余长度，传超大值是安全的。
            # 注意不能省略该字段（SGLang 默认 128），也不能开 ignore_eos。
            "max_new_tokens": a.max_new_tokens,
        },
    }

    try:
        async with session.post(a.endpoint, json=payload) as response:
            if response.status != 200:
                raise RuntimeError(await response.text())
            async for line in response.content:
                line = line.strip()
                if line.startswith(b"data: "):
                    line = line[6:]
                if not line or line == b"[DONE]":
                    continue
                chunk = json.loads(line)
                if chunk.get("meta_info"):
                    meta = chunk["meta_info"]
                text = chunk.get("text", "")
                if text:
                    first_at = first_at or iso()
                    answer = text
    except Exception as e:
        error = str(e)

    actual = start - begin

    # ===== 服务端新增的 PD 分离计时字段 =====
    d_recv_ts = meta.get("d_received_from_p_ts")       # D端从P端接收完成时刻
    d_done_ts = meta.get("decode_completion_ts")       # D端真正结束时刻
    gpu_total_s = meta.get("decode_gpu_total_time")    # 整个 decode 的 GPU 总耗时（秒）
    true_wait_s = meta.get("d_true_wait_s")            # 真正的 D 端等待（P完成→首次decode）
    gpu_fraction = meta.get("d_gpu_fraction")          # 最终指标：GPU时间/(true_wait+d_total)
    p_finish_ts = meta.get("p_prefill_finished_ts")    # P 端 prefill 完成的 wall-clock
    decode_life = meta.get("d_decode_life_fraction")   # decode GPU / (到达P→D完成) 生命周期占比
    prefill_life = meta.get("d_prefill_life_fraction") # prefill GPU / (到达P→D完成) 生命周期占比
    p_arrival_ts = meta.get("p_arrival_ts")            # 请求到达 P 端的 wall-clock
    prefill_gpu_ms = meta.get("prefill_gpu_total_time")  # P 端 prefill GPU 实际耗时（秒）

    d_total_ms = round((d_done_ts - d_recv_ts) * 1000, 3) if d_recv_ts and d_done_ts else None

    return {
        "index": index, "prompt": req["prompt"],
        "reference_answer": req["reference"], "generated_answer": answer,
        "planned_send_offset_s": planned, "actual_send_offset_s": round(actual, 6),
        "send_drift_ms": round((actual - planned) * 1000, 3),
        "sent_at": sent_at, "first_token_at": first_at, "finished_at": iso(),
        "latency_ms": round((time.perf_counter() - start) * 1000, 3),
        # ===== 新增字段 =====
        "d_received_from_p_ts": d_recv_ts,
        "decode_completion_ts": d_done_ts,
        "d_total_ms": d_total_ms,                       # D端排队+decode 总耗时
        "decode_gpu_time_ms": round(gpu_total_s * 1000, 3) if gpu_total_s else None,
        "true_wait_ms": round(true_wait_s * 1000, 3) if true_wait_s is not None else None,
        "gpu_fraction": round(gpu_fraction, 6) if gpu_fraction is not None else None,
        "p_prefill_finished_ts": p_finish_ts,
        "decode_life_fraction": round(decode_life, 6) if decode_life is not None else None,
        "prefill_life_fraction": round(prefill_life, 6) if prefill_life is not None else None,
        "p_arrival_ts": p_arrival_ts,
        "prefill_gpu_time_ms": round(prefill_gpu_ms * 1000, 3) if prefill_gpu_ms else None,
        "completion_tokens": meta.get("completion_tokens"),  # 真实输出长度（不再固定）
        "finish_reason": (meta.get("finish_reason") or {}).get("type"),
        "success": error is None, "error": error,
    }


async def main(a):
    workload, begin = load_workload(a), time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(*[
            send(session, a, req, i, begin) for i, req in enumerate(workload)
        ])
    # 默认保存到数据分析文件夹（data_analysis/qwen3-8B/），目录不存在则自动创建
    output = a.output or str(
        Path("data_analysis") / "qwen3-8B" / datetime.now().strftime("lmsys_cn_%Y%m%d_%H%M%S.jsonl")
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("".join(
        json.dumps(x, ensure_ascii=False) + "\n" for x in results
    ), encoding="utf-8")
    print(f"完成：{output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--endpoint", default="http://127.0.0.1:8000/generate")
    p.add_argument("--workload", default="lmsys_cn_workload.json")
    p.add_argument("--num-requests", type=int, default=50)
    p.add_argument("--request-rate", type=float, default=2)
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--max-output-tokens", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output")
    asyncio.run(main(p.parse_args()))
