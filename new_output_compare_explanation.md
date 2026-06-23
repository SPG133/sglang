# `main...46fc0ba67` 新增代码说明

对比链接：

<https://github.com/sgl-project/sglang/compare/main...46fc0ba674ad570caf66a34033845f7627e01776>

本地等价命令：

```bash
git diff --stat main...46fc0ba674ad570caf66a34033845f7627e01776
git diff main...46fc0ba674ad570caf66a34033845f7627e01776
```

本次 compare 的实际变更很集中：

```text
python/sglang/srt/managers/schedule_batch.py |  5 ++++
python/sglang/srt/managers/scheduler.py      | 39 ++++++++++++++++++++++++++++
2 files changed, 44 insertions(+)
```

整体来看，这批新增代码不是修改 MLFQ 的调度排序规则，而是给已有的调度实验字段增加“请求级 timing 落盘”能力。也就是说，原来 request 内部已经记录了 `scheduler_enqueue_time`、`prefill_execution_time`、`decode_execution_time`、`kv_transfer_time`、`waiting_time`、`mlfq_level` 等字段；这次新增代码负责在请求完成时，把这些字段写入独立 JSONL 文件，方便 benchmark 结束后离线分析。

## 1. `schedule_batch.py`

文件：

```text
python/sglang/srt/managers/schedule_batch.py
```

### 1.1 新增 `timing_dumped`

新增位置在 `Req.__init__` 的调度实验字段附近：

```python
self.timing_dumped = False
```

### 作用

这是每个 `Req` 实例自己的 timing dump 标记，用来表示这个请求对象是否已经把 timing 记录写入文件。

没有这个标记时，如果同一个 finished request 多次进入完成处理逻辑，就可能重复写多条 JSONL 记录。对 benchmark 分析来说，这会污染统计结果。

### 举例

假设请求 `rid="req-001"` 已经生成完成：

```text
第一次进入 finished 分支：
  timing_dumped = False
  写入 request_timing.jsonl
  timing_dumped = True

第二次又进入 finished 分支：
  timing_dumped = True
  直接跳过，不再写入
```

这样可以保证同一个 `Req` 实例最多只写一条 timing 记录。

## 2. `finalize_scheduler_timing` 变成幂等

新增代码：

```python
if self.release_time > 0.0:
    # overlap / 延迟输出路径下，同一个 finished request 可能再次走到
    # 收口逻辑。release_time / waiting_time 只应记录第一次完成时刻。
    return
```

### 作用

`finalize_scheduler_timing()` 用来在请求完成时计算最终时间：

```python
total_time = release_time - scheduler_enqueue_time
waiting_time = total_time - actual_execution_time - kv_transfer_time
```

新增的 guard 表示：如果 `release_time` 已经设置过，就说明这个请求已经完成收口过，后续重复调用直接返回。

### 为什么需要

SGLang 的 scheduler 在 overlap、延迟输出、batch 过滤等路径下，可能让同一个已完成请求再次走到 finished 分支。如果第二次调用又重新设置 `release_time`，会把请求完成时间往后推，从而把 `waiting_time` 算大。

### 举例

假设一个请求真实情况如下：

```text
scheduler_enqueue_time = 100.0
真实完成时间 release_time = 105.0
prefill_execution_time = 0.5
decode_execution_time = 1.0
kv_transfer_time = 0.2
```

第一次 finalize：

```text
actual_execution_time = 0.5 + 1.0 = 1.5
total_time = 105.0 - 100.0 = 5.0
waiting_time = 5.0 - 1.5 - 0.2 = 3.3
```

如果没有这次新增的 guard，第二次在 `106.0` 又 finalize：

```text
release_time 被错误覆盖成 106.0
total_time = 106.0 - 100.0 = 6.0
waiting_time = 6.0 - 1.5 - 0.2 = 4.3
```

这样等待时间会虚高 `1.0s`。新增 guard 后，第二次 finalize 会直接返回，保留第一次的真实完成时间。

## 3. `scheduler.py`

文件：

```text
python/sglang/srt/managers/scheduler.py
```

### 3.1 新增 `json` 依赖

新增代码：

```python
import json
```

### 作用

用于把 request timing record 序列化成 JSON 字符串，并以 JSONL 格式写入文件。

JSONL 是一行一个 JSON 对象，适合 benchmark 后用 Python、pandas、jq 或脚本逐行读取分析。

## 4. Scheduler 级别新增 rid 去重集合

新增代码：

```python
self._dumped_request_timing_rids: set[str] = set()
```

### 作用

这是 scheduler 级别的 request id 去重集合。它和 `req.timing_dumped` 是两层保险：

```text
req.timing_dumped:
  防止同一个 Req 实例重复 dump

self._dumped_request_timing_rids:
  防止同一个 rid 对应的不同 Req 实例重复 dump
```

### 为什么只靠 `req.timing_dumped` 不够

注释里写得很明确：在一些路径，尤其是 overlap 路径中，同一个 `rid` 可能以不同 `Req` 实例再次走到 finished 分支。

### 举例

假设同一个请求 `rid="abc"` 出现了两个 Python 对象：

```text
Req 对象 A:
  rid = "abc"
  timing_dumped = True

Req 对象 B:
  rid = "abc"
  timing_dumped = False
```

如果只检查 `req.timing_dumped`，对象 B 会再次写一条记录。

新增 scheduler 级集合后：

```text
第一次写入后：
  _dumped_request_timing_rids = {"abc"}

对象 B 再次尝试写入：
  "abc" in _dumped_request_timing_rids
  直接跳过
```

这能保证同一个 request id 在一个 scheduler 生命周期中最多写一次 timing 记录。

## 5. 新增 `_dump_request_timing_record`

新增函数：

```python
def _dump_request_timing_record(self, req: Req):
    """把 request 级执行画像额外落到独立文件，便于 benchmark 后单独分析。"""
    if req.timing_dumped or req.rid in self._dumped_request_timing_rids:
        return
    dump_path = os.environ.get(
        "SGLANG_REQUEST_TIMING_DUMP_FILE",
        os.path.expanduser("~/sglang/request_timing.jsonl"),
    )
    record = {
        "rid": req.rid,
        "scheduler_enqueue_time": req.scheduler_enqueue_time,
        "release_time": req.release_time,
        "prefill_execution_time": req.prefill_execution_time,
        "decode_execution_time": req.decode_execution_time,
        "actual_execution_time": req.actual_execution_time,
        "waiting_time": req.waiting_time,
        "kv_transfer_time": req.kv_transfer_time,
        "mlfq_level": req.mlfq_level,
        "mlfq_tokens_in_level": req.mlfq_tokens_in_level,
        "prompt_len": len(req.origin_input_ids),
        "output_len": len(req.output_ids),
        "finished_reason": (
            req.finished_reason.to_json() if req.finished_reason else None
        ),
    }
    try:
        with open(dump_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        req.timing_dumped = True
        self._dumped_request_timing_rids.add(req.rid)
    except Exception:
        logger.exception("Failed to dump request timing record for rid=%s", req.rid)
```

### 作用

这个函数把单个 request 的调度画像写入一个独立文件。默认路径是：

```text
~/sglang/request_timing.jsonl
```

也可以通过环境变量指定：

```bash
export SGLANG_REQUEST_TIMING_DUMP_FILE=/path/to/request_timing.jsonl
```

### 写入格式

每个完成请求写一行 JSON：

```json
{"rid":"...","scheduler_enqueue_time":...,"release_time":...}
```

多请求时文件类似：

```jsonl
{"rid":"req-1","scheduler_enqueue_time":100.0,"release_time":101.2,"prefill_execution_time":0.08,"decode_execution_time":0.20,"actual_execution_time":0.28,"waiting_time":0.92,"kv_transfer_time":0.0,"mlfq_level":1,"mlfq_tokens_in_level":0,"prompt_len":64,"output_len":32,"finished_reason":{"type":"length"}}
{"rid":"req-2","scheduler_enqueue_time":100.1,"release_time":106.5,"prefill_execution_time":0.50,"decode_execution_time":1.10,"actual_execution_time":1.60,"waiting_time":4.50,"kv_transfer_time":0.30,"mlfq_level":2,"mlfq_tokens_in_level":1,"prompt_len":2048,"output_len":256,"finished_reason":{"type":"length"}}
```

## 6. 输出字段逐项解释和例子

### `rid`

请求 ID。

例子：

```json
{"rid": "chatcmpl-123"}
```

用途：把 benchmark 客户端记录、server 日志、timing dump 对齐到同一个请求。

### `scheduler_enqueue_time`

请求首次进入 scheduler 队列的时间，使用 `time.monotonic()`。

例子：

```text
scheduler_enqueue_time = 100.0
```

它不是墙钟时间，而是单调时钟，适合做耗时差值。

### `release_time`

请求完成并释放时的时间。

例子：

```text
release_time = 105.0
```

端到端 server 侧总耗时可以近似看作：

```text
release_time - scheduler_enqueue_time
```

### `prefill_execution_time`

请求在 prefill 阶段真正占用 GPU forward 的累计时间。

例子：

```text
长 prompt 被 chunked prefill 切成 3 个 chunk：
chunk 1 GPU 时间 0.04s
chunk 2 GPU 时间 0.05s
chunk 3 GPU 时间 0.06s

prefill_execution_time = 0.15s
```

用途：判断长 prompt 的 prefill 计算成本。

### `decode_execution_time`

请求在 decode 阶段真正占用 GPU forward 的累计时间。

例子：

```text
生成 100 个 token
平均每个 decode step 对该请求摊到 0.004s

decode_execution_time = 0.4s
```

用途：判断输出长度导致的 decode 计算成本。

### `actual_execution_time`

实际计算时间：

```text
actual_execution_time = prefill_execution_time + decode_execution_time
```

例子：

```text
prefill_execution_time = 0.15s
decode_execution_time = 0.40s
actual_execution_time = 0.55s
```

### `kv_transfer_time`

PD 分离场景下 KV cache 传输耗时。

例子：

```text
P 端完成 prefill 后，需要把 KV 传到 D 端
传输耗时 0.30s

kv_transfer_time = 0.30s
```

如果不是 PD 分离，或者没有记录 KV 传输，这个值通常是 `0.0`。

### `waiting_time`

等待时间，计算方式是：

```text
waiting_time =
  release_time
  - scheduler_enqueue_time
  - actual_execution_time
  - kv_transfer_time
```

它表示除 GPU 计算和 KV 传输以外，request 花在排队、资源等待、调度间隔、被低优先级压后等地方的时间。

例子：

```text
scheduler_enqueue_time = 100.0
release_time = 106.0
actual_execution_time = 1.2
kv_transfer_time = 0.3

total_time = 6.0
waiting_time = 6.0 - 1.2 - 0.3 = 4.5
```

这个例子说明请求慢主要不是算得慢，而是等得久。

### `mlfq_level`

请求完成时所在的 MLFQ 队列层级。

```text
level 越小，优先级越高
0 = 最高优先级
1 = 中间优先级
2 = 最低优先级
```

例子：

```json
{"mlfq_level": 2}
```

表示请求最终处在较低优先级队列，通常意味着它已经消耗过一定 token quantum，或者本身是长 prompt/长任务。

### `mlfq_tokens_in_level`

请求在当前 MLFQ level 已经消耗的 token 计数。

例子：

```text
MLFQ quanta = (1, 2, 4)

当前 level = 1
level 1 的 quantum = 2
mlfq_tokens_in_level = 1
```

表示它在 level 1 已经消耗了 1 个 token，再消耗 1 个 token 就会降到 level 2。

### `prompt_len`

输入 prompt 的 token 数：

```python
len(req.origin_input_ids)
```

例子：

```json
{"prompt_len": 2048}
```

用途：判断 prefill 压力。

### `output_len`

当前已经生成的 token 数：

```python
len(req.output_ids)
```

例子：

```json
{"output_len": 256}
```

用途：判断 decode 压力。

### `finished_reason`

请求完成原因。

例子：

```json
{"finished_reason": {"type": "length"}}
```

或者：

```json
{"finished_reason": {"type": "stop", "matched": "</s>"}}
```

用途：区分正常完成、达到长度上限、命中 stop、abort 等情况。

## 7. `_finalize_finished_request_timing` 中新增 dump 调用

新增代码：

```python
def _finalize_finished_request_timing(self, batch: ScheduleBatch):
    now = time.monotonic()
    for req in batch.reqs:
        if req.finished():
            req.finalize_scheduler_timing(now)
            self._dump_request_timing_record(req)
```

### 作用

以前这个函数只负责在请求完成时调用：

```python
req.finalize_scheduler_timing(now)
```

现在多做一步：

```python
self._dump_request_timing_record(req)
```

也就是只要请求 finished，就把该请求的 timing record 写入 JSONL。

### 举例

一个 batch 里有 4 个请求：

```text
req-1 finished
req-2 running
req-3 finished
req-4 running
```

这次 `_finalize_finished_request_timing` 会：

```text
dump req-1
skip req-2
dump req-3
skip req-4
```

最终 JSONL 新增两行。

## 8. 对 MLFQ 实验有什么用

这批新增代码最直接的价值是：可以把每个请求完成时的 MLFQ 状态和耗时拆开分析。

### 例子 1：判断短请求是否被长请求拖慢

假设 dump 里有：

```json
{"rid":"short-1","prompt_len":32,"output_len":16,"mlfq_level":0,"actual_execution_time":0.05,"waiting_time":0.02}
{"rid":"long-1","prompt_len":4096,"output_len":512,"mlfq_level":2,"actual_execution_time":2.50,"waiting_time":0.80}
```

可以看出：

```text
short-1:
  prompt/output 都短
  level=0
  waiting_time 很低
  说明 MLFQ 确实优先照顾了短请求

long-1:
  prompt/output 都长
  level=2
  actual_execution_time 高
  waiting_time 也更高
  符合长任务被逐步降级的预期
```

### 例子 2：判断慢是因为 GPU 计算还是排队

```json
{"rid":"req-a","actual_execution_time":0.30,"kv_transfer_time":0.00,"waiting_time":4.70}
```

解释：

```text
总耗时里绝大部分是 waiting_time
瓶颈更可能是排队、batch admission、KV cache 不足、调度策略导致等待
不是 GPU forward 本身慢
```

另一个请求：

```json
{"rid":"req-b","actual_execution_time":5.20,"kv_transfer_time":0.00,"waiting_time":0.30}
```

解释：

```text
主要时间花在 GPU 实际计算
可能是 prompt 太长、output 太长、模型太大或 batch 太重
```

### 例子 3：分析 PD 分离里的 KV 传输

```json
{"rid":"pd-1","actual_execution_time":0.90,"kv_transfer_time":1.20,"waiting_time":0.40}
```

解释：

```text
KV 传输时间比实际计算时间还高
PD 分离下的网络/传输层可能是瓶颈
```

如果大量请求都有高 `kv_transfer_time`，就应该检查：

```text
P/D 节点网络
KV transfer backend
batch size
chunked prefill
跨节点带宽
```

### 例子 4：发现重复完成路径问题

这次新增了两层去重：

```text
req.timing_dumped
_dumped_request_timing_rids
```

如果没有这些保护，JSONL 可能出现：

```jsonl
{"rid":"same-req","release_time":105.0,"waiting_time":3.3}
{"rid":"same-req","release_time":106.0,"waiting_time":4.3}
```

这会让平均 latency、P95 latency、平均 waiting time 都被污染。

新增代码后，同一个 `rid` 只保留第一次完成时的记录。

## 9. 使用方式示例

启动 server 前设置输出路径：

```bash
export SGLANG_REQUEST_TIMING_DUMP_FILE=/tmp/request_timing.jsonl
```

跑 benchmark 后，查看前几行：

```bash
head -n 3 /tmp/request_timing.jsonl
```

用 Python 简单分析：

```python
import json

rows = []
with open("/tmp/request_timing.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

avg_wait = sum(r["waiting_time"] for r in rows) / len(rows)
avg_exec = sum(r["actual_execution_time"] for r in rows) / len(rows)
avg_kv = sum(r["kv_transfer_time"] for r in rows) / len(rows)

print("avg waiting:", avg_wait)
print("avg execution:", avg_exec)
print("avg kv transfer:", avg_kv)
```

按 MLFQ level 分组：

```python
from collections import defaultdict

by_level = defaultdict(list)
for r in rows:
    by_level[r["mlfq_level"]].append(r)

for level, group in sorted(by_level.items()):
    avg_wait = sum(r["waiting_time"] for r in group) / len(group)
    avg_prompt = sum(r["prompt_len"] for r in group) / len(group)
    avg_output = sum(r["output_len"] for r in group) / len(group)
    print(level, len(group), avg_wait, avg_prompt, avg_output)
```

可能输出：

```text
0 120 0.08 48.2 18.5
1 300 0.35 256.7 96.3
2 80 1.90 4096.0 512.0
```

解释：

```text
level 0 请求更短，等待时间更低
level 2 请求更长，等待时间更高
```

这可以用来验证 MLFQ 是否达到了“短请求优先、长请求逐步下沉”的实验目标。

## 10. 总结

这次 `main...46fc0ba67` compare 新增的 44 行代码，核心不是改调度策略，而是增强可观测性：

```text
1. Req 增加 timing_dumped，防止同一对象重复写 timing。
2. finalize_scheduler_timing 增加 release_time guard，防止重复 finalize 覆盖完成时间。
3. Scheduler 增加 _dumped_request_timing_rids，防止同一 rid 的不同 Req 实例重复写 timing。
4. 新增 _dump_request_timing_record，把 request 级画像写入 JSONL。
5. 请求完成时自动调用 dump，benchmark 后可以离线分析 waiting、execution、KV transfer 和 MLFQ 状态。
```

它对实验的价值是把每个请求拆成：

```text
总耗时 = 等待时间 + GPU 实际执行时间 + KV 传输时间
```

并额外记录：

```text
MLFQ level
MLFQ 当前 level token 消耗
prompt 长度
output 长度
完成原因
```

这样就可以回答这些问题：

```text
短请求是否真的被优先服务？
长请求是否被降级？
慢请求到底是在排队，还是在 GPU 上执行，还是卡在 PD KV 传输？
overlap / 延迟输出路径有没有导致重复统计？
```
