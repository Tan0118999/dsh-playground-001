---
name: nvtx-ecosystem-and-automation
title: NVTX 的消费端：ncu 过滤、SQLite 与自动化
description: 已有 NVTX、想只 profile 某个 range 里的 kernel、想用 ncu 缩小范围、想把 NVTX/时间线数据接入脚本或 CI、或分不清 NVTX 与 PyTorch Profiler 时读这条。覆盖 ncu 的 NVTX 过滤、nsys SQLite、CSV/JSON 和 CUDA Graphs 投影。
category: performance
tags: [nvtx, ncu, nsys, sqlite, automation, pytorch]
symptoms:
  - 只想 profile 某个阶段的 kernel
  - ncu 一次采太多 kernel，想知道怎么按 NVTX 过滤
  - 想把 NVTX 统计接入脚本或 CI
  - SQLite 里 registered string 的名字是 NULL
  - CUDA Graphs 和 NVTX 对不上
stacks: [nvtx, nsys, ncu, cuda, pytorch, python]
updated: 2026-11-03
---

# NVTX 的消费端：ncu 过滤、SQLite 与自动化

> 一句话经验：**NVTX 只是标注；端到端看 nsys 时间线，单 kernel 用 ncu 的 NVTX 过滤，脚本化分析用 nsys SQLite/CSV，算子级用 PyTorch Profiler——先确定你要回答的问题，再选消费端。**

## 适用场景

- 已经给代码加了 NVTX，不想只会在 GUI 里手点；
- 只想用 ncu 采某个 range 里的 kernel，避免 `--set full` 扫全场；
- 想把 NVTX/时间线数据接入 Python 脚本、回归对比或 CI；
- 需要从 SQLite 里正确读取 registered string、domain、payload；
- 用 CUDA Graphs，发现 NVTX range 与 graph 节点对不上。

不解决：NVTX 的基本写法（见 [第一次用 NVTX](nvtx-first-range.md) 和 [进阶 API](nvtx-advanced-api.md)）；也不解决 nsys 的采集参数（见 [nsys 工作流](nvtx-nsys-workflow.md)）。

## 症状与信号

- ncu 对一个训练 step 采了太多 kernel，报告巨大，真正关心的只有 attention 那一段；
- `ncu --nvtx-include` 报错或一个 kernel 都没采到；
- 用 Python 读 SQLite，按 `text` 筛 range 名，registered string 全部漏掉；
- 想比较两次运行的阶段耗时，但每次都在 GUI 里手抄；
- CUDA Graph replay 之后，时间线里 NVTX range 和节点对不上；
- 分不清 PyTorch Profiler 的 label 和 nsys 的 NVTX。

## 快速判断

| 你要回答的问题 | 先用什么 |
| --- | --- |
| 端到端时间花在哪些阶段/拷贝/同步上 | nsys 时间线 + `nvtx_sum` |
| 某个 range 里的 kernel 为什么慢 | ncu + `--nvtx-include` |
| 某个 PyTorch 算子的 CPU/CUDA 时间 | PyTorch Profiler（`record_function`/Kineto） |
| 批量对比/回归/CI | nsys SQLite 或 CSV/JSON |
| 图形 API 的 frame/range | Nsight Graphics（不是本文重点） |

## ncu：只 profile 某个 NVTX range

ncu 的 NVTX 过滤需要先打开 `--nvtx`：

```bash
# start/end range：Domain A 里的 Range A
ncu --nvtx --nvtx-include "Domain A@Range A" \
  --set basic -o attention-report ./app

# 默认 domain 的 push/pop range；末尾的 / 表示 push/pop
ncu --nvtx --nvtx-include "attention/" \
  --set basic -o attention-report ./app
```

要点：

- `--nvtx` 是总开关；只写 `--nvtx-include` 不写 `--nvtx` 通常不会按预期过滤；
- start/end range 用 `domain@range`；push/pop range 的语法带 `/`、`[`、`]` 等栈位置标记，完整语法见 [Nsight Compute CLI 的 NVTX Filtering](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvtx-filtering)；
- 跨线程 push/pop 可以用 `--nvtx-push-pop-scope=process` 改变作用域（默认 `thread`）；
- 先确认你要的 kernel 确实在 range 内；range 外 launch 的 kernel 不会因为“名字像”被选中；
- ncu 采计数器需要权限；没有权限时先解决 `ERR_NVGPUCTRPERM`，不要反复重试（见 [ncu 权限与容器](../gpu-profiling/gpu-profiling-permissions-and-containers.md)）。

> 本机 WSL2 没有 GPU 计数器权限，因此本文的 ncu 过滤只核对了 `ncu 2026.1.1 --help` 和官方文档，没有做计数器实测。

## nsys SQLite：把 NVTX 接入脚本

```bash
nsys export --type sqlite -o report.sqlite report.nsys-rep
```

常用表：

| 表 | 内容 |
| --- | --- |
| `NVTX_EVENTS` | 所有 mark/range/category/domain 事件 |
| `StringIds` | 字符串表；registered string 的名字在这里 |
| `ThreadNames` | 线程名（`nvtxNameOsThread` 的结果） |
| `CUPTI_ACTIVITY_KIND_KERNEL` | kernel 明细（表名随 nsys 版本可能变化） |

`NVTX_EVENTS` 的关键列见 [nsys 工作流](nvtx-nsys-workflow.md)；最容易踩的坑是 **registered string 的 `text` 为 NULL**。按名字筛 range 时要同时处理 `text` 和 `textId`：

```python
import sqlite3

con = sqlite3.connect("report.sqlite")

# 按 range 名聚合总时长；同时覆盖普通字符串和 registered string
rows = con.execute("""
    SELECT COALESCE(s.value, n.text) AS name,
           COUNT(*) AS instances,
           SUM(n.end - n.start) AS total_ns
    FROM NVTX_EVENTS AS n
    LEFT JOIN StringIds AS s ON s.id = n.textId
    WHERE n.eventType IN (59, 60) AND n.end IS NOT NULL
    GROUP BY name
    ORDER BY total_ns DESC
    LIMIT 20
""").fetchall()

for name, instances, total_ns in rows:
    print(f"{name}: {instances} 次, {total_ns / 1e6:.3f} ms")
```

查 mark：

```python
for row in con.execute("""
    SELECT n.start, COALESCE(s.value, n.text) AS name, n.globalTid
    FROM NVTX_EVENTS AS n
    LEFT JOIN StringIds AS s ON s.id = n.textId
    WHERE n.eventType = 34
    ORDER BY n.start
"""):
    print(row)
```

查跨线程 range：

```python
for row in con.execute("""
    SELECT n.rangeId, n.globalTid, n.endGlobalTid, n.start, n.end
    FROM NVTX_EVENTS AS n
    WHERE n.eventType = 60 AND n.globalTid != n.endGlobalTid
"""):
    print(row)
```

注意：

- 表名/列名会随 nsys 版本变化；先 `.schema NVTX_EVENTS` 或 `PRAGMA table_info` 核对；
- `start`/`end` 是纳秒时间戳；跨进程比较时要考虑时钟/进程起点；
- SQLite 可能比 `.nsys-rep` 旧；`nsys stats` 提示 stale 时加 `--force-export=true`。

## nsys CSV/JSON：给 CI 或对比脚本

`nsys stats` 支持 `--format csv`、`tsv`、`json`：

```bash
nsys stats --report nvtx_sum --format csv report.nsys-rep
nsys stats --report nvtx_sum --format json report.nsys-rep
```

适合做“两次运行的阶段耗时对比”。注意：

- 输出前面可能混有 `Generating SQLite file...` 等日志；脚本要从表头开始解析，或把 stderr/stdout 分开处理；
- `nvtx_sum` 的 `Time%` 不是 wall time，比较时优先用 `Total Time (ns)` 和 `Instances`；
- 采集本身会扰动程序；对比时固定 GPU、时钟、负载和采集参数。

## PyTorch Profiler：和 NVTX 各管一段

- `torch.profiler` 的 `record_function` 是 **PyTorch Profiler 的 label**，默认只出现在 Kineto/Chrome trace/算子表里，不是 nsys 的 NVTX；
- `torch.cuda.nvtx` / `nvtx.annotate` 是真正的 NVTX，出现在 nsys 的 NVTX 行；
- `torch.autograd.profiler.emit_nvtx` 或 nsys 的 `--pytorch=autograd-nvtx` 可以给 autograd 算子批量打 NVTX；
- 一个阶段可以两边都标，分别服务“算子表”和“端到端时间线”。

具体写法见 [Python 与 PyTorch](nvtx-python-pytorch.md)。

## CUDA Graphs、通信库与多进程

- CUDA Graphs 的 NVTX 投影由 nsys 的 `--cuda-graph-trace` 控制，`nvtx-live` 是默认值，`nvtx-precapture` 用于采集前已经构建好的 graph；需要 node 粒度、CUDA trace 和 NVTX trace。该行为随 nsys 版本变化，先用 `nsys profile --help` 核对；
- NCCL、MPI、NVSHMEM 等库可能自带 NVTX domain；先看 `nvtx_sum` 里的 `domain:range`，再用 `--nvtx-domain-include/exclude` 过滤；
- 多进程/多 worker 时，SQLite 里会有多个 PID 的线程；按 `globalTid`/PID 分组，不要假设只有一个进程；
- 不要同时接多个 NVTX 工具；也不要期望 NVTX 自己完成跨进程 correlation，需要自己把 step/batch 写进名字或 payload。

## 修复与规避

- 先确定消费端：端到端用 nsys，单 kernel 用 ncu，算子表用 torch.profiler；
- ncu 过滤前先确认 range 真的包住了目标 kernel；
- 脚本读取 SQLite 时统一通过 `COALESCE(StringIds.value, NVTX_EVENTS.text)` 取名；
- CI 里固定采集参数和基线，记录 nsys/ncu/CUDA 版本；
- CUDA Graphs 场景先确认投影模式，再解释时间线；
- 不要把 `record_function` 的 label 当作 NVTX 证据。

## 验证

- ncu 报告里的 kernel 全部位于目标 range 的时间窗口内；
- SQLite 聚合出的 range 名、实例数、总时长与 `nvtx_sum` 对得上；
- CSV/JSON 解析在两次运行间可重复，字段顺序不依赖人工查看；
- PyTorch Profiler 的 label 与 nsys 的 NVTX 分别出现，语义清楚；
- CUDA Graphs 场景下 NVTX range 与 graph 节点的对应关系符合预期。

## 常见坑

1. **ncu 只写 `--nvtx-include` 不写 `--nvtx`**：过滤不生效；
2. **push/pop 和 start/end 的过滤语法混用**：一个用 `domain@range`，一个带 `/`/`[`/`]`；
3. **SQLite 只看 `text`**：registered string 全部丢失；
4. **把 `nvtx_sum` 的 `Time%` 当 wall time**：嵌套 range 重复计入；
5. **CI 里没固定版本/时钟**：两次结果不可比；
6. **CUDA Graphs 采集前构建**：默认 `nvtx-live` 可能看不到，需确认 `nvtx-precapture`；
7. **把 `record_function` 当 NVTX**：两条链路，不要互相替代；
8. **ncu 权限不足时反复重试**：先解决权限/容器限制。

## 参考

- Nsight Compute CLI 的 NVTX Filtering：<https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvtx-filtering>
- Nsight Systems User Guide（NVTX、capture、CUDA Graphs、SQLite）：<https://docs.nvidia.com/nsight-systems/UserGuide/index.html#nvtx-trace>
- NVTX 官方总览：<https://github.com/NVIDIA/NVTX>
- NVTX Python 文档：<https://nvidia.github.io/NVTX/python/>
- PyTorch Profiler：<https://docs.pytorch.org/docs/stable/profiler.html>
- 本仓 [nsys 工作流](nvtx-nsys-workflow.md) 和 [Python 与 PyTorch](nvtx-python-pytorch.md)
