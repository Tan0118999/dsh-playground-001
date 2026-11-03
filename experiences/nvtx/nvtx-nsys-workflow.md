---
name: nvtx-nsys-workflow
title: 用 Nsight Systems 采集和读 NVTX：trace、capture-range、report、SQLite
description: 打了 NVTX 但报告里没有、报告太大、只想采某一段、想用命令行统计或脚本分析时读这条。给出 --trace=nvtx、--capture-range=nvtx 的 registered string 要求、domain 过滤、nvtx_sum 读法、--filter-nvtx 和 SQLite 查询。
category: performance
tags: [nvtx, nsys, nsight-systems, capture-range, sqlite]
symptoms:
  - nsys 报告里没有 NVTX 行
  - 报告太大，想只采一个阶段
  - capture-range 不触发，生成不了报告
  - 想按某个 range 过滤 kernel 统计
  - 想用脚本读取 NVTX 数据
stacks: [nvtx, nsys, cuda]
updated: 2026-11-03
---

# 用 Nsight Systems 采集和读 NVTX：trace、capture-range、report、SQLite

> 一句话经验：**`--trace=nvtx` 决定采不采 NVTX，`--capture-range=nvtx` 决定采哪一段；后者的消息默认只匹配 registered string，自定义 domain 还要写 `@domain`。**

## 适用场景

- 代码里已经有 NVTX，想在 Nsight Systems 里看时间线或做统计；
- 程序很大，只想采集某个 epoch、某个 step 或某个请求；
- 想用 `nsys stats` 或 SQLite 做批量分析，而不是手点 GUI；
- 需要按 domain 过滤掉框架/通信库的标注。

不解决：NVTX API 怎么用（见 [第一次用 NVTX](nvtx-first-range.md) 和 [进阶 API](nvtx-advanced-api.md)）；也不解决单 kernel 的微架构瓶颈（用 Nsight Compute）。

## 症状与信号

- 时间线只有 CUDA HW 和 API 行，没有 NVTX 行；
- 有 NVTX 行，但 `nvtx_sum` 是空的；
- 报告几个 GB，打开很慢，实际只关心某 200 ms；
- 执行 `--capture-range=nvtx` 后输出 “No reports were generated”；
- `nsys stats` 报 “Existing SQLite export found ... File is older than input file”；
- 想知道某个 range 里到底跑了哪些 kernel，但不想重新采集。

## 快速判断

1. **报告里有没有 NVTX 事件？** 先跑 `nsys stats --report nvtx_sum report.nsys-rep`；为空时看 [看不到 NVTX 的排查](nvtx-not-showing.md)。
2. **是不是采集时没开 `--trace=nvtx`？** 显式写出来，不要依赖默认值。
3. **是不是只想采一段？** 用 `--capture-range=nvtx`，并确认消息是 registered string、domain 写对。
4. **是不是报告太大？** 先 capture-range/domain 过滤，再考虑降低采样和 trace 集合。
5. **是不是想换一种读法？** 端到端看时间线，分组看 `nvtx_sum`，只看某个 range 用 `--filter-nvtx`，自动化用 SQLite。

## 采集

最常用的命令：

```bash
# NVTX + OS runtime；如果还要看 kernel，把 cuda 加进去
nsys profile --trace=cuda,nvtx,osrt --stats=true -o report ./app
```

- `--stats=true`：采集结束后直接跑一批统计报告（包括 `nvtx_sum`）；
- `-o report`：生成 `report.nsys-rep`（以及可能的 `.sqlite`）；
- 打开 GUI：`nsys-ui report.nsys-rep`；
- 不加 `cuda` 时，如果程序没有 CUDA 调用，报告里仍然可以有 NVTX 和 OS runtime 事件。

> 不要无脑 `--trace=all`。trace 集合越大，报告越大，运行时扰动也越大；先只开你需要的。

## 只采一段：capture-range

NVTX capture-range 适合“程序跑很久，只关心某个阶段”的场景：

```bash
nsys profile \
  --trace=nvtx \
  --capture-range=nvtx \
  --nvtx-capture=TARGET@roi \
  --capture-range-end=stop \
  -o roi-report ./app
```

关键点：

- **默认只匹配 registered string**。用 `nvtxRangePushA("TARGET")` 这种普通字符串时，nsys 可能完全不触发；需要改用 `nvtxDomainRegisterStringA` / `nvtx3::registered_string` / Python `nvtx.annotate`（后者会注册字符串）。
- **domain 要写对**：`TARGET@roi` 表示 `roi` domain 里的 `TARGET`；不写 `@domain` 只匹配默认 domain；`TARGET@*` 可以匹配任意 domain。
- **只触发第一个 capture range**：CLI 默认只认第一个匹配的 range；需要重复采集时看 `--capture-range-end=repeat:N`。
- `--capture-range-end=stop` 只停止采集，应用继续跑；默认的 `stop-shutdown` 会在结束后关闭 session。想保留应用后续行为时用 `stop`。

如果必须用普通字符串，可以显式关闭“只匹配注册字符串”的优化（会增加字符串匹配开销）：

```bash
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 nsys profile \
  --trace=nvtx --capture-range=nvtx --nvtx-capture=TARGET@roi \
  --capture-range-end=stop -o roi-report ./app
```

实测对比（nsys 2025.6.3）：普通字符串 + 自定义 domain 不触发；改成 registered string 后正常生成报告；设置 `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` 后普通字符串也能触发。

## 读报告：先看哪些表

```bash
nsys stats \
  --report nvtx_sum,nvtx_pushpop_sum,nvtx_startend_sum,nvtx_gpu_proj_sum \
  report.nsys-rep
```

| report | 回答什么 |
| --- | --- |
| `nvtx_sum` | 所有 range 的聚合：次数、总时长、平均/中位/最小/最大、PushPop/StartEnd |
| `nvtx_pushpop_sum` | 只看 push/pop range |
| `nvtx_startend_sum` | 只看 start/end range |
| `nvtx_gpu_proj_sum` | 每个 NVTX range 投影到 GPU 上的时间、GPU op 数 |
| `nvtx_kern_sum` | 按 NVTX range 归组的 kernel 统计 |
| `cuda_gpu_kern_sum` | 按 kernel 聚合的 GPU 时间（可用 `--filter-nvtx` 限定 range） |

真实 `nvtx_sum` 片段（本仓探测程序，默认 domain + 自定义 domain 混用）：

```text
Time (%)  Total Time (ns)  Instances  ...  Style     Range
--------  ---------------  ---------  ...  --------  -----------------------------------
   29.9        220931696          1  ...  PushPop   :outer
   24.4        180685950          1  ...  StartEnd  :cross-thread
   17.2        127228259          5  ...  PushPop   com.example.nvtxlab:registered-loop
   16.3        120489035          1  ...  PushPop   :inner
```

怎么读：

- **`Time%` 的分母是 range 总时间，不是 wall time**；嵌套/重叠的 range 会重复计入，所以不要把 `outer` 和 `inner` 的占比相加当总耗时；
- `Instances` 包含预热/重复调用；想看稳态就 capture-range 或用 `--filter-nvtx`/时间过滤；
- `Avg` 与 `Med` 差很多时说明有抖动，先看分布再看单次；
- **mark 不在 `nvtx_sum` 里**：mark 没有持续时间，去时间线或 SQLite 看。

## 只分析某个 range：`--filter-nvtx`

采集一次后，可以用 `--filter-nvtx` 把统计限制在某个 range 的时间窗口内，不用重新跑程序：

```bash
# 默认 domain 的 outer；只取第一个实例
nsys stats --report cuda_gpu_kern_sum --filter-nvtx outer report.nsys-rep

# 自定义 domain 必须带 @domain
nsys stats --report nvtx_sum \
  --filter-nvtx 'registered-loop@com.example.nvtxlab' report.nsys-rep
```

- 过滤的是“第一个匹配实例”的时间窗口；要选第 N 个实例，用 `range@domain/index`（索引从 0 开始）；
- 只保留与该时间窗口有重叠的事件，所以 range 边界处的 kernel 也可能被算进来；
- 这是“先采全量、后切 ROI”的快捷方式；如果采集本身开销就很大，还是用 capture-range。

## 按 domain 过滤

```bash
# 只保留自定义 domain
nsys profile --trace=cuda,nvtx \
  --nvtx-domain-include=com.example.data \
  -o only-data ./app

# 排除默认 domain，保留其他 domain
nsys profile --trace=cuda,nvtx \
  --nvtx-domain-exclude=default \
  -o no-default ./app
```

`--nvtx-domain-include` 与 `--nvtx-domain-exclude` 只能二选一；`default` 表示没有 domain 的默认事件。实测两个选项都能把 `nvtx_sum` 限制到目标 domain。

## 自动化：SQLite

```bash
nsys export --type sqlite -o report.sqlite report.nsys-rep
```

`NVTX_EVENTS` 的关键列：

| 列 | 含义 |
| --- | --- |
| `start` / `end` | 纳秒时间戳；mark 的 `end` 为 NULL |
| `eventType` | 34=mark，59=push/pop range，60=start/end range，33=category 命名，75/76=domain 创建/销毁 |
| `text` | 普通字符串消息；**registered string 这里是 NULL** |
| `textId` | registered string 的 ID，要 join `StringIds` |
| `domainId` | 0=默认 domain，自定义 domain 从 1 开始 |
| `category` | 整数 category |
| `color` | ARGB 编码的颜色 |
| `globalTid` / `endGlobalTid` | 开始/结束线程；跨线程 range 两者不同 |
| `int64Value` / `doubleValue` / ... | payload 的各种类型 |

查询示例（Python 标准库，不需要装驱动）：

```python
import sqlite3

con = sqlite3.connect("report.sqlite")
for row in con.execute("""
    SELECT n.start, n.end, s.value AS range_name, n.domainId, n.category
    FROM NVTX_EVENTS AS n
    LEFT JOIN StringIds AS s ON s.id = n.textId
    WHERE n.eventType IN (59, 60)
    ORDER BY (n.end - n.start) DESC
    LIMIT 10
"""):
    print(row)
```

> 用 `text` 列直接筛 range 名会漏掉 registered string（例如 Python `nvtx.annotate` 和 C++ 的 `registered_string`）；要同时看 `text` 和 `textId`。

如果 `nsys stats` 提示已有 SQLite 比 `.nsys-rep` 旧，加 `--force-export=true` 重新导出，避免读到旧数据。

## 在 GUI 里读

- NVTX 行在对应 CPU 线程下面；嵌套 range 会分层显示；
- GPU 行会把 CPU range 投影到 GPU 时间轴，方便看“这段 range 里跑了哪些 kernel”；
- mark 是时间点，不是有宽度的条；
- 右键/过滤器可以按 domain 筛选；Events View 能看精确 start/end/duration；
- 先 Fit to screen 看整体，再放大看 range 与 kernel 的边界；不要只看颜色，颜色是工具分配的。

## 修复与规避

- 采集前先决定“我要回答什么问题”，只开需要的 trace；
- 大程序用 capture-range，且用 registered string；
- 只关心某个阶段时，优先 `--filter-nvtx` 复用已有报告，避免重复采集；
- 报告里混入多个进程时，先按 PID 确认目标进程；子进程/容器场景要确认它们确实被 trace；
- 用 SQLite 做自动化时，把“registered string 的 `text` 为 NULL”当成一等公民处理；
- 不要用 `--trace=all` 代替思考；也不要在采集期间跑其他 profiler（NVTX 只支持一个工具接入）。

## 验证

- `nvtx_sum` 的 range 数量与代码中的调用次数一致；
- capture-range 报告的第一个事件就在目标 range 开始附近，且总时长与目标阶段同量级；
- `--filter-nvtx` 前后的 `cuda_gpu_kern_sum` 只差目标 range 内的 kernel；
- SQLite 查询结果与 `nvtx_sum` 的实例数、总时长能对上；
- 日志里出现 “NVTX injection initialized successfully”，且没有 “wrong thread ID” 之类的告警。

## 常见坑

1. **`--trace=nvtx` 没开**：报告里没有 NVTX；
2. **capture-range 用了普通字符串**：默认不触发；换 registered string 或 `NSYS_NVTX_PROFILER_REGISTER_ONLY=0`；
3. **自定义 domain 忘了 `@domain`**：`TARGET` 只匹配默认 domain；
4. **把 `Time%` 当 wall time**：嵌套 range 会重复计入；
5. **用 `nsys stats` 读旧 `.sqlite`**：加 `--force-export=true`；
6. **以为 mark 会进 `nvtx_sum`**：mark 没有 duration，去时间线/SQLite；
7. **用 `--capture-range-end=stop` 后奇怪为什么 session 没关**：`stop` 只停采集，应用继续跑；要关 session 用默认的 `stop-shutdown`；
8. **多进程只看到主进程**：先确认 PID，再检查 fork/exec、容器和 nsys 的多进程选项。

## 参考

- Nsight Systems User Guide（NVTX Trace、capture range、domain 过滤）：<https://docs.nvidia.com/nsight-systems/UserGuide/index.html#nvtx-trace>
- `nsys stats --help` / `nsys profile --help`：本机 nsys 2025.6.3 的 report、capture、filter 选项以这里为准
- NVTX registered string 说明：<https://github.com/NVIDIA/NVTX>
- 本仓 GPU demo：[`toolbox/gpu-profiling-demo`](../../toolbox/gpu-profiling-demo/README.md)
- 本仓时间线读法：[第一次读 Nsight Systems 时间线](../gpu-profiling/nsight-systems-timeline-reading.md)
