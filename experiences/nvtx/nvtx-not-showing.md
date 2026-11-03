---
name: nvtx-not-showing
title: NVTX 打了但 Nsight Systems 里看不到：从 trace 到 range 的排查顺序
description: 代码里明明有 nvtxRangePush、报告里却没有 NVTX 行、某个 range 缺失、名字不对、只有部分线程、capture-range 不触发时读这条。按 trace 开关、注入、range 生命周期、domain、进程和 registered string 逐层排查。
category: performance
tags: [nvtx, nsys, troubleshooting, capture-range, domains]
symptoms:
  - Nsight Systems 报告里没有 NVTX 行
  - nvtx_sum 输出为空
  - 只有部分 range 或部分线程出现
  - range 名字为空或不完整
  - capture-range 不触发，提示 No reports were generated
stacks: [nvtx, nsys, cuda, python]
updated: 2026-11-03
---

# NVTX 打了但 Nsight Systems 里看不到：从 trace 到 range 的排查顺序

> 一句话经验：**先证明“采到了 NVTX”，再证明“range 真的执行并闭合”，最后才查 domain、进程和 capture-range 的匹配条件；不要一上来改 API。**

## 适用场景

- 代码里已经有 `nvtxRangePushA` / `nvtx.annotate` / `torch.cuda.nvtx`，但报告里没有；
- 有一半 range 出现、另一半消失；
- 多线程/多进程时只看到主线程；
- 用 `--capture-range=nvtx` 时 nsys 输出 “No reports were generated”；
- `nvtx_sum` 为空，但你以为只是“名字没显示”。

不解决：NVTX 的语义和写法（见 [range 语义](nvtx-range-model.md)）；也不解决具体的 nsys 统计读法（见 [nsys 工作流](nvtx-nsys-workflow.md)）。

## 症状与信号

- 时间线只有 CUDA HW / API 行，没有 NVTX；
- `nsys stats --report nvtx_sum report.nsys-rep` 只有表头，没有数据；
- 报告里只有 `:outer`，没有 `:inner`；
- 只在主线程看到 range，子线程没有；
- `--nvtx-capture=TARGET` 执行后输出 “No reports were generated”；
- `nsys stats` 提示 SQLite 比 `.nsys-rep` 旧，或读到了上一次的结果。

## 快速判断

1. **先看 `nvtx_sum` 是否为空**：空说明不是“名字不对”，而是根本没有 NVTX 事件。
2. **看 nsys 日志/诊断**：有没有 “NVTX injection initialized successfully”、 “Number of NVTX events collected”；有没有 “wrong thread ID” 之类的告警。
3. **最小程序能否复现**：用 1 秒的 `nvtxRangePushA("test")` 跑一遍，排除 API 用法问题。
4. **range 是否真的执行并闭合**：分支、提前 `return`、异常、线程结束都可能让 `pop` 没走到。
5. **domain/trace 是否被过滤**：`--nvtx-domain-include/exclude`、`--capture-range`、GUI 过滤器都可能藏掉事件。
6. **进程/PID 是否对**：多进程、子进程、容器、`python -m` 启动方式都要确认报告里有目标 PID。

## 根因（按可能性排序）

| 可能性 | 现象 | 检查 |
| --- | --- | --- |
| 采集时没开 NVTX trace | 报告完全没有 NVTX 行 | `nsys profile --trace=nvtx ...` |
| 程序不是从 nsys 启动 | 没有报告或报告里没有目标进程 | 用 `nsys profile` 包住程序；检查子进程 |
| range 代码没执行 | 只有部分 range | 加日志/断点；检查分支和循环次数 |
| push/pop 没闭合 | 后续 range 嵌套错、时长异常 | 用 RAII；在进程结束前检查栈 |
| 在错误的线程上打点 | 某个线程没有 range | 检查 `globalTid`；push/pop 不能跨线程 |
| domain 被过滤掉 | 只有默认 domain 或其他 domain | 检查 `--nvtx-domain-include/exclude`、GUI 过滤 |
| capture-range 条件不匹配 | “No reports were generated” | 用 registered string；写对 `@domain` |
| NVTX 被编译掉 | 所有 NVTX 都没有 | 检查 `-DNVTX_DISABLE` |
| 注入被显式关闭 | nsys 无法接管 | 检查 `NVTX_INJECTION64_PATH` / `NVTX_INJECTION32_PATH` |
| 另一个工具已接入 | 行为异常或没有数据 | NVTX 同一进程只支持一个工具接入 |
| SQLite 是旧的 | 统计和报告不一致 | `--force-export=true` 重新导出 |

## 排查步骤

### 第 1 步：用最小程序确认链路

```cpp
// nvtx-smoke.cpp
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <thread>

int main() {
  nvtxRangePushA("smoke");
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  nvtxRangePop();
  return 0;
}
```

```bash
NVTX_INC=${NVTX_INC:-/usr/local/cuda/include}
g++ -O2 -std=c++17 -I"$NVTX_INC" nvtx-smoke.cpp -o nvtx-smoke
nsys profile --trace=nvtx --stats=true -o smoke ./nvtx-smoke
```

如果 `nvtx_sum` 里没有 `:smoke`，问题在采集/注入，不在你的业务代码；如果有，问题在真实程序的执行路径、range 生命周期或过滤条件。

### 第 2 步：看 nsys 注入和事件计数

`--stats=true` 的日志里通常能看到类似：

```text
NVTX injection initialized successfully.
Number of NVTX events collected: 15.
```

如果连 “NVTX injection initialized” 都没有：

- 确认 `--trace=nvtx`（或 trace 集合里包含 nvtx）；
- 确认程序确实被 nsys 启动（不是先启动、后 attach；attach 场景另查）；
- 检查环境变量是否显式关闭了注入：`NVTX_INJECTION64_PATH=./.` 会让 NVTX 停止搜索注入库；
- 检查是否有其他 profiler 已经接入；
- 容器/WSL 下确认 nsys 与目标进程的权限、PID namespace 一致。

### 第 3 步：确认 range 真的执行并闭合

- 在 range 前后加普通日志，确认代码路径真的走到；
- 检查提前 `return`、异常、`break`、线程提前退出；
- 把 C/C++ 的手动 push/pop 改成 RAII：

```cpp
#include <nvtx3/nvtx3.hpp>

void f() {
  NVTX3_FUNC_RANGE();          // 自动闭合
  nvtx3::scoped_range r{"step"}; // 异常/return 也会 pop
  // ...
}
```

实测未闭合 range 的行为：后面的 range 会嵌进未闭合的 range，nsys 可能把它延长到进程结束，不会主动报错。因此“只少了一个 range”经常是前面某条路径漏了 `pop`。

### 第 4 步：检查线程和 domain

- 用 SQLite 看 `NVTX_EVENTS.globalTid` / `endGlobalTid`，确认 range 在哪个线程：

```bash
nsys export --type sqlite -o report.sqlite report.nsys-rep
```

```python
import sqlite3
con = sqlite3.connect("report.sqlite")
for row in con.execute("""
    SELECT n.start, n.end, s.value, n.domainId, n.globalTid
    FROM NVTX_EVENTS n
    LEFT JOIN StringIds s ON s.id = n.textId
    WHERE n.eventType IN (59, 60)
    ORDER BY n.start
"""):
    print(row)
```

- push/pop 只在同一线程配对；跨线程用 `nvtxRangeStartA` / `nvtxRangeEnd`；
- 检查是否用了 `--nvtx-domain-include` / `--nvtx-domain-exclude`，或者 GUI 里过滤了 domain；
- 自定义 domain 的名字在 `nvtx_sum` 里表现为 `domain:range`，不要按 `:range` 找不到就以为丢了。

### 第 5 步：capture-range 不触发

这是最常见的“代码有 NVTX，但报告没有”之一。检查：

1. 是否用了 `--capture-range=nvtx`；
2. 消息是否是 **registered string**（`nvtxRangePushA("TARGET")` 默认不匹配）；
3. 自定义 domain 是否写成 `TARGET@domain`；
4. 是否只触发了第一个匹配 range（CLI 默认行为）；
5. 是否在 range 开始前就退出了，或 range 太短、采集还没开始就结束。

修复方式：

```bash
# 用 registered string：C 里 nvtxDomainRegisterStringA，C++ 里 nvtx3::registered_string
# Python 的 nvtx.annotate 默认就是 registered string
nsys profile --trace=nvtx --capture-range=nvtx \
  --nvtx-capture=TARGET@roi --capture-range-end=stop \
  -o roi-report ./app
```

或者临时关闭“只匹配注册字符串”的优化：

```bash
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 nsys profile \
  --trace=nvtx --capture-range=nvtx --nvtx-capture=TARGET@roi \
  --capture-range-end=stop -o roi-report ./app
```

### 第 6 步：排除“看起来没显示”的假象

- **`nvtx_sum` 不统计 mark**：mark 只出现在时间线或 SQLite，不要用它判断“NVTX 没生效”；
- **`nvtx_sum` 的 `Time%` 不是 wall time**：嵌套/重叠 range 会重复计入；
- **SQLite 旧了**：`nsys stats` 会优先用已有 `.sqlite`，加 `--force-export=true`；
- **报告里有多个进程**：先按 PID 区分，确认不是只看了主进程；
- **`nvtxNameOsThread` 告警**：Linux 上把 `pthread_self()` 传给期望 OS tid 的 API 会得到 “wrong thread ID argument”；改用 `syscall(SYS_gettid)`。这只影响线程名，不影响 range 是否存在。

## 修复与规避

- 采集命令固定成模板，显式写 `--trace=nvtx`，不要依赖默认；
- 业务代码用 RAII/`with`，避免漏 pop；
- 跨线程用 start/end，不要跨线程 pop；
- 给库/模块建 domain，并在报告/采集命令里明确过滤条件；
- capture-range 用 registered string，domain 写全；
- 大程序先用最小 range 验证，再逐步加阶段；
- 不用 `-DNVTX_DISABLE` 的 release 构建去排查“为什么没有 NVTX”；
- 多进程先确认 PID，再改代码。

## 验证

- 最小程序能在 `nvtx_sum` 里出现，且时长符合预期；
- 真实程序的 range 数量与预期调用次数一致；
- 每个 range 的 `globalTid`/`endGlobalTid` 符合设计（同线程或跨线程）；
- capture-range 报告从目标 range 开始，且只包含目标阶段；
- 没有未闭合 range 导致的“幽灵父 range”；
- 关闭工具直接运行时程序输出不变。

## 常见坑

1. **用 `nvtx_sum` 找 mark**：找不到是正常的；
2. **只检查报告，不检查日志**：注入失败/被禁用时日志更直接；
3. **把所有 range 都放在主线程**：子线程的工作不会自动出现在主线程行；
4. **在 capture-range 里用普通字符串**：默认不触发；
5. **忘记 `@domain`**：自定义 domain 的 range 用默认 domain 过滤不到；
6. **`--trace=all` 或数据量过大**：报告可能大到难以打开，先缩小 trace/ROI；
7. **SQLite 旧数据**：统计和 `.nsys-rep` 不一致时先 `--force-export=true`；
8. **版本变化**：`nvtx_sum`/capture 选项、GUI 行为随 nsys 版本变化；先看本机 `--help`。

## 参考

- Nsight Systems User Guide（NVTX Trace、capture range、domain 过滤）：<https://docs.nvidia.com/nsight-systems/UserGuide/index.html#nvtx-trace>
- NVTX 官方总览（注入、registered string、单工具限制）：<https://github.com/NVIDIA/NVTX>
- NVTX C/C++ 示例（RAII、start/end）：<https://github.com/NVIDIA/NVTX/blob/release-v3/c/README.md>
- 本仓 [nsys 工作流](nvtx-nsys-workflow.md) 和 [range 语义](nvtx-range-model.md)
