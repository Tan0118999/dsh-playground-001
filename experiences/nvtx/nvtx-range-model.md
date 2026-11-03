---
name: nvtx-range-model
title: NVTX range 的语义：mark、push/pop、start/end、线程与 domain
description: 分不清 mark 和 range、range 嵌套错乱、pop 关错范围、想跨线程标注、多个库的标注混在一起时读这条。讲清 push/pop 的线程栈、start/end 句柄、domain 隔离、category 分组和资源命名的选择。
category: methodology
tags: [nvtx, ranges, domains, threads, nsys]
symptoms:
  - range 嵌套层级和代码结构对不上
  - pop 之后发现关错了 range
  - 想标一个跨线程的范围，不知道用哪个 API
  - 多个库的 NVTX 名字混在一起
  - 不知道 category 和 domain 有什么区别
stacks: [nvtx, nsys, cuda, python]
updated: 2026-11-03
---

# NVTX range 的语义：mark、push/pop、start/end、线程与 domain

> 一句话经验：**push/pop 是“同一线程、同一 domain 内的栈”；start/end 是“拿着句柄、可以跨线程和重叠的区间”；跨库隔离用 domain，组内分组用 category。**

## 适用场景

- 已经能打出一个 range，但 range 的嵌套、开始/结束位置和代码结构对不上；
- 想标注一个从线程 A 开始、线程 B 结束的范围（例如异步任务、回调、通信）；
- 代码里同时有框架、通信库、自研模块的 NVTX，想按模块过滤；
- 需要给 range 加颜色、category、payload，但不知道应该用哪个 API。

不解决：怎么采集、怎么在报告里查看（见 [nsys 工作流](nvtx-nsys-workflow.md)）；也不解决“完全看不到 NVTX”的排查（见 [看不到 NVTX 的排查](nvtx-not-showing.md)）。

## 症状与信号

- 代码里先 `push A`、再 `push B`、然后 `pop`，但时间线上 `B` 的结束位置跑到 `A` 后面；
- 函数提前 `return` 或抛异常，后面的 range 全部嵌进一个“看不见的父 range”；
- 在子线程里 `pop` 主线程 `push` 的 range，报告里层级混乱或直接没有；
- 两个库都用了 `forward` 这个名字，在 `nvtx_sum` 里合并成一行，无法区分；
- 想只统计某个模块的 range，却只能看到一堆默认 domain 的名字。

## 快速判断

1. **需要“时间点”还是“时间段”？** 时间点用 mark，时间段用 range。
2. **开始和结束在同一线程吗？** 是：优先 push/pop；否：用 start/end。
3. **需要嵌套吗？** push/pop 天然是栈；start/end 可以任意重叠，不要求嵌套。
4. **是库/框架的标注吗？** 用独立 domain，避免和业务代码重名。
5. **需要在 domain 内再分组/过滤？** 用 category；需要颜色/附加数据？用 attributes（进阶篇）。
6. **线程行名字是数字？** 用 `nvtxNameOsThread*` 命名；Linux 上注意传 `gettid()`，不是 `pthread_self()`。

## 四种标注

| 类型 | C API | 生命周期 | 能否跨线程 | 典型用途 |
| --- | --- | --- | --- | --- |
| mark | `nvtxMarkA` / `nvtxMarkEx` | 瞬时 | 无持续时间 | 标记一个事件点，例如“开始保存 checkpoint” |
| push/pop | `nvtxRangePushA(Ex)` / `nvtxRangePop` | 同一线程、同一 domain 的栈 | 否 | 函数/阶段嵌套，最常用 |
| start/end | `nvtxRangeStartA(Ex)` / `nvtxRangeEnd` | 句柄 `nvtxRangeId_t`，显式结束 | 是 | 异步任务、跨线程回调、可重叠区间 |
| resource naming | `nvtxNameOsThreadA`、`nvtxNameCudaStreamA` 等 | 给对象绑定名字 | 不适用 | 让线程/stream 行可读 |

C++ 封装把 push/pop 做成 RAII（`nvtx3::scoped_range`、`NVTX3_FUNC_RANGE`），把 start/end 做成可移动的 `nvtx3::unique_range`；Python 的 `nvtx.annotate` 是上下文管理器/装饰器，底层仍然是 push/pop。

## 机制：为什么 pop 会关错

NVTX 为**每个线程、每个 domain** 维护一个独立的 range 栈：

```text
线程 T1 / 默认 domain:
  push "outer"     -> 栈: [outer]
    push "inner"   -> 栈: [outer, inner]
    pop            -> 关 inner，栈: [outer]
  pop              -> 关 outer，栈: []

线程 T2 / 默认 domain:
  push "worker"    -> 独立栈: [worker]
```

因此：

- `pop` 不需要参数，它关的是**当前线程、当前 domain 最近一次未关闭的 push**；
- 在 T2 里 `pop` 不会关掉 T1 的 range；push/pop 不能跨线程配对；
- 未关闭的 range 会一直留在栈上。实测中，进程结束时 nsys 可能把它延长到进程结束，于是后面的 range 全都变成它的子 range，时长看起来偏大；
- domain 是命名空间：不同 domain 有各自的栈、category 编号和 registered string 表，所以同名 range 可以在不同 domain 共存。

start/end 不依赖栈：`nvtxRangeStartA` 返回一个 `nvtxRangeId_t`，你把句柄传给 `nvtxRangeEnd`，因此开始和结束可以在不同线程，多个区间也可以任意重叠。代价是你要自己保存句柄、保证只 end 一次。

## 代码：嵌套、跨线程、domain

### push/pop 嵌套（C）

```cpp
#include <nvtx3/nvToolsExt.h>

void step() {
  nvtxRangePushA("step");
  nvtxRangePushA("inner");
  // ...
  nvtxRangePop();  // 关 inner
  nvtxRangePop();  // 关 step
}
```

### 跨线程 start/end（C）

```cpp
#include <nvtx3/nvToolsExt.h>
#include <thread>

nvtxRangeId_t id = nvtxRangeStartA("async-job");
std::thread t([id] {
  // ... 在另一个线程完成工作
  nvtxRangeEnd(id);  // 用同一个句柄结束
});
t.join();
```

### domain + category + 线程名（C）

```cpp
#include <nvtx3/nvToolsExt.h>
#include <sys/syscall.h>
#include <unistd.h>

int main() {
  // Linux 上要传 OS 线程 ID（gettid），不是 pthread_self() 的 pthread_t
  nvtxNameOsThreadA((uint32_t)syscall(SYS_gettid), "main-thread");

  nvtxDomainHandle_t dom = nvtxDomainCreateA("com.example.data");
  nvtxDomainNameCategoryA(dom, 1, "io");

  nvtxEventAttributes_t attr = {};
  attr.version = NVTX_VERSION;
  attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attr.message.ascii = "load-batch";
  attr.category = 1;

  nvtxDomainRangePushEx(dom, &attr);
  // ... 读数据
  nvtxDomainRangePop(dom);

  nvtxDomainDestroy(dom);
}
```

在 `nvtx_sum` 里，默认 domain 显示为 `:range`，自定义 domain 显示为 `domain:range`。实测：

```text
Time (%)  ...  Style     Range
--------  ...  --------  -----------------------------------
   29.9   ...  PushPop   :outer
   24.4   ...  StartEnd  :cross-thread
   17.2   ...  PushPop   com.example.nvtxlab:registered-loop
   16.3   ...  PushPop   :inner
```

`outer` 包含 `inner` 时，两者的时间会**重复计入**；`nvtx_sum` 的 `Time%` 不是 wall time 占比，不要把它们直接相加当成总时间。

### Python 的嵌套

```python
import time
import nvtx

with nvtx.annotate("step", color="green", category="pipeline"):
    with nvtx.annotate("inner"):
        time.sleep(0.05)
```

Python 的 `with` 保证异常时也会 `pop`；比手动 `push_range`/`pop_range` 安全。需要跨线程时，`nvtx.start_range` 返回一个句柄，`nvtx.end_range(handle)` 结束；但跨线程传递句柄要自己保证生命周期。

## 修复与规避

- **优先 RAII**：C++ 用 `nvtx3::scoped_range` / `NVTX3_FUNC_RANGE()`；Python 用 `with nvtx.annotate(...)`；
- **一个库一个 domain**：框架、通信库、自研模块各自建 domain，避免同名合并；
- **category 只做组内分组**：例如 `io`、`compute`、`comm`，不要用 category 代替 domain；
- **线程名用 `gettid()`**：Linux 上 `pthread_self()` 是 `pthread_t`，传给期望 OS tid 的 API 会得到 “wrong thread ID argument” 告警；
- **不要把 range 放进微秒级热循环**：官方建议，被标注代码通常小于 1 µs 时要谨慎；同时打开的 range 也不要太多（经验上控制在几十个以内）；
- **domain 销毁前确认没有未结束的 range**：销毁 domain 会注销它关联的对象和已开始的 range，容易让报告缺段。

## 验证

- `nvtx_sum` 的 `Style` 列能区分 `PushPop` 和 `StartEnd`；
- 跨线程 range 的时长覆盖两个线程上的实际工作，且只出现一次；
- 自定义 domain 的 range 在报告里带 `domain:` 前缀；
- 用 SQLite 检查 `NVTX_EVENTS.domainId`、`category`、`globalTid`/`endGlobalTid` 是否符合预期（见 [消费端与自动化](nvtx-ecosystem-and-automation.md)）；
- 制造一次提前 `return` 的路径，确认 RAII 版本没有留下未闭合 range。

## 常见坑

1. **在异常路径漏 `pop`**：nsys 不一定报错，但后续 range 会嵌进“幽灵父 range”；
2. **跨线程 `pop`**：push/pop 的栈是线程私有的，不能配对；
3. **用 `pthread_self()` 命名线程**：Linux 上会告警；用 `syscall(SYS_gettid)` 或 `gettid()`；
4. **把 category 当 domain**：category 只是整数分组，不能隔离 range 栈和字符串表；
5. **同名 range 来自不同库**：没有 domain 时会在统计里合并；
6. **domain 销毁太早**：还有活跃 range 时销毁，报告可能缺段；
7. **把 mark 当 range 统计**：`nvtx_sum` 不统计 mark，mark 要去看时间线或 SQLite。

## 参考

- NVTX 总览（Markers、Ranges、Domains、资源命名）：<https://github.com/NVIDIA/NVTX>
- NVTX C/C++ 示例（push/pop、start/end、线程命名）：<https://github.com/NVIDIA/NVTX/blob/release-v3/c/README.md>
- NVTX C API 参考：<https://nvidia.github.io/NVTX/doxygen/>
- Nsight Systems NVTX Trace：<https://docs.nvidia.com/nsight-systems/UserGuide/index.html#nvtx-trace>
- NVTX Python 文档：<https://nvidia.github.io/NVTX/python/>
