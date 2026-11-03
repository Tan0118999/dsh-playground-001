---
name: nvtx-first-range
title: 第一次用 NVTX：5 分钟给时间线加上阶段名
description: 从没接触过 NVTX、Nsight Systems 时间线里只有 kernel 名、看不出数据加载/forward/backward 对应哪段 GPU 活动时读这条。给出 C++ 与 Python 的最小可运行示例、nsys 采集命令和 nvtx_sum 验证方式。
category: methodology
tags: [nvtx, nsys, cuda, profiling, quickstart]
symptoms:
  - 第一次接触 NVTX，不知道从哪里开始
  - 时间线里只有 kernel 名，看不出训练或推理的阶段
  - 在代码里加了 NVTX，但报告里找不到
stacks: [nvtx, nsys, cuda, python]
updated: 2026-11-03
---

# 第一次用 NVTX：5 分钟给时间线加上阶段名

> 一句话经验：**先用一个包住阶段的 push/pop range 跑通 `nsys profile --trace=nvtx --stats=true`，在 `nvtx_sum` 里看到名字，再考虑嵌套、domain 和 payload。**

## 适用场景

- 你有一个 C/C++ 或 Python 程序，已经能用 Nsight Systems 采到 kernel，但时间线只有 kernel 名；
- 你想知道“数据加载 / forward / backward / 通信”各占多久，而不是逐个猜 kernel；
- 你准备把 NVTX 加进代码，但不想一开始就学完整 API。

不解决：单 kernel 内部的瓶颈（用 [Nsight Compute 第一轮](../gpu-profiling/nsight-compute-first-pass.md)）；也不解决“NVTX 打了但看不到”的复杂情况（见 [看不到 NVTX 的排查](nvtx-not-showing.md)）。

## 症状与信号

- 时间线里只有 `matmul`、`elementwise` 这类 kernel 名，看不出它们属于哪个业务阶段；
- 你知道某个阶段慢，但不知道该阶段对应哪些 kernel、是否包含拷贝或同步；
- 代码里写了 `nvtxRangePush`，但 `nsys stats --report nvtx_sum` 输出为空；
- 加完 NVTX 后行为没变化，怀疑根本没生效。

## 快速判断

1. 程序是不是**在 nsys 下启动**的？直接运行 `./app` 不会产生任何报告。
2. 采集时有没有 `--trace=nvtx`？默认 trace 集合通常包含 NVTX，但显式写出来最稳。
3. 报告里有没有 NVTX 行？先跑 `nsys stats --report nvtx_sum report.nsys-rep`。
4. 有 NVTX 行但没有你的名字：检查 range 是否真的执行、是否在目标线程闭合。
5. 完全没有 NVTX 行：按 [看不到 NVTX 的排查](nvtx-not-showing.md) 逐层查。

## 最小 C++ 示例

NVTX v3 的 C 头文件是 header-only 的，**不需要链接 NVTX 库**：

```cpp
// first.cpp
#include <nvtx3/nvToolsExt.h>

#include <chrono>
#include <thread>

int main() {
  nvtxRangePushA("main");
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  nvtxRangePop();
  return 0;
}
```

编译（`NVTX_INC` 指向 CUDA Toolkit 的 include 目录；CUDA 11.6+ 自带 `nvtx3/`）：

```bash
NVTX_INC=${NVTX_INC:-/usr/local/cuda/include}
g++ -O2 -std=c++17 -I"$NVTX_INC" first.cpp -o first
```

采集并直接打印统计：

```bash
nsys profile --trace=nvtx --stats=true -o first-report ./first
```

也可以分两步：

```bash
nsys profile --trace=nvtx -o first-report ./first
nsys stats --report nvtx_sum first-report.nsys-rep
```

期望在 `nvtx_sum` 里看到类似：

```text
 Time (%)  Total Time (ns)  Instances   Avg (ns)  ...  Style    Range
 --------  ---------------  ---------  ----------  ...  -------  -------
    100.0        100123456          1  100123456.0  ...  PushPop  :main
```

- 行首的 `:` 表示默认 domain（没有指定 domain）；
- `PushPop` 表示这是 `nvtxRangePushA` / `nvtxRangePop` 配对；
- `Instances=1`、时长约 100 ms，说明 range 确实包住了 `sleep`。

如果暂时没有 CUDA Toolkit，也可以从 Nsight Systems 安装目录的 `target-<arch>/nvtx/include` 或 [NVIDIA/NVTX](https://github.com/NVIDIA/NVTX) 取头文件；老 glibc（< 2.34）编译时可能需要加 `-ldl`。

## 最小 Python 示例

Python 用官方的 `nvtx` 包：

```bash
python3 -m pip install nvtx
```

```python
# first.py
import time

import nvtx

with nvtx.annotate("main", color="green"):
    time.sleep(0.1)
```

```bash
nsys profile --trace=nvtx --stats=true -o py-report python3 first.py
nsys stats --report nvtx_sum py-report.nsys-rep
```

期望看到 `:main`，时长约 100 ms。`nvtx.annotate` 既可以是上下文管理器，也可以当装饰器用；当装饰器且不传 `message` 时，默认用函数名。

## 为什么这样就够了

- NVTX 调用本身只是把标签交给“注入的工具”。**没有工具时，第一次调用会初始化并关闭后续调用**，之后基本是几条指令；
- nsys 通过注入库接收 NVTX 事件，所以必须从 `nsys profile` 启动程序；
- 采集时必须打开 NVTX trace，否则注入库不会记录这些事件；
- `push/pop` 是**同一线程内的栈**：`pop` 关的是该线程、该 domain 最近一个 `push`。因此最小示例要把 `pop` 放在所有返回路径都会经过的位置；C++ 里更推荐 RAII（见 [进阶 API](nvtx-advanced-api.md)）。

## 修复与规避

- **只包阶段，不包热循环**：先包 `dataloader`、`forward`、`backward`、`optimizer.step`、`all_reduce` 这类毫秒级阶段；
- **名字带关键参数**：`forward bs=8 seq=2048` 比 `forward` 更有用；但频繁变化的字符串会带来格式化开销，固定模板 + payload 更好（进阶篇）；
- **先跑通再加属性**：颜色、category、payload 都建立在你已经能在 `nvtx_sum` 里看到 range 的基础上；
- **异常路径用 RAII**：C++ 用 `nvtx3::scoped_range` 或 `NVTX3_FUNC_RANGE()`，避免 `return`/`throw` 漏 `pop`；Python 用 `with nvtx.annotate(...)`。

## 验证

- `nvtx_sum` 里出现你的 range，`Instances` 和预期调用次数一致；
- range 的时长与你在代码里看到的阶段耗时同量级（例如 `sleep(100ms)` 对应约 100 ms）；
- 不在 nsys 下运行时程序输出不变，说明 NVTX 没改变程序行为；
- 想确认“没有工具时是 no-op”，可以用 `-DNVTX_DISABLE` 重新编译，报告里应该没有任何 NVTX range（实测有效）。

## 常见坑

1. **只运行程序，没有用 nsys 启动**：不会有报告，也不会有 NVTX 行。
2. **`--trace` 没写 `nvtx`**：报告里可能只有 kernel，没有 NVTX。
3. **把 `record_function` 当成 NVTX**：PyTorch 的 `torch.profiler.record_function` 默认只进 PyTorch Profiler 的 trace，不等于 NVTX；用 `torch.cuda.nvtx` 或 `nvtx.annotate`（见 [Python 与 PyTorch](nvtx-python-pytorch.md)）。
4. **range 没闭合**：后面的 range 会嵌进未闭合的 range，时长看起来“多了一截”；nsys 可能不报错，但数据不可信。
5. **链接了旧库**：v3 头文件是 header-only 的，不需要 `-lnvToolsExt`；如果构建脚本还在链接旧库，先确认 include 的是 `nvtx3/` 而不是旧的 `nvToolsExt.h`。
6. **第一次调用的初始化开销**：第一次 NVTX 调用可能加载注入库；不要在延迟敏感的第一次调用处测量。

## 参考

- NVTX 官方总览：<https://github.com/NVIDIA/NVTX>
- NVTX C/C++ 示例（`scoped_range`、`NVTX3_FUNC_RANGE`）：<https://github.com/NVIDIA/NVTX/blob/release-v3/c/README.md>
- Nsight Systems NVTX Trace：<https://docs.nvidia.com/nsight-systems/UserGuide/index.html#nvtx-trace>
- NVTX Python 文档：<https://nvidia.github.io/NVTX/python/>
- 本仓可复现 GPU workload：[`toolbox/gpu-profiling-demo`](../../toolbox/gpu-profiling-demo/README.md)
