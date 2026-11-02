---
name: cudagraph-capture-problems
title: CUDA Graphs 捕获失败、回退与捕获后正确性问题
description: 使用 torch.compile 的 reduce-overhead、torch.cuda.graph 或推理图捕获时，出现 operation not permitted when stream is capturing、捕获失败回退、捕获后结果错误或卡住，按“捕获期非法操作—预热—静态输入—多流/集合通信”逐项排查。
category: performance
tags: [cuda-graphs, torch.compile, inductor, capture, inference]
symptoms:
  - CUDA error: operation not permitted when stream is capturing
  - torch.compile 日志提示 skipping cudagraphs，性能没有提升
  - 捕获成功但 replay 结果错误或第一次正确、第二次错误
  - 捕获阶段卡住、超时或进程 abort
  - 多卡/NCCL 场景捕获失败
stacks: [torch.compile, cuda, triton]
updated: 2026-11-02
---

# CUDA Graphs 捕获失败、回退与捕获后正确性问题

> 一句话经验：捕获期只能做“图内合法”的操作；先看日志确认是否回退，再用同步告警和最小捕获复现找出捕获期的分配、同步或通信。

## 适用场景

- `torch.compile(mode="reduce-overhead")`、`torch.cuda.graph`/`torch.cuda.CUDAGraph` 或推理框架的图捕获。
- 捕获报错、捕获被跳过（回退到普通执行）、捕获后结果错误/卡住。
- 多流、NCCL、动态 shape、含 CPU 逻辑的模型。

## 症状与信号

- `RuntimeError: CUDA error: operation not permitted when stream is capturing`：捕获期间执行了不允许的操作（分配、释放、同步、事件查询等）。
- 日志出现 `skipping cudagraphs due to ...`：Inductor 主动回退，原因通常写在后面（动态 shape、CPU 操作、不支持的算子等）。
- 捕获成功但 replay 结果错误：静态 buffer 被覆盖、输入没有拷入静态 buffer、随机数状态、跨流依赖缺失。
- 只有第二次及以后调用出错：图复用了固定的内存地址，而外部仍持有/修改旧 tensor。
- 多卡捕获卡住：集合通信在捕获期执行，或 rank 间捕获顺序不一致。

## 快速判断

1. **确认是否真的启用**：看 `TORCH_LOGS=perf_hints`，确认 cudagraph 没有被跳过。
2. **打开同步告警**：`torch.cuda.set_sync_debug_mode("warn")`，捕获期出现同步操作会给出提示。
3. **关闭 cudagraph 对比**：`mode="default"` 或关闭对应 config，若问题消失，锁定在图捕获。
4. **最小捕获复现**：脱离 `torch.compile`，用 `torch.cuda.CUDAGraph` 捕获单层/单算子，复现后再加回复杂度。
5. **检查静态输入**：确认每次 replay 前把新输入拷进固定的静态输入 buffer，而不是重新分配。

## 排查步骤

### 1. 看日志与回退原因

```bash
TORCH_LOGS=perf_hints python3 run.py 2>&1 | grep -i -E "cudagraph|skipping|capture"
```

`skipping cudagraphs due to ...` 属于 `perf_hints` 日志（默认关闭，必须显式点名），只开 `+inductor` 不一定能看到。常见原因：输入被原地修改、图里有 CPU 操作、多设备、动态 shape、含不安全的 custom op、不兼容的 aten op（如 `.item()` 触发的 `_local_scalar_dense`）。

另一个常见错误是输出 buffer 被下一次 replay 覆盖：

```text
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run.
```

处理方式：在每步开始处调用 `torch.compiler.cudagraph_mark_step_begin()`，或在图外 `clone()` 输出后再使用。先按原因处理，不要盲目强制开启。

### 2. 打开同步告警

```python
import torch
torch.cuda.set_sync_debug_mode("warn")
```

在捕获前后运行；如果出现 “called a synchronizing CUDA operation” 警告，定位到对应 Python/C++ 调用。该模式是实验性的，且不覆盖 `torch.distributed`/`torch.sparse`。捕获期禁止的典型操作：

- `cudaMalloc`/`cudaFree`（包括第一次触发新 kernel 的模块加载、临时 tensor 分配）。
- `cudaDeviceSynchronize`、`cudaStreamSynchronize`、`cudaEventSynchronize`、`cudaEventQuery`。
- 页锁内存之外的同步拷贝、`cudaMemcpy` 的同步变体。
- 某些 NCCL 集合通信、CPU 侧分支、文件/网络 I/O。

### 3. 用最小 CUDAGraph 复现

```python
import torch

model = model.cuda().eval()
static_in = torch.randn(8, 128, device="cuda")
static_out = None

# 预热：让 kernel 模块加载、autotune、内存池都在捕获前完成
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        model(static_in)
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = model(static_in)

# replay 前把新输入拷贝到固定 buffer
static_in.copy_(new_input)
g.replay()
torch.cuda.synchronize()
print(static_out)
```

要点：

- 预热必须覆盖所有会首次加载/autotune 的 kernel，否则捕获期会触发模块加载或分配。
- 输入/输出地址在捕获后固定；每次 replay 用 `copy_` 更新输入。
- 捕获后不要依赖捕获前创建的临时 tensor；它们可能被图内存池复用。
- 用 `torch.cuda.graph` 上下文管理器，它会处理内存池；手写 `CUDAGraph.capture_begin/capture_end` 更容易出错。

### 4. 检查 torch.compile 的 cudagraph 配置

```python
import torch._inductor.config as config

print("cudagraphs:", config.triton.cudagraphs)
print("cudagraph_trees:", config.triton.cudagraph_trees)
config.triton.cudagraphs = False   # 仅用于对比定位
```

配置项名称和默认值随版本变化，以当前版本为准。关闭后问题消失，说明问题在捕获路径；再逐项排查动态 shape、CPU 操作、多流和通信。

### 5. 多流与多卡

- 捕获必须在非默认流上进行；跨流依赖要用 event 显式建立，捕获期不能动态创建/查询 event。
- NCCL 集合通信在捕获期执行时，必须使用支持图捕获的路径，且所有 rank 的捕获顺序一致；否则会卡住或超时。
- 多卡场景先退化为单卡验证图本身，再恢复多卡；用 `nsys` 看是否有 rank 在等待。

## 根因

1. **捕获期分配/释放**：首次 kernel 加载、临时 tensor、allocator 扩张都会触发 `cudaMalloc`。
2. **捕获期同步**：打印、`.item()`、CPU 分支、`torch.cuda.synchronize()`、事件查询。
3. **动态 shape/控制流**：每次输入 shape 不同，图无法固定地址和 launch 配置。
4. **静态 buffer 误用**：replay 时没有把新输入拷入固定 buffer，或外部仍引用旧输出。
5. **多流/通信**：event、stream、NCCL 的捕获语义不满足。
6. **随机数/状态**：RNG 状态、in-place 更新、优化器状态没有在 replay 中正确推进。

## 修复与规避

- 预热所有 kernel 与 autotune；把分配移出捕获区；用图内存池。
- 固定输入 shape，或为不同 shape 分别捕获；不要把动态 shape 图强行套 cudagraph。
- 把 `.item()`、日志、CPU 分支移到捕获外；用张量操作替代。
- 多卡场景确认通信库版本支持图捕获；不支持时对通信部分关闭 cudagraph。
- 规避：`mode="default"`、`config.triton.cudagraphs=False`、推理框架的 cudagraph 开关；需说明性能代价。

## 验证

- 捕获后连续 replay 多次，结果与 eager 一致（含随机性处理）。
- 每次换输入都用 `copy_` 更新静态 buffer，确认没有读到旧数据。
- 多卡场景覆盖 2/4 卡，重复捕获/释放，确认没有泄漏和超时。
- 用 `nsys` 或 profiler 确认 replay 阶段没有隐式同步/分配。
- 性能对比：捕获前后的端到端延迟和显存占用。

## 常见坑

- 只做一次 warmup 就捕获：autotune 可能还没结束，捕获期仍会编译/分配。
- 捕获后继续用捕获前的输出 tensor，或对它做 in-place 修改。
- 认为“捕获成功”就代表正确：必须对拍 replay 结果。
- 在捕获期使用 `torch.cuda.synchronize()` 或打印，导致直接报错或回退。
- 多卡只在一张卡上验证捕获，忽略 rank 间顺序和通信。
- 捕获失败后不清空图对象就重试，可能留下半初始化的状态；重建 `CUDAGraph`。

## 参考

- PyTorch CUDA Graphs：<https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs>
- `torch.cuda.CUDAGraph`：<https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html>
- `torch.cuda.graph`：<https://pytorch.org/docs/stable/generated/torch.cuda.graph.html>
- `torch.cuda.set_sync_debug_mode`：<https://pytorch.org/docs/stable/generated/torch.cuda.set_sync_debug_mode.html>
- CUDA Graphs 编程指南：<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs>
