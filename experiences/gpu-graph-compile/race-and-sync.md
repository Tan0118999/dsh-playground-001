---
name: race-and-sync
title: 数据竞争与同步缺失：偶发错误、结果随运行变化
description: 结果偶发错误、重复运行不一致、只在多 warp/多 block/多卡出现，或 racecheck 报 hazard 时，按“重复实验—racecheck—单 warp 复现—加同步验证”的顺序定位共享内存、原子操作和 stream 同步问题。
category: correctness
tags: [race-condition, synchronization, triton, cuda, sanitizer]
symptoms:
  - 同一输入重复运行，结果偶尔不同或偶尔崩溃
  - 单 block/单 warp 正确，多 block/多 warp 错误
  - 只在多卡、多流或高并发下出现
  - compute-sanitizer racecheck 报 shared memory hazard
  - 加一句 torch.cuda.synchronize() 后问题消失
stacks: [triton, cuda, torch.compile, xla]
updated: 2026-11-02
---

# 数据竞争与同步缺失：偶发错误、结果随运行变化

> 一句话经验：先证明它是“非确定性”的，再用 racecheck 和单 warp 复现定位；不要用“加锁/加同步后好了”代替根因分析。

## 适用场景

- 结果偶发错误、偶发 NaN、偶发非法访存，重复运行复现率不同。
- 多 warp/多 block/多卡/多流才出问题，单线程或单 block 正常。
- CUDA C++ kernel 中共享内存或 warp 级通信没有正确同步。
- Triton kernel 使用原子操作、跨 program 通信，或依赖未定义执行顺序。

## 症状与信号

- 同一输入、同一进程内多次运行结果不同；或换一个无关的 kernel 后结果变化。
- 错误率随 block 数、warp 数、并发度、GPU 型号变化。
- `compute-sanitizer --tool racecheck` 报告 hazard。
- 在关键位置加 `torch.cuda.synchronize()`、`cudaDeviceSynchronize()` 或 `__syncthreads()` 后暂时消失。
- 只在 `cudagraph`、多流、NCCL 场景出现。

## 快速判断

1. **证明非确定性**：同一输入重复运行 50–100 次，记录失败率；每次重置随机种子和输入。
2. **同步化**：加 `CUDA_LAUNCH_BLOCKING=1` 或在每次 kernel 后 `torch.cuda.synchronize()`。若问题消失，说明是异步执行暴露顺序问题，不一定是同步缺失本身。
3. **缩小并发**：把 grid/block 降到 1 个 block、1 个 warp，逐步加回，找到触发并发度的阈值。
4. **racecheck**：对可疑 kernel 跑 `compute-sanitizer --tool racecheck`，优先看共享内存 hazard。
5. **改调度参数**：改变 `num_warps`、`num_stages`、block size，若只在某些配置出错，通常是同步或共享内存复用问题。

## 排查步骤

### 1. 建立可重复的失败率

```python
import torch

torch.manual_seed(0)
x = torch.randn(1024, 1024, device="cuda")
ref = None
fail = 0
for i in range(100):
    out = kernel(x)
    if ref is None:
        ref = out.clone()
    elif not torch.equal(out, ref):
        fail += 1
print("failures:", fail, "/ 100")
```

不要只看一次运行；竞态可能只有 1% 复现率。

### 2. 用 racecheck 定位共享内存 hazard

```bash
compute-sanitizer --tool racecheck --print-limit 20 python3 repro.py
```

racecheck 主要检查共享内存的读写 hazard，能给出 kernel、地址和冲突的访问类型。它不会覆盖所有全局内存顺序问题，也不替代 memcheck；如果同时有越界，先用 `--tool memcheck` 排掉。

### 3. 检查同步点

CUDA C++ 中重点看：

- `__syncthreads()` 是否覆盖所有线程（放在条件分支里是经典错误）。
- 共享内存写后读、读后写、写后写之间是否有 barrier。
- 是否错误依赖 warp-synchronous 执行（Volta 之后不能假设 warp 内指令严格同步，需要 `__syncwarp()`）。
- 跨 block 的依赖是否通过全局内存 + `__threadfence()` + 原子操作正确建立顺序。
- 原子操作是否只保证原子性，不保证你需要的顺序。

Triton 中重点看：

- 大多数跨线程数据交换由编译器自动插入同步；不要假设可以手工控制共享内存布局。
- `tl.debug_barrier()` 是调试/特殊场景的逃生口，不是通用修复；滥用会降低性能或引入死锁。
- 多个 program 同时写同一全局地址时，用 `tl.atomic_add` 等原子操作，并确认顺序无关或先做归约。
- `num_stages` 软件流水线依赖编译器正确插入 barrier；自定义内联汇编或非常规访存可能破坏假设。

### 4. 检查 stream 与图捕获

- 多流场景：用 `torch.cuda.Stream`、`wait_stream`、`record_event` 明确依赖；不要依赖默认流的隐式顺序。
- `torch.cuda.synchronize()` 只用于调试定位，不要把它当修复。
- CUDA Graphs 捕获会固定执行顺序和内存地址；捕获期间跨流/多进程通信容易出问题，见[CUDA Graphs 捕获问题](cudagraph-capture-problems.md)。
- 多卡：collective 的调用顺序和数量必须在所有 rank 上一致，见[分布式图编译](distributed-graph-compile.md)。

### 5. 用最小改动验证假设

- 在可疑读写之间插入 barrier，看失败率是否归零；归零只能证明“这里缺同步”，还要确认修复没有改变语义。
- 把共享内存复用改为独立 buffer，排除生命周期重叠。
- 把并行归约改为串行/单 block，确认结果正确后逐级恢复并行。

## 根因

1. **共享内存生命周期重叠**：同一块共享内存先后存不同数据，中间缺少 barrier。
2. **条件同步**：`__syncthreads()` 在分支内，部分线程不执行，导致未定义行为或死锁。
3. **warp 同步假设**：依赖 Volta 之前的隐式 warp 同步语义。
4. **跨 block 顺序**：没有用原子/ fence 建立 happens-before，却假设 block 执行顺序。
5. **stream/图顺序**：多个 stream 或 cudagraph 捕获后，依赖关系没有显式表达。
6. **原子操作误用**：把原子性当成顺序保证；或对浮点原子累加的顺序敏感。

## 修复与规避

- 首选补齐同步或重新划分阶段；其次用双缓冲/独立 buffer 消除复用。
- 对跨 block 归约，用“每个 block 写自己的槽位 + 第二个 kernel 汇总”替代跨 block 自旋等待。
- 浮点原子累加结果不确定时，改为确定性归约树，或接受误差并在验证中覆盖。
- 规避手段（单流、强制同步、单 block）只用于定位或临时止损，需说明性能代价。

## 验证

- 同一输入重复 100 次，结果完全一致（或误差在明确容差内）。
- `compute-sanitizer --tool racecheck` 无 hazard；同时跑 `--tool memcheck` 确认没有越界。
- 覆盖不同 block/warp 数、不同 GPU 架构、不同并发度。
- 多卡场景覆盖 2 卡和 4 卡，重复多次并注入延迟/故障。
- 性能没有因为新增同步而明显回退；若回退，记录基线。

## 常见坑

- racecheck 通过不代表没有竞态：它主要覆盖共享内存，且依赖实际执行路径。
- 加 `torch.cuda.synchronize()` 后问题消失，可能只是把异步错误变成了同步错误，根因仍在。
- 用 `volatile` 或 `__threadfence()` 代替正确的同步设计，可能在不同架构上失效。
- 在 Triton 里手工加 `tl.debug_barrier()` 解决所有问题，会掩盖任务映射或原子使用错误。
- 只在一个 GPU 型号上验证；不同架构的 warp 调度和内存模型可能不同。

## 参考

- NVIDIA Compute Sanitizer（racecheck）：<https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html>
- CUDA C++ Programming Guide（同步与内存模型）：<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>
- Triton `debug_barrier`：<https://triton-lang.org/main/python-api/generated/triton.language.debug_barrier.html>
- PyTorch CUDA 语义：<https://pytorch.org/docs/stable/notes/cuda.html>
