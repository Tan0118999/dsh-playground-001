---
name: nsight-systems-gpu-idle-and-launch-overhead
title: GPU 空转与 launch-bound：时间线 gap 怎么归因
description: GPU 利用率锯齿、时间线大片空白、成百上千个微秒级 kernel、cudaLaunchKernel 次数很多、GPU 行空但 CPU 很忙时读这条。区分 CPU launch、同步、dataloader、host 初始化，并给出 CUDA Graphs/融合的验证方法。
category: performance
tags: [nsys, gap, launch-overhead, cuda-graphs, dataloader, idle]
symptoms:
  - GPU 时间线周期性空白
  - 大量 kernel 只有几微秒
  - cudaLaunchKernel 次数很多
  - GPU 行空但 CPU 很忙
  - 用了 CUDA Graphs 但不知道有没有用
stacks: [nsys, cuda, pytorch]
updated: 2026-11-02
---

# GPU 空转与 launch-bound

> 一句话经验：**GPU 空闲时不要优化 kernel；先看同一时间 CPU/API 在做什么。GPU busy 时间远小于 NVTX 跨度，就是 launch/CPU 瓶颈。**

## 适用场景

- GPU 利用率呈锯齿状，step time 波动；
- 时间线上 CUDA HW 行有长段空白；
- 一个循环里 launch 很多小 kernel；
- GPU kernel 时间很短，但端到端很久。

不解决：kernel 内部效率（见 [Nsight Compute 第一轮](nsight-compute-first-pass.md)）；也不替代 CPU 代码优化。

## 症状与信号

- CUDA HW 行空白，CPU 线程 running；
- `nsys stats` 里 kernel 总时间远小于 wall time；
- `cudaLaunchKernel` 数百/数千次，单次几微秒；
- NVTX range 跨度 10 ms，但 kernel 合计 0.25 ms；
- 多流/异步拷贝没有带来收益。

## 快速判断

1. 算比例：kernel 总时间 / wall time；如果 <50%，先看 gap；
2. 找最大 gap，看同一时间的 CUDA API 行：
   - 长 `cudaMalloc`/`cudaHostAlloc` → host 初始化；
   - 长 `cudaDeviceSynchronize`/`cudaStreamSynchronize` → 同步等待；
   - 密集 `cudaLaunchKernel` → launch-bound；
3. 看 CPU 线程：Python/数据预处理/锁；
4. 看 kernel 数量：是不是成百上千个小块；
5. 看 gap 是否周期性：dataloader/日志/周期同步。

## 根因

### 1. host 初始化/分配

首次 `cudaMalloc`、`cudaMallocHost`、stream 创建、模块加载、JIT 都会让 GPU 空闲。本仓 demo 中，P4 到 P5 之间有约 29 ms 空白，其中 `cudaMalloc` 单次 134 ms（WSL 下尤其慢）、`cudaHostAlloc` 18.8 ms。这些是**一次性/低频**成本，应移出测量区间或复用 buffer。

### 2. launch overhead

每次 kernel launch 有 CPU/驱动开销（几微秒到几十微秒）。500 个 0.5 µs 的 kernel，真正 GPU 时间只有 0.25 ms，其余都是 launch 和 profiler 开销。本仓多次采集：原生 3–10 ms，nsys 下 4.7–11.5 ms。

### 3. 同步点

`.item()`、`.cpu()`、`print`、`cudaDeviceSynchronize`、`cudaStreamSynchronize` 都会让 CPU 等 GPU，或让 GPU 等 CPU；时间线上表现为 gap 前后有长 API 调用。

### 4. 数据供给不足

dataloader/预处理/IO 让 GPU 等数据；CPU 线程 running，GPU 行空。用 [PyTorch Profiler](pytorch-profiler-and-memory-snapshot.md) 或 py-spy 确认。

## 排查步骤

```bash
# 1. 看 kernel 总时间和 top
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum report.nsys-rep

# 2. 看 API 调用次数和平均时间
nsys stats --report cuda_api_sum report.nsys-rep

# 3. 只看一段，避免启动阶段干扰
nsys profile --trace=cuda,nvtx --capture-range=nvtx --capture-range-end=stop -o roi ./app
```

在时间线上：

- 空白期间 CPU 在 running？→ CPU-bound/dataloader；
- 空白期间 CPU 在 sleeping？→ 同步/等待；
- 空白前后是 `cudaMalloc`/`cudaHostAlloc`？→ 初始化；
- 空白期间 API 行非常密？→ launch-bound。

## 修复与规避

### 减少 launch 次数

- **融合**：把多个小 kernel 合成一个；
- **batch**：一次处理更多任务；
- **persistent kernel**：一次 launch 处理多轮；
- **CUDA Graphs**：捕获后重放，降低 launch 开销。

### 处理 host 初始化

- 预热，把初始化移出计时区间；
- 复用 buffer，避免每步 `cudaMalloc`；
- pinned memory + 异步拷贝；
- 区分“首次启动”和“稳态”。

### 处理同步/数据

- 去掉不必要的 `.item()`/`.cpu()`/print；
- dataloader 多 worker、prefetch、pinned memory；
- 预处理搬到 GPU；
- 用 NVTX 包住 dataloader/forward，确认 gap 归属。

## 验证

- `cudaLaunchKernel` 次数/总时间下降；
- NVTX 跨度与 GPU busy 比值接近 1；
- 时间线 gap 消失或变短；
- 端到端时间下降，且输出正确；
- CUDA Graphs 场景下，确认捕获后的 graph 重放没有改变语义。

## 常见坑

- **把 profiler 下的小 kernel 时间当真**：nsys 会放大 launch/采集开销；
- **只看 kernel 平均时间，不看数量**：500 × 0.5 µs 也是负担；
- **用 CUDA Graphs 但每步重新捕获**：捕获本身有开销；
- **把 host 初始化算进稳态**：首次 100 ms 不代表每次都慢；
- **多流反而更慢**：依赖/资源竞争，见 [多流 overlap](gpu-multi-stream-overlap.md)。

## 参考

- Nsight Systems User Guide: <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>
- CUDA C++ Best Practices Guide（launch/stream）: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
- 本仓 demo: [`toolbox/gpu-profiling-demo`](../../toolbox/gpu-profiling-demo/README.md)
