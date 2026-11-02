---
name: npu-to-gpu-profiling-mapping
title: 从 NPU 到 GPU：profiling 概念对照与迁移陷阱
description: 有昇腾/CANN 优化经验、第一次看 GPU profile，不知道 SM/warp/occupancy 与 AI Core/UB/流水线如何对应时读这条。给出概念对照、不能直接类比的点和推荐阅读顺序，帮助把 NPU 经验安全地迁移到 GPU。
category: methodology
tags: [npu, ascend, cuda, sm, warp, occupancy, mental-model]
symptoms:
  - 只会看 msprof/MindStudio Insight，不知道 Nsight 从哪看起
  - 用 NPU 的流水线/tiling 思维调 GPU，越调越慢
  - 不理解 SM、warp、occupancy、sector 这些词
  - 想把 AI Core/UB 的经验直接套到 GPU
stacks: [cuda, npu, nsys, ncu]
updated: 2026-11-02
---

# 从 NPU 到 GPU：profiling 概念对照与迁移陷阱

> 一句话经验：**GPU 靠大量 warp 做延迟隐藏，NPU 更依赖显式流水和片上存储编排；两者都怕 HBM，但“喂饱计算单元”的方式不同。**

## 适用场景

- 你做过昇腾/CANN 或其他 NPU 的算子优化，熟悉 tiling、双缓冲、UB、流水；
- 第一次读 GPU profile，看到 `SM`、`warp`、`occupancy`、`sector` 不知道对应什么；
- 想把“减少搬运、提高复用”的经验迁移到 GPU，但不确定哪些能直接套。

不解决：具体某个 kernel 的瓶颈（用 [Nsight Compute 第一轮](nsight-compute-first-pass.md)），也不替代 CUDA 编程模型文档。

## 症状与信号

- 看到 `nvidia-smi` 100% 就以为 GPU 很忙；
- 以为“occupancy 越高越好”，把所有 kernel 都去调 block/寄存器；
- 按 NPU 的流水级数去理解 GPU 的 warp 切换；
- 把 UB 容量和 shared memory 容量直接比较；
- 优化了某个 kernel，端到端却没变（见 [分层定位总纲](gpu-profiling-triage.md)）。

## 快速判断

1. 先问：这个 kernel 是 **memory-bound、compute-bound 还是 latency-bound**？不要先问“AI Core 对应哪个”；
2. 看它占端到端多少时间；占比很小就先别优化；
3. 看 GPU 用什么方式隐藏延迟：有足够 warp/ILP 吗？
4. 看数据怎么搬：合并了吗？复用了吗？sector 放大了吗？
5. 最后才看具体硬件单元（Tensor Core/FP32 pipe）的利用率。

## 根因：三个必须换掉的观念

### 1. 延迟隐藏：从“显式流水”到“海量 warp”

NPU 上你习惯把数据切成 tile，用双缓冲把搬运和计算重叠，靠流水线藏 HBM 延迟。流水线是**显式**的。

GPU 的主流方式是**线程级并行（TLP）**：SM 上同时驻留很多 warp，某个 warp 等内存时，调度器切到另一个 ready 的 warp。你不需要手写双缓冲（也可以），但需要**足够的并发 warp**和**足够好的访存**，否则调度器没有东西可切。

这解释了：

- 为什么 `occupancy`（驻留 warp 比例）在 GPU 上是常用指标；
- 为什么“计算量不大但访存很散”的 kernel 会慢：warp 都在等内存，能切换的 warp 不够。

### 2. 执行模型：SIMT 的“32 个线程一组”

GPU 以 **warp**（NVIDIA 为 32 线程）为调度单位。一个 warp 的 32 个线程执行同一条指令（SIMT），但可以有各自的寄存器和分支路径；分支发散时不同路径串行执行。

你写的 `threadIdx` 是线程视角，硬件看到的是 warp。**很多指标按 warp 或 SMSP（SM sub-partition）统计**，不是按线程。

### 3. 存储层级：scratchpad 与 cache 的差别

- **shared memory / L1**：每个 SM 一块；shared memory 由软件显式管理（类似 scratchpad），L1 由硬件自动缓存；
- **L2**：全 GPU 共享，硬件管理；
- **HBM/GDDR**：全局显存，带宽是大多数 kernel 的硬上限。

GPU 的 L1/L2 是**自动缓存**，你无法像 UB 那样精确控制“哪块数据一定在片上”；能控制的是访问顺序、合并程度、shared memory 复用。分析时因此要看 **hit rate + sectors/request**，而不只是“有没有放进 shared memory”。

## 概念对照表

再强调：这是**概念对照**，不是一一等价。

| NPU 概念（以昇腾为例） | GPU 概念 | 相同点 | 关键差异 |
| --- | --- | --- | --- |
| AI Core | SM | 都是“计算簇 + 本地存储 + 调度” | SM 内是大量细粒度 lane + warp 调度；AI Core 的 Cube/Vector 分工更显式 |
| Cube 单元 | Tensor Core | 都做矩阵乘加速 | Tensor Core 通过 MMA/WGMMA 等指令使用（warp 或 warpgroup 级），shape/dtype/对齐要求不同 |
| Vector 单元 | CUDA core（FP32/INT pipe） | 都做逐元素/向量运算 | GPU lane 更细，依赖 warp 级 SIMT |
| Unified Buffer（UB） | shared memory + L1 | 都是片上 SRAM | UB 是软件管理的 scratchpad；GPU shared mem 按 CTA 分配且有 bank；L1 是自动 cache |
| HBM | HBM / GDDR | 都是全局带宽瓶颈 | 分析思路一致：bytes、sector、hit rate |
| 手工 tiling / 双缓冲 | shared-memory tiling / async copy / TMA | 都为复用和重叠 | GPU 更多依赖编译器调度和 warp 并发；TMA 是 Hopper 之后的异步搬运路径 |
| Stream | CUDA stream | 都是任务队列 | 语义接近；GPU 还有 copy engine 与 kernel 并发 |
| `msprof` / MindStudio Insight | nsys + ncu | 都分“时间线”和“单算子” | GPU 的 kernel replay、源码/SASS 关联更细 |
| 图编译 / GE | CUDA Graphs / 编译器融合 | 都减少 launch 和调度开销 | 不是同一个抽象层，不能直接类比 |

**最不该直接类比的两件事：**

1. 不要把“AI Core 数量 × 单核算力”直接映射成“SM 数量 × 单 SM 算力”。峰值算力的口径（dtype、稀疏、Tensor Core 代际）不同。
2. 不要把 UB 容量当作 shared memory 容量。UB 的共享范围、bank、生命周期和 GPU shared memory 都不一样；GPU 还有 L1/L2 自动缓存这一层。

## 推荐阅读顺序

1. 本文：建立概念地图；
2. [分层定位总纲](gpu-profiling-triage.md)：知道先看哪一层；
3. [Nsight Systems 时间线](nsight-systems-timeline-reading.md)：读端到端；
4. [Nsight Compute 第一轮](nsight-compute-first-pass.md)：读单 kernel；
5. [roofline 与 occupancy](gpu-roofline-and-occupancy.md)：判断还有多少空间。

## 验证

- 能用自己的话解释：为什么 occupancy 低可能慢、为什么 `nvidia-smi` 100% 不代表效率高、为什么“减少字节”对 memory-bound kernel 有效；
- 拿一个自己的 kernel，能判断它更接近 memory-bound 还是 compute-bound，并说出下一步用哪个工具；
- 不再用“AI Core 数量 × 单核算力”估算 GPU 峰值。

## 常见坑

- **把 NPU 的“搬运/计算”二分直接当 GPU 的 memory/compute 二分**：GPU 还有 latency-bound 这一大类，表现为两者都不高；
- **把 UB 当 shared memory**：容量、bank、生命周期都不同；
- **把 CUDA core 当 Vector 单元一一对应**：SIMT 的执行粒度不同；
- **忽略 warp 调度**：只按线程数算并行度，忽略一个 block 内 warp 的调度和发散。

## 参考

- CUDA C++ Programming Guide: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>
- CUDA C++ Best Practices Guide: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
- Nsight Compute Kernel Profiling Guide: <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>
- 本仓实验（真实数据）: [`toolbox/gpu-profiling-demo`](../../toolbox/gpu-profiling-demo/README.md)
