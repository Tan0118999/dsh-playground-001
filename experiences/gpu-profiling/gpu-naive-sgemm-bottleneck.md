---
name: gpu-naive-sgemm-bottleneck
title: naive SGEMM 只有峰值 6%：compute-bound 也该先看访存和复用
description: 手写 GEMM 吞吐远低于峰值、离 compute roof 很远、不知道该 tiling 还是换 Tensor Core、加 shared memory 后提升有限时读这条。用真实 SGEMM 数据分析指令/复用/occupancy，并给出验证顺序。
category: performance
tags: [gemm, matmul, sgemm, tiling, tensor-core, cublas, compute-bound]
symptoms:
  - GEMM 只有几 TFLOP/s
  - compute-bound 但 FMA 没打满
  - 加了 shared memory 提升有限
  - Tensor Core 改完结果不对
  - 不知道先 tiling 还是先换 dtype
stacks: [cuda, ncu, cublas]
updated: 2026-11-02
---

# naive SGEMM 只有峰值 6%

> 一句话经验：**“compute-bound”不等于 FMA 单元忙；naive GEMM 的瓶颈常常是 load/store 指令和片上复用不足，先看 Compute Workload Analysis/InstructionStats 再决定 tiling 还是 Tensor Core。**

## 适用场景

- 手写矩阵乘/卷积，吞吐远低于峰值；
- 不确定该做 shared-memory tiling、register blocking 还是 Tensor Core；
- 换了 Tensor Core 但结果/性能不对；
- 想知道“库为什么比手写快”。

不解决：库的选择/安装；也不解决数值精度定义。

## 症状与信号

- 1024×1024 FP32 GEMM 约 20–22 ms / 20 次，约 1.9–2.1 TFLOP/s；RTX 5070 FP32 峰值约 30.9 TFLOP/s，利用率 6%–7%；
- nsys 里 `matmul_naive` 占 GPU kernel 时间约 95%；
- 每次 kernel 约 1.03 ms，20 次背靠背，kernel 间 gap 1–4 µs → 不是 launch-bound；
- ncu 预期：Compute 不高、L1/LSU 较忙、load/store 指令占比高（待验证，本机无计数器）。

## 快速判断

1. 先确认不是 launch-bound（kernel 时长/数量）；
2. 算 GFLOPS：`2*N^3 / time`；
3. 看 Compute Workload Analysis 的 pipe 细分：FMA 是否接近 100%？
4. 看 InstructionStats：load/store 指令比例；
5. 看 Memory Workload Analysis：L1/L2 复用、sectors/request；
6. 再看 occupancy/寄存器。

## 根因

naive 版本每个线程算一个输出，循环 1024 次，每次从 global 读 A、B 两个元素：

- A：同一行的 16 个线程读同一个元素（broadcast，L1 友好）；一个 warp 覆盖两行；
- B：同一 k 的 16 个连续列被合并读取（coalesced），一个 warp 里有两组重复；
- 但每个线程仍要发 2048 次 load，load/store 指令比例高，FMA pipe 利用率低；
- 每个 block 只有 16×16 的输出复用，A/B 的片上复用不足。

结果：虽然算术上是 compute-bound，实际瓶颈更接近**指令吞吐 + 片上复用**，不是 FMA 峰值。

## 修复顺序

1. **shared-memory tiling**：每块加载一次 A/B tile，复用 32×32 次，显著减少 load 指令；
2. **register blocking**：每个线程算多个输出（如 4×4），提高 FMA/load 比；
3. **Tensor Core**：FP32 可用 TF32 Tensor Core（注意精度），或 FP16/BF16；
4. **用库**：cuBLAS/CUTLASS 已做上述优化，先对比库再决定是否手写。

每次只改一层，测一次。

## 排查步骤

```bash
# 只采目标 kernel，先 basic
ncu --set basic -k regex:matmul_naive -c 1 -o matmul_basic ./gpu_demo

# 需要看 pipe/指令
ncu --section ComputeWorkloadAnalysis --section InstructionStats \
    --section MemoryWorkloadAnalysis --section Occupancy \
    -k regex:matmul_naive -c 1 ./gpu_demo
```

本仓 demo 可直接复现：

```bash
cd toolbox/gpu-profiling-demo
./run_profile.sh
```

## 验证

- 统一用 `2*N^3 / time` 算 GFLOPS；
- 每次只改一层，记录前后；
- 对拍：与 CPU/参考实现比较，TF32/FP16 设容差；
- 回 nsys 确认端到端收益（本案例占约 95%，收益会明显）；
- 覆盖不同 N（非 2 的幂、非 32 倍数）。

## 常见坑

- 把“compute-bound”直接等同于“FMA 忙”；
- 一上来换 Tensor Core，精度/对齐没处理；
- tiling 后 occupancy 下降，收益被吃掉；
- 只测一个 N；
- 用 profiler 下的绝对时间当性能；
- 忽略库已经做过同样优化。

## 参考

- Nsight Compute Profiling Guide: <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>
- CUDA C++ Best Practices Guide: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
- 本仓 demo: [`toolbox/gpu-profiling-demo`](../../toolbox/gpu-profiling-demo/README.md)
