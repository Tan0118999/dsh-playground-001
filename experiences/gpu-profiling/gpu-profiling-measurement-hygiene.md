---
name: gpu-profiling-measurement-hygiene
title: profiling 测量卫生：预热、时钟、profiler 开销和统计
description: 两次测量差很多、优化后无法证明收益、profiler 下比原生慢几倍、报告数字被质疑、时钟/温度导致降频时读这条。给出预热、多次/中位数、时钟、profiler 开销、单位和交叉验证的清单与报告模板。
category: methodology
tags: [measurement, warmup, variance, throttling, profiling-overhead, statistics]
symptoms:
  - 结果跳动大，不知道信哪个
  - profiler 下 kernel 慢很多
  - 优化后看不出端到端收益
  - 时钟/温度导致降频
  - 两次运行差 20% 但代码没变
stacks: [cuda, nsys, ncu, pytorch]
updated: 2026-11-02
---

# profiling 测量卫生

> 一句话经验：**没有 baseline 和误差范围的“优化”不可信；profiler 会改变被测对象，关键数字要用第二种方法复核。**

## 适用场景

- 性能数字波动大，无法判断优化是否有效；
- profiler 下的数字和真实运行差很多；
- 需要写一份别人可以复现的性能报告；
- 时钟/温度/功耗/多租户导致结果不可比。

不解决：具体瓶颈判读（见 [分层定位](gpu-profiling-triage.md)）；也不解决正确性。

## 症状与信号

- 同一份代码两次运行差 10%–30%；
- nsys/ncu 下 tiny kernel 慢 2–4 倍；
- 只跑一次就宣布“提速 30%”；
- 不记录 GPU/驱动/CUDA/框架版本；
- 用 `time.time()` 包异步 launch；
- GB/s 和 GiB/s 混用。

## 快速判断

1. 是否预热？第一次是否包含 JIT/模块加载/allocator？
2. 是否跑了 ≥5 次并报中位数/p95？
3. 是否固定输入 shape/dtype/batch/seed？
4. 是否记录时钟、温度、功耗、是否降频？
5. 是否用 CUDA events（而不是 wall clock）测 GPU？
6. 是否用两种方法交叉验证（profiler vs 关 profiler）？
7. 单位是否一致（GB/s vs GiB/s、TFLOP/s）？

## 根因

- **首次运行成本**：模块加载、JIT、cuBLAS 算法选择、allocator 首次分配；
- **时钟/功耗/温度**：boost 不稳定，笔记本/共享 GPU 更明显；
- **profiler 开销**：nsys 的采集、ncu 的 replay/清 cache/抬时钟；
- **统计方法**：单次、最大值、平均值掩盖长尾；
- **输入/环境变化**：shape、batch、精度、其他进程、MIG；
- **单位/口径**：GB vs GiB、active vs elapsed、峰值 vs 可达。

## 排查步骤

### 1. 预热

至少 2–5 次；或用 ncu `--launch-skip`、torch profiler 的 `schedule(wait, warmup, ...)`、nsys 的 capture range。

### 2. 多次运行

至少 5 次，报 **median** 和 **p95**；不报“最好的一次”。对比两个版本时交替 A/B/A/B，减少时间漂移。

### 3. GPU 计时用 events

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
start.record()
for _ in range(reps):
    run_once()
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end) / reps, "ms")
```

`time.time()` 包异步 launch 只测到“发射时间”。

### 4. 记录时钟/温度/功耗

```bash
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active \
  --format=csv
```

如果两次时钟不同，性能差异可能不是代码造成的。ncu 默认 `--clock-control boost` 会改变时钟；做结论用 `base`，并用关掉 profiler 的 CUDA events 复核。

### 5. profiler 开销

- nsys 对 tiny kernel 的开销尤其大：本仓 500 个 tiny kernel 原生 3–10 ms，nsys 下 4.7–11.5 ms；
- ncu 的 replay 会清 cache/抬时钟；
- 相对占比用 nsys/ncu，绝对时间用关掉 profiler 的 events；
- 两种方法冲突时，以更接近真实执行的方法为准，并说明差异。

### 6. 单位

- 带宽：GB/s = 10⁹ B/s；GiB/s = 2³⁰ B/s；
- 算力：TFLOP/s = 10¹² FLOP/s；
- 显存：GB vs GiB；
- 写报告时标清楚，避免“差了 7%”其实是单位问题。

## 验证

- 同输入、同 build、同测量方法；
- 至少 5 次，median/p95；
- kernel 指标 + 端到端都改善；
- 结论在误差范围之外；
- 换一台机器/一个版本仍能复现（或说明限制）。

## 报告模板

```markdown
# <优化项> 性能分析

## 结论
- 端到端：<before> → <after>（median，N 次）
- 关键 kernel：<before> → <after>
- 是否采纳：是/否；原因

## 环境
- GPU / 驱动 / CUDA / 框架 / 工具版本
- 输入 shape、dtype、batch、seed
- 时钟/温度/是否 MIG/容器

## 证据
- nsys：总时长、top kernel 占比、gap、copy/overlap
- ncu：SOL、occupancy、stall、sectors/request、duration
- 对照实验：只改了什么

## 分析与假设
- 现象 → 机制 → 假设 → 预期

## 改动
- 代码/参数/编译选项；为什么这样改

## 验证
- 正确性对拍
- 多次测量（median/p95）
- 端到端收益
- 回归风险

## 限制
- 适用 shape/架构/版本
- 未验证的部分
```

## 常见坑

| # | 误判 | 为什么错 | 怎么避免 |
| --- | --- | --- | --- |
| 1 | 只跑一次 | 时钟/温度/调度波动 | 5 次以上，median/p95 |
| 2 | 用 `time.time()` 测 GPU | 只测到 launch | CUDA events + synchronize |
| 3 | 把 nsys 下 tiny kernel 当真 | profiler 放大 | 关掉 profiler 复核 |
| 4 | 把 ncu replay 当真 | 清 cache/抬时钟 | `--clock-control base`、交叉验证 |
| 5 | 不同 batch/shape 直接比 | 工作量不同 | 固定输入 |
| 6 | 不同精度直接比 | TF32/FP16 峰值不同 | 标清 dtype |
| 7 | 忽略降频 | 时钟下降 | 记录 clocks/throttle |
| 8 | 用平均值掩盖长尾 | p95 可能差很多 | 同时报 p50/p95 |
| 9 | 忽略 allocator cache | reserved 高但 allocated 正常 | memory snapshot |
| 10 | kernel 快了端到端没变 | 被其他阶段吃掉 | 回 nsys 看占比 |

## 参考

- Nsight Systems User Guide: <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>
- Nsight Compute Profiling Guide: <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>
- CUDA C++ Best Practices Guide: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
- PyTorch Profiler: <https://pytorch.org/docs/stable/profiler.html>
