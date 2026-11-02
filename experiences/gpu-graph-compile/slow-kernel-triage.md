---
name: slow-kernel-triage
title: 慢 kernel 分层排查：端到端、访存、计算还是调度
description: 编译通过、结果正确但性能明显回退或波动时，先用 nsys 区分端到端瓶颈（同步、拷贝、启动、空洞），再用 ncu 指标判断 kernel 是访存、计算、延迟还是占用率受限，最后才调 tile 和 launch 配置。
category: performance
tags: [performance, nsight-systems, nsight-compute, profiling, memory-bound]
symptoms:
  - 能跑但比 baseline 慢很多，或性能波动大
  - 融合/cudagraph 开启后没有预期提升
  - GPU 利用率低、CPU 与 GPU 之间有空档
  - kernel 数量很多、每个都很短
  - 不确定该优化访存、计算还是 launch 开销
stacks: [torch.compile, triton, cuda, xla]
updated: 2026-11-02
---

# 慢 kernel 分层排查：端到端、访存、计算还是调度

> 一句话经验：先用时间线回答“时间花在哪”，再用计数器回答“这个 kernel 为什么慢”；顺序反过来容易优化错地方。

## 适用场景

- 模型/算子能正确运行，但比基线慢或抖动大。
- 怀疑编译融合、autotune、cudagraph、内存布局导致性能问题。
- 需要判断瓶颈在 CPU、GPU kernel、数据搬运还是同步。
- 需要在多个优化方案之间做有依据的选择。

## 症状与信号

- 端到端慢，但单个 kernel 时间正常：问题可能在 launch 次数、CPU 侧、同步、拷贝或空洞。
- kernel 时间慢，GPU 利用率低：可能 occupancy 低、stall 高、访存不合并。
- 首次慢、后续快：编译/autotune/模块加载，不是稳态性能问题。
- 性能随输入 shape 剧烈变化：重编译、尾部浪费、算法切换或 autotune 选择不同。
- 单次测量抖动大：时钟、功耗、其他租户、缓存状态或测量方法问题。

## 快速判断

1. **排除编译与预热**：先跑 5–10 次，取稳态；确认没有重编译（见[编译缓存与重编译](compile-cache-and-recompiles.md)）。
2. **看时间线**：用 nsys 看 CPU/GPU 重叠、同步、memcpy、kernel 间隙。
3. **看单 kernel**：用 ncu 看 SpeedOfLight、Occupancy、MemoryWorkloadAnalysis、WarpStateStats。
4. **判断瓶颈类型**：DRAM 高 → 访存；SM 高 → 计算；都低且 stall 高 → 延迟/占用率；kernel 多而短 → launch/调度。
5. **单变量优化**：一次只改一项，固定输入和测量方法。

## 排查步骤

### 1. 建立可比较的测量

```python
import torch, time

def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2], times[int(len(times) * 0.9)]

median, p90 = bench(lambda: model(x))
print(f"median {median:.3f} ms, p90 {p90:.3f} ms")
```

固定输入、dtype、batch、GPU、时钟状态；报告 p50/p90 而不是单次值。

### 2. 用 nsys 看端到端时间线

```bash
nsys profile --trace=cuda,nvtx,osrt -o report python3 run.py
nsys stats report.nsys-rep
```

关注：

- CPU 与 GPU 是否重叠，是否存在大段空洞（CPU 在准备数据/等锁）。
- 是否有频繁 `cudaMemcpy`/`cudaMemset`/同步。
- kernel 数量与单 kernel 时长；大量短 kernel 说明 launch 开销或融合不足。
- 多流/多进程是否真的并行，还是互相等待。
- DataLoader、NCCL、CPU 预处理是否成为瓶颈。

nsys 适合回答“什么时候、什么顺序”，不要用它判断单个 kernel 的微架构瓶颈。

### 3. 用 ncu 分析单 kernel

```bash
ncu --set full -o prof python3 run.py
# 或只开需要的 section，减少开销
ncu --section SpeedOfLight --section Occupancy \
    --section MemoryWorkloadAnalysis --section SchedulerStats \
    --section WarpStateStats -o prof python3 run.py
```

- 用 `--kernel-name regex:<name>` 聚焦目标 kernel，`--launch-count 1` 限制采集次数。
- 无权限时会报 `ERR_NVGPUCTRPERM`；需要给用户/容器授予 GPU performance counter 权限，或由管理员配置。
- ncu 会 replay/serialize kernel，**不能用 ncu 下的耗时判断端到端性能**，只能看计数器和比例。

关键指标：

| 指标 | 用途 |
| --- | --- |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 计算吞吐占峰值 |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM 带宽占峰值 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 实际占用率 |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | issue slot 利用率 |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | 等全局内存 |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | 共享内存/特殊单元压力 |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` | 全局 load 每 sector 字节数，低说明不合并 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | 共享内存 bank conflict |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | tensor core 利用率 |
| `launch__registers_per_thread` | 寄存器数，配合 spill 判断 |

### 4. 按瓶颈类型处理

| 观察 | 判断 | 优先动作 |
| --- | --- | --- |
| DRAM 吞吐高、SM 低 | 访存受限 | 合并访存、提高复用、减少中间 tensor、用更低精度/更紧凑布局 |
| SM 吞吐高、DRAM 低 | 计算受限 | 减少指令、用 tensor core、提高 ILP、避免分支发散 |
| 两者都低、stall 高 | 延迟受限 | 提高 occupancy/ILP、预取、减少依赖链 |
| 大量短 kernel、间隙大 | launch/调度受限 | 融合、CUDA Graphs、减少同步、批处理 |
| bank conflict 高 | 共享内存布局问题 | padding、改变访问模式 |
| tensor pipe 低但理论算力高 | 没有用上 tensor core 或 tile 不匹配 | 检查 dtype/shape/tile，确认走了 MMA |

### 5. 常见工程因素

- 动态 shape 导致重编译或 autotune 重新选择。
- 额外的 layout 转换（NCHW/NHWC、transpose、contiguous）和拷贝。
- 融合把大 kernel 拆成多个小 kernel，或融合后寄存器压力上升。
- cudagraph 未启用/回退导致 launch 开销高。
- 批处理太小，GPU 利用率天然低；尾部 batch 浪费。
- autotune 在真实输入分布上选了不适合的配置。

## 根因

1. **测量方法错误**：没有预热、没有同步、单次测量、在 profiler 下比较。
2. **端到端瓶颈不在 kernel**：CPU 预处理、同步、拷贝、launch 开销、通信。
3. **kernel 微架构瓶颈**：访存不合并、bank conflict、occupancy 低、spill、指令依赖。
4. **图层面问题**：未融合、重复计算、多余布局转换、动态 shape 重编译。
5. **环境因素**：时钟/功耗限制、MIG、其他租户、驱动版本差异。

## 修复与规避

优化顺序建议：

1. 先消除算法/图层面的浪费（融合、去掉多余拷贝、缓存中间结果）。
2. 再改数据布局和访存模式（合并、对齐、复用）。
3. 再调 tile/`num_warps`/`num_stages`/occupancy。
4. 最后处理 launch 开销（CUDA Graphs、批处理）。

每一步都保留基线和单变量对比；不要同时改多项。

## 验证

- 固定输入、预热、多次运行，报告中位数和 p90。
- 端到端与单 kernel 分开测量；ncu 数据只用于分析瓶颈，不用于性能结论。
- 覆盖真实输入分布（shape、batch、序列长度），确认不是只在测试尺寸上变快。
- 与 baseline 使用相同环境、相同精度、相同 batch；记录 GPU 型号和时钟状态。
- 正确性回归：性能优化不应改变结果。

## 常见坑

- 用 ncu 报告的 kernel 时间比较优化前后：replay 和 serialization 会改变时序。
- 只看 GPU 时间忽略 CPU 侧：数据加载和 Python 开销可能才是瓶颈。
- 测量前没有 `torch.cuda.synchronize()`，把异步 launch 当成执行完成。
- 把首次编译/autotune 时间算进稳态性能。
- 在共享 GPU/MIG 上不控制变量，结论不可复现。
- 过度追求 occupancy，忽略带宽/延迟的实际限制。

## 参考

- Nsight Systems User Guide：<https://docs.nvidia.com/nsight-systems/UserGuide/index.html>
- Nsight Compute Profiling Guide：<https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>
- Nsight Compute CLI：<https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html>
- PyTorch profiler：<https://pytorch.org/docs/stable/profiler.html>
