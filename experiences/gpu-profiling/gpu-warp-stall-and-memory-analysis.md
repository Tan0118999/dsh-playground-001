---
name: gpu-warp-stall-and-memory-analysis
title: warp stall 与访存效率：long scoreboard、sectors/request、bank conflict
description: Nsight Compute 里 long scoreboard/barrier/math throttle 很高、带宽上不去、global load sectors/request 远高于 4、怀疑非合并访存或 shared bank conflict、transpose 或 attention 访存效率差时读这条。
category: performance
tags: [ncu, warp-stall, coalescing, sectors, shared-memory, bank-conflict, transpose]
symptoms:
  - long scoreboard 占比高
  - global load sectors/request 是 16
  - shared bank conflict 高
  - 带宽只有峰值的 30%
  - transpose/attention 的 HBM 流量远大于有效数据
stacks: [ncu, cuda]
updated: 2026-11-02
---

# warp stall 与访存效率

> 一句话经验：**stall 是症状不是根因；long scoreboard 高时先看 sectors/request 和 hit rate，不要盲目加 occupancy。**

## 适用场景

- SOL 显示 memory 高或 compute/memory 都低；
- Warp State 里某一类 stall 占比很高；
- 带宽利用率低，怀疑非合并访存；
- transpose/attention 等访存密集 kernel 明显慢。

不解决：occupancy 限制项和 roofline 分类（见 [roofline 与 occupancy](gpu-roofline-and-occupancy.md)）；端到端（见 [Nsight Systems](nsight-systems-timeline-reading.md)）。

## 症状与信号

- `long scoreboard` 高：等 global/local 内存；
- `short scoreboard` 高：等 shared/MIO/特殊函数；
- `math_pipe_throttle` 高：计算 pipe 忙；
- `barrier` 高：等 `__syncthreads`；
- `not_selected` 高：warp 很多但没被选中；
- global load `sectors/request` 远高于 4。

## 快速判断

1. 看 stall 第一名；
2. 如果是 `long_scoreboard`，先看 Memory Workload Analysis：
   - `sectors/request` 理想值取决于访问宽度（32×4B 连续 = 4）；
   - L1/L2 hit rate；
   - DRAM bytes 是否远大于“有效数据”；
3. 如果 `sectors/request` 明显偏高 → 非合并访存；
4. 如果 sectors 正常但带宽低 → 可能是 occupancy/ILP/延迟；
5. 如果 `math_pipe_throttle` → 看 Compute Workload Analysis 哪个 pipe 满；
6. 如果 `barrier` → 看 block 内负载是否不均。

## 根因

### 1. 32 B sector 与合并访存

GPU 的 L1/L2 以 **32 B sector** 为传输粒度。一个 warp 的 32 个线程各读 1 个 `float`（4 B）：

- **连续地址**：32×4 B = 128 B = 4 个 sector → 1 次请求、4 sector，利用率 100%；
- **跨步 32 个 float（128 B）**：每个线程落在不同 128 B 行 → 最多 32 个 sector（1024 B）只取回 128 B 有用数据 → **8× 放大**。

所以“我只读了 4 B”不代表“只搬了 4 B”。

### 2. 复用差

如果 L2 hit 低、DRAM bytes 接近“每次重新读”，说明数据被反复从 HBM 搬。用 tiling/shared memory/调整循环顺序提高复用。

### 3. shared memory bank conflict

同一个 bank 被多个线程同时访问会串行化。`[32][33]` 的 padding 是经典修法；但 padding 会改变 shared mem 占用，可能影响 occupancy。

### 4. local memory / spill

寄存器不足时变量落到 local memory，表现为 local load/store 高、寄存器限制明显。降 register pressure 或改 block，但小心 spill 和 occupancy 的权衡。

## 排查步骤

```bash
# 只采与 stall/内存相关的 section
ncu --section SpeedOfLight --section WarpStateStats \
    --section MemoryWorkloadAnalysis --section Occupancy \
    -k regex:my_kernel -c 1 ./app

# 关键指标
ncu --metrics \
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
dram__bytes_read.sum,dram__bytes_write.sum \
    -k regex:my_kernel -c 1 ./app
```

计算：

```
sectors/request = l1tex__t_sectors... / l1tex__t_requests...
有效带宽 = (dram__bytes_read + dram__bytes_write) / duration
```

## 真实案例：transpose 的写跨步

`transpose_naive` 读连续、写跨步：warp 的 32 个线程写同一列，地址间隔 `N*4 = 4 KiB`，每个线程落在不同 sector。结果：

- global store sectors/request 高；
- DRAM write bytes 远大于有效 4 MiB；
- 有效带宽约 220–228 GB/s。

改成 shared-memory tiled transpose（`tile[32][33]`）后，读写都连续，sectors/request 回到理想值，带宽约 387–622 GB/s，中位数约 2×。

```cuda
__global__ void transpose_tiled(const float* in, float* out, int n) {
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (x < n && y < n) tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    if (x < n && y < n) out[y * n + x] = tile[threadIdx.x][threadIdx.y];
}
```

注意：重复运行时 4 MiB 矩阵可能驻留 L2，折算出的“带宽”会偏高；做 HBM 结论要用 cold-cache 或 ncu 的 DRAM bytes。

## 修复与规避

| 问题 | 修复 | 代价 |
| --- | --- | --- |
| 非合并访问 | 调整 thread→data 映射、float4、转置/分块 | 需要改索引；边界处理 |
| 复用差 | tiling、shared memory、循环顺序 | shared 占用/occupancy |
| bank conflict | padding、改变 stride | shared 占用变化 |
| spill | 降寄存器、改 block、`__launch_bounds__` | 可能 spill/占用变化 |
| 对齐不足 | 128/64-bit 向量 load，保证对齐 | 边界/尾部处理 |
| 延迟高 | 提 occupancy 或 ILP/预取 | 资源竞争 |

先做“减少字节/合并访存”，再考虑 occupancy；顺序反了会白忙。

## 验证

- `sectors/request` 回到理想值；
- DRAM bytes 接近理论最小值；
- 有效带宽提升，且端到端改善；
- 用非 32 倍数/非对齐 shape 测试边界；
- 正确性对拍。

## 常见坑

- 把 `sectors/request` 4 当成所有访问的理想值（16 B 向量 load 可能是 8）；
- 只看 L1 hit rate：流式访问低是正常的；
- 加 shared memory 后 occupancy 下降，收益被吃掉；
- padding 改变 shared mem 大小，影响 block 数；
- 用 profiler 下 warm cache 的带宽当 HBM 带宽。

## 参考

- Nsight Compute Profiling Guide: <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>
- CUDA C++ Best Practices Guide（coalescing/shared）: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
- 本仓 demo: [`toolbox/gpu-profiling-demo`](../../toolbox/gpu-profiling-demo/README.md)
