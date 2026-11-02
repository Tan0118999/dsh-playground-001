---
name: register-spill-occupancy
title: 寄存器溢出与占用率：tile 调大反而更慢
description: kernel 编译通过但变慢，ptxas 报告 spill stores/loads，或 occupancy 低、寄存器限制占用率时，用 cuobjdump -res-usage、ptxas -v、nsight compute 指标确认是否溢出，再通过缩小 tile、减少活跃变量和拆分 kernel 修复。
category: performance
tags: [registers, spill, occupancy, ptxas, sass, nsight-compute]
symptoms:
  - ptxas 输出非零 spill stores / spill loads
  - 调大 tile/BLOCK 后性能反而下降
  - nsight compute 显示 occupancy 受寄存器限制
  - SASS 中出现大量 LDL/STL（local memory）
  - kernel 理论算力够，但吞吐上不去
stacks: [cuda, triton, torch.compile]
updated: 2026-11-02
---

# 寄存器溢出与占用率：tile 调大反而更慢

> 一句话经验：先看 spill 和寄存器限制的 occupancy，再决定“缩 tile”还是“换算法”；不要只看 occupancy 数字，带宽受限时高占用率未必更快。

## 适用场景

- 手写 CUDA/Triton kernel 或编译器生成的 kernel 比预期慢。
- 调大 tile/block/`num_stages` 后性能不升反降。
- 怀疑寄存器压力、local memory 溢出或 occupancy 不足。
- 需要判断该优化访存、减少重算，还是拆分 kernel。

## 症状与信号

- `ptxas` 输出 `N bytes spill stores, M bytes spill loads` 且非零。
- `cuobjdump -res-usage` 显示 `STACK`/`LOCAL` 非零。
- SASS 中出现 `LDL`/`STL`（local load/store），而预期是寄存器操作。
- Nsight Compute 中 `launch__occupancy_limit_registers` 是主要限制项。
- 同一条 kernel 在小 tile 下更快，大 tile 下更慢。
- 理论 FLOPs/带宽利用率都不高，warp stall 集中在 long scoreboard / math pipe throttle。

## 快速判断

1. **看编译报告**：`ptxas -v` 或 `cuobjdump -res-usage` 确认寄存器数、spill、shared memory。
2. **看受限于什么**：Nsight Compute 的 `launch__occupancy_limit_*` 指出寄存器/共享内存/warp/block 哪个先到上限。
3. **看实际占用率**：`sm__warps_active.avg.pct_of_peak_sustained_active`，不要只看理论值。
4. **看是否访存受限**：`dram__throughput.avg.pct_of_peak_sustained_elapsed` 高时，先优化访存，而不是继续压寄存器。
5. **做单变量实验**：只改 tile 或 `num_stages`，固定输入和 launch 配置，测中位数。

## 排查步骤

### 1. 读取寄存器与 spill

CUDA C++ 编译时：

```bash
nvcc -arch=sm_90 -cubin -Xptxas -v kernel.cu -o kernel.cubin
```

典型输出（措辞随版本略有差异）：

```text
ptxas info    : Compiling entry function '_Z6kernelPf' for 'sm_90'
ptxas info    : Function properties for _Z6kernelPf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 64 registers, used 1 barriers, 4096 bytes smem
```

- `spill stores/loads` 非零：寄存器不够，值被放到 local memory。
- `stack frame` 非零：也可能来自局部数组或函数调用。

聚合查看：

```bash
cuobjdump -res-usage kernel.cubin
# REG:64 STACK:0 SHARED:4096 LOCAL:0 ...
```

注意 `cuobjdump` 的 `SHARED` 可能包含驱动保留字节，和 `ptxas` 报告的静态 smem 不完全一致；以 `ptxas` 看静态占用，以 `-res-usage` 看整体。

Triton 生成物：

- 缓存目录（默认 `~/.triton/cache`，可用 `TRITON_CACHE_DIR` 覆盖）里找 `.cubin`，对它们跑 `cuobjdump -res-usage` / `nvdisasm`。
- 也可以从 `triton.compile(...)` 的 `CompiledKernel.asm["cubin"]` 取出二进制再分析。
- Triton 调用的 ptxas 默认带 `-v`，寄存器数和 spill 通常已经出现在 ptxas 日志里；用 `TRITON_DUMP_PTXAS_LOG=1` 保留日志。也可以对缓存里的 `.cubin` 执行 `cuobjdump -res-usage`/`nvdisasm` 做静态检查。

### 2. 判断 occupancy 限制

```bash
ncu --section Occupancy --section LaunchStats -o prof python3 run.py
```

关键指标：

| 指标 | 含义 |
| --- | --- |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 实际达成的占用率（活跃 warp 占峰值） |
| `launch__registers_per_thread` | 每线程寄存器数 |
| `launch__occupancy_limit_registers` | 寄存器允许的最大占用率/上限 |
| `launch__occupancy_limit_shared_mem` | 共享内存允许的上限 |
| `launch__occupancy_limit_warps` | warp 数上限 |
| `launch__occupancy_limit_blocks` | block 数上限 |
| `launch__shared_mem_per_block_static` / `_dynamic` | 每 block 静态/动态共享内存 |

如果 `launch__occupancy_limit_registers` 明显低于其他项，且实际占用率接近该上限，优化寄存器才有意义。

### 3. 判断是否值得降寄存器

```bash
ncu --section SpeedOfLight --section WarpStateStats --section SchedulerStats -o prof python3 run.py
```

- `dram__throughput.avg.pct_of_peak_sustained_elapsed` 很高：访存受限，先看合并访存和复用，不要盲目降寄存器。
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` 很高：计算受限，继续提高 ILP/减少指令。
- `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` 高：在等内存，增加并发或预取。
- `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` 高：共享内存/特殊功能单元压力。
- `smsp__issue_active.avg.pct_of_peak_sustained_active` 低且 stall 高：延迟受限，occupancy 或 ILP 不足。

### 4. 降低寄存器压力

按优先级：

1. **缩小 tile/BLOCK**：最直接，减少同时活跃的元素和累加器。
2. **降低 unroll/`num_stages`**：流水线更深会同时保留更多中间值。
3. **减少活跃变量**：合并临时变量、缩短生命周期、避免一次性加载整个 tile。
4. **用 32 位索引**：64 位地址运算会占用成对寄存器；确认元素数不会溢出时用 `int`/`unsigned`。
5. **重算代替保存**：代价低的地址/索引可以重算，省寄存器。
6. **拆分 kernel**：把寄存器压力大的阶段拆出去，用额外一次读写换寄存器。
7. **`__launch_bounds__` / `maxrregcount`**：限制每线程寄存器数，迫使编译器 spill 或优化；这是权衡，不是免费优化。

Triton 中对应：减小 `BLOCK_SIZE`、`num_warps`、`num_stages`，把大 kernel 拆成多个小 kernel。

### 5. 检查 SASS 确认 spill

```bash
cuobjdump -sass kernel.cubin | grep -E "LDL|STL|LDG|STG" | head
nvdisasm -plr kernel.cubin   # 查看寄存器 live range
```

- 出现 `LDL`/`STL` 基本可确认走了 local memory。
- `nvdisasm -plr` 可看每条指令的寄存器活跃区间，帮助定位“哪个值活得太久”。

## 根因

1. **tile 太大**：每个线程承担过多元素，累加器/中间结果超过寄存器预算。
2. **循环展开/流水线过深**：同时活跃的 load 和中间值增多。
3. **64 位索引与地址计算**：地址寄存器成对占用。
4. **复杂控制流/函数调用**：编译器难以复用寄存器，产生栈帧。
5. **共享内存换寄存器失败**：把数据放 shared memory 需要额外地址/同步开销，未必更省。
6. **架构差异**：不同 sm_XX 的寄存器文件、warp 调度和指令吞吐不同，同一配置表现不同。

## 修复与规避

- 先做单变量实验：只改 tile 或 `num_stages`，记录寄存器数、spill、occupancy 和耗时。
- 找到“性能拐点”而不是追求 occupancy 最大：带宽受限时降低 occupancy 反而可能减少 cache 抖动。
- 用 `__launch_bounds__` 时同时测 spill：限制过严会把压力转移到 local memory，可能更慢。
- 如果 spill 无法消除，评估“重算 vs 读 local”的代价，选择更便宜的一侧。

## 验证

- `ptxas`/`cuobjdump` 确认 spill 为 0 或显著下降，寄存器数符合预期。
- Nsight Compute 确认 occupancy 提升且 stall 结构改善。
- 固定输入、预热后取多次运行中位数，报告 p50/p90；不要用单次测量。
- 覆盖不同 shape/批次，确认不是只在某个尺寸上变快。
- 数值结果与修改前一致（tile 变化不应改变语义）。

## 常见坑

- 只看 occupancy：占用率高但访存不合并的 kernel 仍然慢。
- 只看寄存器数：寄存器多不一定是问题，spill 和实际 occupancy 才是信号。
- 用 `maxrregcount` 强行压低寄存器，导致大量 local memory 访问。
- 在 Nsight Compute 下比较性能：ncu 会 replay/serialize kernel，耗时不能代表真实端到端。
- 忽略 clock/power/温度和其他租户干扰；测量前固定 GPU 状态并重复。
- Triton 缓存命中旧 cubin，分析的是旧版本产物；改代码后用 `TRITON_ALWAYS_COMPILE=1` 或换缓存目录。

## 参考

- CUDA Binary Utilities（`cuobjdump`/`nvdisasm`）：<https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html>
- Nsight Compute Occupancy section：<https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>
- CUDA C++ Programming Guide（寄存器与 launch bounds）：<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launch-bounds>
- Triton `num_warps`/`num_stages`：<https://triton-lang.org/main/python-api/triton.html>
