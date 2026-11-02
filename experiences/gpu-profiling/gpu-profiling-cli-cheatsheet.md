---
name: gpu-profiling-cli-cheatsheet
title: GPU profiling 命令与指标速查：nsys、ncu、torch、nvidia-smi
description: 需要一条可复制的 nsys/ncu/torch profiler/nvidia-smi 命令、常见指标名、阈值或“症状对应先看什么”时读这条。包含采集、统计、报告导出、指标表和快速判读阈值。
category: performance
tags: [nsys, ncu, pytorch, nvidia-smi, metrics, cheatsheet, commands]
symptoms:
  - 忘了 nsys stats 的报告名
  - ncu 指标名是什么
  - 怎么只采一个 kernel
  - 压测要看哪些 nvidia-smi 字段
  - 只有 10 分钟，想按清单走
stacks: [nsys, ncu, pytorch, cuda]
updated: 2026-11-02
---

# GPU profiling 命令与指标速查

> 一句话经验：**先用 nsys 找“优化谁”，再用 ncu 找“为什么”，最后用 CUDA events 复核绝对时间。**

## 适用场景

- 需要可复制的命令；
- 忘了 report/metric 名称；
- 想要阈值和症状表；
- 做一次快速排查。

## 快速判断

| 你要做什么 | 用哪条 |
| --- | --- |
| 端到端/找 gap | Nsight Systems |
| 单 kernel 瓶颈 | Nsight Compute |
| PyTorch 算子/shape/显存 | PyTorch Profiler |
| 健康/降频/拓扑 | `nvidia-smi`/DCGM |
| CPU 侧 | `py-spy`/`perf` |
| 不采 profiler 的绝对时间 | CUDA events |

## Nsight Systems

```bash
# 采集
nsys profile --trace=cuda,nvtx,osrt --sample=process-tree --cpuctxsw=process-tree \
  --force-overwrite=true -o report ./app

# 只采一段 NVTX
nsys profile --trace=cuda,nvtx --capture-range=nvtx --capture-range-end=stop -o roi ./app

# 采集并直接出统计
nsys profile --trace=cuda,nvtx --stats=true -o report ./app

# 统计
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum report.nsys-rep

# 导出 SQLite / 专家规则
nsys export --type sqlite -o report.sqlite report.nsys-rep
nsys analyze report.nsys-rep
```

| report | 内容 |
| --- | --- |
| `cuda_gpu_kern_sum` | 按 kernel 聚合的 GPU 时间 |
| `cuda_gpu_mem_time_sum` | memcpy/memset 时间 |
| `cuda_api_sum` | CUDA API 调用时间 |
| `nvtx_sum` | NVTX range 汇总 |
| `cuda_gpu_trace` | 逐个 kernel 明细 |
| `mpi_event_sum` / `mpi_event_trace` | MPI（如启用） |

## Nsight Compute

```bash
# 第一轮
ncu --set basic -k regex:my_kernel -c 1 -o report ./app

# 指定 section
ncu --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
  -k regex:my_kernel -c 1 ./app

# 指定指标
ncu --metrics gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
  -k regex:my_kernel -c 1 ./app

# 跳过预热
ncu --set detailed -k regex:my_kernel -s 5 -c 1 ./app

# 读报告
ncu -i report.ncu-rep --page details --print-details all
ncu -i report.ncu-rep --page source --print-source sass
ncu -i report.ncu-rep --page raw --csv

# 查本机元数据
ncu --list-sets
ncu --list-sections
ncu --query-metrics --chip <chip>
```

## PyTorch Profiler

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./log"),
    record_shapes=True, profile_memory=True,
) as prof:
    for step, batch in enumerate(loader):
        if step >= 6: break
        train_step(batch)
        prof.step()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")
```

```bash
tensorboard --logdir ./log
```

## 监控 / CPU

```bash
nvidia-smi
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,\
temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv -l 1
nvidia-smi dmon -s pucvmet -d 1
nvidia-smi topo -m
nvidia-smi -q -d CLOCK,PERFORMANCE,POWER

dcgmi dmon
dcgmi diag -r 2

py-spy record -o profile.svg --pid <pid> --duration 30
perf record -F 99 -g -- python train.py && perf report
```

## CUDA events（关掉 profiler 的绝对计时）

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

## 指标

| 指标 | 含义 |
| --- | --- |
| `gpu__time_duration.sum` | kernel 时长（ns） |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | compute（SM）占峰值 |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM 占峰值 |
| `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | L1TEX 占峰值 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | achieved occupancy |
| `smsp__warps_eligible.avg.per_cycle_active` | 每周期 eligible warp 数 |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | issue slot 利用率 |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | long scoreboard 强度 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | global load sector 数 |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | global load 请求数 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` | global store sector 数 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | shared load bank conflict |
| `dram__bytes_read.sum` / `dram__bytes_write.sum` | DRAM 读写字节 |

> 指标名随架构/版本变化；用 `ncu --query-metrics --chip <chip>` 核对。

## 阈值（起点，不是结论）

| 指标 | 经验阈值 | 误读 |
| --- | --- | --- |
| DRAM/memory throughput | >80% 接近带宽瓶颈；>85% 余量小 | L1 高不等于 DRAM 高 |
| Compute throughput | >80% 接近计算瓶颈 | 可能只是某个 pipe 忙 |
| 有效带宽/峰值 | >85% 很好；<50% 有空间 | 峰值口径/单位要一致 |
| Occupancy | achieved 明显低于 theoretical 才查 | 25% 也可能打满带宽 |
| Eligible warps/cycle | <1 倾向 latency-bound | 不是越高越好 |
| Sectors/request（32×4B） | 理想 4；8/16/32 说明非合并 | 32 B sector，不是 4 B |
| L1 hit rate | 流式访问低是正常的 | 不要追 100% |
| Shared bank conflict | 接近 0 最好 | padding 可能改变占用 |
| 小 kernel | <10 µs 考虑 launch 成本 | profiler 下会被放大 |

## 症状与先看什么

| 症状 | 先看 | 条目 |
| --- | --- | --- |
| GPU 时间线空白 | CPU/API/gap | [gap](nsight-systems-gpu-idle-and-launch-overhead.md) |
| 一个 kernel 占 90% | SOL 分类 | [ncu](nsight-compute-first-pass.md) |
| 很多小 kernel | launch 次数/Graphs | [gap](nsight-systems-gpu-idle-and-launch-overhead.md) |
| 带宽低 | sectors/request、hit | [访存](gpu-warp-stall-and-memory-analysis.md) |
| occupancy 低 | 限制项 | [roofline/occupancy](gpu-roofline-and-occupancy.md) |
| long scoreboard 高 | 访存/occupancy | [stall](gpu-warp-stall-and-memory-analysis.md) |
| 拷贝不重叠 | stream/pinned/同步 | [overlap](gpu-multi-stream-overlap.md) |
| step time 锯齿 | dataloader/CPU | [torch profiler](pytorch-profiler-and-memory-snapshot.md) |
| 显存 OOM | allocated/reserved/碎片 | [torch profiler](pytorch-profiler-and-memory-snapshot.md) |
| 多卡扩展差 | NCCL/拓扑/重叠 | [监控](gpu-monitoring-and-throttling.md) |
| 采不到计数器 | 权限/容器/WSL/MIG | [权限](gpu-profiling-permissions-and-containers.md) |

## 10 分钟例程

1. Fit to screen，记住总时长；
2. `nsys stats` 找 top kernel；
3. 找最大 gap，看 CPU/API；
4. 区分大 kernel / 很多小 kernel；
5. 看 memcpy 是否重叠；
6. 算 top kernel 占端到端比例（Amdahl）；
7. 对 top kernel 跑 `ncu --set basic`；
8. SOL 分类 → occupancy 限制项 → 第一名 stall；
9. 提出可证伪假设，做最小改动；
10. 多次测量，回 nsys 确认端到端。

## 验证

- 每条命令能在你的版本上跑通（先 `--version`/`--help`）；
- report/metric 名称与本机版本一致；
- 采集出来的报告能在 GUI 打开；
- 指标口径/单位与结论一致。

## 常见坑

- report 名、metric 名、CLI 选项随版本变化；
- `Time%` 不是 wall time；
- GB/s 与 GiB/s 混用；
- profiler 下的小 kernel 时间被放大；
- 没有计数器权限却反复重试。

## 参考

- Nsight Systems: <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>
- Nsight Compute CLI: <https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html>
- Nsight Compute Profiling Guide: <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>
- PyTorch Profiler: <https://pytorch.org/docs/stable/profiler.html>
