---
name: distributed-graph-compile
title: 分布式图编译：多卡 hang、NCCL 超时与 rank 间图不一致
description: 单卡正常、多卡 hang/超时/结果不一致，或 torch.compile 与分布式集合通信、CUDA Graphs 组合出问题时，按“控制流一致—collective 顺序—shape 对齐—超时与 flight recorder—通信与图捕获”逐项排查。
category: engineering
tags: [distributed, nccl, torch.compile, cudagraph, multi-gpu]
symptoms:
  - NCCL timeout / watchdog 报错，进程挂起不退出
  - 单卡正确，多卡结果不一致或 loss 发散
  - 只有部分 rank 抛异常，其他 rank 卡住
  - torch.compile 后多卡 hang，eager 正常
  - CUDA Graphs 捕获期集合通信失败
stacks: [torch.compile, cuda, nccl]
updated: 2026-11-02
---

# 分布式图编译：多卡 hang、NCCL 超时与 rank 间图不一致

> 一句话经验：多卡问题先确认“所有 rank 是否执行了相同顺序、相同数量的 collective”，再看 shape、超时和通信后端；图编译只是放大不一致，不改变这个前提。

## 适用场景

- `torchrun`/`torch.distributed` 多卡训练或推理 hang、超时。
- 单卡正常，多卡结果不一致、loss 异常或梯度不同步。
- `torch.compile`、CUDA Graphs、NCCL 组合使用后出现问题。
- 需要定位是哪个 rank、哪个 collective、哪一步开始不一致。

## 症状与信号

- `ProcessGroupNCCL` watchdog 报 `NCCL timeout` / `Watchdog caught collective operation timeout`。
- 日志里只有部分 rank 有异常，其他 rank 停在 `all_reduce`/`barrier`。
- 所有 rank 都能跑完但 loss/梯度不一致：可能是 collective 顺序错、shape 不一致或部分 rank 跳过了同步。
- 开启 `torch.compile`/cudagraph 后 hang，关闭后正常。
- 增加/减少 GPU 数、改变 batch 或序列长度后复现率变化。

## 快速判断

1. **分离 rank 日志**：每个 rank 单独写文件，避免交错；用 `torchrun --redirects 1` 或应用内按 rank 建 log。
2. **确认卡在哪**：用 `py-spy dump`/`faulthandler` 看每个 rank 的 Python 栈，或看 NCCL watchdog 报告。
3. **打开调试信息**：`TORCH_DISTRIBUTED_DEBUG=DETAIL`、`NCCL_DEBUG=INFO`，确认 collective 顺序和 size。
4. **缩小规模**：2 卡复现；关闭 `torch.compile`/cudagraph；单进程多卡改单卡。
5. **检查控制流**：是否有 `if rank == 0`、数据依赖分支、动态 shape 导致某 rank 跳过了 collective。

## 排查步骤

### 1. 让每个 rank 的日志可读

```bash
torchrun --nproc_per_node=2 --redirects 3 run.py
# 3 = stdout + stderr 都重定向到日志文件；也可在应用内按 rank 建 log
```

把 `RANK`、step、collective 名称、tensor shape 打进日志。只有分离日志，才能判断是哪个 rank 先偏离。

### 2. 抓取卡住时的调用栈

```bash
py-spy dump --pid <pid>
```

或应用内注册：

```python
import faulthandler
faulthandler.dump_traceback_later(60, repeat=True)
```

看每个 rank 停在 Python 的哪一行、是否在 `all_reduce`/`barrier`/`wait`。如果所有 rank 都停在同一个 collective，通常是调用次数/顺序不一致；如果部分 rank 停在计算，说明它没走到 collective。

### 3. 打开分布式与 NCCL 调试

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO python3 run.py
```

- `TORCH_DISTRIBUTED_DEBUG=DETAIL`：DDP 场景下打印参数/模块级别的同步信息，帮助定位未使用参数、bucket 不一致。
- `NCCL_DEBUG=INFO`：打印通信初始化、拓扑、collective 调用；日志量很大，只在复现时短时间开启。
- 设置 NCCL 超时（如 `TORCH_NCCL_BLOCKING_WAIT`/`timeout` 参数）要谨慎：超时太短会把慢 rank 误判为故障，太长会拖慢定位。

### 4. 用 flight recorder 看最后的通信

PyTorch 的 NCCL flight recorder 会保留最近一段 collective 记录，超时后导出：

```bash
TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 \
TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
python3 run.py
```

具体 dump 位置和解析方式随版本变化，按当前 PyTorch 文档操作。它能回答“哪个 rank 最后发起了哪个 collective、其他 rank 在等什么”。

### 5. 检查图编译与捕获

- 关闭 `torch.compile`（`mode="default"` 或直接 eager）和 cudagraph，确认问题是否仍存在。
- 图编译会改变执行顺序和融合边界，但要求所有 rank 的图结构一致；按数据/rank 分支会导致不同 rank 走不同图。
- CUDA Graphs 捕获期执行 NCCL 需要通信库支持，且所有 rank 捕获顺序一致；不满足时先对通信部分关闭 cudagraph。
- 动态 shape 在不同 rank 上产生不同图/不同 collective 参数，会直接导致 hang 或结果错误。

### 6. 检查数据与并行策略

- DDP：每个 rank 的 batch 数是否一致；`find_unused_parameters` 是否掩盖了真实问题。
- FSDP/张量并行：分片 shape 是否一致；某 rank 是否因数据不足跳过 step。
- 数据加载：某 rank 卡在 DataLoader/IO，其他 rank 在等 collective。
- 混合精度/梯度裁剪：某 rank 出现 NaN 后提前退出，其他 rank 继续等待。

## 根因

1. **collective 顺序/数量不一致**：条件分支、异常提前退出、不同 step 数。
2. **shape 不一致**：动态 shape、变长序列、不均匀 batch 导致 all_reduce 的 tensor 对不上。
3. **超时设置不当**：过短导致误杀，过长导致定位慢；慢 rank 本身可能是 CPU/IO 瓶颈。
4. **图编译/捕获差异**：rank 间图不同、cudagraph 捕获顺序不同、通信未被图支持。
5. **环境问题**：NCCL 拓扑/网卡选择、容器网络、共享文件系统、端口冲突、MIG。
6. **进程组生命周期**：提前 `destroy_process_group`、fork 后使用父进程的组、异常路径未清理。

## 修复与规避

- 让所有 rank 的控制流、step 数、collective 顺序严格一致；分支条件用全局张量同步，而不是只看本地 rank。
- 对齐 shape：padding/truncate 到固定长度，或使用支持变长的通信原语；不要在不同 rank 上静默改变 batch。
- 为慢 rank 留足超时，同时监控各 rank 的 step 时间；区分“真故障”和“慢”。
- 图编译场景先保证 eager 多卡正确，再逐步开启 compile/cudagraph，每次只开一项。
- 规避：关闭 cudagraph、单流、减少融合；需说明性能代价。

## 验证

- 2 卡、4 卡（有条件再加 8 卡）重复运行，覆盖冷启动和异常恢复。
- 注入延迟/慢 rank，确认超时和错误信息可读。
- 与单卡结果对拍，确认数值一致（在容差内）。
- 用 `nsys` 看多进程时间线，确认没有 rank 在无谓等待。
- 记录 NCCL/驱动/PyTorch 版本和拓扑，便于复现。

## 常见坑

- 只看 rank 0 的日志，忽略其他 rank 的异常。
- 把 NCCL timeout 当成根因：它通常只是“某个 rank 没到达”的结果。
- 用 `barrier()` 到处加同步来“修复”hang，可能掩盖真正的不一致。
- 在 timeout 后复用同一个进程组继续训练，状态已不可靠。
- 只在单机多卡验证，忽略多机网络/拓扑差异。
- `torch.compile` 的动态 shape 让不同 rank 生成不同图，表面看是通信问题。

## 参考

- PyTorch 分布式概览：<https://pytorch.org/docs/stable/distributed.html>
- PyTorch `torchrun`：<https://pytorch.org/docs/stable/elastic/run.html>
- NCCL 文档：<https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html>
- PyTorch flight recorder：<https://pytorch.org/docs/stable/torch_nccl_environment_variables.html>
- PyTorch CUDA Graphs（与通信组合的限制）：<https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs>
