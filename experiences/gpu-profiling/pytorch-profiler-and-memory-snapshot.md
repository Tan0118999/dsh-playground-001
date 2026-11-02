---
name: pytorch-profiler-and-memory-snapshot
title: PyTorch Profiler 与显存快照：先找 top 算子，再看 OOM
description: 用 PyTorch 训练/推理，想快速找 top 算子、shape、调用栈、显存峰值，或排查 OOM/碎片、dataloader 瓶颈时读这条。含 schedule、TensorBoard、memory snapshot 和与 nsys 的分工。
category: performance
tags: [pytorch, torch-profiler, tensorboard, memory, oom, dataloader]
symptoms:
  - 不知道哪个 Python 算子慢
  - 同一个算子不同 shape 时间差很多
  - 显存 OOM 但不知道谁占的
  - dataloader 是否是瓶颈
  - 不想先学 Nsight
stacks: [pytorch, cuda]
updated: 2026-11-02
---

# PyTorch Profiler 与显存快照

> 一句话经验：**用 PyTorch，先跑 torch profiler 找 top 算子/shape，比直接上 nsys 更快；显存问题用 memory snapshot，不要靠猜。**

## 适用场景

- 模型用 PyTorch，想快速定位 top 算子；
- 怀疑某个 shape/batch 导致慢；
- 显存 OOM/碎片/峰值异常；
- 需要在 TensorBoard 里对比两个版本。

不解决：CUDA API/stream/copy 的精确时间线（用 [Nsight Systems](nsight-systems-timeline-reading.md)）；单 kernel 硬件计数器（用 [ncu](nsight-compute-first-pass.md)）。

## 症状与信号

- 知道 step 慢，但不知道是哪个 op；
- 同一个 op 在某个 shape 下特别慢；
- `torch.cuda.memory_allocated` 不高但 OOM；
- GPU 利用率周期性下降，怀疑 dataloader；
- 只在 eval/train 某个阶段慢。

## 快速判断

1. `key_averages()` 按 CUDA/device time 排序，找 top 算子；
2. 看 `# Calls`：很多小调用 → launch/框架开销；
3. 开启 `record_shapes`，看 shape 是否敏感；
4. 看 CPU total：是 GPU 慢还是 CPU 发得慢；
5. 显存：看 allocated vs reserved，再抓 snapshot；
6. 仍不够 → nsys/ncu。

## 最小可用代码

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./log"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        if step >= 6:          # wait 1 + warmup 1 + active 3 + repeat 1
            break
        train_step(batch)
        prof.step()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")
```

说明：

- `schedule(wait, warmup, active, repeat)`：先跳过 `wait`，再 warmup，再采 `active`；避免首次编译/分配；
- `record_shapes=True`：表里带 shape；
- `profile_memory=True`：记录显存分配；
- `with_stack=True`：带 Python 调用栈，开销更大；
- `self_cuda_time_total` 在较新 PyTorch 中逐步改为 `device_time_total`，以你版本的列为准。

## 表里看什么

| 列 | 含义 |
| --- | --- |
| `Name` | 算子/kernel 名 |
| `Self CPU` / `CPU total` | 自身/含子调用的 CPU 时间 |
| `Self CUDA` / `CUDA total` | 自身/含子调用的 GPU 时间 |
| `# Calls` | 调用次数 |
| `Input Shapes` | 开启 `record_shapes` 后出现 |

- 按 CUDA/device time 排序，先看 GPU 大头；
- 对照 CPU total，判断 GPU 慢还是 CPU 发得慢；
- 调用多、单次短 → launch/框架开销；
- 同一算子不同 shape → 形状敏感；
- 某个 op 的 CPU 时间高但 GPU 低 → host-bound。

## TensorBoard

```bash
tensorboard --logdir ./log
```

“PyTorch Profiler” 标签页：

- **Trace**：CPU/GPU 时间线（以算子为中心）；
- **Operator**：按算子聚合；
- **Memory**：显存随时间变化；
- **Overview**：step 时间、GPU 利用率、top ops。

适合快速找 top 算子、把 Python 路径和 GPU 算子对应起来、对比两个版本。不适合精确 CUDA API/stream/copy 重叠。

## 显存：OOM/碎片

```python
# 记录分配历史（有开销，短时间开）
torch.cuda.memory._record_memory_history(max_entries=100000)

# ... 跑 workload ...

torch.cuda.memory._dump_snapshot("mem.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
```

把 `mem.pickle` 拖到 <https://pytorch.org/memory_viz>，可以看到：

- 每个 allocation 的调用栈、大小、地址；
- 峰值时刻谁占着显存；
- 碎片（allocated vs reserved 的差距）；
- activation 没释放、cache 太大、临时 tensor 太多。

辅助：

```python
print(torch.cuda.memory_summary())
print(torch.cuda.max_memory_allocated() / 2**30, "GiB")
print(torch.cuda.memory_allocated() / 2**30, "GiB")
print(torch.cuda.memory_reserved() / 2**30, "GiB")
```

`allocated` 是 tensor 实际占用，`reserved` 是 allocator 向驱动要的池子；差距大 = 碎片/缓存。`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 在支持的版本上可缓解碎片，但要看版本/平台。

## 和 nsys/ncu 的分工

| 维度 | PyTorch Profiler | Nsight Systems |
| --- | --- | --- |
| 上手成本 | 低（几行代码） | 中（采集 + GUI/CLI） |
| 算子/shape/调用栈 | 强 | 需要 NVTX 辅助 |
| CUDA API/stream/copy | 中 | 强 |
| 开销 | 中–高 | 可调 |
| 非 PyTorch | 不适用 | 通用 |

实践顺序：**torch profiler 找 top 算子 → nsys 看时间线上下文 → ncu 分析单 kernel。**

## 验证

- top 算子/占比与 nsys 的 kernel 汇总对得上；
- 改动后同一 schedule/输入下对比；
- 显存优化后用 snapshot 确认峰值/碎片下降；
- 输出正确性不变。

## 常见坑

- 不 warmup，把首次编译/分配算进去；
- 忘记 `prof.step()`，schedule 不推进；
- `with_stack=True` 开销大，长时间开启影响结果；
- 把 allocator 的 reserved 当泄漏；
- 只看 CPU time，不看 CUDA time；
- 不同 batch/shape 直接比较；
- 把 torch profiler 的时间线当 CUDA API 级时间线（精度/字段不同）。

## 参考

- PyTorch Profiler: <https://pytorch.org/docs/stable/profiler.html>
- PyTorch Profiler Recipe: <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>
- PyTorch 显存理解: <https://pytorch.org/blog/understanding-gpu-memory-1/>
- 显存可视化: <https://pytorch.org/memory_viz>
