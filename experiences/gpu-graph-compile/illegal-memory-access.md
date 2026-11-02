---
name: illegal-memory-access
title: 非法访存与异步 CUDA 错误定位
description: 出现 illegal memory access、misaligned address、device-side assert、unspecified launch failure，或错误在无关调用处才抛出时，用 CUDA_LAUNCH_BLOCKING、compute-sanitizer 和 cuda-gdb 把异步错误定位到具体 kernel 与越界点。
category: crash
tags: [cuda, illegal-memory-access, sanitizer, cuda-gdb, torch.compile]
symptoms:
  - CUDA error: an illegal memory access was encountered
  - misaligned address / unspecified launch failure / bus error
  - device-side assert triggered，但不知道是哪个 kernel
  - 报错位置在无关的 tensor 操作或下一次迭代
  - 进程直接 abort，没有 Python traceback
stacks: [cuda, torch.compile, triton, tvm, xla]
updated: 2026-11-02
---

# 非法访存与异步 CUDA 错误定位

> 一句话经验：先把异步变同步，再用 memcheck 拿到“kernel + 地址 + 访问类型”，最后回到索引、mask、对齐和生命周期四类根因。

## 适用场景

- CUDA 报 `illegal memory access`、`misaligned address`、`unspecified launch failure`、`device-side assert triggered`。
- 错误出现在与根因无关的 API 调用上，或同一进程后续所有 CUDA 调用都报错。
- 多流、CUDA Graphs、Triton kernel、手写 CUDA 扩展中的越界/非法指针。
- PyTorch 报 device-side assert，需要定位到具体算子/索引。

## 症状与信号

- `CUDA error: an illegal memory access was encountered`：通常是越界读写、野指针或已释放内存。
- `misaligned address`：向量化访问（`float4`/`int4`、128-bit load）的指针对齐不满足要求。
- `device-side assert triggered`：kernel 内 `assert`/`CUDA_KERNEL_ASSERT` 失败，常见于索引越界（embedding、gather、cross entropy）。
- `unspecified launch failure` / `bus error`：可能是越界、硬件/驱动问题，也可能是 MIG/多进程环境问题。
- 报错发生在“下一次迭代”或“析构时”：典型的异步上报。
- 同一进程后续调用全部失败：context 已进入 sticky error 状态。

## 快速判断

1. **同步化**：设置 `CUDA_LAUNCH_BLOCKING=1` 重跑，让错误在真正的 kernel 调用处抛出。
2. **memcheck**：用 `compute-sanitizer --tool memcheck` 重跑，拿到 kernel 名、地址、访问类型和越界方向。
3. **设备断言**：PyTorch 官方 wheel 默认没有设备端断言行号；需要行号时只能用源码构建并启用 `TORCH_USE_CUDA_DSA`（构建期开关，不是运行时环境变量）。常规排查靠 memcheck + 缩小到具体算子。
4. **禁用缓存分配器**：`PYTORCH_NO_CUDA_MEMORY_CACHING=1` 让越界更容易被 memcheck 捕获，但会显著变慢。
5. **缩小范围**：固定输入、单卡单进程，关闭 cudagraph/多流/NCCL 后重跑。

## 排查步骤

### 1. 让错误同步暴露

```bash
CUDA_LAUNCH_BLOCKING=1 python3 repro.py
```

同步化后，报错位置通常会前移到真正出错的 kernel。它只用于定位，不要长期开启（会破坏异步流水、显著变慢）。

### 2. 用 compute-sanitizer 定位

```bash
compute-sanitizer --tool memcheck --print-limit 20 --padding 32 python3 repro.py
```

- 关注输出里的 kernel 名、源文件/行号（若有 lineinfo）、地址、读写方向和越界字节数。
- PyTorch 的 caching allocator 会保留已释放显存，越界写进“已释放但未归还”的区域可能不被 memcheck 发现；此时用 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 复跑。
- 多进程要加 `--target-processes all`；启动慢的任务调大 `--launch-timeout`。
- 同时有越界和竞态时，先 memcheck 再 racecheck（见[数据竞争与同步](race-and-sync.md)）。

### 3. 设备端断言与打印

```bash
CUDA_LAUNCH_BLOCKING=1 python3 repro.py
```

- PyTorch 的 `device-side assert triggered` 常见于 embedding/gather 索引越界、`topk` 的 k 过大、cross entropy target 越界；用 `CUDA_LAUNCH_BLOCKING=1` 先定位到具体算子，再检查 Python 侧的索引范围。
- 想要设备端断言的**文件和行号**，需要在源码构建 PyTorch 时启用 `TORCH_USE_CUDA_DSA`（构建期开关）；官方 wheel 不带，不能靠运行时设置开启。
- 自定义 kernel 中可用 `printf` 打印 block/thread 索引，但输出量可能很大，先限制到可疑区间。
- Triton 用 `tl.device_print` 输出运行期值；`tl.device_assert` 只有在 `TRITON_DEBUG=1` 时才触发（见[Triton 编译错误](triton-kernel-compile-error.md)）。

### 4. 用 cuda-gdb 抓现场

```bash
cuda-gdb --args python3 repro.py
(cuda-gdb) set cuda break_on_launch application
(cuda-gdb) set cuda api_failures stop
(cuda-gdb) run
```

- `set cuda break_on_launch application` 在每个 kernel 启动时停下，可查看 `threadIdx`/`blockIdx` 和调用栈；`set cuda api_failures stop` 在 API 返回错误时停下。
- **`set cuda memcheck on` 从 CUDA 12.0 起已随 cuda-memcheck 移除**，不要照旧资料使用。内存错误优先用 compute-sanitizer；需要事后分析时用 `compute-sanitizer --generate-coredump` 生成 core，再用 `cuda-gdb -c core` 打开。
- 附着调试需要 `ptrace_scope` 允许；任一 GPU 命中断点会暂停所有 GPU。
- cuda-gdb 与多进程、CUDA Graphs、部分驱动版本配合可能不稳定；优先用 memcheck 定位，再上 gdb。

### 5. 回到四类根因

| 类别 | 典型错误 | 检查点 |
| --- | --- | --- |
| 索引越界 | illegal access、assert | 尾部 mask、`<` vs `<=`、负索引、stride 计算、`gather`/`embedding` 的索引范围 |
| 对齐 | misaligned address | 向量化 load/store 的指针对齐、`reinterpret_cast`、slice 后的偏移 |
| 生命周期 | illegal access、随机值 | tensor 是否提前释放、异步复用、stream 间生命周期、cudagraph 固定地址 |
| 动态共享内存 | illegal access | `extern __shared__` 大小与 launch 配置、Triton 的 shared memory 需求 |

## 根因

1. **索引/边界计算错误**：最有效的手段是对比 CPU 参考实现，并在 kernel 里断言索引范围。
2. **mask 写反或缺失**：尾部元素、非连续输入、动态 shape 下最容易出现。
3. **步长/布局假设**：把非连续张量当连续处理，或忽略转置后的 stride。
4. **对齐问题**：向量化访问要求 16 字节对齐；切片/偏移后不再满足。
5. **内存生命周期**：kernel 异步执行时，输入 tensor 已被释放或复用；cudagraph 会放大这类问题。
6. **PyTorch 算子参数非法**：embedding 索引越界、`topk` 的 k 大于维度、cross entropy 的 target 越界，表现为 device-side assert。

## 修复与规避

- 越界/对齐：修索引和边界，必要时把向量化访问降级为标量访问验证；确认修复后再恢复向量化。
- 生命周期：显式持有输入引用直到同步完成；避免在异步执行未完成时复用 buffer。
- 设备断言：在 Python 侧先做范围校验，给出更友好的错误；不要只依赖设备端 assert。
- 临时规避：`CUDA_LAUNCH_BLOCKING=1`、单流、关闭 cudagraph 只用于定位；不要作为最终方案。

## 验证

- `compute-sanitizer --tool memcheck` 无报错；必要时再跑 `--tool initcheck` 检查未初始化读。
- 覆盖边界 shape（尾部、batch=1、非 2 的幂）、非连续输入、不同 dtype。
- 重复运行至少 10 次，确认不是偶发；多流/多卡场景单独覆盖。
- 修复后性能与基线对比，确认没有因为加同步或降级向量化而明显回退。

## 常见坑

- 看到 `illegal memory access` 就去查报错那一行的 Python 代码：根因通常在之前异步执行的 kernel 里。
- memcheck 没报错就认为安全：caching allocator、未初始化读、竞态都可能漏检；配合 `initcheck`/`racecheck`。
- 用 `torch.cuda.synchronize()` 掩盖问题：同步只改变上报位置，不修复越界。
- 在 `try/except` 里捕获 CUDA 错误后继续用同一个 context：sticky error 后 context 已不可用，必须重启进程。
- `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 会显著变慢且改变内存行为，只用于调试，不要留在生产配置。

## 参考

- NVIDIA Compute Sanitizer：<https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html>
- CUDA-GDB：<https://docs.nvidia.com/cuda/cuda-gdb/index.html>
- PyTorch CUDA 内存管理：<https://pytorch.org/docs/stable/notes/cuda.html#memory-management>
- PyTorch 环境变量（`PYTORCH_NO_CUDA_MEMORY_CACHING` 等）：<https://pytorch.org/docs/stable/torch_environment_variables.html>
