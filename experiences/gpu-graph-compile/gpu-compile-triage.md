---
name: gpu-compile-triage
title: GPU 图编译问题分层定位总纲
description: GPU 图编译或执行出现报错、崩溃、结果错误、性能回退时，先用“环境—捕获—编译—加载—执行”分层和最小化流程确定问题层，再进入对应专题。适用于 torch.compile、Triton、TVM、XLA/JAX 和底层 CUDA 工具链。
category: methodology
tags: [triage, torch.compile, triton, tvm, xla, cuda, debugging]
symptoms:
  - 有 Python traceback，但不确定问题在前端、编译器还是 runtime
  - 报错指向底层 CUDA，但复现不稳定，或换个无关调用才报错
  - 编译能过，运行时崩溃、卡住或结果不对
  - 能跑但性能明显回退，不知道先看哪一层
stacks: [torch.compile, triton, tvm, xla, cuda]
updated: 2026-11-02
---

# GPU 图编译问题分层定位总纲

> 一句话经验：先确定错误“第一次发生”的层，再选工具；不要从 traceback 最后一行直接猜根因。

## 适用场景

适用于“图被编译器接管之后”的问题：`torch.compile`、Triton kernel、TVM 调度/构建、XLA/JAX 的编译或执行异常。也适用于编译成功但运行崩溃、结果错误、性能回退的情况。

不适用于纯 eager 模式下的普通 Python/CUDA API 误用；这类问题直接用常规调试手段即可。

## 分层模型

| 层 | 典型组件 | 常见信号 | 第一动作 |
| --- | --- | --- | --- |
| L0 环境与输入 | 驱动、CUDA runtime、`ptxas`、GPU 架构、shape/dtype | 版本不匹配、`no kernel image`、分配失败 | 记录版本矩阵，核对架构 |
| L1 图捕获/前端 | TorchDynamo、JAX tracing、TVM 前端 | graph break、unsupported op、guard 失败、recompile | 看捕获日志，确认完整图 |
| L2 图优化/代码生成 | Inductor、Triton 编译器、TVM Pass/TIR、XLA HLO | `CompilationError`、`Check failed`、调度错误、`ptxas` 报错 | 打印中间 IR，定位到 pass |
| L3 模块加载 | cubin/PTX 加载、module init | `no kernel image`、`invalid device function`、PTX JIT 失败 | 检查二进制与架构 |
| L4 执行与同步 | kernel 执行、CUDA Graphs、NCCL | illegal memory access、assert、错误结果、hang | 同步化、缩小到单个 kernel |

关键点：CUDA 错误是异步的。L4 的越界可能在几十个 API 调用之后才由 `cudaGetLastError` 抛出，表面上的“报错位置”通常不是根因位置。

## 症状与信号

- Python traceback 里出现 `torch._dynamo`、`torch._inductor`、`triton/compiler`、`xla`：优先看 L1/L2。
- 进程直接 abort、段错误、core dump，没有完整 Python traceback：优先看 L3/L4 或 C++ 编译器崩溃。
- `CUDA error: an illegal memory access was encountered` 出现在无关调用上：异步错误，按 L4 处理。
- 只有特定 shape、batch、GPU 架构或多卡组合失败：优先 L0/L1，再查边界条件。
- 结果错但没有报错：不要从日志找，直接进入数值对拍流程。
- 首次运行慢、后续正常：先排除编译与 autotune，再谈性能。

## 快速判断

1. **固定证据**：保存完整命令、环境版本、输入 shape/dtype、完整日志（含 stderr），不要只截最后一行。
2. **判断是否编译期**：报错前是否已经产生可执行 kernel？能否在 `torch.compile` 之外用 eager 跑通？
3. **同步化**：设置 `CUDA_LAUNCH_BLOCKING=1` 再跑一次。报错位置改变，说明是异步错误。
4. **对拍**：用 eager/CPU/float64 参考结果判断是“崩溃”还是“算错”。
5. **去缓存**：关闭编译缓存和 autotune 再跑，排除旧产物命中。
6. **单卡单进程**：排除多进程日志交错、NCCL、共享 cache 目录的干扰。

## 排查步骤

### 第一步：采集环境与版本（5 分钟）

```bash
python3 - <<'PY'
import platform, sys
import torch

print("python:", sys.version.split()[0], platform.platform())
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda, "cudnn:", torch.backends.cudnn.version())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu:", p.name, "sm_%d%d" % (p.major, p.minor),
          "mem MiB:", p.total_memory // 2**20)
PY
nvcc --version
nvidia-smi
pip freeze > env.txt
```

同时记录：是否容器/WSL/MIG、是否多卡、`TORCH_CUDA_ARCH_LIST`、`CUDA_VISIBLE_DEVICES`、Triton/TVM/XLA 版本。

### 第二步：确认“第一次失败”的位置

- 从 traceback 的**最后一帧反向**找第一处位于自己代码或用户模型中的帧；框架内部帧只是传播路径。
- 没有 Python traceback 时看退出码：`137` 常见于被 OOM killer 杀掉，`134` 是 `SIGABRT`，`139` 是 `SIGSEGV`。
- 用 `2>&1 | tee run.log` 保存完整输出；编译器和驱动有时把关键信息写到 stderr 或系统日志（`dmesg`）。

### 第三步：按层缩小

| 问题 | 缩小手段 |
| --- | --- |
| 不确定是否编译器 | 关闭 `torch.compile`/JIT，用 eager 跑同一输入 |
| 怀疑异步错误 | `CUDA_LAUNCH_BLOCKING=1` |
| 怀疑缓存 | 换新的缓存目录，或禁用编译缓存 |
| 怀疑某个 shape | 只改 shape，保持其余输入不变 |
| 怀疑 GPU 代码生成 | 换 CPU 后端（TVM `llvm`、`JAX_PLATFORMS=cpu`、Triton `TRITON_INTERPRET=1`） |
| 怀疑多卡 | 单卡单进程复现 |
| 怀疑数值 | 与 float64/eager 参考对拍，逐算子比较 |

### 第四步：进入专题条目

| 现象 | 转入 |
| --- | --- |
| 编译期 Python 异常、graph break、recompile | [torch.compile 后端报错](torch-compile-backend-error.md) |
| Triton kernel 编译失败 | [Triton 编译错误](triton-kernel-compile-error.md) |
| TVM `Check failed`/调度错误 | [TVM 构建错误](tvm-build-error.md) |
| XLA/JAX 编译错误 | [XLA 编译错误](xla-compile-error.md) |
| 非法访存、异步错误、设备断言 | [非法访存定位](illegal-memory-access.md) |
| 不报错但结果错 | [静默错误结果](silent-wrong-results.md) |
| 偶发、随运行变化 | [数据竞争与同步](race-and-sync.md) |
| 能跑但慢 | [慢 kernel 分层排查](slow-kernel-triage.md) |
| 寄存器溢出/占用率低 | [寄存器溢出与占用率](register-spill-occupancy.md) |
| CUDA Graphs 捕获失败 | [CUDA Graphs 捕获问题](cudagraph-capture-problems.md) |
| 反复重编译、编译慢 | [编译缓存与重编译](compile-cache-and-recompiles.md) |
| 多卡 hang/超时/结果不一致 | [分布式图编译](distributed-graph-compile.md) |
| 架构/驱动/工具链不匹配 | [工具链与架构不匹配](toolchain-arch-mismatch.md) |
| 精度偏差、NaN | [精度与 fast-math](precision-and-fastmath.md) |
| 需要最小复现 | [最小复现与配置二分](minimal-repro.md) |

## 根因

问题被误分层通常来自四个原因：

1. **异步上报**：kernel 执行是异步的，越界/断言在后续同步点才报错。表面位置与根因位置相差很远。
2. **缓存与产物复用**：旧 cubin、旧 autotune 结果、旧编译缓存让“改了代码没生效”或“在新机器上行为不同”。
3. **包装异常**：`BackendCompilerFailed`、`XlaRuntimeError` 等是包装层，真正原因在 `__cause__` 或更内层的日志里。
4. **多进程/多卡混合**：日志交错、rank 间控制流不一致、共享 cache 竞争，让单卡单进程无法复现。

## 修复与规避

- 先修根因，再做规避。常见规避（关融合、关 autotune、关 cudagraph、降级到 eager）只用于确认问题层或临时止损，需在结果中说明代价。
- 不要用 `suppress_errors`、`try/except` 或清缓存来“消除”报错；这些操作会丢失证据。
- 对编译器/驱动问题，保留最小复现、IR、版本矩阵和完整日志，便于向上游报告。

## 验证

- 同一份输入重复运行至少 10 次（含冷启动和热启动），结果一致。
- 与 eager/CPU/float64 参考对拍，明确容差。
- 崩溃类问题在 `compute-sanitizer` 下无报错（见[非法访存定位](illegal-memory-access.md)）。
- 性能问题给出固定输入、预热后的中位数和基线，不用单次测量下结论。

## 常见坑

- 把编译期 OOM 当成模型 OOM：编译 worker 的峰值内存与训练进程不同。
- 看到 `graph break` 就认为失败：它是性能/覆盖问题，不一定是错误。
- 清空缓存后问题消失就结案：可能只是旧产物被删，根因仍在。
- 只跑一次就判断“偶发问题已修复”：竞态和越界需要重复与 sanitizer 验证。
- 用多卡日志的最后几行判断 rank：先按 rank 分离日志再看。

## 参考

- PyTorch `torch.compile` troubleshooting：<https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html>
- PyTorch logging（`TORCH_LOGS`）：<https://pytorch.org/docs/stable/logging.html>
- NVIDIA Compute Sanitizer：<https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html>
- Triton debugging：<https://triton-lang.org/main/programming-guide/chapter-3/debugging.html>
