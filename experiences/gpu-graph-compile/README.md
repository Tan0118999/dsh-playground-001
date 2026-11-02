# GPU 图编译 debug

本目录收录 GPU 图编译链路的调试经验，覆盖 `torch.compile`/TorchInductor、Triton、TVM、XLA/JAX，以及 compute-sanitizer、cuda-gdb、Nsight、SASS/PTX 等底层工具。

## 建议阅读顺序

1. 先读 [GPU 图编译问题分层定位总纲](gpu-compile-triage.md)，确定问题层。
2. 按症状进入对应条目（见 [经验条目索引](../README.md#症状速查)）。
3. 需要报 bug 或反复复现时，读 [最小复现与配置二分](minimal-repro.md)。

## 分层模型

```text
L0 环境/输入      驱动、CUDA runtime、ptxas、GPU 架构、shape/dtype
        ↓
L1 图捕获/前端    TorchDynamo、JAX trace、TVM 前端
        ↓
L2 优化/代码生成  Inductor、Triton、TVM pass/TIR、XLA HLO
        ↓
L3 模块加载       cubin/PTX 加载、module init
        ↓
L4 执行/同步      kernel、CUDA Graphs、NCCL
```

先确定“错误第一次发生”的层，再选工具。CUDA 错误是异步的，L4 的越界可能在很晚才上报。

## 环境变量与开关速查

| 开关 | 作用 | 条目 |
| --- | --- | --- |
| `CUDA_LAUNCH_BLOCKING=1` | 让 kernel 同步执行，错误在真正调用处抛出 | [非法访存](illegal-memory-access.md) |
| `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | 关闭 caching allocator，让 memcheck 看到分配边界（很慢） | [非法访存](illegal-memory-access.md) |
| `CUDA_DEVICE_WAITS_ON_EXCEPTION=1` | 设备异常时等待调试器附着 | [非法访存](illegal-memory-access.md) |
| `TORCH_LOGS` | 打开 Dynamo/Inductor 日志；`+` 为 DEBUG，裸名为 INFO | [torch.compile](torch-compile-backend-error.md) |
| `TORCH_LOGS_OUT` | 日志额外写入文件 | [torch.compile](torch-compile-backend-error.md) |
| `TORCH_COMPILE_DEBUG=1` | 保存 FX graph、IR、`output_code.py`、编译 profile | [torch.compile](torch-compile-backend-error.md) |
| `TORCHDYNAMO_REPRO_AFTER=dynamo\|aot` | 在捕获边界/AOTAutograd 后包装后端，生成复现 | [最小复现](minimal-repro.md) |
| `TORCHDYNAMO_REPRO_LEVEL=1..4` | 控制 repro/minifier 产物，4 做精度最小化 | [最小复现](minimal-repro.md) |
| `TORCHINDUCTOR_CACHE_DIR` | Inductor 编译缓存目录 | [编译缓存](compile-cache-and-recompiles.md) |
| `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` | 强制禁用 Inductor 缓存 | [编译缓存](compile-cache-and-recompiles.md) |
| `TORCH_TRACE=/tmp/dir` | 记录编译轨迹，配合 `tlparse` 生成 HTML 报告 | [编译缓存](compile-cache-and-recompiles.md) |
| `TRITON_INTERPRET=1` | 在 CPU 上用 numpy 解释执行，调试逻辑 | [Triton](triton-kernel-compile-error.md) |
| `TRITON_ALWAYS_COMPILE=1` | 绕过缓存，强制重新编译 | [Triton](triton-kernel-compile-error.md) |
| `TRITON_KERNEL_DUMP=1` + `TRITON_DUMP_DIR` | dump 各阶段 IR/PTX/cubin | [Triton](triton-kernel-compile-error.md) |
| `TRITON_REPRODUCER_PATH` | 失败时保存 MLIR reproducer | [Triton](triton-kernel-compile-error.md) |
| `TRITON_DUMP_PTXAS_LOG=1` | 保留 ptxas 日志（含 `-v` 寄存器/spill） | [Triton](triton-kernel-compile-error.md) |
| `MLIR_ENABLE_DUMP=1` / `MLIR_DUMP_PATH` | dump 每个 pass 前后的 IR（输出量大） | [Triton](triton-kernel-compile-error.md) |
| `TRITON_CACHE_DIR` | Triton 缓存目录，默认 `~/.triton/cache` | [Triton](triton-kernel-compile-error.md) |
| `TVM_LOG_DEBUG="file=level"` | TVM C++ 日志，文件路径相对 `src/` | [TVM](tvm-build-error.md) |
| `XLA_FLAGS="--xla_dump_to=..."` | dump XLA HLO/优化过程 | [XLA/JAX](xla-compile-error.md) |
| `XLA_CLIENT_MEM_FRACTION` | 控制 XLA 预分配显存比例；旧名已弃用 | [XLA/JAX](xla-compile-error.md) |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | 关闭预分配，区分预分配 OOM 和真实峰值 | [XLA/JAX](xla-compile-error.md) |
| `JAX_COMPILATION_CACHE_DIR` | JAX/XLA 持久化编译缓存 | [XLA/JAX](xla-compile-error.md) |
| `JAX_DEBUG_NANS=1` / `JAX_DEBUG_INFS=1` | 每个 op 后检查 NaN/inf 并定位 | [XLA/JAX](xla-compile-error.md) |
| `NCCL_DEBUG=INFO` | NCCL 初始化/通信日志 | [分布式](distributed-graph-compile.md) |
| `TORCH_DISTRIBUTED_DEBUG=DETAIL` | DDP/分布式详细日志 | [分布式](distributed-graph-compile.md) |
| `TORCH_NCCL_TRACE_BUFFER_SIZE` / `TORCH_NCCL_DUMP_ON_TIMEOUT` | NCCL flight recorder，超时导出通信记录 | [分布式](distributed-graph-compile.md) |

## 常用命令

```bash
# 内存/竞态/未初始化检查
compute-sanitizer --tool memcheck --padding 32 python3 repro.py
compute-sanitizer --tool racecheck python3 repro.py
compute-sanitizer --tool initcheck python3 repro.py

# 单 kernel 计数器与端到端时间线
ncu --set full -o prof python3 run.py
nsys profile --trace=cuda,nvtx,osrt -o report python3 run.py && nsys stats report.nsys-rep

# 二进制检查
cuobjdump -res-usage kernel.cubin
cuobjdump -sass kernel.cubin
cuobjdump -lelf libkernel.so
nvdisasm -plr kernel.cubin

# 卡住时看 Python 栈
py-spy dump --pid <pid>

# torch.compile 捕获分析
python3 -c "import torch; print(torch._dynamo.explain(f)(x))"

# 生成编译轨迹报告
TORCH_TRACE=/tmp/tracedir python3 run.py
tlparse /tmp/tracedir
```

## 版本差异提醒

- **PyTorch 2.4–2.7 vs 2.8+**：重编译上限叫 `cache_size_limit`/`accumulated_cache_size_limit`，2.8+ 改名为 `recompile_limit`/`accumulated_recompile_limit`；`InductorError` 约 2.9 才作为独立异常类出现。
- **PyTorch 分配器变量**：2.4–2.9 用 `PYTORCH_CUDA_ALLOC_CONF`；`PYTORCH_ALLOC_CONF` 只在更新的版本里作为回退。
- **TVM**：`tvm.lower` 在 v0.21 起移除；`tvm.build` 在 v0.27 已弃用；`tvm.contrib.debugger`/`debug_executor` 已移除；`TVM_BACKTRACE` 是旧变量。
- **Triton**：`TRITON_REPRODUCER_ALWAYS` 不存在；dump 目录按源码 hash、cache 目录按完整 cache key，不能互相拷贝。
- **XLA/JAX**：`--xla_dump_stage` 不是当前真实 flag；`XLA_PYTHON_CLIENT_MEM_FRACTION` 已弃用；`as_text()` 默认输出 StableHLO。
- **CUDA 12+**：cuda-gdb 的 `set cuda memcheck on` 已随 cuda-memcheck 移除，内存检查用 compute-sanitizer。

## 条目

- 方法与流程：[分层定位总纲](gpu-compile-triage.md)、[最小复现](minimal-repro.md)
- 编译错误：[torch.compile](torch-compile-backend-error.md)、[Triton](triton-kernel-compile-error.md)、[TVM](tvm-build-error.md)、[XLA/JAX](xla-compile-error.md)
- 崩溃：[非法访存](illegal-memory-access.md)
- 正确性：[静默错误](silent-wrong-results.md)、[竞争与同步](race-and-sync.md)、[精度与 fast-math](precision-and-fastmath.md)
- 性能：[慢 kernel](slow-kernel-triage.md)、[寄存器/占用率](register-spill-occupancy.md)、[CUDA Graphs](cudagraph-capture-problems.md)
- 工程：[编译缓存](compile-cache-and-recompiles.md)、[分布式](distributed-graph-compile.md)、[工具链/架构](toolchain-arch-mismatch.md)

新增条目请复制 [`templates/experience.md`](../../templates/experience.md)，并更新本文件、[经验条目索引](../README.md) 和 `catalog.json`。
