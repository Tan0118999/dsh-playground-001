# 经验条目

共 **31 条**经验，分为两个主题：

- [`gpu-graph-compile/`](gpu-graph-compile/)：GPU 图编译 debug，16 条；
- [`gpu-profiling/`](gpu-profiling/)：GPU Profiling 与性能分析，15 条。

按现象查表，不要只按标题猜。

## 症状速查

| 你看到的现象 | 先读 |
| --- | --- |
| 有报错但不知道是前端、编译器还是 runtime；报错位置飘忽 | [GPU 图编译问题分层定位总纲](gpu-graph-compile/gpu-compile-triage.md) |
| 需要把大模型问题缩成单文件、向上游报 bug | [最小复现与配置二分](gpu-graph-compile/minimal-repro.md) |
| `torch.compile` 抛 `BackendCompilerFailed`、graph break 多、反复重编译 | [torch.compile 后端报错](gpu-graph-compile/torch-compile-backend-error.md) |
| Triton 报 `CompilationError`、shared memory/寄存器超限、`PTXASError` | [Triton kernel 编译错误](gpu-graph-compile/triton-kernel-compile-error.md) |
| TVM 报 `Check failed`、`ScheduleError`，或旧 API 不存在 | [TVM 构建与调度错误](gpu-graph-compile/tvm-build-error.md) |
| JAX/XLA 报 `XlaRuntimeError`、`INVALID_ARGUMENT`、`RESOURCE_EXHAUSTED` | [XLA/JAX 编译错误](gpu-graph-compile/xla-compile-error.md) |
| `illegal memory access`、`misaligned address`、`device-side assert` | [非法访存与异步 CUDA 错误定位](gpu-graph-compile/illegal-memory-access.md) |
| 不报错但结果错、偶发错误、精度差 | [静默错误结果](gpu-graph-compile/silent-wrong-results.md) |
| 重复运行结果不同、只在多 warp/多卡出错 | [数据竞争与同步](gpu-graph-compile/race-and-sync.md) |
| 出现 NaN/inf、TF32/fp16 偏差、与参考差 1e-3 | [精度与 fast-math](gpu-graph-compile/precision-and-fastmath.md) |
| 能跑但慢、GPU 利用率低、融合没生效 | [慢 kernel 分层排查](gpu-graph-compile/slow-kernel-triage.md) |
| `ptxas` 报 spill、occupancy 低、tile 调大更慢 | [寄存器溢出与占用率](gpu-graph-compile/register-spill-occupancy.md) |
| CUDA Graphs 捕获失败/回退/捕获后结果错 | [CUDA Graphs 捕获问题](gpu-graph-compile/cudagraph-capture-problems.md) |
| 每次启动都编译、cache 目录暴涨、编译 OOM | [编译缓存与重编译](gpu-graph-compile/compile-cache-and-recompiles.md) |
| 多卡 hang、NCCL 超时、rank 间结果不同 | [分布式图编译](gpu-graph-compile/distributed-graph-compile.md) |
| `no kernel image`、`sm_90a` 未定义、驱动不足 | [工具链与架构不匹配](gpu-graph-compile/toolchain-arch-mismatch.md) |

## Profiling 症状速查

| 你看到的现象 | 先读 |
| --- | --- |
| 只会看 msprof，不知道 Nsight 从哪看起 | [NPU→GPU 概念对照](gpu-profiling/npu-to-gpu-profiling-mapping.md) |
| 能跑但慢，不知道先优化谁 | [分层定位总纲](gpu-profiling/gpu-profiling-triage.md) |
| 打开 nsys 一屏彩色条，不知道点哪 | [Nsight Systems 时间线](gpu-profiling/nsight-systems-timeline-reading.md) |
| 时间线大片空白 / GPU 利用率锯齿 | [GPU 空转与 launch](gpu-profiling/nsight-systems-gpu-idle-and-launch-overhead.md) |
| 大量 kernel 只有几微秒 | [GPU 空转与 launch](gpu-profiling/nsight-systems-gpu-idle-and-launch-overhead.md) |
| ncu 几十个 section 不知道先看哪 | [Nsight Compute 第一轮](gpu-profiling/nsight-compute-first-pass.md) |
| `ERR_NVGPUCTRPERM`，采不到计数器 | [权限与容器](gpu-profiling/gpu-profiling-permissions-and-containers.md) |
| long scoreboard / barrier 很高 | [warp stall 与访存](gpu-profiling/gpu-warp-stall-and-memory-analysis.md) |
| 带宽上不去、sectors/request 高 | [warp stall 与访存](gpu-profiling/gpu-warp-stall-and-memory-analysis.md) |
| 不确定 kernel 还有多少优化空间 | [roofline 与 occupancy](gpu-profiling/gpu-roofline-and-occupancy.md) |
| GEMM 只有峰值的几个百分点 | [naive SGEMM](gpu-profiling/gpu-naive-sgemm-bottleneck.md) |
| 用了多流/async 但没变快 | [stream 与 copy engine](gpu-profiling/gpu-multi-stream-overlap.md) |
| 结果跳动大 / profiler 下慢很多 | [测量卫生](gpu-profiling/gpu-profiling-measurement-hygiene.md) |
| 想找 top 算子 / 显存 OOM | [PyTorch Profiler](gpu-profiling/pytorch-profiler-and-memory-snapshot.md) |
| 怀疑降频 / 时钟 / 多卡拓扑 | [监控与降频](gpu-profiling/gpu-monitoring-and-throttling.md) |
| 想查命令/指标/阈值 | [命令与指标速查](gpu-profiling/gpu-profiling-cli-cheatsheet.md) |
| 手头是 AMD/Intel/Apple/TPU | [跨平台工具](gpu-profiling/gpu-other-platform-tools.md) |

## 图编译按类别

| 类别 | 条目 |
| --- | --- |
| 方法（2） | [分层定位总纲](gpu-graph-compile/gpu-compile-triage.md) · [最小复现](gpu-graph-compile/minimal-repro.md) |
| 编译（4） | [torch.compile](gpu-graph-compile/torch-compile-backend-error.md) · [Triton](gpu-graph-compile/triton-kernel-compile-error.md) · [TVM](gpu-graph-compile/tvm-build-error.md) · [XLA/JAX](gpu-graph-compile/xla-compile-error.md) |
| 崩溃（1） | [非法访存与异步错误](gpu-graph-compile/illegal-memory-access.md) |
| 正确性（3） | [静默错误](gpu-graph-compile/silent-wrong-results.md) · [竞争与同步](gpu-graph-compile/race-and-sync.md) · [精度与 fast-math](gpu-graph-compile/precision-and-fastmath.md) |
| 性能（3） | [慢 kernel](gpu-graph-compile/slow-kernel-triage.md) · [寄存器/占用率](gpu-graph-compile/register-spill-occupancy.md) · [CUDA Graphs](gpu-graph-compile/cudagraph-capture-problems.md) |
| 工程（3） | [编译缓存](gpu-graph-compile/compile-cache-and-recompiles.md) · [分布式](gpu-graph-compile/distributed-graph-compile.md) · [工具链/架构](gpu-graph-compile/toolchain-arch-mismatch.md) |

## Profiling 按类别

| 类别 | 条目 |
| --- | --- |
| 方法（4） | [NPU→GPU 对照](gpu-profiling/npu-to-gpu-profiling-mapping.md) · [分层定位](gpu-profiling/gpu-profiling-triage.md) · [roofline/occupancy](gpu-profiling/gpu-roofline-and-occupancy.md) · [测量卫生](gpu-profiling/gpu-profiling-measurement-hygiene.md) |
| Nsight Systems（3） | [时间线](gpu-profiling/nsight-systems-timeline-reading.md) · [gap/launch](gpu-profiling/nsight-systems-gpu-idle-and-launch-overhead.md) · [stream overlap](gpu-profiling/gpu-multi-stream-overlap.md) |
| Nsight Compute/kernel（3） | [ncu 第一轮](gpu-profiling/nsight-compute-first-pass.md) · [stall/访存](gpu-profiling/gpu-warp-stall-and-memory-analysis.md) · [naive SGEMM](gpu-profiling/gpu-naive-sgemm-bottleneck.md) |
| 工具/环境（4） | [PyTorch Profiler](gpu-profiling/pytorch-profiler-and-memory-snapshot.md) · [监控](gpu-profiling/gpu-monitoring-and-throttling.md) · [权限/容器](gpu-profiling/gpu-profiling-permissions-and-containers.md) · [跨平台](gpu-profiling/gpu-other-platform-tools.md) |
| 速查（1） | [命令与指标](gpu-profiling/gpu-profiling-cli-cheatsheet.md) |

## 怎么读一条经验

1. 先看 frontmatter 的 `description` 和 `symptoms`，确认命中。
2. 执行“快速判断”，先做只读、可回退、低成本的检查。
3. 按“排查步骤”缩小范围，记录每一步的观察。
4. 用“验证”确认修复，而不是看到报错消失就结束。
5. 条目是经验而非保证；版本行为冲突时以实测为准，并反馈更新条目。

新增条目请按 [CONTRIBUTING.md](../CONTRIBUTING.md) 操作，并运行 `python3 toolbox/build_catalog.py`。
