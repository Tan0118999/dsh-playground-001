---
name: xla-compile-error
title: XLA/JAX 编译错误：HLO dump、形状错误与显存预分配
description: JAX/XLA 报 XlaRuntimeError、INVALID_ARGUMENT、RESOURCE_EXHAUSTED、INTERNAL 或编译缓慢时，用 as_text 和 XLA_FLAGS dump 出 HLO，区分 trace/形状、编译优化、显存预分配和数值问题。
category: compile
tags: [xla, jax, hlo, compilation-error, dump]
symptoms:
  - jax.errors.JaxRuntimeError / XlaRuntimeError
  - INVALID_ARGUMENT，错误里带大段 HLO
  - RESOURCE_EXHAUSTED，但单步显存看起来没超
  - 编译时间很长或卡住，首次调用特别慢
  - 同一函数因 shape 变化反复重新编译
  - GPU 上出现 NaN，eager 参考却正常
stacks: [xla, jax]
updated: 2026-11-02
---

# XLA/JAX 编译错误：HLO dump、形状错误与显存预分配

> 一句话经验：先区分“trace 阶段的 Python 错误”“XLA 编译错误”“运行时 OOM/数值错误”，再拿 HLO 定位到具体 op 和 shape。

## 适用场景

- JAX/XLA 在 `jit` 编译或首次执行时报错。
- 错误信息包含 HLO、`INVALID_ARGUMENT`、`RESOURCE_EXHAUSTED`。
- 编译时间异常、缓存不生效、shape 变化导致反复编译。
- 需要看优化前后的 HLO，判断是哪个 pass/op 引入问题。

## 症状与信号

- `XlaRuntimeError` 包装 XLA status：`RESOURCE_EXHAUSTED`（多为显存）、`INVALID_ARGUMENT`（shape/dtype/layout）、`INTERNAL`/`UNIMPLEMENTED`（编译器内部或不支持）。
- 错误信息里出现完整 HLO 文本：XLA 常把相关模块打印出来，直接搜 `%` 变量和 shape。
- 首次调用慢、后续快：编译与 autotune；如果每次都慢，检查缓存和动态 shape。
- 只有某个 shape/设备数失败：优先看 sharding、padding、layout。
- GPU 上 NaN 但 CPU/eager 正常：先查精度和 `JAX_DEBUG_NANS`。

## 快速判断

1. **拿 HLO**：`jax.jit(f).lower(*args).as_text('hlo')` 看未优化 HLO；`.compile().as_text()` 看优化后 HLO。
2. **dump 全流程**：`XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"`，看 `before_optimizations`/`after_optimizations`。
3. **判断阶段**：错误发生在 `jit` 调用时（编译）还是第一次执行时（运行/OOM）。
4. **查显存**：`XLA_CLIENT_MEM_FRACTION`、`XLA_PYTHON_CLIENT_PREALLOCATE`，区分预分配和真实峰值。
5. **查数值**：`JAX_DEBUG_NANS=1`、`JAX_DEBUG_INFS=1`。

## 排查步骤

### 1. 打印 HLO

```python
import jax

f = jax.jit(model_fn)
lowered = f.lower(x)
print(lowered.as_text("hlo"))          # 未优化 HLO；默认 dialect 是 stablehlo
compiled = lowered.compile()
print(compiled.as_text())              # 优化后 HLO；后端不提供时可能返回 None
print(compiled.cost_analysis())
print(compiled.memory_analysis())
```

- `as_text()` 默认输出 StableHLO；要看 HLO 必须显式传 `"hlo"`。
- `Lowered.compiler_ir("hlo")` 可拿到对象，再调用其文本接口；不同版本 API 略有差异。
- 也可以给单次编译传 dump 选项，而不是全局设 `XLA_FLAGS`。

### 2. dump 优化过程

```bash
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump \
  --xla_dump_hlo_as_text \
  --xla_dump_hlo_as_html \
  --xla_dump_hlo_pass_re=.*" \
python3 run.py
```

- 文件名通常包含 `before_optimizations`/`after_optimizations` 和 pass 名；先看优化前后差异。
- `--xla_dump_hlo_pass_re=.*` 会 dump 每个 pass，目录可能很大；先限定到可疑 pass。
- 不设 `--xla_dump_to` 而只设其他 dump flag 时，内容可能直接输出到 stdout。
- **`--xla_dump_stage` 不是当前 XLA 的真实 flag**，不要照旧资料使用；以 `--help`/源码里的 dump flags 为准。
- 生成的 HTML 适合用浏览器看模块结构和 layout，文本适合 grep shape/op。

### 3. 按错误码分流

| 错误 | 常见原因 | 动作 |
| --- | --- | --- |
| `RESOURCE_EXHAUSTED` | 显存预分配/峰值超限、编译期常量太大 | 调 `XLA_CLIENT_MEM_FRACTION`、关预分配、减小 batch |
| `INVALID_ARGUMENT` | shape/dtype/layout 不匹配、sharding 不一致 | 对照 HLO 里的 shape 和输入 |
| `INTERNAL` | 编译器内部不变量失败、版本 bug | 最小复现 + HLO + 版本上报 |
| `UNIMPLEMENTED` | 当前后端不支持某 op/dtype | 换实现/精度，或退回 CPU 验证 |

显存相关：

```bash
# 新版变量名；旧名 XLA_PYTHON_CLIENT_MEM_FRACTION 已弃用
XLA_CLIENT_MEM_FRACTION=.5 python3 run.py
XLA_PYTHON_CLIENT_PREALLOCATE=false python3 run.py
```

- 默认预分配约占 GPU 显存的 75%；`RESOURCE_EXHAUSTED` 可能发生在预分配阶段，而不是模型真实峰值。
- `XLA_PYTHON_CLIENT_ALLOCATOR=platform` 会按需分配/释放，能跑但很慢，仅用于确认 OOM。

### 4. 查 trace/形状问题

- JAX 是静态图：Python 的 `if`/`for`/`.shape` 在 trace 时只执行一次；tensor 值驱动的分支要用 `jax.lax.cond`/`scan`/`while_loop`。
- shape 变化会触发重新编译；需要动态 batch 时用 padding 或 `jax.jit` 的静态参数设计。
- `jax.debug.print`/`jax.debug.breakpoint` 会引入 host 交互，可能影响编译/执行，不要留在性能路径。
- 多设备：检查 `jax.devices()`、sharding 和 `pjit`/`shard_map` 的输入布局是否一致。

### 5. 查数值

```bash
JAX_DEBUG_NANS=1 JAX_DEBUG_INFS=1 python3 run.py
```

- 开启后 JAX 会在每个 op 后检查 NaN/inf，并重跑未编译版本以定位来源；会显著变慢。
- 需要更高精度参考时：`jax.config.update("jax_enable_x64", True)`（会改变计算图/性能）。
- 用 `jax.debug.print` 打印中间值（受异步执行影响，加 `ordered=True` 可保序）。

### 6. 检查编译缓存

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache python3 run.py
```

- 持久化缓存默认有最小编译时间阈值（约 1 秒），太小的编译不会入缓存。
- 缓存被信任，可能掩盖代码变化；调试时换新目录，确认问题不是旧缓存。
- 用 `JAX_DEBUG_LOG_MODULES`、`jax_explain_cache_misses=True` 观察 miss 原因（版本相关）。

## 根因

1. **trace 语义错误**：Python 控制流/形状依赖，导致图与预期不符。
2. **shape/dtype/layout 不匹配**：输入、sharding、padding 不一致。
3. **显存预分配/峰值**：默认 75% 预分配，或编译期常量/中间 buffer 过大。
4. **不支持的 op/dtype**：后端未实现，报 `UNIMPLEMENTED`。
5. **编译器 bug**：`INTERNAL`/不变量失败，需要最小复现。
6. **缓存/版本**：旧缓存、JAX/XLA/驱动版本组合问题。

## 修复与规避

- trace 问题：用 `jax.lax` 控制流替代 Python 分支；固定 shape 或显式 padding。
- OOM：减小 batch/序列长度、调 `XLA_CLIENT_MEM_FRACTION`、关闭预分配确认峰值、检查编译期常量。
- 不支持 op：换等价实现或精度，必要时回退 CPU 验证语义。
- 编译器 bug：保留 `lower().as_text('hlo')`、dump 目录、版本矩阵，提交上游。
- 规避：关闭预分配、降低并行度、禁用某些 pass（需确认 flag 存在）；记录性能代价。

## 验证

- 在目标设备上从冷启动跑通，覆盖不同 shape/batch 和多设备。
- 用 `--xla_dump_to` 确认优化后 HLO 符合预期，没有意外的 layout 转换/拷贝。
- 数值与 float64/CPU 参考对拍，明确容差。
- 编译时间与缓存命中：第二次启动应显著更快；若不变，检查缓存目录和阈值。
- 多设备场景验证 sharding 后结果与单设备一致。

## 常见坑

- 把 `as_text()` 默认输出当成 HLO：默认是 StableHLO，需要显式 `"hlo"`。
- 使用不存在的 `--xla_dump_stage` 或旧变量名 `XLA_PYTHON_CLIENT_MEM_FRACTION`（新版本已弃用）。
- 把预分配 OOM 当成模型真实峰值，盲目减小模型。
- dump 所有 pass 导致磁盘爆满/编译变慢；先限定 pass 或模块。
- 用旧编译缓存验证“已修复”，实际跑的还是旧图。
- 在 `jit` 内使用 Python `print`/`.item()`/shape 分支，导致 trace 行为与预期不同。

## 参考

- XLA HLO dumps：<https://openxla.org/xla/hlo_dumps>
- XLA 错误码：<https://openxla.org/xla/error_codes>
- JAX debugging：<https://docs.jax.dev/en/latest/201/debugging.html>
- JAX GPU memory：<https://docs.jax.dev/en/latest/201/gpu-memory.html>
- JAX persistent compilation cache：<https://docs.jax.dev/en/latest/persistent_compilation_cache.html>
