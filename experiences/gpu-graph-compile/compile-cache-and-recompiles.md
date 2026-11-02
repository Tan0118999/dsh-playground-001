---
name: compile-cache-and-recompiles
title: 编译缓存与重编译：每次启动都变慢、cache 目录暴涨
description: torch.compile/JAX/Triton 首次编译慢、每次启动都重编译、recompile 计数暴涨、cache 目录膨胀或编译 OOM 时，用 recompiles 日志、TORCH_TRACE/tlparse、动态 shape 标注和缓存目录排查。
category: engineering
tags: [torch.compile, cache, recompile, dynamic-shapes, compile-time]
symptoms:
  - 每次启动都要重新编译，第二次也不快
  - TORCH_LOGS=recompiles 显示同一函数反复重编译
  - torch._dynamo hit config.cache_size_limit / recompile_limit
  - 编译缓存目录持续膨胀或占满磁盘
  - 多进程训练时每个 rank 各编译一份，启动很慢
stacks: [torch.compile, triton, jax]
updated: 2026-11-02
---

# 编译缓存与重编译：每次启动都变慢、cache 目录暴涨

> 一句话经验：先区分“没命中缓存”和“根本不该重复编译”；前者查目录/进程，后者查动态 shape 和 guard。

## 适用场景

- `torch.compile`/JAX/Triton 首次编译时间很长，或每次启动都重新编译。
- 同一函数因输入 shape/dtype 变化反复重编译，最终触发 `cache_size_limit` 回退 eager。
- 编译缓存目录（Inductor/Triton/XLA）过大、无法复用、多进程重复编译。
- 需要在不牺牲正确性的前提下缩短冷启动。

## 症状与信号

- `TORCH_LOGS=recompiles` 输出 `Recompiling function ... triggered by the following guard failure(s)`，并指出 shape/dtype 差异。
- 警告 `torch._dynamo hit config.cache_size_limit (8)`；2.8+ 改名为 `recompile_limit`。
- 同一模型不同 batch/序列长度各编译一次；服务 latency 随输入抖动。
- 多进程/多卡日志里每个 rank 都在编译，缓存目录出现多份相似产物。
- 编译时间占比高，但 kernel 执行时间正常。

## 快速判断

1. **确认是否真的重编译**：`TORCH_LOGS=recompiles`，看 guard failure 的具体维度。
2. **确认缓存目录**：打印 `TORCHINDUCTOR_CACHE_DIR`、Triton cache、XLA cache 实际路径。
3. **确认动态 shape 策略**：`dynamic=True/None`、`mark_dynamic` 是否覆盖真实输入范围。
4. **确认进程模型**：多进程是否共享缓存目录，是否有写权限和文件锁。
5. **确认不是调试开关**：`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` 会直接禁用缓存；`TORCH_COMPILE_DEBUG=1` 会增加产物并改变编译时序，比较启动时间时要关掉。

## 排查步骤

### 1. 看重编译原因

```bash
TORCH_LOGS=recompiles python3 run.py 2>&1 | tee recompiles.log
TORCH_LOGS=recompiles_verbose python3 run.py
```

- `recompiles` 只打印首次失败的 guard；`recompiles_verbose` 打印全部。
- 典型输出会指出 `tensor 'x' size mismatch at index 0, expected 3, actual 4`，直接告诉你哪个维度在变。
- 结合 `TORCH_LOGS=graph_breaks`，排除 graph break 导致的分段编译。

### 2. 用 TORCH_TRACE/tlparse 看全貌

```bash
TORCH_TRACE=/tmp/tracedir python3 run.py
pip install tlparse
tlparse /tmp/tracedir
```

生成 HTML 报告，可以看到每个 frame 的编译次数、graph break、编译错误和生成代码，比在终端里 grep 日志更直观。

### 3. 处理动态 shape

```python
import torch

torch._dynamo.mark_dynamic(x, 0, min=1, max=64)
compiled = torch.compile(model, dynamic=True)
```

- `dynamic=True` 让 Dynamo 使用符号 shape；`None`（默认）在首次重编译后自动转动态；`False` 强制静态。
- `mark_dynamic(t, index, min=, max=)` 要在编译函数外对输入调用；范围要覆盖真实分布，否则仍会重编译。
- `maybe_mark_dynamic` 只在观察到变化时标记；`mark_static` 强制静态。
- 动态 shape 处理不好会引入额外的 guard/分支，性能和编译时间都要实测。
- 对服务场景，按业务 shape 分桶（padding 到固定桶）通常比完全动态更稳。

### 4. 检查重编译上限

2.4–2.7：

```python
import torch._dynamo.config as config
print(config.cache_size_limit)              # 默认 8
print(config.accumulated_cache_size_limit)  # 默认 256
```

- 命中上限后 Dynamo 会回退 eager，表现为“越跑越慢”。
- 2.8+ 改名为 `recompile_limit`/`accumulated_recompile_limit`，旧名保留为别名。
- **不要**直接把上限调大当作修复：那只是让重编译继续发生、消耗更多内存/时间；先消除 shape 变化原因。

### 5. 检查缓存目录

```bash
python3 - <<'PY'
import tempfile, os
print("tmp:", tempfile.gettempdir())
print("TORCHINDUCTOR_CACHE_DIR:", os.environ.get("TORCHINDUCTOR_CACHE_DIR"))
print("TRITON_CACHE_DIR:", os.environ.get("TRITON_CACHE_DIR"))
print("JAX_COMPILATION_CACHE_DIR:", os.environ.get("JAX_COMPILATION_CACHE_DIR"))
PY
```

- Inductor 默认在系统临时目录下 `torchinductor_<user>`；临时目录被清理会导致每次冷编译。
- Triton 默认 `~/.triton/cache`，可用 `TRITON_CACHE_DIR` 覆盖；Inductor 的 Triton 子目录在 `$TORCHINDUCTOR_CACHE_DIR/triton/<device>`。
- XLA/JAX 持久化缓存用 `JAX_COMPILATION_CACHE_DIR`，默认有最小编译时间阈值（约 1 秒），太小的编译不会入缓存。
- 缓存目录要有稳定持久化存储；容器重建、临时盘回收都会让缓存失效。

### 6. 多进程与共享缓存

- 每个进程默认各自编译；多卡训练时 N 个 rank 可能编译 N 份相同 kernel。
- 可以让所有 rank 指向同一个持久化缓存目录，但要注意并发写和文件锁；共享文件系统上的锁行为要实测。
- 服务化场景可在启动时用固定输入预热，把编译成本移到健康检查/启动阶段。
- 缓存目录被多个版本/配置共享时，key 可能包含版本/环境，升级后自然失效；不要手工混用不同版本的产物。

### 7. 缩短编译时间

- 避免/减少 `max-autotune` 等昂贵模式，或只在离线调优时启用。
- 减少 graph break 和不必要的动态 shape，让图更完整。
- 预编译/预热常用 shape，避免请求路径上首次编译。
- 用 `TORCH_COMPILE_DEBUG=1` 的 `compile.prof`/`compile.stats` 看编译时间花在哪个阶段（这是 cProfile，不是执行时间线）。
- 编译 OOM 时减小图/模型、拆分编译、增加内存；编译 worker 的峰值内存通常高于训练进程。

## 根因

1. **输入 shape/dtype/device 变化**：guard 失效，触发重编译。
2. **Python 侧状态变化**：全局变量、随机数、可变默认参数被 guard 捕获。
3. **缓存目录不持久/不共享**：临时目录被清理，或多进程各自编译。
4. **缓存被禁用**：`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`、调试开关、只读目录。
5. **上限触发回退**：`cache_size_limit` 命中后回退 eager，看起来像“编译没生效”。
6. **版本/配置变化**：缓存 key 变化导致全部 miss。

## 修复与规避

- 消除根因：按业务 shape 分桶、固定 dtype/device、避免 Python 状态进入图。
- 需要动态 shape 时显式 `mark_dynamic` 并覆盖真实范围；不要只调大重编译上限。
- 把缓存目录设到持久化路径，多进程共享前先验证并发安全。
- 预热/预编译常用 shape；把编译成本移出请求路径。
- 规避：临时提高 `cache_size_limit`、减少 autotune；记录内存/时间代价，不作为长期方案。

## 验证

- 同一进程第二次调用不触发重编译（`TORCH_LOGS=recompiles` 无新记录）。
- 新进程从冷启动能命中持久化缓存，启动时间显著下降。
- 覆盖真实 shape 分布，确认重编译次数在可接受范围。
- 缓存目录大小可控，没有无限增长；清理策略明确。
- 正确性回归：缓存复用不能改变结果。

## 常见坑

- 每次启动前清缓存“保证干净”，结果永远冷编译；只在排查缓存污染时清。
- 把 `cache_size_limit` 调大后不再回退，但重编译仍在发生，延迟抖动更大。
- 多进程共享缓存目录但文件系统不支持可靠锁，出现半成品产物。
- 用 `torch._dynamo.utils.counters` 这类内部结构做长期监控；它可能随版本变化，优先用日志和 `compile_times()`。
- 把 `compile.prof` 当成执行性能分析；它只记录编译阶段。
- 认为“第二次快”就一定命中缓存：也可能只是页缓存/预热，换进程再验证。

## 参考

- torch.compile caching 教程：<https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html>
- torch.compile FAQ（编译慢、重编译）：<https://docs.pytorch.org/docs/stable/torch.compiler_faq.html>
- 动态 shape：<https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html>
- `TORCH_LOGS`：<https://docs.pytorch.org/docs/stable/logging.html>
- JAX persistent compilation cache：<https://docs.jax.dev/en/latest/persistent_compilation_cache.html>
