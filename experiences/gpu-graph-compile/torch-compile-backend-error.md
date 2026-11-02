---
name: torch-compile-backend-error
title: torch.compile 后端报错：BackendCompilerFailed、graph break 与重编译
description: torch.compile 首次运行抛出 BackendCompilerFailed、SubgraphLoweringException、OperatorIssue 或 Inductor 内部异常时，按“Dynamo 捕获—AOTAutograd—Inductor lowering/codegen”分层定位，并用 explain、日志、ablation backend 和 minifier 缩到最小复现。
category: compile
tags: [torch.compile, dynamo, inductor, graph-break, recompile]
symptoms:
  - torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised
  - SubgraphLoweringException / LoweringException / OperatorIssue
  - torch.compile 后首次运行报错，eager 正常
  - 日志里大量 graph break，或 recompile 次数异常
  - 升级 PyTorch 后原本能编译的模型失败
stacks: [torch.compile]
updated: 2026-11-02
---

# torch.compile 后端报错：BackendCompilerFailed、graph break 与重编译

> 一句话经验：先看 `BackendCompilerFailed` 包住的真正异常，再用 ablation backend 判断是 Dynamo 捕获、AOTAutograd 还是 Inductor 代码生成的问题。

## 适用场景

- `torch.compile` 首次执行时抛异常，eager 模式正常。
- 报错被 `BackendCompilerFailed` 包装，看不到真正的根因。
- 模型能跑但 graph break 很多、recompile 次数暴涨、性能不及预期。
- 需要向上游提交最小复现。

## 症状与信号

- `torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:`：后端异常被包装，真正原因在 `__cause__`/内层 traceback。
- `SubgraphLoweringException`、`LoweringException`、`OperatorIssue`：Inductor lowering 阶段的算子/形状问题（2.4–2.7 常见）。
- `InductorError`：约 2.9 起才作为独立异常类出现；旧版本不要按这个名字找。
- `TORCH_LOGS=graph_breaks` 输出 `Graph break in user code at file:line` 和 `Reason: ...`。
- `TORCH_LOGS=recompiles` 输出 `Recompiling function ... triggered by guard failure`。
- 命中 `cache_size_limit` 后回退 eager，表现为“越跑越慢”。

## 快速判断

1. **找内层异常**：从 `BackendCompilerFailed` 的 `__cause__` 或完整 traceback 里找最后一个用户代码帧。
2. **分层 ablation**：分别用 `backend="eager"`、`backend="aot_eager"`、`backend="inductor"` 跑同一输入。
3. **看捕获质量**：`torch._dynamo.explain(model)(x)` 看 graph 数、break 数、break 原因。
4. **看日志**：`TORCH_LOGS="+dynamo,+inductor"`、`TORCH_LOGS=graph_breaks`、`TORCH_LOGS=recompiles`。
5. **最小化**：设置 `TORCHDYNAMO_REPRO_AFTER=aot` 和 `TORCHDYNAMO_REPRO_LEVEL=4` 让 minifier 产出复现。

## 排查步骤

### 1. 分层 ablation

```python
import torch

for backend in ("eager", "aot_eager", "inductor"):
    try:
        fn = torch.compile(model, backend=backend)
        fn(x)
        print(backend, "OK")
    except Exception as e:
        print(backend, "FAILED:", type(e).__name__, e)
```

- `eager` 失败：问题在 Dynamo 捕获/前端，不是 Inductor。
- `aot_eager` 失败：问题在 AOTAutograd（autograd/fake tensor/分解）。
- 只有 `inductor` 失败：问题在 Inductor lowering/codegen，继续看 `TORCH_COMPILE_DEBUG=1` 的产物。

### 2. 捕获诊断

```python
expl = torch._dynamo.explain(model)(x)
print(expl)                 # Graph Count / Graph Break Count / Op Count / Break Reasons
print(expl.break_reasons)
print(expl.ops_per_graph)
```

`explain` 会在调用前后重置 Dynamo 状态并执行一次函数，适合分析捕获，不要放在性能路径里。

graph break 的常见原因：

- 不支持的 Python 内建/第三方库调用（`open`、`print`、日志）。
- 数据依赖控制流（`if tensor.item() > 0`、动态 shape 的 Python 分支）。
- `.item()`、`.cpu()`、`torch.nonzero()` 等导致同步/动态形状的算子。
- 自定义 op 没有 fake/meta 实现。
- 原地修改、全局状态、异常路径。

### 3. 打开日志

```bash
TORCH_LOGS="+dynamo,+inductor" python3 run.py 2>&1 | tee compile.log
TORCH_LOGS=graph_breaks python3 run.py
TORCH_LOGS=recompiles python3 run.py
TORCH_LOGS=perf_hints python3 run.py    # cudagraph 跳过等原因
TORCH_LOGS="+inductor" TORCH_LOGS_OUT=/tmp/inductor.log python3 run.py
```

- `TORCH_LOGS` 用逗号分隔，`+component` 是 DEBUG，裸组件是 INFO，`-component` 是 ERROR。
- `perf_hints` 默认关闭，必须显式写出。
- 日志可能非常大，先重定向到文件再 grep。

### 4. 保存编译产物

```bash
TORCH_COMPILE_DEBUG=1 python3 run.py
```

产物默认写到 `$CWD/torch_compile_debug/`（可用 `TORCH_COMPILE_DEBUG_DIR` 改根目录），每个图一个子目录，常见内容：

- `fx_graph_readable.py`、`fx_graph_runnable.py`：可独立运行的图复现。
- `fx_graph_transformed.py`、`ir_pre_fusion.txt`、`ir_post_fusion.txt`：图变换与融合前后 IR。
- `output_code.py`：生成的 Triton/C++ 代码。
- `compile.prof`/`compile.stats`：编译耗时的 cProfile（不是执行时间线）。
- `aot_<graph>_debug.log`：AOTAutograd 日志。

先看 `fx_graph_runnable.py` 能否独立复现，再看 `output_code.py` 是否生成了非法代码。

### 5. 动态 shape 与 guard

```python
torch._dynamo.mark_dynamic(x, 0, min=1, max=64)
```

- `torch.compile(dynamic=True)` 启用动态 shape；默认 `None` 表示首次重编译后自动转动态。
- `torch.compile` 没有 `dynamic_shapes=` 参数；`dynamic_shapes` 是 `torch.export` 的概念。
- `mark_dynamic` 要在编译函数外、对输入调用；如果该维度被 specialize，会报错。
- 动态 shape 处理不好会导致重复编译（见[编译缓存与重编译](compile-cache-and-recompiles.md)）。

### 6. 生成最小复现

```bash
TORCHDYNAMO_REPRO_AFTER=aot \
TORCHDYNAMO_REPRO_LEVEL=4 \
python3 run.py
```

- `TORCHDYNAMO_REPRO_AFTER` 取 `dynamo` 或 `aot`，分别在图捕获边界/AOTAutograd 之后包装后端。
- `TORCHDYNAMO_REPRO_LEVEL`：1 只 dump `repro.py`；2（默认）dump `minifier_launcher.py`；3 即使没崩溃也 dump；4 做精度最小化（仅 Inductor）。
- 产物在 `torch_compile_debug/minifier/`，`repro.py` 会复制到当前目录。
- 不要引用不存在的 `torch._dynamo.config.repro_path`/`TORCHDYNAMO_REPRO_PATH`。

### 7. 常见根因与修复

| 根因 | 信号 | 修复 |
| --- | --- | --- |
| 不支持的 op/库调用 | graph break，`Reason: Unsupported` | 改写为可捕获操作，或用 `torch._dynamo.allow_in_graph` 标记纯函数 |
| 自定义 op 缺 fake/meta | `UnsupportedOperator`/fake tensor 错误 | 用 `torch.library.custom_op` 注册并实现 fake/meta |
| 动态 shape/guard | 重编译、`size mismatch` | `mark_dynamic`、固定 shape、避免 Python 形状分支 |
| 数据依赖控制流 | `Data-dependent jump` | 用 `torch.cond`/掩码实现 |
| dtype/device 不一致 | lowering 报类型错误 | 统一输入 dtype/device，避免隐式转换 |
| 生成代码非法 | `output_code.py` 报错 | 缩到单算子，作为 Inductor bug 上报 |
| 编译期 OOM | 编译 worker 被杀 | 减小图/模型、增加内存、拆分编译 |

## 根因

1. **Dynamo 前端**：不支持的 Python 语义、graph break、fake tensor 传播失败。
2. **AOTAutograd**：自定义 autograd function、原地修改、分解规则缺失。
3. **Inductor lowering**：算子/形状/dtype 组合没有 lowering 规则或生成非法代码。
4. **运行时假设**：动态 shape、别名、内存布局、cudagraph 兼容性。
5. **版本差异**：异常类型、config 名称、默认值随版本变化（如 `InductorError` 约 2.9 才有）。

## 修复与规避

- 优先改写模型/算子让图完整捕获；其次用 `torch.library.custom_op` 补齐 fake/meta。
- 对暂时不支持的子图，允许 graph break 回退 eager，但要评估性能；`suppress_errors=True` 会静默回退，不适合作为最终方案。
- 动态 shape 用 `mark_dynamic`/固定 padding；不要在编译函数内用 Python 分支处理 shape。
- 确认是编译器 bug 时，用 minifier 产物 + `output_code.py` + 版本矩阵上报，并先关闭对应优化止损。
- 规避：`fullgraph=False`、换 backend、关闭 cudagraph/autotune；记录功能/性能代价。

## 验证

- `torch._dynamo.explain` 的 graph break 数、graph 数符合预期；没有意外回退 eager。
- 第二次运行不再重编译（`TORCH_LOGS=recompiles` 无新记录）。
- 与 eager 结果对拍，覆盖动态 shape 和边界输入。
- 编译时间、端到端性能与基线对比；确认修复没有引入新的 graph break。
- 在目标 PyTorch 版本上验证，不假设异常类名/配置名跨版本一致。

## 常见坑

- 只看 `BackendCompilerFailed` 最后一层，忽略 `__cause__` 里的真正异常。
- 用 `suppress_errors=True` 让报错消失，实际回退到 eager，性能和覆盖都变差。
- 把 graph break 当成编译失败：它只是回退，过多才影响性能。
- 在 2.4–2.7 找 `InductorError` 类：该名称约 2.9 才出现，旧版本应看 `BackendCompilerFailed`/`LoweringException`。
- 给 `torch.compile` 传 `dynamic_shapes=`：它不是该 API 的参数。
- 清理缓存或重启进程后“好了”，但没有确认根因和重编译行为。

## 参考

- torch.compile troubleshooting：<https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html>
- 旧版 troubleshooting（minifier/explain/TORCH_COMPILE_DEBUG）：<https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting_old.html>
- `TORCH_LOGS`：<https://docs.pytorch.org/docs/stable/logging.html>
- 动态 shape：<https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html>
- torch.compile FAQ：<https://docs.pytorch.org/docs/stable/torch.compiler_faq.html>
