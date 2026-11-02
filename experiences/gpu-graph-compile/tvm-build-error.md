---
name: tvm-build-error
title: TVM 构建与调度错误：Check failed、IR 定位和版本 API 差异
description: TVM 报 Check failed、InternalError、ScheduleError 或 tvm.build 失败时，先用当前版本的 API 打印 lowered TIR/CUDA 源码，再用 opt_level=0 和 disabled_pass 二分定位；同时注意 tvm.lower、debug_executor 等旧教程 API 已被移除。
category: compile
tags: [tvm, tir, schedule, build, compilation-error]
symptoms:
  - TVMError / InternalError / Check failed，附带文件行号和 TIR 片段
  - ScheduleError 提示某个 axis 无法 fuse/compute_at/vectorize
  - tvm.build 在 CUDA target 上失败，但 LLVM target 正常
  - 按旧教程调用 tvm.lower / tvm.contrib.debugger 报 AttributeError
  - 升级 TVM 后同一份 schedule 代码不能编译
stacks: [tvm]
updated: 2026-11-02
---

# TVM 构建与调度错误：Check failed、IR 定位和版本 API 差异

> 一句话经验：先确认 TVM 版本和 API 是否存在，再打印 lowered IR 定位到具体 op/axis；很多“报错”其实是照了旧版教程的 API。

## 适用场景

- TE/TensorIR schedule 报 `ScheduleError`、`Check failed`、`InternalError`。
- `tvm.build` 在 CUDA target 上失败，或生成的代码运行崩溃/结果错误。
- 从旧项目/教程迁移到新版本 TVM 时出现 `AttributeError` 或行为变化。
- 需要把问题缩到一个算子、一个 schedule 原语。

## 症状与信号

- `Check failed: ...`：通常是 C++ `ICHECK`/`LOG(FATAL)`，会带 `file:line` 和当前 TIR。
- `ScheduleError`：某个 schedule 原语的前提不满足，错误里通常打印相关 PrimFunc/Stmt。
- `TVMError`/`InternalError`：可能是 pass 不变量、类型检查或后端 codegen 失败。
- `AttributeError: module 'tvm' has no attribute 'lower'`：版本 API 已变化（`tvm.lower` 在 v0.21 起移除）。
- CUDA target 失败但 `llvm` target 正常：问题更可能在 GPU codegen/thread binding/内存 scope，而不是 schedule 本身。

## 快速判断

1. **先看版本**：`tvm.__version__`，确认 `tvm.lower`、`tvm.build`、`tvm.contrib.debugger` 等 API 是否存在。
2. **读报错附件**：`Check failed` 的 `file:line` 和 TIR 片段通常已经指出哪个 op/axis。
3. **降 opt_level**：`PassContext(opt_level=0)` 看问题是否由某个优化 pass 引入。
4. **换 target**：先用 `llvm` 构建，分离“schedule/IR 问题”和“CUDA codegen 问题”。
5. **缩到单算子**：只保留失败的 compute 和 schedule 原语，去掉融合/并行化。

## 排查步骤

### 1. 确认 API 与版本

```python
import tvm
print(tvm.__version__)
for name in ("lower", "build", "compile", "tirx"):
    print(name, hasattr(tvm, name))
```

重要版本差异（以官方 CHANGELOG/源码为准）：

| API | 状态 |
| --- | --- |
| `tvm.lower` | v0.19 仍可用；v0.21 起移除 |
| `tvm.build` | v0.27 仍存在但已弃用，委托给 `tvm.tirx.build` |
| `tvm.contrib.debugger` / `debug_executor` | v0.21 起移除，旧教程的逐算子计时不可用 |
| `TVM_BACKTRACE` | 旧版环境变量；当前版本错误已自带 backtrace，只有 `TVM_BACKTRACE_LIMIT` |
| `tvm.ir.save_json` / `load_json` | 仍可用 |

不要直接照搬旧教程；先用 `hasattr` 或当前版本文档确认。

### 2. 打印 lowered IR 与生成源码

旧版本（≤ v0.19）：

```python
s = tvm.te.create_schedule(op.op)
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

新版本（≥ v0.21，`tvm.lower` 已移除）：按当前版本的 `tvm.compile`/`tvm.tirx` API 获取 lowered IR；TE 路径仍可用 `tvm.build`（弃用但可用）：

```python
lib = tvm.build(s, [A, B, C], target="cuda")
print(lib.get_source("cuda"))
```

- `get_source("cuda")` 能看到 thread binding、shared memory、循环结构，是定位 GPU codegen 问题的最直接材料。
- TIR/IRModule 可以 `tvm.ir.save_json(mod, "mod.json")` 保存，便于对比/上报。

### 3. 用 PassContext 二分

```python
with tvm.transform.PassContext(opt_level=0):
    lib = tvm.build(s, [A, B, C], target="cuda")
```

- `opt_level=0` 通过后，再逐级提高，找到引入问题的优化级别。
- `disabled_pass=["..."]` 可以禁用具体 pass；可用的 pass 名称以当前版本为准，不要凭旧文档猜。
- 需要看 pass 配置项时，在 TVM 源码里搜索 `TVM_REGISTER_PASS_CONFIG_OPTION`，确认 key 存在再用。

### 4. 打开 C++ 日志

```bash
TVM_LOG_DEBUG="DEFAULT=2" python3 run.py
# 或只开某个文件（路径相对 src/）：
TVM_LOG_DEBUG="ir/transform.cc=2,te/schedule/schedule_lang.cc=2" python3 run.py
```

- 格式是逗号分隔的 `file=level`；`DEFAULT=<n>` 设置默认级别；负值可关闭某个文件。
- 文件名要相对 `src/`，例如 `ir/transform.cc`，不要写 `src/ir/transform.cc`。
- 日志量可能很大，先设 `DEFAULT=1` 或用单文件级别。

### 5. 缩到单算子并换 target

```python
# 1) 只保留失败的 compute
# 2) 去掉 fuse/compute_at/parallel/vectorize，逐个加回
# 3) 先 target="llvm" 构建并运行，再切 target="cuda"
```

- `llvm` 通过、`cuda` 失败：重点查 thread binding、shared/local scope、`vectorize` 宽度、目标 `-arch`。
- 两边都失败：重点查 schedule 轴、buffer shape/dtype、auto-scheduler 约束。
- 对 `ScheduleError`，检查 `fuse` 的轴是否相邻、`compute_at` 的 scope 是否合法、`vectorize` 的 extent 是否满足对齐。

### 6. 检查 target 与架构

```python
target = tvm.target.Target("cuda -arch=sm_90")
```

- 交叉编译时 `-arch` 必须匹配目标 GPU；不匹配会在加载/运行时失败（见[工具链与架构不匹配](toolchain-arch-mismatch.md)）。
- 多 GPU 环境确认 `tvm.cuda(0)` 的 device id 与 `CUDA_VISIBLE_DEVICES` 一致。

## 根因

1. **schedule 原语前提不满足**：轴不连续、scope 不合法、extent 不是常量等。
2. **shape/dtype 不一致**：buffer 声明与实际 tensor 不符，或 `te.var` 推断失败。
3. **GPU codegen/thread binding 问题**：`bind` 的线程轴与 block 维度不匹配、共享内存超限。
4. **优化 pass 触发不变量**：`opt_level` 提高后某个 pass 暴露了非法 IR。
5. **版本 API/行为变化**：旧教程的 `tvm.lower`、`debug_executor` 已移除。
6. **target/arch 不匹配**：编译目标与运行 GPU 不同。

## 修复与规避

- 先修 schedule/IR 的非法前提，不要靠关闭所有优化绕过。
- 需要逐算子定位时，用当前版本支持的 profiling 手段（如 `tvm.runtime.Profiler`/`time_evaluator` 做整体计时）；旧 `debug_executor` 已不可用。
- 版本迁移时，先跑官方对应版本的教程/测试，确认 API 映射，再改业务代码。
- 规避：`opt_level=0`、单 target、禁用特定 pass；记录性能/功能代价。

## 验证

- `llvm` 和 `cuda` 两个 target 都能构建；CUDA 在目标 GPU 上实际运行。
- 数值结果与 NumPy/参考实现一致（覆盖边界 shape）。
- `get_source("cuda")` 检查 thread binding、shared memory 和循环结构符合预期。
- 升级 TVM 后重新跑构建 + 数值 + 性能基线，不只验证 import。

## 常见坑

- 照抄旧教程的 `tvm.lower`/`tvm.contrib.debugger`，在新版本直接 `AttributeError`。
- 把 `Check failed` 当成 TVM 内部 bug：多数是 schedule/IR 前提不满足，报错里的 TIR 片段就是线索。
- 只调 `opt_level=0` 让问题消失就结案：可能是某个优化 pass 暴露了非法 IR，根因仍在。
- 在编译机上用 `-arch` 编译，部署到不同架构 GPU。
- `TVM_LOG_DEBUG` 路径写成绝对路径或 `src/` 前缀，导致匹配不到、日志为空。

## 参考

- TVM 错误处理：<https://tvm.apache.org/docs/errors.html>
- TVM 日志与调试：<https://tvm.apache.org/docs/contribute/error_handling.html>
- TVM `tvm.build`/`tvm.compile` API：<https://tvm.apache.org/docs/reference/api/python/driver.html>
- TVM `tvm.ir.save_json`：<https://tvm.apache.org/docs/reference/api/python/ir.html>
