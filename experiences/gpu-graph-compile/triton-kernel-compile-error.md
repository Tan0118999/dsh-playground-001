---
name: triton-kernel-compile-error
title: Triton kernel 编译错误：从前端、MLIR 到 ptxas 的分层定位
description: Triton kernel 报 CompilationError、MLIR verifier 错误、OutOfResources 或 PTXASError 时，先判断错误发生在前端/TTIR-TTGIR 优化/LLVM/PTX/ptxas 哪一层，再用 kernel dump、reproducer、override 和 interpreter 定位到具体约束。
category: compile
tags: [triton, mlir, ptxas, compilation-error, gpu]
symptoms:
  - triton.compiler.errors.CompilationError，附带源码位置和 caret
  - MLIR 报 'tt.load'/'tt.dot' 等 op 的 verifier 错误
  - out of resource: shared memory, Required: N, Hardware limit: M
  - PTXASError，ptxas stderr 里是寄存器/共享内存/架构问题
  - 同样的 kernel 在别的 GPU 或 triton 版本能编译
stacks: [triton]
updated: 2026-11-02
---

# Triton kernel 编译错误：从前端、MLIR 到 ptxas 的分层定位

> 一句话经验：先看异常类型和报错阶段，再用 dump/override 把问题钉到某个 pass；不要一上来就改 kernel 逻辑。

## 适用场景

- `triton.jit` kernel 编译失败，或换 GPU 架构、升级 Triton 后编译失败。
- 需要判断是 kernel 写法问题、资源超限，还是编译器/ptxas 的问题。
- 需要向上游提交可复现的 MLIR reproducer。

## 症状与信号

| 异常 | 阶段 | 典型含义 |
| --- | --- | --- |
| `CompilationError` | Python AST/前端或 MLIR verifier | 语法、类型、shape、op 约束不满足；通常带源码 caret |
| `OutOfResources` | TTGIR/资源分配 | shared memory 超限；错误文本会给出 Required 和 Hardware limit |
| `PTXASError` | PTX → cubin | ptxas 失败，含 ptxas stderr 和 repro command |
| `MLIR` 报 `error: 'tt.xxx' op ...` | TTIR/TTGIR pass | 某个 op 的 verifier 或 pass 不变量失败 |
| 编译卡住/极慢 | 某 pass 或 autotune | 需要 `MLIR_ENABLE_TIMING` 或减少 autotune |

## 快速判断

1. **看异常类型**：`CompilationError`、`OutOfResources`、`PTXASError` 分别对应不同层。
2. **读完整报错**：Triton 会打印源码片段和 caret；先定位到具体行，再看是类型、shape 还是资源。
3. **绕过缓存**：`TRITON_ALWAYS_COMPILE=1`，排除旧产物。
4. **关掉异常包装**：`TRITON_FRONT_END_DEBUGGING=1`，拿到完整 traceback。
5. **用 interpreter 验逻辑**：`TRITON_INTERPRET=1` 先把“写法/索引”问题和“GPU 编译”问题分开。

## 排查步骤

### 1. 判断错误阶段

```bash
TRITON_FRONT_END_DEBUGGING=1 TRITON_ALWAYS_COMPILE=1 python3 run.py
```

- 报错在 `triton/compiler/compiler.py` 的 `ast_to_ttir` 附近：前端/AST 问题（类型、`constexpr`、不支持的语法）。
- 报错在 `ttir_to_ttgir`/`ttgir_to_llir`：MLIR pass 或 op 约束问题。
- 报错在 `ptxas`：资源、架构或 PTX 生成问题。
- 只有 cache 命中时失败：清缓存或 `TRITON_ALWAYS_COMPILE=1` 后重试。

### 2. Dump 每个阶段的 IR

```bash
rm -rf /tmp/triton_dump
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/triton_dump python3 run.py
ls -t /tmp/triton_dump | head
```

产物按源码 hash 分目录，包含 `.ttir`、`.ttgir`、`.llir`、`.ptx`、`.cubin`。找到失败阶段的前后文件，对比 pass 前后的 op。

需要看“每个 pass 之前”的 IR：

```bash
MLIR_ENABLE_DUMP=1 MLIR_DUMP_PATH=/tmp/mlir_dump python3 run.py
```

- `MLIR_ENABLE_DUMP=1` 会 dump 所有 pass，输出量巨大；也可以设成某个 kernel 名，只 dump 指定 kernel。
- 缓存可能让 dump 不触发；先清 `~/.triton/cache` 或用 `TRITON_ALWAYS_COMPILE=1`。
- `MLIR_ENABLE_TIMING=1` 可定位编译时间花在哪个 pass。

### 3. 生成 reproducer

```bash
TRITON_REPRODUCER_PATH=/tmp/triton_repro.mlir python3 run.py
```

发生崩溃时，该文件保存失败 pass 之前的状态，可直接提交给上游或本地重放。注意：`TRITON_REPRODUCER_ALWAYS` 不存在，不要引用。

### 4. 用 override 做单点实验

```bash
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/dump python3 run.py
cp -r /tmp/dump/<src_hash> /tmp/override/<src_hash>
# 编辑 /tmp/override/<src_hash>/kernel.ttgir
TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=/tmp/override python3 run.py
```

- dump 目录按源码 hash 命名，cache 目录按完整 cache key 命名，两者不能互相拷贝。
- override 成功时会打印 `Overriding kernel with file <path>`；没打印说明路径/哈希不对。

### 5. 用 interpreter 分离逻辑问题

```bash
TRITON_INTERPRET=1 python3 run.py
```

- 在 CPU 上用 numpy 逐个 program 执行，适合验证索引、mask、归约逻辑。
- 已知限制：不支持 bfloat16、不支持间接/指针追逐访存；共享内存语义不建模（不能用来验证 bank conflict 或 swizzle）；atomics 可用但语义与 GPU 不完全一致。
- interpreter 通过但 GPU 编译失败，说明问题在编译/资源/架构；两者都失败，先修 kernel 逻辑。

### 6. 读 ptxas 错误

`PTXASError` 会包含 ptxas stderr 和 repro command。常见原因：

- `out of resource: shared memory, Required: N, Hardware limit: M`：减小 `BLOCK_SIZE`、`num_stages`，或把大 tile 拆开。
- 寄存器溢出：ptxas 默认带 `-v`，日志里有寄存器数和 spill；用 `TRITON_DUMP_PTXAS_LOG=1` 保留日志，必要时用 `PTXAS_OPTIONS` 添加选项。
- 架构不支持：例如目标 GPU 不支持某条指令；检查 Triton 版本和 `sm_XXa` 特性。
- PTX 版本与驱动不匹配：见[工具链与架构不匹配](toolchain-arch-mismatch.md)。

### 7. 常见 kernel 约束

- `tl.arange(0, n)` 的 `n` 必须是 2 的幂；block 大小通常也需要是 2 的幂。
- `tl.dot` 对 M/N/K 的最小维度和 dtype 有约束（常见要求 ≥ 16，随版本/精度变化）；不满足时改用其他实现或 padding。
- 动态 shape 需要显式 `constexpr` 或运行时参数；不要把 Python 值混进 kernel 的 tensor 逻辑。
- mask 的 dtype 要和 `other` 匹配；`other=0` 与 `other=-inf` 在 softmax 中语义完全不同。
- 条件分支里调用 `tl.dot`/`tl.load` 时，确保所有 program 的 shape 一致。

## 根因

1. **shape/block 不满足约束**：非 2 的幂、`tl.dot` 维度太小、mask 形状不匹配。
2. **资源超限**：shared memory、寄存器、线程数超过硬件限制。
3. **类型错误**：dtype 不支持、累加器类型不匹配、`constexpr` 使用错误。
4. **MLIR verifier 失败**：op 的操作数数量/类型/shape 不符合定义，通常是前端生成了非法 IR。
5. **架构/工具链**：Triton 版本、ptxas 版本、目标 `sm_XX` 不匹配。
6. **缓存/产物问题**：旧 cubin 或 dump 与当前源码不一致，误导判断。

## 修复与规避

- 先改 kernel 的 tile/shape/dtype，再考虑升级工具链。
- 资源超限优先减小 tile/`num_stages`，而不是盲目加 `num_warps`。
- 确认是编译器 bug 时，用 `TRITON_REPRODUCER_PATH` 生成最小 MLIR，附上 Triton/ptxas/GPU 版本提交上游。
- 规避：换用更小的 tile、拆分 kernel、关闭 autotune；记录性能代价。
- 不要用 `try/except` 吞掉 `CompilationError` 后回退到未验证的实现，除非有等价性测试。

## 验证

- `TRITON_ALWAYS_COMPILE=1` 下能稳定编译，换新缓存目录也能编译。
- `cuobjdump -res-usage` 确认寄存器/共享内存在预期内。
- 数值结果与参考实现一致（覆盖边界 shape）。
- 在目标 GPU 架构上实际运行，不只验证编译。
- 如果改了 tile，重新测性能，避免“编译过了但更慢”。

## 常见坑

- 只清 `~/.triton/cache` 不够：`TRITON_CACHE_DIR`、`TRITON_DUMP_DIR`、`TRITON_OVERRIDE_DIR` 可能在不同位置。
- `MLIR_ENABLE_DUMP=1` 输出巨大，可能拖慢或撑爆磁盘；用 `MLIR_DUMP_PATH` 写到文件并限制范围。
- 用 interpreter 验证性能或共享内存行为：它不建模这些。
- 把 `OutOfResources` 当成驱动问题：先看 Required 与 Hardware limit 的差值。
- 在错误的 GPU 架构上编译，再用另一张卡运行；见[工具链与架构不匹配](toolchain-arch-mismatch.md)。

## 参考

- Triton debugging guide：<https://triton-lang.org/main/programming-guide/chapter-3/debugging.html>
- Triton README "Tips for hacking"（dump/override）：<https://github.com/triton-lang/triton#tips-for-hacking>
- Triton `tl.dot`：<https://triton-lang.org/main/python-api/generated/triton.language.dot.html>
- Triton `debug_barrier`/`device_print`：<https://triton-lang.org/main/python-api/triton.language.html>
