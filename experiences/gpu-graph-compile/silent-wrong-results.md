---
name: silent-wrong-results
title: 静默错误结果：不报错但算错了
description: 编译和运行都没有异常，但输出与 eager/CPU/float64 参考不一致，或偶发错误、精度下降、NaN。按“建立参考—逐层对拍—关闭优化—极端输入”的顺序定位，区分真实 bug 与数值差异。
category: correctness
tags: [correctness, torch.compile, triton, numerical, debugging]
symptoms:
  - eager 结果正确，torch.compile/JIT 后结果错误
  - CPU 正确、GPU 错误，或小 shape 正确、大 shape 错误
  - 没有报错，但 loss/梯度异常、输出出现 NaN 或 inf
  - 同一输入重复运行，结果偶尔不同
  - 误差不大但下游任务明显退化
stacks: [torch.compile, triton, tvm, xla, cuda]
updated: 2026-11-02
---

# 静默错误结果：不报错但算错了

> 一句话经验：先建立可信参考并量化误差，再逐层缩小；不要在整图输出上直接猜某个 kernel 有问题。

## 适用场景

- 编译后的图能跑完，但结果与参考不一致。
- 精度下降、偶发 NaN/inf、偶发错误，或只在特定 shape/dtype/设备组合下出错。
- 需要判断是“编译器/生成代码的 bug”还是“本来就存在的数值差异/未定义行为”。

不适用于明确的崩溃或非法访存：先按[非法访存与异步错误定位](illegal-memory-access.md)处理。

## 症状与信号

- 误差是稳定的（每次相同）还是随机的（竞态、未初始化内存、原子操作顺序）。
- 误差集中在尾部元素、padding 区、特定通道/头，还是全局均匀。
- 只有 `float16`/`bfloat16` 错，还是 `float32` 也错。
- 只有融合/autotune/cudagraph 开启时错，还是 eager 也错。
- 首次正确、后续错误：优先怀疑状态复用、buffer 覆盖、多流未同步。

## 快速判断

1. **建立参考**：用 eager 和（可行时）CPU/float64 跑同一输入，确认参考本身可信。
2. **量化误差**：看最大绝对误差、相对误差、误差分布和首个出错位置，不只看一个数。
3. **单算子对拍**：逐层比较，找到第一处显著偏离的算子；融合后无法逐层时，先关闭融合。
4. **重复运行**：同一输入跑 5–10 次，结果变化说明存在竞态或未初始化读。
5. **关闭优化**：依次关闭 cudagraph、autotune、融合，判断哪项引入错误。
6. **极端输入**：加入 0、±inf、NaN、denormal、极大值、边界 shape，暴露被正常数据掩盖的问题。

## 排查步骤

### 1. 固定输入并建立参考

```python
import torch

torch.manual_seed(0)
x = torch.randn(8, 128, device="cuda", dtype=torch.float32)

with torch.no_grad():
    ref = eager_model(x)          # 参考一：eager
    out = compiled_model(x)       # 参考二：编译后

print(torch.max(torch.abs(out - ref)))
print(torch.isnan(out).sum(), torch.isinf(out).sum())
torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
```

`assert_close` 失败时会给出最大差异的位置和数值，比手写 `allclose` 更有用。参考实现尽量用更高精度（float64）或独立实现，避免“用同一个 bug 验证自己”。

### 2. 逐层定位第一处偏离

```python
ref_acts, out_acts = {}, {}

def make_hook(store, name):
    def hook(_module, _inputs, output):
        store[name] = output.detach()
    return hook

for name, module in eager_model.named_modules():
    module.register_forward_hook(make_hook(ref_acts, name))
for name, module in compiled_model.named_modules():
    module.register_forward_hook(make_hook(out_acts, name))

for name in ref_acts:
    if name not in out_acts:
        continue
    diff = (ref_acts[name] - out_acts[name]).abs().max().item()
    if diff > 1e-4:
        print("first divergence:", name, diff)
        break
```

注意：编译/融合可能改变模块边界，hook 数量对不上；此时以算子为单位，而不是模块。

### 3. 单独验证可疑 kernel

- 把该 kernel 的输入保存下来，单独运行，和参考实现对比。
- Triton 可先用 `TRITON_INTERPRET=1` 在 CPU 上跑同一 kernel（见[Triton 编译错误](triton-kernel-compile-error.md)）。
- 用 `torch.testing.assert_close` 对拍时，明确容差与 dtype 的关系；不要用默认容差判断 `float16` 的大量累加。

### 4. 二分配置与 IR

依次关闭可疑优化，每次只改一项并记录结果：

```bash
# 关闭 cudagraph、autotune、融合等（具体开关以当前版本为准）
python3 run.py --no-cudagraph --no-autotune
```

- `torch.compile`：用 `TORCH_COMPILE_DEBUG=1` 保存生成的代码/IR，逐版本比较；关闭 cudagraph 后先排除捕获期 buffer 复用。
- Triton：保存 `.ttir`/`.ttgir`/`.ptx`，对比优化前后；用 `TRITON_KERNEL_DUMP=1`（见[Triton 编译错误](triton-kernel-compile-error.md)）。
- TVM：旧版可用 `tvm.lower` 打印 TIR（新版本已移除，需用当前版本 API）；逐 pass 关闭用 `PassContext(disabled_pass=[...])`。
- XLA：`--xla_dump_to` 保存 HLO，用 `--xla_disable_hlo_passes` 二分（flag 是否存在以当前版本为准，见[XLA/JAX 编译错误](xla-compile-error.md)）。

### 5. 用极端输入暴露未定义行为

```python
cases = [
    torch.zeros_like(x),
    torch.full_like(x, 1e4),
    torch.full_like(x, float("inf")),
    torch.full_like(x, float("nan")),
]
```

- inf/NaN 参与代数化简时，`0 * inf = NaN`、重结合、快速数学都可能改变结果。
- 大量 0 或极端值容易暴露未初始化共享内存、越界读和错误的 mask。
- 边界 shape（非 2 的幂、尾部不足一个 tile、batch=1）容易暴露越界和 mask 错误。

## 根因

按经验排序：

1. **数值精度**：fp16/bf16 累加、TF32、fast-math、FMA 重结合、denormal 被 flush。见[精度与 fast-math](precision-and-fastmath.md)。
2. **竞态与未初始化内存**：缺失同步、跨 warp 依赖、复用共享内存、越界读写。见[数据竞争与同步](race-and-sync.md)。
3. **边界与任务映射错误**：mask 条件写反、尾部处理错误、grid/block 与 shape 不匹配。
4. **布局/步长错误**：把非连续张量当成连续、转置后 stride 假设错误、broadcast 维度错位。
5. **编译器优化 bug**：代数化简、循环变换、向量化在特定模式下的不变量被破坏；通常需要最小复现才能确认。
6. **状态/图问题**：cudagraph 复用 buffer、in-place 更新、RNG 状态、多流未同步。

## 修复与规避

- 边界/同步/精度问题直接修根因，不要用裁剪或容差掩盖。
- 确认是编译器 bug 时，保留最小复现、IR、版本矩阵，先关闭对应优化止损，再向上游报告。
- 不要用 `nan_to_num`、`clamp` 或放宽容差来“让测试通过”；这些会掩盖真实错误并影响下游。
- 用更高精度累加（fp32 accumulate）或分块归约改善精度，是合理的工程修复。

## 验证

- 覆盖多个 seed、边界 shape（尾部、非 2 的幂、batch=1）、dtype（fp32/fp16/bf16）和极端值。
- 同一输入重复运行至少 10 次，确认结果稳定。
- 与 eager/float64 参考对拍，容差按 dtype 和归约规模设定，并在测试里写明依据。
- 若修的是竞态，额外跑 `compute-sanitizer --tool racecheck`（见[数据竞争与同步](race-and-sync.md)）。

## 常见坑

- `torch.testing.assert_close` 的默认容差对 float32 是 `rtol=1.3e-6, atol=1e-5`，对长归约或 fp16 往往过严；但放宽前要确认差异确实来自精度。
- 只看最大绝对误差，会漏掉大面积的小偏差；同时看相对误差和分布。
- 在同一台机器、同一个缓存、同一套随机数据上“复现”，会把缓存和随机性一起带进来。
- 用 GPU 上的同一实现做参考，会把同样的 bug 当成正确行为。
- 训练场景下，第一次迭代正确不代表后续正确：优化器状态、in-place buffer、梯度累积都可能引入问题。

## 参考

- `torch.testing.assert_close`：<https://docs.pytorch.org/docs/stable/testing.html>
- PyTorch 可复现性说明：<https://pytorch.org/docs/stable/notes/randomness.html>
- PyTorch 数值精度说明：<https://pytorch.org/docs/stable/notes/numerical_accuracy.html>
- XLA HLO dump：<https://openxla.org/xla/hlo_dumps>
