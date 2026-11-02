---
name: precision-and-fastmath
title: 精度与 fast-math：区分数值差异和真实 bug
description: 编译后结果与参考不一致、出现 NaN/inf、fp16/bf16/TF32 误差超预期时，先确认误差来源是数据类型、累加精度、TF32、fast-math 还是算法 bug，再决定修精度还是报编译器问题。
category: correctness
tags: [precision, tf32, fp16, bf16, fast-math, numerical]
symptoms:
  - float32 结果与参考差 1e-3 量级，疑似 TF32
  - fp16/bf16 训练出现 NaN、inf 或 loss 爆炸
  - 关闭 torch.compile 后误差消失，但找不到具体算子
  - 加长归约或增大 batch 后误差明显变大
  - 同一模型换 GPU 架构后精度不同
stacks: [torch.compile, triton, cuda, xla]
updated: 2026-11-02
---

# 精度与 fast-math：区分数值差异和真实 bug

> 一句话经验：先用更高精度参考算出“理论误差上限”，再对照 dtype、累加方式、TF32 和 fast-math 设置；超过上限才按 bug 查。

## 适用场景

- 输出与 eager/CPU 参考的差异在“可能是精度”的量级，需要判断是否可接受。
- fp16/bf16/TF32 训练出现 NaN、inf、梯度爆炸或收敛变差。
- 同一份代码在不同 GPU 架构、不同 PyTorch/Triton 版本上精度不同。
- 需要为测试设置合理的 `rtol`/`atol`，或决定是否关闭 fast-math。

不适用于明显的越界/竞态导致的错误：先按[静默错误结果](silent-wrong-results.md)做逐层对拍。

## 症状与信号

- 误差量级与 dtype 精度吻合：fp16 约 1e-3、bf16 约 1e-2、TF32 约 1e-3（相对误差，随规模变化）。
- 误差随归约长度/矩阵 K 维增大而累积，属于正常的浮点累加误差。
- 只有开启 `torch.compile`、autotune、融合或 TF32 时误差变大。
- 出现 NaN/inf：优先查溢出（fp16 上限约 65504）、除零、`log(0)`、`sqrt(负数)`、未初始化内存。
- 同一输入重复运行误差不同：不是精度问题，优先查竞态。

## 快速判断

1. **算误差画像**：最大绝对误差、最大相对误差、误差随规模的增长趋势。
2. **提高参考精度**：用 float64/CPU 参考，确认参考本身不是同精度。
3. **逐项关闭精度捷径**：TF32、fp16 reduced reduction、cudnn benchmark、fast-math。
4. **看 dtype 传播**：中间累加是否意外变成 fp16/bf16；`sum`/`mean`/`matmul` 的 accumulate dtype。
5. **看数值范围**：是否接近 dtype 上限、是否出现 denormal、是否 inf/NaN 参与运算。

## 排查步骤

### 1. 建立 float64 参考并量化误差

```python
import torch

torch.manual_seed(0)
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)

ref = (x.double() @ x.double()).float()
out = x @ x

diff = (out - ref).abs()
print("max abs:", diff.max().item())
print("mean abs:", diff.mean().item())
print("max rel:", (diff / ref.abs().clamp_min(1e-30)).max().item())
```

矩阵乘的误差会随 K 维增长；对比不同 K（256/1024/4096）看增长是否符合预期，比单个数值更有判断力。

### 2. 检查 PyTorch 的精度开关

```python
import torch

print("matmul precision:", torch.get_float32_matmul_precision())
print("matmul allow_tf32:", torch.backends.cuda.matmul.allow_tf32)
print("cudnn allow_tf32:", torch.backends.cudnn.allow_tf32)
print("cudnn benchmark:", torch.backends.cudnn.benchmark)
print("fp16 reduced reduction:",
      torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
```

- `torch.set_float32_matmul_precision("highest"|"high"|"medium")` 控制 float32 matmul 的精度策略；`high`/`medium` 可能启用 TF32 或 bf16 算法。
- `torch.backends.cuda.matmul.allow_tf32`、`torch.backends.cudnn.allow_tf32` 控制 TF32；不同版本默认值不同，必须显式打印确认。
- `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction`（以及 bf16 对应项）允许 fp16/bf16 matmul 用低精度累加，默认可能为 `True`。
- `torch.backends.cudnn.benchmark=True` 会让 cudnn 选不同算法，结果可能有细微差异，也可能在输入 shape 变化时抖动。

### 3. 检查归约与累加 dtype

```python
# 错误示例：fp16 输入，累加也在 fp16
s16 = x16.sum()                      # 可能溢出/误差大
s32 = x16.sum(dtype=torch.float32)   # 更稳
m16 = x16.mean(dtype=torch.float32)
```

- 大 K 归约、softmax、LayerNorm、loss 计算优先用 fp32 累加。
- 检查自定义 Triton/CUDA kernel 的累加器类型；`tl.sum` 的输入 dtype 决定累加精度。
- 检查 `torch.compile` 生成的代码是否把中间结果降精度（看 IR 或生成代码）。

### 4. 隔离 fast-math 与融合

- CUDA C++：`-use_fast_math` 会启用近似除法/开方、flush denormal、重结合；对比关闭后的结果。
- Triton：部分版本支持 `TRITON_F32_DEFAULT=ieee`（或 `tf32`/`tf32x3`）控制 float32 点积精度；先确认当前版本是否支持。
- 编译融合可能重排 `a*b+c` 为 FMA，改变舍入；这是允许的浮点优化，不是 bug。
- 如果关闭融合/fast-math 后误差回到预期，说明差异来自数值策略，优先调整容差或精度设置，而不是直接报编译器 bug。

### 5. 处理 NaN/inf

```python
torch.autograd.set_detect_anomaly(True)   # 定位反向传播中的 NaN 来源
```

- 用 `torch.isnan`/`torch.isinf`/`torch.isfinite` 定位第一次出现的张量和算子。
- fp16 上限约 65504；检查 exp、softmax、loss、梯度累积是否溢出。
- `log(0)`、`0/0`、`sqrt(负数)` 会产生 -inf/NaN；检查 mask 和 clamp 的位置。
- NaN 会通过 `0 * NaN = NaN` 传播；不要用 `nan_to_num` 掩盖，先定位来源。

## 根因

1. **dtype 精度不足**：fp16/bf16 的尾数位少，累加和大 K 归约误差明显。
2. **TF32 / reduced precision reduction**：默认或显式开启，相对误差约 1e-3。
3. **fast-math / 近似函数**：`rsqrt`、`exp`、除法近似、flush denormal。
4. **FMA/重结合**：编译器把 `a*b+c` 合并或重排，舍入不同但数学等价。
5. **算法本身的数值不稳定**：softmax 未减最大值、方差不加 epsilon、长链乘除。
6. **真实 bug**：越界、竞态、未初始化内存，表现也可能像“精度问题”，需要用对拍和 sanitizer 排除。

## 修复与规避

- 提高累加精度（fp32 accumulate）、softmax 减最大值、方差加 epsilon、损失用 log-sum-exp 形式。
- 对精度敏感的 matmul 关闭 TF32 或使用 `torch.set_float32_matmul_precision("highest")`。
- 测试容差按 dtype 和归约规模设定，并在注释里写明依据；不要用全局放宽的容差掩盖回归。
- 确认是编译器重排/近似导致且数学上可接受时，记录版本和设置，不必强行关闭优化。
- 确认是编译器 bug 时，保留最小复现和 IR，先关闭对应优化止损。

## 验证

- 用 float64 参考计算误差上限，覆盖不同规模（K=256/1024/4096）和 dtype。
- 覆盖极端值：0、±inf、NaN、denormal、接近 dtype 上限的值。
- 同一输入重复运行，确认误差稳定（不稳定说明是竞态，不是精度）。
- 训练场景看多个 step 的 loss/梯度，而不是只看第一步。
- 关闭/开启 TF32、fast-math 各跑一遍，记录差异，作为配置基线。

## 常见坑

- `torch.testing.assert_close` 的默认容差随 dtype 变化；对 float32 是 `rtol=1.3e-6, atol=1e-5`，对长归约往往过严，但不要因此全局放宽。
- 只看最大绝对误差：大值处的绝对误差天然更大，应同时看相对误差。
- 用同一个编译后的实现做参考，会把同样的舍入/近似当成“正确答案”。
- 不同 GPU 架构的 FMA/denormal/近似指令行为可能不同，换机器复现前先固定架构。
- 把 `nan_to_num`、`clamp` 当成修复：会改变语义并掩盖上游错误。

## 参考

- PyTorch 数值精度：<https://pytorch.org/docs/stable/notes/numerical_accuracy.html>
- `torch.set_float32_matmul_precision`：<https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html>
- PyTorch 可复现性（含确定性算法）：<https://pytorch.org/docs/stable/notes/randomness.html>
- CUDA C++ Programming Guide（浮点与 FMA）：<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions-appendix>
