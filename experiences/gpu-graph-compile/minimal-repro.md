---
name: minimal-repro
title: 最小复现与配置二分：把 GPU 图编译问题缩到可提交
description: 需要定位或向上游报告 GPU 图编译问题时，用“固定输入—剥离模型—配置二分—版本二分”得到单文件可运行的最小复现，并收集日志、IR、环境矩阵和期望/实际结果。
category: methodology
tags: [repro, bisect, torch.compile, triton, tvm, xla, debugging]
symptoms:
  - 大模型能复现、小图不能复现
  - 换个 shape/进程/机器就不复现
  - 需要向 PyTorch/Triton/TVM/XLA 上游报 bug
  - 不知道该收集哪些证据才能让别人复现
stacks: [torch.compile, triton, tvm, xla, cuda]
updated: 2026-11-02
---

# 最小复现与配置二分：把 GPU 图编译问题缩到可提交

> 一句话经验：先让失败“稳定、单文件、可一键运行”，再谈根因；一个可靠的复现比十个猜测更有价值。

## 适用场景

- 已确认是编译/执行链路的问题，但模型太大、依赖太多，无法快速迭代。
- 需要向上游提交 bug，或让同事在另一台机器复现。
- 需要判断某个配置/版本是否引入问题。

## 症状与信号

- 原始脚本依赖训练数据、多卡/NCCL、长时间 step，单次迭代就要几分钟。
- 报错只在特定 shape、dtype、GPU 架构或配置组合下出现。
- 复现率不稳定，别人按描述无法重现。
- 已经尝试关闭某些优化，但不确定是哪一项导致问题。

## 快速判断

1. 先用原始脚本记录一次完整失败日志和环境版本，确保“失败”本身可观察。
2. 固定 seed、shape、dtype 和输入生成方式，让失败率可量化。
3. 从外到内剥离模型/训练循环，每一步都验证“仍然失败”。
4. 用单变量配置二分找到触发开关，再用版本二分锁定引入点。
5. 最后整理成一条命令可运行的复现包。

## 最小复现的验收标准

一条合格的复现应满足：

1. **单文件**：一个脚本（或一个目录 + 一条命令）能跑。
2. **依赖明确**：只依赖框架和必要的库，`env.txt` 记录精确版本。
3. **稳定失败**：连续运行多次都能复现；偶发问题要给出复现率。
4. **最小输入**：固定 seed、固定 shape/dtype/device，不依赖外部数据。
5. **有期望/实际**：打印或断言期望结果与实际结果，而不是只“报错”。
6. **可开关**：能通过一个 flag 在“失败配置”和“正常配置”之间切换，证明差异来源。

## 排查步骤

### 1. 冻结环境

```bash
python3 -V > env.txt
nvidia-smi >> env.txt
nvcc --version >> env.txt
pip freeze >> env.txt
python3 - <<'PY' >> env.txt
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("arch list", torch.cuda.get_arch_list())
print("device", torch.cuda.get_device_properties(0))
PY
```

同时记录：容器镜像、驱动版本、`TORCH_CUDA_ARCH_LIST`、`CUDA_VISIBLE_DEVICES`、多卡拓扑、相关环境变量（`TORCH_LOGS`、`TRITON_*`、`XLA_FLAGS`）。

### 2. 固定输入与随机性

```python
import torch

def make_input(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(8, 128, device="cuda", generator=g)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
x = make_input()
```

- 输入用确定性生成；不要用真实数据集或随机 DataLoader。
- 竞态问题不要只跑一次，记录 50–100 次的失败率。
- 如果问题依赖特定数值（NaN/inf/极端值），显式构造这些值。

### 3. 剥离模型

从外到内逐层删除，保持失败：

1. 把训练循环缩成单次 forward/backward。
2. 用 `model.eval()`、`torch.no_grad()` 排除 dropout/BN/优化器状态。
3. 按 block/layer 删除，直到只剩触发问题的子图。
4. 把子图替换成等价的单算子/最小 kernel。
5. 保留最小的 autograd 路径（如果问题只在 backward 出现）。

每一步都确认“仍然失败”；删过头导致不复现时，回退到上一步并换一个维度继续删。

### 4. 配置二分

一次只改一项，记录结果：

- `torch.compile`：`mode="default"` vs `"reduce-overhead"`/`"max-autotune"`；`dynamic=True/False`；`fullgraph=True`；关闭 cudagraph/autotune/融合。
- Triton：`num_warps`、`num_stages`、`BLOCK_SIZE`；`TRITON_ALWAYS_COMPILE=1`；是否命中缓存。
- TVM：`opt_level=0`、`disabled_pass`、target `llvm` vs `cuda`。
- XLA：`--xla_disable_hlo_passes`、dump 前后对比、CPU vs GPU。
- 环境：单卡 vs 多卡、单流 vs 多流、不同 GPU 架构。

用表格记录：配置 | 结果 | 失败率 | 备注。二分时优先切分“可疑维度”，不要同时改多项。

### 5. 版本二分

- 用容器/虚拟环境固定其他依赖，只改一个组件（PyTorch、Triton、TVM、XLA、驱动）。
- 对编译器仓库本身，用 `git bisect` 找到引入问题的 commit。
- 记录每个版本的结果，不要只写“升级后坏了”。

### 6. 收集工具产物

**torch.compile**

```bash
TORCHDYNAMO_REPRO_AFTER=aot \
TORCHDYNAMO_REPRO_LEVEL=4 \
TORCH_COMPILE_DEBUG=1 \
python3 repro.py
```

- minifier 产物在 `torch_compile_debug/minifier/`，`repro.py` 会复制到当前目录。
- `TORCH_COMPILE_DEBUG=1` 还会保存 `fx_graph_runnable.py`、`output_code.py`、`ir_pre_fusion.txt`/`ir_post_fusion.txt`、`compile.prof` 等。
- `TORCHDYNAMO_REPRO_LEVEL=3` 适合段错误（即使没抛异常也 dump）。

**Triton**

```bash
TRITON_ALWAYS_COMPILE=1 \
TRITON_REPRODUCER_PATH=/tmp/repro.mlir \
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/triton_dump \
python3 repro.py
```

- reproducer 保存失败 pass 之前的 MLIR 状态；dump 目录里有 `.ttir/.ttgir/.llir/.ptx/.cubin`。
- 不要引用不存在的 `TRITON_REPRODUCER_ALWAYS`。

**TVM**

```python
import tvm
# 保存 IRModule；版本相关 API 以当前版本为准
tvm.ir.save_json(mod, "mod.json")
print(lib.get_source("cuda"))
```

- 不同 TVM 版本 API 差异很大（`tvm.lower` 已在新版本移除）；复现里写明版本。

**XLA/JAX**

```python
print(jax.jit(f).lower(x).as_text("hlo"))
```

```bash
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text --xla_dump_hlo_as_html" python3 repro.py
```

- `as_text()` 默认是 StableHLO，要 HLO 需显式传 `"hlo"`。
- 不要使用不存在的 `--xla_dump_stage`。

### 7. 组织复现包

推荐结构：

```text
repro/
├── repro.py          # 一条命令可运行
├── run.sh            # 固定环境变量与命令
├── env.txt           # 版本矩阵
├── logs/             # 完整 stdout/stderr
├── ir/               # HLO/TTGIR/FX graph/output_code
└── README.md         # 现象、期望/实际、复现率、已排除项
```

`README.md` 里写清：什么现象、怎么运行、期望结果、实际结果、失败率、已尝试的规避、相关版本。

## 根因

最小复现不是找根因，而是把搜索空间压到可管理：

1. **无关复杂度**：模型层数、数据加载、训练循环掩盖了触发条件。
2. **非确定性**：随机输入、竞态、autotune 让失败不可稳定观察。
3. **环境漂移**：版本/驱动/架构差异导致“你这也复现不了”。
4. **配置组合**：多个优化叠加，单独关闭任一项都不复现。

## 修复与规避

- 先用最小复现确认修复有效，再回到完整模型验证。
- 上游报告时附复现包和版本矩阵，不要只贴一段 traceback。
- 规避：关闭触发问题的优化、固定 shape/架构；在复现里保留开关，方便对比。

## 验证

- 在一台干净机器/新容器里按 `run.sh` 能稳定复现。
- 复现脚本在“正常配置”下通过、在“失败配置”下失败。
- 覆盖至少两个 shape/seed，确认不是单点偶然。
- 修复后复现脚本转为通过，并保留为回归测试。

## 常见坑

- 用真实数据集做复现：数据加载和随机性会掩盖问题。
- 只保留最后一次报错，不保存完整日志和 IR。
- 把“换机器不复现”当成问题消失：可能是架构/驱动/缓存差异。
- 在复现里开着调试工具（sanitizer、profiler、`TORCH_COMPILE_DEBUG`）改变时序，导致竞态消失或变慢。
- 删代码时没有逐步验证，最后得到一个与原问题无关的“新 bug”。
- 忘了记录失败率，把 1% 的偶发问题当成必现。

## 参考

- PyTorch minifier/`TORCHDYNAMO_REPRO_*`：<https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting_old.html>
- `TORCH_COMPILE_DEBUG`：<https://docs.pytorch.org/docs/stable/torch.compiler_faq.html>
- Triton debugging（dump/reproducer）：<https://triton-lang.org/main/programming-guide/chapter-3/debugging.html>
- XLA HLO dumps：<https://openxla.org/xla/hlo_dumps>
- TVM 错误处理：<https://tvm.apache.org/docs/errors.html>
