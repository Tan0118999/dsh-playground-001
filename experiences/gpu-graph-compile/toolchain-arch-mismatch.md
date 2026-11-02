---
name: toolchain-arch-mismatch
title: 工具链与架构不匹配：no kernel image、sm_90a 未定义、驱动过旧
description: 编译成功但运行时报 no kernel image、invalid device function、CUDA driver version is insufficient，或 ptxas 不识别 sm_90a 时，按“运行时/驱动/ptxas/目标 GPU 架构”四层核对版本与 cubin/PTX，再决定重编译、补 PTX 还是升级驱动。
category: engineering
tags: [cuda, arch, ptxas, driver, compatibility, no-kernel-image]
symptoms:
  - no kernel image is available for execution on the device
  - CUDA driver version is insufficient for CUDA runtime version
  - ptxas fatal 提示 sm_90a/sm_100a 等架构值未定义
  - 编译机正常，目标机运行失败
  - 换 GPU 型号或容器后同一 wheel 不能跑
  - invalid device function / too many resources requested
stacks: [cuda, torch.compile, triton, tvm, xla]
updated: 2026-11-02
---

# 工具链与架构不匹配：no kernel image、sm_90a 未定义、驱动过旧

> 一句话经验：先把“编译时目标架构、运行时 GPU 架构、驱动支持的 PTX 版本、ptxas 版本”四者列出来对齐，再决定重编译还是升级环境。

## 适用场景

- wheel/镜像在开发机正常，部署到目标机报 `no kernel image` 或 `invalid device function`。
- 升级 GPU（如 Hopper/Blackwell）后旧环境无法运行。
- 编译时 `ptxas` 不认识 `sm_90a`/`sm_100a` 等架构。
- 容器内 CUDA runtime 与宿主机驱动版本不匹配。
- 多架构机器上同一份二进制行为不同。

## 症状与信号

- `no kernel image is available for execution on the device`（`cudaErrorNoKernelImageForDevice`, 209）：没有匹配当前 GPU 的 cubin，也没有可 JIT 的 PTX。
- `CUDA driver version is insufficient for CUDA runtime version`（`cudaErrorInsufficientDriver`, 35）：运行时比驱动新。
- `ptxas fatal: Value 'sm_90a' is not defined`：`ptxas` 版本早于该架构；`sm_90a` 是 Hopper 专属特性（wgmma/TMA 等），不能向前兼容。
- `invalid device function`：加载了不适用于当前设备的 kernel 映像。
- 同一 wheel 在 A100 正常、H100 失败，或反过来：架构列表没有覆盖目标 GPU。

## 快速判断

1. **列版本**：驱动、CUDA runtime、`nvcc`/`ptxas`、PyTorch/Triton/TVM 版本。
2. **列架构**：当前 GPU 的 compute capability（`sm_XX`），以及二进制里包含哪些 cubin/PTX。
3. **对目标机验证**：不要只在编译机上判断；在目标 GPU 上跑最小 kernel。
4. **看是否 `a` 架构**：`sm_90a`/`sm_100a` 只在该代芯片可用，不能指望在下一代上运行。
5. **确认 PTX fallback**：二进制是否带 PTX，驱动是否支持该 PTX 版本。

## 排查步骤

### 1. 采集版本与架构

```bash
nvidia-smi
nvcc --version
ptxas --version
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("capability:", torch.cuda.get_device_capability())
print("arch list:", torch.cuda.get_arch_list())
PY
```

- `nvidia-smi` 右上角的 `CUDA Version` 是驱动支持的最高 runtime 版本，不是已安装的 toolkit 版本。
- `torch.version.cuda` 是 PyTorch 编译时使用的 CUDA 版本；`torch.cuda.get_arch_list()` 列出 wheel 里包含的架构（如 `sm_80`、`sm_90`）。
- Triton/TVM 各自的编译目标架构可能独立于 PyTorch，需要分别确认。

### 2. 检查二进制里的 cubin 与 PTX

```bash
cuobjdump -lelf libkernel.so      # 列出内嵌的 ELF/cubin 架构
cuobjdump -lptx libkernel.so      # 列出内嵌的 PTX 版本
cuobjdump -sass libkernel.so | head
```

- 只有 `sm_80` 的 cubin，却在 H100（`sm_90`）上运行，且没有 PTX，就会报 `no kernel image`。
- 有 PTX 但驱动太旧、无法 JIT 到目标架构，同样失败。
- 对 `.cubin` 也可以直接 `cuobjdump -lelf kernel.cubin`。

### 3. 对齐编译目标

CUDA C++：

```bash
# 为 Hopper 生成 cubin；需要 CUDA 12.0+ 的 ptxas
nvcc -gencode arch=compute_90,code=sm_90 ...
# 需要 Hopper 专属特性（wgmma/TMA）时
nvcc -gencode arch=compute_90a,code=sm_90a ...
```

PyTorch 扩展 / 自定义算子：

```bash
# 官方文档形式：空格分隔，可加 +PTX
export TORCH_CUDA_ARCH_LIST="8.0 9.0+PTX"
```

- 近期 PyTorch 也广泛接受分号形式和 `9.0a` 后缀，但这不保证所有版本一致；设置后用 `torch.cuda.get_arch_list()` 或构建日志确认实际架构。
- `+PTX` 会嵌入 PTX，允许在更高架构上用 JIT 运行，但可能损失该架构的专用优化。
- 只列目标机器实际需要的架构，避免 wheel 体积和编译时间膨胀。

### 4. 对齐驱动与 runtime

- 驱动版本必须 ≥ runtime 要求的版本；容器内通常只装 runtime，驱动来自宿主机。
- 驱动过旧时，升级驱动，或使用 NVIDIA 的 forward compatibility / `cuda-compat` 包（具体支持范围以版本矩阵为准）。
- 不要只看 `nvidia-smi` 的 CUDA Version 就认为 toolkit 匹配；用 `nvcc --version` 和 `torch.version.cuda` 分别确认。
- 多架构集群要按节点架构构建或分发对应 wheel；一份二进制覆盖所有架构会增加体积，但可避免部署错配。

### 5. 在目标机做最小验证

```bash
python3 -c "import torch; a=torch.randn(8, device='cuda'); print((a+a).sum().item())"
```

再用实际 kernel/模型跑一次；只验证 import 或 `cuda.is_available()` 不足以发现 `no kernel image`。

## 根因

1. **架构列表缺失**：编译只覆盖了部分 `sm_XX`，目标 GPU 不在其中。
2. **PTX 缺失或过新**：没有 PTX fallback，或 PTX 版本高于驱动可 JIT 的版本。
3. **驱动过旧**：runtime 比驱动新，或容器驱动与宿主机不匹配。
4. **`a` 架构误用**：把 `sm_90a` 二进制部署到非 Hopper，或以为它向前兼容。
5. **缓存旧产物**：编译缓存里仍是旧架构的 cubin，改配置后没有失效。
6. **多版本工具链混用**：`nvcc`、`ptxas`、`cuda runtime`、PyTorch 内置 CUDA 版本不一致。

## 修复与规避

- 补全架构列表并重新编译；需要 PTX fallback 时加 `+PTX`。
- 升级驱动或改用匹配的 runtime；容器场景确认宿主机驱动版本。
- `sm_90a`/`sm_100a` 特性单独构建，部署时按 GPU 型号选择，不要混用。
- 清理旧缓存后重编译，避免加载旧 cubin。
- 无法升级驱动时，降级 runtime 或使用兼容包；记录版本矩阵。

## 验证

- `cuobjdump -lelf`/`-lptx` 确认目标架构和 PTX 存在。
- 在目标 GPU 上跑最小 kernel 和实际模型，覆盖冷启动。
- 多架构集群按节点抽样验证，不要只测一台。
- 记录 `nvidia-smi`、`nvcc --version`、`torch.version.cuda`、`torch.cuda.get_arch_list()`，作为可复现的环境基线。

## 常见坑

- 把 `nvidia-smi` 的 `CUDA Version` 当成已安装 toolkit 版本。
- 在编译机上验证通过就认为部署没问题；编译机和运行机的架构/驱动可能不同。
- 只加 `+PTX` 不重新编译 cubin，导致目标架构无法使用专用指令，性能下降。
- 容器里升级了 runtime 但宿主机驱动没变，仍然报驱动不足。
- 缓存目录里残留旧架构产物，改 `TORCH_CUDA_ARCH_LIST` 后没有清理。
- 多 GPU 混合架构机器上，`CUDA_VISIBLE_DEVICES` 指到了错误型号。

## 参考

- CUDA 兼容性指南：<https://docs.nvidia.com/deploy/cuda-compatibility/index.html>
- CUDA 二进制工具（`cuobjdump`）：<https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html>
- PyTorch `TORCH_CUDA_ARCH_LIST`：<https://pytorch.org/docs/stable/cpp_extension.html>
- PyTorch CUDA 语义：<https://pytorch.org/docs/stable/notes/cuda.html>
