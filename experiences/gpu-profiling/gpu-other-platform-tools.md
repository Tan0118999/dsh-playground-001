---
name: gpu-other-platform-tools
title: 不止 Nsight：AMD、Intel、Apple、TPU 和图形工具怎么选
description: 手头不是 NVIDIA GPU、要分析图形管线、或不确定该用 torch profiler / Nsight / 厂商工具时读这条。给出按问题选工具的表、AMD rocprofv3、Intel unitrace、Apple Metal System Trace、TPU/NPU 工具和边界。
category: engineering
tags: [amd, rocm, intel, apple, metal, tpu, nsight-graphics, tooling]
symptoms:
  - 手头是 AMD/Intel/Apple GPU
  - 不知道什么时候用 torch profiler
  - 图形管线问题
  - TPU/NPU 上怎么 profile
  - 工具太多不知道选哪个
stacks: [pytorch, rocm, sycl, metal, tpu]
updated: 2026-11-02
---

# 不止 Nsight：跨平台与图形工具怎么选

> 一句话经验：**先看问题层次（端到端/单算子/图形管线），再选平台工具；不同厂商的“利用率”口径通常不同。**

## 适用场景

- 不在 NVIDIA GPU 上；
- 需要分析图形渲染，而不是 compute kernel；
- 想用更轻量的 PyTorch Profiler；
- TPU/NPU 或跨平台对比。

不解决：NVIDIA GPU 的具体 kernel 分析（见 [Nsight Compute 第一轮](nsight-compute-first-pass.md)）。

## 快速判断

1. 先确认平台：NVIDIA / AMD / Intel / Apple / TPU / NPU；
2. 再确认问题层次：端到端、单算子、微架构、还是图形管线；
3. 选平台对应的工具；没有计数器权限时，用 CUDA events/框架 profiler 做相对对比；
4. 跨平台比较前，先对齐 dtype/精度/“利用率”口径。

## 按问题选工具

| 问题 | 首选 | 说明 |
| --- | --- | --- |
| NVIDIA 端到端 | Nsight Systems | 见 [时间线](nsight-systems-timeline-reading.md) |
| NVIDIA 单 kernel | Nsight Compute | 见 [ncu](nsight-compute-first-pass.md) |
| PyTorch 算子/shape | PyTorch Profiler | 见 [torch profiler](pytorch-profiler-and-memory-snapshot.md) |
| 图形管线 | Nsight Graphics / RenderDoc / PIX | 不是 compute 工具 |
| AMD GPU | rocprofv3 / rocprof-sys / RGP / amd-smi | ROCm 生态 |
| Intel GPU | unitrace / GPA / VTune | oneAPI 生态 |
| Apple GPU | Metal System Trace / Metal Debugger | Xcode |
| TPU | jax.profiler / XLA profile | TensorBoard |
| 昇腾/NPU | msprof / MindStudio Insight | 见 [NPU 对照](npu-to-gpu-profiling-mapping.md) |

## AMD

- **rocprofv3**：HIP/HSA trace、kernel trace、性能计数器（取决于 GPU/权限）；
- **rocprof-sys**（原 Omnitrace）：系统级 tracing；
- **Radeon GPU Profiler（RGP）**：图形/计算 kernel 的 UI 分析；
- **amd-smi**：监控。

命令风格和 nsys 类似，但选项/报告格式不同；用 `rocprofv3 --help` 查当前版本。

参考：<https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html>

## Intel

- **unitrace**：Intel GPU 的 trace/性能工具（oneAPI）；
- **Intel GPA**：图形/计算分析；
- **VTune**：CPU/GPU 混合分析。

参考：<https://github.com/intel/pti-gpu/tree/master/tools/unitrace>

## Apple

- **Metal System Trace**（Xcode Instruments）：Metal 命令、GPU 时间线；
- **Metal Debugger**：捕获一帧，分析 shader/带宽。

```bash
xctrace record --template 'Metal System Trace' --launch -- ./your_app
```

## TPU / NPU

- **XLA/TPU**：`jax.profiler`、TensorBoard 的 XLA profile、`--xla_dump_to`；
- **昇腾/CANN**：`msprof`、MindStudio Insight；
- **其他 NPU**：厂商 profiler。

跨平台对比时，先对齐“问题层次”（端到端/单算子/微架构），再比较数字；不同工具的“利用率”口径通常不同。

## 图形 vs 计算

- Nsight Graphics、RenderDoc、PIX 关注 **graphics pipeline**（draw call、shader、stage）；
- CUDA 计算优化不要用它们；
- 图形问题先看 GPU 时间线/管线 stage，再用 shader 调试。

## 修复与规避

- 平台没有对应工具时，用 CUDA events/框架 profiler 做相对对比；
- 跨平台移植时，先确认 dtype/精度/峰值口径；
- 不要把一个平台的“利用率”直接套到另一个平台；
- 图形和计算分开分析。

## 验证

- 工具能覆盖当前问题层次；
- 指标口径明确（峰值/可达/时间占比）；
- 跨平台结论注明平台和版本；
- 关键结论有两种方法交叉验证。

## 常见坑

- 用 NVIDIA 工具分析 AMD/Intel/Apple；
- 用图形工具分析 compute kernel；
- 把不同平台的利用率直接比较；
- 忽略 ROCm/oneAPI 版本差异；
- TPU/NPU 的 profile 与 GPU 不是同一抽象层。

## 参考

- rocprofv3: <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html>
- Intel unitrace: <https://github.com/intel/pti-gpu/tree/master/tools/unitrace>
- Nsight Graphics: <https://docs.nvidia.com/nsight-graphics/UserGuide/index.html>
- PyTorch Profiler: <https://pytorch.org/docs/stable/profiler.html>
