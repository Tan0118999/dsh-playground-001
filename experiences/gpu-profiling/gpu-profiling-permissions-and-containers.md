---
name: gpu-profiling-permissions-and-containers
title: 采不到数据：ncu 计数器权限、容器、WSL 和 MIG
description: ncu 报 ERR_NVGPUCTRPERM、容器里 nsys 失败/没有数据、WSL 下 ncu 不可用、MIG 实例指标缺失、远程/云上采不到计数器时读这条。给出权限模型、容器/Slurm 配置和替代测量方案。
category: engineering
tags: [ncu, nsys, permissions, container, docker, wsl, mig, slurm]
symptoms:
  - ERR_NVGPUCTRPERM
  - 容器里 nsys 没有数据
  - WSL 下 ncu 失败
  - MIG 上部分指标不可用
  - 远程机器采完报告打不开
stacks: [ncu, nsys, cuda]
updated: 2026-11-02
---

# 采不到数据：ncu 计数器权限、容器、WSL 和 MIG

> 一句话经验：**先分清“采集工具没装/没权限”和“代码有问题”；计数器权限是驱动/平台策略，不是 kernel 的错。**

## 适用场景

- `ncu` 报 `ERR_NVGPUCTRPERM`；
- nsys/ncu 在容器、Slurm、云主机、WSL 里失败或没有数据；
- MIG 实例上部分指标缺失；
- 远程采集的报告在本地打不开。

不解决：采集成功后的指标判读（见 [Nsight Compute 第一轮](nsight-compute-first-pass.md)）。

## 症状与信号

- `==ERROR== ERR_NVGPUCTRPERM`；
- nsys 报告只有 CPU 行，没有 CUDA HW；
- 容器里 `nvidia-smi` 正常但 `ncu` 失败；
- MIG 下只能看到部分 SM/指标；
- 本地 GUI 提示报告版本不兼容。

## 快速判断

1. `nvidia-smi` 是否能看到 GPU？
2. `nsys --version` / `ncu --version` 是否可用？
3. 错误是权限、驱动能力、还是工具版本？
4. 是否在容器/MIG/WSL/云主机？
5. 报告版本和 GUI 版本是否兼容？

## 根因

### 1. GPU 性能计数器权限

`ncu` 需要访问 GPU 性能计数器；默认只有管理员/特定配置允许。Linux 通过驱动参数控制，Windows 通过 NVIDIA 控制面板，容器需要额外 capability，MIG/WSL/云平台可能直接不开放。

### 2. 容器能力

容器默认没有 `SYS_ADMIN`/perf 权限，也未必有 CUDA/CUPTI 工具；`NVIDIA_DRIVER_CAPABILITIES` 不含 `compute,utility` 时工具会缺能力。

### 3. WSL/虚拟化

WSL2 和部分虚拟化/云环境不开放计数器；这不是代码问题。

### 4. MIG

只能分析分配给当前实例的 SM，部分指标不可用；整卡指标不能直接对比。

## 排查步骤

```bash
nvidia-smi
nsys --version
ncu --version
ncu --set basic -k regex:my_kernel -c 1 ./app
```

看完整错误：

- `ERR_NVGPUCTRPERM` → 权限；
- `no CUDA devices` → 驱动/容器透传；
- `unsupported chip` → 工具版本/架构；
- 报告打不开 → 版本。

## 修复与规避

### Linux 裸机

在 `/etc/modprobe.d/` 加入：

```text
options nvidia NVreg_RestrictProfilingToAdminUsers=0
```

然后重启或重载驱动。**需要管理员权限，按你的环境策略执行；共享生产机不要随意改。**

### Windows

NVIDIA 控制面板 → Developer → Manage GPU Performance Counters → 允许所有用户访问（或按组织策略只允许管理员）。

### 容器

```bash
docker run --gpus all \
  --cap-add=SYS_ADMIN \
  --security-opt seccomp=unconfined \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v $PWD:/workspace -w /workspace \
  your_image \
  nsys profile --trace=cuda,nvtx -o /workspace/report ./app
```

- `SYS_ADMIN` 权限很大，只在受控环境使用；
- `ncu` 还需要计数器权限，容器通常需要额外 capability 或 privileged，且宿主机允许；
- 镜像里要有 nsys/ncu 或挂载工具；
- 报告在宿主机/本地 GUI 打开，注意版本兼容。

### Slurm/多机

```bash
srun --ntasks=1 --gpus=1 nsys profile --trace=cuda,nvtx -o report ./app
```

每个 rank 一个报告；多机先确认 MPI/NCCL trace；报告可能很大，用 duration/capture range 限制。

### WSL/云/MIG

- 很多环境不开放计数器；先查平台文档；
- 不要在采不到时反复重试；
- 退回 nsys + CUDA events + PyTorch Profiler，给出“没有硬件计数器”的限制说明；
- MIG 下确认实例类型和可用指标。

## 替代测量方案

采不到 ncu 时：

1. nsys 看端到端/kernel 时间；
2. CUDA events 测绝对时间；
3. 微基准验证假设（如合并访存、block 大小）；
4. PyTorch Profiler 看算子/shape；
5. 在另一台有权限的机器上采同一个最小复现。

## 验证

- `ncu --set basic` 能产出报告；
- nsys 报告有 CUDA HW 行；
- 容器/远程报告能在本地打开；
- MIG 下明确知道哪些指标不可用；
- 替代方案能支持当前决策。

## 常见坑

- 把权限错误当 kernel 错误；
- 在生产机随意改驱动参数；
- 容器里用 `--privileged` 而不评估风险；
- 报告/GUI 版本不兼容；
- MIG 下拿整卡指标对比；
- 采不到计数器就编造结论。

## 参考

- Nsight Compute 文档入口: <https://docs.nvidia.com/nsight-compute/>
- Nsight Systems User Guide: <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>
- CUDA C++ Best Practices Guide: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
