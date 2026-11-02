# GPU profiling demo

一个约 200 行的 CUDA workload，用来复现 GPU 性能分析里的五类常见现象：

1. `vector_add4`：纯 memory-bound，测可达带宽；
2. `matmul_naive`：compute-bound 但效率很低的 SGEMM；
3. `transpose_naive` vs `transpose_tiled`：同一份数据、不同访存 pattern；
4. 500 个 tiny kernel：launch overhead 主导；
5. 两个 stream：计算与 128 MiB 拷贝重叠。

配套脚本会编译、原生跑 5 次、用 Nsight Systems 采集、生成统计文本和时间线图；如果本机有 Nsight Compute 且计数器权限正常，再采一个 kernel。

> 这是需要 GPU 的工具；本仓库的 CI 不运行它。命令、版本和限制见下文。

## 前置条件

- NVIDIA GPU、可用驱动；
- CUDA Toolkit（`nvcc`），建议 CUDA 11.6+（需要 `nvtx3` 头文件）；
- Nsight Systems（`nsys`），可选 Nsight Compute（`ncu`）；
- Python 3 + matplotlib（仅用于从 SQLite 重绘时间线）。

```bash
nvidia-smi
nvcc --version
nsys --version
ncu --version        # 可选
```

## 快速开始

```bash
cd toolbox/gpu-profiling-demo
./run_profile.sh
```

脚本会：

1. 查找 `nvcc`/`nsys`（可用 `NVCC=`、`NSYS=` 覆盖）；
2. 自动定位 NVTX 库（CUDA 13 的 `libnvtx3interop` 或旧版 `libnvToolsExt`）；
3. 用 `-arch=native -lineinfo` 编译到 `build/gpu_demo`；
4. 原生运行 5 次，写 `results/run-native.txt`；
5. `nsys profile` 采集到 `results/demo_profile.nsys-rep`；
6. 导出 SQLite，生成 `results/nsys-*.txt`；
7. 用 matplotlib 生成 `results/figures/*.png`；
8. 尝试 `ncu --set basic`（没有权限时打印提示并继续）。

常用覆盖：

```bash
NVCC_ARCH=sm_90 ./run_profile.sh
NVTX_DIR=/path/to/cuda/targets/x86_64-linux/lib ./run_profile.sh
NVCC=/usr/local/cuda-12.8/bin/nvcc NSYS=/opt/nvidia/nsight-systems/2025.6.3/bin/nsys ./run_profile.sh
```

> 如果编译报 `nvtx3/nvToolsExt.h: No such file`，说明 CUDA 安装缺 NVTX 头文件；装对应 `cuda-nvtx` 包，或不带 `-DUSE_NVTX` 编译（会没有 NVTX 阶段条）。

## 本机实测输出

环境：RTX 5070 12 GB（GB205，sm_120，48 SM）、驱动 616.92、CUDA 13.2、Nsight Systems 2025.6.3、WSL2、2026-11-02。

`results/run-native.txt` 保存 5 次原始输出；下面是其中一次：

```text
GPU: NVIDIA GeForce RTX 5070 (sm_120, 48 SMs)
vector_add      :    0.325 ms    620.4 GB/s (read+write 3x64MB)
matmul_naive    :   20.743 ms   2070.6 GFLOP/s (42.95 GFLOP)
transpose_naive :    0.038 ms    220.7 GB/s
transpose_tiled :    0.013 ms    622.1 GB/s
500 tiny kernels:    5.258 ms  (10.5 us each, launch included)
overlap phase   :   10.254 ms
done
```

多次运行的量级：

| 阶段 | 原生 | nsys 下 | 解读 |
| --- | --- | --- | --- |
| vector add | 0.32–0.39 ms，510–625 GB/s（多数 0.32–0.34 / 600–625） | 0.36 ms，554 GB/s | 多数运行达到 90%–93% 峰值带宽 |
| matmul naive | 20.7–22.6 ms / 20 次 | 20.4–22.2 ms | 约 1.9–2.1 TFLOP/s，FP32 峰值的 6.2%–6.7% |
| transpose naive → tiled | 0.037–0.038 → 0.013–0.022 ms | 0.038 → 0.015 ms | 约 220–228 → 387–622 GB/s |
| 500 tiny kernels | 3–10 ms | 4.7–11.5 ms | kernel 自身合计仅约 0.25 ms |
| overlap | 10.3–11.5 ms | 10.4–12.2 ms | 拷贝基本被 10 次 matmul 遮住 |

**不要把这些数字当 benchmark**：WSL2、温度、时钟、后台负载都会让结果波动；重点看比例和结论（是否到 roof、是否 launch-bound、是否重叠）。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `build/gpu_demo` | 编译后的程序 |
| `results/run-native.txt` | 5 次原生输出 |
| `results/run-under-nsys.txt` | nsys 下的程序输出 |
| `results/demo_profile.nsys-rep` | Nsight Systems 报告 |
| `results/demo_profile.sqlite` | SQLite 导出 |
| `results/nsys-kern-sum.txt` | kernel 汇总 |
| `results/nsys-mem-sum.txt` | memcpy/memset 汇总 |
| `results/nsys-api-sum.txt` | CUDA API 汇总 |
| `results/nsys-nvtx-sum.txt` | NVTX range 汇总 |
| `results/figures/*.png` | 从 SQLite 重绘的时间线 |
| `results/matmul_basic.ncu-rep` | 可选：ncu 报告（有权限时） |

```bash
nsys-ui results/demo_profile.nsys-rep
ncu-ui results/matmul_basic.ncu-rep
```

## 只重画时间线

```bash
python3 plot_timeline.py results/demo_profile.sqlite results/figures
```

脚本假设报告里有五个 NVTX range（`phase 1`…`phase 5`），并按名字找 `matmul`/`overlap` 阶段；换 workload 需要相应调整。

## Nsight Compute 权限

脚本会尝试：

```bash
ncu --set basic -k regex:matmul_naive -c 1 -o results/matmul_basic ./build/gpu_demo
```

没有权限时典型报错：

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

这不是代码问题。处理方式见 [Nsight Compute 权限、容器、WSL 和 MIG](../../experiences/gpu-profiling/gpu-profiling-permissions-and-containers.md)。本仓实验环境（WSL2）没有开放计数器，因此没有本机 ncu 数据。

## 已知的坑

1. `-arch=native` 需要较新的 nvcc；老版本用 `NVCC_ARCH=sm_80` 之类显式指定；
2. NVTX 库名随 CUDA 版本变化（CUDA 13 用 `libnvtx3interop`，旧版用 `libnvToolsExt`）；
3. WSL2 计时波动大，ncu 通常不可用；
4. transpose 的“带宽”会受 L2 影响，可能超过 HBM 峰值；
5. nsys 会放大 tiny kernel 的开销；
6. `float4` 要求 16 B 对齐；
7. 不要提交 `.nsys-rep`/`.sqlite`/`.ncu-rep` 和 `figures/`（`.gitignore` 已处理）。

## 相关条目

- [GPU Profiling 与性能分析总览](../../experiences/gpu-profiling/README.md)
- [Nsight Systems 时间线](../../experiences/gpu-profiling/nsight-systems-timeline-reading.md)
- [GPU 空转与 launch](../../experiences/gpu-profiling/nsight-systems-gpu-idle-and-launch-overhead.md)
- [Nsight Compute 第一轮](../../experiences/gpu-profiling/nsight-compute-first-pass.md)
- [跨平台工具](../../experiences/gpu-profiling/gpu-other-platform-tools.md)

## 参考

- Nsight Systems: <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>
- Nsight Compute CLI: <https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html>
- CUDA Best Practices: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
