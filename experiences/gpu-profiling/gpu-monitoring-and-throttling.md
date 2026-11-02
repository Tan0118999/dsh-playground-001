---
name: gpu-monitoring-and-throttling
title: 先排除环境问题：nvidia-smi、DCGM、降频和拓扑
description: 性能波动大、怀疑降频/功耗/温度、nvidia-smi util 100% 但吞吐低、多卡走 PCIe 而不是 NVLink、需要集群监控时读这条。给出查询命令、指标判读、DCGM 和常见误读。
category: engineering
tags: [nvidia-smi, dcgm, throttling, topology, monitoring, clocks]
symptoms:
  - 时钟低于 boost
  - 温度/功耗高，性能忽快忽慢
  - util 100% 但吞吐低
  - 多卡走 PCIe 而不是 NVLink
  - 需要长期监控 GPU 健康
stacks: [cuda]
updated: 2026-11-02
---

# 先排除环境问题：nvidia-smi、DCGM、降频和拓扑

> 一句话经验：**在做任何性能结论前，先确认 GPU 没有降频、没有多租户干扰、拓扑符合预期；`util 100%` 不等于效率高。**

## 适用场景

- 两次运行差异很大，怀疑时钟/温度/功耗；
- 多卡扩展效率低，怀疑 PCIe/NVLink；
- 需要看 GPU 是否健康、是否被其他进程占用；
- 生产集群需要趋势监控。

不解决：kernel 内部瓶颈（用 ncu）；端到端时间线（用 nsys）。

## 症状与信号

- `clocks.sm` 明显低于 boost；
- 温度接近上限、`power.draw` 接近 TDP；
- `utilization.gpu` 100% 但吞吐低；
- `nvidia-smi topo -m` 显示 PCIe；
- 同一个模型在不同机器上差异大。

## 快速判断

1. 看 `nvidia-smi` 的 GPU 型号、显存、温度、功耗、时钟、进程；
2. 用 query 命令连续采样，确认是否降频；
3. 用 `nvidia-smi topo -m` 看多卡拓扑；
4. 集群/长期问题用 DCGM/exporter；
5. 排除其他进程/MIG/容器干扰。

## 常用命令

```bash
nvidia-smi

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,\
temperature.gpu,power.draw,clocks.sm,clocks.mem \
  --format=csv -l 1

nvidia-smi dmon -s pucvmet -d 1

nvidia-smi topo -m

nvidia-smi -q -d CLOCK,PERFORMANCE,POWER

nvidia-smi mig -lgip
```

（`-l 1`/`dmon` 会持续采样，Ctrl-C 退出。）

## 指标判读

| 指标 | 含义 | 误读 |
| --- | --- | --- |
| `utilization.gpu` | 采样周期内有 kernel 活动的时间比例 | 不等于 SM 效率，也不等于算力利用率 |
| `memory.used` | 显存占用 | 包括 cache/reserved，不等于模型权重 |
| `temperature.gpu` | 温度 | 高温度可能触发降频 |
| `power.draw` | 功耗 | 接近上限可能降频 |
| `clocks.sm` / `clocks.mem` | 当前时钟 | 和峰值/boost 对比，判断是否降频 |
| `pstate` | 性能状态 | P0 最高，P12 最低 |
| `ecc.errors` | ECC 错误 | 数据中心卡才常见 |

**判断降频**：同时看 `clocks.sm`、`temperature.gpu`、`power.draw`、`nvidia-smi -q -d PERFORMANCE` 里的 throttle 原因。如果 SM 时钟明显低于 boost，性能对比就不可信。

## DCGM

DCGM 适合**集群/长期监控**，不是单次 kernel 分析：

```bash
dcgmi discovery -l          # 发现 GPU
dcgmi dmon -e <field ids>   # 采集指定指标
dcgmi diag -r 2             # 运行诊断
```

生产环境常用 `dcgm-exporter` + Prometheus + Grafana，采集利用率、显存、温度、功耗、ECC、XID。它回答“集群是否健康、是否有异常趋势”，不回答“这个 kernel 为什么慢”。

## 多卡拓扑

```bash
nvidia-smi topo -m
```

- `NV#` = NVLink，`PIX/PXB/PHB` = PCIe 不同层级；
- all-reduce 走 PCIe 时带宽明显低；
- 多机还要看 IB/RoCE 和 NCCL 配置；
- 拓扑异常时，先修环境再调 kernel。

## 修复与规避

- 降频：改善散热/功耗策略，或锁时钟（需要权限）；
- 其他进程占用：隔离/MIG/独占；
- 拓扑：把通信密集的 rank 放到 NVLink 域内；
- 监控：用 DCGM exporter 长期采集，设置 XID/ECC/温度告警；
- 做性能结论前记录时钟/温度/功耗，作为报告的一部分。

## 验证

- 性能对比期间时钟稳定；
- 温度/功耗没有触发 throttle；
- 拓扑与预期一致；
- 多卡通信时间/扩展效率改善；
- 监控能看到异常趋势。

## 常见坑

- 把 `util 100%` 当“GPU 很忙/很快”；
- 把 reserved 显存当泄漏；
- 在降频状态下做 A/B；
- 忽略其他进程/多租户；
- MIG 下用整卡指标对比；
- 用 `nvidia-smi` 代替 nsys/ncu 做瓶颈分析。

## 参考

- `nvidia-smi` 文档: <https://docs.nvidia.com/deploy/nvidia-smi/index.html>
- DCGM: <https://docs.nvidia.com/datacenter/dcgm/latest/index.html>
- CUDA C++ Best Practices Guide: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
