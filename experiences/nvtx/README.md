# NVTX 标注：从第一次 range 到可自动化的时间线

本目录收录 NVIDIA NVTX（NVIDIA Tools Extension）的使用经验，面向**完全没接触过 NVTX、只会打开 Nsight Systems 看 kernel** 的工程师。和其他主题一样，条目按“症状驱动”组织；本页负责给学习顺序、概念地图、教程入口和证据边界。

> 一句话：NVTX 不是计时器，而是**给代码打标签**。标签本身几乎不耗时；只有当程序在 nsys、ncu 等工具下运行时，这些标签才会变成时间线上的 range、统计里的分组或过滤条件。先用它把“数据加载 / forward / backward / 通信”对上 GPU 时间线，再谈优化。

## 先看什么

| 你的情况 | 阅读顺序 |
| --- | --- |
| 完全没接触过，想先跑通 | [第一次用 NVTX](nvtx-first-range.md) |
| 能跑通，但 range 嵌套、跨线程或 domain 搞不清 | [range 语义](nvtx-range-model.md) |
| 已经打了 range，想在 nsys 里采集、查看、只采一段 | [nsys 工作流](nvtx-nsys-workflow.md) |
| 在 Python 或 PyTorch 训练/推理里加标注 | [Python 与 PyTorch](nvtx-python-pytorch.md) |
| 打了但报告里没有、名字不对、capture-range 不触发 | [看不到 NVTX 的排查](nvtx-not-showing.md) |
| 想加 payload、降低开销、用 C++ RAII 封装 | [进阶 API](nvtx-advanced-api.md) |
| 想用 ncu 只测某段 kernel，或把结果接入脚本 | [消费端与自动化](nvtx-ecosystem-and-automation.md) |

如果只读一条：读 [第一次用 NVTX](nvtx-first-range.md)，先拿到一次 `nvtx_sum` 输出；概念可以后补。

## 学习路线（0 → 3）

| 阶段 | 目标 | 读什么 | 验收 |
| --- | --- | --- | --- |
| 0 入门 | 5 分钟内让时间线出现一个自己命名的 range | [第一次用 NVTX](nvtx-first-range.md) | `nsys stats --report nvtx_sum` 能看到 `:main` |
| 1 说清楚 | 分清 mark、push/pop、start/end、domain、category | [range 语义](nvtx-range-model.md) | 能解释为什么 range 嵌套错、跨线程该用哪个 API |
| 2 用起来 | 采一段、过滤、统计，接到真实 workload | [nsys 工作流](nvtx-nsys-workflow.md)、[Python 与 PyTorch](nvtx-python-pytorch.md) | 能只采一个阶段，并解释 top range |
| 3 进阶 | 控制开销、只 profile 某段 kernel、脚本化 | [进阶 API](nvtx-advanced-api.md)、[消费端与自动化](nvtx-ecosystem-and-automation.md) | 能设计 domain/registered string/payload，并从 SQLite 取数 |

## 概念地图

| 术语 | 含义 | 容易混淆的点 |
| --- | --- | --- |
| **mark** | 一个时间点，没有持续时间 | 不是 range；`nvtx_sum` 不统计 mark，要去看时间线或 SQLite |
| **push/pop range** | 同一线程内嵌套的范围，按栈配对 | `pop` 关的是同一线程、同一 domain 最近一次 `push`；异常路径容易漏 `pop` |
| **start/end range** | 用句柄显式开始/结束，可跨线程、可重叠 | 不是栈；必须保存 `nvtxRangeId_t` 并传给 end |
| **domain** | 标注的命名空间，可按库/模块隔离 | 每个 domain 有自己的 range 栈、category 和 registered string 表 |
| **category** | domain 内的整数分组，可命名、过滤 | 不是字符串；名字要用 `nvtxNameCategory*` 绑定 |
| **payload** | 单个标量（int/uint/double/float）附加数据 | 不要用 `sprintf` 拼字符串；可变数据优先放 payload |
| **registered string** | 注册过的消息句柄，工具只需匹配一次 | nsys 的 capture-range 默认只认 registered string |

NVTX 是**宿主侧** API，不能在 `__device__` 代码里打点。GPU 内部的细粒度标记要用 `pmevent`/硬件计数器或 Nsight Compute 的方式，不要指望 NVTX。

## 教程链接

按“先跑起来，再看原理，最后看工具”的顺序：

1. [NVTX 官方总览（GitHub README）](https://github.com/NVIDIA/NVTX)：NVTX 是什么、支持哪些工具、开销和最佳实践。
2. [NVTX C/C++ 示例](https://github.com/NVIDIA/NVTX/blob/release-v3/c/README.md)：`nvtx3::scoped_range`、`NVTX3_FUNC_RANGE`、push/pop 和 start/end 的最小例子。
3. [Nsight Systems User Guide 的 NVTX Trace 一节](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#nvtx-trace)：时间线里 NVTX 行怎么读、常用 API 和采集步骤。
4. [NVTX C API 参考](https://nvidia.github.io/NVTX/doxygen/)、[C++ API 参考](https://nvidia.github.io/NVTX/doxygen-cpp/)：属性结构、domain、payload、资源命名的权威签名。
5. [NVTX Python 文档](https://nvidia.github.io/NVTX/python/)：`pip install nvtx`、`annotate` 装饰器/上下文管理器、颜色和 domain。
6. [PyTorch `torch.cuda.nvtx`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.nvtx.range.html)：`range` 上下文管理器、`range_push`/`range_pop`、`mark`。
7. [PyTorch `emit_nvtx`](https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.emit_nvtx)：给 autograd 算子自动打 NVTX，适合搭配 nsys。
8. [Nsight Compute CLI 的 NVTX Filtering](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvtx-filtering)：只 profile 某个 NVTX range 里的 kernel。
9. 本仓 [GPU profiling demo](../../toolbox/gpu-profiling-demo/README.md)：一个真实 CUDA workload，已经带 NVTX 阶段条和 nsys 采集脚本。

> 链接是入口，不是结论。API 名称和 CLI 选项随 CUDA、Nsight 版本变化；执行前先用 `--version`、`--help` 核对。

## 证据边界

本目录的实测环境（2026-11-03）：

- GPU：RTX 5070 12 GB（WSL2，驱动 616.92），CUDA 13.2，Nsight Systems 2025.6.3，Nsight Compute 2026.1.1；
- 编译/运行：g++ 11.4、glibc 2.35、Python 3.10、`nvtx` Python 包 0.2.16；
- **实测过**：C++ NVTX C API 与 `nvtx3.hpp` 封装、Python `nvtx` 包、nsys 的 NVTX trace、`nvtx_sum`/`nvtx_pushpop_sum`/`nvtx_startend_sum`/`nvtx_gpu_proj_sum`、`--capture-range=nvtx`、domain include/exclude、`--filter-nvtx`、SQLite `NVTX_EVENTS`、线程命名、未闭合 range 的行为；
- **没有实测**：PyTorch（本机未安装 torch）、ncu 的计数器采集（WSL2 没有权限；ncu 的 NVTX 过滤语法来自 `ncu 2026.1.1 --help` 和官方文档）、Nsight Systems GUI 的新截图（GUI 行为参考官方文档和本仓已有截图）；
- 具体数字只用于说明量级，不是 benchmark；WSL2、时钟、后台负载都会让结果波动。

本目录不包含私有代码、凭据或内部链接。示例都是最小可复现片段；PyTorch 相关内容明确标为“文档整理”。

## 条目

### 入门与语义

- [第一次用 NVTX：5 分钟给时间线加上阶段名](nvtx-first-range.md)
- [NVTX range 的语义：mark、push/pop、start/end、线程与 domain](nvtx-range-model.md)

### 工具与集成

- [用 Nsight Systems 采集和读 NVTX：trace、capture-range、report、SQLite](nvtx-nsys-workflow.md)
- [Python 与 PyTorch 里的 NVTX：annotate、torch.cuda.nvtx、emit_nvtx](nvtx-python-pytorch.md)
- [NVTX 打了但 Nsight Systems 里看不到：从 trace 到 range 的排查顺序](nvtx-not-showing.md)

### 进阶

- [NVTX 进阶：attributes、payload、domain、registered string 与 C++ RAII](nvtx-advanced-api.md)
- [NVTX 的消费端：ncu 过滤、SQLite 与自动化](nvtx-ecosystem-and-automation.md)

---

新增条目请按 [CONTRIBUTING.md](../../CONTRIBUTING.md) 和 [模板](../../templates/experience.md) 操作，并更新本页、[条目索引](../README.md)和 `catalog.json`。
