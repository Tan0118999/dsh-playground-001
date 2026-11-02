# 经验库（Experience Library）

面向工程师和 AI agent 的可复用调试经验库。每条经验按“症状驱动”组织：**什么现象 → 怎么快速判断 → 根因机制 → 怎么排查 → 怎么修 → 怎么验证**，可以直接当作检查清单使用。条目 frontmatter 提供 `name`/`description`/`symptoms`/`tags`，便于像 skill 一样检索和按需加载。

当前首批收录 **GPU 图编译 debug 经验 16 条**，覆盖 `torch.compile`/TorchInductor、Triton、TVM、XLA/JAX，以及底层 CUDA 工具链（compute-sanitizer、cuda-gdb、Nsight、SASS/PTX 检查）。后续可以按主题继续扩展，也可以加入 toolbox 类可执行工具。

## 目录结构

```
.
├── README.md                 # 本文件：定位、使用方式、总索引
├── AGENTS.md                 # agent 消费与维护本库的约定
├── CONTRIBUTING.md           # 收录标准、字段规范、写作要求
├── catalog.json              # 机器可读索引（由脚本生成，勿手改）
├── templates/
│   └── experience.md         # 新增条目模板
├── experiences/
│   ├── README.md             # 全部条目索引
│   └── gpu-graph-compile/    # 首批：GPU 图编译 debug 经验
└── toolbox/                  # 可复用脚本（catalog 构建/校验）
```

## 怎么用

### 人工排查

1. 从 [症状速查](experiences/README.md#症状速查) 找到最接近的现象。
2. 打开条目，按“快速判断”从左到右执行；优先只读、可回退、低成本的检查。
3. 用条目末尾的“验证”确认修复，而不是看到报错消失就结束。

### Agent 使用

1. 先读 `catalog.json` 或 [条目索引](experiences/README.md)，用 `description`、`symptoms`、`tags`、`stacks` 匹配问题。
2. 完整读取命中的条目，把“快速判断/排查步骤”当检查清单；执行前确认版本、GPU 架构和当前目录。
3. 条目是经验而非保证；与实测冲突时以实测为准，并说明差异。约定见 [AGENTS.md](AGENTS.md)。

## 条目总览

| 类别 | 条目 | 解决什么 | 典型症状 |
| --- | --- | --- | --- |
| 方法 | [GPU 图编译问题分层定位总纲](experiences/gpu-graph-compile/gpu-compile-triage.md) | 先定位问题在哪一层，再选工具 | 有报错但不知道是前端、编译器还是 runtime；报错位置飘忽 |
| 方法 | [最小复现与配置二分](experiences/gpu-graph-compile/minimal-repro.md) | 把问题缩到单文件、单 kernel、固定输入 | 大模型能复现、小图不能；需要向上游报 bug |
| 编译 | [torch.compile 后端报错](experiences/gpu-graph-compile/torch-compile-backend-error.md) | 处理 `BackendCompilerFailed`/`InductorError`、graph break、重编译 | `torch.compile` 首次运行直接抛异常；日志淹没在框架内部 |
| 编译 | [Triton kernel 编译错误](experiences/gpu-graph-compile/triton-kernel-compile-error.md) | 定位 MLIR/LLVM/`ptxas` 各阶段错误 | `CompilationError`、shared memory 超限、寄存器溢出 |
| 编译 | [TVM 构建错误](experiences/gpu-graph-compile/tvm-build-error.md) | 定位 schedule、TIR pass、`tvm.build` 错误 | `Check failed`、`InternalError`、调度非法 |
| 编译 | [XLA/JAX 编译错误](experiences/gpu-graph-compile/xla-compile-error.md) | 用 HLO dump 定位 XLA 编译问题 | `XlaRuntimeError`、`INVALID_ARGUMENT`、编译卡住 |
| 崩溃 | [非法访存与异步错误定位](experiences/gpu-graph-compile/illegal-memory-access.md) | 把异步的 CUDA 错误定位到 kernel 和越界点 | `illegal memory access`、`misaligned address`、`device-side assert` |
| 正确性 | [静默错误结果](experiences/gpu-graph-compile/silent-wrong-results.md) | 不报错但结果不对时做分层对拍 | 与 eager 不一致、偶发错误、精度差 |
| 正确性 | [数据竞争与同步](experiences/gpu-graph-compile/race-and-sync.md) | 定位缺失 barrier、atomic、stream 同步 | 偶发错误、重复运行结果不同、只在多 warp/多卡出现 |
| 正确性 | [精度与 fast-math](experiences/gpu-graph-compile/precision-and-fastmath.md) | 区分真实 bug 与数值精度差异 | TF32/fp16 偏差、NaN、denormal、FMA 重排 |
| 性能 | [慢 kernel 分层排查](experiences/gpu-graph-compile/slow-kernel-triage.md) | 从端到端到 kernel 逐层定位瓶颈 | 能跑但明显变慢、波动大、融合没生效 |
| 性能 | [寄存器溢出与占用率](experiences/gpu-graph-compile/register-spill-occupancy.md) | 判断慢是否来自 spill/低 occupancy | `ptxas` 报 spill、occupancy 低、tile 调大后更慢 |
| 性能 | [CUDA Graphs 捕获问题](experiences/gpu-graph-compile/cudagraph-capture-problems.md) | 处理捕获失败、回退和捕获后正确性问题 | `operation not permitted when stream is capturing`、results 错 |
| 工程 | [编译缓存与重编译](experiences/gpu-graph-compile/compile-cache-and-recompiles.md) | 减少动态 shape 重编译、复用缓存 | 每次启动都编译、recompile 计数暴涨、cache 目录膨胀 |
| 工程 | [分布式图编译](experiences/gpu-graph-compile/distributed-graph-compile.md) | 排查多卡 hang、超时、rank 间图不一致 | NCCL timeout、collective 顺序错、多卡结果不同 |
| 工程 | [工具链与架构不匹配](experiences/gpu-graph-compile/toolchain-arch-mismatch.md) | 处理驱动/runtime/`ptxas`/`sm_XXa` 兼容问题 | `no kernel image`、`sm_90a is not defined`、换机器就失败 |

## 内容状态与边界

- 条目由公开文档和通用调试实践整理，每条末尾列出参考来源；仓库目前没有 GPU CI，**未逐条在本机 GPU 环境实测**，使用前请按条目里的“快速判断”确认版本行为。
- 命令、环境变量默认值和报错文本会随 PyTorch、Triton、TVM、XLA、CUDA 版本变化；无法确定版本的结论会标注“推断/待验证”。
- 不包含私有代码、客户数据、凭据或内部链接。
- 本库不替代官方文档：涉及语义和兼容性时，以对应版本的官方文档为准。

## 维护

```bash
python3 toolbox/build_catalog.py          # 新增/删除条目后重写 catalog.json
python3 toolbox/build_catalog.py --check  # 校验索引是否最新
```

收录标准、frontmatter 字段和写作要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。
