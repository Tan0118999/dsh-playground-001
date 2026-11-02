---
name: dsh-model-architecture-visualization
title: 用 DSH 插件画出任意模型的架构示意图：全流程、上下游、token 构成与消费端
description: 想让 DSH 画出自己模型的全流程架构示意图、又不想每次换模型就改提示词时读这条：给出 agent 从工作目录的模型项目文件提取信息的步骤、可替换的 JSON 清单契约、通用创造模式提示词模板和验收清单；换模型只换清单，不在插件里硬编码结构。
category: methodology
tags: [dsh, creator-mode, model-visualization, architecture-diagram, token-budget, plugin, cordis, data-contract]
symptoms:
  - 官方提示词只覆盖 Transformer，想画自己项目里的模型架构
  - 生成的图只有模块名，看不到上下游依赖、token 长度构成和消费端
  - 不知道 agent 该从工作目录的哪些文件、按什么结构整理这些信息
stacks: [dsh, cordis]
updated: 2026-11-03
---

# 用 DSH 插件画出任意模型的架构示意图

> 一句话经验：先让 agent 把模型信息整理成一份 JSON 清单（组件、上下游、token 构成、消费端），再用一段通用提示词让 DSH 创造模式做“读清单渲染”的插件；换模型只换清单，不重写插件。

## 适用场景

- 适用：任何有本地项目文件的模型——训练脚本、推理服务、agent 系统、多模态流水线——希望在 DSH 侧栏出现一个能讲清“数据怎么流、token 花在哪、谁在消费”的可视化插件。
- 不适用：只要论文级静态插图（用绘图工具）；或只想看单个模块的内部实现（直接读源码更快）。
- 前提：agent 能读工作目录里的项目文件；清单里的数字必须标来源，拿不到就标 `unknown`，不要估算成精确值。

## 输入与输出

输入（agent 自己在工作目录里找）：

- 模型结构：`config.json`、`modeling_*.py`、`model.py`、`modules/`
- token 相关：tokenizer 配置、prompt 模板、`generation_config.json`、上下文截断/压缩逻辑
- 消费端：服务入口、agent loop、工具执行器、UI/网关、计费或日志
- 文档与测试：README、`tests/`、示例请求

输出：

1. 一份 `model-architecture.json` 清单（契约见下）；
2. 一个 DSH 插件：左侧边栏入口，读清单渲染全流程、上下游、token 构成、消费端；
3. 验收记录：哪些数字来自配置/实测，哪些是 `unknown`。

## 接口设置（必须保留，其余自由）

这几项是插件能长期复用的最小接口，写进提示词，别让 agent 自己发明：

| 接口 | 要求 |
| --- | --- |
| 入口 | 左侧边栏入口，和官方 Demo 形态一致 |
| 数据源 | 工作区内的清单文件路径（插件设置里可改），插件只读它，不内置模型结构 |
| 刷新 | 启动时读一次 + 手动刷新；可选文件监听 |
| 详情 | 点选组件显示 `detail`、`evidence`、`confidence` |
| 导出 | PNG/SVG |
| 平台 | macOS 下标题和工具栏避开窗口按钮 |

## 清单契约（换模型只换这个文件）

要求 `schema_version=1`；字段缺数据时填 `"unknown"`，不要删字段、不要编造。

```json
{
  "schema_version": 1,
  "model": {"name": "<model>", "source": ["config.json"]},
  "context_window": "<int or unknown>",
  "stages": [
    {"id": "input", "label": "输入组装", "order": 1},
    {"id": "encode", "label": "编码/推理", "order": 2},
    {"id": "output", "label": "输出与消费", "order": 3}
  ],
  "components": [
    {
      "id": "tokenizer",
      "label": "Tokenizer",
      "stage": "input",
      "kind": "module|service|consumer",
      "detail": "一句话说明；关键参数放这里",
      "evidence": "tokenizer_config.json",
      "confidence": "measured|config|inferred|unknown"
    }
  ],
  "flows": [
    {"from": "tokenizer", "to": "embedding", "label": "token ids", "evidence": "modeling_x.py:120"}
  ],
  "token_budget": {
    "input": [{"part": "system prompt", "tokens": "<int or unknown>", "evidence": "..."}],
    "per_stage": [{"stage": "encode", "tokens": "<int or unknown>", "note": "..."}],
    "output": [{"part": "answer", "tokens": "<int or unknown>", "evidence": "..."}],
    "total": {"tokens": "<int or unknown>", "context_window": "<int or unknown>"}
  },
  "consumers": [
    {"id": "ui", "label": "Web UI", "consumes": ["streamed tokens"], "stage": "output", "evidence": "..."}
  ]
}
```

填写规则：

- `components` 覆盖全流程，不只主干：输入装配、tokenize、embedding、每类层/模块、采样、工具调用、输出解析、消费端。
- `flows` 必须能连成从头到尾的路径；允许分支（工具调用、并行流、多路召回）。
- `token_budget` 分三段：输入构成、各阶段、输出构成；有 context window 就标出占比。
- `consumers` 写清“谁消费什么”（tokens、logits、embeddings、日志），以及它在哪个阶段。
- 每个数值带 `evidence`（文件:行、配置键或实测命令）；同一字段多来源冲突时保留来源并在 `note` 说明。

## 通用提示词模板（创造模式）

把 `<清单路径>` 换成实际路径，其余照抄：

```text
创建一个带左侧边栏入口的 DeepSeek Harness 插件，展示任意模型的全流程架构。数据只来自工作区文件 <清单路径>（JSON，schema_version=1），不要在代码里硬编码模型名称或结构；换一份清单就应展示另一种模型。

功能要求：
1. 按 stages 顺序展示全流程；每个阶段内按 components 排列，可折叠、可按阶段/kind 过滤。
2. 用 flows 画组件间的有向连线并标注数据名；点击组件高亮其上游和下游，详情面板显示 detail、evidence、confidence。
3. 展示 token 构成：输入各部分、各阶段、输出各部分，以及 context window 占比；悬停显示 evidence。
4. 展示 consumers：每个消费端消费什么、处在哪个阶段。
5. 支持缩放、平移、搜索组件、导出 PNG/SVG；unknown 字段原样显示，不要编造。
6. 面板标题和工具栏避开 macOS 原生窗口按钮。

先读清单确认字段齐全再实现；缺字段时在界面上标 unknown，并告诉我缺哪些。
```

想要 3D 或更炫的效果，在“功能要求”后追加一句即可，数据契约不变；官方 Transformer 3D 原文见文末附录。

## 在 dsh 里怎么用

1. 打开 dsh，新建任务，在模式选择器选“创造模式”；或从侧栏「插件」→「添加插件」进创造流程。判断标准是当前 preset 为 `cordis`，菜单文案随版本变化。
2. 把清单放到工作区任意位置（例如 `model-architecture.json`），把模板里的 `<清单路径>` 换成相对路径，整段粘贴发送。
3. 等插件生成：按提示词要求，完成后左侧边栏出现入口，点开就是全流程视图；插件以 bundle 形式安装，可在「插件」页启停。
4. 按“验收”核对；缺字段就补清单再刷新，结构不对就继续对话，不要重写插件。

## Agent 操作步骤

1. 定位项目：先列工作目录，找 `config.json`、`modeling_*.py`、tokenizer/prompt 模板、服务入口、agent loop、UI/网关代码。
2. 提取结构：用 `rg` 找模块注册、前向调用链、服务入口；每条结论记录文件:行。
3. 提取 token 规则：context window、prompt 分段、截断/压缩逻辑、`max_new_tokens`、KV cache 或批处理参数。
4. 提取消费端：谁读 token/logits、谁做流式输出、谁计费/记日志、谁消费工具结果。
5. 写清单：缺数据标 `unknown`，不猜；不写私密 prompt 原文或真实用户数据。
6. 在 DSH 创造模式里用上面的模板生成插件（见“在 dsh 里怎么用”），先跑通最小版本。
7. 按“验收”逐条核对，不符就继续对话补，不要重写插件。
8. 清单随模型项目走，插件当通用渲染层：下一个模型只换清单。

## 验收

- 换一份 `model-architecture.json` 后界面展示另一种模型，代码不用改——这是“通用”的硬标准。
- 全流程从输入到消费端连续可见，没有断开的组件。
- 每个组件都能看到上游和下游；点击高亮正确。
- token 构成包含输入、各阶段、输出三部分，数字带 evidence，`unknown` 明确显示。
- 消费端完整：至少覆盖 UI/调用方、agent loop 或服务层、日志/计费。
- 交互可用：缩放、过滤、搜索、导出；大图不重叠、可读。
- 清单里没有私密 prompt 原文、真实用户数据或凭据。

## 常见坑

- 把官方 Transformer 3D 提示词直接当“任意模型”用：它只描述了一个固定的 Transformer 演示。
- 只画模块框，不画 flows 和 consumers：像架构图，但答不了“数据从哪来、到哪去、token 被谁消费”。
- token 数字没有来源：宁可标 `unknown`，也不要把估算写成实测。
- 把结构硬编码进插件：换模型就要改代码，失去通用性。
- 清单太大导致图不可读：先按 stage 折叠，再提供过滤和搜索。
- 把私密 prompt 或真实会话内容贴进清单：清单进仓库前先脱敏。

## 附：官方 Transformer 3D 原文（最小起点）

先跑通“能动的 3D 插件”时可以照抄；它只覆盖 Transformer 演示，通用做法按上面的清单 + 模板。

```text
创建一个带左侧边栏入口的 DeepSeek Harness 插件，做成可交互的 3D 演示游戏，帮助我理解 Transformer 内部的计算逻辑。确保面板标题和工具栏避开 macOS 原生窗口按钮，不发生重叠。
```

## 参考

- 官方创造模式提示词合集（含上述 Transformer 3D 原文）：https://cdn.deepseek.com/harness/dsh-creator-mode-prompt-collection-zh.html
- 英文版合集：https://cdn.deepseek.com/harness/dsh-creator-mode-prompt-collection-en.html
- 发布视频（简介称“模型可视化工具”）：https://www.bilibili.com/video/BV1i24P6rEmt/
- 创造模式入口：模式选择器选“创造模式”，或在插件页「添加插件」里进创造流程（preset `cordis`，菜单文案随版本变化）；本机 dsh 版本 0.2.1-rc.3。
