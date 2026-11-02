---
name: dsh-creator-model-architecture-prompt
title: DSH 创造模式做“模型架构可视化”插件：官方提示词原文与用法
description: 想在 DeepSeek Harness 里用一句提示词生成“模型架构示意图/模型可视化”插件时读这条：收录官方创造模式提示词合集里 Transformer 3D 可视化的中英文原文，说明创造模式的入口、粘贴与迭代方式、生成物怎么验收，并指出该合集并没有单独命名为“模型架构示意图”的条目。
category: methodology
tags: [dsh, creator-mode, prompt, model-visualization, architecture-diagram, plugin, cordis]
symptoms:
  - 想复现 DSH 创造模式视频里的“模型可视化/模型架构示意图”插件，找不到提示词原文
  - 把提示词粘进 dsh 后，一次生成的插件和视频效果不一样，不知道是提示词问题还是 one-shot 差异
  - 找不到 dsh 的创造模式入口，或不确定生成物装到了哪里、怎么再打开
stacks: [dsh, cordis]
updated: 2026-11-02
---

# DSH 创造模式做“模型架构可视化”插件：官方提示词原文与用法

> 一句话经验：先原样粘贴官方合集第 2 条“Transformer 3D 可视化”提示词，生成一个能跑的最小插件，再按“换展示对象、加交互”的方向继续对话；不要一上来就自己写长提示词。

## 适用场景

- 适用：dsh Web/桌面端（本机 0.2.1-rc.3 的界面文案与包文档中确认有创造模式，preset `cordis`），想在侧栏加一个“模型架构/模型可视化”插件，用来讲清 Transformer 这类模型内部在算什么。
- 不适用：要论文级静态插图或可编辑矢量图（用绘图工具或 Mermaid/Graphviz 类插件），或要 2D 架构图导出 PNG/SVG——官方这条是交互式 3D 演示，不是绘图库。
- 名称说明：官方合集（2026-11-02 抓取）只有 7 条 Demo，**没有**名为“模型架构示意图”的条目；最接近的是第 2 条“Transformer 3D 可视化”，B 站发布视频的简介称其为“模型可视化工具”。本条目收录的就是这一条的原文。

## 症状与信号

- 在合集页找不到“模型架构示意图”这个词，只看到 3D 地标馆、Transformer 3D、像素风、海报、视频、论文、会话统计 7 条。
- 提示词粘进去后插件能生成，但面板样式/交互和视频里不一样。
- 不知道创造模式在哪里：模式选择器里没有认出来，或插件页的添加入口点了没反应（`cordis` preset 不可用时菜单会禁用并说明原因）。

## 快速判断

1. 先确认合集版本：页面标题应为“DeepSeek Harness 创造模式 Demo 提示词合集”，正文写“七个 Demo”。不是这个版本，先换回官方链接再谈复现。
2. 确认当前 dsh 有创造模式：设置或模式选择器里能看到“创造模式”，或插件页「添加插件」菜单里的创造入口可用。
3. 确认你要的是“3D 演示”还是“2D 架构图”：前者直接抄第 2 条；后者要改写展示对象，见下面的改写示例。
4. 确认平台：提示词里的“避开 macOS 原生窗口按钮”只对 macOS 客户端有意义，Linux/Windows 下这段是空约束。

## 为什么第 2 条最接近“模型架构示意图”

这是本库按页面清单和视频简介做的对应，不是官方命名：合集 7 条里只有第 2 条讲模型内部结构，视频简介也把这一类叫“模型可视化工具”。

创造模式（preset `cordis`）把自然语言需求交给 agent，用 Cordis 插件开发指引生成一个插件 bundle。提示词里的每个限定词都会变成验收点：左侧边栏入口、可交互 3D、讲清 Transformer 内部计算、面板标题和工具栏避开 macOS 红绿灯。

官方合集自己也说明：有些插件经过多轮对话反复打磨，one-shot 提示词不一定完全复刻视频效果。所以效果有差距时，通常不是提示词坏了，而是要补第二轮对话。

## 提示词原文（官方，逐字）

中文：

```text
创建一个带左侧边栏入口的 DeepSeek Harness 插件，做成可交互的 3D 演示游戏，帮助我理解 Transformer 内部的计算逻辑。确保面板标题和工具栏避开 macOS 原生窗口按钮，不发生重叠。
```

English（官方 EN 页）：

```text
Create a DeepSeek Harness plugin with a left sidebar entry: an interactive 3D demonstration game that helps me understand the computation inside a Transformer. Keep the panel title and toolbar clear of the native macOS window buttons so they do not overlap.
```

## 在 dsh 里怎么用

1. 打开 dsh（Web 或桌面端），新建任务。
2. 选“创造模式”：在模式选择器里选，或从侧栏「插件」→「添加插件」→「让 Agent 创建插件」进入。不同版本菜单文案可能不同；判断标准是当前 preset 为 `cordis`。
3. 把上面中文提示词整段粘贴进输入框发送；要英文界面就粘 EN 版。
4. 等 agent 生成插件 bundle。按提示词要求，完成后左侧边栏会出现新入口，点开就是 3D 面板；插件以 bundle 形式安装，可在「插件」页启停。
5. 效果不满意就继续在同一任务里对话，例如“把节点改成按层分组”“加一个点击模块显示输入输出 shape 的侧栏”；不要重新开一个任务从零描述。

## 验收

- 侧栏出现入口且点击能打开 3D 面板，场景可交互（旋转/缩放/点选）。
- 面板标题和工具栏在 macOS 上不压住红绿灯；其他平台看布局不重叠即可。
- 刷新或重开 dsh 后插件仍在，且能在「插件」页启停——这是“插件”而非一次性页面的判断标准。
- 3D 面板空白时先查 WebGL：浏览器或客户端禁用硬件加速会只出空白画布。

## 改成“模型架构示意图”的改写示例（未实测）

官方那条是“3D 演示游戏”；要更接近“模型架构示意图”，保留入口和平台约束，只换展示对象与交互：

```text
创建一个带左侧边栏入口的 DeepSeek Harness 插件：用可缩放、可平移的 2D/3D 架构示意图展示一个 Transformer 编码器-解码器，按 Embedding → N×（Self-Attention → FFN）→ LM Head 分组画模块和数据流；悬停或点击模块显示输入输出 shape 与一句话说明，并支持导出 PNG。面板标题和工具栏避开 macOS 原生窗口按钮。
```

这是基于官方提示词的改写，不是官方原文，也没有在本机实测；第一次生成后按上面的验收项逐条核对。

## 常见坑

- 把“模型架构示意图”当成合集里的独立条目去找：合集没有这个名字，只有 7 条，第 2 条才是模型可视化。
- 在 Linux/WSL 上纠结 macOS 红绿灯约束：这段对非 macOS 客户端是空约束，删掉不影响结果。
- 一次生成不理想就换提示词重来：官方明确说多轮打磨，先补对话再改提示词。
- 把第三方画图插件（Excalidraw、draw.io、Mermaid 渲染类）当成这条提示词的产物：它们解决的是“画图”，这条解决的是“用 3D 演示讲模型计算”。
- 提示词里只写“画个架构图”不写入口：不写“左侧边栏入口”，生成物可能做成会话内页面，和视频的插件形态不一致。

## 参考

- 中文提示词合集（7 条 Demo）：https://cdn.deepseek.com/harness/dsh-creator-mode-prompt-collection-zh.html
- 英文提示词合集：https://cdn.deepseek.com/harness/dsh-creator-mode-prompt-collection-en.html
- 发布视频（简介称“模型可视化工具”并给出合集链接）：https://www.bilibili.com/video/BV1i24P6rEmt/
- 本机 dsh 版本 0.2.1-rc.3；创造模式对应 preset `cordis`。
