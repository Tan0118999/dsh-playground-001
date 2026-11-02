# AGENTS.md

本仓库是**经验库**，不是应用代码。Agent 消费内容来排查问题，或维护条目。以下约定同时适用于两类工作。

## 分支与 MR 流程

仓库用 `develop` 做日常集成、`main` 做稳定版本：

- 新任务从最新 `develop` 切分支，命名 `<type>-<短描述>`（`type` 用 `issue`、`feature`、`bugfix` 等），例如 `issue-4-develop-workflow`；不要直接改 `develop` 或 `main`。
- 任务完成后向 `develop` 提 MR，写清变更与验证方式，并请求 @Tan0118999 检视。
- 只有 @Tan0118999 明确要求时，才把 `develop` 合入 `main`。
- 详细规则见 [CONTRIBUTING.md 的分支与协作流程](CONTRIBUTING.md#分支与协作流程)。

## 消费：检索与使用

1. 先读 `catalog.json`（机器索引）或 `experiences/README.md`（人读索引），不要通读整个仓库。
2. 用 `description`、`symptoms`、`tags`、`stacks` 匹配当前问题；命中后完整读取条目，不要只看 frontmatter。
3. 把条目的“快速判断”当作检查清单：先做只读、可回退、低成本的检查，再改代码或重编译。
4. 执行命令前确认版本、GPU 架构和当前工作目录；条目中的默认值可能随版本变化。
5. 条目是经验，不是保证。与当前版本实测冲突时，以实测为准，并在结果中说明差异。
6. 引用条目时给出路径和标题（例如 `experiences/gpu-graph-compile/illegal-memory-access.md`），不要整段复制。
7. 不要根据条目推断用户环境；缺少版本、shape、日志时先补齐证据。

## 维护：新增与修改

1. 复制 `templates/experience.md`，文件名使用 kebab-case slug，并与 frontmatter 的 `name` 完全一致。
2. 必填字段见 `CONTRIBUTING.md`；`category` 只能取允许值。
3. `description` 必须写清“什么时候该读这条”，包含用户会搜索的症状词和技术栈，供检索匹配。
4. 命令要可复制；不确定的结论标注“推断/待验证”；不编造 issue 编号、版本、性能数字或“亲测”经历。
5. 不写入私有数据、凭据、内部链接、客户信息。
6. 新增/删除/重命名条目后：
   ```bash
   python3 toolbox/build_catalog.py
   python3 toolbox/build_catalog.py --check
   ```
   并同步更新 `experiences/<topic>/README.md` 与顶层 `README.md` 的索引。
7. 实质修改条目时更新 `updated` 字段；在 PR 描述中区分“实测验证”和“基于文档整理”。
8. 修改后至少运行 `python3 toolbox/build_catalog.py --check`；如果改了脚本，再运行 `python3 -m py_compile toolbox/build_catalog.py` 并验证 `--check` 的失败路径。

## 内容质量底线

- 一条经验必须能回答：什么现象、怎么判断、根因是什么、怎么修、怎么验证。
- 先给低风险路径，再给高风险操作；清理缓存、关闭校验、忽略错误只能作为临时手段，并要说明代价。
- 同一根因只在一个条目里详述，其他条目链接过去，避免多份互相矛盾的版本。
- 超过半年未复核的条目，在再次使用前先确认版本行为是否变化。
