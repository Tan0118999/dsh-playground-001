# 贡献指南

本仓库是经验库：内容按“可检索、可执行、可验证”的标准组织。新增或修改条目时，请先读本文件，再参考 `templates/experience.md`。

## 分支与协作流程

仓库使用两条长期分支：

- `main`：稳定分支，保持可发布状态；只有 @Tan0118999 明确要求时，才把 `develop` 合入 `main`。
- `develop`：日常集成分支；任务分支从它切出，任务完成后也向它提 MR。

新建任务时：

1. 同步 `develop` 并切出任务分支：

   ```bash
   git fetch origin
   git switch develop
   git pull --ff-only
   git switch -c issue-4-develop-workflow   # <type>-<短描述>
   ```

2. 任务分支命名 `<type>-<短描述>`，`type` 取 `issue`、`feature`、`bugfix` 等，例如 `feature-gpu-profile-guide`、`bugfix-catalog-check`。
3. 在任务分支上提交，不要直接向 `develop` 或 `main` 推送。
4. 任务完成并自测通过后，从任务分支向 `develop` 提 MR；描述写清变更内容、验证方式和已知风险。
5. 请求 @Tan0118999 检视 MR，评审通过后由评审人合并到 `develop`。
6. 除非 @Tan0118999 明确要求，不要发起或执行 `develop` → `main` 的合并。

任务分支合入 `develop` 后即可删除；`develop` 与 `main` 的同步只在 @Tan0118999 要求时进行。

## 收录标准

一条经验值得入库，需要同时满足：

1. **有明确的触发条件**：能用报错原文、可观察行为或性能现象描述“什么时候会遇到”。
2. **有可执行的排查路径**：给出命令、代码或检查清单，而不是只给结论。
3. **有验证方式**：说明怎样确认修复生效、需要覆盖哪些 shape/dtype/架构/规模。
4. **有边界**：说明适用版本、技术栈、失效条件或已知误报。
5. **可公开**：不含 token、内部地址、客户数据、未公开的私有实现。

以下内容不收：纯 API 教程、没有排查过程的结论、无法复现的个例、只针对某次内部故障且无法泛化的记录。

## 目录约定

```
experiences/<topic>/<slug>.md   # 一条经验一个文件，<slug> 与 frontmatter 的 name 一致
templates/experience.md         # 新增条目的模板
toolbox/                        # 可复用脚本（构建索引、校验等）
catalog.json                    # 由脚本生成的机器索引，不要手改
```

主题域（`experiences/` 下一级目录）按领域划分，例如 `gpu-graph-compile`。同一主题内的条目保持扁平，用 `category` 和标签区分，不建更深的目录。

## frontmatter 字段

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `name` | 是 | kebab-case 唯一标识，必须与文件名一致，例如 `triton-kernel-compile-error` |
| `title` | 是 | 面向症状的中文标题，便于人快速判断 |
| `description` | 是 | 30–500 字，写清“什么时候该读这条”，包含症状/技术栈关键词，供检索与 agent 匹配 |
| `category` | 是 | `methodology` / `compile` / `crash` / `correctness` / `performance` / `engineering` 之一 |
| `tags` | 是 | 检索标签，小写，建议 3–8 个 |
| `symptoms` | 是 | 可观察现象列表，尽量保留报错原文 |
| `stacks` | 是 | `torch.compile`、`triton`、`tvm`、`xla`、`cuda` 等 |
| `updated` | 是 | 最近一次实质更新的日期，`YYYY-MM-DD` |

字段值尽量写成单行；需要冒号时用双引号包裹。列表既可以用 `[a, b]`，也可以用缩进的 `- item`。不支持块标量（`|`/`>`）和嵌套映射。

## 正文结构

优先使用模板中的顺序：

1. `一句话经验`：最重要的判断或动作。
2. `适用场景`：解决什么、不解决什么。
3. `症状与信号`：报错原文与可观察现象。
4. `快速判断`：3–6 步分流，优先低风险、可回退的检查。
5. `根因`：按可能性排序，解释机制。
6. `排查步骤`：可复制或改写的命令与观察点。
7. `修复与规避`：首选修复与可接受规避，写明代价。
8. `验证`：对拍方式、容差、覆盖范围、性能测量方法。
9. `常见坑`：误判、掩盖问题的操作、版本/缓存干扰。
10. `参考`：公开链接；没有外部来源时写“经验总结，无外部引用”。

## 写作要求

- **命令可复制**：路径、变量用占位符时要解释；需要 GPU、重编译或会显著变慢的步骤要标注。
- **区分事实与推断**：不确定的结论标“推断/待验证”，不要编造 issue 编号、版本号、性能数字或“亲测”经历。
- **精确到版本**：默认值和行为会随版本变化；无法确定版本时给出判断方法。
- **先低风险**：排查顺序从只读、可回退、低成本的检查开始；需要改代码或重编译的步骤放在后面。
- **不重复**：同一根因写入最合适的条目，其他条目用相对链接引用。
- **面向症状**：标题和 description 用用户会搜索的词，而不是内部术语。

## 索引与校验

新增、删除、重命名条目后，必须重新生成索引：

```bash
python3 toolbox/build_catalog.py          # 重写 catalog.json
python3 toolbox/build_catalog.py --check  # 校验 catalog.json 与 experiences/README.md 是否最新
```

脚本会检查：

- 必填字段齐全且非空；
- `name` 为 kebab-case、与文件名一致、全局唯一；
- `category` 在允许集合内；
- `updated` 为日期；
- `description` 长度在 30–500 之间；
- 每个条目都能从 `experiences/README.md` 找到。

若条目有新增/删除，还要同步更新：

- `experiences/<topic>/README.md` 的条目列表与症状速查；
- 顶层 `README.md` 的索引表。

## 提交前自查

- [ ] `python3 toolbox/build_catalog.py --check` 通过；
- [ ] 新增条目已加入对应主题的 README 和顶层索引；
- [ ] 命令、报错原文、版本信息没有被截断或改写；
- [ ] 没有私有数据、凭据、内部链接；
- [ ] 更新了 `updated` 字段；
- [ ] 需要时在 PR 描述里说明“哪些内容经过实测、哪些仅基于文档整理”。
