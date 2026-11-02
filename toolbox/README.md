# toolbox

可复用的脚本与工具。经验条目描述“怎么判断、怎么修”，这里放可以重复执行的部分。

## 现有工具

### `gpu-profiling-demo/`

一个可复现的 CUDA 性能实验：memory-bound 加法、低效 SGEMM、转置访存、tiny kernel launch、双 stream overlap。需要 NVIDIA GPU；包含采集脚本、真实结果和时间线重绘脚本。

```bash
cd toolbox/gpu-profiling-demo
./run_profile.sh
```

说明见 [`gpu-profiling-demo/README.md`](gpu-profiling-demo/README.md)，对应经验见 [`experiences/gpu-profiling/`](../experiences/gpu-profiling/README.md)。

> 这是唯一需要 GPU 的 toolbox 项，作为复现实验保留；它不进入默认 CI。下面的“添加工具的要求”针对默认工具链。

### `dsh-html-experience/`

一个可安装的 DSH 插件（cordis 插件 + 无依赖 CLI）：接收任意提示词，把指定的经验条目渲染成自包含的 HTML 阅读页，带四套风格预设和页面内读者样式面板。

```bash
# 在 DSH 里安装到某个 profile
dsh plugin --profile <profile> add file:$PWD/toolbox/dsh-html-experience

# 不装 DSH 也能直接生成
node toolbox/dsh-html-experience/bin/html-experience.js \
  --doc experiences/dsh-creator-mode/dsh-model-architecture-visualization.md \
  --prompt "给第一次用创造模式做插件的人" --style paper --out /tmp/reading.html
```

不需要 `npm install`；测试用 `cd toolbox/dsh-html-experience && node --test test/*.test.js`。说明见 [`dsh-html-experience/README.md`](dsh-html-experience/README.md)。

### `build_catalog.py`

构建并校验机器可读索引 `catalog.json`，同时检查每个条目都能从 `experiences/README.md` 找到。只依赖 Python 标准库。

```bash
python3 toolbox/build_catalog.py          # 重写 catalog.json
python3 toolbox/build_catalog.py --check  # 校验索引是否最新
```

新增条目后必须运行，否则 `--check` 失败。

## 添加工具的要求

- 单文件、标准库优先，避免要求用户安装额外依赖；DSH 插件/前端演示这类确实需要多文件的工具可以建子目录，自带 README、测试和安装说明。
- 顶部 docstring 写明用途、用法、输入输出和失败行为。
- 默认只读或幂等；会修改文件的工具必须提供 `--check` 或 `--dry-run`。
- 不依赖 GPU 才能运行；需要 GPU 的工具在文档中明确标注。
- 在 PR 中说明测试方式（至少覆盖正常路径和一个失败路径）。
