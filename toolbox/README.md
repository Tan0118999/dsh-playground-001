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

### `build_catalog.py`

构建并校验机器可读索引 `catalog.json`，同时检查每个条目都能从 `experiences/README.md` 找到。只依赖 Python 标准库。

```bash
python3 toolbox/build_catalog.py          # 重写 catalog.json
python3 toolbox/build_catalog.py --check  # 校验索引是否最新
```

新增条目后必须运行，否则 `--check` 失败。

## 添加工具的要求

- 单文件、标准库优先，避免要求用户安装额外依赖。
- 顶部 docstring 写明用途、用法、输入输出和失败行为。
- 默认只读或幂等；会修改文件的工具必须提供 `--check` 或 `--dry-run`。
- 不依赖 GPU 才能运行；需要 GPU 的工具在文档中明确标注。
- 在 PR 中说明测试方式（至少覆盖正常路径和一个失败路径）。
