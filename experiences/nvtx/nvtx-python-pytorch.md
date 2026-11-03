---
name: nvtx-python-pytorch
title: Python 与 PyTorch 里的 NVTX：annotate、torch.cuda.nvtx、emit_nvtx
description: 在 Python 训练或推理里加 NVTX、分不清 nvtx.annotate、torch.cuda.nvtx、record_function、emit_nvtx、不知道 DataLoader 多进程怎么标注时读这条。含可运行示例、PyTorch API 对照、autograd NVTX 和常见误用。
category: performance
tags: [nvtx, python, pytorch, nsys, dataloader]
symptoms:
  - Python 程序里不知道怎么打 NVTX
  - 分不清 torch.cuda.nvtx 和 record_function
  - 想给 autograd 算子自动加标注
  - DataLoader 多进程里打了 range 但看不到
  - PyTorch Profiler 的 trace 和 nsys 时间线对不上
stacks: [nvtx, python, pytorch, nsys]
updated: 2026-11-03
---

# Python 与 PyTorch 里的 NVTX：annotate、torch.cuda.nvtx、emit_nvtx

> 一句话经验：**纯 Python 用 `nvtx.annotate`，PyTorch 里 `torch.cuda.nvtx.range` 是 NVTX，`torch.profiler.record_function` 是 PyTorch Profiler 的标签；`emit_nvtx` 则是给 autograd 算子批量打 NVTX。**

## 适用场景

- 训练/推理脚本是 Python，想在 nsys 时间线里区分 dataloader、forward、backward、optimizer、通信；
- 已经在用 PyTorch Profiler，想搞清楚它和 NVTX 的关系；
- 想给每个 autograd 算子自动加 range，定位是哪个 op 慢；
- 用 DataLoader 多进程，range 出现在 worker 里但主进程时间线看不到。

不解决：C/C++ 侧 NVTX 的底层语义（见 [range 语义](nvtx-range-model.md)）；也不解决 nsys 采集/过滤（见 [nsys 工作流](nvtx-nsys-workflow.md)）。

## 症状与信号

- 代码里用了 `record_function`，但 nsys 时间线没有对应的 NVTX range；
- `torch.cuda.nvtx.range_push` 和 `range_pop` 嵌套错乱，或者异常时没有 pop；
- 想看 backward 的每个 op，但 `nvtx_sum` 里只有自己手打的几个阶段；
- DataLoader 的 worker 里打了 NVTX，报告里只有主进程；
- PyTorch Profiler 的 Chrome trace 里有 label，nsys 里却没有。

## 快速判断

1. **只标业务阶段？** Python 用 `nvtx.annotate`；PyTorch 用 `torch.cuda.nvtx.range` / `range_push` / `range_pop`。
2. **要标 autograd 的每个 op？** 用 `torch.autograd.profiler.emit_nvtx`，或 nsys 的 `--pytorch=autograd-nvtx`。
3. **要标每个 `nn.Module.forward`？** 看 nsys 的 `--pytorch=functions-trace`（版本相关）。
4. **只是用 torch.profiler 看算子表？** 用 `record_function`，它不是 NVTX，不要期待出现在 nsys 的 NVTX 行。
5. **DataLoader worker 里看不到？** 先确认报告里有没有对应 PID；再决定是否只在主进程标阶段。

## Python：`nvtx` 包

安装：

```bash
python3 -m pip install nvtx
```

最小示例（实测可运行）：

```python
import time
import nvtx

@nvtx.annotate("load", color="blue")
def load():
    time.sleep(0.06)

@nvtx.annotate()  # 默认用函数名 "forward"
def forward():
    time.sleep(0.12)

def main():
    nvtx.mark("begin", color="yellow")
    with nvtx.annotate("step", color="green", category="pipeline"):
        load()
        forward()
    nvtx.push_range("manual", domain="data")
    time.sleep(0.05)
    nvtx.pop_range(domain="data")
    with nvtx.annotate("TARGET", domain="roi"):
        time.sleep(0.10)

main()
```

```bash
nsys profile --trace=nvtx --stats=true -o py-report python3 demo.py
```

实测的 `nvtx_sum`：

```text
Time (%)  Total Time (ns)  Instances  ...  Style     Range
--------  ---------------  ---------  ...  --------  -----------
   35.3        180390332          1  ...  PushPop   :step
   23.5        120190738          1  ...  PushPop   :forward
   19.6        100178317          1  ...  PushPop   roi:TARGET
   11.8         60187700          1  ...  PushPop   :load
    9.8         50205035          1  ...  PushPop   data:manual
```

注意：

- `nvtx.annotate` 的消息会被缓存并注册成 registered string，所以 nsys 的 capture-range 可以直接用它；
- 上下文管理器不传 `message` 时默认为空字符串，建议显式命名；装饰器不传时用函数名；
- `push_range` / `pop_range` 是线程内栈，异常时会漏 `pop`；优先用 `with nvtx.annotate(...)`；
- `start_range` 返回句柄，`end_range(handle)` 结束，适合跨线程；跨线程传递句柄时自己保证只 end 一次；
- 颜色需要工具支持；`matplotlib` 可用时颜色写法更多，没装也不影响 NVTX 本身。

## PyTorch：`torch.cuda.nvtx`

PyTorch 自带一层 Python 封装，底层还是 NVTX 的 push/pop：

```python
import torch

torch.cuda.nvtx.range_push("forward")
# ...
torch.cuda.nvtx.range_pop()

# 或上下文管理器/装饰器；msg 里的 {} 会用参数格式化
with torch.cuda.nvtx.range("forward bs={}", 8):
    # ...
    pass

torch.cuda.nvtx.mark("step-start")
```

跨线程：

```python
range_id = torch.cuda.nvtx.range_start("async-job")
# 在另一个线程完成工作后
torch.cuda.nvtx.range_end(range_id)
```

对照表：

| API | 类型 | 同线程 | 说明 |
| --- | --- | --- | --- |
| `torch.cuda.nvtx.range_push(msg)` / `range_pop()` | push/pop | 是 | 返回当前深度；异常路径要自己保证 pop |
| `torch.cuda.nvtx.range(msg, *args)` | push/pop 的上下文管理器/装饰器 | 是 | 额外参数走 `msg.format()`，注意热循环里的格式化开销 |
| `torch.cuda.nvtx.range_start(msg)` / `range_end(id)` | start/end | 否 | 返回 `uint64_t` 句柄，可跨线程 |
| `torch.cuda.nvtx.mark(msg)` | mark | 不适用 | 瞬时事件 |

这些 API 在 PyTorch 2.x 的 `torch.cuda.nvtx` 文档里有说明；本机没有安装 PyTorch，行为以官方文档为准。

## PyTorch：`emit_nvtx` 与 `--pytorch`

`torch.autograd.profiler.emit_nvtx` 会让**每个 autograd 操作**发出一个 NVTX range，适合在 nsys 里看 forward/backward 的算子级时间线：

```python
import torch

with torch.autograd.profiler.emit_nvtx(record_shapes=False):
    loss = model(x).sum()
    loss.backward()
```

- `record_shapes=True` 会在 range 里附带张量 shape，但会增加开销；
- 旧文档里它配合 nvprof 使用；在 nsys 下仍然可以使用，nsys 也提供了更自动的入口：

```bash
# 让 PyTorch 自动调用 emit_nvtx（record_shapes=False）
nsys profile --trace=cuda,nvtx --pytorch=autograd-nvtx -o report python3 train.py

# 带 shape
nsys profile --trace=cuda,nvtx --pytorch=autograd-shapes-nvtx -o report python3 train.py

# 给 torch.Module.forward 生成 range（nsys 版本相关）
nsys profile --trace=cuda,nvtx --pytorch=functions-trace -o report python3 train.py
```

`--pytorch` 的具体取值以本机 `nsys profile --help` 为准；这些选项会隐式打开 NVTX trace。

## 不要混淆：`record_function` 不是 NVTX

`torch.profiler.record_function`（实现上是 `torch.autograd.profiler.record_function`）的作用是：

> 在 autograd profiler 运行时给一段代码加 label；只有开启 CPU activity tracing 才会出现在 PyTorch Profiler 的结果里。

它适合 `torch.profiler` / TensorBoard / Chrome trace 的算子表，**默认不会变成 nsys 的 NVTX range**。想让同一个阶段同时出现在两边，可以两个都加，或者统一用 NVTX + nsys：

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with torch.profiler.record_function("forward"):  # PyTorch Profiler 的 label
        with torch.cuda.nvtx.range("forward"):      # nsys 的 NVTX range
            y = model(x)
    y.sum().backward()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
```

## DataLoader 与多进程

- DataLoader 的 worker 是独立进程；在 `Dataset.__getitem__` 里打的 NVTX 属于 worker 的 PID，不会自动出现在主进程的线程行里；
- 先在报告里确认 worker PID 是否存在；没有就检查采集方式、fork/exec、容器 PID namespace 和 nsys 的多进程选项；
- worker 里每个 batch 都打 range 可能非常密，建议只在关键路径（如解码、增强、collate）打，或用采样/计数控制频率；
- 跨进程 correlation 不是 NVTX 自动完成的；需要把 batch/step 等标识放进 range 名或 payload，再在报告里对齐。

## 修复与规避

- **阶段标注统一用一种入口**：业务代码用 `nvtx.annotate` 或 `torch.cuda.nvtx.range`，不要一半 `record_function` 一半 NVTX；
- **异常安全**：Python 用 `with`；手动 push/pop 要配 `try/finally`；
- **不要在每个 micro-op 上打 range**：autograd 级用 `emit_nvtx`，手写阶段只包毫秒级区间；
- **动态字符串少用**：`f"forward bs={bs}"` 会付出格式化成本；固定模板 + payload 更好（见 [进阶 API](nvtx-advanced-api.md)）；
- **DataLoader 先确认 PID**：报告里没有 worker，先在采集侧解决，不要改 NVTX 代码；
- **PyTorch Profiler 与 NVTX 是两条链路**：需要 nsys 时间线就用 NVTX，需要算子表和 Chrome trace 就用 `torch.profiler`。

## 验证

- `nvtx_sum` 里能看到 `:step`、`:forward`、`roi:TARGET` 等名字，实例数与调用次数一致；
- Python `nvtx.annotate` 生成的是 registered string，`nsys --capture-range=nvtx --nvtx-capture=TARGET@roi` 能触发（实测）；
- 用 `emit_nvtx` 后，nsys 里出现 autograd 算子级 range；关掉后这些 range 消失；
- 用 `record_function` 时，PyTorch Profiler 表里有 label，但 nsys 的 NVTX 行不一定有；这是预期行为，不是 bug；
- DataLoader worker 的 range 能在对应 PID 下找到，且没有把每个 batch 都打成海量事件。

## 常见坑

1. **把 `record_function` 当 NVTX**：它只进 PyTorch Profiler 的 trace；
2. **`range_push` 后异常没 `pop`**：后续 range 全部嵌套错；用 `with`；
3. **在 `forward` 里用 `msg.format` 拼动态字符串**：格式化发生在 NVTX 调用之前，工具不在也会付出成本；
4. **worker 里打了但报告没有**：先查 PID/多进程采集，不要先怀疑 API；
5. **`emit_nvtx` 打开后事件太多**：autograd 级别会产生大量 range；先缩小 step 或用 capture-range；
6. **PyTorch 版本差异**：`range` 上下文管理器、`--pytorch` 取值和 `record_function` 的行为随版本变化，执行前看对应版本文档和 `nsys --help`；
7. **把 NVTX 当同步/计时 API**：它不改变程序行为，也不保证精确计时；计时用 CUDA events 或 profiler。

## 参考

- NVTX Python 文档：<https://nvidia.github.io/NVTX/python/>
- PyTorch `torch.cuda.nvtx.range`：<https://docs.pytorch.org/docs/stable/generated/torch.cuda.nvtx.range.html>
- PyTorch `torch.cuda.nvtx.range_push`：<https://docs.pytorch.org/docs/stable/generated/torch.cuda.nvtx.range_push.html>
- PyTorch `emit_nvtx`：<https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.emit_nvtx>
- PyTorch `record_function`：<https://docs.pytorch.org/docs/stable/generated/torch.autograd.profiler.record_function.html>
- PyTorch Profiler：<https://docs.pytorch.org/docs/stable/profiler.html>
- Nsight Systems NVTX Trace：<https://docs.nvidia.com/nsight-systems/UserGuide/index.html#nvtx-trace>
