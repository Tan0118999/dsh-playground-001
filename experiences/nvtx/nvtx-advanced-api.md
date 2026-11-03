---
name: nvtx-advanced-api
title: NVTX 进阶：attributes、payload、domain、registered string 与 C++ RAII
description: 想给 range 加 shape/step 等附加数据、按库分 domain、减少字符串开销、用 C++ RAII 或自动函数标注、控制 NVTX 开销时读这条。含 C/C++/Python 属性写法、registered string、payload 和开销边界。
category: methodology
tags: [nvtx, attributes, payload, domains, cpp, raii]
symptoms:
  - 想给 range 附加 step、shape、loss 等数值
  - 多个库的 range 混在一起，需要隔离
  - 动态字符串让 NVTX 开销变大
  - 手动 push/pop 容易漏 pop
  - 想知道 NVTX 在热路径里能不能用
stacks: [nvtx, cuda, cpp, python, nsys]
updated: 2026-11-03
---

# NVTX 进阶：attributes、payload、domain、registered string 与 C++ RAII

> 一句话经验：**固定的名字用 registered string，变化的数据用 payload，库/模块用 domain，生命周期用 RAII；不要把格式化和字符串拼接放进热路径。**

## 适用场景

- 已经会打基本 range，想让它携带更多信息（step、batch、shape、loss）；
- 代码里有多个库/模块都在打 NVTX，需要按 domain 隔离；
- 热路径里的 NVTX 开销需要控制；
- C++ 代码经常提前 `return`/抛异常，想保证 range 一定闭合；
- 想用 `NVTX3_FUNC_RANGE()` 自动标注函数。

不解决：怎么在 nsys 里采集/过滤（见 [nsys 工作流](nvtx-nsys-workflow.md)）；也不解决 ncu 的 NVTX 过滤（见 [消费端与自动化](nvtx-ecosystem-and-automation.md)）。

## 症状与信号

- `nvtx_sum` 里所有 step 都叫 `train-step`，无法区分第几步；
- 用 `sprintf`/f-string 拼 range 名，关掉 profiler 后程序仍变慢；
- 框架和自研代码都有 `forward`，统计时合并成一行；
- 函数中途 `return`，后面的 range 嵌进一个“幽灵父 range”；
- 想加颜色/分类，但不知道 `nvtxEventAttributes_t` 哪些字段必须填；
- 不确定 NVTX 能不能放进微秒级循环。

## 快速判断

1. **名字是否固定？** 固定 → registered string；不要每次拼字符串。
2. **是否有可变数值？** 有 → payload（int/uint/double/float），不要塞进 message。
3. **是否跨库/跨模块？** 是 → 独立 domain；domain 内再按 category 分组。
4. **C++ 是否有异常/提前返回？** 用 `nvtx3::scoped_range` / `NVTX3_FUNC_RANGE()`，不要手写裸 push/pop。
5. **是否跨线程？** 用 `unique_range` 或 start/end 句柄，不要跨线程 pop。
6. **是否在 <1 µs 的热路径？** 默认不要打；先测量工具关闭和打开两种情况的差异。

## 三层 API

| 层 | 头文件/模块 | 适合 |
| --- | --- | --- |
| C | `<nvtx3/nvToolsExt.h>` | 最大控制力；结构体属性、domain、payload、资源命名 |
| C++ | `<nvtx3/nvtx3.hpp>` | RAII、类型安全、`NVTX3_FUNC_RANGE()`、domain 模板 |
| Python | `nvtx` 包 / `torch.cuda.nvtx` | 装饰器/上下文管理器、颜色、category、payload |

三层最终都落到同一套 NVTX 事件；选你所在语言最自然的入口即可。

## attributes：颜色、category、payload

C 的 `nvtxEventAttributes_t` 必须设置 `version` 和 `size`，否则工具可能忽略：

```cpp
#include <nvtx3/nvToolsExt.h>

void train_step(int step, float loss) {
  nvtxEventAttributes_t attr = {};
  attr.version = NVTX_VERSION;                    // 必须
  attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;      // 必须

  attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attr.message.ascii = "train-step";              // 固定名字
  attr.category = 1;
  attr.colorType = NVTX_COLOR_ARGB;
  attr.color = 0xFF00FF00;                        // ARGB：不透明绿色

  attr.payloadType = NVTX_PAYLOAD_TYPE_INT64;
  attr.payload.llValue = step;                    // 变化的数据放 payload

  nvtxRangePushEx(&attr);
  // ...
  nvtxRangePop();
}
```

payload 类型与 union 字段：

| `payloadType` | union 字段 |
| --- | --- |
| `NVTX_PAYLOAD_TYPE_UNSIGNED_INT64` | `payload.ullValue` |
| `NVTX_PAYLOAD_TYPE_INT64` | `payload.llValue` |
| `NVTX_PAYLOAD_TYPE_DOUBLE` | `payload.dValue` |
| `NVTX_PAYLOAD_TYPE_UNSIGNED_INT32` | `payload.uiValue` |
| `NVTX_PAYLOAD_TYPE_INT32` | `payload.iValue` |
| `NVTX_PAYLOAD_TYPE_FLOAT` | `payload.fValue` |

payload 在 nsys 导出的 SQLite 里分别落到 `uint64Value`、`int64Value`、`doubleValue` 等列；GUI 的展示位置随版本变化，但数据在报告里。一个事件只有一个 payload；需要多个字段时，优先拆成多个 range，或用 NVTX 的扩展 payload（见官方文档）。

## domain 与 category

- **domain** 是命名空间：每个 domain 有自己的 range 栈、category 和 registered string 表；
- **category** 是 domain 内的整数分组，工具可用它过滤/着色；
- 默认 domain 用 `NULL` 表示，显示为 `:range`；自定义 domain 显示为 `domain:range`。

```cpp
nvtxDomainHandle_t dom = nvtxDomainCreateA("com.example.data");
nvtxDomainNameCategoryA(dom, 1, "io");

nvtxEventAttributes_t attr = {};
attr.version = NVTX_VERSION;
attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
attr.message.ascii = "load-batch";
attr.category = 1;

nvtxDomainRangePushEx(dom, &attr);
// ...
nvtxDomainRangePop(dom);

nvtxDomainDestroy(dom);
```

命名建议：domain 用反向域名或 `模块.子模块`（例如 `com.example.data`、`myteam.inference`），category 用 `io`、`compute`、`comm` 这类短词。nsys 支持 `--nvtx-domain-include/exclude` 按 domain 过滤。

## registered string

固定消息每次 `push` 都会复制字符串；工具还要反复匹配。registered string 把消息注册一次，之后只传句柄：

```cpp
nvtxDomainHandle_t dom = nvtxDomainCreateA("com.example.train");
nvtxStringHandle_t msg = nvtxDomainRegisterStringA(dom, "train-step");

nvtxEventAttributes_t attr = {};
attr.version = NVTX_VERSION;
attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
attr.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
attr.message.registered = msg;
attr.category = 1;

nvtxDomainRangePushEx(dom, &attr);
// ...
nvtxDomainRangePop(dom);
```

为什么值得做：

- 减少字符串复制和工具端匹配；
- nsys 的 `--capture-range=nvtx` **默认只匹配 registered string**；
- SQLite 里 registered string 的 `text` 是 NULL，名字在 `StringIds` 表；脚本要 join 才能看到名字。

C++ 可以用 `nvtx3::registered_string_in<domain>::get<Tag>()` 做“首次使用时注册”，比手写全局句柄更安全；具体签名见 C++ API 参考。

## C++ RAII：scoped_range、unique_range、函数范围

```cpp
#include <nvtx3/nvtx3.hpp>

struct my_domain {
  static constexpr char const* name{"com.example.cpp"};
};

__attribute__((noinline)) void my_function() {
  NVTX3_FUNC_RANGE();  // 用函数名做 registered string，函数返回时自动 pop
  // ...
}

void f() {
  // 默认 domain 的 RAII range；异常/return 都会 pop
  nvtx3::scoped_range outer{"outer"};

  // 自定义 domain + 属性
  using my_range = nvtx3::scoped_range_in<my_domain>;
  my_range r{nvtx3::event_attributes{
      "attributed", nvtx3::rgb{0, 128, 255}, nvtx3::payload{42}, nvtx3::category{1}}};

  // 需要跨线程/可移动时用 unique_range；析构时结束
  nvtx3::unique_range u{"async"};
  // std::move(u) 到另一个线程，让它在那边析构即可
}
```

实测 `nvtx_sum` 会把 `NVTX3_FUNC_RANGE()` 显示为 `:my_function`，把 `unique_range` 显示为 `StartEnd`，把自定义 domain 显示为 `com.example.cpp:attributed`。`scoped_range` 是栈对象，析构顺序严格嵌套；`unique_range` 可移动、可在不同线程析构，但有额外开销，普通场景优先 `scoped_range`。

## Python 的属性

```python
import nvtx

with nvtx.annotate(
    "train-step",
    color="green",
    category="train",
    payload=step,
    domain="app",
):
    # ...
    pass

nvtx.mark("checkpoint-saved", color="yellow", payload=loss)
```

`nvtx.annotate` 的消息会被缓存并注册为 registered string；`payload` 支持整数/浮点等标量（也可传数组类，具体支持范围见 Python 文档）。不要在热循环里用 f-string 拼消息；固定模板 + payload 更稳。

## 开销与边界

官方对开销的描述可以总结成：

- **没有工具接入**：第一次调用初始化后，后续调用基本是几条指令；但**调用前的参数准备不会被跳过**，`sprintf`/f-string 仍然要花时间；
- **有工具接入**：开销由工具决定；nsys 的 range 记录本身也有成本；
- **第一次调用**：可能加载注入库，延迟敏感场景可用 `nvtxInitialize` 提前初始化；
- **微秒级代码**：官方建议，被标注代码通常小于 1 µs 时要谨慎；nsys 文档也建议同时打开的 range 不要太多（经验上几十个以内）；
- **registered string**：减少字符串复制和工具端匹配，热路径首选；
- **payload**：比格式化字符串便宜，但不要把一个事件塞成日志系统；
- **NVTX 是宿主侧 API**：不支持在 `__device__` 代码里打点；GPU 内部的细粒度事件用 `pmevent`/硬件计数器；
- **单工具限制**：同一进程只支持一个 NVTX 工具接入，不要同时挂多个 profiler。

发布构建可以用 `-DNVTX_DISABLE` 把所有 NVTX 调用预处理掉（实测报告里不再出现 range）；代价是该构建完全失去标注能力。

## 修复与规避

- 固定名字用 registered string；变化数据用 payload；
- 每个库/模块一个 domain，domain 内用 category 分组；
- C++ 用 RAII，禁止在异常路径上依赖手写 pop；
- 跨线程用 `unique_range`/start-end，不要跨线程 pop；
- 热路径先测量再决定是否保留；不要为了“完整”给每个算子都打 range；
- 需要多个字段时，拆成多个嵌套 range 或用扩展 payload，不要拼字符串；
- 发布构建用 `NVTX_DISABLE` 前，确认没有代码依赖 NVTX 返回值（NVTX 本身也不应该影响程序行为）。

## 验证

- `nvtx_sum` 中出现 domain 前缀、正确的 Style 和实例数；
- SQLite 中 `category`、`color`、`int64Value`/`doubleValue`、`domainId` 与代码一致；
- `NVTX3_FUNC_RANGE()` 的 range 名与函数名一致；
- 异常/提前 `return` 后没有未闭合 range；
- 用 `-DNVTX_DISABLE` 编译后报告里没有 NVTX 事件；
- 热路径做 A/B：工具关闭时 NVTX 版本与无 NVTX 版本差异在可接受范围内，工具打开时明确知道开销来自 profiler。

## 常见坑

1. **忘记设置 `attr.version` / `attr.size`**：属性可能被忽略；
2. **payloadType 和 union 字段不匹配**：读出来是垃圾值；
3. **把可变字符串拼进 message**：工具不在时也有格式化开销；
4. **registered string 在 domain 销毁后继续使用**：句柄失效；
5. **domain 销毁时还有活跃 range**：报告缺段；
6. **`unique_range` 当 `scoped_range` 用**：它有额外开销，且生命周期更容易写错；
7. **在 device 代码里调用 NVTX**：不支持；
8. **多个 profiler 同时接入**：NVTX 只支持一个工具。

## 参考

- NVTX 官方总览（attributes、domains、registered string、开销、`NVTX_DISABLE`）：<https://github.com/NVIDIA/NVTX>
- NVTX C/C++ 示例（`scoped_range`、`unique_range`、`NVTX3_FUNC_RANGE`）：<https://github.com/NVIDIA/NVTX/blob/release-v3/c/README.md>
- NVTX C API 参考（`nvtxEventAttributes_t`、payload、domain）：<https://nvidia.github.io/NVTX/doxygen/>
- NVTX C++ API 参考：<https://nvidia.github.io/NVTX/doxygen-cpp/>
- NVTX Python 文档：<https://nvidia.github.io/NVTX/python/>
- Nsight Systems NVTX Trace（开销和 range 数量建议）：<https://docs.nvidia.com/nsight-systems/UserGuide/index.html#nvtx-trace>
