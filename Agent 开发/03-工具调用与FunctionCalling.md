# 03 - 工具调用与 Function Calling

## 本篇目标

1. 理解原生 Function Calling 的工作原理（重点：**模型不执行工具，只输出调用请求**）；
2. 掌握工具描述（JSON Schema）的写法——它直接决定调用质量；
3. 写一个健壮的工具执行器：参数校验、错误回喂、并发执行。

## 1. Function Calling 的真相：模型只是"点菜"

初学者最常见的误解是以为模型"调用"了函数。实际上：

```
┌────────┐  ① 请求 + 工具清单(JSON Schema)   ┌────────┐
│ 你的代码 │ ──────────────────────────────→ │  LLM    │
│        │                                  │        │
│        │  ② 返回"我想调 read_file，参数是…"  │        │
│        │ ←──────────────────────────────  │        │
│        │
│        │  ③ 你的代码执行 read_file("a.txt")
│        │  ④ 把执行结果作为新消息发回 LLM
│        │ ←──────────── ⑤ 模型继续推理 ────→ …循环…
└────────┘
```

- **第②步模型输出的是结构化的调用意图**（函数名 + JSON 参数），不是执行结果；
- **第③步执行永远发生在你的代码里**。模型没有任何执行能力；
- 第④步回喂后模型继续，直到它不再请求工具、给出最终回答。

这套机制最早由 OpenAI 于 2023 年 6 月在 API 中推出（gpt-3.5-turbo-0613 / gpt-4-0613），随后 Anthropic（Claude，称为 tool use）、Google（Gemini）等主流模型 API 都提供了类似能力，具体字段名各家略有差异，但结构一致：**工具清单进 → 调用请求出 → 结果回喂**。

以 OpenAI Python SDK（v1.x）为例的最小骨架：

```python
from openai import OpenAI
import json

client = OpenAI()  # 从环境变量读取 OPENAI_API_KEY

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的当前天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，如：北京"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                         "description": "温度单位，默认 celsius"}
            },
            "required": ["city"]
        }
    }
}]

messages = [{"role": "user", "content": "北京今天多少度？"}]

resp = client.chat.completions.create(
    model="gpt-4o-mini",   # 示例模型，可替换
    messages=messages,
    tools=tools,
)
msg = resp.choices[0].message

# 模型决定调用工具
if msg.tool_calls:
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)   # 注意：这是 JSON 字符串
        result = get_weather(**args)               # ← 执行在你的代码里
        messages.append(msg)                       # 助手的"调用意图"入历史
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result, ensure_ascii=False),
        })
    # 第二次请求：模型看到工具结果，继续推理
    resp2 = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=tools)
    print(resp2.choices[0].message.content)
```

三个最容易踩的坑已在注释里：`function.arguments` 是 **JSON 字符串**不是对象，要 `json.loads`；助手的工具调用消息必须原样入历史；工具结果要用 `role: "tool"` 并带上 `tool_call_id`。漏掉任何一步，API 都会报 400 错误。

## 2. 工具描述：Agent 质量的第一决定因素

模型选不选工具、传什么参数，**几乎完全取决于工具的 name/description/schema**。把工具描述当成"给一个聪明但毫无上下文的新同事的操作手册"来写：

**坏例子：**

```json
{"name": "search", "description": "搜索", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}
```

**好例子：**

```json
{
  "name": "search_orders",
  "description": "在订单系统中按条件搜索订单。仅当用户询问订单状态、物流、退款时使用；不要用于查询商品信息（用 get_product）。返回最多 20 条，按创建时间倒序。",
  "parameters": {
    "type": "object",
    "properties": {
      "status": {"type": "string", "enum": ["pending", "shipped", "refunded"],
                 "description": "订单状态过滤"},
      "keyword": {"type": "string", "description": "商品名关键词，模糊匹配"}
    },
    "required": ["keyword"]
  }
}
```

实践准则（业界共识性经验，非某篇论文的结论）：

1. **描述"何时用 + 何时不用"**：像好例子里"不要用于查询商品信息（用 get_product）"这样的负向指引，能显著减少误选；
2. **用 enum 收窄参数**：自由字符串换成枚举， hallucination 参数的概率大幅下降；
3. **参数宁少勿多**：每个可选参数都是模型犯错的机会；
4. **工具数量控制在合理的规模**：几十个以上时选择准确率会下降，应做工具检索/分组（本仓库 VLM & VLA Stage 9 也讨论过同样问题在多模态场景的形态）；
5. **返回结构要"面向模型"**：返回 JSON 比"漂亮格式化的句子"更好解析；错误时返回可操作的提示（"未找到订单，试试订单号 XXX 格式"）而不是 bare 的 `Error`。

## 3. 健壮的工具执行器

生产级的执行器要做四件事：**校验、隔离、回喂、限流**。

```python
import json
import time
from typing import Any, Callable

class ToolError(Exception):
    """带用户可读信息的工具错误——这类信息会回喂给模型"""

def execute_tool(
    registry: dict[str, Callable[..., Any]],
    name: str,
    args: dict,
    *,
    max_output_chars: int = 4000,
) -> str:
    """执行一个工具调用，永远返回字符串（成功或错误），绝不向上抛异常。"""
    if name not in registry:
        return f"错误：未知工具 {name}。可用工具：{sorted(registry)}"

    fn = registry[name]
    t0 = time.monotonic()
    try:
        result = fn(**args)
    except TypeError as e:
        # 参数名/个数不匹配
        return f"错误：参数不匹配（{e}）。请对照 schema 重新生成参数。"
    except ToolError as e:
        return f"工具执行失败：{e}"
    except Exception as e:  # noqa: BLE001
        return f"工具内部错误：{type(e).__name__}"

    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    # 截断保护：一个工具返回 500KB 会瞬间吃掉上下文窗口
    if len(text) > max_output_chars:
        text = text[:max_output_chars] + f"\n…[输出被截断，原始长度 {len(text)} 字符]"
    return text
```

设计要点：

- **永不向上抛异常**：异常变成文本回喂（02 篇的要点 2 在这里的落地）；
- **输出截断**：单个工具的返回必须限长。一次 `cat` 大文件就能让后续所有轮次退化（07 篇"上下文腐化"）；
- **TypeError 单独处理**：LLM 生成错参数名是高频错误，专属的错误文案能引导它自纠。

> 并发执行：当模型一次请求返回多个工具调用（`tool_calls` 是列表），彼此无副作用的调用可用 `concurrent.futures.ThreadPoolExecutor` 并行执行，能显著降低总延迟。有副作用（写文件、下单）的调用务必串行。

## 4. 工具调用的安全前置：能力边界

本篇先立规矩，08 篇展开实现：

- **写操作 > 读操作的危险等级**。读文件、查询可以放行；写文件、发请求、删数据必须经过权限检查或人审；
- **工具注册表本身就是权限边界**：不注册 = 不存在 = 模型无法调用。最小权限原则从"少给工具"开始；
- **不要把 shell 当万能工具**暴露给模型。"执行任意命令"工具会把上面所有设计一夜间作废。

## 自测

1. Function Calling 中，"执行工具"发生在哪里？模型的输出本质上是什么？
2. `function.arguments` 的类型是什么？工具结果用什么 role 回喂？
3. 写出三条工具描述的实践准则，并解释每条为什么能减少错误。

## 下一篇

[04 - MCP 协议详解](04-MCP协议详解.md)——工具、资源与提示的开放标准。

## 参考来源

- OpenAI, *Function calling and other API updates*（2023-06-13，发布 function calling）: https://openai.com/blog/function-calling-and-other-api-updates （现已归档于 openai.com/index/...）
- OpenAI API 文档·Function Calling: https://platform.openai.com/docs/guides/function-calling （`arguments` 为 JSON 字符串、`role: "tool"` 回喂等细节以此文档为准）
- Anthropic API 文档·Tool use: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use
