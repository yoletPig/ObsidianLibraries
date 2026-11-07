# 02 - ReAct：推理与行动循环

## 本篇目标

1. 理解 ReAct 论文的思想：让模型交替输出"推理"和"行动"；
2. 不用任何框架，手写一个约 100 行的最小 ReAct Agent；
3. 理解循环的终止条件与失败模式。

## 1. ReAct 是什么

**ReAct = Reasoning + Acting**，来自论文 *ReAct: Synergizing Reasoning and Acting in Language Models*（Shunyu Yao 等人，普林斯顿大学与 Google Research，2022 年 10 月发布于 arXiv，发表于 ICLR 2023）。

论文的出发点：LLM 有两种内在能力——

- **推理（reasoning）**：链式思考（Chain-of-Thought），一步步想，但想法可能与世界脱节，容易幻觉；
- **行动（acting）**：直接输出答案或动作，但没有中间推理，错了没法自我修正。

ReAct 让模型在解决问题时**交替产生推理轨迹（thought）和行动（action）**，行动会从外部环境（搜索引擎、计算器等）带回**观测（observation）**，观测又被纳入下一轮推理。三者组成循环：

```
Thought: 我需要先查X公司的总部所在地 →
Action: Search[X公司 总部] →
Observation: X公司总部位于上海 →
Thought: 有了地点，接下来查当地天气 →
Action: Search[上海 今日天气] →
Observation: 晴，28℃ →
Answer: ...
```

论文在 HotpotQA、FEVER 和 ALFWorld 等任务上验证了这种方式优于单纯的 CoT 或单纯的行动，并且人类可以透过 thought 轨迹直接审查模型的推理过程、诊断错在哪一步。

> 注意一个常见误传：ReAct **不是** "Function Calling 的别名"。Function Calling（下一篇）是模型 API 层面结构化输出工具调用的机制；ReAct 是一种**提示词驱动的交替推理-行动模式**。现代 Agent 框架常用"原生工具调用"来实现 ReAct 循环，但 ReAct 的提示词写法至今仍然有用（模型不支持原生工具调用时，或需要显式思考轨迹时）。

## 2. 最小实现：只用 Python 标准库

下面这个 Agent 不依赖任何 LLM SDK——`llm()` 是占位函数，接入任何一家 API 都行。它包含 4 个工具：计算器、模拟搜索、当前时间、任务完成声明。

```python
# minimal_react.py — 最小 ReAct Agent（约 100 行，无框架依赖）
import json
import re
import datetime

def llm(prompt: str) -> str:
    """占位：接入任意 LLM API。prompt 进，文本出。"""
    raise NotImplementedError("接入你的 LLM API")

# ---------- 工具实现 ----------
def tool_calculator(expression: str) -> str:
    # 演示用；生产环境绝不应对 LLM 产出直接 eval，
    # 见 08 篇权限与沙箱。
    allowed = re.fullmatch(r"[\d\s+\-*/().]+", expression)
    if not allowed:
        return "错误：表达式包含不允许的字符"
    return str(eval(expression))  # noqa: S307（已白名单校验，仅演示）

def tool_search(query: str) -> str:
    # 演示用的假搜索；真实实现接搜索 API
    return f"（模拟搜索结果）关于「{query}」：这里是检索到的资料……"

def tool_now(_: str) -> str:
    return datetime.datetime.now().isoformat()

TOOLS = {
    "Calculator": tool_calculator,
    "Search": tool_search,
    "Now": tool_now,
}

# ---------- 提示词 ----------
SYSTEM = """你是一个使用 ReAct 模式的助手。每轮严格输出以下两种格式之一：

Thought: <你的推理，一句话>
Action: <工具名>[<工具输入>]

可用工具：
{tool_desc}

当信息足够回答问题时，输出：
Answer: <最终答案>

规则：一次只输出一个 Thought 和一个 Action（或 Answer）。"""

def build_tool_desc() -> str:
    return "\n".join(f"- {name}: {fn.__doc__ or ''}" for name, fn in TOOLS.items())

MAX_STEPS = 8  # 关键安全阀：防止死循环

def run_agent(question: str) -> str:
    trajectory = [f"Question: {question}"]
    for step in range(MAX_STEPS):
        prompt = SYSTEM.format(tool_desc=build_tool_desc()) + "\n\n" + "\n".join(trajectory)
        output = llm(prompt).strip()

        ans = re.search(r"Answer:\s*(.+)", output, re.S)
        if ans:
            return ans.group(1).strip()

        act = re.search(r"Action:\s*(\w+)\[(.*)\]", output, re.S)
        if not act:
            # 模型输出不合规：把错误观测回喂，让它自己纠正
            trajectory.append(f"Observation: 输出格式错误，请严格使用 Action: 工具名[输入]")
            continue

        name, arg = act.group(1), act.group(2).strip()
        if name == "Finish":  # 允许模型显式声明完成
            return arg
        if name not in TOOLS:
            obs = f"错误：未知工具 {name}，可用工具：{list(TOOLS)}"
        else:
            try:
                obs = TOOLS[name](arg)
            except Exception as e:  # 工具错误也作为观测回喂
                obs = f"工具执行出错: {e}"

        trajectory.append(output)          # Thought + Action
        trajectory.append(f"Observation: {obs}")

    return "已达最大步数仍未得出答案，任务终止。"

if __name__ == "__main__":
    print(run_agent("现在离 2030 年 1 月 1 日还有多少天？"))
```

## 3. 逐段拆解：四个设计要点

### 要点 1：观测（Observation）是 Agent 的"感官"

工具结果必须**回喂**到下一轮 prompt 里。上面代码用 `trajectory.append(f"Observation: {obs}")` 实现这一点。Agent 的智能很大程度上来自"观测质量"：报错信息写得清楚，模型自纠错的能力就强；返回 `None` 或空字符串，模型就会瞎猜。

### 要点 2：错误也是观测

注意 `except Exception` 和"未知工具"分支——**不要在工具失败时抛异常终止循环**，而是把错误信息作为 Observation 喂回去。这是 Agent 自我修正的核心机制（也是 07 篇要讲的"错误级联"问题的第一道防线）。

### 要点 3：终止条件必须有冗余

代码里有两层终止：模型显式输出 `Answer:`，以及 `MAX_STEPS` 硬上限。**永远不要只依赖模型自己说停**——循环、重复调用同一工具是 ReAct Agent 最常见的故障（07 篇详述）。工业级 Agent 通常还有第三层：token/预算上限。

### 要点 4：格式解析要宽容 + 可恢复

正则解析失败时不该崩溃，而是回喂"格式错误"观测让模型重试。这是提示词驱动 ReAct 的固有脆弱性；原生 Function Calling（03 篇）把工具调用变成结构化 JSON，就是为了消掉这一层脆弱性。

## 4. ReAct 的已知局限

论文和后续研究指出的几点，写 Agent 前应心里有数：

1. **循环与跑偏**：模型可能反复执行同一个动作。对策：步数上限 + 在观测中提示"你已连续 N 次调用同一工具"。
2. **推理幻觉**：thought 可能陈述不存在的事实（论文称之为通过假想动作"欺骗"自己）。对策：关键事实必须经由工具确认，而非依赖模型记忆。
3. **对提示词格式敏感**：格式约束的微小变化可能显著影响效果，这是早期论文实验中观察到的现象。
4. **长轨迹的上下文压力**：每步都在增长的历史会吃掉上下文窗口（05 篇专门处理）。

## 自测

1. ReAct 循环中，Observation 的两种来源是什么？（提示：工具结果，和……）
2. 为什么 `MAX_STEPS` 是必须的？删掉它会怎样？
3. 不看代码，说出 `run_agent` 主循环的五个步骤。

## 下一篇

[03 - 工具调用与 Function Calling](03-工具调用与FunctionCalling.md)——用 API 原生能力替代脆弱的正则解析。

## 参考来源

- Yao, S. et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, arXiv:2210.03629（2022-10），ICLR 2023: https://arxiv.org/abs/2210.03629
- 论文项目页（含 HotpotQA/FEVER/ALFWorld 实验与交互演示）: https://react-lm.github.io/
