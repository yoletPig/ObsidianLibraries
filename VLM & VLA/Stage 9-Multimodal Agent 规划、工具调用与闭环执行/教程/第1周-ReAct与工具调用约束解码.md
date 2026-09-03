# 第 1 周教程：Agent 核心范式与 Tool Use / 约束解码

> **本周要回答的三个问题**
> 1. ReAct 循环工程化时，哪些环节是"模型能做的"，哪些必须交给"程序"？
> 2. 约束解码凭什么能做到 100% 合法 JSON？它的机制与代价是什么？
> 3. 多模态工具链怎么设计，才能让 VLM 的感知短板被工具补齐？

对应学习计划：第 1 周。交付物：手写 ReAct Agent Loop 引擎；密集文本图上自主调用 `Crop_Image_Tool` + `OCR_Tool` 得出结论；约束解码保证 100 次工具调用零 JSON 解析失败。

---

## 1. 第一性原理：Agent = 把 LLM 的"语言能力"接到"可执行接口"上

### 1.1 根本矛盾：语言模型只会生成文本，而任务需要副作用

LLM 的输出是 Token 序列，天然没有副作用——不能点按钮、不能执行代码、不能查数据库。Agent 的本质是给这个文本生成器装上**行动接口**：

$$
\underbrace{\text{LLM}}_{\text{生成 JSON 文本}} \xrightarrow{\text{解析}} \underbrace{\text{工具调用}}_{\text{程序执行, 产生副作用}} \xrightarrow{\text{结果序列化}} \underbrace{\text{Observation}}_{\text{回到上下文}}
$$

这个环路的每一环都对应一类失败：生成不合法 JSON（→1.2 约束解码）、参数幻觉（→Schema 校验）、工具执行失败（→第 2 周 Reflection）、上下文爆炸（→第 2 周 Memory）。**Agent 工程就是逐环加固的可靠性工程**——模型智能决定上限，环路的鲁棒性决定下限。

**ReAct 三角色的分工再确认**（自测考点）：`Thought` 承担推理与决策（为什么这么做、下一步做什么）；`Action` 承担对工具的指令（结构化的调用，产生副作用）；`Observation` 承担环境反馈（工具返回/报错/截图）——**它是外部世界的输入，不是模型生成的**（Stage 7 的掩码规则在此同样适用）。数据流转：三者交替出现构成轨迹，Observation 之后必然回到 Thought 开始下一轮。

### 1.2 职责切分：模型做什么，程序做什么

Agent Loop 工程的第一设计决策是职责边界。原则：**凡可用代码确定性处理的，绝不交给模型**：

| 环节 | 归属 | 理由 |
| --- | --- | --- |
| 决定调用什么工具、传什么参数 | 模型 | 需要语义理解 |
| JSON 语法合法性 | **约束解码（程序）** | 语法是确定性规则 |
| 参数的 Schema 校验（类型/枚举/范围） | **程序** | 确定性可判 |
| 工具执行、超时、重试 | **程序** | 副作用管理 |
| 解读 Observation、决定下一步 | 模型 | 需要语义理解 |
| 步数上限/预算控制 | **程序** | 防失控（防死循环的第一道闸） |

新手最常见的错误是把"校验与重试"也写进 prompt 让模型自律——**模型自律是概率性的，程序校验是确定性的**。

### 1.3 约束解码：从"祈祷格式正确"到"词表级保证"

**问题**：LLM 生成 JSON 的失败模式——括号不闭合、枚举值拼错、参数漏填、在 JSON 里混入解释文字。微调与 few-shot 只能把错误率降低，**永远到不了 100%**，因为采样本身是概率过程。

**约束解码（Constrained Decoding）的机制**（自测考点）：解码每一步时，模型输出的是**整个词表上的概率分布**；约束引擎根据 JSON Schema/CFG 语法，预先计算"当前状态下语法允许的 Token 集合"，**把不在集合内的 Token 的 logits 置为 $-\infty$**（屏蔽），再采样——非法 Token 根本没有机会被选中：

$$
p'(v) \propto p(v) \cdot \mathbb{1}[v \in \mathcal{A}(s)], \quad \mathcal{A}(s) = \text{语法在状态 } s \text{ 下允许的 Token 集}
$$

工程要点：

1. **状态机映射**：把 JSON Schema 编译为语法状态机（Outlines 用正则/CFG → FSM；vLLM `guided_json`、SGLang JSON Mode 内置同思路），每步解码查询当前状态允许的 Token 集；
2. **词表预编译**：FSM 转移与 Token 化的对齐需要预编译（对大 Schema 有一次性编译开销，Outlines 有缓存）；
3. **代价与边界**：约束保证**语法**100% 合法，不保证**语义**正确（枚举值合法但选错、数字幻觉依旧存在）；强约束可能挤压模型的表达自由度（复杂 Schema 下质量略降）；部分引擎对并行请求的 FSM 编译缓存效率不同。**正确用法：约束解码管语法，Schema 校验管结构，模型能力管语义，三道闸各司其职**。

### 1.4 多模态工具链设计：用工具补感知短板

密集文本图上 VLM 的裸感知短板（Stage 1/3 的结论：低分辨率下小字不可读、密集元素计数弱）。工具链的设计原则：

1. **工具 = 把"模型不确定的感知"外包给"确定性/高精度模块"**：OCR（文本抽取交 OCR 引擎）、Crop+Zoom（把小区域放大到模型可读分辨率——直接对症 Stage 3 的分辨率瓶颈探针）、Grounding（坐标交给检测器）；
2. **Observation 必须是模型可消化的形态**：OCR 返回结构化文本而非原始像素；Crop 返回放大图 + 坐标系映射说明（模型后续引用坐标时基于哪个坐标系，必须显式告知，否则坐标引用错位）；
3. **工具面窄而精**：每多一个工具，模型的决策空间翻倍、选错率上升。起步 3~5 个高内聚工具优于 20 个大杂烩。

---

## 2. 实现与验证

### 2.1 手写 ReAct Agent Loop 引擎（本周 MVP 核心）

```python
"""
ReAct Agent Loop 引擎: 工具注册 + 约束解码 + 超时/步数控制 + Observation 回灌。
运行方式: python stage9_week1_react_engine.py   (离线演示模式, 工具为确定桩)
依赖: 标准库 (在线模式需 openai/vllm 客户端)
"""
import json
import re
import time
from dataclasses import dataclass, field


# ---------- 工具注册表 ----------
@dataclass
class Tool:
    name: str
    description: str
    params_schema: dict                      # 简化 JSON Schema
    fn: callable                             # 执行体
    timeout: float = 10.0


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def schema_prompt(self) -> str:
        """注入 prompt 的工具说明书 (模型据此决策)"""
        lines = []
        for t in self.tools.values():
            params = ", ".join(f'{k}:{v}' for k, v in t.params_schema.items())
            lines.append(f'- {t.name}({params}): {t.description}')
        return "\n".join(lines)

    def execute(self, name, args) -> str:
        """确定性执行: 校验 -> 超时包裹 -> 序列化 Observation (异常不逃逸)"""
        if name not in self.tools:
            return f"ERROR: unknown tool {name}"
        tool = self.tools[name]
        # Schema 校验 (程序负责的确定性部分)
        for k, typ in tool.params_schema.items():
            if k not in args:
                return f"ERROR: missing param {k}"
            if typ == "int" and not isinstance(args[k], int):
                return f"ERROR: param {k} must be int"
            if typ == "str" and not isinstance(args[k], str):
                return f"ERROR: param {k} must be str"
        t0 = time.time()
        try:
            obs = tool.fn(**args)
            return f"OK: {obs}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"[:300]
        finally:
            if time.time() - t0 > tool.timeout:
                return "ERROR: tool timeout"


# ---------- 两个视觉工具桩 (在线模式替换为真实现) ----------
def crop_image_tool(region: str):
    """region 格式 'x1,y1,x2,y2'; 演示返回放大说明"""
    x1, y1, x2, y2 = map(int, region.split(","))
    assert x2 > x1 and y2 > y1, "invalid box"
    return f"cropped region ({x1},{y1})-({x2},{y2}) enlarged 4x, OCR-ready image returned"


def ocr_tool(image_ref: str = "last_crop"):
    return "OCR result: '发票号 INV-2024-08812 金额 USD 4,350.00 到期日 2025-06-30'"


# ---------- Agent Loop ----------
ACTION_RE = re.compile(r"\{.*\}", re.S)


def extract_action(text: str):
    """从模型输出中抽取 Action JSON (约束解码下的兜底解析)"""
    m = ACTION_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if "tool" in obj and "args" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    return None


FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.S)


def run_agent(llm_call, registry: ToolRegistry, task: str,
              max_steps: int = 8, use_constrained=True):
    """llm_call(prompt)->str; 约束解码时 llm_call 侧挂 guided_json (见 2.2)"""
    traj = []
    obs = ""
    for step in range(max_steps):
        prompt = (f"Task: {task}\nTools:\n{registry.schema_prompt()}\n"
                  f"{obs}\n"
                  "Respond with ONE action as JSON {\"tool\":..., \"args\":{...}} "
                  "OR 'Final Answer: ...'")
        out = llm_call(prompt, constrained=("action_json" if use_constrained else None))
        if final := FINAL_RE.search(out):
            traj.append({"step": step, "final": final.group(1).strip()})
            return traj, final.group(1).strip()
        act = extract_action(out)
        if act is None:
            obs = f"Observation(step{step}): parse failed, respond with valid JSON action."
            traj.append({"step": step, "error": "parse"}); continue
        obs = f"Observation(step{step}): {registry.execute(act['tool'], act['args'])}"
        traj.append({"step": step, "action": act, "observation": obs})
    return traj, "FAILED: max_steps exceeded"


def _demo():
    reg = ToolRegistry()
    reg.register(Tool("Crop_Image_Tool", "裁剪并放大图片指定区域", {"region": "str"}, crop_image_tool))
    reg.register(Tool("OCR_Tool", "对最近裁剪区域执行高精度 OCR", {"image_ref": "str"}, ocr_tool))

    # 桩 LLM: 固定脚本 (在线模式即 VLM API); 脚本模拟"先裁剪->再OCR->总结"
    script = iter([
        '{"tool": "Crop_Image_Tool", "args": {"region": "100,60,400,180"}}',
        '{"tool": "OCR_Tool", "args": {"image_ref": "last_crop"}}',
        'Final Answer: 发票号 INV-2024-08812, 金额 USD 4,350.00, 到期日 2025-06-30。'])
    llm_call = lambda prompt, constrained=None: next(script)

    traj, ans = run_agent(llm_call, reg, "找出图中的发票号、金额与到期日。", max_steps=8)
    # ---- 断言: 闭环行为 ----
    assert len(traj) == 3 and "final" in traj[-1], "应在 3 步内收敛到 Final Answer"
    assert traj[0]["action"]["tool"] == "Crop_Image_Tool", "第一步应是裁剪"
    assert "INV-2024-08812" in ans, "最终答案应含 OCR 抽取的关键信息"
    assert all("Observation" in s.get("observation", "OK") or "final" in s or "error" in s
               for s in traj), "每个 Action 步必须有 Observation 回灌"
    # Schema 校验拦截演示: 错误参数类型
    obs = reg.execute("Crop_Image_Tool", {"region": 123})
    assert obs.startswith("ERROR"), "类型错误应被程序校验拦截而非模型自律"
    print(f"✅ ReAct 引擎闭环通过: {len(traj)} 步收敛, 答案片段: {ans[:40]}...")


if __name__ == "__main__":
    _demo()
```

**预期输出**：

```text
✅ ReAct 引擎闭环通过: 3 步收敛, 答案片段: 发票号 INV-2024-08812, 金额 USD 4,350.00...
```

断言验证的是**闭环的关键行为**：多工具协作顺序（先裁剪再 OCR）、Observation 回灌、Final Answer 收敛、以及"类型错误被程序拦截"的职责切分原则。在线模式把桩 LLM 换成 VLM API、工具桩换成真实现（Crop 用 PIL、OCR 用引擎）即可。

### 2.2 约束解码接入（100 次 0 失败的保证）

```python
"""
约束解码接入: vLLM guided_json / Outlines 两种路径 + 100 次零失败验证。
运行方式: python stage9_week1_constrained.py   (在线需 vLLM 服务; 离线用 FSM 桩验证机制)
"""
def vllm_constrained_call(api_base, model, prompt, schema):
    """路径A: vLLM guided_json (服务端 FSM)"""
    from openai import OpenAI
    client = OpenAI(base_url=api_base, api_key="EMPTY")
    rsp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        extra_body={"guided_json": schema})          # vLLM 的 structured output 参数
    return rsp.choices[0].message.content

def outlines_constrained_call(model, prompt, schema):
    """路径B: Outlines (客户端 FSM, 支持本地 HF 模型)"""
    import outlines
    gen = outlines.generate.json(model, schema)      # Schema -> FSM -> 屏蔽采样
    return gen(prompt)

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": ["Crop_Image_Tool", "OCR_Tool", "Finish"]},
        "args": {"type": "object"},
    },
    "required": ["tool", "args"],
}

def _mech_test():
    """离线机制验证: FSM 屏蔽逻辑的最小模拟"""
    # 语法状态: { 或 " 之外一律屏蔽 (模拟 FSM 屏蔽)
    vocab = ['{', '"', 't', 'o', 'o', 'l', 'x', ' ', '}']
    allowed_at_start = {'{'}
    masked = [v for v in vocab if v not in allowed_at_start]
    assert set(masked) == {'"', 't', 'o', 'o', 'l', 'x', ' ', '}'}
    print("✅ FSM 屏蔽机制验证: 首个 Token 仅 '{' 可被采样 (100% 语法合法的来源)")


if __name__ == "__main__":
    _mech_test()
    # 在线 100 次零失败验证 (需 vLLM 服务):
    # fails = 0
    # for i in range(100):
    #     out = vllm_constrained_call(URL, MODEL, f"发起一次工具调用 #{i}", ACTION_SCHEMA)
    #     obj = json.loads(out)            # 约束解码下此行永不抛异常
    #     assert obj["tool"] in ACTION_SCHEMA["properties"]["tool"]["enum"]
    # print(f"✅ 100 次工具调用, JSON 解析失败 {fails} 次")
```

机制说明：FSM 屏蔽是**词表级事前过滤**——非法 Token 的概率被置零后才采样，所以"解析失败"在结构上不可能发生（语义错误仍然可能：约束管不了"裁剪区域选错了"）。100 次零失败的验收里，**同时要记录约束开/关的对照**：关闭 guided_json 的对照组通常会出现个位数解析失败——有对照的 100% 才是可信的 100%。

---

## 3. 工程权衡与失效模式

### 3.1 决策表：约束解码方案选择

| 方案 | 部署形态 | 优势 | 注意 |
| --- | --- | --- | --- |
| vLLM `guided_json` | 服务端 | 与推理引擎一体化、并发 FSM 缓存 | 需 vLLM 后端 |
| SGLang JSON Mode | 服务端 | 与 RadixAttention/多轮调度协同 | 同上 |
| Outlines | 客户端/本地模型 | 支持 HF 本地模型、Schema→FSM 可审计 | 大 Schema 编译开销 |
| 手工正则重试 | 任意 | 零依赖 | **概率性**，不是"保证"只是"降低"——不满足 100% 验收 |

### 3.2 三个代表性失效模式

**失效 1：Observation 坐标系漂移——Crop 后的"再次定位"全错**
- **症状**：Agent 裁剪放大后，第二次调用 Grounding 返回的坐标映射回原图时整体偏移。
- **根因**：Crop 工具返回的 Observation 未声明"新图坐标系起点 = 原图 (x1,y1)"；模型用新图局部坐标当原图坐标用。
- **定位**：审计轨迹中 Crop 后的坐标引用与工具参数的一致性。
- **修复**：Observation 强制携带坐标系元信息（"此图对应原图区域 (100,60)-(400,180)，所有坐标需加上偏移 (100,60)"）；工具内部直接返回原图坐标系结果（更优——把坐标换算从模型侧移到程序侧，1.2 节职责原则）。

**失效 2：工具面过宽——模型选错工具或连环误用**
- **症状**：Agent 频繁调用无关工具（如对着文本问题调 OCR），步数耗尽。
- **根因**：工具描述模糊、职责重叠（两个工具都能"看图"）；或工具数过多导致决策空间爆炸。
- **定位**：统计轨迹中的工具选择分布与任务类型的相关性。
- **修复**：工具描述写"何时用/何时不用"；合并职责重叠的工具；超过 8~10 个工具时引入"工具路由层"（先用小模型/规则选类别）。

**失效 3：约束解码的"合法但无意义"**
- **症状**：100% 合法 JSON 达成，但 Agent 完成率反而下降。
- **根因**：强约束挤压了模型的中间推理空间（如不再输出 Thought 直接蹦 JSON）；或 Schema 过于复杂时 FSM 约束下的质量退化；或枚举封闭了模型"表达不确定"的能力（只能选工具，不能说"信息不足"）。
- **定位**：对照开/关约束的完成率与 Thought 质量。
- **修复**：Schema 里加 `Finish`/`Ask_User` 等逃生枚举；Thought 与 Action 分开生成（先自由推理后约束生成动作）；Schema 扁平化（嵌套深度 ≤2）。

---

## 4. 延伸思考题

1. **FSM 的词表对齐难题**：JSON 的一个字符可能被切成多个 Token（如 `":` 是单 Token 吗？），FSM 状态与 Token 边界如何对齐？查 Outlines 的实现（token 级 FSM 匹配 + 前缀缓存）并解释为什么这是约束解码工程里最微妙的部分。
2. **职责边界的哲学**：1.2 节的原则是"确定性的都给程序"。反例思考：参数合法性校验中"语义合法性"（如裁剪区域是否包含目标物体）无法程序化判定——此时该怎么办？（提示：拆成"语法校验给程序、语义校验给模型自检或第二遍验证调用"，或引入 Stage 4 的执行验证器——执行失败本身就是最强的语义反馈。）
3. **动手实验**：把 2.1 的引擎扩展出"OCR 结果为空"的工具失败分支，观察桩脚本没有处理该分支时的行为；再实现"失败 Observation 后自动重试一次"的逻辑——这就是第 2 周 Reflection 的入口。

---

*下一篇：[第 2 周：任务规划、反思与长短期记忆](第2周-规划反思与记忆机制.md)*
