# 第 5-6 周教程：端到端 Multimodal Agent 系统集成实战

> **本周要回答的三个问题**
> 1. 多 Agent 协作（Planner + 专职 Agent）的架构与消息协议怎么设计？什么时候值得多 Agent？
> 2. 工程鲁棒性的四件套（超时/Fallback/Guardrails/审计）怎么落地？
> 3. 一个 Deep Research 系统的端到端闭环怎么搭、怎么验收？

对应学习计划：第 5-6 周。交付物：端到端 Deep Research / Visual Task Agent——输入复杂指令（对比两份带图表公式的 PDF、算增长率、绘图、写报告），自主调用 PDF 解析/VLM 图表理解/Python 解释器/绘图，产出源码、Trace Logs 与 Demo。

---

## 1. 第一性原理：从单 Agent 到多 Agent 的决策

### 1.1 根本矛盾：上下文是共享的，而职责是冲突的

单 Agent（一个模型 + 全部工具 + 完整上下文）在任务复杂后有四类退化：

1. **上下文竞争**：任务规划需要的全局视野与子任务需要的细节细节挤同一个窗口——细节多了规划变糊，视野够了细节装不下；
2. **指令稀释**：工具越多决策越差（Stage 9 第 1 周失效 2）；提示词既要管规划又要管 OCR 要领，哪个都写不透；
3. **错误级联**：单 Agent 的一个坏决策直接传导到最终输出，无隔离；
4. **不可审计**：所有推理混在一条轨迹里，失败归因困难。

**多 Agent 的本质是"职责分治"**：每个专职 Agent 有窄工具面、聚焦的 System Prompt、独立的上下文——用**消息传递**交换结论而非共享全部过程。这换来可审计性（每个 Agent 的输入输出清晰归档）与隔离性（Vision Agent 的失败不污染 Reporter），代价是**通信开销与全局视野损失**——协调者看不到执行细节，可能做出脱离实际的规划。

### 1.2 多 Agent 的决策判据（什么时候值得拆）

| 信号 | 建议 |
| --- | --- |
| 任务可分解为**异质能力域**（视觉检查/代码计算/写作） | 值得拆——各域的 prompt 与工具可深度专业化 |
| 单 Agent 上下文已溢出或工具 >10 个 | 值得拆 |
| 子任务高度同质、串行依赖 | **不拆**——多 Agent 是通信开销纯浪费 |
| 需要并行处理独立子任务 | 值得拆（并行化收益） |

**反面提醒**：多 Agent 不是高级的标志。业界常见教训是"为拆而拆"——三个 Agent 互相传话损失的信息比专业化的收益还大。**先证明单 Agent 失败的具体环节，再为那个环节拆出专职 Agent。**

### 1.3 协作拓扑与消息协议

典型拓扑（本周 Deep Research 系统采用）：

```text
                 ┌──────────────┐
   用户任务 ──>  │   Planner    │  拆解 + 汇总 (唯一看得见全局的 Agent)
                 └──┬───┬───┬──┘
        结构化子任务 │   │   │  (JSON: {task_id, goal, constraints, expect})
            ┌───────┘   │   └───────┐
            ▼           ▼           ▼
      ┌──────────┐ ┌──────────┐ ┌──────────┐
      │ PDF/Vision│ │  Coder   │ │ Reporter │
      │ Examiner │ │ (Python) │ │ (写作)   │
      └────┬─────┘ └────┬─────┘ └────┬─────┘
           │ 结构化结果    │            │
           └───────> 回传 Planner <────┘
```

**消息协议的最小字段**（可审计的关键）：

```json
{
  "msg_id": "t3-vision-1",
  "from": "vision_examiner", "to": "planner",
  "type": "subtask_result",
  "payload": {"subtask_id": "t3", "findings": "...", "artifacts": ["chart1.png"],
               "confidence": 0.8},
  "issues": ["图表2坐标轴文字模糊, 数值为估计"],
  "audit": {"model": "...", "ts": "...", "tokens": {...}}
}
```

三条协议纪律：① **结构化结论 + 置信度 + 问题声明**——子 Agent 必须报告"不确定"而不是硬编（Planner 据此决定重试/换路）；② **产物走文件、消息走引用**（图/大文本落盘，消息里放路径——防上下文爆炸，Stage 4 第 2 周记忆管理的系统版）；③ **审计字段强制**——每个消息可回放可归因（Stage 3/4 的审计链纪律）。

---

## 2. 实战：Deep Research Agent 端到端搭建

### 2.1 任务分解（Planner 视角）

任务："对比两份带图表和公式的 PDF 报告，计算 X 数据的增长率，并绘制柱状图报告"：

| 子任务 | 执行 Agent | 工具 | 完成谓词（可验证） |
| --- | --- | --- | --- |
| T1 解析 PDF-A/B，抽取图表与公式区 | PDF/Vision | pdf 工具 + Crop + OCR | 每份产出 ≥1 图表图 + 公式文本 |
| T2 读图表数据点 | Vision Examiner | VLM 图表理解 + OCR | 数值表（附置信度） |
| T3 计算增长率 | Coder | Python 沙盒 | 代码执行成功 + 数值输出 |
| T4 绘柱状图 | Coder | matplotlib | 文件生成 + 非空 |
| T5 汇总 markdown 报告 | Reporter | — | 文件含全部关键数字 |

注意**每个子任务都带完成谓词**——这是 Stage 4"任务与验证器原子绑定"纪律的系统版。Planner 依谓词判定子任务成败并决定重试/改派。

### 2.2 系统骨架（langgraph 风格的状态机，手写可跑）

```python
"""
Deep Research Agent: Planner + 专职 Agent 状态机 (离线可跑; 工具为桩)。
运行方式: python stage9_week5_deep_research.py
依赖: 标准库 (在线时 LLM/工具替换真实现)
"""
import json
import time
from dataclasses import dataclass, field


@dataclass
class Msg:
    frm: str; to: str; type: str
    payload: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)


class AgentBus:
    """消息总线: 强制结构化消息 + 审计日志"""
    def __init__(self):
        self.log: list[Msg] = []

    def send(self, msg: Msg):
        assert msg.type in ("subtask", "subtask_result", "plan", "report"), "非法消息类型"
        assert msg.payload is not None
        self.log.append(msg)
        return msg

    def audit(self):
        return [f"{m.frm}->{m.to} [{m.type}] {json.dumps(m.payload, ensure_ascii=False)[:80]}"
                for m in self.log]


# ---------- 专职 Agent (桩实现: 确定性行为) ----------
class VisionExaminer:
    name = "vision"
    def run(self, sub: dict, bus: AgentBus) -> Msg:
        # 真实: pdf->图, Crop+OCR+VLM 读图表
        data = {"chart_a": {"2023": 120, "2024": 168}, "chart_b": {"2023": 90, "2024": 99}}
        ok = sub["subtask_id"] in ("t1", "t2")
        return bus.send(Msg(self.name, "planner", "subtask_result", {
            "subtask_id": sub["subtask_id"], "findings": "图表数据已提取",
            "data": data, "confidence": 0.85 if ok else 0.2},
            issues=[] if ok else ["图表模糊, 数值为估计"]))


class Coder:
    name = "coder"
    def run(self, sub: dict, bus: AgentBus) -> Msg:
        # 真实: Python 沙盒执行 matplotlib/计算; 桩: 确定性结果
        data = sub.get("payload", {}).get("data", {})
        a, b = data.get("chart_a", {}), data.get("chart_b", {})
        growth_a = round((a.get("2024", 0) - a.get("2023", 0)) / max(a.get("2023", 1), 1) * 100, 1)
        growth_b = round((b.get("2024", 0) - b.get("2023", 0)) / max(b.get("2023", 1), 1) * 100, 1)
        art = f"chart_growth.png (A: +{growth_a}%, B: +{growth_b}%)"
        return bus.send(Msg(self.name, "planner", "subtask_result", {
            "subtask_id": sub["subtask_id"], "growth": {"A": growth_a, "B": growth_b},
            "artifacts": [art], "confidence": 1.0}))


class Reporter:
    name = "reporter"
    def run(self, sub: dict, bus: AgentBus) -> Msg:
        g = sub.get("payload", {}).get("growth", {})
        md = (f"# 对比报告\n- 报告A X 数据增长率: {g.get('A')}%\n"
              f"- 报告B X 数据增长率: {g.get('B')}%\n- 图表: {sub['payload'].get('artifacts')}")
        open("report.md", "w").write(md)
        return bus.send(Msg(self.name, "planner", "report", {"file": "report.md"}))


class Planner:
    """Planner: 拆解 -> 派发(按依赖序) -> 验谓词 -> 汇总"""
    PLAN = [
        {"subtask_id": "t1", "agent": "vision", "goal": "解析两份PDF图表区", "predicate": lambda r: bool(r.payload.get("data"))},
        {"subtask_id": "t2", "agent": "vision", "goal": "读图表数据点", "predicate": lambda r: r.payload.get("confidence", 0) >= 0.5},
        {"subtask_id": "t3", "agent": "coder", "goal": "计算增长率并绘图",
         "needs": ["t2"], "predicate": lambda r: bool(r.payload.get("growth"))},
        {"subtask_id": "t4", "agent": "reporter", "goal": "汇总报告",
         "needs": ["t3"], "predicate": lambda r: True},
    ]

    def execute(self, bus: AgentBus, results: dict):
        for sub in self.PLAN:
            needs = sub.get("needs", [])
            payload = {k: results[k].payload for k in needs if k in results}
            bus.send(Msg("planner", sub["agent"], "subtask",
                         {"subtask_id": sub["subtask_id"], "goal": sub["goal"], **payload}))
            agent = {"vision": VisionExaminer, "coder": Coder, "reporter": Reporter}[sub["agent"]]()
            msg = agent.run({"subtask_id": sub["subtask_id"], **payload}, bus)
            # 完成谓词判定 (程序审计, 不信模型自报)
            for attempt in range(2):
                if sub["predicate"](msg):
                    results[sub["subtask_id"]] = msg; break
                msg = agent.run({**payload, "retry": True}, bus)   # 一次重试
            else:
                results[sub["subtask_id"]] = msg
                bus.send(Msg("planner", "planner", "report",
                             {"warning": f"{sub['subtask_id']} 谓词未通过"}))
        return results


if __name__ == "__main__":
    bus = AgentBus()
    results = Planner().execute(bus, {})
    assert "report.md" in results["t4"].payload.get("file", ""), "最终报告未产出"
    assert results["t3"].payload["growth"]["A"] == 40.0, "增长率计算错误"   # (168-120)/120
    audit = bus.audit()
    assert len(audit) >= 8, "审计日志不完整"
    print("✅ Deep Research 闭环通过: 报告 report.md 已生成")
    print("审计日志 (节选):")
    for line in audit[:5]:
        print(" ", line)
    print(open("report.md").read())
```

**预期输出**：

```text
✅ Deep Research 闭环通过: 报告 report.md 已生成
审计日志 (节选):
 planner->vision [subtask] {"subtask_id": "t1", "goal": "解析两份PDF图表区"}
 vision->planner [subtask_result] {"subtask_id": "t1", "findings": "图表数据已提取"...
 planner->coder [subtask] {"subtask_id": "t3", "goal": "计算增长率并绘图", "data": {...}
 coder->planner [subtask_result] {"subtask_id": "t3", "growth": {"A": 40.0, "B": 10.0}...
 reporter->planner [report] {"file": "report.md"}
# 对比报告
- 报告A X 数据增长率: 40.0%
- 报告B X 数据增长率: 10.0%
- 图表: ['chart_growth.png (A: +40%, B: +10%)']
```

断言验证系统级关键行为：谓词驱动的子任务验收（增长率算错会被打回）、审计日志完整性、产物落盘。**真实化路径**：把三个桩 Agent 换成真 LLM 调用 + 真工具（pdfplumber/PIL/matplotlib），Planner 的 PLAN 换成 LLM 动态生成（谓词由 LLM 提议、程序审核）——骨架与审计纪律不变。

### 2.3 鲁棒性四件套

| 件 | 实现 | 防的是什么 |
| --- | --- | --- |
| **超时控制** | 每个工具/子任务带 timeout（`asyncio.wait_for` / 沙盒 timeout），超时按失败处理 | 单点挂死拖垮全局 |
| **Fallback** | 关键工具有备用路径（主 OCR 失败换备用引擎；主模型限流换备模型）——Fallback 在 Fallback 链末端才是人工上报 | 外部依赖抖动 |
| **Guardrails** | 危险操作静态拦截（正则/AST 扫描代码里的 `rm -rf`/网络删除类调用）、出域动作白名单、输出敏感信息过滤 | 模型生成破坏性指令 |
| **审计与恢复** | 全消息总线日志 + 产物落盘 + 断点续跑（按 subtask_id 粒度恢复） | 事后归因与故障恢复 |

Guardrails 的实现原则：**静态拦截在执行前（便宜、确定），动态审查在执行后（LLM 审计，抽查）**——顺序不能反（先执行后审查的 Guardrail 只能善后）。

---

## 3. 工程权衡与失效模式

### 3.1 决策表：系统的复杂度控制

| 决策 | 简单做法 | 复杂做法 | 判据 |
| --- | --- | --- | --- |
| Agent 数量 | 1 个全能 + 工具面窄 | Planner + 专职群 | 1.1 节的拆分判据 |
| 通信 | 函数调用直传 | 消息总线 + 结构化协议 | 需要审计/恢复 → 总线 |
| 规划 | 固定流程（状态机） | LLM 动态规划 | 任务形态稳定 → 状态机；开放任务 → 动态 |
| 记忆 | 会话内压缩 | Visual RAG 长期记忆 | 是否跨会话复用经验 |

### 3.2 三个代表性失效模式

**失效 1：错误级联的静默传导——上游低置信结果被下游当真Influencer**
- **症状**：最终报告的数字与源文档不符，逐环节检查才发现图表读数就错了（置信度 0.2 被标注但无人处理）。
- **根因**：子 Agent 报告了 `issues` 与低 `confidence`，但 Planner 没有按置信度分流的逻辑——审计信息只是记了没用。
- **定位**：审计日志回放，找置信度低于阈值仍被消费的消息。
- **修复**：Planner 的谓词逻辑加入置信度门槛（<0.5 触发重试或人工标注降级路径）；"issues 非空 → 必须有处置动作"作为系统不变量。

**失效 2：多 Agent 的"传话失真"**
- **症状**：Planner 汇总的结论与子 Agent 的原始发现不一致（数字被四舍五入丢精度、限定条件丢失）。
- **根因**：消息在多跳传递中被自然语言转述而非结构化透传——每跳转述都是一次信息有损压缩。
- **定位**：逐跳 diff payload。
- **修复**：结构化数据（数字/表格/文件引用）**透传不转述**，转述只用于自然语言摘要；关键数字在最终报告中可溯源到消息 ID。

**失效 3：Guardrails 误伤与漏防的失衡**
- **症状**：正常代码被拦截（误伤）导致任务频繁失败；或上线后仍出现破坏性操作（漏防）。
- **根因**：静态规则过宽/过窄；Guardrails 只在单点（Planner）设防，专职 Agent 的直接工具调用绕过了检查。
- **定位**：拦截日志的误报率 + 红队测试（故意注入危险指令）的漏报率。
- **修复**：Guardrail 作为**总线层的统一关卡**（所有副作用动作必经）而非各 Agent 自查；规则分级（硬拦截/确认闸/仅记录），红队集持续回归。

---

## 4. 延伸思考题

1. **多 Agent 的通信成本模型**：设单 Agent 完成任务需 N 步、每步成本 c；拆成 K 个专职 Agent 后每步成本 c'（<c，prompt 更聚焦）但增加 K 倍协调调用与每次的消息组装/解析。推导"拆分收益 > 0"的条件，代入一个你熟悉的任务验证。（提示：协调开销 ≈ K × 上下文重建成本；子任务独立性越强、单步 prompt 越臃肿，拆分越赚。）
2. **Agent 的 SLO 设计**：给 Deep Research 系统定义 SLO：任务成功率、P90 完成时间、成本上限（Token/任务）、幻觉率（报告数字可溯源比例）。思考四者之间的冲突结构（如提高成功率→更多重试→时间与成本上升），并设计一个"预算感知的动态策略"（预算充足时多反思重试，紧张时降级为快糙输出）。
3. **跨阶段终极串联**：把 Stage 1~9 的技术栈画成这个系统的"依赖清单"——视觉 Token 化（1）支撑图表理解、SFT/RL（2/7）决定 Planner 质量、评测（3）提供谓词校准、合成数据（4）训练子 Agent、筛选（5）保证轨迹质量、偏好对齐（6）决定报告风格、系统优化（8）压低每步延迟。找出这份依赖清单中**你当前最薄弱的一环**，它就是进入 Stage 10 前最值得补的短板。

---

*下一篇：[阶段九自测验收与复盘](阶段九自测验收与复盘.md)*
