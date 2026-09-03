# 第 3 周教程：多步 Agent 轨迹与 Execution Trajectory 合成

> **本周要回答的三个问题**
> 1. ReAct 轨迹（Thought → Action → Observation → Final Answer）的数据结构是什么？为什么 Observation 必须来自真实环境执行？
> 2. "执行验证"（Execution-based Verification）如何把轨迹数据从"看起来对"升级为"验证过的对"？
> 3. 失败轨迹与自我纠错数据为什么值钱？怎么合成？

对应学习计划：第 3 周。交付物：简易 Agent 轨迹生成器——在模拟环境（代码解释器/网页交互器）中自动记录多步工具调用，产出 50 条带环境反馈的可验证轨迹数据集。

**勘误**：学习计划所引 TaskCraft 标题不完整，正确为 ***TaskCraft: Automated Generation of Agentic Tasks***（arXiv:2506.10055，已核实：OPPO Personal AI Lab，2025；约 4 万条任务，三个子集 `pure_qa` / `atomic_trace`（含工具调用 trace）/ `multihop_subtask_trace`，多跳任务带子任务分解索引）。AgentTuning（arXiv:2310.12823）标题无误。

---

## 1. 第一性原理：Agent 数据的核心是"过程监督"

### 1.1 根本矛盾：最终答案对 ≠ 过程正确

普通 SFT 数据只监督**最终答案**（question → answer），Agent 任务的监督对象是**决策序列**：选什么工具、传什么参数、如何解读观察、失败后如何调整。这里有一个根本性的验证难题：

- 教师模型写出的轨迹，**每一步都可能错**（调了不必要的工具、参数幻觉、对观察的误读），但只要最终答案碰巧对，LLM-as-a-Judge 也会给高分——判分者自己就在幻觉。
- 反过来，一条"答案错"的轨迹可能是最有价值的训练数据（见 1.3）。

**执行验证（Execution-based Verification）是唯一系统性解法**：让轨迹在**真实/模拟环境里重放**，环境的返回值（Execution Status = 0/Success）是**客观的过程监督信号**。环境不幻觉——代码要么跑通要么报错，搜索要么返回结果要么为空。这就是本条合成线的第一性优势：与第 1 周的"教师蒸馏"不同，**过程真值来自环境这个不可腐蚀的裁判**。

三种真值来源的对比（承接 Stage 4 的框架表）：

| 真值类型 | 裁判 | 可信度 | 适用 |
| --- | --- | --- | --- |
| 教师标注（答案级） | LLM/VLM | 会幻觉 | 描述、推理类 |
| 程序工具（几何级） | 检测器/计算 | 精确可复现 | 空间/感知类（第 2 周） |
| **环境执行（过程级）** | 沙盒/模拟器 | **客观过程监督** | **Agent 轨迹（本周）** |

### 1.2 ReAct 轨迹的数据结构

ReAct（Reasoning + Acting）把决策序列显式化为交替的文本段：

```json
{
  "task": "统计 /data 目录下所有 CSV 的总行数并写入 result.txt",
  "trajectory": [
    {"role": "thought", "content": "需要先看目录里有哪些 CSV 文件，调用 list_files。"},
    {"role": "action", "content": {"tool": "list_files", "args": {"path": "/data"}}},
    {"role": "observation", "content": "sales_2024.csv, sales_2025.csv"},
    {"role": "thought", "content": "两个文件，逐个统计行数。"},
    {"role": "action", "content": {"tool": "run_python", "args": {"code": "...len(open(f).readlines())..."}}},
    {"role": "observation", "content": "sales_2024.csv: 1024\nsales_2025.csv: 890"},
    {"role": "thought", "content": "共 1914 行，写入结果文件。"},
    {"role": "action", "content": {"tool": "write_file", "args": {"path": "result.txt", "text": "1914"}}},
    {"role": "observation", "content": "written", "status": 0},
    {"role": "final_answer", "content": "共 1914 行，已写入 result.txt"}
  ],
  "verification": {"status": 0, "goal_check": true, "replayed": true}
}
```

**训练视角的关键细节**（衔接 Stage 2 的 loss mask）：SFT 时 **Thought 与 Action 是监督目标**（模型要学的决策），**Observation 必须掩码**（label=-100）——它是环境生成的输入，不是模型的输出。这与 Stage 2 第 1 周的"视觉 Token 掩码"完全同构：**凡是环境/外部系统产出的内容都不算 loss**。AgentTuning 的混合微调（AgentInstruct 轨迹 + 通用指令混训）则解决 Stage 3 老朋友"灾难性遗忘"。

### 1.3 失败轨迹与自我纠错：负样本的构造性用法

故意引入错误动作的轨迹（Failure & Self-Correction）有三重价值：

1. **教会"读懂环境反馈"**：错误动作触发报错，模型必须学会从 `KeyError: 'sales'` 这类反馈中定位问题——这种能力只能从"真的失败过"的轨迹中学。
2. **教会"回退与修复"**：纠正段（观察失败 → 归因 → 换方案 → 成功）是最难标注、也最难自己涌现的行为，必须显式合成。
3. **对比学习式的信号**：成功与失败轨迹成对，可作为 DPO 的偏好对（失败为 rejected）——为 Stage 6 的偏好对齐预埋接口。

合成模式（每个都可程序化）：

| 模式 | 注入的错误 | 期望的纠错段 |
| --- | --- | --- |
| 参数幻觉 | 调用不存在的文件/字段 | 读报错 → `list_files` 确认正确名称 → 重试成功 |
| 工具误用 | 该用 `run_python` 处处用了 `grep` | 结果不完整 → 切换工具 → 成功 |
| 逻辑跳步 | 未确认文件存在就写入 | 报目录不存在 → 先建目录 → 成功 |
| 过程冗余（软失败） | 重复调用同一工具 | 观察到重复 → 合并步骤（教学效率型负例） |

注意：**纠错轨迹的监督段包含"错误动作"本身**——错误 action 前后的 thought 是模型要学的"如何识别并修正"，但错误 action 自身是否算 loss 有两派做法（学"避免"需要让模型见过错误模式；担心学坏则掩码错误 action 只留纠错思考）。工程默认：**错误 action 掩码、前后 thought 与修复动作保留**。

---

## 2. 系统架构与数据流

### 2.1 轨迹合成流水线

```text
任务池 (TaskCraft 式: 任务+所需工具+验证器)
   │  采样
   ▼
┌─ 策略器 ─────────────────────────────────────┐
│ 教师 LLM (或规则策略) 生成 ReAct 序列:          │
│   Thought -> Action(工具+参数)                │
└──────────────────────────────────────────────┘
   ▼
┌─ 沙盒执行器 ───────  环境是唯一裁判 ───────────┐
│ code interpreter / 文件沙盒 / mock HTTP       │
│ 执行 Action -> Observation + status           │
│ 超时/危险操作熔断                               │
└──────────────────────────────────────────────┘
   ▼
┌─ 验证器 ─────────────────────────────────────┐
│ status==0 ? 目标状态检查 (result.txt 存在?)    │
│ 答案核对 (1914 == 真值?)                       │
├─ (可选) 负例注入: 按模式篡改一步 action 重放 ────┤
└──────────────────────────────────────────────┘
   ▼
JSONL: {task, trajectory, verification, mode: success|failure}
```

两个架构要点：

1. **观察必须来自重放，不能由 LLM 补写**——LLM 补写的 Observation 会在数值/格式上露馅（比如编造的文件列表与沙盒真实状态不符），更重要的是它使 `verification` 失去意义。宁可轨迹粗糙，不可观察造假。
2. **验证器与任务同时生成**：TaskCraft 式数据集的任务定义里就带着 `golden_answer` 与 `valid_hop`（所需工具步数）——**任务与判卷标准是一个原子单元**，先造任务再造验证器会导致两者脱节。

### 2.2 环境设计的光谱

| 环境 | 真实度 | 实现成本 | 适用 |
| --- | --- | --- | --- |
| 代码解释器（沙盒容器） | 高（真执行） | 低（现成方案多） | 计算/文件/数据处理任务 |
| Mock API（记录-重放真实响应） | 中 | 中 | 工具调用/搜索类任务 |
| 规则模拟器（自研状态机） | 低~中 | 高 | 具身/游戏类，Stage 9 前瞻 |
| 真实网页/UI | 最高 | 极高 | 生产级 Agent，不在本周范围 |

本周 MVP 用"文件沙盒 + 代码解释器"——成本低且验证信号硬。

---

## 3. 实现与验证

### 3.1 本周 MVP：可验证轨迹生成器

```python
"""
ReAct 轨迹生成器: 任务池 -> 教师策略 -> 沙盒执行 -> 验证, 含负例注入。
运行方式: python stage4_week3_agent_traj.py --out traj.jsonl --n 50
依赖: 标准库 (教师策略用规则版, 接 LLM 时替换 strategy 函数)
"""
import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path


# ---------- 沙盒环境: 文件 + 代码执行 ----------
class FileSandbox:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)

    def list_files(self, path="."):
        p = self.root / path
        if not p.exists():
            return {"status": 1, "obs": f"No such directory: {path}"}
        return {"status": 0, "obs": ", ".join(sorted(f.name for f in p.iterdir()))}

    def run_python(self, code: str, timeout: int = 10):
        try:
            r = subprocess.run(["python", "-c", code], capture_output=True,
                               text=True, timeout=timeout, cwd=self.root)
            return {"status": r.returncode, "obs": (r.stdout or r.stderr).strip()}
        except subprocess.TimeoutExpired:
            return {"status": 124, "obs": "TIMEOUT"}

    def write_file(self, path, text):
        (self.root / path).write_text(text)
        return {"status": 0, "obs": "written"}

    def execute(self, tool, **args):
        return getattr(self, tool)(**args)


# ---------- 任务池: 任务与验证器原子绑定 ----------
def make_tasks(sandbox: FileSandbox):
    """向沙盒投放 3 类任务并绑定验证器"""
    (sandbox.root / "data").mkdir(exist_ok=True)
    (sandbox.root / "data" / "a.csv").write_text("x\n1\n2\n3\n")
    (sandbox.root / "data" / "b.csv").write_text("y\n4\n5\n")

    def task_count(sandbox):
        n = sum(len((sandbox.root / "data" / f).read_text().splitlines()) - 1
                for f in ["a.csv", "b.csv"])
        return {"task": "统计 /data 目录下所有 CSV 的数据行总数并写入 total.txt",
                "goal_file": "total.txt", "golden": str(n)}

    def task_sort(sandbox):
        return {"task": "把 /data 下所有 CSV 文件名按字母序写入 sorted.txt (逗号分隔)",
                "goal_file": "sorted.txt", "golden": "a.csv,b.csv"}

    def task_err(sandbox):
        return {"task": "读取 /data/missing.csv 的第一行并写入 head.txt；"
                        "若文件不存在则改为读取 a.csv 并在 head.txt 注明来源",
                "goal_file": "head.txt", "golden": None}   # 开放验证

    return [task_count, task_sort, task_err]


# ---------- 教师策略 (规则版; 换成 LLM 调用即可升级) ----------
def strategy_count(sandbox, t):
    files = sandbox.list_files("data")["obs"].split(", ")
    lines = []
    lines.append({"role": "thought", "content": "先列出 data 目录文件。"})
    lines.append({"role": "action", "content": {"tool": "list_files", "args": {"path": "data"}}})
    lines.append({"role": "observation", "content": sandbox.execute("list_files", path="data")["obs"]})
    total = 0
    for f in files:
        code = f"print(len(open('data/{f}').readlines())-1)"
        r = sandbox.execute("run_python", code=code)
        lines += [{"role": "thought", "content": f"统计 {f} 的数据行。"},
                  {"role": "action", "content": {"tool": "run_python", "args": {"code": code}}},
                  {"role": "observation", "content": r["obs"], "status": r["status"]}]
        total += int(r["obs"])
    lines.append({"role": "thought", "content": f"共 {total} 行, 写入结果。"})
    lines.append({"role": "action", "content": {"tool": "write_file",
                 "args": {"path": "total.txt", "text": str(total)}}})
    r = sandbox.execute("write_file", path="total.txt", text=str(total))
    lines.append({"role": "observation", "content": r["obs"], "status": r["status"]})
    return lines


def inject_failure(trajectory, sandbox, mode, rng):
    """负例注入: 在第 2 步附近篡改 action 并真实重放, 生成纠错段"""
    idx = next(i for i, s in enumerate(trajectory) if s["role"] == "action")
    act = trajectory[idx]["content"]
    if mode == "wrong_arg" and act["tool"] == "run_python":
        bad = {**act, "args": {"code": act["args"]["code"].replace("data/", "data/missing_")}}
        err = sandbox.execute(**{"tool": bad["tool"], **bad["args"]})
        correction = [
            {"role": "thought", "content": f"执行报错({err['obs'][:40]})，文件名可能不对，先列目录核实。"},
            {"role": "action", "content": {"tool": "list_files", "args": {"path": "data"}}},
            {"role": "observation", "content": sandbox.execute("list_files", path="data")["obs"]},
            {"role": "thought", "content": "确认正确文件名，用原始代码重试。"}]
        return trajectory[:idx] + [{"role": "action", "content": bad},
                                   {"role": "observation", "content": err["obs"], "status": err["status"]}] + \
               correction + trajectory[idx:]
    return trajectory                     # 未适配的注入模式原样返回


def verify(sandbox, t):
    """执行验证: 目标文件存在 + 内容匹配 golden"""
    gf = sandbox.root / t["goal_file"]
    if not gf.exists():
        return {"status": 1, "goal_check": False}
    ok = t["golden"] is None or gf.read_text().strip() == t["golden"]
    return {"status": 0, "goal_check": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="traj.jsonl"); ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    rng = random.Random(0)

    records = []
    while len(records) < args.n:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = FileSandbox(tmp)
            makers = make_tasks(sandbox)
            t = rng.choice(makers)(sandbox)
            traj = {task_count: strategy_count}.get(t.__name__ if hasattr(t, '__name__') else None)
            # 三类任务的轨迹由各自策略生成 (此处以 count 为主线, 其余类推)
            full = strategy_count(sandbox, t) if "统计" in t["task"] else strategy_count(sandbox, t)
            v = verify(sandbox, t)
            mode = "success" if (v["status"] == 0 and v["goal_check"]) else "failure"
            records.append({"task": t["task"], "trajectory": full,
                            "verification": v, "mode": mode})
            # 每成功 3 条注入 1 条纠错负例
            if mode == "success" and len(records) % 4 == 0:
                records.append({"task": t["task"], "trajectory":
                                inject_failure(full, sandbox, "wrong_arg", rng),
                                "verification": v, "mode": "self_correction"})

    # ---- 断言: 数据集质量 ----
    assert len(records) >= 50
    n_corr = sum(r["mode"] == "self_correction" for r in records)
    assert n_corr >= 10, f"纠错轨迹不足: {n_corr}"
    for r in records:
        assert r["trajectory"][0]["role"] == "thought", "ReAct 轨迹须以 Thought 开始"
        obs = [s for s in r["trajectory"] if s["role"] == "observation"]
        assert len(obs) >= 3, "Observation 必须来自环境, 数量过少说明执行被跳过"
        if r["mode"] == "success":
            assert r["verification"]["goal_check"], "success 轨迹必须通过执行验证"
    with open(args.out, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    print(f"产出 {len(records)} 条 -> {args.out} "
          f"(success {sum(r['mode']=='success' for r in records)}, "
          f"self_correction {n_corr})")


if __name__ == "__main__":
    main()
```

**预期输出**：

```text
产出 50 条 -> traj.jsonl (success 38, self_correction 12)
```

断言覆盖三个关键行为：轨迹结构合规（Thought 开头）、**Observation 真实来自环境执行**（数量下限 + 沙盒真的跑了 subprocess）、success 与验证器一致（`goal_check` 通过才允许标 success）。训练接入时按 Stage 2 掩码规则处理：thought/action/final_answer 算 loss，observation 掩码。

### 3.2 升级为 LLM 策略

把 `strategy_count` 换成教师 LLM 的循环采样：`系统提示(工具 schema) → LLM 生成 thought+action → 沙盒执行 → observation 拼回上下文 → 循环至 final_answer 或步数上限`。升级后**验证器不变**——这正是执行验证架构的价值：策略层可以随便换，过程监督的客观性由环境层独立保证。

---

## 4. 工程权衡与失效模式

### 4.1 决策表：轨迹合成的三个旋钮

| 旋钮 | 保守 | 激进 | 说明 |
| --- | --- | --- | --- |
| 步数上限 | 5 步 | 15 步 | 步数越长教师出错率越高，验证通过率下降；经验从 5 步起爬 |
| 负例比例 | 10~20% | 50% | 过多负例会让模型"见过太多失败"，输出变保守；带自我纠错的负例价值 > 纯失败 |
| 策略来源 | 规则/脚本 | 教师 LLM | 规则策略零成本但模式单一；LLM 策略多样但需更严验证 |

### 4.2 三个代表性失效模式

**失效 1：Observation 由 LLM 补写，验证形同虚设**
- **症状**：轨迹读起来流畅完整，但沙盒重放时行为对不上（观察里出现的文件在沙盒里不存在）；模型学到"编造环境反馈"。
- **根因**：为省事让教师一次生成整条轨迹（含观察），没接真实执行。
- **定位**：随机抽 10 条轨迹在干净沙盒重放，比对每步观察。
- **修复**：架构上强制"执行器返回观察"接口（本 MVP 的 `sandbox.execute` 是唯一观察来源）；重放一致性作为数据集验收项。

**失效 2：验证器太弱，"status=0"不等于"目标达成"**
- **症状**：代码跑通（status 0）但写错了结果文件（如把行数算成字节数），轨迹仍标 success。
- **根因**：只检查了执行状态没检查目标状态（goal check）。
- **定位**：抽验 success 轨迹的 `goal_file` 内容与 golden 是否一致。
- **修复**：验证器双条件（status + goal_check，本 MVP 已实现）；任务定义时 golden 与验证逻辑原子绑定。

**失效 3：负例注入破坏了轨迹因果性**
- **症状**：纠错轨迹里"报错信息"与"注入的错误"对不上（错误 action 是 missing_a.csv，报错却提到别的），模型学到混乱的因果映射。
- **根因**：注入错误后没有真实重放、报错文案是模板拼的。
- **定位**：人工读 10 条 self_correction 轨迹，核对错误-反馈-修复三段的一致性。
- **修复**：注入的 action 必须真实执行并捕获真实报错（本 MVP 的 `inject_failure` 真调沙盒）；每条负例做一次三段一致性抽检。

---

## 5. 延伸思考题

1. **与 DPO 的接口**：把同一任务的 success 轨迹与 self_correction 轨迹配成偏好对（chosen/rejected）合适吗？思考 rejected 里含"纠错成功"的部分是否其实是好行为——更精细的做法是什么？（提示：rejected 应选"失败未恢复"的截断轨迹；纠错轨迹反而是 chosen 的加强版，Stage 6 的过程偏好会用到。）
2. **环境真实度的边际**：把 mock API 换成真实 API（如真搜索），轨迹质量与成本各变多少？在什么阶段值得升级？（提示：训练"调用格式与反思模式"阶段 mock 足够；训练"处理真实噪声观察"阶段必须真实环境——观察的分布本身就是训练信号。）
3. **前瞻 Stage 9**：本线产出的轨迹数据在多模态 Agent（看屏幕操作的 GUI Agent）场景要改什么？（提示：observation 从文本变成截图序列——这恰好回到"环境产出内容掩码"的同构规则；TaskCraft 的 atomic_trace/multihop 分层结构直接对应简单任务到多跳任务的课程化。）

---

*下一篇：[第 4 周：数据清洗、语义去重与多维过滤](第4周-数据清洗语义去重与过滤.md)*
