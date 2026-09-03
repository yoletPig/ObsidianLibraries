# 第 2 周教程：自动化评测框架 VLMEvalKit 实战

> **本周要回答的三个问题**
> 1. VLMEvalKit 的架构是怎么分层的？为什么接入新模型只需实现一个 `generate_inner()`？
> 2. 如何用 vLLM/LMDeploy 后端把评测从几小时压到几分钟？分布式评测怎么切分任务？
> 3. 推理结果的缓存与断点续评机制是什么？评测中断后如何不浪费已花掉的 GPU 时？

对应学习计划：第 2 周。交付物：VLMEvalKit 跑通 Qwen2.5-VL-7B 与 LLaVA-1.5-7B 在 MME、POPE 上的全自动评测，生成对比柱状图并撰写差异笔记。

本篇属于**系统与基础设施**主题（评测框架本质是一个批处理推理系统），分析重点是吞吐、并行切分与 I/O 复用。

---

## 1. 第一性原理：评测系统的根本矛盾

### 1.1 根本矛盾：任务并行度 vs GPU 资源形态

评测的工作负载特征与训练截然不同：

- **训练**：固定模型 + 固定形状的 batch，长时间满负荷计算；
- **评测**：固定模型 + **数千条异构短请求**（MME 两千多条、POPE 九千条、MMMU 近万条），每条 = 一次前向 + 短生成。

矛盾在于：单条请求的 GPU 利用率极低（prefill 短、decode 步数少、大量时间在 CPU 数据准备与调度上），而评测任务之间**天然无依赖**（样本 $i$ 与样本 $j$ 独立），并行度理论上是无限的。评测系统的全部设计——推理后端、多卡切分、缓存——都在回答同一个问题：**如何把无限的任务并行度塞进有限的 GPU 资源里，同时保证可复现与可恢复**。

### 1.2 VLMEvalKit 的架构分层

（接口与机制已对照 `open-compass/VLMEvalKit` main 分支核实，2026 年持续维护，220+ 模型、80+ 基准。）

```text
run.py (CLI 入口: --model --data --verbose --work-dir --nproc ...)
   │
   ▼
vlmeval/config.py          # 模型注册表: 模型名 -> 类与构造参数 (含 use_vllm 等加速开关)
   │
   ▼
┌─────────────── 推理层 ───────────────────────┐
│ vlmeval/infer/                               │
│   InferenceExecutor (多进程/多卡任务切分)      │
│   每个进程: sample -> model.generate_inner() │
└──────────────────────────────────────────────┘
   │  预测结果 (TSV/XLSX) 落盘 -> outputs/{model}/{dataset}/
   ▼
┌─────────────── 评测层 ───────────────────────┐
│ vlmeval/dataset/  各基准的 Dataset 类         │
│   .evaluate() : 读取预测文件 -> 打分           │
│   MCQ: 正则/LLM 提取选项 (第1周的三级提取)     │
│   开放题: judge 模型评分 (GPT-4/Qwen-Max...)  │
└──────────────────────────────────────────────┘
   │  分数 (acc CSV) + 评分日志 (judge 对话记录)
   ▼
outputs/.../{dataset}_acc.csv  |  {dataset}_eval.judge.log
```

三个关键设计决策：

1. **推理与评测解耦，以文件为界**：预测落盘后才进入打分阶段。好处是打分可单独重跑（换裁判/换提取逻辑不必重新推理）、预测文件本身就是可审计的中间产物。
2. **统一 generation-based 评测**：官方明确说明对**所有**模型用生成式评测（而非 SEED-Bench 原版的 ppl 打分），以保证跨模型可比性——代价是与部分基准原论文的数字不完全一致（官方 README 也声明了这一点）。读别人报告的分数时，注意区分"原论文协议"与"VLMEvalKit 协议"两套数字。
3. **新模型接入面收窄为 `generate_inner()`**：数据下载、prompt 构造、多进程调度、预测落盘、指标计算全部由框架承担。这就是"1 小时接入自定义模型"（自测考点）的架构基础。

### 1.3 推理后端：为什么 vLLM 能快一个量级

`transformers` 原生 `generate()` 是**朴素逐请求调度**：一个 batch 内所有序列同步走完 decode，短序列被长序列拖住（无 continuous batching）。vLLM 的核心是：

- **PagedAttention**：把 KV Cache 按"页"管理（类比操作系统虚拟内存），显存碎片从 60~80% 降到 <4%，同显存可容纳的并发序列数成倍增加；
- **Continuous batching**：某序列生成结束立即退出 batch、新请求立即加入，GPU 永远满载；
- 对评测场景的放大效应：评测请求普遍短且数量大，正是 continuous batching 的最优区间。

**实测量级参考**（经验区间，具体取决于模型大小、序列长度与卡型）：7B 模型在 A100 上，transformers 后端跑 MME 约需 20~60 分钟，vLLM 后端可压缩到 5~10 分钟量级——3~8 倍加速。**务必用你自己的环境实测并记录**，不要引用他人的倍数。

VLMEvalKit 的接入方式（已核实）：在 `vlmeval/config.py` 的自定义模型配置中添加 `use_vllm=True`（或 `use_lmdeploy=True`，后者对 InternVL/Qwen 系支持良好，还支持多节点分布式推理）。注意约束：vLLM 后端仅对适配的模型架构生效（Qwen-VL 系、LLaMA4 等已适配；某些依赖 HF 复杂前向逻辑的模型不可用）。

### 1.4 多卡切分与缓存

- **任务切分**：VLMEvalKit 的 `--nproc N` 启动 N 个工作进程，按样本 index 把数据集切成 N 份独立推理，最后合并预测文件。**每个进程独立加载一份模型权重**——因此多进程评测吃的是"多份数据并行"而非"单模型张量并行"（除走 vLLM/LMDeploy 的分布式路径外）。8 卡跑评测的显存规划要按"8 份模型副本"算。
- **缓存与断点续评**：预测文件以 TSV 逐条追加（或按 chunk 落盘），重跑时框架检测到已有预测的样本（按 index 对齐）会**跳过推理直接复用**。这带来两个工程红利：
  1. **断点续评**：OOM/网络中断后直接重跑同一条命令，已完成的样本不重算；
  2. **裁判复用**：打分阶段读的是同一份预测文件，换裁判模型只重跑评测层。
- **缓存失效的正确姿势**：改了 prompt 模板或模型权重后必须清掉对应 work-dir（或换 `--work-dir`），否则旧预测被静默复用——这是评测工程里最阴险的错误来源（详见失效模式 3）。

---

## 2. 实现与验证

### 2.1 环境：两个后端的部署

```bash
# 框架本体 (Python 3.10+)
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit && pip install -e .

# 可选: vLLM 加速后端 (版本需与 transformers 匹配, 首次先跑通再加)
pip install vllm

# API 裁判 (开放式题/MCQ 兜底提取需要)
export OPENAI_API_KEY=sk-xxx            # 或其他兼容 OpenAI 协议的裁判服务
export OPENAI_API_BASE=https://xxx/v1
```

版本对齐警告（VLMEvalKit 官方维护着一张 transformers 版本推荐表）：不同 VLM 家族对 `transformers` 版本敏感（如 Qwen2.5-VL 需要较新版本，LLaVA-1.5 系对 4.37 依赖较强）。**两模型同评测时若版本冲突，标准做法是分两个虚拟环境各跑各的**，共享预测目录结构。

### 2.2 跑通四个评测任务

```bash
# Qwen2.5-VL-7B: MME + POPE (vLLM 加速)
python run.py \
  --model qwen2.5_vl_7b \
  --data MME POPE \
  --work-dir outputs/qwen25vl7b \
  --nproc 2 --verbose

# LLaVA-1.5-7B: 同基准 (注意其 transformers 版本要求, 建议独立环境)
python run.py \
  --model llava_v1.5_7b \
  --data MME POPE \
  --work-dir outputs/llava15_7b \
  --nproc 2 --verbose
```

产物结构（断点续评就绪）：

```text
outputs/qwen25vl7b/
├── MME/
│   ├── MME.xlsx 或 .tsv      # 预测文件 (可审计)
│   ├── MME_acc.csv           # 各子任务分数
│   └── MME_eval.judge.log    # 裁判对话记录 (若走 judge)
└── POPE/
    ├── POPE.xlsx
    └── POPE_acc.csv
```

### 2.3 结果对比可视化（本周交付的柱状图）

```python
"""
对比两模型在 MME 子任务与 POPE 上的得分, 输出柱状图。
运行方式: python stage3_week2_compare.py
依赖: matplotlib, pandas
"""
import pandas as pd
import matplotlib.pyplot as plt

# MME_acc.csv 为 长表: columns = [category, split, score]
# POPE_acc.csv 为 宽表: columns = [split, f1, accuracy, precision, recall, yes_ratio]
MME_KEY_TASKS = ["existence", "count", "position", "color",  # 感知-细粒度
                 "code_reasoning", "numeric_calculation",     # 认知-推理
                 "text_translation", "posters"]


def load_scores(qwen_dir: str, llava_dir: str) -> pd.DataFrame:
    rows = []
    for name, d in [("Qwen2.5-VL-7B", qwen_dir), ("LLaVA-1.5-7B", llava_dir)]:
        mme = pd.read_csv(f"{d}/MME/MME_acc.csv")
        pope = pd.read_csv(f"{d}/POPE/POPE_acc.csv")
        for task in MME_KEY_TASKS:
            row = mme[mme["category"] == task]
            if len(row):
                rows.append({"model": name, "task": task,
                             "score": float(row["score"].iloc[0])})
        # POPE 三种采样取 F1 均值作为单一画像点
        rows.append({"model": name, "task": "POPE(F1 avg)",
                     "score": float(pope["f1"].mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_scores("outputs/qwen25vl7b", "outputs/llava15_7b")
    assert df["model"].nunique() == 2, "必须有两个模型的数据才能对比"
    pivot = df.pivot(index="task", columns="model", values="score")

    ax = pivot.plot(kind="bar", figsize=(12, 5), width=0.75)
    ax.set_title("MME 子任务与 POPE: Qwen2.5-VL-7B vs LLaVA-1.5-7B")
    ax.set_ylabel("score / F1")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("stage3_week2_compare.png", dpi=150)
    print("图已保存: stage3_week2_compare.png")
    print(pivot.round(1))
```

### 2.4 差异笔记的写作框架（交付的一部分）

对比笔记必须回答"细粒度感知 vs 逻辑推理"的维度差异，建议结构：

1. **总分画像表**：两模型在所选子任务的分数并排；
2. **维度结论**：预期模式是 Qwen2.5-VL（更新的数据配方与动态分辨率）在 `count/position/text` 类细粒度任务上显著领先，LLaVA-1.5（固定 336 分辨率）在位置/计数类任务上系统性偏弱——**用你的实际数字验证或修正这个预期**，Stage 1 第 3 周的架构分析（固定 vs 动态分辨率）正好解释这类差异，把评测数字与架构原因接通是本阶段的核心训练；
3. **可复现性脚注**：框架 commit hash、transformers 版本、是否 vLLM、裁判模型、抽取失败率。

---

## 3. 工程权衡与失效模式

### 3.1 决策表：评测规模 vs 资源

| 场景 | 推荐配置 | 理由 |
| --- | --- | --- |
| 快速迭代（改了 LoRA 想看效果） | POPE + MME 感知子集，transformers 后端 | 分钟级反馈，够诊断方向 |
| 发布前全量评测 | 全基准 + vLLM/LMDeploy + `--nproc`=卡数 | 吞吐优先，缓存兜底 |
| 多模型横向对比 | 统一后端、统一裁判、统一框架版本 | 消除协议噪声，否则数字不可比 |
| 疑似分数异常 | 关闭缓存重跑子集 + 人工抽 30 条预测 | 审计预测文件 |

### 3.2 三个代表性失效模式

**失效 1：两模型评分协议不同导致假差异**
- **症状**：对比笔记里 LLaVA 的 MME 分数远低于其官方报告。
- **根因**：官方数字可能用不同 prompt 模板或 ppl 协议；或 LLaVA 环境的 transformers 版本不对导致输出质量退化；或一个走了 judge 提取、另一个只走正则。
- **定位**：抽 30 条 LLaVA 的原始预测肉眼检查（输出是否正常、是否被截断）；对照官方推荐版本表核对环境。
- **修复**：同协议重跑；两模型分环境部署；报告里写清协议。

**失效 2：`--nproc` 开多了，OOM 或预测文件互相踩踏**
- **症状**：8 卡全开 `--nproc 8` 后部分进程 OOM，或合并后的预测文件缺行。
- **根因**：7B 模型每进程占约 16~20GB（bf16 + 激活），40G 卡上 3~4 个进程即触顶；旧版本框架在预测文件追加时存在并发写风险。
- **定位**：`nvidia-smi` 看每进程显存；检查预测文件行数是否等于数据集行数。
- **修复**：`nproc` ≤ `floor(显存 / 每副本占用)`；预测文件行数对不上时删除该 chunk 重跑（缓存机制保证只重算缺失部分）。

**失效 3：缓存复用了旧模型的预测，新模型分数是"别人的"**
- **症状**：换 checkpoint 后重跑评测，分数与旧模型一模一样。
- **根因**：`--work-dir` 未变，框架检测到预测文件已存在，按 index 复用旧预测。
- **定位**：对比新旧预测文件的 MD5/修改时间；或对某条样本的预测文本与新模型手动推理结果比对。
- **修复**：纪律化——每次评测用带时间戳/commit 的新 `work-dir`；或评测前清理。这是评测工程里**发生频率最高、最隐蔽**的事故，建议写进组内 checklist。

---

## 4. 延伸思考题

1. **吞吐核算**：POPE 共约 9000 条、每条生成约 10 个 Token。用 vLLM 的吞吐公式估算：7B 模型在单张 A100 上（decode 吞吐约 2000~4000 Token/s 的并发总量），纯 decode 时间的下限是多少？再算上 prefill（每条约 600 视觉+文本 Token），解释为什么 prefill 往往占评测耗时的大头、而 vLLM 的 chunked prefill 如何缓解。
2. **可复现性设计**：评测报告要能做到"三个月后一键复现"，列出你会在报告里归档的全部要素（提示：≥8 项——模型 hash、框架 commit、各库版本、裁判模型与 prompt、后端类型、work-dir 的预测文件、随机性说明、硬件环境）。
3. **接入练习（自测考点预演）**：读 VLMEvalKit 的 `vlmeval/infer/` 与某个简单模型的包装类（如 `mminstruct` 系或 `qwen2_vl.py`），花 1 小时实现一个返回固定字符串的 DummyModel 并跑通 POPE 的前 20 条（`--data POPE` + 截断数据集）。这个练习完成后，自测清单第 3 项即达成。

---

*下一篇：[第 3 周：视觉幻觉与空间能力深挖](第3周-视觉幻觉与空间能力深挖.md)*
