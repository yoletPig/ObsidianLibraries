# 第 1 周教程：基础评价指标——从 NLL、PPL 到模态对齐得分

> **本周要回答的三个问题**
> 1. NLL/PPL 的"意外程度"怎么计算？它凭什么给数据打分？
> 2. 为什么 NLL 分布的**两端**都是坏样本？Middle 90% 里藏着什么？
> 3. CLIP Score 与预测熵各自补足了 NLL 的什么盲区？

对应学习计划：第 1 周。交付物：PyTorch 批处理脚本，对 1,000 条多模态 SFT 样本计算每条回答的平均 NLL、PPL 与 CLIP Score，绘制分布直方图并可视化 Top 5% 与 Bottom 5% 样本的内容差异。

---

## 1. 第一性原理：用参考模型的"无知"度量数据

### 1.1 根本矛盾：数据质量无法直接观测

"这条训练数据好不好"没有直接的测量仪器——好数据的定义（能让模型学到东西、且教的是对的东西）依赖两个不可观测的量：**目标模型缺什么**、**标注本身是否正确**。数据筛选的全部指标设计，都是构造这两个量的**代理变量（proxy）**。

本阶段的核心代理是**参考模型（Reference Model）的损失**：用一个固定的预训练模型（如 Qwen2-VL 底座）对每条样本的回答计算 NLL——模型觉得"意外"（loss 高）的样本，要么是**模型没学过的新知识**（高价值），要么是**乱码/错标**（噪声）；模型觉得"理所当然"（loss 低）的样本，要么是常识（无用），要么是模型已掌握（冗余）。**单一指标无法区分"高价值"与"噪声"**——这个根本局限正是第 1 周要建立的最重要的认知，也是后续两周组合指标的动机。

### 1.2 NLL 与 PPL：定义与物理含义

对样本 $(x_{img}, Q, A = (a_1, \dots, a_T))$，参考模型对回答部分逐 Token 的负对数似然：

$$
NLL = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(a_t \mid x_{img}, Q, a_{<t})
$$

PPL（困惑度）是它的指数化：

$$
PPL = \exp(NLL)
$$

物理直觉：$\exp(-\log P) = 1/P$，即**模型认为每个位置"实际上有 1/PPL 个候选Token 竞争"**。PPL=1 表示模型完全确定（每个位置概率 1）；PPL=1000 表示每个位置模型在约一千个候选间犹豫。NLL 取平均（per-token）而 PPL 取指数，因此 PPL 对长句中偶发的极端低概率 Token 更敏感（一个 $\log P = -10$ 的 Token 就把 PPL 拉高 $e^{10/T}$）——**长回答比较时用 NLL 更稳，短回答/单 Token 判断（如 Yes/No）时两者趋同**。

三个工程细节（每个都影响打分口径）：

1. **只在回答 Token 上算**：prompt 与图像 Token 不算 loss（Stage 2 的掩码机制），否则分数被问题措辞的难度污染；
2. **长度归一化**：per-token 平均（上式）vs 整句求和（$\sum$ 不除 $T$）——求和版本偏爱长样本，筛选时若回答长度与质量相关会引入偏置；
3. **参考模型与目标模型的关系**：参考模型越强、与目标模型越同源，代理越准。经验做法：用**同底座、未微调**的模型做参考（衡量"底座还不会什么"），或用教师模型做参考（衡量"离专家还有多远"）——两者筛出的子集画像不同，需按目标选择。

### 1.3 NLL 两极诊断：为什么两端都是坏样本

以参考模型的视角把 NLL 轴分成三段：

| 区段 | 样本画像 | 命运 |
| --- | --- | --- |
| **极低（Bottom ~20%）** | 常识问答、模板化回答、"1+1=2"级难度；模型早已掌握，梯度信号趋近于零 | **丢弃**（过简单） |
| **中间（Sweet Spot）** | 模型"会一半"的样本：难度处于能力临界点，梯度既非零又可学习 | **保留主力**（第 3 周展开） |
| **极高（Top ~5%）** | OCR 乱码、标注错误、图文错配、代码/公式噪声；模型"学不会也不该学" | **丢弃**（噪声） |

注意三个重要推论：

1. **不能只挑最高 NLL**（自测清单考点）：最高 NLL 段的富集物是 Outliers——损坏图片、错误标注、乱码。训练它们等于教模型"输出不连贯的内容"，极端情况直接训崩（loss 爆炸）。正确做法是**两端截断**。
2. **截断比例不是普适常数**：Bottom 20% / Top 5% 是学习计划的合理起点，但最佳比例依赖参考模型强度与数据噪声率——噪声率高的爬取数据应放宽 Top 截断，参考模型较弱时应收缩 Bottom 截断（因为它低估"简单"）。**截断比例本身是消融对象**（第 4 周）。
3. **NLL 是相对指标**：换参考模型后分数分布整体平移，筛选结论可能翻转。**报告必须绑定参考模型版本**。

### 1.4 CLIP Score：图文相关性的独立裁判

NLL 有一个结构性盲区：它度量"模型输出回答的难度"，但**无法察觉回答与图片无关**。一条（错误标注的）图文错配样本，回答可能流畅自然（NLL 中等），CLIP 语义空间里却是两个不相干的世界：

$$
\text{CLIP Score} = \cos\big(E_{img}(x_{img}),\ E_{txt}(\text{caption} \oplus Q)\big)
$$

其中 $E_{img}$、$E_{txt}$ 是 CLIP/SigLIP 的双塔编码器（Stage 1 第 1 周的老朋友），文本侧通常取"问题 + 回答摘要"或 caption。低分样本 = 图与文各说各话 = 图文错配/标注噪声的高发区。SigLIP 的分数分布与 CLIP 不同（Sigmoid 训练导致整体相似度偏低），**阈值不可跨模型迁移**。

**CLIP Score 的第二个用途**：它是第 2 周 Image Necessity 的"静态版"先导——静态地度量图文关联强度，而 Image Necessity 动态地度量"图像对预测的因果贡献"。两者互补：CLIP Score 高但 $\Delta L_{Img} \approx 0$ 的样本（图相关但答案不依赖图）正是"伪多模态"的典型画像。

### 1.5 预测分布熵：不确定性的另一面

NLL 度量"真实答案让模型多意外"，熵度量"模型自身的输出分布多犹豫"：

$$
H = -\sum_{v \in V} P_\theta(v \mid x, Q, a_{<t}) \log P_\theta(v \mid x, Q, a_{<t})
$$

（实际计算常取 Top-k 或回答首 Token 的分布熵以控制词表求和成本。）两者的组合很有诊断力：**NLL 高 + 熵低** = 模型很自信但错了（典型的先验陷阱样本，值得高亮审查）；**NLL 高 + 熵高** = 模型不确定且答案出乎意料（新知识或噪声，需进一步区分）；**NLL 低 + 熵低** = 已掌握（冗余）。

---

## 2. 实现与验证

### 2.1 本周 MVP：批处理打分脚本

```python
"""
多模态 SFT 数据打分: NLL / PPL / CLIP Score 批处理计算 + 分布可视化。
运行方式: python stage5_week1_scoring.py --data sft_1000.jsonl --out scores.jsonl --plot
依赖: torch, transformers, pillow, matplotlib
"""
import argparse
import json
import torch
from PIL import Image


@torch.no_grad()
def batch_nll_ppl(model, processor, batch, device):
    """对一批 (image, question, answer) 计算回答区的平均 NLL。
    返回每条的 [avg_nll]。掩码逻辑与 Stage 2 collator 同构。"""
    texts, images = [], []
    for r in batch:
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": r["question"]}]}]
        prompt = processor.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        texts.append(prompt + r["answer"])            # prompt + 回答拼成完整序列
        images.append(r["image"])
    inputs = processor(text=texts, images=images, return_tensors="pt",
                       padding=True).to(device)
    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    # 定位回答区起点: 渲染两次取长度差 (prompt-only 长度), 与 Stage2 教程同构
    plens = []
    for i, t in enumerate(texts):
        ans_head = r_answer_head(batch[i])
        plens.append(len(processor.tokenizer(t[: len(t) - len(ans_head)],
                                             add_special_tokens=False)["input_ids"]))
    for i, L in enumerate(plens):
        labels[i, :L] = -100

    out = model(**inputs, labels=labels)
    # 手动 per-token 重算以取每条平均 (labels=-100 的位置已被模型忽略)
    logits = out.logits[:, :-1]
    tgt = labels[:, 1:]
    mask = tgt != -100
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_nll = -logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    per_sample = (tok_nll * mask).sum(1) / mask.sum(1).clamp(min=1)
    return per_sample.tolist()


def r_answer_head(r):
    """回答原文 (用于定位回答区起点)"""
    return r["answer"]


@torch.no_grad()
def batch_clip_score(clip_model, clip_proc, batch, device):
    imgs = clip_proc(images=[r["image"] for r in batch], return_tensors="pt").to(device)
    txts = clip_proc(text=[f'{r["question"]} {r["answer"][:120]}' for r in batch],
                     return_tensors="pt", padding=True).to(device)
    ie = clip_model.get_image_features(**imgs)
    te = clip_model.get_text_features(**txts)
    ie, te = ie / ie.norm(dim=-1, keepdim=True), te / te.norm(dim=-1, keepdim=True)
    return (ie * te).sum(-1).tolist()                 # 余弦相似度


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="scores.jsonl")
    ap.add_argument("--plot", action="store_true"); ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration,
                              CLIPModel, CLIPProcessor)
    MID = "Qwen/Qwen2-VL-2B-Instruct"                 # 参考模型 (版本必须归档!)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MID, torch_dtype=torch.bfloat16).to(device).eval()
    processor = AutoProcessor.from_pretrained(MID)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    rows = [json.loads(l) for l in open(args.data)]
    assert len(rows) >= 100, "样本过少, 分布无统计意义"
    scores = []
    for i in range(0, len(rows), args.bs):
        batch = rows[i: i + args.bs]
        nlls = batch_nll_ppl(model, processor, batch, device)
        clips = batch_clip_score(clip_model, clip_proc, batch, device)
        for r, nll, cs in zip(batch, nlls, clips):
            scores.append({**{k: r[k] for k in ("question", "answer", "image")},
                           "nll": round(nll, 4), "ppl": round(pow(2.718281828, nll), 2),
                           "clip": round(cs, 4)})
        print(f"\r{i + len(batch)}/{len(rows)}", end="")
    with open(args.out, "w") as f:
        f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in scores)

    # ---- 分布诊断与两端样本抽取 ----
    import statistics as st
    nlls = sorted(s["nll"] for s in scores)
    q = lambda p: nlls[int(p * (len(nlls) - 1))]
    print(f"\nNLL 分布: min={q(0):.3f} p25={q(.25):.3f} 中位={q(.5):.3f} "
          f"p95={q(.95):.3f} max={q(1):.3f}")
    by_nll = sorted(scores, key=lambda s: -s["nll"])
    top5 = by_nll[: max(1, len(by_nll) // 20)]
    bottom5 = by_nll[-max(1, len(by_nll) // 20):]
    json.dump({"top5_high_nll": top5, "bottom5_low_nll": bottom5},
              open("extremes.json", "w"), ensure_ascii=False, indent=1)

    # ---- 断言: 指标行为正确 ----
    assert all(0 <= s["clip"] <= 1 for s in scores), "CLIP 余弦应落在 [-1,1]"
    assert all(s["nll"] >= 0 for s in scores), "NLL 非负"
    # 教学断言: 随机数据上 NLL 与 CLIP 的相关性应较弱 (度量的是不同维度)
    import math
    n = [s["nll"] for s in scores]; c = [s["clip"] for s in scores]
    def pearson(x, y):
        mx, my = st.mean(x), st.mean(y)
        cov = sum((a-mx)*(b-my) for a, b in zip(x, y))
        return cov / math.sqrt(sum((a-mx)**2 for a in x) * sum((b-my)**2 for b in y))
    print(f"corr(NLL, CLIP) = {pearson(n, c):.3f}  (弱相关为正常)")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(n, bins=50); ax[0].set_title("NLL distribution (per-token avg)")
        ax[0].axvline(q(.05), ls="--", c="r"); ax[0].axvline(q(.80), ls="--", c="orange")
        ax[1].hist(c, bins=50); ax[1].set_title("CLIP Score distribution")
        plt.tight_layout(); plt.savefig("stage5_week1_dist.png", dpi=150)
        print("图已保存: stage5_week1_dist.png")


if __name__ == "__main__":
    main()
```

**预期输出形态**（数字随数据而变）：

```text
1000/1000
NLL 分布: min=0.112 p25=0.684 中位=1.052 p95=2.871 max=6.413
corr(NLL, CLIP) = 0.142  (弱相关为正常)
图已保存: stage5_week1_dist.png
```

**人工验收（交付的关键部分）**：打开 `extremes.json` 肉眼对比两端样本——预期 Top 5%（高 NLL）富集乱码回答/明显错标/OCR 噪声，Bottom 5%（低 NLL）富集"看图说话式"模板句与常识问答。**若两端看起来一样**，说明参考模型选错（过强/过弱）或掩码实现有 bug（比如把 prompt 也算了分）。

### 2.2 掩码实现的快速自检

回答区起点定位是本脚本最容易出错处（Stage 2 失效 2 的教训），上线的三重自检：

1. 取 1 条样本，decode 掉 `labels != -100` 的部分，确认**恰好等于回答原文**（含 EOS、不含 prompt 尾部）；
2. 人为把某条的 answer 换成空串，断言该条 NLL 无效（mask 全零触发 `clamp(min=1)` 保护）而非 0 分混入；
3. 对同一批数据跑两次（`shuffle=False`），断言分数逐条完全一致（批处理 padding 不应影响 per-token loss——若不一致，检查 padding 方向与 attention mask）。

---

## 3. 工程权衡与失效模式

### 3.1 打分管线效率速查（10k 样本的参考量级）

| 环节 | 实现 | 量级（单卡 A100, 7B 参考） |
| --- | --- | --- |
| NLL/PPL | 本地 7B 前向（不训练），bf16 | 10k 条 × ~2k Token：30~90 分钟 |
| CLIP Score | base 级 CLIP 前向 | 分钟级 |
| 批处理优化 | `--bs` 调大 + padding 到 multiple-of-16（Stage 2 技巧复用） | 提速 20~50% |
| 熵 | 与 NLL 同一次前向附带计算 | 几乎零增量 |

第 4 周会把 NLL 打分迁到 vLLM 的 `prompt_logprobs` 模式再压一个量级；本周用 transformers 原生实现，**先把口径做对，再谈快**。

### 3.2 三个代表性失效模式

**失效 1：回答区掩码错位，NLL 混入 prompt 分数**
- **症状**：全部样本 NLL 异常偏低（如普遍 <0.3），分布几乎无区分度。
- **根因**：`plens` 定位偏差——`add_generation_prompt` 的模板尾部（`<|im_start|>assistant\n`）被划进回答区，或 chat 模板变体把换行符归错侧（Stage 2 同源问题）。
- **定位**：2.2 节自检第 1 条——decode 监督区肉眼比对。
- **修复**：内容锚点法替代长度法；模板升级后重跑自检。

**失效 2：参考模型选型不当，筛选方向反了**
- **症状**：Bottom 20% 里混着大量高质量复杂样本（如数学推理），Top 5% 反而都是正常样本。
- **根因**：参考模型太弱（2B 参考给 72B 目标筛数据，弱模型"觉得难"的可能是好数据）；或参考模型与数据同源（打分模型自己生成的数据，NLL 系统性偏低）。
- **定位**：两端样本人工审 + 对比不同参考模型的分布形状。
- **修复**：参考模型与目标模型同源同量级；打分模型自产的数据换另一个参考模型打分（利益回避）。

**失效 3：CLIP 阈值照搬他人数字，误杀细粒度数据**
- **症状**：按"CLIP < 0.2 丢弃"清洗后，OCR/细粒度定位类样本几乎全灭，下游对应能力退化。
- **根因**：细粒度任务的文本（如坐标串、代码）在 CLIP 空间与图片的相似度天然偏低——**阈值与任务类型强耦合**，全局单一阈值系统性歧视某些任务。
- **定位**：分任务维度统计 CLIP 分布——应看到不同维度分布中心显著不同。
- **修复**：分维度设阈值或用分布分位数（如"各维度内 CLIP 最低 5%"）替代全局绝对阈值。

---

## 4. 延伸思考题

1. **口径敏感性实验**：把 per-token 平均改成整句求和重跑打分，对比两端样本集合的重合率；构造 10 对"同题不同长度"的样本，验证长度归一化对排序的影响。结论写成一段"口径选择声明"放进筛选报告。
2. **NLL 高 + 熵低的先验陷阱**：从你的数据中挖出该象限的样本（1.5 节的组合），人工审查它们是否多为"模型靠语言先验作答但答案碰巧对/错"的类型。这个象限与第 2 周的 Image Necessity 是什么关系？（提示：$\Delta L_{Img}$ 正是给这个象限定量的工具。）
3. **参考模型的"知识时差"**：假设用 2024 年发布的参考模型给 2026 年的合成数据打分，哪些类型的样本会被系统性高估 NLL？这对"高 NLL = 高价值"的假设构成什么挑战？（提示：新事实/新术语类样本会被误判为噪声——两端截断的比例需要按数据的新鲜度调整。）

---

*下一篇：[第 2 周：双向信息增益与模态依赖分析](第2周-双向信息增益与模态依赖.md)*
