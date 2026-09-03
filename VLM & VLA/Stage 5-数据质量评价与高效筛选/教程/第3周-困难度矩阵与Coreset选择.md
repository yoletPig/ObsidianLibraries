# 第 3 周教程：困难度筛选矩阵、多样性与 Coreset 选择

> **本周要回答的三个问题**
> 1. 困难度-质量二维矩阵为什么把 "Sweet Spot" 放在中间而非两端？
> 2. 只按 Loss 筛选为什么会"全挤在一个领域"？Embedding 聚类怎么解？
> 3. K-Means 与最远点采样（FPS）各自在保证什么？怎么组合成两阶段 Pipeline？

对应学习计划：第 3 周。交付物：复合筛选 Pipeline——第一阶段用 NLL 剔除最高 5%（噪声）与最低 20%（过简单）；第二阶段抽取 LLM 倒数第二层 Embedding，K-Means 聚 10 簇，簇内按 NLL 均匀抽取，产出兼具困难度与多样性的 30% 精炼数据集。

---

## 1. 第一性原理：从"样本好坏"到"集合构成"

### 1.1 根本矛盾：独立打分 ≠ 组合最优

第 1-2 周的所有指标都是**逐样本独立打分**——每条数据拿到一个分数，排序截断。但这忽略了一个关键事实：**训练的价值取决于样本集合的构成，而非样本个体的绝对分数**。两个集合层面的失败模式：

1. **领域坍缩（Domain Collapse）**：单纯按 NLL 排序取 Top，选出的几乎全是同一类"难"样本（如全是复杂数学题/长 OCR）——NLL 高与"任务类型难"高度相关。训练分布严重偏斜，其他能力遗忘（Stage 2 第 3 周的机制在数据选择中的重现）。
2. **语义冗余**：分数相近的样本可能内容高度雷同（同一模板的变体），保留 10 条学不到比保留 1 条更多的东西（Stage 4 语义去重的"集合版"问题）。

**Coreset（核心集）选择的本质**：在预算约束（30%）下，选出一个**子集，使其对参数空间的影响（梯度/损失变化）尽可能逼近全量数据**。直接优化"影响逼近"计算不可行（需要梯度内积，$O(n^2)$），因此用两类代理：**质量代理**（困难度——这条数据"值不值得学"）与**覆盖代理**（多样性——这类数据"有没有被代表过"）。本周的两阶段 Pipeline 就是这两个代理的串联。

### 1.2 困难度-质量二维矩阵

把第 1 周的 NLL（困难度轴）与噪声判定（质量轴：CLIP Score 低/规则过滤命中/两端异常）组合成二维决策矩阵：

| | **质量高** | **质量低** |
| --- | --- | --- |
| **困难度低**（NLL 低） | Too Easy：模型已掌握，**丢弃** | 无价值噪声，**丢弃** |
| **困难度中**（Sweet Spot） | **核心资产：能力临界点样本，保留** | 混杂：逐条审查 |
| **困难度高**（NLL 高） | **难题：有条件保留**（如带验证的推理链） | Too Hard/Noisy：乱码/错标，**丢弃** |

"Sweet Spot 在中间"的机制解释：学习信号 $\propto$ 梯度大小 $\approx$ 模型的预测误差。误差趋零（太简单）→ 无梯度；误差巨大且结构混乱（太难/噪声）→ 梯度方向不可信、损失面震荡。**可学习的最优区间在"模型踮脚够得着"的能力边界（Margin Zone）**。这个论断有个重要推论：**Sweet Spot 的位置随参考模型移动**——用 2B 参考筛出的"难"，对 7B 目标可能只是"中等"。因此矩阵的困难度轴应该用**与目标模型同源的参考模型**打分，且同一数据集在不同训练阶段重新打分（能力边界推移了）。

DEITA（已核实：*What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning*, ICLR 2024）给出的三维框架与此同构——**复杂度（Complexity）、质量（Quality）、多样性（Diversity）**——并验证了"按此三维度量 + 简单选择策略，6K 样本可匹敌基线"的 Less-is-More 结论。困难度矩阵是其在多模态场景的操作化。

### 1.3 多样性的几何：Embedding 空间中的覆盖

多样性需要一个"语义距离"的定义。第 4 阶段用过 CLIP 相似度（字面/浅语义），本阶段升级为**目标模型自身的隐层表征**：

- **倒数第二层 hidden state**（回答区 Token 的平均池化）：它是 LLM 对样本"理解"的内部编码，比 CLIP 更贴近"训练动力学的相似性"——两个在该空间中靠近的样本，对模型的影响也更相似（XMAS 等工作证明了注意力/隐层特征与梯度影响的关联，arXiv:2510.01454）。
- 高维（4k）不能直接聚类（维度灾难：距离集中效应），先 **PCA 降到 64~128 维**，保留主要方差方向。

两个覆盖算法的分工：

| 算法 | 保证 | 复杂度 | 适用 |
| --- | --- | --- | --- |
| **K-Means 聚类 + 簇内配额采样** | "每个语义区域按比例有代表" | $O(nkdi)$（近似线性） | 量大（>5k）、需要任务配额控制 |
| **最远点采样（FPS）** | "选中点两两距离最大化"——均匀覆盖整个空间 | $O(nk)$ 朴素实现 | 量中小、追求几何均匀 |

K-Means 的簇天然对应"任务/领域"（无需人工标注的任务边界），簇内按 NLL 分层采样同时满足困难度与多样性；FPS 则不假设簇结构、几何上更均匀但对离群点敏感。**两阶段 Pipeline 选 K-Means 为主**（配额可控、可解释——10 个簇可以人工抽查对应什么任务）。

---

## 2. 实现与验证

### 2.1 本周 MVP：两阶段复合筛选 Pipeline

```python
"""
两阶段复合筛选: NLL 两端截断 -> 倒数第二层 Embedding -> KMeans 簇内均匀采样。
运行方式:
  步骤A(打分+抽特征): python stage5_week3_coreset.py --data sft.jsonl --extract
  步骤B(选择):        python stage5_week3_coreset.py --select --keep 0.30 --clusters 10
依赖: torch, transformers, scikit-learn, numpy
"""
import argparse
import json
import numpy as np


# ---------- 第一阶段: NLL 两端截断 ----------
def stage1_filter(scores_path, drop_high=0.05, drop_low=0.20):
    scores = [json.loads(l) for l in open(scores_path)]   # 第1周产物
    s = sorted(scores, key=lambda r: r["nll"])
    n = len(s)
    lo, hi = int(n * drop_low), n - int(n * drop_high)
    kept = s[lo:hi]                                       # 剔除两端
    # 质量底线: CLIP 过低者一并出局 (第1周指标串联)
    kept = [r for r in kept if r.get("clip", 1.0) > 0.15]
    print(f"阶段1: {n} -> {len(kept)} (剔除低NLL {lo} + 高NLL {n - hi} + CLIP低分)")
    return kept


# ---------- 第二阶段特征: 倒数第二层 hidden state ----------
@torch.no_grad()
def extract_penult_embeddings(rows, model, processor, device):
    """回答区最后一 Token 的倒数第二层 hidden state 作为样本表示"""
    embs = []
    for i in range(0, len(rows), 4):                      # 小批
        batch = rows[i:i + 4]
        texts, images = [], []
        for r in batch:
            msgs = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": r["question"]}]}]
            p = processor.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
            texts.append(p + r["answer"]); images.append(r["image"])
        inputs = processor(text=texts, images=images, return_tensors="pt",
                           padding=True).to(device)
        out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states[-2]                          # 倒数第二层 [B,L,d]
        # 用每条最后一个非 padding 位置的向量作为样本表示
        lens = inputs["attention_mask"].sum(1) - 1
        emb = h[torch.arange(h.size(0)), lens]
        embs.append(emb.float().cpu().numpy())
    return np.concatenate(embs)


def stage2_select(kept, embs, keep_ratio=0.30, k=10, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    X = PCA(n_components=64, random_state=seed).fit_transform(embs)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
    labels = km.labels_

    n_keep = int(len(kept) * keep_ratio)
    # 簇配额: 按簇大小比例分配 (保底每簇 1)
    quota = np.maximum(1, (np.bincount(labels, minlength=k) / len(labels) * n_keep).astype(int))
    while quota.sum() > n_keep:                            # 配额超支时从最大簇扣
        quota[np.argmax(quota)] -= 1

    chosen = []
    for c in range(k):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue
        # 簇内按 NLL 均匀分层取: 排序后等间隔采样 (覆盖困难度谱系)
        idxs = idxs[np.argsort([kept[i]["nll"] for i in idxs])]
        take = quota[c]
        if len(idxs) <= take:
            chosen.extend(idxs.tolist())
        else:
            pos = np.linspace(0, len(idxs) - 1, take).round().astype(int)
            chosen.extend(idxs[pos].tolist())
    return sorted(set(chosen)), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sft.jsonl"); ap.add_argument("--scores", default="scores.jsonl")
    ap.add_argument("--extract", action="store_true"); ap.add_argument("--select", action="store_true")
    ap.add_argument("--out", default="coreset_30.jsonl")
    ap.add_argument("--keep", type=float, default=0.30); ap.add_argument("--clusters", type=int, default=10)
    args = ap.parse_args()

    kept = stage1_filter(args.scores)
    if args.extract:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        MID = "Qwen/Qwen2-VL-2B-Instruct"
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MID, torch_dtype=torch.bfloat16).to(device).eval()
        processor = AutoProcessor.from_pretrained(MID)
        rows = {id(r): r for r in json.load(open(args.data))}
        # 与 scores 对齐: scores 保留了 question 作为键
        q2row = {r["question"]: r for r in json.load(open(args.data))}
        rows_ord = [q2row[r["question"]] for r in kept]
        embs = extract_penult_embeddings(rows_ord, model, processor, device)
        np.save("embs.npy", embs)
        json.dump([r["question"] for r in kept], open("kept_keys.json", "w"), ensure_ascii=False)
        print(f"特征已抽取: embs.npy {embs.shape}")
        return

    if args.select:
        embs = np.load("embs.npy")
        assert len(embs) == len(kept), "特征与打分记录错位, 重跑 --extract"
        chosen, labels = stage2_select(kept, embs, args.keep, args.clusters)

        # ---- 断言: 筛选质量的关键统计 ----
        assert len(chosen) >= int(len(kept) * args.keep * 0.9), "产出不足"
        sel_nll = [kept[i]["nll"] for i in chosen]
        all_nll = [r["nll"] for r in kept]
        import statistics as st
        # 均值应接近 (保留中间困难度, 不因聚类而偏移)
        assert abs(st.mean(sel_nll) - st.mean(all_nll)) < 0.5, "簇内采样偏移了困难度分布"
        # 多样性: 各簇均应有代表
        from collections import Counter
        cnt = Counter(labels[i] for i in chosen)
        assert len(cnt) == args.clusters and min(cnt.values()) >= 1, "存在空簇, 多样性失败"
        q2row = {r["question"]: r for r in json.load(open(args.data))}
        with open(args.out, "w") as f:
            for i in chosen:
                f.write(json.dumps(q2row[kept[i]["question"]], ensure_ascii=False) + "\n")
        print(f"簇分布: {dict(sorted(cnt.items()))}")
        print(f"最终 Coreset: {len(chosen)} 条 ({len(chosen)/len(kept):.0%} of 阶段1产出) -> {args.out}")


if __name__ == "__main__":
    main()
```

**预期输出形态**：

```text
阶段1: 1000 -> 742 (剔除低NLL 200 + 高NLL 50 + CLIP低分)
特征已抽取: embs.npy (742, 3584)
簇分布: {0: 22, 1: 24, 2: 21, 3: 25, 4: 23, 5: 20, 6: 24, 7: 22, 8: 23, 9: 21}
最终 Coreset: 225 条 (30% of 阶段1产出) -> coreset_30.jsonl
```

**人工验收**：抽 3 个簇各看 5 条样本，判断簇是否对应可命名的任务类型（如"图表问答簇""空间关系簇"）——簇的可解释性是"多样性覆盖"有效性的直接证据；若所有簇内容雷同，说明 Embedding 未捕获语义（检查是否忘了 PCA 归一化，或 hidden state 取错层）。

---

## 3. 工程权衡与失效模式

### 3.1 决策表：选择算法的参数空间

| 参数 | 起点 | 说明 |
| --- | --- | --- |
| 两端截断比例 | 高 5% / 低 20% | 按数据噪声率与参考模型强度调整（第 1 周 1.3 节） |
| 聚类特征 | 倒数第二层 + 回答区池化 | 换层（如中层）会改变语义粒度——中层更句法、深层更任务化 |
| PCA 维度 | 64~128 | 过高受维度灾难，过低丢多样性信息 |
| K（簇数） | 10 | 用轮廓系数（silhouette）辅助扫描 5~30；簇要可人工命名 |
| 簇内采样 | NLL 等间隔分层 | 也可按 ΔL_Img 分层（强化视觉多样性） |

### 3.2 三个代表性失效模式

**失效 1：聚类特征坍缩到"长度方向"**
- **症状**：簇与"回答长短"强相关（短答案一簇、长答案一簇），而非任务类型。
- **根因**：hidden state 的范数/位置编码残留与序列长度耦合，PCA 第一主成分被长度占据。
- **定位**：对 Embedding 与样本长度做相关分析；看第一主成分的载荷。
- **修复**：L2 归一化（已内置）；或改用"均值池化 + 去长度中心化"；或对回答 Token 做 mean-pooling 而非取末 Token。

**失效 2：K-Means 簇极不均衡，配额保底反而引入噪声簇**
- **症状**：一个巨型簇占 80%，其余小簇每簇只被保底采 1 条——小簇的 1 条可能恰是离群噪声。
- **根因**：数据本身主导任务集中（真实分布如此），均匀配额与保底机制放大了小簇的话语权。
- **定位**：打印簇大小分布；对最小簇的样本人工审。
- **修复**：配额按 $\sqrt{\text{簇大小}}$ 或温度公式（Stage 4 第 3 周的 $\alpha$ 采样）软平衡而非硬保底；最小簇样本数 <5 时并入邻近簇。

**失效 3：两端截断把"目标模型真正要学的新任务"扔掉了**
- **症状**：30% Coreset 训练后，通用能力持平但新兴任务（如新 benchmark）表现反不如随机子集。
- **根因**：参考模型"没见过"的新任务类型 NLL 天然极高，被当作噪声截断——**参考模型的时差**（第 1 周思考题 3 的机制）在矩阵阶段兑现为损失。
- **定位**：被截断的高 NLL 段抽样人工审——区分"乱码噪声"与"新任务好数据"。
- **修复**：高 NLL 段不直接丢弃，先过质量闸（CLIP/规则/人工抽检），质量过关的"难而正确"样本单独成簇保留；或用更强参考模型（教师）重新打分这一段。

---

## 4. 延伸思考题

1. **FPS 对比实验**：把第二阶段替换为最远点采样（k-center：贪心选距已选集最远的点），在同预算下与 K-Means 方案对比：两子集的 Jaccard 重合度、簇覆盖率、以及 30% 预算下的训练效果（第 4 周）。写一段"算法敏感性"结论。
2. **Sweet Spot 的动态性**：设计一个"迭代式筛选"方案——第一轮用底座模型打分筛选、训练后用**新模型**重打分、再筛再训。这个自举循环的收敛条件与风险（如分布越走越窄）是什么？（提示：这是 curriculum / self-paced learning 的思想；防窄化手段是每轮强制保留一定比例的随机样本。）
3. **理论拉伸**：Coreset 的严格定义要求"子集训练的损失变化 ε-逼近全量训练"。本章的"质量代理 + 覆盖代理"在什么假设下是这个定义的合理近似？什么数据分布会使其失效？（提示：假设"隐层近邻 → 梯度相似"（XMAS 的经验结论）；失效场景——两个隐层近但梯度异号的样本，覆盖代理会误判冗余。）

---

*下一篇：[第 4 周：筛选算法验证与 "Less is More" 消融实验](第4周-LessIsMore消融验证.md)*
