# 第 4 周教程：端到端神经分离（EEND）与重叠处理

> **本周要回答的三个问题**
> 1. PIT（Permutation Invariant Training）损失在解决什么问题？没有它会怎样？
> 2. EEND 的输出空间与聚类路线有什么本质不同？为什么这让它天然支持重叠？
> 3. EEND 的阿喀琉斯之踵是什么？EEND-EDA / TS-VAD 如何补救？

对应学习计划：第 4 周。交付物：在重叠率不同的数据上对比聚类法与端到端法的 DER，输出"什么场景选什么方案"的结论表。

---

## 1. 第一性原理：把分离重新定义为"多标签逐帧分类"

### 1.1 聚类路线的输出空间缺陷（回顾）

第 3 周的结论：聚类路线每段只能输出**一个**说话人标签（互斥标签），重叠语音处理不了。根因在**输出空间设计**。

### 1.2 EEND 的重新定义

EEND（End-to-End Neural Diarization）把分离定义为：

$$
\text{输入：音频特征} \quad \longrightarrow \quad \text{输出：每帧} \times \text{每说话人的活动概率}
$$

即输出矩阵 $\mathbf{Y} \in [0,1]^{T \times S}$（$S$ = 最大说话人数），$Y_{t,s}$ = "时刻 $t$ 说话人 $s$ 在说话"的概率。**每帧可以同时有多个说话人为 1**——重叠被输出空间天然支持。

### 1.3 训练目标：逐帧二元交叉熵

对每个说话人维度独立做二分类：

$$
\mathcal{L} = \frac{1}{T \cdot S} \sum_{t,s} \text{BCE}(Y_{t,s}, \hat{Y}_{t,s})
$$

但有个致命问题：**说话人编号是任意的**。训练样本里"说话人 1"和"说话人 2"的编号可以互换而语义不变——网络不知道哪个输出神经元该对应哪个人。

---

## 2. PIT：置换不变训练

### 2.1 问题

若直接用固定编号算损失，网络会被"编号"本身干扰：同一内容换个编号顺序，损失就不同。这会让训练信号充满噪声。

### 2.2 PIT 的解法

对每个样本，**枚举所有说话人置换**，取损失最小的那个置换作为该样本的监督方式：

$$
\mathcal{L}_{\text{PIT}} = \min_{\pi \in \mathcal{P}_S} \; \frac{1}{T \cdot S} \sum_{t,s} \text{BCE}(Y_{t,s}, \hat{Y}_{t,\pi(s)})
$$

其中 $\mathcal{P}_S$ 是 $S$ 个说话人的所有置换集合（$S!$ 种）。

**直觉**：不告诉网络"1 号神经元必须是谁"，只要求"存在某种一一对应，使预测与真值吻合"。网络因此可以自由组织输出神经元，专注学"有几个不同的声音、各自的活动区间"。

**代价**：$S!$ 随 $S$ 阶乘增长。$S = 3$ 时有 6 种置换，$S = 5$ 时有 120 种——**这是 EEND 说话人数上限受限的根源**。

---

## 3. EEND 家族演进

### 3.1 EEND（基础版）

Transformer Encoder + 线性层 → $T \times S$ 概率。适合 2-3 人的固定上限场景。

### 3.2 EEND-EDA：动态说话人数

**问题**：基础版 $S$ 固定，但真实会议人数不定。

**EEND-EDA（Encoder-Decoder Attractor）**：用一个吸引子（attractor）解码器**动态生成**说话人表示，生成到"没有更多说话人"为止。人数不再是超参数，而是模型输出。

### 3.3 EEND-VC：跨会话一致性

**问题**：EEND 每次处理一段音频，不同会议里的"说话人 1"无法对应同一个人。

**EEND-VC（Vector Clustering）**：局部 EEND + 全局声纹聚类结合——先局部块内分离，再用声纹向量跨块/跨会话聚类对齐。

### 3.4 TS-VAD：目标说话人路线

**TS-VAD（Target-Speaker VAD）**：给定每个目标说话人的参考声纹（或前一轮估计），模型输出"这些特定说话人各自的活动"。

$$
\text{输入：音频 + } S \text{ 个目标声纹} \quad \to \quad \text{输出：这 } S \text{ 人各自的活动区间}
$$

**优势**：迭代精化（用上一轮输出作参考再跑一轮），DIHARD 竞赛冠军方案常用。
**前提**：需要说话人参考（注册声纹或聚类初始化）。

---

## 4. 两条路线的本质对比（本周核心表）

| 维度 | 聚类路线 | EEND / TS-VAD |
| --- | --- | --- |
| 输出空间 | 每段一个互斥标签 | 每帧多标签（支持重叠） |
| 重叠处理 | 先天缺陷，需 OSD 补丁 | 天然支持 |
| 说话人数 | 任意（聚类自适应） | 受 $S!/S$ 上限约束 |
| 长音频 | 适合（分块聚类） | 受上下文长度限制 |
| 计算 | 声纹提取 + 聚类（$O(n^2)$） | Transformer 前向（重） |
| 工业成熟度 | 高（pyannote/NeMo） | 中（研究前沿 + 竞赛） |

**一句话决策**：
- **人数少、重叠多、短片段** → EEND/TS-VAD；
- **人数多、长会议、重叠少** → 聚类路线。
- **当前工业最佳实践**：聚类骨架 + 重叠检测补丁，或聚类初始化 + TS-VAD 精化。

---

## 5. 实现与验证（交付核心）

### 5.1 对比实验设计

```python
# 数据：LibriMix 或 VoxConverse（含重叠标注）
# 目标：按重叠率分档，对比两条路线的 DER

overlap_bins = {
    "0%（无重叠）":   [],   # 填充对应的音频段
    "10%（少量重叠）": [],
    "20%（大量重叠）": [],
}
```

```python
from pyannote.metrics.diarization import DiarizationErrorRate

metric = DiarizationErrorRate()
results = {}

for bin_name, items in overlap_bins.items():
    der_cluster, der_eend = [], []
    for audio, reference in items:
        # 路线 1：聚类法（ECAPA + 谱聚类，可复用第 3 周代码）
        hyp_cluster = cluster_pipeline(audio)
        der_cluster.append(metric(reference, hyp_cluster))
        # 路线 2：端到端（pyannote 或 EEND 实现）
        hyp_eend = eend_pipeline(audio)
        der_eend.append(metric(reference, hyp_eend))
    results[bin_name] = {
        "聚类法 DER": np.mean(der_cluster),
        "端到端 DER": np.mean(der_eend),
    }

print(f"{'重叠率':<16}{'聚类法':>10}{'端到端':>10}")
for name, r in results.items():
    print(f"{name:<16}{r['聚类法 DER']*100:>9.2f}%{r['端到端 DER']*100:>9.2f}%")
```

**预期结果模式**：
- 0% 重叠：两法接近，聚类法甚至略优（更稳）；
- 10-20% 重叠：端到端法显著领先——**重叠越多，端到端优势越大**。

这张表就是你"什么场景选什么方案"结论的数据支撑。

### 5.2 手写聚类部分（谱聚类）

```python
import numpy as np
from sklearn.cluster import SpectralClustering

def cluster_embeddings(embs: np.ndarray, n_speakers: int):
    """embs: (N, D) 声纹矩阵 -> 每段的说话人标签"""
    # 亲和矩阵：余弦相似度，裁剪到 [0, 1]
    embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    affinity = np.clip(embs_n @ embs_n.T, 0, 1)
    sc = SpectralClustering(n_clusters=n_speakers,
                            affinity="precomputed",
                            assign_labels="kmeans",
                            random_state=0)
    return sc.fit_predict(affinity)
```

**验证点**：同一批声纹，$n_{\text{speakers}}$ 给对时标签应与真实说话人高度一致（ARI > 0.9）；给错时观察"同人被拆"现象。

---

## 6. 工程权衡与失效模式

### 6.1 权衡

- **$S$ 上限**：EEND 的 $S!$ 置换代价限制说话人数。3-4 人是实用上限；
- **上下文长度**：Transformer 窗口限制，长音频要分块 + 跨块对齐（EEND-VC）；
- **计算成本**：端到端前向远贵于声纹提取，端侧慎用。

### 6.2 失效模式

1. **人数超出 $S$**：真实 5 人但 $S=3$，模型只能硬凑。修复：EEND-EDA 或切段处理。
2. **跨会话混淆**：两段音频的"说话人 1"不是同一人。修复：EEND-VC / TS-VAD + 全局声纹。
3. **重叠漏检**：EEND 在重叠段置信度低。修复：加重叠辅助损失、提升训练数据重叠比例。
4. **长音频漂移**：分块处理时块间标签不一致。修复：声纹对齐块间标签。

---

## 7. 延伸思考题（含解析）

**Q1**：PIT 损失为什么必要？不用会怎样？
**A**：说话人编号是任意的，固定编号监督会给网络引入与内容无关的编号噪声。PIT 枚举所有置换取最优，让网络只学"内容分组"不学"编号"。不用会训练不稳、收敛到与编号绑定的错误解。

**Q2**：聚类法与 EEND 的重叠处理能力差异的根因？
**A**：在输出空间设计。聚类法每段输出互斥单标签，重叠段只能选一个；EEND 输出每帧多标签，重叠天然可表达。这是架构层面的差异，不是调参能弥补的。

**Q3**：两人同时说话怎么办？
**A**：① 若用聚类路线：加 OSD 检测重叠区，标"多说话人"，但拆不开内容；② 若用端到端：EEND/TS-VAD 直接输出两人各自的活动区间。要求精确拆分时必须走端到端。

**Q4**：EEND 为什么有说话人数上限？
**A**：PIT 需枚举 $S!$ 置换，且输出层维度随 $S$ 线性增长，训练与推理成本阶乘/线性上升。实用上限 3-4 人。EEND-EDA 用吸引子动态生成缓解，但仍有天花板。

**Q5**：你的结论表会怎么写"什么场景选什么方案"？
**A**：短片段、≤4 人、重叠多 → 端到端（EEND/TS-VAD）；长会议、人数多、重叠少 → 聚类路线；工业折中 → 聚类骨架 + 重叠检测，或聚类初始化 + TS-VAD 精化。用实测 DER 数据支撑。

---

## 本周交付清单

- [ ] 准备不同重叠率的测试数据（LibriMix/VoxConverse 或自混）。
- [ ] 跑通聚类法（手写谱聚类）与端到端法（pyannote）。
- [ ] 输出重叠率 × 方法 的 DER 对比表，得出结论。
- [ ] 能闭卷解释：PIT 原理、输出空间差异、EEND 家族演进。
