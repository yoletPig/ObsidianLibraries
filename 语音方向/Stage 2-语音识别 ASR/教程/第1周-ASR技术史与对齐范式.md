# 第 1 周教程：ASR 技术史与对齐范式——CTC / Attention / 非自回归

> **本周要回答的三个问题**
> 1. ASR 的根本矛盾是什么？为什么"对齐"是所有方案的出发点？
> 2. CTC 的空白符（blank）如何让我们**不需要逐帧标注**就能训练？前向算法在算什么？
> 3. 自回归（Attention）与非自回归（Paraformer）各自的失效模式是什么？端侧为什么偏爱后者？

对应学习计划：第 1 周。交付物：手推 CTC 对齐路径与前向递推式；画 Whisper vs Paraformer 解码流程对比图。

---

## 1. 第一性原理：ASR 的根本矛盾——对齐未知

### 1.1 问题定义

ASR 要学的是条件分布：给定音频特征序列 $\mathbf{x} = (x_1, \dots, x_T)$（$T$ 帧，每秒约 100 帧），输出文字序列 $\mathbf{y} = (y_1, \dots, y_U)$（$U$ 个字符/子词）。

根本矛盾：**$T$ 与 $U$ 不仅不相等，而且对应关系未知**。一句 3 秒的"你好"：$T \approx 300$ 帧，$U = 2$ 个字。哪些帧对应"你"、哪些对应"好"？标注数据只给了整句文字，没给逐帧对齐。这个"对齐缺失"问题，是三代技术各自回答的核心。

### 1.2 三代答案速览

| 时代 | 代表 | 对齐策略 | 致命弱点 |
| --- | --- | --- | --- |
| 传统 | HMM-GMM | 显式对齐：HMM 状态机 + Viterbi | 模型弱、流水线复杂（声学/发音/语言三模型手工拼接） |
| 端到端 1.0 | CTC / LAS | 隐式对齐：边缘化（CTC）或注意力（Attention） | 条件独立假设（CTC）/ 自回归慢（LAS） |
| 端到端 2.0 | Whisper / Paraformer | 大规模弱监督 + 强骨干；NAR 单步解码 | 数据成本 / 长度预测难 |

下面逐一拆解对齐机制——本周重点是 CTC 的数学。

---

## 2. 前代方案：HMM-GMM 为什么被淘汰

### 2.1 流水线结构

传统 ASR 是三个独立模型的拼接：

$$
\hat{w} = \arg\max_w P(w) \cdot P(\mathbf{x} \mid w)
$$

- **声学模型** $P(\mathbf{x}\mid w)$：GMM 估计每个 HMM 状态发射特征的概率；
- **发音词典**：词 → 音素序列的映射；
- **语言模型** $P(w)$：N-gram 统计。

解码用 WFST（加权有限状态转换器）把三者编译成一张大搜索图，Viterbi 找最优路径。

### 2.2 被淘汰的三个原因

1. **模型容量弱**：GMM 无法学习深层非线性特征，必须手工设计特征（MFCC——第 3 周讲过它正是为此而生）；
2. **误差级联**：三模块独立优化，接口处损失无法端到端反传；
3. **工程复杂**：对齐、聚类、决策树等大量手工流程。

深度学习的胜利在于：**一个神经网络端到端，从原始特征直接学映射**。但"对齐未知"仍在——CTC 与 Attention 是两条端到端的破局路线。

---

## 3. CTC：用边缘化消灭对齐标注

### 3.1 核心思想

既然不知道对齐，那就**对所有可能的对齐求和（边缘化）**。CTC 引入一个特殊的**空白符（blank, $\epsilon$）**，允许：

- 帧可以发射空白（"这一帧不属于任何字符"）；
- 同一字符可以连续发射多次（"好——好"折叠成一个"好"）。

定义折叠函数 $\mathcal{B}$：去掉空白、合并重复字符。例如路径 $\pi = (\epsilon, 你, 你, \epsilon, 好)$ 折叠为 $\mathcal{B}(\pi) = (你, 好)$。

目标概率 = 所有折叠后等于 $\mathbf{y}$ 的路径概率之和：

$$
P(\mathbf{y} \mid \mathbf{x}) = \sum_{\pi : \mathcal{B}(\pi) = \mathbf{y}} P(\pi \mid \mathbf{x})
$$

**关键假设**：各帧条件独立，$P(\pi\mid\mathbf{x}) = \prod_{t=1}^{T} P(\pi_t \mid x_t)$。每帧的发射概率由网络最后一层（softmax，词表 + 空白符）给出。

### 3.2 路径数为什么必须用动态规划

路径数是天文数字：每帧有 $|V|+1$ 种选择，共 $(|V|+1)^T$ 条。直接枚举不可能。但折叠函数有**最优子结构**——前缀路径的折叠只依赖前缀本身——因此可以用**前向算法（forward algorithm）**在 $O(T \cdot U)$ 内求和。

### 3.3 前向算法推导（必须能手写）

构造**扩展标签序列** $\mathbf{y}'$：在 $\mathbf{y}$ 的首、尾、每两个字符之间插入空白符。若 $\mathbf{y}$ 长 $U$，则 $\mathbf{y}'$ 长 $2U+1$。例如 $\mathbf{y} = (你, 好)$ → $\mathbf{y}' = (\epsilon, 你, \epsilon, 好, \epsilon)$。

定义前向变量 $\alpha(t, s)$ = 在时刻 $t$、恰好发射到扩展序列第 $s$ 个位置、且折叠后是 $\mathbf{y}$ 前缀的所有路径概率之和。

**递推**。时刻 $t$ 到达位置 $s$，前一时刻 $t-1$ 只可能在：

- 位置 $s$（原地：重复字符或空白）；
- 位置 $s-1$（前进一步）。

若 $\mathbf{y}'_s$ 是**非空白字符**且与 $\mathbf{y}'_{s-2}$ 不同，还可从 $s-2$ 跳过中间空白直接到达。于是：

$$
\alpha(t, s) =
\begin{cases}
\big[\alpha(t-1, s) + \alpha(t-1, s-1)\big] \cdot P(\mathbf{y}'_s \mid x_t), & \mathbf{y}'_s = \epsilon \text{ 或 } \mathbf{y}'_s = \mathbf{y}'_{s-2} \\
\big[\alpha(t-1, s) + \alpha(t-1, s-1) + \alpha(t-1, s-2)\big] \cdot P(\mathbf{y}'_s \mid x_t), & \text{否则}
\end{cases}
$$

**边界**：$\alpha(1, 0) = P(\epsilon \mid x_1)$，$\alpha(1, 1) = P(\mathbf{y}'_1 \mid x_1)$，其余为 0。

**总概率**：序列必须以空白结尾，故

$$
P(\mathbf{y}\mid\mathbf{x}) = \alpha(T, 2U) + \alpha(T, 2U+1 - 1)
$$

即最后两个位置（末字符与末尾空白）之和。取负对数即为 CTC 损失，可用 $\alpha$ 与对称的后向变量 $\beta$ 高效求梯度。

**数值实现必须在 log 域**：$\alpha$ 是大量小概率连乘，直接算会下溢为 0。用 $\log$-sum-exp：

$$
\log(e^a + e^b) = \max(a,b) + \log(1 + e^{-|a-b|})
$$

### 3.4 CTC 的可验证实现

```python
import numpy as np

NEG_INF = -float("inf")

def logsumexp(a, b):
    """log(exp(a) + exp(b))，数值稳定。"""
    if a == NEG_INF: return b
    if b == NEG_INF: return a
    m = max(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))

def ctc_forward(log_probs: np.ndarray, label_ids: list) -> float:
    """
    CTC 前向算法（log 域）。
    log_probs: (T, V) 每帧每符号的 log 概率，符号 0 约定为 blank。
    label_ids: 目标标签的符号索引列表（不含 blank）。
    返回 log P(label | x)。
    """
    T, V = log_probs.shape
    blank = 0
    # 扩展序列：[blank, l1, blank, l2, blank, ...]
    ext = [blank]
    for l in label_ids:
        ext.extend([l, blank])
    S = len(ext)                       # 2U + 1

    # alpha[t][s]: log 前向概率
    alpha = [[NEG_INF] * S for _ in range(T)]
    alpha[0][0] = log_probs[0, ext[0]]
    if S > 1:
        alpha[0][1] = log_probs[0, ext[1]]

    for t in range(1, T):
        for s in range(S):
            # 原地 + 前进一步
            acc = logsumexp(alpha[t-1][s],
                            alpha[t-1][s-1] if s >= 1 else NEG_INF)
            # 跳过中间空白：当前是非 blank，且与 s-2 不同
            if ext[s] != blank and s >= 2 and ext[s] != ext[s-2]:
                acc = logsumexp(acc, alpha[t-1][s-2])
            alpha[t][s] = acc + log_probs[t, ext[s]]

    # 末尾两位置之和
    return logsumexp(alpha[T-1][S-1], alpha[T-1][S-2])

# ---- 验证 ----
T, V = 5, 4          # 4 个符号：0=blank, 1='你', 2='好', 3='他'
np.random.seed(0)
logits = np.random.randn(T, V)
log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

# 目标 "你好" = [1, 2]
ll = ctc_forward(log_probs, [1, 2])
print(f"log P(你好 | x) = {ll:.4f}")

# 暴力枚举验证：列出所有折叠后为 [1,2] 的路径并求和
from itertools import product
def collapse(path):
    out = []
    for s in path:
        if s == 0: continue
        if out and out[-1] == s: continue
        out.append(s)
    return out

brute = 0.0
for path in product(range(V), repeat=T):
    if collapse(path) == [1, 2]:
        p = 1.0
        for t, s in enumerate(path):
            p *= np.exp(log_probs[t, s])
        brute += p
print(f"暴力枚举 log P = {np.log(brute):.4f}")
assert abs(ll - np.log(brute)) < 1e-6, "前向算法与暴力枚举不一致！"
print("前向算法实现正确 ✓")
```

**预期输出**：两个对数概率相等（误差 < 1e-6），断言通过。这就是本周交付的核心——**用暴力枚举交叉验证你的前向算法**，这是检验实现正确性的黄金标准。

### 3.5 解码：贪心与束搜索

训练边缘化所有路径，推理要找出**最优**路径：

- **贪心**：每帧取 argmax，再折叠。快，但忽略序列整体概率；
- **CTC beam search**：保留 top-k 前缀路径，考虑"重复字符合并"与"空白跨越"的概率合并。精度更高。

---

## 4. Attention-based Seq2Seq：另一条路

### 4.1 LAS（Listen, Attend and Spell）

不边缘化对齐，而是**用注意力隐式学对齐**：Encoder 读音频，Decoder 自回归生成文字，每步用 cross-attention 看向音频的某个区域。对齐由注意力权重自动涌现，无需空白符。

**优势**：建模标签间依赖（语言模型内置）、无条件独立假设。
**劣势**：自回归——逐字生成，延迟随句长线性增长；且注意力可能重复/跳字。

### 4.2 CTC + Attention 联合（Hybrid）

ESPnet 的标准配方：两个损失加权

$$
\mathcal{L} = \lambda \mathcal{L}_{\text{CTC}} + (1-\lambda)\mathcal{L}_{\text{Att}}, \quad \lambda \approx 0.3
$$

CTC 帮助对齐快速收敛、Attention 负责语言流畅；解码时两者分数融合（shallow fusion）。

---

## 5. 非自回归（NAR）：Paraformer 与 CIF

### 5.1 为什么需要快

自回归每生成一个字都要一次前向。端侧芯片算力有限，一句话 10 个字就是 10 次串行前向——延迟不可接受。**非自回归（NAR）**一步并行输出所有字。

### 5.2 难题：输出长度未知

并行输出要求**预先知道 $U$**（输出多少个字），但 $U$ 正是未知的。Paraformer 的答案是 **CIF（Continuous Integrate-and-Fire）**。

### 5.3 CIF 机制（直觉版）

CIF 给每帧预测一个"发射权重" $w_t \in [0,1]$（该帧承载多少字符信息），沿时间**累积积分**：

$$
\text{acc}_t = \text{acc}_{t-1} + w_t
$$

当累积量 $\text{acc}_t \ge 1$ 时"开火"（fire）——表示攒够了一个字的信息，触发一次输出，并把累积量减 1 继续。**总开火次数 ≈ 字数 $U$**。这样：

1. 模型自己预测出输出长度 $U$（开火次数）；
2. 每次开火位置的声学特征被提取为该字的表示，并行解码。

于是 NAR 一步生成 $U$ 个字，速度是自回归的数十倍——这就是"SenseVoice 比 Whisper 快 15 倍"的根源。

### 5.4 Whisper（AR）vs Paraformer（NAR）对比

| 维度 | Whisper（自回归） | Paraformer（CIF-NAR） |
| --- | --- | --- |
| 解码 | 逐字生成，$U$ 次前向 | 一步并行，约 1 次前向 |
| 延迟 | 随句长线性增长 | 近恒定，端侧友好 |
| 质量 | 高（充分语言建模） | 略低（长度预测误差、无充分自回归语言建模） |
| 失效模式 | 慢、长音频易幻觉重复 | 字数预测不准 → 漏字/多字 |

**本周交付图**：把这两种解码流程画成对比图（可用文字流程描述或 Mermaid），标注延迟与质量权衡——这是你第 6 周自测"端侧为什么选 NAR"的答案。

---

## 6. 工程权衡与失效模式

### 6.1 权衡

- **CTC 条件独立**：帧间独立假设丢弃了标签依赖，纯 CTC 模型语言流畅度差 → 需要外挂 LM 或换 Attention。
- **AR vs NAR**：质量 vs 延迟的直接交换。云端追求质量用 AR（Whisper），端侧追求延迟用 NAR（Paraformer/SenseVoice）。
- **beam 宽度**：宽 → 精度↑、延迟↑。端侧常用贪心或极小 beam。

### 6.2 失效模式

1. **CTC 重复坍缩**：模型学会几乎全发射空白或反复同一字符。症状：输出大量重复或缺失；根因：训练初期对齐未收敛或梯度问题；修复：联合 Attention 损失、检查学习率。
2. **Attention 跳字/重复**：注意力对齐漂移。症状：漏词或循环重复；修复：注意力约束、联合 CTC。
3. **NAR 长度误差**：CIF 开火次数 ≠ 真实字数 → 整句漏/多。症状：特定长度句子系统性错误；修复：长度校准、混合解码。
4. **长音频幻觉**（Whisper 特有）：30 秒窗口下，静音段被"脑补"出文字。修复：VAD 分段、温度回退解码。

---

## 7. 延伸思考题（含解析）

**Q1**：为什么 CTC 要在标签之间强制插入空白符？直接允许重复不行吗？
**A**：空白符区分"同一字符的连续重复"与"两个独立字符"。没有空白，无法表达"啊啊"（两个字）与"啊"的长音之间的区别——折叠函数无法区分路径 (啊,啊) 该折成一个还是两个。空白提供了必要的分隔符号。

**Q2**：CTC 前向算法的复杂度是多少？为什么比暴力枚举可行？
**A**：$O(T \cdot U)$（$U$ 为扩展标签长度），暴力枚举是 $O((|V|+1)^T)$。动态规划利用折叠的最优子结构把指数级压到线性级——与 HMM 前向、注意力 DP 同源的思想。

**Q3**：为什么实现必须在 log 域？不用会怎样？
**A**：$\alpha$ 是大量小于 1 的概率连乘，几十帧后就下溢为 0（float 最小约 1e-308）。log 域把乘法变加法，避免下溢，且 log-sum-exp 数值稳定。

**Q4**：Paraformer 的 CIF 如何同时解决"长度未知"和"并行解码"两个问题？
**A**：积分发射机制让总开火次数自然等于预测字数 $U$（解决长度未知），每个开火点的特征独立解码（支持并行）。一步前向出全句。

**Q5**：端侧离线助手为什么选 SenseVoice-Small（NAR）而不是 Whisper？
**A**：① NAR 单步解码延迟近恒定，适配算力受限的 NPU/CPU；② SenseVoice 针对中文优化且附带标点/情感多任务；③ 显存/内存占用小。代价是极端长尾质量略逊，但端侧交互场景可接受。

---

## 本周交付清单

- [ ] 手推 CTC 扩展序列与三类转移的递推式（能闭卷写出）。
- [ ] 跑通 `ctc_forward()`，暴力枚举交叉验证误差 < 1e-6。
- [ ] 画 Whisper（AR）vs Paraformer（CIF-NAR）解码对比图，标注延迟/质量权衡。
