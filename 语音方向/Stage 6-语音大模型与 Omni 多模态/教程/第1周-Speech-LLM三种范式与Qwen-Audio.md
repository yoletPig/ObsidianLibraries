# 第 1 周教程：Speech-LLM 三种范式与 Qwen-Audio——把你学过的 LLaVA 听一遍

> **本周要回答的四个问题**
> 1. 让 LLM「听懂语音」有三条路：级联包装、音频特征注入、Speech-to-Speech——各自的误差传播与延迟天花板是什么？
> 2. Qwen-Audio 与 SALMONN 的音频塔、Projector 各怎么设计？为什么都要做「token 降采样」？
> 3. 音频塔 ≈ Vision Tower、Projector 完全同构——这个类比的逐项对照表是什么？差异点在哪？
> 4. 级联方案丢掉的「副语言信息」（语气、情绪、说话人）如何在特征注入范式里被找回来？

对应学习计划：第 1 周。交付物：三范式架构图 + 与 VLM 阶段 LLaVA 图的逐项对照清单（同构点与差异点）。

---

## 1. 第一性原理：语音进 LLM 的根本矛盾

### 1.1 矛盾定义

文本是**离散符号序列**，天然适配 LLM 的词表。语音是**连续的一维时序信号**（16 kHz 时每秒 16000 个采样点），且携带两层信息：

- **语义层**：说了什么（可转写为文字）；
- **副语言层**（paralinguistic）：谁说的、什么情绪、什么语气、环境声是什么。

根本矛盾：**想让 LLM 理解语音，要么先转成文字（丢副语言信息、级联误差），要么把连续信号映射进 LLM 的离散嵌入空间（对齐难、序列长）**。三种范式就是对这一矛盾的三种回答。

### 1.2 三范式速览

| 范式 | 结构 | 信息保真 | 延迟 | 误差传播 |
| --- | --- | --- | --- | --- |
| A：级联包装 | ASR → 文本 → LLM | 丢副语言 | 两段串行相加 | 乘法级联 |
| B：特征注入（类 LLaVA） | 音频塔 → Projector → LLM | 保留 | 单模型 | 端到端可联合优化 |
| C：Speech-to-Speech | 音频 ⇄ 离散语音 token ⇄ LLM | 全保留 | 最低（可全双工） | 端到端，但训练最难 |

---

## 2. 范式 A：级联包装——误差乘法与延迟加法

### 2.1 结构

最简单的方案：把 Stage 2 学的 ASR 当作 LLM 的前端麦克风。

$$
\text{波形} \xrightarrow{\text{ASR}} \hat{w} \xrightarrow{\text{拼进 prompt}} \text{LLM} \to \text{回答}
$$

你 Stage 7 要做的端侧助手，本质就是这一范式的工程化（SenseVoice → Qwen2.5 → TTS）。它的天花板必须提前算清楚。

### 2.2 误差乘法

设 ASR 整句正确率 $P_{\text{ASR}}$，LLM 在正确转写下的回答正确率 $P_{\text{LLM}}$。级联后端到端正确率（近似，假设 ASR 错误必然导致回答错误）：

$$
P_{\text{end2end}} \approx P_{\text{ASR}} \cdot P_{\text{LLM}}
$$

数字感受：ASR 90%、LLM 80% → 端到端只剩 72%。**ASR 的 10% 错误是 LLM 永远无法修复的**——这就是误差传播天花板。特征注入范式让音频特征直接进 LLM，LLM 有机会「结合上下文猜回」ASR 会错的部分，联合训练还能反传梯度给编码器。

### 2.3 延迟加法

级联是串行的：用户说完话后，总等待 = ASR 全部解码 + LLM 首 token。而范式 B 在音频流入时编码器就能并行工作。可运行的量化见第 6 节代码。

### 2.4 丢掉的副语言信息

「我没事」（平静地说）和「我没事！」（带着哭腔），转写完全相同，语义天差地别。级联范式把这一层全部抹掉——这是特征注入范式存在的第二个理由。

### 2.5 级联何时仍是正确答案

级联在技术上「旧」，但工程上有三个不可替代的优点：

1. **可审计**：ASR 文本是天然的日志与安全过滤点，纯语音输出没有中间态可查；
2. **模块可替换**：ASR 与 LLM 可各自独立升级、端侧可自由混搭（你 Stage 7 的 SenseVoice + Qwen2.5 正是这个组合）；
3. **数据门槛低**：文本指令数据远比语音指令数据丰富——语音指令数据的稀缺，正是第 5-6 周要动手补的课。

所以不要预先判定范式优劣：**云端理解用 B/C，端侧产品用 A**——这个结论会在 Stage 7 再验证一次。

---

## 3. 范式 B：音频特征注入——把你学过的 LLaVA 听一遍

### 3.1 结构同构

$$
\text{波形} \xrightarrow{\text{音频塔}} \mathbf{h}_a \xrightarrow{\text{Projector}} \mathbf{z} \xrightarrow{\text{占位符替换}} \text{LLM}
$$

你在 VLM 阶段对这条链路做过维度追踪（CLIP → MLP Projector → LLaMA）。音频版**逐项同构**：

| LLaVA（你已学） | Speech-LLM 对应 | 备注 |
| --- | --- | --- |
| Vision Tower（CLIP ViT） | 音频塔（Whisper encoder / BEATs） | 都是冻结或弱微调的预训练编码器 |
| 图像 → 576 个 patch token | 音频 → $T$ 帧特征（50 Hz × 秒数） | 音频是 1D 时序，长度随语音时长**无上限** |
| MLP Projector $W \in \mathbb{R}^{d_{enc} \times d_{LLM}}$ | 同款 MLP，或 Q-Former | 把编码器空间映射进 LLM 嵌入空间 |
| `<image>` 占位符替换为图像特征 | `<audio>` 占位符替换为音频特征 | 模板拼接逻辑完全一致 |
| 图文对齐预训练（两阶段） | 音频-文本对齐预训练 | 第一阶段只训 Projector |

**差异点（面试必答）**：图像是二维、定长（固定分辨率 → 固定 576 token）；音频是一维时序，**帧率敏感**——30 秒音频按 50 Hz 就是 1500 token，直接塞进 LLM 上下文会爆炸。所以音频侧的 Projector 额外承担**降采样**职责（这是与视觉侧最大的工程差异）。

### 3.2 Qwen-Audio：多任务音频塔 + 单层 cross-attention 采样

Qwen-Audio（阿里，*Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models*，2023）的音频理解范围远超语音：语音、环境声、音乐都能问答。

- **音频塔**：Whisper-large-v2 的 encoder（约 640M），输出 50 Hz、1280 维特征；
- **Projector**：单层 cross-attention（交叉注意力）采样器——用**可学习查询向量**对音频特征做注意力池化，把 1 秒 50 帧压到约 12.5 个 token（4 倍降采样）。这与 BLIP-2 的 Q-Former 思想同源，你在 VLM 阶段见过；
- **多任务联合**：ASR、音频描述（AudioCaps）、声音事件分类（AudioSet）、音乐理解、语音情感识别，统一成「音频 + 任务指令 → 文本」格式联合训练；
- **谱系**：Qwen-Audio → Qwen2-Audio（编码器不变，指令跟随更强，Stage 6 第 5-6 周微调就用它）。

### 3.3 SALMONN：双编码器 + 窗口级 Q-Former

SALMONN（字节 + 清华，*SALMONN: Towards Generic Hearing Abilities for Large Language Models*，2023）回答的是另一个问题：**语音和非语音（环境声、音乐）用同一个编码器不够好，怎么办？**

- **双编码器**：Whisper encoder（擅长语音语义）+ BEATs（自监督音频事件编码器，擅长非语音声音）并行提取，特征拼接；
- **Window-level Q-Former**：把音频按时间窗切块，每窗用一组可学习查询向量做 cross-attention 压缩（窗口内因果，保证可流式扩展）；
- **三阶段训练**：① 对齐（训 Q-Former + 线性层）→ ② 指令微调 → ③ 激活（用带噪声/音乐的数据激活推理与鲁棒能力，论文称之为 activation tuning）。

**双编码器的启示**：单塔覆盖不了所有听感任务时，就并联多个专家塔再融合——这个「多塔 + 融合」模式你在 VLM 的多分辨率方案里也见过。

### 3.4 为什么必须降采样（定量）

$$
\text{token 数} = \text{时长(s)} \times \text{帧率(Hz)} \div \text{降采样倍率}
$$

30 秒音频：50 Hz 不降采样 = 1500 token；4 倍降采样 = 375 token。LLM 上下文是稀缺资源（尤其端侧），且注意力计算 $O(L^2)$ 随序列长度平方增长——**音频 token 率是 Speech-LLM 的第一性能变量**。第 4 周的 codec 深挖会从生成侧再次撞上这个问题。

---

## 4. 范式 C：Speech-to-Speech——LLM 直接开口

范式 B 让 LLM 听懂了，但输出仍是文字（还要外接 TTS）。范式 C 让 LLM 直接生成**离散语音 token**，再由声码器还原为波形——你在 Stage 5 第 2 周学过的 EnCodec/RVQ 在这里登场：

$$
\text{音频} \xrightarrow{\text{codec 编码}} \text{语音 token} \xrightarrow{\text{LLM 建模}} \text{语音 token} \xrightarrow{\text{codec 解码}} \text{波形}
$$

- **Moshi**（Kyutai）：真正的**全双工**——同一条音频流里同时建模「用户声道」和「模型声道」，模型边听边说，没有轮流；第 3 周细拆；
- **GLM-4-Voice**、**Qwen2.5-Omni 的 Talker**：半双工但端到端，第 2 周细拆。

范式 C 的独特价值：**副语言信息可以生成**。级联 + 外挂 TTS 的回答永远是「播音腔」；端到端模型可以带着笑声、叹息、犹豫的语气回答。代价是训练数据（成对的语音对话）极其稀缺，这是它至今不是主流交付形态的原因。

---

## 5. 三范式定位总结

```
级联包装：工程最快、可控性最强、误差乘法          → 端侧产品首选（Stage 7）
特征注入：理解侧最优、多任务广度最大              → 音频理解模型主流（Qwen2-Audio）
Speech-to-Speech：延迟最低、副语言可生成、训练最难 → 交互终局形态（Moshi/Omni）
```

三者不是替代关系：**Qwen2.5-Omni 的理解侧是范式 B，生成侧是范式 C，而你的端侧助手是范式 A 的极致工程化**。同一个知识体系的三个切面。

---

## 6. 实现与验证

### 6.1 级联误差乘法模拟

```python
import numpy as np

rng = np.random.default_rng(0)
N = 20000           # 模拟样本数
p_asr, p_llm = 0.90, 0.80

# ASR 阶段：每句以 p_asr 概率转写正确
asr_correct = rng.random(N) < p_asr
# LLM 阶段：转写正确时以 p_llm 答对；转写错误时假设必然答错
answer_correct = asr_correct & (rng.random(N) < p_llm)

cascade_acc = answer_correct.mean()
oracle_acc = p_llm                      # 若 ASR 完美，上界就是 p_llm
print(f"级联端到端正确率: {cascade_acc:.3f} (理论 {p_asr*p_llm:.3f})")
print(f"误差传播损失: {oracle_acc - cascade_acc:.3f}")
assert abs(cascade_acc - p_asr * p_llm) < 0.02, "模拟应与理论乘法一致"
print("级联误差乘法验证通过 ✓")
```

**预期输出**：端到端正确率 ≈ 0.72，损失 ≈ 0.08——ASR 的 10% 错误在级联里不可修复。

### 6.2 音频 Projector：MLP 版与 Q-Former 版（纯 PyTorch）

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
B, T, d_enc, d_llm = 1, 1500, 1280, 896   # 30s×50Hz 音频，Whisper 1280 维

audio_feats = torch.randn(B, T, d_enc)    # 音频塔输出

# ---- 方案一：MLP Projector（LLaVA 同款，不降采样）----
mlp_proj = nn.Sequential(nn.Linear(d_enc, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm))
z_mlp = mlp_proj(audio_feats)
assert z_mlp.shape == (B, T, d_llm)       # token 数不变：1500 个

# ---- 方案二：cross-attention 采样器（Qwen-Audio 式，降采样 4×）----
class AudioSampler(nn.Module):
    """每 4 帧音频用 1 个可学习查询做 cross-attention，输出 1 个 LLM token。"""
    def __init__(self, d_enc, d_llm, downsample=4):
        super().__init__()
        self.k = downsample
        self.q = nn.Parameter(torch.randn(1, 1, d_llm) * 0.02)
        self.Wq = nn.Linear(d_llm, d_llm)
        self.Wkv = nn.Linear(d_enc, d_llm)   # 音频特征 -> K=V
        self.out = nn.Linear(d_llm, d_llm)

    def forward(self, h):                    # h: (B, T, d_enc)
        B, T, _ = h.shape
        T2 = T // self.k
        q = self.Wq(self.q).expand(B, T2, -1)          # (B, T/4, d_llm)
        k = v = self.Wkv(h.reshape(B * T2, self.k, -1))  # (B*T2, 4, d_llm)
        attn = torch.softmax(q.reshape(B*T2, 1, -1) @ k.transpose(-1, -2)
                             / (q.shape[-1] ** 0.5), dim=-1)
        return self.out((attn @ v).reshape(B, T2, -1))   # (B, T/4, d_llm)

sampler = AudioSampler(d_enc, d_llm, downsample=4)
z_s = sampler(audio_feats)
assert z_s.shape == (B, T // 4, d_llm), "降采样后应为 375 个 token"
print(f"MLP 方案: {z_mlp.shape[1]} token | 采样器方案: {z_s.shape[1]} token")
print("两种 Projector 形状验证通过 ✓")
```

**预期输出**：`MLP 方案: 1500 token | 采样器方案: 375 token`。对照第 3.1 节表格：方案一 = LLaVA 的 MLP 原样搬来；方案二 = Qwen-Audio 的 cross-attention 采样，**音频侧特有的降采样需求**就体现在这里。

### 6.3 上下文预算计算器（MVP 用）

```python
def audio_token_budget(duration_s, hz=50, downsample=4, text_tokens=128, ctx=4096):
    n = int(duration_s * hz / downsample) + text_tokens
    assert n < ctx, f"超出上下文: {n} >= {ctx}"
    return n

# Qwen2-Audio 支持最长 30s 音频；算一下 4× 降采样后的占用
print("30s 音频占用:", audio_token_budget(30), "token")   # 375 + 128 = 503
print("剩余上下文:", 4096 - audio_token_budget(30), "token")
```

**预期输出**：503 / 3593。不降采样则是 1628，剩余上下文直接砍半——这就是「帧率敏感」的代价。

### 6.4 级联延迟加法模拟

```python
def cascade_vs_parallel(asr_ms, llm_ms):
    """级联：LLM 必须等 ASR 全部解码完；特征注入：编码器与 LLM 流水重叠。"""
    cascade = asr_ms + llm_ms                 # 串行相加
    overlap_encoder_ms = asr_ms * 0.7         # 特征注入下编码器工作大部分被重叠
    parallel = overlap_encoder_ms + llm_ms    # 粗估：重叠掉编码器主体耗时
    return cascade, parallel

c, p = cascade_vs_parallel(150, 400)
print(f"级联首响应 {c} ms | 特征注入约 {p:.0f} ms")
assert c > p, "级联延迟必然更高"
print("延迟加法验证通过 ✓")
```

**预期输出**：550 ms vs ~505 ms。真实重叠收益取决于实现，但「加法」性质不变——第 2 周的流式双轨流水线会把重叠做到极致。

---

## 7. 工程权衡与失效模式

### 7.1 权衡

- **级联的确定性**：ASR 输出可审计、可替换、出错可定位——产品级系统（尤其端侧）首选；特征注入是黑盒，错了不知道是编码器还是 LLM 的锅。
- **降采样倍率**：压得越狠上下文越省，但细粒度时间信息（精确时间戳、快速连续指令）丢失越多。Qwen-Audio 的 4× 是理解任务的甜点；ASR 子任务要时间戳时必须保留更高分辨率。
- **单塔 vs 双塔**：SALMONN 双编码器能力覆盖广，但参数量、对齐复杂度翻倍；Qwen-Audio 单塔靠数据规模取胜。
- **冻结音频塔与否**：冻结省显存、防灾难性遗忘；解冻（第二阶段轻微微调）能提升对齐精度——与 VLM 阶段 ViT 冻结策略的权衡一模一样。

### 7.2 失效模式

1. **长音频 token 爆炸**：忘记降采样或超过模型支持时长 → 截断或 OOM。症状：长音频回答驴唇不对马嘴（被截断）；定位：打印实际 token 数；修复：前端 VAD 分段（你 Stage 2 处理过 Whisper 30s 窗口的同类问题）。
2. **音频幻觉**：低质量/静音音频下，LLM「脑补」出并不存在的内容（与 Whisper 长音频幻觉同源）。修复：VAD 前置过滤 + 训练时加入负样本（「这段音频里有什么？」→「无有效语音」）。
3. **副语言信息在 Projector 处漏掉**：对齐阶段只用 ASR 数据训练，Projector 学会只传语义、丢掉情绪信息。症状：情感识别子任务准确率显著低于编码器探针（probe）上限；修复：对齐数据混入情感/事件描述任务（Qwen-Audio、SALMONN 的多任务配方）。
4. **级联边界的延迟尖峰**：ASR 遇到长句时整句解码慢，LLM 只能干等。症状：偶发的超高首字延迟；修复：流式 ASR 分段喂给 LLM，或换 NAR 模型（Stage 2 结论复用）。

---

## 8. 延伸思考题（含解析)

**Q1**：级联范式误差乘法 $P_{\text{ASR}} \cdot P_{\text{LLM}}$ 在什么假设下成立？现实中会更乐观还是更悲观？
**A**：假设「ASR 错 → 回答必错」。现实中略乐观：LLM 有语言先验，个别转写错误能靠上下文纠正；但也可能更悲观：转写错误把 LLM 带偏到自信的错误答案。所以该式是量级估计而非精确值——工程上用真实测试集测端到端准确率。

**Q2**：为什么说音频塔与 Vision Tower 同构，但 Projector 不同构？
**A**：同构在「冻结预训练编码器 + 可学习映射层 + 占位符替换」三件套；不同构在音频是 1D 时序、长度随时长无上限，Projector 必须兼任降采样（cross-attention 采样/池化），而图像固定分辨率下 token 数恒定，MLP 直投即可。

**Q3**：Qwen-Audio 用单层 cross-attention 采样器而不用更深的 Q-Former，可能出于什么考虑？
**A**：单层参数少、不易过拟合、梯度直达音频塔便于联合微调；且 4× 降采样的信息压缩任务并不复杂，深 Q-Former 的收益有限而训练成本翻倍。这与 BLIP-2 需要重 Q-Former 弥合「冻结 ViT 与冻结 LLM」的鸿沟不同——Qwen-Audio 的 LLM 侧训练更充分。

**Q4**：SALMONN 为什么需要 BEATs 这个第二编码器？只用 Whisper 不行吗？
**A**：Whisper 在海量语音上训练，特征聚焦「说了什么」；环境声（狗叫、玻璃碎）、音乐结构等事件信息不在其分布内。BEATs 在 AudioSet 上自监督训练，补上非语音声音的表征。双塔拼接 = 能力并集。

**Q5**：Speech-to-Speech 范式为什么至今没有成为主流交付形态？
**A**：① 训练数据稀缺——成对的「语音问-语音答」远少于文本数据，多数系统仍要借道文本做监督；② 语音 token 序列极长（第 4 周细算），训练推理都贵；③ 文本输出可审计可控，纯语音输出难以做安全过滤与日志追溯。Omni 模型的折中是 Thinker 出文本、Talker 出语音（第 2 周）。

---

## 9. 本周附录：音频理解的「广度」一览

学习计划里强调：音频不只是语音。三大非语音子任务与对应数据，决定了音频塔的「通才」程度：

| 子任务 | 代表数据集 | 任务形态 |
| --- | --- | --- |
| 音频描述（Audio Captioning） | AudioCaps、Clotho | 听一段环境声 → 文字描述 |
| 声音事件分类 | AudioSet（527 类） | 狗叫/玻璃碎/警笛 → 类别 |
| 音乐理解 | MusicCaps、MTG-Jamendo | 风格/情绪/乐器问答 |

Qwen-Audio 与 SALMONN 的多任务预训练都覆盖了这三类——这也是「通用听觉」（generic hearing）与「只做语音」的分界线。你的端侧助手只需要语音能力，但面试谈音频理解广度时，这三类是必答点。

---

## 本周交付清单

- [ ] 画出三范式架构图（级联 / 特征注入 / Speech-to-Speech），标注信息流与延迟特征。
- [ ] 把 VLM 阶段的 LLaVA 图与本周特征注入图并排放，完成逐项对照清单（≥5 个同构点 + ≥3 个差异点）。
- [ ] 跑通 6.1 / 6.2 / 6.3 三段代码，断言全部通过。
- [ ] 查表记录：Qwen2-Audio 音频塔帧率、降采样倍率、最长支持音频时长（写进对照清单）。
- [ ] 能口述：误差乘法、延迟加法、token 爆炸三个天花板的数字。
