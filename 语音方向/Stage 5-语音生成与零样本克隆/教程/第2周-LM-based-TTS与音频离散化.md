# 第 2 周教程：LM-based TTS 与音频离散化——语音作为"语言"

> **本周要回答的三个问题**
> 1. VALL-E 的核心洞察是什么？把语音离散成 token 后，TTS 如何变成语言建模问题？
> 2. EnCodec 的 RVQ（残差向量量化）是什么？多层 codebook 各建模什么？比特率怎么算？
> 3. 为什么工业界从 VALL-E 的 AR+NAR 路线转向 flow matching？

对应学习计划：第 2 周。交付物：用音频 codec 把语音离散成 token，打印每层 codebook 的序列长度与词表大小，计算"1 秒语音 = 多少 token"，并用 LLM 计算该 token 序列的困惑度。

---

## 1. 第一性原理：语音可以当"语言"来建模

### 1.1 VALL-E 的核心洞察

传统 TTS 生成的是**连续**的梅尔谱。VALL-E（Neural Codec Language Models are Zero-Shot TTS）的洞察：

> 如果把语音编码成**离散的 codec token**，那么 TTS 就变成了一个标准的**语言建模问题**——给定文本 token，自回归地预测语音 token。

$$
\text{文本 token} \;\to\; \text{语音 codec token} \;\xrightarrow{\text{codec 解码}}\; \text{波形}
$$

这让 TTS 直接复用 LLM 的全部技术：大规模预训练、in-context learning、zero-shot 能力。

### 1.2 为什么这能实现零样本克隆

VALL-E 把零样本克隆变成 **in-context learning**：

$$
\text{prompt：参考音频的 codec token} + \text{目标文本} \;\to\; \text{目标语音的 codec token}
$$

模型见过海量"同一说话人不同句子"的数据，学会了"参考音频 → 该音色"的映射。给它 3-10 秒参考音频作为 prompt，它就能续写出同样音色的新内容——**不需要任何微调**。

这就是零样本克隆的通用配方：

$$
\text{参考音频} \xrightarrow{\text{离散化}} \text{prompt token} \;\to\; \text{条件生成目标语音}
$$

---

## 2. 音频离散化：EnCodec 与 RVQ

### 2.1 Neural Audio Codec 是什么

音频 codec（编解码器）把波形压缩成紧凑的离散表示，再重建波形。EnCodec / SoundStream / DAC 都是这类。结构：

$$
\text{波形} \xrightarrow{\text{Encoder}} \text{连续特征} \xrightarrow{\text{RVQ}} \text{离散 token} \xrightarrow{\text{Decoder}} \text{重建波形}
$$

### 2.2 RVQ（Residual Vector Quantization，残差向量量化）

这是离散化的核心，也是与"量化方向"联动的关键。

**普通向量量化（VQ）**：把连续向量映射到最近的码本（codebook）向量。但单个码本表达力有限。

**RVQ 的递进思想**：第一层量化后，对**残差**（原向量 - 第一层重建）再量化，层层递进：

$$
r_0 = z \quad \text{（原始连续特征）}
$$
$$
c_1 = \text{Quantize}(r_0), \quad r_1 = r_0 - c_1
$$
$$
c_2 = \text{Quantize}(r_1), \quad r_2 = r_1 - c_2
$$
$$
\cdots
$$

最终重建 = 各层码本向量之和：

$$
\hat{z} = c_1 + c_2 + \cdots + c_N
$$

**每层 codebook 建模什么**：
- **第 1 层**：最主要的信息——**语义/内容**（说了什么）；
- **后续层**：逐层细化的**声学细节**（音色、音质）。

**这个分层结构与量化方向的联系**：
- RVQ 是**向量量化**（把向量映射到码本）；
- GPTQ/AWQ 是**标量量化**（把单个权重映射到离散电平）；
- 两者互为镜像：都是"用离散近似连续"，只是量化的对象和粒度不同。

### 2.3 比特率计算（本周自测题）

EnCodec 的关键参数：
- 采样率 24 kHz；
- 帧率 75 Hz（每秒 75 个时间步）；
- 8 层 RVQ，每层 codebook 大小 1024（$2^{10}$，即每个时间步每层 10 bit）。

**计算 1 秒语音的 token 数**：

$$
75 \text{ 帧/秒} \times 8 \text{ 层} = 600 \text{ token/秒}
$$

**计算比特率**：

$$
75 \times 8 \text{ 层} \times 10 \text{ bit} = 6000 \text{ bit/s} = 6 \text{ kbps}
$$

对比：无损音频约 1411 kbps（CD），EnCodec 用 6 kbps 就能重建出高质量语音——压缩比惊人。这正是"语音可以被极度压缩成离散表示"的证据，也是 LM-based TTS 可行的基础。

**不同码率**：EnCodec 支持 1.5/3/6/12/24 kbps，通过**只用前几层 RVQ** 实现（码率越低用越少层，音质越差）。

---

## 3. VALL-E 的两段式架构

### 3.1 AR + NAR

VALL-E 把 8 层 RVQ 分两阶段生成：

1. **AR（自回归）模型**：生成**第 1 层** codec token（沿时间自回归）。第 1 层承载语义，决定"说什么、什么韵律"；
2. **NAR（非自回归）模型**：以第 1 层 + 文本为条件，**并行**生成第 2-8 层（声学细节）。

### 3.2 为什么这样分

- 第 1 层（语义）需要时序建模 → 适合自回归；
- 后续层（声学细节）依赖第 1 层但彼此并行 → 适合非自回归。

**这个"先语义后声学"的两段式，是后续所有 LM-based TTS 的共同骨架**（CosyVoice、MaskGCT 都是变体）。

### 3.3 为什么工业界转向 flow matching

VALL-E 系的问题：
- AR 解码慢、有误码累积；
- codec token 是**声学重建导向**的，语义信息不够干净。

后续方案（MaskGCT、Seed-TTS、CosyVoice）的改进：
- 用**语义 token**（从 ASR encoder 提取，更干净的"内容"表示）替代纯声学 codec token；
- 用 **flow matching / diffusion** 生成声学细节，质量更稳、可控性更好。

这为第 3-4 周的 CosyVoice 拆解铺路：**语义 token + flow matching** 是当前工业 TTS 的主流配方。

---

## 4. 实现与验证（交付核心）

### 4.1 用 EnCodec 离散化语音

```bash
pip install encodec soundfile
```

```python
import torch
import soundfile as sf
import numpy as np
from encodec import EncodecModel
from encodec.utils import convert_audio

# 加载 EnCodec 24kHz 模型
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)        # 6 kbps -> 用 8 层中的前几层

# 载入音频并适配
wav, sr = sf.read("my_voice.wav", dtype="float32")
if wav.ndim == 1:
    wav = wav[None, :]                  # (1, N)
wav_t = torch.from_numpy(wav).unsqueeze(0)   # (1, 1, N)
wav_t = convert_audio(wav_t, sr, model.sample_rate, model.channels)

with torch.no_grad():
    encoded_frames = model.encode(wav_t)

# 提取 codes: (batch, layers, time)
codes = torch.cat([f[0] for f in encoded_frames], dim=-1)
print("codes shape:", codes.shape)      # (1, n_layers, T)

n_layers, T = codes.shape[1], codes.shape[2]
duration = len(wav) / sr
tokens_per_sec = T / duration
print(f"时长 {duration:.2f}s, 帧率 {tokens_per_sec:.1f} Hz")
print(f"层数: {n_layers}, 每层词表: 1024")
print(f"1 秒语音 = {tokens_per_sec:.0f} 帧 × {n_layers} 层 = {tokens_per_sec*n_layers:.0f} token")
```

**预期**：24 kHz、6 kbps 下，帧率约 75 Hz，8 层 → 1 秒约 600 token。

### 4.2 验证"第 1 层是语义、后续层是音质"

```python
# 只用第 1 层解码：内容对但音色丢失
with torch.no_grad():
    # 只保留第 1 层，其余置零/丢弃（按 API 调整）
    first_layer_only = codes[:, :1, :]
    # 重建（需要把单层的 codes 喂回 decoder）
    # reconstructed = model.decode(...)
print("对比：全 8 层重建（完整音质） vs 仅第 1 层重建（内容对、音色丢）")
```

**观察点**：仅用第 1 层重建，能听出"说了什么"，但音色、音质明显丢失——**直接证明第 1 层承载语义、后续层承载声学细节**。这是本周最有价值的实验。

### 4.3 用 LLM 算语音 token 的困惑度（进阶）

```python
# 把第 1 层 token 序列当作"语言"，用一个小型 LM 计算困惑度
# 目的：直观感受"语音作为语言"的可建模性
# 若困惑度显著低于随机（词表 1024 的随机 PPL=1024），
# 说明语音 token 序列有很强的可预测结构 -> 可以用 LM 建模
```

**预期**：语音第 1 层 token 的困惑度远低于 1024（随机基线），说明它有强结构、可被语言模型预测——这就是 LM-based TTS 可行的量化证据。

---

## 5. 工程权衡与失效模式

### 5.1 权衡

- **码率**：高（12/24 kbps）音质好但 token 多、LM 负担重；低（1.5 kbps）省但音质差；
- **语义 token vs 声学 codec token**：语义干净利于内容控制，声学完整利于音质重建；
- **AR vs NAR**：AR 韵律自然但慢，NAR 快但需条件。

### 5.2 失效模式

1. **AR 误码累积**：自回归生成中一个错误传播到后续。修复：温度回退、NAR 声学层。
2. **音色泄漏**：prompt 太短或音色特征不足，克隆不像。修复：加长参考音频、用说话人 embedding 增强。
3. **内容漂移**：生成的语音说了与文本不符的内容。修复：更强的文本-语音对齐、约束解码。
4. **codec 重建上限**：再好的 LM 也超不过 codec 的重建质量。修复：用更高码率/更好的 codec。

---

## 6. 延伸思考题（含解析）

**Q1**：EnCodec 24 kHz、8 层 RVQ、每层 1024 词表，1 秒语音的 token 数与比特率？
**A**：75 帧/秒 × 8 层 = 600 token/秒；比特率 = 75 × 8 × 10 bit = 6 kbps。

**Q2**：RVQ 为什么分层？每层在建模什么？
**A**：第 1 层量化主体（语义/内容），后续层逐层量化残差（声学细节、音色）。分层让"内容"与"音质"解耦，可按需取用（低码率只用前几层）。

**Q3**：为什么 CosyVoice 选"语义 token + flow"两段式，而不是 VALL-E 的纯声学 codec？
**A**：语义 token（ASR encoder 提取）比声学 codec 的第 1 层更干净地表达内容，利于可控与对齐；flow matching 生成声学细节质量更稳、可流式。纯声学 codec 的语义不够干净、AR 易误码。

**Q4**：VALL-E 如何实现零样本克隆？为什么不需要微调？
**A**：把克隆变成 in-context learning：参考音频的 codec token 作为 prompt 拼在目标文本前，模型从海量"同人多句"数据中学会了"参考 → 音色"的映射，续写即克隆。无需针对目标人微调。

**Q5**：RVQ 与 GPTQ/AWQ 的关系是什么？（跨方向联动题）
**A**：RVQ 是向量量化（连续向量 → 码本向量之和），GPTQ/AWQ 是标量量化（权重 → 离散电平）。都是"离散近似连续"，互为镜像：一个量化激活/特征的向量，一个量化模型权重。理解其一有助于理解另一个。

---

## 本周交付清单

- [ ] 用 EnCodec 把一段语音离散化，打印层数、帧率、词表、每秒 token 数。
- [ ] 验证第 1 层（语义）与全层（音质）的重建差异，试听记录。
- [ ] 手推 6 kbps 的 token 数与比特率计算。
- [ ] 能解释：RVQ 分层、VALL-E 的 AR+NAR、为何转向 flow matching。
