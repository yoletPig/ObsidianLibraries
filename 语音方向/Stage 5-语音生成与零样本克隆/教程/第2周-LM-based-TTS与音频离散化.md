# 第 2 周教程：LM-based TTS 与音频离散化——语音作为"语言"

> **本周要回答的三个问题**
> 1. VALL-E 的核心洞察是什么？把语音离散成 token 后，TTS 如何变成语言建模问题？
> 2. EnCodec 的 RVQ（残差向量量化）是什么？多层 codebook 各建模什么？比特率怎么算？
> 3. 为什么工业界从 VALL-E 的 AR+NAR 路线转向 flow matching？

对应学习计划：第 2 周。交付物：用音频 codec 把语音离散成 token，打印每层 codebook 的序列长度与词表大小，计算"1 秒语音 = 多少 token"，并用 LLM 计算该 token 序列的困惑度。

---

## 1. 第一性原理：语音可以当"语言"来建模

### 1.1 VALL-E 的核心洞察

传统 TTS 生成的是**连续**的梅尔谱。VALL-E（Wang et al., Microsoft, 2023, arXiv:2301.02111，全名 *Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers*）的洞察：

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

模型见过海量"同一说话人不同句子"的数据，学会了"参考音频 → 该音色"的映射。给它几秒参考音频（论文用 3 秒录音）作为 prompt，它就能续写出同样音色的新内容——**不需要任何微调**。

这就是零样本克隆的通用配方：

$$
\text{参考音频} \xrightarrow{\text{离散化}} \text{prompt token} \;\to\; \text{条件生成目标语音}
$$

---

## 2. 音频离散化：EnCodec 与 RVQ

### 2.1 Neural Audio Codec 是什么

音频 codec（编解码器）把波形压缩成紧凑的离散表示，再重建波形。代表工作：**SoundStream**（Zeghidour et al., Google, 2021，首次把 RVQ 用进神经音频编解码）、**EnCodec**（Défossez et al., Meta, 2022）、**DAC / Descript Audio Codec**（Kumar et al., Descript, 2023，改进 RVQGAN，主打 44.1 kHz 高保真）。结构都是同一副骨架：

$$
\text{波形} \xrightarrow{\text{Encoder}} \text{连续特征} \xrightarrow{\text{RVQ}} \text{离散 token} \xrightarrow{\text{Decoder}} \text{重建波形}
$$

各环节的分工：Encoder 是个下采样卷积网络，把每秒几万个采样点压成每秒几十~上百个连续向量；RVQ 把每个向量变成一小组离散码；Decoder 是上采样网络，把离散码还原回波形——**端到端训练，目标是"重建出来的波形听不出和原声的差别"**。

### 2.2 RVQ（Residual Vector Quantization，残差向量量化）

这是离散化的核心，也是与"量化方向"联动的关键。

**普通向量量化（VQ）**：把连续向量 $z$ 映射到码本（codebook）里**最近**的那个向量。**这个操作在干什么**：相当于"只允许用字典里的词说话"——$z$ 是千变万化的连续点，码本是有限的 $K$ 个"标准点"，量化 = 找到离你最近的标准点并记住它的编号（编号就是 token）。**代价**：最近的替代点与原点的距离就是量化误差；单个码本（哪怕 1024 个向量）表达力有限，误差太大。

**RVQ 的递进思想**：一次量化不准，就对**误差**再量化一次，层层套娃，误差越滚越小：

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

**逐符号拆解**：$r_i$ 是第 $i$ 步开始时"还没被表达出来的误差"；$\text{Quantize}(\cdot)$ 是"在码本里找最近向量"；$c_i$ 是这一层选中的码字（它的**索引**才是存下来的 token）；$r_{i} = r_{i-1} - c_i$ 把已表达的部分减掉，剩下的残差交给下一层。**因为 $r_0 = r_1 + c_1$ 恒成立、逐层展开，所以 $N$ 层之后 $z = (c_1 + \cdots + c_N) + r_N$——只要层数够多，误差 $r_N$ 想多小有多小。**

**手算小例**（把向量简化成一维标量感受一下）：设 $z = 7.3$，每层码本粒度逐层变细——第 1 层码本是整数，第 2 层是 $\{0, \pm 0.25, \pm 0.5, \dots\}$，第 3 层是 $\{0, \pm 0.05, \dots\}$：

| 层 | 输入残差 $r_{i-1}$ | 选中码字 $c_i$ | 剩余残差 $r_i$ |
| --- | --- | --- | --- |
| 1 | 7.3 | 7 | 0.3 |
| 2 | 0.3 | 0.25 | 0.05 |
| 3 | 0.05 | 0.05 | 0.0 |

三层之后重建 $\hat{z} = 7 + 0.25 + 0.05 = 7.3$，误差归零；只用第 1 层则差 0.3。**这就是"套娃"：每层只负责修正上一层没画好的部分，层码本不需要表达"全世界"，只需要表达"剩下的误差"。**

**每层 codebook 大致在建模什么**（经验规律，不是严格定理）：
- **第 1 层**：主体信息——**内容与韵律骨架**（说了什么、怎么抑扬顿挫）；
- **后续层**：逐层细化的**声学细节**（音色纹理、相位、音质）。

注意：这是 EnCodec 这类**声学重建导向** codec 的经验分层，第 1 层仍是"声学 token"而非干净的语义表示——它混合了内容与音色。这正是第 3-4 周要讲的 CosyVoice 改用"监督语义 token"的动机。

> **动手验证**：2.3 节 4.2 小节的实验就是这个分层的直接证据——只用第 1 层重建 vs 全层重建，用耳朵对比。

**这个分层结构与量化方向的联系**：
- RVQ 是**向量量化**（把向量映射到码本）；
- GPTQ/AWQ 是**标量量化**（把单个权重映射到离散电平）；
- 两者互为镜像：都是"用离散近似连续"，只是量化的对象和粒度不同。

### 2.3 比特率计算（本周自测题）

EnCodec 24 kHz 模型的关键参数（Défossez et al., 2022）：
- 采样率 24 kHz，**单声道**；
- 帧率 75 Hz（每秒 75 个时间步，encoder 总下采样步长 320）；
- RVQ **最多 32 层**，每层 codebook 大小 1024（$2^{10}$，即每个时间步每层 10 bit）；
- 支持的码率档位：1.5 / 3 / 6 / 12 / 24 kbps，**分别对应只启用前 2 / 4 / 8 / 16 / 32 层**。

**计算 1 秒语音的 token 数（6 kbps 档 = 8 层，即 VALL-E 用的配置）**：

$$
75 \text{ 帧/秒} \times 8 \text{ 层} = 600 \text{ token/秒}
$$

**计算比特率**：

$$
75 \times 8 \text{ 层} \times 10 \text{ bit} = 6000 \text{ bit/s} = 6 \text{ kbps}
$$

对比：无损 CD 音频约 1411 kbps，EnCodec 用 6 kbps 就能重建出高质量语音——压缩比超过 200 倍。这正是"语音可以被极度压缩成离散表示"的证据，也是 LM-based TTS 可行的基础。

| 码率 | 启用层数 | 1 秒 token 数 |
| --- | --- | --- |
| 1.5 kbps | 2 | 150 |
| 3 kbps | 4 | 300 |
| 6 kbps | 8 | 600 |
| 12 kbps | 16 | 1200 |
| 24 kbps | 32 | 2400 |

> **一句话总结**：token 越多 = 比特率越高 = 音质越好 = 后续语言建模负担越重。VALL-E 选 8 层，就是在"音质"与"序列长度"之间取的折衷。

---

## 3. VALL-E 的两段式架构

### 3.1 AR + NAR

VALL-E 用 EnCodec 的**前 8 层** RVQ（即 6 kbps 档配置），把生成分两阶段：

1. **AR（自回归）模型**：沿时间轴逐帧生成**第 1 层** codec token。第 1 层承载内容与韵律骨架，决定"说什么、什么节奏"；
2. **NAR（非自回归）模型**：以第 1 层 + 文本为条件，生成第 2–8 层（声学细节）。**注意"非自回归"的方向**：沿**时间**是并行的（一整句的所有帧一次算完），沿**层间**仍是串行的——生成第 $i$ 层时要看着第 $i-1$ 层，一层一层补细节。

### 3.2 为什么这样分

- 第 1 层（内容/韵律）需要长程时序建模 → 适合自回归（和时间轴逐点"接龙"）；
- 后续层（声学细节）给定第 1 层后基本是"逐帧填色"，无长程依赖 → 沿时间并行，速度快。

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

# 加载 EnCodec 24kHz 模型（单声道；首次运行会自动下载权重）
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)        # 6 kbps -> 启用前 8 层 RVQ（档位见 2.3 节表格）

# 载入音频并适配
wav, sr = sf.read("my_voice.wav", dtype="float32")
if wav.ndim == 1:
    wav = wav[None, :]                  # (1, N)
wav_t = torch.from_numpy(wav).unsqueeze(0)   # (B=1, C=1, N)
wav_t = convert_audio(wav_t, sr, model.sample_rate, model.channels)  # 重采样到 24kHz、混缩声道

with torch.no_grad():
    encoded_frames = model.encode(wav_t)   # 返回 list，每项是 (codes, scale)

# 提取 codes: (batch, n_q, time)，n_q = 当前带宽档位启用的层数
codes = torch.cat([f[0] for f in encoded_frames], dim=-1)
print("codes shape:", codes.shape)      # (1, 8, ~75T)

n_layers, T = codes.shape[1], codes.shape[2]
duration = len(wav) / sr
tokens_per_sec = T / duration
print(f"时长 {duration:.2f}s, 帧率 {tokens_per_sec:.1f} Hz")
print(f"层数: {n_layers}, 每层词表: 1024")
print(f"1 秒语音 = {tokens_per_sec:.0f} 帧 × {n_layers} 层 = {tokens_per_sec*n_layers:.0f} token")
```

**预期**：24 kHz、6 kbps 档下，帧率约 75 Hz、8 层 → 1 秒约 600 token；把 `set_target_bandwidth` 改成 1.5，`codes.shape[1]` 会变成 2（2 层）——这就是"低码率 = 少用几层套娃"的直接证据。

### 4.2 验证"第 1 层保内容骨架、后续层保音质"

最省事的对比方法：用带宽档位控制层数（1.5 kbps 档只启用前 2 层，6 kbps 档启用 8 层），同一段音频各重建一次：

```python
import torchaudio

for bw, n_q in [(1.5, 2), (6.0, 8)]:      # 24kHz 模型码率档位 -> 启用层数（见 2.3 节表格）
    model.set_target_bandwidth(bw)
    with torch.no_grad():
        frames = model.encode(wav_t)
        recon = model.decode(frames)          # (1, 1, N)，重建波形
    torchaudio.save(f"recon_{bw}kbps.wav", recon.squeeze(0), model.sample_rate)
    print(f"{bw} kbps 档（启用前 {n_q} 层）：已保存 recon_{bw}kbps.wav")
```

**观察点**：1.5 kbps（2 层）版本能听出"说了什么"、韵律还在，但音质明显劣化、发闷发糊；6 kbps（8 层）版本接近原声。这印证了 2.2 节的分层规律——**前几层承载内容与韵律骨架，后续层补齐音色与音质细节**。注意结论要留有余地：EnCodec 是声学重建导向的，第 1 层并不是"纯语义"，只是信息占比上更偏内容。这是本周最有价值的实验。

### 4.3 用 LLM 算语音 token 的困惑度（进阶）

困惑度（Perplexity, PPL）定义为平均负对数似然的指数：

$$
\text{PPL} = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log p(x_i \mid x_{<i}) \right)
$$

**这个公式在干什么**：把"模型每一步的预测分布"折算成一个"等效候选数"。若模型预测很准（$p \to 1$），$\log p \to 0$，PPL $\to 1$（等于闭卷抄答案）；若模型完全瞎猜（词表 $V=1024$ 上的均匀分布，$p = 1/1024$），PPL $= 1024$——**这就是"随机基线 PPL = 词表大小"的由来**。PPL = 50 的直观含义：模型每一步的不确定性约等于"在 50 个候选里挑"。

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

**Q1**：EnCodec 24 kHz、启用前 8 层 RVQ（即 6 kbps 档）、每层 1024 词表，1 秒语音的 token 数与比特率？
**A**：75 帧/秒 × 8 层 = 600 token/秒；比特率 = 75 × 8 × 10 bit = 6 kbps。注意 8 层不是 EnCodec 的上限——24 kHz 模型最多 32 层（24 kbps），VALL-E 只用前 8 层。

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
