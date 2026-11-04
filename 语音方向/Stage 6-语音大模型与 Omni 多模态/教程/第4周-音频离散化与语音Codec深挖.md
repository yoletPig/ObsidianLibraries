# 第 4 周教程：音频离散化与语音 Codec 深挖——RVQ 与标量量化的镜像

> **本周要回答的四个问题**
> 1. SoundStream / EnCodec / Mimi 三代 codec 的训练目标与关键参数各是什么？
> 2. Semantic RVQ 的设计哲学：为什么「第一层语义、后续层声学」？只用第一层解码为何「内容对但音色丢」？
> 3. RVQ（向量量化）与 GPTQ/AWQ（标量量化）如何互为镜像？面试怎么讲？
> 4. token 率（12.5 / 50 / 75 Hz）如何决定 LLM 的上下文压力与实时性？

对应学习计划：第 4 周。交付物：同一段语音过 EnCodec（1.5/6/12 kbps）与 Mimi，测重建质量，画「码率-质量曲线」；解释语义层单独解码的现象。

参考文献：*SoundStream: An End-to-End Neural Audio Codec*（arXiv:2107.03312）、*High Fidelity Neural Audio Compression*（EnCodec，arXiv:2210.13438）、*Moshi: a speech-text foundation model for real-time dialogue*（arXiv:2410.00037，Mimi 出处）。

---

## 1. 第一性原理：为什么语音必须离散化

### 1.1 两个动机

1. **压缩**：16 kHz 单声道原始音频 = 256 kbps（16 bit 量化）。神经网络 codec 把它压到 1~6 kbps 仍可听懂——压缩比 50~250 倍，证明语音信息高度冗余；
2. **接入 LLM**：LLM 只会预测下一个**离散** token。语音要成为 LLM 的输入输出模态，必须先变成有限词表上的序列——这正是你 Stage 5 第 2 周学过的 VALL-E 前提，本周把它补完整。

### 1.2 重建目标的数学形式

**这个公式在干什么**：它是神经 codec 的「总损失账单」。codec 学习一个编码-解码对：波形 $x$ → 离散码 $q$ → 重建 $\hat{x}$。只看「波形像不像」（纯重建损失）训出来的东西模糊发闷，所以要让多个各管一摊的损失项相加、互相制衡：

$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{rec}}}_{\text{时域+频域重建}} + \underbrace{\mathcal{L}_{\text{adv}} + \mathcal{L}_{\text{fm}}}_{\text{对抗+特征匹配}} + \underbrace{\mathcal{L}_{\text{vq}}}_{\text{量化}}
$$

逐项拆解（$\mathcal{L}_{\text{rec}}$ 管「像」，$\mathcal{L}_{\text{adv}}$ 管「真」，$\mathcal{L}_{\text{vq}}$ 管「贴」）：

- $\mathcal{L}_{\text{rec}}$：波形 L1 + 多分辨率 STFT 幅度谱距离——直接度量重建波形与原波形在时域和多个频域分辨率上的差距，保证「听得出来是原声」；
- $\mathcal{L}_{\text{adv}}$：对抗损失，多尺度 STFT 判别器（SoundStream）/ 多周期多尺度判别器（EnCodec）。L1/STFT 损失倾向于输出「平均解」导致声音发闷，判别器逼着生成器给出「听起来像真录音」的锐利细节——这与 GAN 的博弈同源；
- $\mathcal{L}_{\text{vq}}$：commitment loss，约束编码器输出贴近码本向量（你学过的 VQ-VAE 同款）。因为离散取码本那一步不可导，靠它让编码器「主动凑近」码本，训练才稳定。

### 1.3 RVQ 复习（一分钟，细节见 Stage 5 第 2 周）

**这组公式在干什么**：它定义**残差向量量化（Residual Vector Quantization, RVQ）**——用「多层叠加的近似」把一个连续向量压成有限码字的组合。单个码本容量有限（比如 1024 个码字，10 bit），压 1280 维连续向量误差很大；RVQ 的思路是：第一层给个粗略近似，第二层专门量化「第一层没解释掉的部分」（残差），第三层再解释第二层剩下的……逐层精修。

$$
r_0 = z, \quad c_k = \text{VQ}(r_{k-1}), \quad r_k = r_{k-1} - c_k, \quad \hat{z} = \sum_{k=1}^{N} c_k
$$

符号拆解：$z$ 是编码器输出的连续向量；$r_k$ 是第 $k$ 层量化后**还没被解释的残差**；$\text{VQ}(\cdot)$ 在码本里找最近邻码字 $c_k$；最终重建 $\hat{z}$ 就是全部码字之和。

**类比**：猜价格游戏。出题人心里有个数 $z$（比如 368 元）。第一位竞猜者给粗估 $c_1 = 350$；主持人报出差额「还差 18」；第二位补 $c_2 = 15$；还差 3，第三位补 $c_3 = 3$……每人（每层码本）只需要在各自的小范围里选值，多层之和却能逼近任意精度。这也是为什么「码率档位 = 层数」：砍层就砍精度（第 8 节 Q2 展开）。

误差逐层单调衰减——6.1 节会用纯 NumPy 亲手验证这条性质，它是本周所有实验的地基。

---

## 2. 三代 Codec 对照

| 维度 | SoundStream (2021) | EnCodec (2022) | Mimi (2024) |
| --- | --- | --- | --- |
| 出品 | Google | Meta | Kyutai（Moshi 配套） |
| 采样率/帧率 | 24 kHz / 50 Hz | 24 kHz / 75 Hz | 24 kHz / **12.5 Hz** |
| RVQ | 首次引入音频 codec | 最多 32 层（按码率取 2~32 层），码本 1024 | 8 层，码本 2048 |
| 码率 | 3~18 kbps | 1.5/3/6/12/24 kbps | ≈1.1 kbps（全层） |
| 第一层性质 | 普通声学 | 普通声学（信息量最大） | **语义层**（蒸馏自 WavLM） |
| 流式 | 支持 | 支持（附熵编码 LM） | 全因果，为双工设计 |

**比特率速算公式**（本周必会）：

$$
\text{kbps} = \text{帧率} \times \text{层数} \times \log_2(\text{码本大小}) \;/\; 1000
$$

**这个公式在干什么**：把 codec 的三个设计参数换算成存储/传输压力。每秒产生「帧率」帧，每帧存「层数」个 token，每个 token 是 $\log_2(\text{码本大小})$ bit（码本 1024 → 10 bit，码本 2048 → 11 bit）——三项相乘就是每秒比特数。它与第 1 周的「音频 token 预算公式」是同一件事的两面：一个算 token 数（上下文压力），一个算 bit 数（码率）。

验证：EnCodec 6 kbps = $75 \times 8 \times 10 / 1000 = 6$ ✓；Mimi 全层 = $12.5 \times 8 \times 11 / 1000 = 1.1$ ✓。EnCodec 的低码率档位靠**减少使用的层数**实现：1.5 kbps = 2 层，3 kbps = 4 层，12 kbps = 16 层（正好对应 RVQ「砍层 = 砍精度」）。

**代际主线**：从「声学重建质量」（SoundStream/EnCodec）走向「语义可建模性」（Mimi）——codec 的评价标准从 PESQ 转向「下游 LM 好不好建模」。

---

## 3. Semantic RVQ：第一层为什么必须是语义

### 3.1 Mimi 的关键设计

Mimi 的第一层码本不是普通声学量化，而是加了一个**语义蒸馏损失**：让第一层解码器的输出特征去逼近 WavLM（自监督语音模型）的特征。结果是：

- **第 1 层 ≈ 语义/内容**：承载「说了什么」，丢音色；
- **第 2~8 层 ≈ 声学残差**：音色、音质、细节。

而 Moshi 的建模分工（上周讲过）正好咬合：主 Transformer 沿时间自回归建模第 1 层（语义需要时序因果），Depth Transformer 在每个 80 ms 时间步内串行补全第 2~8 层声学细节。

### 3.2 「内容对但音色丢」的机理

只用第 1 层解码：$\hat{z} = c_1$，声学残差全部置零。重建出的语音保留了字内容与大致韵律（语义层里有），但说话人音色、气息细节全丢——听起来像「内容正确的人声占位符」。这就是本周 MVP 要亲耳验证的现象。

反过来用：**LM-based TTS 先只生成语义层，声学层交给并行模型补全**（VALL-E 的 AR+NAR，Stage 5 学过）——语义/声学分层正是这套两段式范式的物理基础。

### 3.3 EnCodec 的第一层是语义吗？

严格说不是——它只是「信息量最大的一层」（残差分解的第一主项），语义与音色仍混在一起。Mimi 用蒸馏显式解耦，是设计哲学上的跃迁。面试时这个区分是加分点。

---

## 4. 与标量量化互为镜像（跨方向面试弹药）

你在性能优化方向学过 GPTQ/AWQ。把两者并排：

| 维度 | RVQ（音频） | GPTQ/AWQ（权重） |
| --- | --- | --- |
| 量化对象 | 特征**向量**（整块映射到码本） | 单个权重**标量**（映射到电平网格） |
| 码本/网格 | 学习出的码本（2048 个向量） | 均匀网格 + scale/zero-point |
| 误差补偿 | 残差逐层量化（第 $k$ 层量化第 $k-1$ 层残差） | GPTQ 按列顺序量化并用 Hessian 把误差分摊给未量化权重 |
| 比特率-质量曲线 | kbps ↑ → PESQ ↑，边际递减 | bit 数 ↑ → 困惑度 ↓，4 bit 是甜点 |
| 失效模式 | 码本坍缩（大量码字不用） | 离群权重（outlier）压爆网格（AWQ 专门保它们） |

**镜像命题（背诵级）**：两者都是「用离散码逼近连续值 + 显式误差补偿」，区别只在量化单元的粒度——向量级（联合最优，表达力强，码本难训）vs 标量级（独立量化，工程简单，靠误差重分配补精度）。GPTQ 的「逐列量化 + 残差分摊」本质上就是标量版的 RVQ。

---

## 5. Token 率：决定 LLM 上下文的第一变量

**这个公式在干什么**：把「codec 帧率」翻译成 LLM 工程师最关心的量——**一分钟音频要占多少 token**。它是第 1 周「token 预算公式」的生成侧版本，只是这里关注语义流（LLM 逐帧预测的那一层）：

$$
\text{token/min} = \text{帧率(Hz)} \times 60 \times \text{并行层数}
$$

帧率是每秒多少帧 token；乘 60 换到分钟；并行层数看你的 LM 建模几层（Moshi 的 Depth Transformer 里 8 层都建模，但只有第 1 层沿时间展开，所以「上下文压力」通常按语义流一列算）。

| codec | 帧率 | 语义流 token/min | 60 分钟对话占用 |
| --- | --- | --- | --- |
| Mimi（仅语义层） | 12.5 | 750 | 45,000 |
| SoundStream | 50 | 3,000 | 180,000 |
| EnCodec | 75 | 4,500 | 270,000 |

对照：人类正常语速一分钟的**文本**转写约 200~300 个中文 token。也就是说：

- Mimi 12.5 Hz 的语义流约为同等文本的 3 倍——4K 上下文勉强装下 5 分钟对话；
- 50 Hz 不降采样则 15 倍起步——长对话必炸上下文，这就是「为什么 codec 都在卷更低码率」。

可运行计算见第 6.4 节。

---

## 6. 实现与验证

### 6.1 从零实现 RVQ：验证「逐层误差衰减」（纯 NumPy）

**这段代码在干什么**：把 1.3 节的猜价格类比跑成程序——随机造 6 个码本（每个都塞一个零向量，相当于允许「这一层不改」），对一批随机向量逐层量化，记录每加一层后的残差能量。若 RVQ 性质成立，误差必须一层比一层小。

```python
import numpy as np

rng = np.random.default_rng(42)
N, D, K, n_layers = 2000, 32, 256, 6
data = rng.standard_normal((N, D))

# 每个码本含一个零向量（保证"不变更"选项 -> 误差单调不增）
codebooks = []
for _ in range(n_layers):
    cb = rng.standard_normal((K, D)) * 0.5
    cb[0] = 0.0
    codebooks.append(cb)

def vq(x, cb):
    dist = ((x[:, None, :] - cb[None, :, :]) ** 2).sum(-1)
    return dist.argmin(axis=1)

def rvq_error(x, cbs):
    resid, codes = x.copy(), []
    for cb in cbs:
        idx = vq(resid, cb)
        codes.append(idx)
        resid -= cb[idx]
    return float((resid ** 2).mean()), codes

errs = [rvq_error(data, codebooks[:n])[0] for n in range(1, n_layers + 1)]
print("逐层量化误差:", [f"{e:.3f}" for e in errs])
assert all(b < a for a, b in zip(errs, errs[1:])), "RVQ 每加一层误差必须严格下降"
print("RVQ 残差衰减性质验证通过 ✓")
```

**预期输出**（实测）：`0.782 → 0.613 → 0.490 → 0.403 → 0.343 → 0.302`，误差序列严格递减。注意衰减幅度逐层变小——信息大头在第 1 层，后面都是「锦上添花」，这正是 EnCodec/ semantic-codec 设计「第一层承载语义」的统计基础。这也是「低码率=少层数=音质差」的数学根源。

### 6.2 EnCodec 码率扫描：码率-质量曲线（MVP 核心）

```bash
pip install encodec torch
```

```python
import torch
from encodec import EncodecModel

torch.manual_seed(0)
sr = 24000
t = torch.linspace(0, 3, sr * 3)
# 伪语音信号：谐波+颤动（真实实验请换成 3 秒真实语音 wav）
wav = sum(0.4 / k * torch.sin(2 * 3.14159 * 220 * k * t * (1 + 0.01 * torch.sin(2 * 3.14159 * 4 * t)))
          for k in range(1, 8)).unsqueeze(0).unsqueeze(0)

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)

def mel_of(x, n_fft=1024, hop=256, n_mels=80):
    """简易梅尔谱：三角带近似为矩形带，对教学足够（负对数域便于比较）。"""
    mag = torch.stft(x.squeeze(), n_fft, hop_length=hop,
                     window=torch.hann_window(n_fft), return_complex=True).abs()
    mel_pts = torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + sr / 2 / 700)), n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)      # n_mels+2 个边界频率（含两端）
    bins = (hz_pts * n_fft / sr).long()              # 边界频率 -> FFT bin 序号
    fb = torch.zeros(n_mels, mag.shape[0])
    for i in range(n_mels):
        fb[i, bins[i]:bins[i+1]] = 1.0
        fb[i, bins[i+1]:bins[i+2]] = 1.0
    return torch.log10(mag.T @ fb.T + 1e-6)

def mel_distance(bw):
    model.set_target_bandwidth(bw)
    with torch.no_grad():
        [(codes, scale)] = model.encode(wav)
        recon = model.decode([(codes, scale)])
        n = min(wav.shape[-1], recon.shape[-1])
        return float((mel_of(wav[..., :n]) - mel_of(recon[..., :n])).abs().mean()), codes.shape[1]

results = {bw: mel_distance(bw) for bw in [1.5, 6.0, 24.0]}
for bw, (d, nq) in results.items():
    print(f"{bw} kbps -> 层数={nq}, mel距离={d:.3f}")
assert results[1.5][1] == 2 and results[6.0][1] == 8 and results[24.0][1] == 32, "层数-码率映射错误"
assert results[24.0][0] < results[6.0][0] < results[1.5][0], "质量必须随码率单调提升"
print("码率-质量单调性验证通过 ✓（把 (码率, mel距离) 画成曲线即 MVP 图）")
```

**预期输出**（本机实测，CPU 即可跑，首次运行会下载约 89 MB 权重）：

```
1.5 kbps -> 层数=2, mel距离=0.929
6.0 kbps -> 层数=8, mel距离=0.777
24.0 kbps -> 层数=32, mel距离=0.741
```

层数 2/8/32 与公式推算完全一致，mel 距离随码率严格下降。**交付升级**：换成真实 3 秒语音，用 `pesq`（`pip install pesq`）与 UTMOS（`torch.hub` 的 `tarepan/SpeechMOS`）替换 mel 距离，得到正式版「码率-质量曲线」。

### 6.3 Mimi：语义层单独解码实验

```bash
pip install transformers soundfile
# 首次运行会从 HuggingFace 下载官方 transformers 版权重 kyutai/mimi（约 360 MB），CPU 可跑
```

```python
import soundfile as sf
import torch
from transformers import MimiModel

mimi = MimiModel.from_pretrained("kyutai/mimi")
mimi.eval()
sr = mimi.config.sampling_rate            # 24000
wav = torch.randn(1, 1, sr * 2)           # 真实实验换成 2 秒人声

with torch.no_grad():
    codes = mimi.encode(wav).audio_codes  # (1, 8, T)
    T = codes.shape[-1]
    assert codes.shape[1] == 8, "Mimi 应为 8 层 RVQ"
    assert abs(T - 2 * 12.5) <= 1, f"帧率应为 12.5 Hz，实际 {T / 2}"

    # 只用第 1 层（语义层）解码：量化器支持只传前 k 层码字，声学层视为未激活
    recon_sem = mimi.decode(codes[:, :1]).audio_values
    recon_full = mimi.decode(codes).audio_values
    sf.write("recon_sem.wav", recon_sem[0, 0].numpy(), sr)
    sf.write("recon_full.wav", recon_full[0, 0].numpy(), sr)

print(f"2 秒音频 -> {T} 个时间步（12.5 Hz）；对比 recon_sem.wav 与 recon_full.wav 听感")
# 交付：听两段音频，写下「内容保留 / 音色丢失」的主观记录
```

> 若你装的是 Kyutai 官方 `moshi` 包（`pip install moshi`），等价写法是 `from moshi.models import loaders; mimi = loaders.get_mimi(<moshi 格式权重路径>, device="cpu")`，其 `encode` 返回 `(B, 8, T)`、`decode` 同样支持传入部分层码字——两个包的机制一致，本文用 transformers 版因为它有独立的官方权重仓库、下载体积小。

**预期**：`recon_sem.wav` 能听出说了什么（或随机噪声输入时呈现「内容轮廓还在」的占位人声），但音色与 `recon_full.wav` 明显不同——第 3.2 节的机理被你亲耳验证。

### 6.4 Token 率与上下文预算

```python
def tokens_per_minute(hz, streams=1):
    return int(hz * 60 * streams)

assert tokens_per_minute(12.5) == 750     # Mimi 语义流
assert tokens_per_minute(50) == 3000      # SoundStream
assert tokens_per_minute(75) == 4500      # EnCodec

CTX = 4096
for name, hz in [("Mimi语义", 12.5), ("SoundStream", 50), ("EnCodec", 75)]:
    minutes = CTX / tokens_per_minute(hz)
    print(f"{name}: 4K 上下文约可容纳 {minutes:.1f} 分钟音频")
# 预期：约 5.5 / 1.4 / 0.9 分钟 -> 解释"为什么 codec 卷低码率"
```

---

## 7. 工程权衡与失效模式

### 7.1 权衡

- **码率-质量-可建模性三角**：码率低 → 上下文省、LM 好建模，但声学重建差（要靠额外声学模型补）；Mimi 用「语义层极低帧率 + 声学层另建模」拆解了这个三角。
- **帧率 vs 时序精度**：12.5 Hz 意味着一个 token 代表 80 ms 音频——音素级（20~50 ms）的时间细节在第一层已不可恢复，ASR 类任务必须靠编码器特征而非语义 token。
- **对抗训练的稳定性**：判别器让重建自然，但训练不稳时音质崩塌（模式坍缩）；工业实现普遍要精细调判别器权重。

### 7.2 失效模式

1. **码本坍缩**（codebook collapse）：大量码字从不被使用，有效词表缩水，低码率档音质骤降。症状：码字利用率 <50%；修复：EMA 码本更新、死码字重置策略，或改用 FSQ（Finite Scalar Quantization，有限标量量化——免去码本学习与坍缩问题，Stage 5 量化方向知识）。
2. **帧率与采样率错配**：误把 24 kHz 模型当 16 kHz 用，帧率算错 → 下游所有时长估计全错。症状：token 数与音频时长对不上；修复：始终从模型配置读 `frame_rate`，不硬编码。
3. **语义层误解码当完整音频**：拿第 1 层 token 直接给用户听（或当交付物）。症状：「机器人声、没音色」；修复：明确语义层只供 LM 建模，交付音频必须全层或走声学补全。
4. **评测指标失真**：PESQ 对低码率 codec 区分度差（为传统编解码器设计），UTMOS 对合成语音偏乐观。修复：双指标 + 人工 AB 听测三者对照写报告（本周交付要求）。

---

## 8. 延伸思考题（含解析）

**Q1**：Mimi 为什么要把帧率压到 12.5 Hz 这么激进？
**A**：双工对话要求 LM 实时处理「用户流 + 模型流」两条音频序列，50 Hz 意味着每秒上百个 token 的联合建模，算力撑不住实时；12.5 Hz 把序列砍到 1/4，代价（声学细节）交给后续 7 层声学码与深度 Transformer 补回。帧率是「实时性预算」倒推出来的。

**Q2**：为什么 EnCodec 的码率档位用「层数」而不是「码本大小」来调？
**A**：减层数保持每层 10 bit 的量化精度，只是砍信息总量，实现简单且各档共享同一组码本（一次训练多档可用）；改码本大小要改词表、重训所有下游。这与标量量化里「改 bit 数不改网格结构」是同一个工程直觉。

**Q3**：GPTQ 的误差分摊和 RVQ 的残差量化，哪个更接近「联合最优」？
**A**：RVQ——它在向量空间里联合表示一组维度，码本本身是学出来的联合分布近似；GPTQ 仍是逐标量量化，只是用 Hessian（二阶信息）把误差重分配到未量化权重上，属于「标量量化 + 误差补偿」。表达力上限：向量量化 > 标量量化，代价是训练与推理开销。

**Q4**：如果让你设计一个「面向中文方言」的语义层，蒸馏教师该换成什么？
**A**：换成在大规模中文/方言 ASR 或自监督数据上训练的语音模型（如大规模中文 SSL 模型的 encoder），因为 Mimi 的语义层质量上限由教师决定——WavLM 以英文为主，方言音素区分度不足。蒸馏目标的领域覆盖 = 语义层的领域能力。

**Q5**：一周对话数据（60 分钟/天 × 7 天）用 Mimi 语义流存储，比原始 16 kHz 16bit 音频省多少倍？
**A**：语义流帧率 12.5 token/s，每个 token 11 bit（码本 2048）→ 12.5 × 11 = 137.5 bps ≈ **0.14 kbps**，即约 0.017 KB/s；原始 16 kHz 16 bit = 256 kbps = 32 KB/s。**省约 1900 倍**（420 分钟对话从约 800 MB 压到约 0.4 MB）。若存全层 RVQ（1.1 kbps）则省约 230 倍。这就是「语音可以被压成语义」的存储视角证据。

---

## 本周交付清单

- [ ] 跑通 6.1 的从零 RVQ，断言误差严格递减。
- [ ] 完成 EnCodec 码率扫描（真实语音版），输出码率-质量曲线图（PESQ/UTMOS 双指标 + 人工听测结论）。
- [ ] 完成 Mimi 语义层单独解码实验，记录听感对比。
- [ ] 能默写比特率公式并验证 EnCodec 6 kbps 与 Mimi 1.1 kbps。
- [ ] 写出「RVQ vs GPTQ/AWQ 镜像」一页纸（跨方向面试稿）。
