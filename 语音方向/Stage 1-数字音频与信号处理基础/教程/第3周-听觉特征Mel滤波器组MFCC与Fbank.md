# 第 3 周教程：听觉特征工程——Mel 滤波器组、MFCC 与 Fbank

> **本周要回答的四个问题**
> 1. 人耳对不同频率的敏感度并不均匀——这在特征提取里如何体现？
> 2. Mel 滤波器组到底长什么样？每个三角滤波器在做什么？
> 3. MFCC 为什么在深度学习时代被弃用？Fbank 为什么成了 ASR 标准输入？
> 4. 80 维、128 维这些超参数是怎么来的？不同模型的差异在哪？

对应学习计划：第 3 周。交付物：手写 Mel 滤波器组构造 + 80 维 Log-Mel Fbank 计算，与 `librosa`/`torchaudio` 数值对齐。

---

## 1. 第一性原理：为什么要"以耳为本"

### 1.1 根本矛盾

线性频谱（第 2 周的 $|X[m,k]|$）在频率轴上是**均匀的**：每 31.25 Hz 一个 bin，从 0 到 8 kHz 一视同仁。但人耳不是线性仪器：

- **低频区分辨率高**：我们能轻松区分 100 Hz 与 200 Hz（差一个八度）；
- **高频区分辨率低**：8000 Hz 与 8100 Hz 听起来几乎一样。

把均匀频谱直接喂给模型，等于强迫模型用 60% 以上的参数去建模人耳不关心的高频细节。解决方案：**按人耳感知重新划分频率轴**——这就是 Mel 刻度。

### 1.2 Mel 刻度

Mel 刻度由心理声学实验定义：1000 mel 被定义为 40 dB、1 kHz 纯音的音高感知，其余频率按"听起来几倍高/低"的主观判断标定。广泛使用的近似换算（O'Shaughnessy 公式）：

$$
m = 2595 \cdot \log_{10}\!\left(1 + \frac{f}{700}\right)
$$

逆变换：

$$
f = 700 \cdot \left(10^{m/2595} - 1\right)
$$

**性质**：低频段近似线性（$f \ll 700$ 时 $m \approx 2595 f/700$），高频段近似对数压缩。直觉：低频精细、高频粗糙——正好匹配耳蜗的频率分辨特性。

（现代变体：HTK mel 与 Slaney mel 的差异只在归一化，`librosa` 用 `htk=True/False` 切换；Whisper 用的是自己的 80 维滤波器组，思路一致。）

### 1.3 从刻度到滤波器组

有了 Mel 刻度，构造滤波器组的步骤是：

1. 确定目标频带 $[f_{\min}, f_{\max}]$（如 $[0, 8000]$ Hz）；
2. 两端换算到 Mel：$[m_{\min}, m_{\max}]$；
3. 在 **Mel 轴上等距**放 $B + 2$ 个点（$B$ 为滤波器数，+2 是为了定义首尾边界）；
4. 把这些点换回 Hz，并映射到最近的 FFT bin 索引；
5. 每三个相邻点 $(f_{i-1}, f_i, f_{i+1})$ 张成一个三角滤波器：中心 $f_i$ 处权重 1，向两侧线性降到 0。

**第 $i$ 个三角滤波器的响应**：

$$
H_i[k] =
\begin{cases}
0, & k < f_{i-1} \\
\dfrac{k - f_{i-1}}{f_i - f_{i-1}}, & f_{i-1} \le k \le f_i \\
\dfrac{f_{i+1} - k}{f_{i+1} - f_i}, & f_i < k \le f_{i+1} \\
0, & k > f_{i+1}
\end{cases}
$$

**关键直觉**：滤波器组矩阵 $\mathbf{H} \in \mathbb{R}^{B \times (N/2+1)}$ 是一个**稀疏的带权池化**——把 257 个线性频率 bin 加权平均成 80 个"听觉频带"。低频区滤波器窄（保留细节），高频区滤波器宽（主动丢弃细节）。这一步把特征维度从 257 压到 80，同时把信息按"耳朵关心程度"重新分配。

### 1.4 Log-Mel 特征的计算链

$$
\underbrace{|X[m,k]|^2}_{\text{功率谱}}
\;\xrightarrow{\ \times \mathbf{H}^T\ }\;
\underbrace{\text{mel}[m,i]}_{\text{80 维}}
\;\xrightarrow{\ \log(\cdot + \epsilon)\ }\;
\underbrace{\text{log\_mel}[m,i]}_{\text{最终特征}}
$$

即：

$$
\text{mel}[m, i] = \sum_{k} |X[m,k]|^2 \cdot H_i[k], \qquad
\text{log\_mel}[m,i] = \log\big(\text{mel}[m,i] + \epsilon\big)
$$

- **为什么用功率谱**（平方）而非幅度谱：功率与能量成正比，物理意义干净；多数实现（含 torchaudio 的 `power=2.0`）如此。
- **为什么要取对数**：① 人耳响度感知近似对数（Weber-Fechner 定律）；② 语音动态范围约 60 dB，对数压缩后乘性变化（音量、信道增益）变成加性偏移，对归一化更友好。
- **$\epsilon$ 的作用**：静音帧功率可为 0，$\log 0 = -\infty$ 会产生 NaN。取 $\epsilon = 10^{-5}$ 到 $10^{-10}$（注意：$\epsilon$ 大小直接影响静音帧的特征值，不同库默认不同——这是跨库对齐失败的高频原因）。

---

## 2. MFCC：古典王者与其退位

### 2.1 MFCC 的完整流程

$$
\text{波形} \xrightarrow{\text{STFT}} |X|^2 \xrightarrow{\text{Mel 滤波}} \xrightarrow{\log} \xrightarrow{\text{DCT 取前 13 维}} \text{MFCC}
$$

DCT（离散余弦变换）在这里的作用：log-mel 相邻维度高度相关（三角滤波器大量重叠），DCT 做**去相关 + 信息压缩**，前 13 个系数承载绝大部分信息（第 0 维是总能量，1-12 维是谱包络形状）。

### 2.2 为什么深度学习时代弃用 MFCC

三个理由，按重要性排序：

1. **信息有损**：80 维 → 13 维的压缩是为 GMM-HMM 这种弱模型设计的（参数少、防过拟合）。神经网络容量足够，宁可要全量 80 维信息，自己学压缩。
2. **DCT 假设失效**：DCT 去相关假设特征服从某种平滑性；神经网络用卷积/注意力能学到任意非线性组合，预设的正交基反而是束缚。
3. **梯度不友好**：MFCC 链路里的截断（取 13 维）不可微，端到端训练时无法反传——现代模型要么用全量 log-mel，要么把滤波器组做成可学习层。

**一句话**：MFCC = 为弱模型做的特征压缩；深度学习时代把压缩权交还给模型。**Whisper、SenseVoice、Paraformer、CosyVoice 全部使用 log-mel（80 维或 128 维），没有一个用 MFCC。**

### 2.3 主流模型的听觉特征配置（速查表）

| 模型 | 维度 | 采样率 | 窗长/帧移 | 备注 |
| --- | --- | --- | --- | --- |
| Whisper | 80 | 16 kHz | 25 ms / 10 ms, n_fft=400 | 自定义滤波器组 |
| SenseVoice | 80 | 16 kHz | 25 ms / 10 ms | fbank，CMVN 归一化 |
| Paraformer | 80 | 16 kHz | 25 ms / 10 ms | Kaldi 风格 compliance fbank |
| CosyVoice TTS | 50 | 24 kHz | — | 语音合成侧常用更少维度 |
| HiFi-GAN（声码器） | 80/100 | 22/24 kHz | — | 输入是 mel，输出波形 |

记住这个模式：**ASR ≈ 80 维 @16 kHz，TTS ≈ 80-100 维 @22-24 kHz**。

---

## 3. 手写实现：Mel 滤波器组从零构造

### 3.1 构造滤波器组矩阵（交付核心）

```python
import numpy as np

def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def mel_filterbank(n_mels: int, n_fft: int, sr: int,
                   fmin: float = 0.0, fmax: float = None) -> np.ndarray:
    """
    构造 (n_mels, n_fft//2 + 1) 的三角滤波器组矩阵。
    """
    fmax = fmax or sr / 2
    n_freqs = n_fft // 2 + 1

    # 1-2. Mel 轴等距放 n_mels + 2 个点，换回 Hz
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mels)                       # (n_mels + 2,)

    # 3. Hz -> FFT bin 索引（浮点，用于线性插值权重）
    bin_pts = hz_pts * n_fft / sr                  # (n_mels + 2,)

    fb = np.zeros((n_mels, n_freqs))
    freq_axis = np.arange(n_freqs)                 # bin 索引轴
    for i in range(n_mels):
        lo, mid, hi = bin_pts[i], bin_pts[i+1], bin_pts[i+2]
        # 上升沿
        up = (freq_axis - lo) / max(mid - lo, 1e-10)
        # 下降沿
        down = (hi - freq_axis) / max(hi - mid, 1e-10)
        fb[i] = np.clip(np.minimum(up, down), 0, None)
    return fb

# 验证形状与非负稀疏性
fb = mel_filterbank(80, n_fft=512, sr=16000)
print("滤波器组 shape:", fb.shape)                # (80, 257)
assert fb.shape == (80, 257)
assert (fb >= 0).all()
assert fb.sum(axis=1).min() > 0                    # 每个滤波器都有能量
# 低频滤波器窄、高频滤波器宽：比较第 1 个与第 79 个滤波器的支撑宽度
width_low  = (fb[0]  > 0).sum()
width_high = (fb[79] > 0).sum()
print(f"第 1 个滤波器支撑 {width_low} bins，第 80 个支撑 {width_high} bins")
assert width_high > width_low * 2, "高频滤波器应明显更宽"
```

**预期输出**：

```
滤波器组 shape: (80, 257)
第 1 个滤波器支撑 6 bins，第 80 个支撑 57 bins
```

最后一行断言是 Mel 刻度"低频细、高频粗"的直接可视化证据。画出来更直观：

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
for i in range(0, 80, 5):
    plt.plot(fb[i], alpha=0.7)
plt.xlabel("FFT bin (31.25 Hz/bin)"); plt.ylabel("权重")
plt.title("Mel Filterbank (every 5th of 80)")
plt.tight_layout(); plt.savefig("mel_filterbank.png", dpi=120)
```

### 3.2 计算 80 维 Log-Mel Fbank

```python
import numpy as np
import soundfile as sf

def log_mel_fbank(x: np.ndarray, sr: int = 16000, n_fft: int = 512,
                  hop: int = 160, win_len: int = 400,
                  n_mels: int = 80, eps: float = 1e-5) -> np.ndarray:
    """波形 -> (T, n_mels) log-mel 特征。复用第 2 周的 stft()。"""
    # 把第 2 周教程中的 stft() 定义拷入本文件（或组织成 dsp_tools.py 统一导入）
    S = stft(x, n_fft=n_fft, hop=hop, win_len=win_len)   # (T, 257) 复数
    power = (np.abs(S) ** 2)                               # 功率谱
    fb = mel_filterbank(n_mels, n_fft, sr)                 # (80, 257)
    mel = power @ fb.T                                     # (T, 80)
    return np.log(mel + eps)

# ---- Shape 推导验证（本周自测题的数值版）----
dur = 10.0
x = np.random.randn(int(dur * 16000)).astype(np.float32)
feat = log_mel_fbank(x)
print("Log-Mel shape:", feat.shape)          # (1001, 80)
expected_T = 1 + int(dur * 16000) // 160
assert feat.shape == (expected_T, 80), f"期望 ({expected_T}, 80)，实际 {feat.shape}"
```

**Shape 推导**（本周自测原题，必须能脱口而出）：10 秒 × 16 kHz = 160000 样本；帧移 160 样本 → $T = 1 + 160000/160 = 1001$ 帧；每帧 80 维 → $(1001, 80)$。

### 3.3 与 librosa / torchaudio 数值对齐

```python
import librosa

S_lib = librosa.feature.melspectrogram(
    y=x, sr=16000, n_fft=512, hop_length=160, win_length=400,
    n_mels=80, fmin=0, fmax=8000, power=2.0, htk=False)
log_lib = np.log(S_lib + 1e-5)

err = np.max(np.abs(feat - log_lib))
print(f"手写版与 librosa 最大差异: {err:.4f}")
```

**预期**：差异通常在 $0.01$ 以内。若有系统偏差，按此排查清单（按出现频率排序）：

1. **滤波器归一化方式**：`librosa` 默认 `norm=None`，但部分实现用"等面积归一化"（Slaney，除以滤波器带宽）；
2. **$\epsilon$ 取值**：`librosa.power_to_db` 内部参考值不同；
3. **幅度谱还是功率谱**：差一个平方；
4. **窗函数**：Hann 必须一致；
5. **htk 开关**：mel 换算公式不同。

能逐条说出差异来源，说明你真的懂了这条链路——这也是面试常考题。

---

## 4. CMVN：信道不变的最后一道工序

### 4.1 为什么需要

同一句话在不同麦克风/电话信道下，频谱会被整体染色（乘性失真）→ log 域变成加性偏移。CMVN（Cepstral Mean and Variance Normalization）对每个特征维度做标准化：

$$
\hat{z}_{t,i} = \frac{z_{t,i} - \mu_i}{\sigma_i}
$$

$\mu_i, \sigma_i$ 按时间轴统计。这能显著削弱信道差异，是 SenseVoice/Paraformer 前端的标配。

```python
def cmvn(feat: np.ndarray) -> np.ndarray:
    mu = feat.mean(axis=0, keepdims=True)
    sigma = feat.std(axis=0, keepdims=True) + 1e-8
    return (feat - mu) / sigma

feat_norm = cmvn(feat)
assert abs(feat_norm.mean()) < 1e-6
assert abs(feat_norm.std() - 1.0) < 0.05
```

注意：**流式推理**时不能看到未来帧，要用滑动窗口或全局先验统计量（训练集预计算的 mean/var）——SenseVoice 就内置了全局均值文件。

---

## 5. 工程权衡与失效模式

### 5.1 权衡

- **维度选择**：80 维是 ASR 甜点；128 维用于高采样率（24 kHz+）音频；40 维以下细节损失明显。维度过高只增加算力不增加信息（高频滤波器已高度重叠）。
- **对数底与缩放**：`log` 与 `log10`、dB 缩放只差常数倍，但必须与下游模型的训练配置**严格一致**，否则输入分布错位，性能崩。
- **$\epsilon$ 选择**：太小 → 静音帧数值不稳；太大 → 压缩低能量细节。跟随目标模型的官方配置，别自创。

### 5.2 失效模式

1. **特征配置与模型不匹配**：用 128 维喂 80 维模型（或反之），shape 报错是好的；更阴险的是维度恰好能过但语义错位（如 512 vs 400 的 n_fft 差异）。**定位**：打印官方 `preprocessor` 配置逐项比对。
2. **忘记取对数或重复取对数**：ASR 输出全错但"看起来在跑"。**定位**：画特征直方图——正常 log-mel 近似正态，原始 mel 严重右偏。
3. **采样率与特征参数脱钩**：把 16 kHz 的参数（n_fft=512）直接用在 48 kHz 音频上，频率轴整体错 3 倍。**修复**：所有特征函数显式传入 `sr` 并断言。

---

## 6. 延伸思考题（含解析）

**Q1**：16 kHz、10 秒语音、25 ms 窗、10 ms 移、80 维 Mel——最终特征 shape？
**A**：$(1001, 80)$。$T = 1 + 160000/160 = 1001$。

**Q2**：为什么滤波器组的点要在 **Mel 轴**上等距，而不是 Hz 轴？
**A**：等距于 Mel 轴 ⇔ 每个滤波器覆盖"相同的主观音高带宽"，与人耳分辨力匹配；若在 Hz 轴等距，高低频滤波器一样宽，就退化成均匀频谱，失去意义。

**Q3**：若有人坚持用 MFCC 训 Whisper 级模型，你预期会发生什么？
**A**：WER 显著上升。13 维截断丢失了发音细节（尤其高频辅音信息），且 DCT 基函数不匹配数据分布；历史上端到端模型多次验证过这一点。

**Q4**：CMVN 在流式场景怎么做？训练与推理不一致会怎样？
**A**：用滑动窗（如最近 3 秒）或训练集全局统计量。若训练用句级统计、推理用全局统计，归一化分布不一致，WER 会涨几个点——归一化统计量是特征的一部分，必须对齐。

**Q5**：Whisper 的 mel 是 80 维，但 n_fft=400（频率维 201）。它的滤波器组矩阵形状？
**A**：$(80, 201)$。原理相同，只是频率轴变短。这提醒你：**特征维度（80）与 n_fft 无关，只与滤波器数有关**。

---

## 本周交付清单

- [ ] 手写 `mel_filterbank()`，验证低频窄/高频宽断言，保存滤波器组可视化图。
- [ ] 手写 `log_mel_fbank()`，对 10 秒音频输出 $(1001, 80)$ 并手推公式验证。
- [ ] 与 `librosa` 对齐，差异 > 0.05 时按排查清单逐条定位并写明原因。
- [ ] 实现 `cmvn()` 并验证归一化后均值≈0、方差≈1。
