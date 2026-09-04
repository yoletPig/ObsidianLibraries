# 第 5-6 周教程：工具箱整合与 AudioPipeline 构建

> **本周要回答的四个问题**
> 1. 前四周手写的组件，如何整合成一条**可复用、可配置**的标准流水线？
> 2. SpecAugment 的时间掩码/频率掩码在做什么？为什么它几乎是 ASR 训练的标配？
> 3. torchaudio / librosa / soundfile 各自该用在什么场景？
> 4. 数据增广（加噪、混响、变速、变调）如何模拟真实世界的声学多样性？

对应学习计划：第 5-6 周。交付物：可复用的 `AudioPipeline` 类（输入路径 → 输出 `{wave, fbank, mel_spec}` 三件套，支持 SpecAugment 开关），用 5 条语音跑通全流程并保存可视化。这个类将成为后续所有 Stage 的**数据入口**，值得花时间写扎实。

---

## 1. 工具箱全景：每个库该干什么

### 1.1 选型矩阵

| 库 | 定位 | 用在哪 | 注意 |
| --- | --- | --- | --- |
| `soundfile` | 纯 I/O（读写音频文件） | 文件读写、格式转换 | 不做处理，只做搬运 |
| `librosa` | 分析为主（特征、可视化） | 原型、研究、画图 | 纯 CPU/NumPy，慢但直观 |
| `torchaudio` | PyTorch 生态、可 GPU、可进训练图 | **训练数据流水线（首选）** | 与模型同框架，零拷贝衔接 |
| `audiomentations` | 时域增广（加噪、混响、变调） | 数据增广 | 需配合 soundfile |
| `scipy.signal` | 通用信号处理 | 滤波器、重采样的对照验证 | 教学与验证用 |

**决策规则**：训练数据流水线用 `torchaudio`（与模型同框架、支持批量化与 GPU）；研究分析和画图用 `librosa`；文件 I/O 用 `soundfile`。本教程以 `torchaudio` 为主线构建 `AudioPipeline`。

### 1.2 torchaudio 的关键组件

```python
import torch
import torchaudio

# 读取（返回 (channels, samples) 与采样率）
wave, sr = torchaudio.load("sample.wav")        # (C, N)

# 重采样（生产级，多相滤波）
wave = torchaudio.functional.resample(wave, sr, 16000)

# Kaldi 风格 Fbank（ASR 训练标配，与 ESPnet/FunASR 对齐）
fbank = torchaudio.compliance.kaldi.fbank(
    wave, num_mel_bins=80, sample_frequency=16000,
    frame_length=25.0, frame_shift=10.0)        # (T, 80)

# Mel 谱（更通用的声学特征）
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=512, hop_length=160,
    win_length=400, n_mels=80)
mel = mel_transform(wave)                       # (C, 80, T)
```

**`compliance.kaldi.fbank` vs `MelSpectrogram` 的区别**（面试高频）：前者严格复现 Kaldi 的特征提取（能量归一化、dither、预加重等细节），是 FunASR/ESPnet 系 ASR 的标准输入；后者是通用的可微 Mel 谱，常用于 TTS/声码器。**喂 ASR 用前者，做声学分析用后者**——混用是精度玄学的常见来源。

---

## 2. SpecAugment：ASR 训练的标配增广

### 2.1 动机与原理

SpecAugment（Park et al., 2019）的核心洞察：**对特征谱做随机掩码，等价于教模型"在信息残缺时也能识别"**——模拟真实世界的遮挡、丢帧、噪声突发。它作用在**特征域**（不是波形域），实现简单、几乎零成本，却带来显著的 WER 下降。

三个操作：

1. **时间掩码（Time Masking）**：随机选连续 $t$ 帧，全部置 0（模拟短暂静音/丢帧）。
2. **频率掩码（Frequency Masking）**：随机选连续 $f$ 个频带，全部置 0（模拟窄带噪声/信道凹陷）。
3. **时间扭曲（Time Warping）**：沿时间轴做轻微非线性扭曲（实践中常省略，收益边际）。

### 2.2 实现

```python
import torch

class SpecAugment(torch.nn.Module):
    """特征域增广：时间掩码 + 频率掩码。"""
    def __init__(self, freq_mask_param=27, time_mask_param=100,
                 num_freq_masks=2, num_time_masks=2):
        super().__init__()
        self.fmp, self.tmp = freq_mask_param, time_mask_param
        self.nfm, self.ntm = num_freq_masks, num_time_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """spec: (T, F) 或 (B, T, F)。仅训练时调用。"""
        x = spec.clone()
        squeeze = (x.dim() == 2)
        if squeeze:
            x = x.unsqueeze(0)
        B, T, F = x.shape
        for _ in range(self.nfm):
            f = torch.randint(0, self.fmp + 1, (1,)).item()
            f0 = torch.randint(0, max(F - f, 1), (1,)).item()
            x[:, :, f0:f0 + f] = 0
        for _ in range(self.ntm):
            t = torch.randint(0, self.tmp + 1, (1,)).item()
            t0 = torch.randint(0, max(T - t, 1), (1,)).item()
            x[:, t0:t0 + t, :] = 0
        return x.squeeze(0) if squeeze else x

# 验证：掩码后应有若干行/列为 0
aug = SpecAugment()
feat = torch.randn(1000, 80)
out = aug(feat)
assert out.shape == feat.shape
assert (out.sum(dim=1) == 0).any() or (out.sum(dim=0) == 0).any()
```

**关键工程细节**：
- **只在训练时开**：推理/验证必须关闭，否则引入随机性破坏可复现性；
- 参数 $F, T$ 的上限（如 `freq_mask_param=27`）来自原论文在 LibriSpeech 的配置，中文/长音频可按比例调整；
- 掩码置 0 等价于"把该区域当静音"，与 CMVN 归一化配合时注意：应在归一化**之后**做掩码（置 0 才代表"无信息"）。

---

## 3. 波形域增广：模拟声学多样性

特征域掩码之外，**波形域增广**让模型见到更多声学条件。四类最常用：

### 3.1 加性噪声（SNR 控制）

复用第 4 周的 `mix_at_snr()`：从噪声库随机抽一条，按随机 SNR（如 5-20 dB）混入。这是鲁棒性增广的主力。

### 3.2 混响卷积（RIR）

房间脉冲响应 $h[n]$ 与干净语音卷积即得带混响版本：

$$
y = s * h
$$

```python
import torch
import torch.nn.functional as F

def apply_reverb(speech: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    """speech: (1, N); rir: (1, M)。返回带混响信号并归一化。"""
    # 用 FFT 卷积加速（长信号）
    y = F.conv1d(speech.unsqueeze(0).flip(-1), rir.unsqueeze(0).flip(-1))
    y = y.squeeze(0).flip(-1)[:speech.shape[-1]]
    peak = y.abs().max() + 1e-8
    return y / peak * speech.abs().max()   # 峰值对齐防削波
```

RIR 数据源：OpenAIR、MIT IR Survey，或用 `pyroomacoustics` 数值模拟任意房间。

### 3.3 变速与变调

- **变速（speed perturb）**：改变时长但**不改音高**（如 0.9×/1.1×）——`torchaudio` 无直接支持，用 `sox` 或 `librosa.effects.time_stretch`；
- **变调（pitch shift）**：改音高不改时长——模拟不同说话人。

Kaldi 经典配方是 **0.9/1.0/1.1 三档变速**，等价于数据量 ×3。

### 3.4 音量扰动

随机乘一个增益（如 ±6 dB），教模型对响度不变。注意别推出削波区。

---

## 4. AudioPipeline：把一切焊成可复用类（交付核心）

### 4.1 设计目标

- **输入**：音频路径（任意采样率/声道）；**输出**：`{wave, fbank, mel_spec}` 三件套 + 可选增广；
- 统一重采样到 16 kHz、单声道；
- SpecAugment 与波形增广可开关；
- 输出可直接进模型（张量化）。

### 4.2 完整实现

```python
import torch
import torchaudio
import numpy as np

class AudioPipeline:
    """
    语音数据入口：音频路径 -> {wave, fbank, mel_spec} 三件套。
    - 统一 16kHz 单声道
    - 支持 SpecAugment（特征域）与加性噪声（波形域）增广
    """
    def __init__(self, target_sr=16000, n_mels=80,
                 use_specaugment=False, augment_noise=False,
                 noise_dir=None):
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.specaugment = SpecAugment() if use_specaugment else None
        self.augment_noise = augment_noise
        self.noise_dir = noise_dir

        # Kaldi 风格 Fbank（ASR 标准输入）
        self.fbank = lambda w: torchaudio.compliance.kaldi.fbank(
            w, num_mel_bins=n_mels, sample_frequency=target_sr,
            frame_length=25.0, frame_shift=10.0)
        # 通用 Mel 谱（声学分析 / TTS 侧）
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr, n_fft=512, hop_length=160,
            win_length=400, n_mels=n_mels)

    def _load_mono(self, path):
        wave, sr = torchaudio.load(path)
        if wave.shape[0] > 1:
            wave = wave.mean(dim=0, keepdim=True)      # 混缩单声道
        if sr != self.target_sr:
            wave = torchaudio.functional.resample(wave, sr, self.target_sr)
        return wave                                     # (1, N)

    def _maybe_add_noise(self, wave):
        if not self.augment_noise or self.noise_dir is None:
            return wave
        # 从噪声库随机取一条，按随机 SNR 混入（复用第 4 周逻辑）
        import os, random
        files = os.listdir(self.noise_dir)
        noise, nsr = torchaudio.load(
            os.path.join(self.noise_dir, random.choice(files)))
        noise = noise.mean(dim=0, keepdim=True)
        if noise.shape[-1] < wave.shape[-1]:
            noise = noise.repeat(1, int(np.ceil(wave.shape[-1]/noise.shape[-1])))
        noise = noise[:, :wave.shape[-1]]
        snr = random.uniform(5, 20)
        ps = wave.pow(2).mean() + 1e-12
        pn = noise.pow(2).mean() + 1e-12
        scale = (ps / (pn * 10 ** (snr/10))).sqrt()
        return wave + scale * noise

    def __call__(self, path):
        wave = self._load_mono(path)
        wave = self._maybe_add_noise(wave)

        fbank = self.fbank(wave)                        # (T, 80)
        mel_spec = self.mel_transform(wave).squeeze(0)  # (80, T)

        if self.specaugment is not None:
            fbank = self.specaugment(fbank)

        return {
            "wave": wave,              # (1, N) 波形
            "fbank": fbank,            # (T, 80) ASR 输入
            "mel_spec": mel_spec,      # (80, T) 声学特征
            "sr": self.target_sr,
        }
```

### 4.3 端到端验证（交付）

```python
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 准备 5 条测试音频（中文语音或任意语音）
paths = sorted(glob.glob("test_audio/*.wav"))[:5]
assert len(paths) >= 5, "请准备至少 5 条测试音频放入 test_audio/"

pipe = AudioPipeline(use_specaugment=True, augment_noise=False)

for i, p in enumerate(paths):
    out = pipe(p)
    wave, fbank, mel = out["wave"], out["fbank"], out["mel_spec"]
    print(f"[{p}] wave={tuple(wave.shape)} fbank={tuple(fbank.shape)} mel={tuple(mel.shape)}")
    # 断言：三件套形状自洽
    assert wave.dim() == 2 and wave.shape[0] == 1
    assert fbank.shape[1] == 80
    expected_T = 1 + (wave.shape[-1] - 400) // 160
    assert abs(fbank.shape[0] - expected_T) <= 2   # 中心填充±1帧容差

    # 保存可视化（波形 + fbank 热力图）
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(wave[0].numpy())
    axes[0].set_title("Waveform")
    im = axes[1].imshow(fbank.T.numpy(), aspect="auto", origin="lower")
    axes[1].set_title("Log-Mel Fbank (80-dim)")
    fig.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig(f"pipeline_out_{i}.png", dpi=110)
    plt.close(fig)
print("全部跑通，可视化已保存。")
```

**预期**：每条音频打印三个 shape，fbank 帧数与公式 $1 + (N-400)/160$ 吻合（±1 容差），并输出波形 + 语谱图可视化。

**验证增广生效**：同一路径分别用 `use_specaugment=True/False` 各跑一次，断言两次 `fbank` 不完全相同（掩码引入差异），证明增广链路接通。

---

## 5. 工程权衡与失效模式

### 5.1 权衡

- **增广强度**：过强（低 SNR 加噪 + 多重掩码）会让训练分布偏离推理分布，反而伤精度；从无增广基线出发，逐项加增广看验证集变化。
- **特征提取一致性**：训练与推理必须用**完全相同**的特征配置（同库、同参数），否则分布漂移。`AudioPipeline` 的价值正在于此——单点定义，处处复用。
- **波形增广在线还是离线**：在线增广每次 epoch 随机不同（多样性好，但慢）；离线预生成（快，但固定）。常用折中：在线波形增广 + 缓存特征。

### 5.2 失效模式

1. **特征配置不一致**：训练用 `compliance.kaldi.fbank`、推理用 `MelSpectrogram`，输入分布错位 → WER 暴涨且难排查。**定位**：打印两路特征均值/方差对比；**修复**：统一走 `AudioPipeline`。
2. **推理时忘关增广**：验证集带随机掩码，指标抖动且偏低。**修复**：`use_specaugment` 与训练阶段绑定，推理强制关。
3. **增广后削波**：加噪/混响后幅值超过 ±1。**修复**：增广后统一峰值归一化。
4. **重采样遗漏**：混合采样率数据集未统一到 16 kHz，帧数公式全错。**修复**：`AudioPipeline` 入口强制重采样并断言。

---

## 6. 端到端预演：把特征喂给一个真实模型

本周收尾任务：不深究模型，只走通"波形 → 特征 → 模型"的流水线。用 `funasr` 或 `transformers` 跑一次语音识别，重点观察**你的 `AudioPipeline` 特征如何接入**：

```python
# 示例：用 funasr 的 SenseVoice 走一次识别（需先 pip install funasr）
from funasr import AutoModel

model = AutoModel(model="iic/SenseVoiceSmall")
res = model.generate(input="test_audio/sample_0.wav")
print(res[0]["text"])
```

观察点：① 模型内部特征提取与你 `AudioPipeline` 的配置是否一致；② 输入波形如何被转成 80 维特征再进 encoder。这为 Stage 2 深拆 ASR 模型做铺垫——**你已经打通了"数据入口"，下周起拆"模型内部"**。

---

## 7. 延伸思考题（含解析）

**Q1**：SpecAugment 为什么作用在特征域而非波形域？两者互补吗？
**A**：特征域掩码直接模拟"信息缺失"，计算几乎免费且对模型输入空间精准；波形增广（加噪/混响）模拟声学条件多样性。二者作用层面不同、互补，实践中叠加使用。

**Q2**：为什么喂 ASR 用 `compliance.kaldi.fbank` 而不用通用 `MelSpectrogram`？
**A**：Kaldi fbank 严格复现训练该模型时的特征规范（能量、dither、预加重），保证输入分布一致；通用 Mel 谱细节不同，会引入分布漂移。特征提取必须与模型训练配置逐项对齐。

**Q3**：`AudioPipeline` 的帧数断言为什么留 ±1 容差？
**A**：不同实现的中心填充/边界处理可能差一帧；断言目标是抓"量级错误"（如采样率错导致帧数差几倍），而非纠结一帧的边界差异。

**Q4**：在线增广和离线预生成增广，各自适合什么场景？
**A**：在线增广多样性好、存储省，适合训练主流程；离线预生成速度快、可复现，适合把昂贵的增广（如混响卷积）固化成数据集。常见组合：昂贵增广离线、轻量增广在线。

**Q5**：如果把 `AudioPipeline` 输出直接喂给 Stage 2 的 Whisper，需要注意什么？
**A**：Whisper 用自己的 80 维、25ms/10ms、n_fft=400 的 log-mel，且内部自动提取——直接喂波形即可，不要喂你预计算的 512-FFT 特征（n_fft 不同）。特征必须与目标模型的官方前处理一致。

---

## 本周交付清单

- [ ] 实现 `SpecAugment`，验证掩码生效断言。
- [ ] 实现 `AudioPipeline` 类，输出三件套并统一 16 kHz 单声道。
- [ ] 用 5 条音频跑通全流程，帧数断言通过，保存波形 + 语谱图可视化。
- [ ] 验证增广开关：开/关 SpecAugment 两次输出有差异。
- [ ] 用 `funasr`/`whisper` 走一次端到端识别，观察特征接入点。

## Stage 1 总结：你现在拥有的地基

完成本阶段，你已能：

1. **徒手推导**：任意音频从波形 → 帧 → 频谱 → 80 维 Log-Mel 的每一步 Shape；
2. **亲手实现**：重采样、STFT/iSTFT、Mel 滤波器组、谱减法、`mix_at_snr`、SpecAugment；
3. **一条标准流水线**：`AudioPipeline`，作为 Stage 2-6 所有实验的统一数据入口。

对照学习计划第 6 周的自测清单逐项核验，全部通过后即可进入 **Stage 2：语音识别 ASR**。
