# 第 2 周教程：AI 深度降噪——DeepFilterNet 系

> **本周要回答的四个问题**
> 1. 从谱减法到维纳滤波都是"估增益"，深度降噪为什么换了一条路？
> 2. DeepFilterNet 的 Deep Filtering 在做什么？为什么比直接预测掩码好？
> 3. 实时约束（20 ms 帧、40 ms 算法延迟、CPU RTF≪1）是怎么做到的？
> 4. Conv-TasNet / DTLN / FullSubNet 这些流派和 DeepFilterNet 差在哪？

对应学习计划：第 2 周。交付物：跑通 DeepFilterNet3 推理，与第 1 周维纳滤波在 DNS 测试集上比 DNSMOS；加载官方 ONNX（或用 PyTorch 后端）测 CPU 延迟。

---

## 1. 第一性原理：从"估增益"到"学滤波"

### 1.1 掩码估计路线的局限

第 1 周的维纳滤波本质是**逐时频单元独立估增益** $G[m,k]$。深度学习最初也走这条路：网络预测一个理想比值掩码（IRM）或复数掩码，乘到带噪谱上：

$$
\hat{S}[m,k] = M[m,k] \cdot Y[m,k]
$$

**这个公式在干什么**：与第 1 周的 $\hat{S} = G \cdot Y$ 同构，只是 $G$ 从手工公式（维纳）换成了神经网络输出的 $M$——可以是实数比值掩码（IRM），也可以是复数掩码（CRM，同时给幅度和相位一个缩放系数）。网络学的还是"每个时频格子该乘多少"。

这条路的天花板在于：**掩码是对每个时频点独立的标量缩放**。它无法表达"用相邻多个时频点的信息加权合成当前点"——而这恰恰是处理混响尾、宽带噪声所需要的。

### 1.2 Deep Filtering：把"增益"升级为"跨帧复数滤波器"

DeepFilterNet 的核心创新：不再预测单个增益，而是为每个频点预测一个**跨多帧的复数滤波器**，用这个滤波器对带噪谱做时间方向的加权合成（DeepFilterNet 论文中滤波阶数 $N=5$）：

$$
\hat{S}[m,k] = \sum_{n=0}^{N-1} h_{m,k}[n] \cdot Y[m+\Delta_k - n,\; k]
$$

**这个公式在干什么**：第 1 周的增益是 $G[m,k]\cdot Y[m,k]$——只用**当前这一帧**的数据；Deep Filtering 把"素材池"扩大到**相邻 $N$ 帧**的同一频点（$\Delta_k$ 是滤波器前瞻，DFN3 取 2 帧），每帧配一个**复数权重** $h_{m,k}[n]$ 加权求和。复数权重同时改幅度和相位——相当于在频域给每个频点配了一个 5 阶小 FIR 滤波器。注意该滤波只作用于低频段（DFN3 中为最低 96 个 bin，约 4.8 kHz）——因为语音的周期性谐波成分集中在这里，高频段用第 1.3 节的 ERB 增益处理即可。

**为什么这更强**：

1. **跨帧利用信息**：谐波（基音及其倍频）在相邻帧间高度相关，用多帧合成能"借"邻近帧的能量去噪，也能平滑跟踪噪声的帧间抖动；
2. **复数滤波**：同时处理幅度与相位，相位估计不再靠"保留带噪相位"这种妥协；
3. **仍是线性滤波**：输出是输入的线性组合，保真度高、不易引入非线性伪影（对比纯生成式方法）。

**直观对比**：掩码估计 = 每个像素单独调亮度；Deep Filtering = 给每个频点学一个跨帧的小卷积核。后者的表达能力是前者的严格超集（掩码 = $N=1$ 且只看当前帧的特例——论文中对应 CRM 复数比值掩码）。论文用实验证明了这一点：在 5–30 ms 的多种 FFT 帧长下，Deep Filtering 的 SI-SDR 全面优于 CRM，且帧长越短优势越大。

### 1.2.1 深入拆解：为什么"复数 + 多帧"是关键

> 维纳滤波（第 1 周）也满足 $G$ 是实数、单帧——它的两个限制恰好被 Deep Filtering 逐一解开。

**（一）实数增益动不了相位。** 噪声和语音混在一起时，带噪相位 ≠ 干净相位。实数 $G$ 只能缩放幅度，相位只能"继承带噪相位"。多数场景下带噪相位尚可接受，但在低 SNR 和谐波密集处，相位误差直接损伤波形结构。复数滤波器 $h\in\mathbb{C}^N$ 把相位也纳入可学习范围。

**（二）单帧信息太少。** 一个时频格里的观测是"语音+噪声"的混合，仅凭本帧无法分辨"这是语音的谐波还是恰好同频的噪声"。但谐波是准周期的：上一帧、上上帧同一频点的值提供了强先验。多帧线性组合 = 对"这个频点在时间上如何演化"建模——这正是卷积的语义。

**（三）代价可控。** 每个频点只有 $N=5$ 个复系数，且只作用在 96 个低频 bin 上——相比"预测全部 481 个频点的复数掩码"（960 点 FFT），系数数量少一个数量级。这是"表达力↑、计算量↓"同时成立的关键。

### 1.3 双分支特征：ERB 包络 + 帧级细节

DeepFilterNet 的 Encoder 走两条路（以默认的 DeepFilterNet3 配置为例，48 kHz、FFT 960 点、帧移 480 点）：

- **ERB 分支**：把 481 个频率 bin 按人耳听觉刻度 **ERB（Equivalent Rectangular Bandwidth，等效矩形带宽）** 合并成 **32 个频带**，对每个频带预测一个**实数增益**——负责重建语音的"频谱包络"（低频一声道密、高频一频带很宽）。对数压缩的 ERB 特征同时利用了"响度感知是对数的、频率感知也是对数的"两条心理声学规律；
- **帧级（fullband）分支**：每帧只取少量标量统计特征（归一化系数、局部 SNR 等），走轻量通路捕捉帧间动态。

两路特征各自经过带分组（grouping）的卷积和 GRU 后，ERB 增益先乘上带噪谱，再由 1.2 节的 Deep Filtering 精修低频段（96 个 bin，约 4.8 kHz），高低两段拼出最终增强谱。**双分支的意义**：人耳对包络远比对相位/精细结构敏感（论文正是以"感知动机"命名），粗粒度的 ERB 增益用极小的计算量就决定了听感主体，细粒度的 Deep Filtering 只需花在刀刃（低频谐波）上——这是"小模型、低延迟、好音质"三角的架构解。

---

## 2. 实时性：工程约束如何塑造架构

### 2.1 延迟预算

DeepFilterNet3 论文给出的设计参数（48 kHz 采样率）：

- 帧长 20 ms（960 点）、帧移 10 ms（480 点）；
- 前瞻（lookahead）2 帧 = 20 ms——模型会"偷看"未来 2 帧再决定当前帧的输出；
- **总算法延迟 = 帧长 + 前瞻 = 40 ms**（帧缓冲 20 ms + 前瞻 20 ms）；
- 计算量：DeepFilterNet（第 1 代）约 1.8M 参数 / 0.35 GMACs；本教程实测 DeepFilterNet3（约 2.14M 参数）在普通笔记本 CPU 单线程处理 2 s 音频仅需 ~0.13 s，**RTF ≈ 0.07**（论文报告单线程 RTF 0.19）。

### 2.2 做到实时的三个手段

1. **短前瞻的准因果设计**：GRU 递归严格因果，卷积只允许 2 帧 lookahead → 延迟可控且有限（不是非因果的"看完整句"）。注意它**并非纯因果**：用 20 ms 前瞻换取质量，这是延迟-质量的显式折中；
2. **Deep Filtering 核很小**：每个频点只有 5 个复系数，且只作用于 96 个低频 bin，计算量远小于全时频注意力；
3. **ERB 压缩 + 分组（grouping）**：长时建模与卷积都在 ERB 粗粒度和分组稀疏化后进行，避免在 481 个 bin 的全分辨率上做大感受野。

**这给端侧部署的启示**：实时降噪不靠堆大模型，靠"短前瞻 + 小核 + 多分辨率"的架构纪律。你的结课助手前端（Stage 7）就用这类模型。

### 2.3 流派全景对比

| 模型 | 域 | 核心机制 | 端侧友好度 |
| --- | --- | --- | --- |
| **DeepFilterNet2/3** | 频域 | ERB 增益 + 跨帧 Deep Filtering | 高（~2M 参数、RTF<0.2） |
| Conv-TasNet | 时域 | 学习的编解码 + 时域掩码 | 中 |
| DTLN | 时域+频域双路 | 轻量 LSTM 双路 | 高（约 1M 参数量级） |
| FullSubNet | 频域 | 全频带+子频带融合 | 中 |

记忆锚点：**DeepFilterNet = 频域 + 跨帧复数滤波；Conv-TasNet = 时域；DTLN = 轻量实时**。

### 2.4 DeepFilterNet 1 / 2 / 3 的演进

| 版本 | 出处 | 相对上一代的改进 |
| --- | --- | --- |
| DeepFilterNet | ICASSP 2022（Schröter, Escalante-B., Rosenkranz, Maier） | 提出 ERB 增益 + Deep Filtering 两级框架 |
| DeepFilterNet2 | IWAENC 2022（同一团队） | 训练策略、数据增强与网络结构优化，嵌入设备实时化（论文报告笔记本 Core-i5 上 RTF 0.04） |
| DeepFilterNet3 | Interspeech 2023 demo 论文《DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement》 | 进一步改进降噪表现与低 SNR 稳健性；是 pip 包的默认模型 |

三者共用同一框架与开源仓库（作者 Hendrik Schröter 等，FAU）；仓库采用 MIT / Apache-2.0 双许可。

---

## 3. 实现与验证（交付核心）

### 3.1 跑通 DeepFilterNet 推理

```bash
pip install deepfilternet
```

```python
from df.enhance import enhance, init_df
import torch
import soundfile as sf
import numpy as np
import librosa

# 初始化模型（首次运行自动下载预训练权重；默认加载 DeepFilterNet3）
model, df_state, _ = init_df()
print("模型采样率:", df_state.sr())          # 48000

noisy, sr = sf.read("noisy.wav", dtype="float32")
if noisy.ndim > 1:
    noisy = noisy.mean(axis=1)               # 混缩为单声道

# DeepFilterNet 工作在 48 kHz 全频带；若输入 16 kHz 先重采样
if sr != 48000:
    noisy48 = librosa.resample(noisy, orig_sr=sr, target_sr=48000)
else:
    noisy48 = noisy

# 注意：enhance() 要求输入形状为 (声道数, 样本数)，一维会报错
audio = torch.from_numpy(noisy48).unsqueeze(0)      # (1, N)
enhanced = enhance(model, df_state, audio)          # 输出 (1, N)
sf.write("enhanced_dfnet.wav", enhanced.squeeze(0).numpy(), 48000)
print("已输出 enhanced_dfnet.wav")
```

**注意**：
- DeepFilterNet 是 **48 kHz 全频带**模型，这是它音质好的原因之一（覆盖全频）。若你的数据是 16 kHz，先升采样到 48 kHz 再增强，输出再降回——升采样本身不引入信息，但让模型在全频带工作；
- `enhance()` 的输入必须是 `(声道, 样本)` 的二维张量，直接喂一维会报 "dimensionality mismatch"；
- 若你的 `torchaudio` 版本较新（2.9+ 移除了 `torchaudio.backend` 模块），旧版 `deepfilternet` 包导入时可能报 `No module named 'torchaudio.backend'`——升级 deepfilternet 或降级 torchaudio 即可。

### 3.2 与维纳滤波比 DNSMOS

DNSMOS 是微软的**无参考**深度学习语音质量打分（1-5，越高越好），不需干净参考，适合打榜与真实场景评估：

```python
# 用微软 DNSMOS ONNX（需从 DNS Challenge 仓库获取模型文件）
# 这里给出评测流程骨架
import onnxruntime as ort

def dnsmos_score(wav_path, sr=16000):
    """调用 DNSMOS ONNX 打分（占位：需本地 SIG/BAK/OVRL 三个模型）。"""
    # sess = ort.InferenceSession("sig.onnx")
    # ... 前向 ...
    # return {"SIG": ..., "BAK": ..., "OVRL": ...}
    raise NotImplementedError("请下载 DNS Challenge 的 DNSMOS ONNX 后实现")
```

**评测流程**（交付）：

1. 取 DNS Challenge 测试集若干条带噪语音（或用你的 `mix_at_snr` 合成）；
2. 分别用「维纳滤波（第 1 周）」与「DeepFilterNet3」增强；
3. 对两路输出算 DNSMOS（或无 DNSMOS 时用 PESQ/STOI + 试听）；
4. 输出对比表：方法 × {OVRL, SIG, BAK}。

**预期**：DeepFilterNet3 的 BAK（背景干净度）显著高于维纳滤波，SIG（语音失真度）接近——即"更干净且不伤语音"。

### 3.3 测 CPU 延迟（端侧预热）

DeepFilterNet 官方仓库的 Release 页直接提供预训练 **ONNX 模型**（`DeepFilterNet3_onnx.tar.gz` 等），配套的 Rust 实现以流式循环跑 ONNX——这正是它作为 LADSPA 插件实时降噪的方式。用 `onnxruntime` 测延迟的通用骨架：

```python
import onnxruntime as ort
import numpy as np
import time

# 模型文件从官方 Release 下载；输入名/形状以 sess.get_inputs() 打印为准
sess = ort.InferenceSession("deepfilternet3.onnx",
                            providers=["CPUExecutionProvider"])
print([(i.name, i.shape) for i in sess.get_inputs()])

# 按模型要求的输入构造 dummy 数据后计时（示例）：
# dummy = np.random.randn(1, 48000).astype(np.float32)   # 1 秒 @ 48kHz
# t0 = time.time()
# _ = sess.run(None, {sess.get_inputs()[0].name: dummy})
# cpu_time = time.time() - t0
# print(f"CPU 处理 1s 音频耗时 {cpu_time:.3f}s, RTF={cpu_time:.3f}")
```

也可以直接用 PyTorch 后端测 RTF（无需 ONNX）：

```python
import time
audio = torch.randn(1, 48000)                 # 1 秒 @ 48 kHz
t0 = time.time()
_ = enhance(model, df_state, audio)
dt = time.time() - t0
print(f"CPU 处理 1s 音频耗时 {dt:.3f}s, RTF={dt:.3f}")   # 期望 < 1
```

（不同机器数值不同；本教程在某笔记本 CPU 上实测 RTF ≈ 0.07。重点是**建立"测 RTF 判断能否端侧"的方法**。ONNX 的输入名与流式接口因版本而异，以 `sess.get_inputs()` 与官方仓库文档为准。）

---

## 4. 工程权衡与失效模式

### 4.1 权衡

- **频带宽度**：48 kHz 全频带音质好但算力高；16 kHz 省算力但高频缺失。端侧按需裁剪；
- **前瞻（lookahead）**：前瞻越多质量越好、延迟越大。DeepFilterNet3 用 20 ms 前瞻；零前瞻方案质量略降但延迟减半。按场景折中；
- **模型大小**：DeepFilterNet3 约 2.1M 参数，极小——这是它端侧可行的根本。不要为了精度换大模型而牺牲实时性。

### 4.2 失效模式

1. **采样率不匹配**：16 kHz 输入直接喂 48 kHz 模型 → 输出错误。修复：统一升/降采样。
2. **过度抑制**：把轻声语音当噪声压掉。症状：弱语音段发虚；修复：调低降噪强度参数。
3. **音乐噪声残留**：极低 SNR 下仍有伪影。深度方法比谱减法好很多，但极端场景仍存在。
4. **音乐/歌唱失真**：降噪模型对"非语音"信号（音乐、歌声）处理差，可能当噪声压。场景不适配时慎用。

---

## 5. 延伸思考题（含解析）

**Q1**：Deep Filtering 相比纯掩码估计的本质提升是什么？
**A**：掩码是对每个时频点的独立标量缩放（且 DFN 的 ERB 增益更是每个频带一个实数）；Deep Filtering 为每个低频频点学一个跨 5 帧的复数滤波器，能利用帧间相关性加权合成、并同时修正相位，表达力是掩码的严格超集（掩码 = 单帧特例），对谐波增强与噪声抖动抑制尤其有效。

**Q2**：ERB 分支为什么能兼顾"大感受野"与"低计算量"？
**A**：481 个频率 bin 先按听觉刻度压缩成 32 个 ERB 频带，卷积与 GRU 都在 32 维的粗粒度上运行，运算量下降一个数量级；而频谱包络——决定听感主体的部分——恰好不需要更细的分辨率。再与帧级分支融合，"金字塔"式的多分辨率换来了效率。

**Q3**：为什么端侧实时降噪强调"短前瞻 + 小核"而不是堆大模型？
**A**：实时通信能接受的算法延迟有限（DeepFilterNet3 为 40 ms：20 ms 帧 + 20 ms 前瞻），看完整句的非因果模型不可用；端侧算力有限，大核/大模型算不动。短前瞻、小滤波核、多分辨率是延迟-精度-算力的工程最优解。

**Q4**：DNSMOS 的"无参考"特性为什么重要？它和 PESQ 的区别？
**A**：真实场景没有干净参考，PESQ 需要参考故只能用合成数据评测；DNSMOS 直接给真实录音打分，适合线上质量监控与打榜。代价是它是学习出来的代理指标，需与主观听感校准。

**Q5**：DeepFilterNet 是 48 kHz 全频带，你的端侧助手是 16 kHz，怎么取舍？
**A**：若端侧只要语音可懂度，可裁剪/换 16 kHz 轻量模型（如 DTLN）省算力；若要全频音质，升采样到 48 kHz 处理再降回，但算力开销大。按助手音质需求与算力预算决策。

---

## 本周交付清单

- [ ] 跑通 DeepFilterNet3 推理，输出增强音频。
- [ ] 与第 1 周维纳滤波对比（DNSMOS 或 PESQ/STOI + 试听），输出对比表。
- [ ] 加载官方 ONNX（或用 PyTorch 后端）测 CPU RTF，验证 < 1。
- [ ] 能解释 Deep Filtering 与掩码估计的本质区别。
