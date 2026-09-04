# 第 2 周教程：AI 深度降噪——DeepFilterNet 系

> **本周要回答的四个问题**
> 1. 从谱减法到维纳滤波都是"估增益"，深度降噪为什么换了一条路？
> 2. DeepFilterNet 的 Deep Filtering 在做什么？为什么比直接预测掩码好？
> 3. 实时约束（10 ms 帧、20 ms 延迟、RTF<1）是怎么做到的？
> 4. Conv-TasNet / DTLN / FullSubNet 这些流派和 DeepFilterNet 差在哪？

对应学习计划：第 2 周。交付物：跑通 DeepFilterNet3 推理，与第 1 周维纳滤波在 DNS 测试集上比 DNSMOS；导出 ONNX 测 CPU 延迟。

---

## 1. 第一性原理：从"估增益"到"学滤波"

### 1.1 掩码估计路线的局限

第 1 周的维纳滤波本质是**逐时频单元独立估增益** $G[m,k]$。深度学习最初也走这条路：网络预测一个理想比值掩码（IRM）或复数掩码，乘到带噪谱上：

$$
\hat{S}[m,k] = M[m,k] \cdot Y[m,k]
$$

这条路的天花板在于：**掩码是对每个时频点独立的标量缩放**。它无法表达"用相邻多个时频点的信息加权合成当前点"——而这恰恰是处理混响尾、宽带噪声所需要的。

### 1.2 Deep Filtering：把"增益"升级为"时频卷积核"

DeepFilterNet 的核心创新：不再预测单个增益，而是预测**每个时频点的一个小型复数滤波器**（沿时间和频率两个方向展开的卷积核），用这个核去卷积带噪谱：

$$
\hat{S}[m,k] = \sum_{\Delta m, \Delta k} h_{m,k}[\Delta m, \Delta k] \cdot Y[m+\Delta m, k+\Delta k]
$$

其中 $h_{m,k}$ 是网络为位置 $(m,k)$ 预测的**时频滤波器系数**（如 $5 \times 5 = 25$ 个复数权重）。

**为什么这更强**：

1. **跨时频利用信息**：每个输出点由邻域加权合成，天然能利用相关性去噪、追踪混响尾；
2. **复数滤波**：同时处理幅度与相位，相位估计不再靠"保留带噪相位"这种妥协；
3. **仍是线性滤波**：输出是输入的线性组合，保真度高、不易引入非线性伪影（对比纯生成式方法）。

**直观对比**：掩码估计 = 每个像素单独调亮度；Deep Filtering = 给每个像素学一个局部卷积核。后者的表达能力是前者的严格超集（掩码是 $1\times1$ 滤波器的特例）。

### 1.3 双分辨率特征：帧级 + 组级

DeepFilterNet 的 Encoder 走两条路：

- **帧级分支**：对每帧 80 维特征做处理，捕捉细节；
- **组级分支（group level）**：把相邻若干帧打包成"组"，下采样后建模更长时序结构（如混响尾、节奏）。

两路特征融合后送入解码。**组级分支的意义**：用低计算量的下采样通路获得大感受野，这是"小模型、低延迟、大感受野"三角的关键——也是它能做到实时的架构原因。

---

## 2. 实时性：工程约束如何塑造架构

### 2.1 延迟预算

DeepFilterNet 的设计目标：

- 帧长 20 ms、帧移 10 ms；
- 算法延迟约 20 ms（一帧缓冲）；
- CPU 单核 **RTF < 1**（比实时快）。

### 2.2 做到实时的三个手段

1. **因果设计**：所有卷积/递归只看过去帧，无未来依赖 → 可流式逐帧处理；
2. **Deep Filtering 核很小**：每个点的滤波器只有 $5\times5$，计算量远小于全时频注意力；
3. **组级下采样**：长时建模走低分辨率通路，避免在高分辨率上算大感受野。

**这给端侧部署的启示**：实时降噪不靠堆大模型，靠"因果 + 小核 + 多分辨率"的架构纪律。你的结课助手前端（Stage 7）就用这类模型。

### 2.3 流派全景对比

| 模型 | 域 | 核心机制 | 端侧友好度 |
| --- | --- | --- | --- |
| **DeepFilterNet2/3** | 频域 | Deep Filtering 时频滤波 | 高（因果、小核） |
| Conv-TasNet | 时域 | 学习的编解码 + 时域掩码 | 中 |
| DTLN | 时域+频域双路 | 轻量 LSTM 双路 | 高（专为实时） |
| FullSubNet | 频域 | 全频带+子频带融合 | 中 |

记忆锚点：**DeepFilterNet = 频域 + 时频滤波；Conv-TasNet = 时域；DTLN = 轻量实时**。

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

# 初始化模型（首次自动下载权重）
model, df_state, _ = init_df()

noisy, sr = sf.read("noisy.wav", dtype="float32")
if noisy.ndim > 1:
    noisy = noisy.mean(axis=1)
assert sr == 48000 or True  # DeepFilterNet 用 48kHz，若 16k 需先升采样

# DeepFilterNet 工作在 48kHz 全频带；若输入 16kHz 先重采样
import librosa
if sr != 48000:
    noisy48 = librosa.resample(noisy, orig_sr=sr, target_sr=48000)
else:
    noisy48 = noisy

enhanced = enhance(model, df_state, torch.from_numpy(noisy48))
sf.write("enhanced_dfnet.wav", enhanced.numpy(), 48000)
print("已输出 enhanced_dfnet.wav")
```

**注意**：DeepFilterNet 是 **48 kHz 全频带**模型，这是它音质好的原因之一（覆盖全频）。若你的数据是 16 kHz，先升采样到 48 kHz 再增强，输出再降回——升采样本身不引入信息，但让模型在全频带工作。

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

### 3.3 导出 ONNX 测 CPU 延迟（端侧预热）

```python
# DeepFilterNet 官方提供 ONNX 导出（df.enhance 生态）
# 导出后用 onnxruntime 测 CPU 延迟
import onnxruntime as ort
import time

sess = ort.InferenceSession("deepfilternet.onnx",
                            providers=["CPUExecutionProvider"])
dummy = np.random.randn(1, 48000).astype(np.float32)  # 1秒示例
t0 = time.time()
_ = sess.run(None, {"input": dummy})
cpu_time = time.time() - t0
print(f"CPU 处理 1s 音频耗时 {cpu_time:.3f}s, RTF={cpu_time:.3f}")
# 期望 RTF < 1（比实时快），证明可在 CPU 端侧运行
```

（实际导出命令以 DeepFilterNet 官方仓库为准；重点是**建立"测 RTF 判断能否端侧"的方法**。）

---

## 4. 工程权衡与失效模式

### 4.1 权衡

- **频带宽度**：48 kHz 全频带音质好但算力高；16 kHz 省算力但高频缺失。端侧按需裁剪；
- **因果性**：因果（实时）精度低于非因果（离线）。实时通信必须因果；
- **模型大小**：DeepFilterNet 约 2M 参数，极小——这是它端侧可行的根本。不要为了精度换大模型而牺牲实时性。

### 4.2 失效模式

1. **采样率不匹配**：16 kHz 输入直接喂 48 kHz 模型 → 输出错误。修复：统一升/降采样。
2. **过度抑制**：把轻声语音当噪声压掉。症状：弱语音段发虚；修复：调低降噪强度参数。
3. **音乐噪声残留**：极低 SNR 下仍有伪影。深度方法比谱减法好很多，但极端场景仍存在。
4. **音乐/歌唱失真**：降噪模型对"非语音"信号（音乐、歌声）处理差，可能当噪声压。场景不适配时慎用。

---

## 5. 延伸思考题（含解析）

**Q1**：Deep Filtering 相比纯掩码估计的本质提升是什么？
**A**：掩码是对每个时频点的独立标量缩放；Deep Filtering 为每个点学一个跨时频的复数卷积核，能利用邻域信息加权合成，表达力是掩码的严格超集（掩码=1×1 特例），尤其利于去混响尾与宽带噪声。

**Q2**：组级（group-level）分支为什么能兼顾"大感受野"与"低计算量"？
**A**：它把相邻帧打包下采样，在低分辨率上建模长时序，感受野大但运算量小；再与帧级高分辨率分支融合。相当于用"金字塔"换效率。

**Q3**：为什么端侧实时降噪强调"因果 + 小核"而不是堆大模型？
**A**：实时要求延迟≤20ms，不能看未来（因果）；端侧算力有限，大核/大模型算不动。因果小核 + 多分辨率是延迟-精度-算力的工程最优解。

**Q4**：DNSMOS 的"无参考"特性为什么重要？它和 PESQ 的区别？
**A**：真实场景没有干净参考，PESQ 需要参考故只能用合成数据评测；DNSMOS 直接给真实录音打分，适合线上质量监控与打榜。代价是它是学习出来的代理指标，需与主观听感校准。

**Q5**：DeepFilterNet 是 48 kHz 全频带，你的端侧助手是 16 kHz，怎么取舍？
**A**：若端侧只要语音可懂度，可裁剪/换 16 kHz 轻量模型（如 DTLN）省算力；若要全频音质，升采样到 48 kHz 处理再降回，但算力开销大。按助手音质需求与算力预算决策。

---

## 本周交付清单

- [ ] 跑通 DeepFilterNet3 推理，输出增强音频。
- [ ] 与第 1 周维纳滤波对比（DNSMOS 或 PESQ/STOI + 试听），输出对比表。
- [ ] 导出/加载 ONNX，测 CPU RTF，验证 < 1。
- [ ] 能解释 Deep Filtering 与掩码估计的本质区别。
