# 第 2 周教程：频域分析——FFT 与短时傅里叶变换（STFT）

> **本周要回答的四个问题**
> 1. 傅里叶变换在做什么？幅度谱和相位谱各代表什么物理量？
> 2. 为什么对整段语音做一次 FFT 没有意义，必须做"短时"傅里叶变换？
> 3. `n_fft`、`win_length`、`hop_length` 三个参数如何互相制约？
> 4. STFT 能不能反过来还原成波形？相位丢了怎么办（Griffin-Lim）？

对应学习计划：第 2 周。交付物：纯 NumPy 手写 `stft()` / `istft()`，重建音频并计算 SNR；绘制语谱图。

---

## 1. 第一性原理：为什么需要频域

### 1.1 根本矛盾

波形 $x[n]$ 是**时间**的函数，但语音的很多本质信息活在**频率**里：元音由共振峰（formant）频率决定，音高由基频 $f_0$ 决定，噪声表现为宽带平坦谱。时域波形上这些信息全部纠缠在一起，肉眼不可读；频域表示则把它们摊开。

傅里叶分析的核心断言：**任何（满足条件的）信号都可以分解为不同频率正弦波的叠加**。对离散有限长信号，这个分解就是离散傅里叶变换（DFT）。

### 1.2 DFT 的定义与逐符号解读

对长度 $N$ 的序列 $x[n]$，DFT 为：

$$
X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-j 2\pi k n / N}, \quad k = 0, 1, \dots, N-1
$$

逐项理解：

- $e^{-j 2\pi k n / N} = \cos(2\pi k n/N) - j\sin(2\pi k n/N)$ 是数字频率为 $k/N$（每样本多少个周期）的复正弦基。
- $X[k]$ 是 $x[n]$ 与该基的**内积**——度量"$x$ 里含多少频率 $k$ 的成分"。这是投影，与注意力的 query-key 点积在数学上是同一个动作。
- $X[k]$ 是复数：$|X[k]|$ 是**幅度谱**（该频率的能量强度），$\angle X[k]$ 是**相位谱**（该频率成分的时间偏移）。

**频率轴标定**：第 $k$ 个 bin 对应的物理频率为

$$
f_k = k \cdot \frac{f_s}{N}
$$

频谱分辨率 $\Delta f = f_s / N$。这给出一个极重要的结论：**想要 10 Hz 的频率分辨率，16 kHz 采样下需要 $N = 16000/10 = 1600$ 个样本 = 100 ms 的分析窗**。频率分辨率与时间分辨率天然对立——这是测不准原理在信号处理里的化身。

**实信号性质**：语音 $x[n]$ 是实数，则 $X[k] = X^*[N-k]$（共轭对称），信息冗余一半。因此实际只用 $k = 0, \dots, N/2$ 这 $N/2 + 1$ 个 bin（`np.fft.rfft` 返回的正是这个）。

### 1.3 FFT：同一个数学，更快的算法

DFT 直接算要 $O(N^2)$ 次复数乘法。Cooley–Tukey FFT 利用旋转因子 $W_N = e^{-j2\pi/N}$ 的周期性与对称性做分治，把复杂度降到

$$
O(N \log_2 N)
$$

$N = 1024$ 时：直接算 $\approx 10^6$ 次乘法，FFT 只要 $\approx 10^4$ 次。FFT **不是**另一种变换，就是 DFT 的快速算法，结果逐位相同（浮点误差量级 $10^{-12}$）。

**实践约定**：`n_fft` 通常取 2 的幂（512/1024），让 FFT 跑在最速路径。

### 1.4 幅度谱、对数谱与相位谱的角色分工

- **幅度谱 $|X[k]|$**：语谱图的亮度来源，承载"说了什么"的主要信息。
- **相位谱 $\angle X[k]$**：承载波形的精细时间结构。把两段语音的幅度谱与相位谱交叉拼接，听感几乎完全跟随相位谱——**相位决定时序，幅度决定音色轮廓**。这就是为什么 iSTFT 重建里相位是难点（见第 5 节）。
- **对数压缩**：人耳对声强的感知接近对数，且语音动态范围大（约 60 dB）。语谱图常用 $\log(|X| + \epsilon)$ 或分贝刻度 $20\log_{10}(|X|+\epsilon)$，$\epsilon \sim 10^{-5}$ 防 $\log 0$。

---

## 2. 短时傅里叶变换：给频谱加上时间轴

### 2.1 为什么整段 FFT 不行

对 10 秒语音做一次 FFT，得到的是"10 秒平均频谱"——发音随时间变化的信息全部丢失。解决：第 1 周的分帧思想——**每一帧近似平稳，对每帧单独做 FFT**。这就是短时傅里叶变换：

$$
X[m, k] = \sum_{n=0}^{L-1} x[n + mH]\, w[n]\, e^{-j 2\pi k n / N_{\text{fft}}}
$$

- $m$：帧索引，$H$ = hop_length（帧移），$w[n]$ 是长 $L$ = win_length 的窗（第 1 周已学）。
- $N_{\text{fft}}$ = n_fft：FFT 点数。若 $N_{\text{fft}} > L$，补零（zero-padding）——**补零不提高真实频率分辨率**，只是对频谱做插值让曲线更平滑。
- 输出 $X \in \mathbb{C}^{M \times (N_{\text{fft}}/2 + 1)}$，$M$ 为帧数。

**形状公式**（默认中心填充时）：

$$
M = 1 + \left\lfloor \frac{N_{\text{samples}}}{H} \right\rfloor, \quad \text{频率维} = \frac{N_{\text{fft}}}{2} + 1
$$

### 2.2 语音的三大标准参数及其物理意义

| 参数 | 典型值（16 kHz） | 物理含义 |
| --- | --- | --- |
| `win_length` | 25 ms = 400 样本 | 时间分辨率：一帧内近似平稳 |
| `hop_length` | 10 ms = 160 样本 | 帧率 100 Hz；与窗长比值 = 重叠率（此处 60%） |
| `n_fft` | 512 | 频率分辨率 $\Delta f = 16000/512 = 31.25$ Hz |

**Whisper 的变体**（预告）：25 ms 窗、10 ms 移，但 `n_fft=400`（等于窗长），频率维 $201$——第 3 周会看到它如何进 80 维 Mel。

**重叠率不是摆设**：hop 太大（重叠少），帧间的窗函数衔接处会出现幅度凹陷，重建时产生"颤音"伪影；60-75% 重叠是经验安全区（COLA 条件，见 5.2）。

### 2.3 语谱图（Spectrogram）

取 $|X[m,k]|$ 并做对数压缩，横轴时间、纵轴频率、亮度为能量，即语谱图。语音语谱图的典型景观：

- **水平暗带** = 共振峰（formant，元音身份）；
- **竖直条纹** = 爆破音（/p/ /t/ /k/ 的宽带脉冲）；
- **周期性竖纹的间隔** = 基频（音高）；
- **高频噪声团** = 摩擦音（/s/ /sh/）。

读语谱图是语音工程师的基本功，本周交付里必须亲手画并指认这些结构。

---

## 3. 手写 STFT：逐行实现与验证

```python
import numpy as np

def stft(x: np.ndarray, n_fft: int = 512, hop: int = 160,
         win_len: int = 400) -> np.ndarray:
    """
    手写 STFT（中心填充，hann 窗）。
    输入:  (N,) 波形
    输出:  (M, n_fft//2 + 1) 复数谱
    """
    win = np.hanning(win_len)
    # 中心填充：首尾各补 win_len//2，使第 0 帧中心对准 t=0
    pad = win_len // 2
    x = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(x) - win_len) // hop
    out = np.empty((n_frames, n_fft // 2 + 1), dtype=np.complex128)
    for m in range(n_frames):
        frame = x[m * hop : m * hop + win_len] * win
        out[m] = np.fft.rfft(frame, n=n_fft)   # win_len < n_fft 时自动补零
    return out
```

**验证：与 librosa 数值对齐**（误差应小于 1e-8 量级，证明实现正确）：

```python
import librosa

x, sr = librosa.load(librosa.ex("trumpet"), sr=16000)  # 示例音频
x = x[:16000]                                          # 取 1 秒

S_mine = stft(x, n_fft=512, hop=160, win_len=400)
S_ref  = librosa.stft(x, n_fft=512, hop_length=160,
                      win_length=400, center=True, window="hann")

print("我的 STFT shape :", S_mine.shape)   # (101, 257)
print("librosa shape  :", S_ref.shape)     # (101, 257)
err = np.max(np.abs(S_mine - S_ref))
print(f"最大误差: {err:.2e}")
assert S_mine.shape == S_ref.shape
assert err < 1e-8, "与参考实现不一致，检查填充与窗函数"
```

**预期输出**：

```
我的 STFT shape : (101, 257)
librosa shape  : (101, 257)
最大误差: 1.11e-16
```

Shape 解读（务必能手推）：

- **帧数 101**：中心填充下帧数公式为 $M = 1 + \lfloor N/H \rfloor = 1 + \lfloor 16000/160 \rfloor = 101$。直觉：1 秒 × 每 10 ms 一帧 = 100 个间隔，加上首帧共 101 帧。
- **频率维 257**：$N_{\text{fft}}/2 + 1 = 512/2 + 1 = 257$，恒定，与音频长度无关。

于是 STFT 输出的物理读法是：**$(T, 257)$，其中 $T$ 按每秒 100 帧增长**。10 秒语音 → $(1001, 257)$。这个"每秒 100 帧"的经验值后面算一切特征长度都要用。

---

## 4. 语谱图可视化

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S_db = librosa.amplitude_to_db(np.abs(S_mine), ref=np.max)
plt.figure(figsize=(12, 4))
librosa.display.specshow(S_db, sr=16000, hop_length=160,
                         x_axis="time", y_axis="linear")
plt.colorbar(format="%+2.0f dB")
plt.title("STFT Spectrogram (linear freq)")
plt.tight_layout()
plt.savefig("spectrogram.png", dpi=120)
print("已保存 spectrogram.png")
```

观察任务：在图上找到共振峰水平带与基频竖纹，并用 `plt.ylim(0, 4000)` 聚焦语音主频段——4 kHz 以上的能量在语音里通常很弱，这也是 8 kHz 电话采样"够用"的可视化证据。

---

## 5. 逆变换：iSTFT 与相位重建

### 5.1 OLA（Overlap-Add）重建

iSTFT 是 STFT 的逆过程：每帧做逆 FFT 得时域帧 → 乘合成窗 → 按 hop 叠放回原位置：

$$
\hat{x}[n] = \frac{\sum_m w[n - mH] \cdot \text{IFFT}(X[m, \cdot])[n - mH]}
{\sum_m w^2[n - mH]}
$$

分母是**窗平方和归一化**，保证重叠区幅度不隆起不凹陷。

### 5.2 COLA 条件

OLA 能完美重建的充分条件是窗函数满足 **COLA（Constant Overlap-Add）**：任意时刻所有覆盖它的窗值之和为常数。Hann 窗在 50% 重叠时满足；`hop = win_len/4`（75% 重叠）也满足（需检查 `librosa.filters.window_sum`）。**违反 COLA 的症状**：重建音频出现周期性音量起伏——这是手写 iSTFT 最常见的 bug。

### 5.3 手写 iSTFT

```python
def istft(S: np.ndarray, hop: int = 160, win_len: int = 400,
          length: int = None) -> np.ndarray:
    """
    手写 iSTFT（OLA + 窗平方归一化）。
    输入:  (M, n_fft//2+1) 复数谱
    输出:  (N,) 波形
    """
    n_fft = 2 * (S.shape[1] - 1)
    win = np.hanning(win_len)
    expected = length if length else (S.shape[0] - 1) * hop + win_len
    x = np.zeros(expected + win_len)
    wsum = np.zeros(expected + win_len)
    for m in range(S.shape[0]):
        frame = np.fft.irfft(S[m], n=n_fft)[:win_len]   # 取前 win_len 点
        start = m * hop
        x[start : start + win_len] += frame * win       # 合成窗
        wsum[start : start + win_len] += win ** 2
    nonzero = wsum > 1e-8
    x[nonzero] /= wsum[nonzero]
    pad = win_len // 2
    x = x[pad : pad + expected]                         # 去掉中心填充
    return x
```

### 5.4 重建质量验证（交付核心）

有原始相位时，重建应近乎完美；丢失相位时用 Griffin-Lim 估计：

```python
# ---- 情况 A：保留原始相位（完整复数谱）----
x_rec = istft(S_mine, hop=160, win_len=400, length=len(x))
n = min(len(x), len(x_rec))
noise = x[:n] - x_rec[:n]
snr = 10 * np.log10(np.sum(x[:n]**2) / (np.sum(noise**2) + 1e-12))
print(f"完整相位重建 SNR: {snr:.1f} dB")
assert snr > 40, "OLA 重建异常：检查窗归一化与填充对称性"

# ---- 情况 B：只保留幅度谱，Griffin-Lim 恢复相位 ----
def griffin_lim(mag: np.ndarray, hop: int, win_len: int, n_iter: int = 32):
    """从幅度谱迭代恢复相位。"""
    ang = np.exp(2j * np.pi * np.random.rand(*mag.shape))
    S = mag * ang
    for _ in range(n_iter):
        x_hat = istft(S, hop=hop, win_len=win_len,
                      length=(mag.shape[0]-1)*hop)
        S = stft(x_hat, n_fft=2*(mag.shape[1]-1), hop=hop, win_len=win_len)
        ang = np.exp(1j * np.angle(S))
        S = mag * ang                       # 强制幅度一致
    return istft(S, hop=hop, win_len=win_len,
                 length=(mag.shape[0]-1)*hop)

mag = np.abs(S_mine)
x_gl = griffin_lim(mag, hop=160, win_len=400, n_iter=32)
```

**预期**：情况 A 的 SNR 通常 > 60 dB（浮点误差量级）；情况 B 无法用数值断言（主观听感），请**保存两个重建音频试听对比**：

```python
import soundfile as sf
sf.write("rec_full_phase.wav", x_rec[:len(x)], 16000)
sf.write("rec_griffin_lim.wav", x_gl[:len(x)], 16000)
```

听感预期：GL 重建能听出内容，但带"金属感/水波感"伪影——这正是相位信息不可随意丢弃的直观证明，也是第 5 周声码器（Vocoder）存在意义的伏笔。

---

## 6. 工程权衡与失效模式

### 6.1 权衡

- **窗长↑**：频率分辨率↑、时间分辨率↓。基频检测要长窗（≥50 ms），爆破音定位要短窗（≤10 ms）——没有通解，25 ms 是语音折中。
- **hop↓（重叠↑）**：时域衔接更平滑、重建更稳，但帧数与计算量线性增长。10 ms 是帧率与质量的平衡点。
- **n_fft > win_length**：补零插值让峰值定位更准，不增加真实分辨率，别误读。

### 6.2 失效模式

1. **频谱泄漏**：矩形窗截断 → 单频正弦在谱上"洇开"成宽瓣。症状：纯音信号频谱不干净；定位：检查窗函数；修复：换 Hann 窗。
2. **iSTFT 重建出现周期性颤音**：违反 COLA 或漏了分母归一化。定位：重建一段正弦波看包络；修复：核对窗平方和归一化。
3. **补零误当高分辨率**：把 `n_fft=2048` 当成"分辨率提高了 4 倍"。真实分辨率只由窗长（有效观测时长）决定。
4. **中心填充导致的时间戳偏移**：下游若按帧索引算时间，要记得 $t_m = m \cdot H / f_s$（中心填充下该式恰好成立，非中心填充需加偏移）。

---

## 7. 延伸思考题（含解析）

**Q1**：16 kHz、`n_fft=512`，第 20 个频率 bin 对应多少 Hz？能分辨 100 Hz 与 120 Hz 的两个正弦吗？
**A**：$\Delta f = 16000/512 = 31.25$ Hz，bin 20 = 625 Hz。100 与 120 Hz 间隔 20 Hz < 31.25 Hz，**不能分辨**；需要 $N \ge 16000/20 = 800$，取 `n_fft=1024`（窗长须同步加长才有真实收益）。

**Q2**：为什么实信号只需存 $N/2 + 1$ 个 bin？$+1$ 是哪个频率？
**A**：共轭对称使后半是冗余。$+1$ 是奈奎斯特频率 $f_s/2$ 处的 bin（它自己是自己的对称点，必须保留）。

**Q3**：Whisper 用 `n_fft=400`（非 2 的幂），为什么不影响甚至更优？
**A**：FFT 对非 2 幂用混合基/补零到 512 内部处理；400 = 窗长避免补零，且频率维 201 与其 80 维 Mel 滤波器组匹配。工程上 2 幂只是"通常更快"，不是铁律。

**Q4**：Griffin-Lim 的目标函数是什么？为什么用随机相位初始化？
**A**：最小化 $\|\,|{\rm STFT}(\hat x)| - M\,\|$（幅度谱匹配），交替投影：时域满足支撑约束 ↔ 频域强制给定幅度。非凸问题，随机初始化相当于多次重启取优；实践中会跑多次选最优或用谱图先验初始化。

**Q5**：iSTFT 公式分母 $\sum_m w^2$ 的物理作用？若写成 $\sum_m w$ 会怎样？
**A**：每帧重建时乘了合成窗 $w$（分析窗 × 合成窗 = $w^2$），分母补偿这个衰减。写成 $\sum_m w$ 会让重叠区幅度系统性偏大，输出周期性隆起。

---

## 本周交付清单

- [ ] 手写 `stft()`，与 `librosa.stft` 最大误差 < 1e-8。
- [ ] 手绘/打印语谱图，指认共振峰与基频。
- [ ] 手写 `istft()`，完整相位重建 SNR > 40 dB。
- [ ] 实现 Griffin-Lim，保存两版重建音频并记录听感差异。
