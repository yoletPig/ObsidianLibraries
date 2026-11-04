# 第 4 周教程：AGC、VAD 与主动降噪（ANC）

> **本周要回答的三个问题**
> 1. AGC 如何"追踪"目标响度？起音/释放时间为什么必须不对称？
> 2. VAD 从能量检测、统计模型（WebRTC 的 GMM）到 DNN（Silero）演进的原因是什么？
> 3. ANC 的次级路径估计 $\hat{S}(z)$ 为什么是系统稳定的命门？失配会怎样？

对应学习计划：第 4 周。交付物：① 手写 AGC（目标响度 RMS 近似 + 不对称平滑增益 + 限幅）；② FXLMS 数值仿真，画次级路径失配对残余噪声的影响；③ 集成 Silero VAD。

---

## 1. AGC：自动增益控制

### 1.1 根本矛盾

说话人远近、音量起伏，导致信号幅度剧烈波动。太大会削波失真，太小会淹没在噪声里。AGC 的目标：**把信号响度稳定追踪到一个目标值**，且过渡平滑无抽吸感。

### 1.2 两级结构

- **数字增益**：在数字域乘一个时变增益 $g[n]$；
- **模拟增益**：调整麦克风/放大器增益（硬件级，范围更大）。

数字域我们聚焦 $g[n]$ 的计算。

### 1.3 核心：包络检测 + 平滑增益

```
目标响度 R_target（如 -23 dBFS RMS；广播行业标准见 1.4）
当前包络 env（每 10 ms 一帧的 RMS）
期望增益 g_desired = R_target / env
平滑：g[k] = 平滑过渡到 g_desired[k]（攻击/释放不对称）
插值到逐样本 → 施加增益 → 限幅防削波
```

**这个流程在干什么**：期望增益 $g_{\text{desired}} = R_{\text{target}}/\text{env}$ 是"响度差的倒数"——当前比目标低多少 dB，就该放大多少。但**绝不能直接用**：语音的瞬时包络起伏剧烈（浊音强、清音弱），若每个样本都贴着期望增益走，增益会跟着波形抖动，把底噪也按增益调制放大，听感是"喘息/抽吸"（pumping/breathing）。所以要过一道**不对称平滑**。

**攻击/释放不对称**（本周自测题）：

- **攻击（Attack）快**：音量突然变大，要快速压低防削波 → 短时间常数（~5 ms）；
- **释放（Release）慢**：音量变小后，缓慢回升避免"抽吸"（pumping）感 → 长时间常数（~50-100 ms）。

> **类比**：好的音响师跟推子——台上突然爆发时手要快（压下去，防啸叫防爆音）；台上安静下来时手要慢（慢慢抬，否则把座椅挪动的沙沙声也抬起来了）。不对称是同一双手的两个速度。

**工程细节（本节实现里最关键的一处）**：包络要按 **10 ms 帧的 RMS** 计算，而**不能逐样本用 $|x[n]|$**。逐样本包络在信号过零处趋近 0，期望增益飙到上限；再叠加"攻击快、释放慢"的不对称，增益会被波形纹波一路"棘轮式"往下压（每次纹波都触发快速攻击、恢复却很慢），输出会越压越轻。帧级 RMS 天然滤掉了这个纹波。

### 1.4 LUFS 响度标准

LUFS（Loudness Units Full Scale）是广播响度标准：ITU-R BS.1770 定义了 K 加权（模拟人耳频响）+ 门限门控的响度测量，EBU R128 规定广播目标为 **-23 LUFS**；流媒体平台常用 -14 ~ -19 LUFS。它比峰值更接近人耳感知。本周教学实现用 **RMS（dBFS）近似响度**（够用且无需 K 加权滤波器），生产实现请用真实 LUFS 表。

### 1.5 实现（交付）

```python
import numpy as np

def agc(x: np.ndarray, sr: int, target_db: float = -23.0,
        attack_ms: float = 5.0, release_ms: float = 50.0,
        max_gain_db: float = 30.0) -> np.ndarray:
    """
    自动增益控制：帧级 RMS 包络 + 不对称平滑增益 + 限幅。
    """
    frame = int(0.010 * sr)                       # 10 ms 一帧
    n_frames = len(x) // frame
    x_used = x[:n_frames * frame]

    # 1. 帧级 RMS 包络（逐样本包络会随波形纹波抖动，见 1.3 节）
    rms_f = np.sqrt(np.mean(x_used.reshape(n_frames, frame) ** 2, axis=1) + 1e-12)

    # 2. 期望增益：响度差 -> 线性增益，并限幅
    target = 10 ** (target_db / 20.0)             # -23 dBFS -> 线性 0.071
    g_desired = np.clip(target / rms_f, 0.0, 10 ** (max_gain_db / 20.0))

    # 3. 不对称一阶平滑（帧级）：增益下降=攻击(快)，上升=释放(慢)
    a_atk = np.exp(-0.010 / (attack_ms / 1000.0)) # 时间常数 5 ms
    a_rel = np.exp(-0.010 / (release_ms / 1000.0))# 时间常数 50 ms
    g = np.zeros(n_frames)
    g[0] = g_desired[0]
    for k in range(1, n_frames):
        if g_desired[k] < g[k-1]:                 # 需要压低 -> 攻击（快）
            g[k] = a_atk * g[k-1] + (1 - a_atk) * g_desired[k]
        else:                                     # 需要提升 -> 释放（慢）
            g[k] = a_rel * g[k-1] + (1 - a_rel) * g_desired[k]

    # 4. 帧间线性插值到逐样本（避免帧边界增益跳变产生"咔哒"声）
    g_up = np.interp(np.arange(len(x_used)), np.arange(n_frames) * frame, g)
    y = x_used * g_up

    # 5. 限幅防削波
    peak = np.abs(y).max()
    if peak > 1.0:
        y = y / peak * 0.99
    return y

# 验证：音量忽大忽小的输入 -> 输出包络稳定
np.random.seed(0)
sr = 16000
t = np.arange(sr * 3) / sr
# 构造幅度起伏的语音样信号（三段音量差 8 倍）
carrier = np.sin(2*np.pi*200*t)
amp = np.where(t < 1.0, 0.1, np.where(t < 2.0, 0.8, 0.2))
x = carrier * amp
y = agc(x, sr)

# 检查三段输出 RMS 是否趋于一致
def rms(seg): return np.sqrt(np.mean(seg**2))
r1, r2, r3 = rms(y[:sr]), rms(y[sr:2*sr]), rms(y[2*sr:])
print(f"三段输出 RMS: {r1:.3f} {r2:.3f} {r3:.3f}")
assert max(r1, r2, r3) / (min(r1, r2, r3) + 1e-9) < 2.0, "AGC 未稳定响度"
print("AGC 已把三段不同音量稳定到相近响度 ✓")
```

**预期输出**：

```
三段输出 RMS: 0.071 0.072 0.069
AGC 已把三段不同音量稳定到相近响度 ✓
```

**预期**：三段输入音量差 8 倍（0.1 / 0.8 / 0.2），输出 RMS 全部稳定到 ≈ 0.07（即 -23 dBFS 目标）。这就是"自动增益"的最小实证。

---

## 2. VAD：语音活动检测

### 2.1 三代演进

| 代 | 方法 | 特点 |
| --- | --- | --- |
| 传统 | 能量/过零率/谱熵 | 简单、快，但抗噪差 |
| 统计模型 | WebRTC VAD（GMM） | 工业标准、极轻量（CPU 上微秒级） |
| 端到端 DNN | **Silero VAD** | 当前事实标准，鲁棒 |

**为什么这么演进**：能量检测在信噪比稍差时立刻失效（噪声比语音还响）；WebRTC VAD 用统计模型解决"噪声也有能量"的问题——它提取每 10 ms 帧的 6 个子带对数能量作为特征，对每个子带用"语音/噪声"两个高斯分布打分，按似然比判决（即一个极小的 GMM，Gaussian Mixture Model），并提供 0-3 四档激进程度（Python 绑定 `webrtcvad`，仅支持 8/16/32/48 kHz 的 10/20/30 ms 帧）；而 DNN 直接从数据里学"什么听起来像语音"，对音乐、咳嗽、键盘声等未建模声源的泛化远超手工特征——代价是计算量与模型分发。

### 2.2 Silero VAD

预训练的轻量 DNN 模型，输出每帧"是语音"的概率。`torch.hub` 一行加载（首次运行自动下载模型）：

```python
import torch

model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad', trust_repo=True)
(get_speech_timestamps, _, read_audio, _, _) = utils

wav = read_audio('test.wav', sampling_rate=16000)
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
print(speech_timestamps)   # [{'start': 320, 'end': 5120}, ...]
```

**输出**：语音段的起止**样本**索引列表（加 `return_seconds=True` 可改返回秒）。用途：① ASR 前端分段；② 降噪的噪声估计（静音段）；③ 录音只存有语音部分。

> **环境提示**：新版权重，`torchaudio ≥ 2.9` 的音频解码改为依赖 `torchcodec`，`read_audio()` 可能报错。绕过方法：用 `soundfile` 自己读入再传张量——`w = torch.from_numpy(sf.read('test.wav', dtype='float32')[0])`，`get_speech_timestamps()` 同样接受普通张量。

### 2.3 集成进 Pipeline

```python
import numpy as np
import torch

class VADFrontend:
    def __init__(self):
        self.model, self.utils = torch.hub.load(
            'snakers4/silero-vad', 'silero_vad', trust_repo=True)
        self.get_ts = self.utils[0]

    def speech_segments(self, wav, sr=16000):
        return self.get_ts(wav, self.model, sampling_rate=sr)

    def keep_speech(self, wav, sr=16000):
        """拼接所有语音段，返回 numpy 波形。"""
        ts = self.speech_segments(wav, sr)
        wav_np = wav.numpy() if hasattr(wav, "numpy") else wav
        if not ts:
            return wav_np
        return np.concatenate([wav_np[t['start']:t['end']] for t in ts])
```

> **注意**：`read_audio()` 返回的是 **torch 张量**，`np.concatenate` 不能直接拼接张量切片——先转 numpy（如上），或用官方 utils 里的 `collect_chunks()`（`self.utils[4]`）完成同样的事。

---

## 3. 主动降噪（ANC）

### 3.1 物理原理

ANC 不靠算法"滤"噪声，而是**产生一个反相声波物理抵消**噪声：

$$
\text{原始噪声 } d[n] \;+\; \text{反相声波 } -d[n] \;=\; 0
$$

**这个式子在干什么**：声波是气压波动，两个等幅、反相（相位差 180°）的波叠加，气压起伏互相抵消，耳朵听到的就是"安静"。类比：往水里同时扔两个石子，波峰遇上波谷，水面恢复平静。

耳机里的麦克风采集外部噪声，处理器生成反相信号从扬声器（耳机喇叭）播出，两者叠加抵消。这是**物理层的消噪**，与 AI 降噪（信号处理层）互补。

### 3.2 FXLMS：为什么需要次级路径估计

难点：反相信号从"误差麦克风"听回来时，经过了**次级路径** $S(z)$（扬声器 + 耳道传播）——它既有增益也有**延迟和相位旋转**。若控制器不知道这个路径，就无法正确生成反相波。

> **类比**：你朝峡谷对岸喊"喂——"想让回声正好抵消原声。如果你在听到自己的声音"该到"的瞬间就喊，声音还在路上，到达时完全错过了抵消时机——**相位错了**。你必须预判"声音穿过峡谷要多久、被削弱多少"，提前、按正确音量喊。次级路径估计 $\hat{S}(z)$ 就是这张"预判表"。

**FXLMS（Filtered-x LMS）** 的"Filtered-x"正是指：参考信号要先通过次级路径估计 $\hat{S}(z)$ 滤波，才能正确更新：

$$
\mathbf{w} \leftarrow \mathbf{w} + \mu\, e[n]\, \underbrace{(\hat{S}(z) * x)[n]}_{\text{filtered-x}}
$$

**这个公式在干什么**：对比第 3 周 NLMS 的更新式 $\hat{h} += \mu e x$——形式几乎一样，唯一的区别是 $x$ 先过了 $\hat{S}(z)$。为什么必须这样？因为控制器权重 $w$ 影响误差的"通路"是：$w \to$ 扬声器 $\to$ **次级路径 $S(z)$** $\to$ 误差麦克风。梯度下降要求更新方向与"误差对 $w$ 的真实敏感度"一致，而这个敏感度恰好是 $S(z)*x$。我们手里只有估计值 $\hat{S}(z)$，就用它代替。**$\hat{S}(z)$ 的相位准确性直接决定系统稳定**——详见 3.3 节仿真：相位误差超过 90° 时梯度方向掉头，系统发散；误差在 90° 以内时只是收敛变慢，仍收敛到同一个最优解（增益误差则几乎无害，只影响等效步长）。

### 3.3 数值仿真（交付）

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def fxlms_sim(d, x, S, S_hat, L=8, mu=0.05):
    """
    FxLMS 仿真。
    d: 误差麦克风处的初级噪声; x: 参考信号（领先 d 若干样本）;
    S: 真实次级路径; S_hat: 控制器所知的次级路径估计。
    返回误差信号 e（残余噪声）。
    """
    n = len(d)
    w = np.zeros(L)                          # 控制器 FIR
    e = np.zeros(n)
    x_filt = np.convolve(x, S_hat)[:n]       # filtered-x：参考先过 S_hat
    y_buf = np.zeros(len(S))                 # 控制器输出历史（供次级路径卷积）
    for i in range(L, n):
        x_vec = x[i-L+1:i+1][::-1]
        y_ctrl = np.dot(w, x_vec)            # 控制器输出（反噪声）
        y_buf = np.roll(y_buf, 1); y_buf[0] = y_ctrl
        y_at_err = np.dot(S, y_buf)          # 反噪声经真实次级路径到达误差麦克风
        e[i] = d[i] - y_at_err
        w += mu * e[i] * x_filt[i-L+1:i+1][::-1]   # FxLMS 更新
    return e

# 合成：窄带低频噪声（ANC 擅长）
sr = 16000
n = sr * 2
t = np.arange(n) / sr
f0 = 300.0
x = np.sin(2*np.pi*f0*t) * 0.8            # 参考麦克风信号（在噪声源头附近）
delay_primary = 6                          # 主路径延迟：参考领先噪声 6 个样本
d = np.zeros_like(x); d[delay_primary:] = x[:-delay_primary]

S = np.zeros(16); S[4] = 0.9; S[5] = 0.2   # 真实次级路径：短延迟 + 衰减

# 场景1：次级路径估计准确
S_hat_good = S.copy()
# 场景2：增益估计偏 50%（相位正确）
S_hat_gain = S.copy(); S_hat_gain[4] = 1.35; S_hat_gain[5] = 0.3
# 场景3：延迟错 16 样本（300Hz 处相位差 108°，>90°）——如耳机佩戴位置变化
S_hat_phase = np.zeros(48); S_hat_phase[20] = 0.9; S_hat_phase[21] = 0.2

res = {}
curves = {}
for name, Sh in [("准确", S_hat_good), ("增益偏50%", S_hat_gain), ("相位差108°", S_hat_phase)]:
    e = fxlms_sim(d, x, S, Sh, mu=0.05)
    curves[name] = 10*np.log10(np.convolve(e**2, np.ones(800)/800, 'same') + 1e-30)
    res[name] = 10*np.log10(np.mean(e[sr:]**2) / np.mean(d[sr:]**2)) \
                if np.isfinite(e).all() else float('inf')
    print(f"次级路径{name}: 残余噪声 {res[name]:.1f} dB")

assert res["准确"] < res["相位差108°"], "相位失配应导致更差的降噪"

plt.figure(figsize=(10,4))
for name, curve in curves.items():
    plt.plot(curve, label=name, alpha=0.8)
plt.ylim(-80, 60); plt.xlabel("样本"); plt.ylabel("残余噪声能量 (dB)")
plt.title("FxLMS: 次级路径估计对降噪的影响"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("anc_fxlms.png", dpi=110)
print("已保存 anc_fxlms.png")
```

**预期输出**：

```
次级路径准确: 残余噪声 -255.5 dB
次级路径增益偏50%: 残余噪声 -255.5 dB
次级路径相位差108°: 残余噪声 2910.7 dB
```

**三个场景各说明一件事**（比"失配 = 变差"精细得多，也是面试加分点）：

1. **准确估计**：残余噪声指数下降到数值精度极限（-255 dB 即完全抵消）——窄带噪声 + 充分自由度的控制器，FxLMS 能学出近乎完美的反相波；
2. **增益偏 50% 但相位正确**：残余与场景 1 完全一样——增益误差只改变等效步长，不改变收敛终点；
3. **相位差 108°（>90°）**：梯度方向整体掉头，控制器越学越错，残余能量指数爆炸直至数值溢出——系统发散。**这就是 $\hat{S}(z)$ 被称为"命门"的准确含义：不是"不准就效果差"，而是"相位错过 90° 就翻车"**。真实 ANC 耳机因此要定期（在线/离线）重辨识次级路径，并在步长上留足稳定裕量。

### 3.4 ANC 与 AI 降噪的分工

| | ANC（物理） | AI 降噪（信号处理） |
| --- | --- | --- |
| 擅长 | 低频稳态噪声（引擎、空调） | 非稳态噪声（键盘、人声背景） |
| 频带 | 低频 | 全频 |
| 实现 | 硬件 + 自适应控制 | 深度学习模型 |

两者**互补**：耳机常用 ANC 消低频 + AI 降噪处理中高频。这是学习计划里"都要学"的原因。

---

## 4. 工程权衡与失效模式

### 4.1 权衡

- **AGC 攻击/释放**：攻击太快会压掉瞬态（辅音）；释放太慢有抽吸感。5/50 ms 是起点；
- **ANC 次级路径精度**：估得越准越稳，但在线辨识增加复杂度；
- **VAD 灵敏度**：太灵敏漏掉轻声语音；太迟钝把噪声当语音。

### 4.2 失效模式

1. **AGC 抽吸感**：释放太快，音量起伏被放大成"呼吸声"。修复：加长释放时间常数。
2. **AGC 削波**：增益过大顶到 ±1。修复：限幅 + 控制 `max_gain`。
3. **ANC 失稳**：次级路径失配或步长过大，系统自激振荡（尖锐啸叫）。修复：精确辨识 $\hat{S}$、降 $\mu$。
4. **VAD 漏检轻声**：低音量语音被判为静音被切掉。修复：调低阈值、用 Silero 这类鲁棒模型。

---

## 5. 延伸思考题（含解析）

**Q1**：AGC 的攻击与释放时间为什么必须不对称？
**A**：音量突增要快速压低防削波（攻击快）；音量减小后要缓慢回升，否则增益快速拉高会把背景噪声也放大，产生"抽吸/呼吸"伪影（释放慢）。不对称是防削波与防抽吸的折中。

**Q2**：LUFS 相比峰值更能代表响度，为什么？
**A**：峰值只反映瞬时最大幅度，而响度是人耳对能量的时间加权感知。LUFS 按 ITU-R BS.1770 做 K 加权 + 时间积分，更贴近主观响度，故广播用 -23 LUFS 而非峰值标准化。

**Q3**：FXLMS 为什么叫 "Filtered-x"？参考信号为什么要先过 $\hat{S}(z)$？
**A**：控制输出经真实次级路径 $S(z)$ 才到达误差麦克风，所以"误差对控制器权重的敏感度"就是 $S(z)*x$；为正确计算梯度，参考信号必须用估计 $\hat{S}(z)$ 先滤波（即"filtered x"）。$\hat{S}$ 的相位误差在 90° 以内只影响收敛速度（仍收敛到同一最优解），超过 90° 梯度方向翻转，系统发散——因此 $\hat{S}(z)$ 是稳定命门，需在线/定期重新辨识。

**Q4**：ANC 消低频、AI 降噪消中高频，为什么这样分工？
**A**：ANC 靠物理反相抵消，低频波长长、相位易控，适合稳态低频；高频波长短、相位难控且指向性强，物理抵消难，交给信号处理的 AI 降噪更合适。

**Q5**：你的结课助手里，ANC 和 AI 降噪分别部署在哪？
**A**：ANC 在耳机/麦克风硬件层（若有）；AI 降噪（如 DeepFilterNet/DTLN）在音频前端软件层，与 AEC/AGC/VAD 一起。两者串联，ANC 先消低频稳态，AI 再清其余。

---

## 本周交付清单

- [ ] 实现 `agc()`，验证三段不同音量被稳定到相近 RMS。
- [ ] 集成 Silero VAD，输出语音段时间戳。
- [ ] 跑通 `fxlms_sim()`，画次级路径准确/失配的残余噪声对比图。
- [ ] 能解释攻击/释放不对称、FXLMS 的 filtered-x、ANC 与 AI 降噪分工。
