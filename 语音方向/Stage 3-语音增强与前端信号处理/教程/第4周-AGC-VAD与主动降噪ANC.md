# 第 4 周教程：AGC、VAD 与主动降噪（ANC）

> **本周要回答的三个问题**
> 1. AGC 如何"追踪"目标响度？起音/释放时间为什么必须不对称？
> 2. 传统自适应降噪（NLMS）与 AI 深度降噪在原理上有什么本质区别？各自适用什么场景？
> 3. ANC 的次级路径估计 $\hat{S}(z)$ 为什么是系统稳定的命门？失配会怎样？

对应学习计划：第 4 周。交付物：① 手写 AGC（目标 LUFS + 平滑增益 + 限幅）；② FXLMS 数值仿真，画次级路径失配对残余噪声的影响；③ 集成 Silero VAD。

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
目标响度 R_target（如 -23 LUFS）
当前包络 env[n]（信号幅度包络）
期望增益 g_desired = R_target / env[n]
平滑：g[n] = 平滑过渡到 g_desired（攻击/释放不对称）
限幅：防削波
```

**攻击/释放不对称**（本周自测题）：

- **攻击（Attack）快**：音量突然变大，要快速压低防削波 → 短时间常数（~5 ms）；
- **释放（Release）慢**：音量变小后，缓慢回升避免"抽吸"（pumping）感 → 长时间常数（~50-100 ms）。

### 1.4 LUFS 响度标准

LUFS（Loudness Units Full Scale）是广播响度标准（EBU R128），比峰值更接近人耳感知。语音通信目标常取 -23 LUFS。本周简化用 RMS 包络近似，生产用真实 LUFS 表。

### 1.5 实现（交付）

```python
import numpy as np

def agc(x: np.ndarray, sr: int, target_db: float = -23.0,
        attack_ms: float = 5.0, release_ms: float = 50.0,
        max_gain_db: float = 30.0) -> np.ndarray:
    """
    自动增益控制：追踪目标响度 + 不对称平滑 + 限幅。
    """
    # 1. 包络检测：全波整流 + 一阶低通平滑
    env = np.abs(x)
    alpha_env = np.exp(-1.0 / (sr * 0.010))     # 10ms 包络平滑
    smooth_env = np.zeros_like(env)
    for n in range(1, len(env)):
        smooth_env[n] = alpha_env * smooth_env[n-1] + (1-alpha_env) * env[n]
    smooth_env[0] = env[0]

    # 2. 目标增益（线性）
    target = 10 ** (target_db / 20.0)
    g_desired = target / (smooth_env + 1e-8)
    max_gain = 10 ** (max_gain_db / 20.0)
    g_desired = np.clip(g_desired, 0, max_gain)

    # 3. 不对称平滑：增益下降用攻击常数，上升用释放常数
    alpha_atk = np.exp(-1.0 / (sr * attack_ms / 1000))
    alpha_rel = np.exp(-1.0 / (sr * release_ms / 1000))
    g = np.zeros_like(g_desired)
    g[0] = g_desired[0]
    for n in range(1, len(g_desired)):
        if g_desired[n] < g[n-1]:       # 需要压低 -> 攻击（快）
            g[n] = alpha_atk * g[n-1] + (1-alpha_atk) * g_desired[n]
        else:                            # 需要提升 -> 释放（慢）
            g[n] = alpha_rel * g[n-1] + (1-alpha_rel) * g_desired[n]

    # 4. 应用增益 + 限幅
    y = x * g
    peak = np.abs(y).max()
    if peak > 1.0:
        y = y / peak * 0.99              # 软限幅防削波
    return y

# 验证：音量忽大忽小的输入 -> 输出包络稳定
np.random.seed(0)
sr = 16000
t = np.arange(sr * 3) / sr
# 构造幅度起伏的语音样信号
carrier = np.sin(2*np.pi*200*t)
amp = np.where(t < 1.0, 0.1, np.where(t < 2.0, 0.8, 0.2))  # 三段不同音量
x = carrier * amp
y = agc(x, sr)

# 检查三段输出 RMS 是否趋于一致
def rms(seg): return np.sqrt(np.mean(seg**2))
r1, r2, r3 = rms(y[:sr]), rms(y[sr:2*sr]), rms(y[2*sr:])
print(f"三段输出 RMS: {r1:.3f} {r2:.3f} {r3:.3f}")
assert max(r1, r2, r3) / (min(r1, r2, r3) + 1e-9) < 2.0, "AGC 未稳定响度"
print("AGC 已把三段不同音量稳定到相近响度 ✓")
```

**预期**：三段输入音量差 8 倍，输出 RMS 趋于一致（比值 < 2）。这就是"自动增益"的最小实证。

---

## 2. VAD：语音活动检测

### 2.1 三代演进

| 代 | 方法 | 特点 |
| --- | --- | --- |
| 传统 | 能量/过零率/谱熵 | 简单、快，但抗噪差 |
| 统计 | WebRTC VAD（GMM） | 工业标准、轻量 |
| 端到端 | **Silero VAD** | 当前事实标准，鲁棒 |

### 2.2 Silero VAD

预训练的轻量模型，输出每帧"是语音"的概率。`torch.hub` 一行加载：

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

**输出**：语音段的起止样本索引列表。用途：① ASR 前端分段；② 降噪的噪声估计（静音段）；③ 录音只存有语音部分。

### 2.3 集成进 Pipeline

```python
class VADFrontend:
    def __init__(self):
        self.model, self.utils = torch.hub.load(
            'snakers4/silero-vad', 'silero_vad', trust_repo=True)
        self.get_ts = self.utils[0]

    def speech_segments(self, wav, sr=16000):
        return self.get_ts(wav, self.model, sampling_rate=sr)

    def keep_speech(self, wav, sr=16000):
        ts = self.speech_segments(wav, sr)
        return np.concatenate([wav[t['start']:t['end']] for t in ts]) \
               if ts else wav
```

---

## 3. 主动降噪（ANC）

### 3.1 物理原理

ANC 不靠算法"滤"噪声，而是**产生一个反相声波物理抵消**噪声：

$$
\text{原始噪声 } d[n] \;+\; \text{反相声波 } -d[n] \;=\; 0
$$

耳机里的麦克风采集外部噪声，处理器生成反相信号从扬声器（耳机喇叭）播出，两者叠加抵消。这是**物理层的消噪**，与 AI 降噪（信号处理层）互补。

### 3.2 FXLMS：为什么需要次级路径估计

难点：反相信号从"误差麦克风"听回来时，经过了**次级路径** $S(z)$（扬声器 + 耳道传播）。若控制器不知道这个路径，就无法正确生成反相波。

**FXLMS（Filtered-x LMS）** 的"Filtered-x"正是指：参考信号要先通过次级路径估计 $\hat{S}(z)$ 滤波，才能正确更新：

$$
\mathbf{w} \leftarrow \mathbf{w} + \mu\, e[n]\, (\hat{S}(z) * x[n])
$$

其中 $\hat{S}(z) * x$ 是"用估计的次级路径滤波后的参考"。**$\hat{S}(z)$ 的准确性直接决定系统稳定**。

### 3.3 数值仿真（交付）

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def fxlms_sim(d, x, S, S_hat, L=32, mu=0.01, n_iter=None):
    """
    FXLMS 仿真。
    d: 期望信号(噪声); x: 参考信号; S: 真实次级路径; S_hat: 估计的次级路径。
    返回误差信号 e（残余噪声）。
    """
    n = len(d) if n_iter is None else n_iter
    w = np.zeros(L)                    # 控制器滤波器
    e = np.zeros(n)
    # 次级路径滤波参考
    x_filt = np.convolve(x, S_hat)[:n]
    for i in range(L, n):
        # 控制器输出（反相波）
        y_ctrl = -np.dot(w, x[i-L+1:i+1][::-1])
        # 误差 = 噪声 + 控制输出经真实次级路径
        y_at_err = np.convolve(np.append([0]*(i-L), [y_ctrl]), S)[i] \
                   if i < len(S)+1 else y_ctrl * S[0]
        e[i] = d[i] + y_at_err
        # FXLMS 更新（用滤波后的参考）
        w += mu * e[i] * x_filt[i-L+1:i+1][::-1]
    return e

# 合成：窄带低频噪声（ANC 擅长）
sr = 16000
n = sr * 2
t = np.arange(n) / sr
d = np.sin(2*np.pi*150*t) * 0.8        # 150Hz 引擎噪声
x = d.copy()                            # 参考麦克风采集

# 真实次级路径：短延迟 + 衰减
S = np.zeros(16); S[4] = 0.9; S[5] = 0.2

# 场景1：次级路径估计准确
S_hat_good = S.copy()
# 场景2：次级路径失配（延迟错 2 样本）
S_hat_bad = np.zeros(16); S_hat_bad[6] = 0.9; S_hat_bad[7] = 0.2

e_good = fxlms_sim(d, x, S, S_hat_good)
e_bad = fxlms_sim(d, x, S, S_hat_bad)

res_good = 10*np.log10(np.mean(e_good[sr:]**2) / (np.mean(d[sr:]**2)+1e-12))
res_bad = 10*np.log10(np.mean(e_bad[sr:]**2) / (np.mean(d[sr:]**2)+1e-12))
print(f"次级路径准确: 残余噪声 {res_good:.1f} dB")
print(f"次级路径失配: 残余噪声 {res_bad:.1f} dB")
assert res_good < res_bad, "失配应导致更差的降噪"

plt.figure(figsize=(10,4))
plt.plot(10*np.log10(np.convolve(e_good**2,np.ones(800)/800,'same')+1e-12), label="准确")
plt.plot(10*np.log10(np.convolve(e_bad**2,np.ones(800)/800,'same')+1e-12), label="失配")
plt.xlabel("样本"); plt.ylabel("残余噪声能量 (dB)")
plt.title("FXLMS: 次级路径估计对降噪的影响"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("anc_fxlms.png", dpi=110)
print("已保存 anc_fxlms.png")
```

**预期**：次级路径准确时残余噪声明显更低；失配时收敛慢、残留高，甚至不稳。这直接证明 **$\hat{S}(z)$ 是 ANC 稳定的命门**。

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
**A**：控制输出经真实次级路径 $S(z)$ 才到达误差麦克风；为正确计算梯度，参考信号必须用次级路径估计 $\hat{S}(z)$ 先滤波（即"filtered x"），否则更新方向错误，系统不收敛甚至发散。

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
