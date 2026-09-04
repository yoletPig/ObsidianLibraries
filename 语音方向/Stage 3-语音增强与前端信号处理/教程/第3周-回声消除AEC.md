# 第 3 周教程：回声消除（AEC）

> **本周要回答的四个问题**
> 1. 免提通话里那个"自己的声音又回来了"的回声，物理上是怎么产生的？
> 2. 自适应滤波器如何用远端参考信号估计回声路径？NLMS 的步长为什么那样归一化？
> 3. 双讲（Double-talk）时为什么必须冻结滤波器更新？不冻结会发生什么灾难？
> 4. 线性滤波消不掉的残余回声，NLP/RES 怎么收尾？为什么 AEC 是 WebRTC 里最难的模块？

对应学习计划：第 3 周。交付物：手写频域自适应回声消除器，合成回声数据，画回声路径收敛曲线（ERLE 随时间）；跑 WebRTC APM 对照。

---

## 1. 第一性原理：回声是"扬声器声音被麦克风重新采集"

### 1.1 物理成因

免提/会议场景中，远端语音 $x[n]$（far-end，对方说话）从**扬声器**播出，经房间传播后被**麦克风**重新采集，与本地说话人语音 $s[n]$ 一起传回给远端——对方就听到了自己声音的延迟版本，即**回声**。

### 1.2 数学模型

麦克风采集信号：

$$
d[n] = s[n] + \underbrace{\sum_{k=0}^{L-1} h[k]\, x[n-k]}_{\text{回声}} + v[n]
$$

- $d[n]$：麦克风信号（desired，含回声）；
- $x[n]$：远端参考信号（扬声器播的，我们**已知**——这是 AEC 能成立的关键）；
- $h[k]$：回声路径（房间 + 扬声器 + 麦克风的脉冲响应），长度 $L$ 可达数百毫秒；
- $s[n]$：近端语音（本地说话人）；$v[n]$：噪声。

**目标**：用已知的 $x[n]$ 估计回声 $\hat{h} * x$，从 $d[n]$ 里减掉，留下 $s[n]$。

### 1.3 核心难点：双讲

当本地说话人**同时**说话（$s[n] \neq 0$），麦克风信号里既有回声又有近端语音。此时若继续用 $d[n]$ 去更新滤波器 $\hat{h}$，近端语音 $s[n]$ 会被当成回声路径的一部分"学进去"，**污染 $\hat{h}$**，导致回声消不干净甚至发散。这就是双讲问题——AEC 的灵魂难题。

---

## 2. 自适应滤波：估计回声路径

### 2.1 LMS / NLMS

用滤波器 $\hat{h}[k]$ 卷积参考信号 $x$ 得到回声估计 $\hat{y}[n] = \sum_k \hat{h}[k] x[n-k]$，误差信号：

$$
e[n] = d[n] - \hat{y}[n]
$$

LMS 按梯度下降更新滤波器：

$$
\hat{h}[k] \leftarrow \hat{h}[k] + \mu\, e[n]\, x[n-k]
$$

**NLMS（归一化 LMS）**——用参考信号能量归一化步长，消除输入幅度影响：

$$
\hat{\mathbf{h}} \leftarrow \hat{\mathbf{h}} + \frac{\mu}{\|\mathbf{x}\|^2 + \delta}\, e[n]\, \mathbf{x}
$$

分母 $\|\mathbf{x}\|^2$ 是参考信号能量：能量大时步长自动变小，防止大步长震荡；$\delta$ 防除零。**$\mu \in (0,2)$** 保证收敛，实际取 0.5-1。

**为什么频域做（FDAF）**：回声路径 $L$ 很长（几百毫秒 = 数千抽头），时域卷积是 $O(L)$ 每样本，太贵。频域用 FFT 把卷积变乘法，复杂度降到 $O(\log L)$。分块自适应滤波（PBFDAF, Partitioned-Block Frequency-Domain Adaptive Filtering）是标准实现。

### 2.2 ERLE：回声抑制效果指标

$$
\text{ERLE}_{\text{dB}} = 10\log_{10} \frac{\mathbb{E}[d^2]}{\mathbb{E}[e^2]}
$$

即"麦克风信号能量 / 误差信号能量"。ERLE 越大，回声消得越干净。**收敛曲线 = ERLE 随时间上升**，是本周交付的核心图。

### 2.3 双讲检测与冻结

双讲期间冻结更新：

```
if 双讲检测到:
    不更新 h（保持上一次估计）
else:
    正常 NLMS 更新
```

双讲检测的常用判据：远端能量高且误差能量也高（说明有额外声源 = 近端说话）。更先进用相干性、卡尔曼滤波。

---

## 3. 实现与验证（交付核心）

### 3.1 合成回声数据

```python
import numpy as np
from scipy import signal as sps

sr = 16000
dur = 4.0
n = int(sr * dur)

# 远端信号（对方说话，用噪声/语音模拟）
np.random.seed(42)
far = np.random.randn(n).astype(np.float32) * 0.5

# 回声路径：模拟房间脉冲响应（指数衰减的随机序列）
L = int(0.15 * sr)                       # 150 ms 回声路径
t = np.arange(L)
h_true = np.random.randn(L) * np.exp(-t / (0.04 * sr))
h_true = h_true / (np.abs(h_true).max() + 1e-9)

# 回声 = 远端 * 路径
echo = sps.fftconvolve(far, h_true)[:n]

# 近端语音：前半段静音（纯回声，用于收敛），后半段有近端（双讲）
near = np.zeros(n, dtype=np.float32)
near[n//2:] = np.random.randn(n - n//2) * 0.3   # 后半段近端说话

# 麦克风信号 = 近端 + 回声
mic = near + echo
print(f"far={far.shape}, echo={echo.shape}, mic={mic.shape}")
```

### 3.2 时域 NLMS 回声消除（教学版）

```python
def nlms_aec(mic, far, L, mu=0.8, delta=1e-4, freeze_doubletalk=True):
    """
    时域 NLMS 回声消除（教学实现，生产用频域）。
    返回 (误差信号, 滤波器历史用于画收敛曲线)。
    """
    n = len(mic)
    h = np.zeros(L)
    err = np.zeros(n)
    erle_hist = []

    # 简易双讲检测：近端能量代理（误差/远端能量比）
    win = 160
    for i in range(L, n):
        x_vec = far[i-L+1:i+1][::-1]          # 参考向量（最新在前）
        y = np.dot(h, x_vec)                  # 回声估计
        e = mic[i] - y                         # 误差
        x_energy = np.dot(x_vec, x_vec) + delta

        # 双讲检测：误差能量远高于回声估计能量 → 可能有近端
        if freeze_doubletalk and i > L:
            recent_err = err[i-win:i]
            if len(recent_err) == win:
                err_pow = np.mean(recent_err ** 2)
                far_pow = np.mean(far[i-win:i] ** 2) + 1e-9
                doubletalk = err_pow > 2.0 * far_pow
            else:
                doubletalk = False
        else:
            doubletalk = False

        if not doubletalk:
            h += (mu / x_energy) * e * x_vec   # NLMS 更新

        err[i] = e
        if i % win == 0 and i > win:
            mic_pow = np.mean(mic[i-win:i] ** 2) + 1e-12
            err_pow = np.mean(err[i-win:i] ** 2) + 1e-12
            erle_hist.append(10 * np.log10(mic_pow / err_pow))
    return err, erle_hist

err, erle = nlms_aec(mic, far, L, mu=0.8)
```

### 3.3 收敛曲线可视化

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
frames = np.arange(len(erle)) * 160 / sr
plt.plot(frames, erle)
plt.axvline(dur / 2, color="r", linestyle="--", label="双讲开始")
plt.xlabel("时间 (s)"); plt.ylabel("ERLE (dB)")
plt.title("回声路径收敛曲线")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("aec_erle.png", dpi=110)
print("已保存 aec_erle.png")
print(f"收敛后稳态 ERLE ≈ {np.mean(erle[len(erle)//4:len(erle)//2]):.1f} dB（双讲前）")
```

**预期**：前 2 秒（纯回声）ERLE 从 ~0 上升到 15-30 dB（收敛）；2 秒后进入双讲，ERLE 因冻结而保持或略降。**观察点**：① 收敛速度；② 双讲时滤波器没被污染（误差仍小）。

### 3.4 WebRTC APM 对照

```bash
pip install webrtc-noise-gain    # 或 webrtc-audio-processing 绑定
```

```python
# WebRTC AEC3 是工业级实现，接口以具体绑定为准
# 这里展示对照思路：把同一段 (mic, far) 喂给 WebRTC，对比 ERLE 与残留
# 重点观察：WebRTC 的延迟估计、双讲处理、残余回声抑制都更稳
print("用同一合成数据跑 WebRTC APM，对比你的 NLMS：")
print("  - ERLE 收敛更快、更稳")
print("  - 双讲期间几乎不污染滤波器")
print("  - 有专门的残余回声抑制（NLP）")
```

**对照结论**（面试可用）：你的教学版能让你理解原理，但工业级（WebRTC AEC3）在**延迟估计、双讲鲁棒、残余抑制、非线性路径**四方面全面更强——这正是"为什么 AEC 是 WebRTC 最难模块"。

---

## 4. 工程权衡与失效模式

### 4.1 权衡

- **滤波器长度**：长 → 能建模长回声路径，但计算与收敛慢；短 → 反之。需覆盖房间实际混响时长；
- **步长 $\mu$**：大 → 收敛快但不稳；小 → 稳但慢。0.5-1 常用；
- **双讲检测灵敏度**：太灵敏 → 频繁冻结、收敛慢；太迟钝 → 污染滤波器。

### 4.2 失效模式

1. **双讲污染**：冻结失效，近端语音被学进 $\hat{h}$，回声残留甚至放大。定位：看双讲段 ERLE 是否骤降；修复：加强双讲检测、用卡尔曼。
2. **延迟失配**：扬声器到麦克风的延迟与参考信号 $x$ 对不齐，滤波器永远学不到。定位：互相关找延迟；修复：延迟估计与对齐（WebRTC 的 delay estimator）。
3. **非线性回声**：扬声器过载产生谐波失真，线性滤波器建模不了。修复：NLP/RES 非线性后处理。
4. **收敛慢**：长路径 + 小步长导致开头几秒回声明显。修复：频域加速、合理步长。

---

## 5. 延伸思考题（含解析）

**Q1**：为什么 AEC 必须依赖"远端参考信号"？没有它行吗？
**A**：回声 = 参考信号经未知路径的卷积。已知参考 $x$，才能用自适应滤波反解路径 $h$。没有参考，退化为盲源分离/纯降噪，难度陡增且无法精准消回声。这是 AEC 与降噪的本质区别。

**Q2**：双讲时为什么要冻结滤波器更新？不冻结会怎样？
**A**：双讲时误差 $e$ 里含近端语音 $s$，若继续更新，$s$ 会被当成回声路径的一部分学进 $\hat{h}$，污染估计，导致回声消不净甚至发散。冻结即"近端说话时不学习"，保护 $\hat{h}$。

**Q3**：NLMS 分母为什么要除以 $\|x\|^2$？
**A**：归一化步长，消除参考信号幅度影响：参考能量大时自动减小步长防止震荡，能量小时放大步长加快收敛。使算法对输入音量鲁棒，收敛行为一致。

**Q4**：线性滤波后为什么还要 NLP/RES？
**A**：线性滤波消不掉的残余回声来自：非线性路径（扬声器失真）、路径估计误差、双讲期间的残留。NLP（非线性处理）/RES（残余回声抑制）用频谱抑制把残余压下去，是最后的收尾。

**Q5**：AEC 为什么要在 NS（降噪）之前？顺序颠倒会怎样？
**A**：AEC 假设回声是参考信号的线性卷积；若先降噪，降噪的非线性谱改动会破坏回声与参考的线性关系，AEC 无法对齐消除。所以流水线必须 AEC → NS → AGC。这是本阶段主线的硬约束。

---

## 本周交付清单

- [ ] 合成回声数据（远端 + 路径 + 双讲近端）。
- [ ] 实现时域 `nlms_aec()`，输出误差信号与收敛曲线 `aec_erle.png`。
- [ ] 观察纯回声段收敛（ERLE 上升）与双讲段冻结行为。
- [ ] 跑 WebRTC APM 对照，总结工业级实现强在哪四点。
- [ ] 能闭卷回答：为什么双讲要冻结、为什么要频域、为什么 AEC 在 NS 前。
