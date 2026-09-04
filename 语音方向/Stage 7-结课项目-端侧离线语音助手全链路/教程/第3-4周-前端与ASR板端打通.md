# 第 3-4 周教程：前端与 ASR 板端打通——从麦克风到文字

> **本周要回答的四个问题**
> 1. ALSA 采集的帧长、缓冲、时间戳怎么配，才能喂好 AEC 又不引入大延迟？
> 2. AEC 的参考信号回路怎么接？为什么「没有参考回路的 AEC 等于没做」？
> 3. WebRTC APM 交叉编译到 ARM 要注意什么？哪些模块可以裁掉？
> 4. SenseVoice-Small 上板先跑 ONNX Runtime CPU，RTF 与云端一致性怎么测才算数？

对应学习计划：第 3-4 周。交付物：板端跑通「说话 → 文字」，实测 RTF 与首字延迟，记录 CPU 利用率；与云端推理一致性差距 < 2%。

---

## 1. 第一性原理：前端存在的理由——信噪比决定一切

### 1.1 链路质量的乘法关系

全链路识别率近似满足：

$$
P_{\text{识别}} \approx P_{\text{前端保真}} \cdot P_{\text{ASR} \mid \text{SNR}}
$$

ASR 模型在干净语音上的能力是固定的，**你能动的变量只有前端送进去的 SNR**。远场 3 米、电视背景音下，裸麦的 SNR 可能只有 5 dB，AEC+NS+AGC 能拉回 15 dB 以上——这一段的收益比换任何 ASR 模型都大。这是你 Stage 3 的结论，本周让它上板。

### 1.2 10 ms 帧的物理约束

WebRTC APM 的处理粒度是 **10 ms 帧**（16 kHz 下 160 样本）。这个数字不可协商——AEC 的滤波器按帧对齐，VAD（Silero）也按 10 ms 设计。整条前端从采集开始就锁死 10 ms 节拍，任何模块引入额外缓冲都在直接加延迟：

$$
T_{\text{前端}} = T_{\text{采集缓冲}} + T_{\text{APM 处理}} \le 10\,\text{ms} + \text{约 5\,ms}
$$

采集缓冲取 1 帧（10 ms）而不是常见的 4~8 帧，是端侧与桌面应用的关键差异——桌面要吞吐，端侧要延迟。

---

## 2. 采集与 AEC 回路

### 2.1 ALSA 采集配置

关键参数：16 kHz、单声道（多通道阵列先混音或走波束成形）、16-bit、period 160 样本（10 ms）、缓冲 2~4 periods。用 `hw` 设备而非 `default`（避开 PulseAudio 的重采样与缓冲黑箱）。

### 2.2 AEC 参考回路（本周最易翻车点）

AEC 的原理：已知送往扬声器的信号 $s_t$（参考），从麦克风信号 $m_t$ 中估计并减去回声 $\hat{h} * s_t$：

$$
\text{clean}_t = m_t - \hat{h} * s_t
$$

**参考信号必须与播放严格同源同时间基**。两种接法：

1. **软件回路（首选）**：播放缓冲里的 PCM 块同时复制一份给 APM 的 `analyze_reverse_stream`——零硬件成本，延迟可知；
2. **硬件回路**：从功放输出回采一路——能消掉功放/扬声器的非线性失真，但需要额外 ADC 通道。

无参考回路时的典型症状：助手在播放 TTS 时被自己的声音触发 VAD，产生自我打断或自问自答。

### 2.3 WebRTC APM 的 ARM 移植与裁剪

交叉编译要点（aarch64-linux-gnu 工具链）：

- 只编 `modules/audio_processing` 子集，裁掉视频与网络模块；
- 保留：AEC3、NS（噪声抑制）、AGC2、VAD（可选，我们用 Silero）；
- 裁掉：回声消除的实验性配置、多通道处理（单声道场景）；
- 编译产物是静态库，Python 侧用 `pybind11` 薄封装或直接用 `sherpa-onnx` 的 VAD + 系统级 APM。

**替代路径**：若自编译时间超 3 天，先用 `sherpa-onnx` 的 Silero VAD + 简化的谱减降噪打通链路，WebRTC APM 作为第 7-8 周回填——**打通优先于完美**，但接口层保持「前端是黑盒模块」以便替换。

---

## 3. ASR 上板：SenseVoice-Small 的部署路径

### 3.1 为什么是 ONNX 先行

部署路径：`FunASR 权重 → ONNX 导出 → (ONNX Runtime CPU 打通) → (RKNN 转换回填)`。

ONNX 先行的理由：① 功能正确性验证与性能优化解耦——先证明链路对，再谈快；② RKNN 转换常遇算子不支持，需要逐个修补，不该阻塞链路打通；③ ONNX Runtime 有成熟的 aarch64 轮子。

**sherpa-onnx 的角色**：它把流式与非流式 ASR、VAD、唤醒词统一成一套 C++/Python API，是端侧语音的「运行时粘合剂」。SenseVoice 走它的非流式接口（短指令场景，NAR 单步解码本就无需流式）。

### 3.2 RTF 与一致性的定义（测量口径必须先冻结）

$$
\text{RTF} = \frac{\text{解码耗时}}{\text{音频时长}}, \qquad
\text{首字延迟} = \text{VAD 端点时刻} \to \text{文本产出时刻}
$$

一致性对比：同 100 条测试音频，板端（ONNX）与云端（FunASR 原始权重）各跑一遍，比较：

$$
\text{一致率} = \frac{\text{转写完全相同的条数}}{100}, \qquad \text{要求} \ge 98\%
$$

注意口径：**一致性比的是「同模型不同运行时」**，不是「板端模型与云端大模型的差距」——后者是模型差距，不是部署差距。混淆这两个口径是压测报告最常见的错误。

---

## 4. 实现与验证

### 4.1 ALSA 采集 + VAD 触发（板端 Python）

```python
import numpy as np
import sounddevice as sd          # 板端底层走 ALSA
from sherpa_onnx import VadModelConfig, VoiceActivityDetector

SR, FRAME = 16000, 160            # 16 kHz, 10 ms/帧

def make_vad():
    cfg = VadModelConfig()
    cfg.silero_vad.onnx = "silero_vad.onnx"
    cfg.silero_vad.threshold = 0.5
    cfg.silero_vad.min_silence_duration = 0.4   # 尾部静默 400 ms 判定说完
    cfg.silero_vad.min_speech_duration = 0.25
    cfg.sample_rate = SR
    return VoiceActivityDetector(cfg, buffer_size_in_seconds=30)

def capture_utterance(vad, stream):
    """阻塞直到采集到一个完整语音段，返回 (音频, 触发前保留的历史)。"""
    while True:
        frame = stream.read(FRAME)                      # int16, (160,)
        vad.accept_waveform(np.frombuffer(frame, np.int16).astype(np.float32) / 32768)
        if not vad.is_empty():
            seg = vad.front
            vad.pop()
            return seg.samples, seg.start               # start 用于延迟埋点
```

**要点**：`buffer_size_in_seconds=30` 的环形缓冲自动保留触发前的音频——第 1-2 周讲的「防截头」在这里落地。

### 4.2 SenseVoice 推理与延迟埋点

```python
import time
import sherpa_onnx

def make_asr():
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx",
        tokens="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
        num_threads=4, use_itn=True, language="auto")

def transcribe_with_timing(asr, samples):
    """返回 (文本, RTF, 解码耗时)。交付时填入板端实测值。"""
    stream = asr.create_stream()
    stream.accept_waveform(SR, samples)
    t0 = time.perf_counter()
    asr.decode_stream(stream)
    dt = time.perf_counter() - t0
    rtf = dt / (len(samples) / SR)
    return stream.result.text.strip(), rtf, dt

# 验收断言（交付时替换为板端真实数字）
def accept_rtf(rtf, audio_s):
    assert rtf < 0.1, f"RTF {rtf:.3f} 超过预算：5s 音频解码应 <500ms 的一半"
    assert audio_s > 0
```

**预期**：RK3588 大核 ×4 线程下，SenseVoice-Small int8 的 RTF ≈ 0.02~0.05（5 秒音频解码 100~250 ms），落在 200 ms 预算内。若超预算：先查是否绑到大核（`taskset`），再查线程数是否与核数匹配。

### 4.3 板云一致性测试脚本（交付核心）

```python
def consistency_report(pairs):
    """pairs: [(音频路径, 板端转写, 云端转写), ...]，100 条。"""
    same = sum(1 for _, b, c in pairs if b == c)
    rate = same / len(pairs)
    print(f"一致率: {rate:.1%} ({same}/{len(pairs)})")
    for path, b, c in pairs:
        if b != c:
            print(f"  差异: {path}\n    板端: {b}\n    云端: {c}")
    assert rate >= 0.98, "同模型不同运行时的一致率必须 ≥98%，否则查量化/预处理"
    return rate

# 差异 >2% 时的排查顺序：
# 1) 预处理差异（fbank 参数、重采样方法）——最常见根因
# 2) int8 量化误差（换 fp32 对比一轮）
# 3) 版本号不一致（ONNX 导出版与云端权重版）
```

**预期输出**：一致率 ≥98%；若 90~98%，九成是预处理（梅尔滤波器参数、重采样器）不一致，先统一再谈量化。

### 4.4 预处理契约：板云一致性的真正变量

一致性差异 >2% 时，90% 的根因不在模型而在预处理。把契约写成可执行的校验：

```python
import numpy as np

PREPROCESS_CONTRACT = {
    "sample_rate": 16000,
    "n_mels": 80,
    "n_fft": 400,
    "hop_length": 160,        # 10 ms 帧移
    "window": "hann",
    "normalize": "log_mel",   # 与导出时的特征提取严格一致
}

def verify_contract(cfg_a, cfg_b):
    """板端与云端配置逐项对比，任何一项不一致直接判差异根因。"""
    diff = {k: (cfg_a.get(k), cfg_b.get(k))
            for k in set(cfg_a) | set(cfg_b) if cfg_a.get(k) != cfg_b.get(k)}
    assert not diff, f"预处理契约不一致: {diff}"
    return True

assert verify_contract(PREPROCESS_CONTRACT, dict(PREPROCESS_CONTRACT))
print("预处理契约校验通过 ✓ —— 板云各存一份此文件，部署时自动断言")
```

**实践**：该文件放进仓库根目录，板端启动与云端推理脚本都 `import` 并断言——配置漂移在启动时就被拦住，而不是在一致性测试里当悬案查。

### 4.5 AEC 效果快速验证

```python
import numpy as np

def echo_suppression_db(mic, ref, out, sr=16000):
    """mic: 含回声麦克风信号; ref: 播放参考; out: APM 输出。
    用「参考与输出的互相关峰值下降」粗估回声抑制量。"""
    def peak_corr(a, b):
        n = min(len(a), len(b))
        a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
        return float(np.max(np.correlate(b[:n//4], a, mode="same"))) / (np.linalg.norm(a)+1e-9)
    before, after = peak_corr(mic, ref), peak_corr(out, ref)
    db = 20 * np.log10((before + 1e-9) / (after + 1e-9))
    print(f"回声相关峰值下降: {db:.1f} dB")
    assert db > 5, "AEC 几乎没生效：检查参考回路是否同源同时间基"
    return db
```

**预期**：回路正确时相关峰值下降 >15 dB；若 <5 dB，按 2.2 节排查参考信号。主观验证：播放音乐时对设备说话，转写不应混入歌词。

---

## 5. 工程权衡与失效模式

### 5.1 权衡

- **CPU 先行 vs NPU 直上**：本周选 CPU（打通优先）；但 ONNX 模型封装成独立模块，RKNN 回填时只换实现不换接口。
- **int8 vs fp32**：int8 内存减半、速度快 ~2 倍，但一致性测试里若差异集中出现（如 >2%），先回 fp32 定位——量化误差与部署 bug 必须分开归因。
- **帧缓冲取舍**：1 帧（10 ms）采集缓冲延迟最低但 CPU 中断频繁；2 帧（20 ms）CPU 负载降 ~30% 延迟 +10 ms。RK3588 上建议 1 帧起步，负载顶不住再放宽。
- **波束成形要不要**：单麦先跑通；若板子是环形阵列，波束成形收益在远场（>2 m）才显著，近场（<1 m）加它纯增加延迟。

### 5.2 失效模式

1. **截头**（丢首字）：VAD 触发前的音频没保留 → 「今天天气」识别成「天天气」。修复：环形缓冲回看 ≥200 ms（4.1 已内置）。
2. **自问自答**：无 AEC 参考回路，TTS 播放被 ASR 听进去。修复：接参考回路；临时方案——播放期间关闭 ASR 触发（半双工降级，打断功能随之失效，写进风险清单）。
3. **重采样污染**：麦克风 48 kHz，链路里多次重采样引入混叠与延迟。修复：全链路锁定 16 kHz，重采样只在采集入口做一次（高质量 sinc 滤波器）。
4. **大核未绑定**：ASR 线程被调度到 A55 小核，RTF 翻 3 倍。症状：延迟时好时坏；修复：`taskset`/`pthread_setaffinity` 绑大核，压测时盯 `/proc` 验证。
5. **一致性差异被误判为量化问题**：实际是预处理参数不一致。修复：固定一份「预处理契约」（采样率、梅尔参数、归一化方式），板云共用。

---

## 6. 延伸思考题（含解析）

**Q1**：为什么 AEC 需要参考信号，而 NS（降噪）不需要？
**A**：回声是**已知源**的干扰（就是自己播出的信号），可以做参考相减；环境噪声源未知，只能靠统计特征估计并抑制。所以 AEC 是确定性减法问题，NS 是统计估计问题——前者没有参考信号就退化成无信息盲减，自然「等于没做」。

**Q2**：RTF < 0.1 为什么还不够？预算是 200 ms，5 秒音频 RTF=0.1 不就是 500 ms 吗？
**A**：好问题——预算表里的 200 ms 针对的是**典型短句**（≤2 s 的指令），长句场景要么超预算要么截断。所以验收要报两个数字：短句（2 s）首字延迟 <200 ms，长句（5 s）RTF <0.1。单一数字会掩盖分布——这是所有延迟指标的通病。

**Q3**：ONNX Runtime CPU 版与未来 RKNN 版，精度口径应该怎么对齐？
**A**：以云端原始权重（fp32）的转写为基准，分别测 ONNX 版与 RKNN 版的一致率；两个运行时各自 ≥98% 才算部署无损。若 RKNN 版单独低，问题在转换（算子替换/量化），与 ONNX 无关——**基准唯一、逐层归因**。

**Q4**：VAD 尾部静默阈值 400 ms 是从哪里来的？能再短吗？
**A**：来自「自然停顿」的统计：中文口语中词语间停顿多为 100~300 ms，轮次结束停顿通常 >400 ms。短于 350 ms 会把思考停顿误判为说完（抢话），长于 600 ms 用户会觉得助手迟钝。400~500 ms 是产品甜点——第 7-8 周用真实用户测试微调。

**Q5**：如果一致性测试板端 95%、云端 100%，但转写错误的 5 条全是带背景噪声的样本，说明什么？
**A**：说明差异不在模型运行时，而在**前端链路**：云端测试用的可能是干净音频或不同的前端处理。修复：把板端前端输出的音频（而非原始录音）作为云端对比输入——对比必须在「同一输入」上才有效。

**Q6**：为什么用 sherpa-onnx 而不是直接写 ONNX Runtime 推理？
**A**：sherpa-onnx 在 ONNX Runtime 之上封装了端侧语音的全部例行公事——音频特征提取、流式/非流式状态管理、VAD、唤醒词、TTS 的统一 API，且有成熟 aarch64 轮子。自己裸写 ONNX Runtime 意味着重造特征提取与状态机，而这两处恰是预处理契约差异的高发区。原则：**通用运行时自己写，领域粘合层用成熟库**。

---

## 7. 附录：板端资源测量速查（交付数据从这来）

```bash
# CPU 占用（分核查看，确认 ASR 跑在大核 core4-7）
top -1                                   # 或 mpstat -P ALL 1

# 内存：进程级
pmap -x $(pidof python3) | tail -3

# NPU 利用率（RKNN 回填后使用）
cat /sys/kernel/debug/rknpu/load 2>/dev/null || rknpu_top

# 温度（压测必记录，第 7-8 周热机分析的数据源）
cat /sys/class/thermal/thermal_zone*/temp
```

**记录表模板**（每次实测填一行）：

| 日期 | 模块 | 运行时 | 绑核 | RTF/延迟 | CPU% | 内存 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 示例 | ASR | ONNX int8 | 4×A76 | RTF 0.04 | 310% | 280 MB | 基线 |

这张表从本周开始积累，第 7-8 周的「与云端对比」「热机降频」分析全靠它的时间序列——**数据不记录等于没测**。

---

## 本周交付清单

- [ ] 板端「说话 → 文字」跑通：ALSA 采集 + VAD 触发 + SenseVoice int8 推理全链路。
- [ ] 实测数据入表：短句首字延迟、长句 RTF、大核利用率、内存占用。
- [ ] 板云一致性 100 条测试完成，一致率 ≥98%（差异样本逐条归因记录）。
- [ ] AEC 参考回路接通并通过 4.4 的验证（播放时说话不串扰）。
- [ ] 设计文档更新：前端与 ASR 的预算行从设计值替换为实测值。
