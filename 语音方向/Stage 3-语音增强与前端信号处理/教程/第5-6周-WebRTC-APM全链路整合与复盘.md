# 第 5-6 周教程：WebRTC APM 全链路整合与阶段复盘

> **本周要回答的四个问题**
> 1. AEC → NS → AGC → VAD 的调用顺序为什么不能乱？
> 2. WebRTC APM 每个模块的保守/激进档位各有什么代价？
> 3. 前端提升对下游 ASR 到底有多大增益？怎么量化？
> 4. 如何把前四周的组件焊成一条可复用的"录音 → 干净文本"流水线？

对应学习计划：第 5-6 周。交付物：构建「录音 → AEC → NS → AGC → VAD → ASR」完整前端，输出各阶段音频对比试听、DNSMOS 分数链、前端开关对 ASR CER 的消融表。

---

## 1. 第一性原理：流水线顺序是因果约束的体现

### 1.1 为什么 AEC 必须在 NS 前

AEC 假设"麦克风信号里的回声 = 参考信号经线性路径卷积"。降噪（NS）对频谱做**非线性**修改，会破坏回声与参考之间的线性对应——AEC 找不到回声、消不掉。因此：

$$
\text{AEC（线性对齐消回声）} \;\to\; \text{NS（非线性去噪）} \;\to\; \text{AGC} \;\to\; \text{VAD}
$$

**这条链在干什么**：把四个模块按"破坏性"排序——越靠前的模块对信号的改动越"温柔、可逆、线性"，越靠后越"自由、非线性"。AEC 做的是减法（减去与参考线性相关的回声），必须最先；NS 做谱整形（乘时变增益）；AGC 乘整体增益（最温和，但会改变后续模块看到的电平）；VAD 只做判决不改信号，放最后还能享受前面所有处理的红利。**任何一个对信号做了非线性改动的模块，都会让排在它后面的"依赖线性结构"的模块失效**——这就是顺序的因果逻辑。

### 1.2 为什么 AGC 在 NS 后

降噪需要稳定的信噪比估计；若先 AGC 放大弱信号，噪声也被放大，干扰降噪的噪声谱估计。先降噪稳住信号，再 AGC 调响度。

### 1.3 为什么 VAD 在最后

VAD 需要尽可能干净的信号来判断"是否有语音"；放在链尾受益于前面所有处理。同时 VAD 的输出（语音段）供 ASR 只处理有效区间。

**这条顺序就是本周交付流水线的骨架**——也是自测题"前端与后端接口"的答案。

---

## 2. WebRTC APM 模块与配置

### 2.1 模块职责

| 模块 | 作用 | 关键配置 |
| --- | --- | --- |
| AEC3 | 回声消除 | 延迟估计、保守/激进 |
| NS（NoiseSuppression） | 噪声抑制 | 档位（低/中/高/极高） |
| AGC2 | 自动增益 | 目标电平、限幅 |
| VAD | 语音检测 | 模式 0-3（激进程度） |

### 2.2 档位权衡

- **保守档**：处理轻、语音失真小，但残留噪声/回声多；
- **激进档**：去噪/回声狠，但可能伤语音（发闷、音乐噪声）。

生产上通常 NS 用中-高档、AEC 用默认、AGC 限幅防削波。**没有万能配置，按场景调**。

### 2.3 Python 调用

```bash
pip install webrtc-noise-gain       # NS + AGC + VAD（基于 freedesktop webrtc-audio-processing）
```

```python
import numpy as np
from webrtc_noise_gain import AudioProcessor

# 参数：auto_gain 0-31（目标 dBFS，0=关闭）；noise_suppression 0-4（抑制强度，0=关闭）
# 输入：16 kHz 单声道 int16；每次处理 10 ms = 160 样本 = 320 字节
ap = AudioProcessor(15, 2)

chunk_in = (np.random.randn(160) * 1000).astype(np.int16).tobytes()
out = ap.Process10ms(chunk_in)
print(out.is_speech)                          # 内置 VAD 的判决
print(np.frombuffer(out.audio, dtype=np.int16).shape)   # (160,) 处理后的 10 ms 音频
```

**关于 AEC 的现实情况**：这个包**不含 AEC**；PyPI 上名为 `webrtc-audio-processing` 的旧绑定（0.1.3）基于约 2016 年的 C++ 库 v0.3（老 AEC 而非 AEC3），维护也停滞。要在 Python 里跑真正的 AEC3，目前需自行编译 freedesktop 的 webrtc-audio-processing（≥1.0 含 AEC3）并写绑定。因此本教程流水线中 AEC 用第 3 周的手写 NLMS 作为占位实现，NS/AGC/VAD 用上面的包真实接入。

---

## 3. 构建完整前端流水线（交付核心）

### 3.1 流水线类

```python
import numpy as np
import torch

class SpeechFrontend:
    """
    录音 -> AEC -> NS -> AGC -> VAD 的完整前端。
    各模块可开关，用于消融实验。
    NS/AGC 用 webrtc-noise-gain 真实接入；AEC 留接入点（第 3 周 NLMS 或自编译 AEC3）。
    """
    def __init__(self, sr=16000,
                 enable_aec=True, enable_ns=True,
                 enable_agc=True, enable_vad=True):
        assert sr == 16000, "webrtc-noise-gain 仅支持 16 kHz"
        self.sr = sr
        self.enable_aec = enable_aec
        self.enable_ns = enable_ns
        self.enable_agc = enable_agc
        self.enable_vad = enable_vad
        self._init_modules()

    def _init_modules(self):
        if self.enable_vad:
            self.vad_model, self.vad_utils = torch.hub.load(
                'snakers4/silero-vad', 'silero_vad', trust_repo=True)
            self.get_ts = self.vad_utils[0]

    def process(self, mic: np.ndarray, far: np.ndarray = None):
        """
        mic: 麦克风信号; far: 远端参考（AEC 用，可空）。
        返回 (增强后波形, 语音段时间戳列表)。
        """
        x = mic.astype(np.float32).copy()

        # 1. AEC（需要远端参考；接入第 3 周的 nlms_aec 或自编译的 AEC3 绑定）
        if self.enable_aec and far is not None:
            # from week3 import nlms_aec
            # x, _ = nlms_aec(x, far, L=int(0.15 * self.sr), freeze=True)
            pass

        # 2+3. NS + AGC（webrtc-noise-gain：一次实例同时做两件事，按开关配置档位）
        if self.enable_ns or self.enable_agc:
            from webrtc_noise_gain import AudioProcessor
            auto_gain = 15 if self.enable_agc else 0     # 0=关闭；[0,31] dBFS
            ns_level = 2 if self.enable_ns else 0        # 0=关闭；[0,4]
            ap = AudioProcessor(auto_gain, ns_level)
            x_i16 = np.clip(x * 32768, -32768, 32767).astype(np.int16)
            out = []
            for j in range(0, len(x_i16) - 159, 160):    # 每次处理 10 ms（不足 10 ms 的尾部丢弃）
                chunk = ap.Process10ms(x_i16[j:j+160].tobytes())
                out.append(np.frombuffer(chunk.audio, dtype=np.int16))
            x = np.concatenate(out).astype(np.float32) / 32768.0

        # 4. VAD：返回语音段（放链尾，享受前面所有处理的红利）
        segments = None
        if self.enable_vad:
            segments = self.get_ts(torch.from_numpy(x).float(),
                                   self.vad_model, sampling_rate=self.sr)
        return x, segments

    def keep_speech(self, x: np.ndarray, segments) -> np.ndarray:
        """按 VAD 时间戳拼接语音段（无语音时返回空数组）。"""
        if not segments:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([x[t['start']:t['end']] for t in segments])
```

**注意**：
- `AudioProcessor(auto_gain, ns_level)` 的两个参数控制 NS/AGC 的档位，把 `0` 传给任一项即关闭它——这正是消融实验需要的"逐模块开关"；
- AEC 是唯一的占位接入点（Python 生态暂无维护良好的 AEC3 绑定，见 2.3 节）——把第 3 周的 `nlms_aec` 接进去即可闭环。这个类的价值在于**统一编排 + 可开关消融**。

### 3.2 端到端：前端 + ASR

```python
import tempfile
import soundfile as sf
from funasr import AutoModel

class FrontendASR:
    def __init__(self):
        self.frontend = SpeechFrontend()
        # SenseVoiceSmall：多语言 ASR，CPU 可跑，首次运行自动下载权重
        self.asr = AutoModel(model="iic/SenseVoiceSmall")

    def transcribe(self, mic, far=None):
        enhanced, segments = self.frontend.process(mic, far)
        # 只把语音段喂给 ASR（VAD 的价值：跳过静音/噪声段）
        speech = self.frontend.keep_speech(enhanced, segments)
        if len(speech) == 0:
            return ""
        # FunASR 对数组输入的支持随版本变化，写临时 wav 最稳
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, speech, self.frontend.sr)
            res = self.asr.generate(input=f.name)
        return res[0]["text"]
```

---

## 4. 消融实验：前端对 ASR 的增益（交付）

### 4.1 实验设计

核心问题：**前端处理到底让 ASR 准了多少？** 设计：

1. 准备 10 条带噪/带回声录音 + 人工标注文本；
2. 四种配置跑 ASR：
   - A：无前端（原始音频直接识别）；
   - B：仅降噪；
   - C：降噪 + AGC；
   - D：完整前端（AEC+NS+AGC+VAD）；
3. 每种配置算 CER，输出消融表。

```python
from jiwer import cer
import numpy as np

def normalize_zh(s):
    import re
    return re.sub(r"[，。！？、,.!?\s\"'《》<>()\[\]]", "", s).lower()

def eval_config(fe_asr, pairs, config):
    """config: dict 控制前端各模块开关"""
    fe_asr.frontend.enable_ns = config.get("ns", False)
    fe_asr.frontend.enable_agc = config.get("agc", False)
    fe_asr.frontend.enable_vad = config.get("vad", False)
    refs, hyps = [], []
    for wav_path, ref in pairs:
        import soundfile as sf
        mic, _ = sf.read(wav_path, dtype="float32")
        if mic.ndim > 1: mic = mic.mean(axis=1)
        hyp = fe_asr.transcribe(mic)
        refs.append(normalize_zh(ref))
        hyps.append(normalize_zh(hyp))
    return cer(refs, hyps)

configs = {
    "A_无前端": {},
    "B_仅降噪": {"ns": True},
    "C_降噪+AGC": {"ns": True, "agc": True},
    "D_完整前端": {"ns": True, "agc": True, "vad": True},
}
# pairs = [(wav_path, ref_text), ...] 你的 10 条测试数据
print(f"{'配置':<14}{'CER':>8}")
for name, cfg in configs.items():
    # c = eval_config(fe_asr, pairs, cfg); print(f"{name:<14}{c*100:>7.2f}%")
    print(f"{name:<14}  (填入实测)")
```

### 4.2 结果解读模板

**预期趋势**：A（无前端）CER 最高；加降噪（B）显著下降；加 AGC/VAD（C/D）进一步小幅改善。这张表直接回答学习计划的问题——"**前端提升 1 分贝，ASR CER 改善多少**"，用你的实测数字填。

**同时记录**：
- 各阶段音频的试听对比（原始/降噪后/完整前端）；
- 若有 DNSMOS，输出分数链（原始 → 各阶段）。

---

## 5. 工程权衡与失效模式

### 5.1 权衡

- **处理强度**：激进前端更干净但可能伤语音、增加延迟；按场景调档；
- **模块级联误差**：每级都可能引入失真，级联放大。需消融验证每级的净收益；
- **实时性**：完整前端有累积延迟，实时通信要控制每级帧长。

### 5.2 失效模式

1. **顺序颠倒**：NS 在 AEC 前 → 回声消不掉（第 1 节原理）。这是最经典的配置错误。
2. **过度处理**：全开激进档，语音发闷、ASR 反而下降。消融表会暴露。
3. **VAD 切词**：VAD 在词中误断，ASR 漏字。调端点参数。
4. **AGC 放大噪声**：降噪不足时 AGC 把噪声也放大。确保降噪在前。

---

## 6. 延伸思考题（含解析）

**Q1**：为什么前端顺序是 AEC → NS → AGC → VAD？任意两级颠倒会怎样？
**A**：AEC 依赖回声与参考的线性关系，必须最先（非线性处理会破坏它）；NS 在 AGC 前以稳定信噪比估计；VAD 最后用干净信号判活。颠倒如 NS→AEC 会让回声消不掉；AGC→NS 会让降噪的噪声谱估计失真。

**Q2**：怎么量化"前端对 ASR 的增益"？只看音频质量指标够吗？
**A**：不够。音频指标（PESQ/DNSMOS）衡量听感，但 ASR 关心可懂度与特征稳定性。必须在下游 ASR 上测 CER 消融，才能证明前端对最终任务的净收益。两者结合：音频指标解释"干净了多少"，CER 回答"识别准了多少"。

**Q3**：完整前端全开激进档，ASR 反而变差，可能为什么？
**A**：过度降噪引入音乐噪声/频谱失真，AGC 放大残留噪声，这些伪影破坏 ASR 特征。激进档的失真代价超过了去噪收益。需消融找到每级的最优强度。

**Q4**：实时通信场景，前端延迟怎么控制？
**A**：每级用因果、短帧设计；AEC/NS 用流式模型（如 DeepFilterNet：20 ms 帧 + 20 ms 前瞻，总延迟 40 ms）；避免需要长上下文的非因果处理。总算法延迟控制在几十毫秒内。

**Q5**：你的前端如何与 Stage 7 的端侧助手对接？
**A**：前端（AEC+NS+AGC+VAD）作为助手的麦克风输入预处理，输出干净语音段喂端侧 ASR（SenseVoice-Small）。前端质量直接决定 ASR 上限，是助手可用性的关键一环。

---

## 本周交付清单

- [ ] 构建 `SpeechFrontend` 流水线类（AEC/NS/AGC/VAD 可开关）。
- [ ] 接入第 2 周降噪、第 3 周 AEC、第 4 周 AGC/VAD。
- [ ] 跑通「前端 + SenseVoice」端到端，输出 10 条测试的转写。
- [ ] 完成四配置消融表（CER），记录各阶段试听与质量分数链。
- [ ] 能解释流水线顺序的因果逻辑。

## Stage 3 总结

完成本阶段，你已能：

1. **推导**：维纳增益 $G=\xi/(1+\xi)$、DD 递推、NLMS 更新、FXLMS filtered-x；
2. **实现**：维纳滤波、DeepFilterNet 推理、NLMS 回声消除、AGC、Silero VAD；
3. **系统**：WebRTC 级前端流水线 + 对 ASR 增益的消融验证。

对照学习计划第 5-6 周自测清单核验后，进入 **Stage 4：说话人分离与声纹识别**。
