# 第 6-8 周教程：TTS 微调实战与推理服务化

> **本周要回答的三个问题**
> 1. 30 分钟干声数据如何变成一份可微调的训练集？录音规范、转写、对齐、过滤各有什么坑？
> 2. CosyVoice SFT / GPT-SoVITS 微调 / F5-TTS LoRA 三条微调路线怎么选？
> 3. 流式 TTS 服务如何压测？并发 10 路时 RTF 与延迟怎么变？

对应学习计划：第 6-8 周。交付物：① 录制/准备 ≥30 分钟干声，完成一次 TTS 微调；② 盲评微调前后的说话人相似度 + 合成语音喂 ASR 测可懂度（WER/CER）；③ FastAPI 流式服务 + 并发压测。

---

## 1. 数据工程：录音 → 训练集

### 1.1 录音规范（决定上限）

- **48 kHz 干声**：无混响、无背景音乐、无噪声；
- **安静环境 + 电容麦**：信噪比 > 40 dB；
- **内容多样**：覆盖陈述、疑问、感叹、数字、中英混读；
- **30 分钟起步**：SFT 级贴合需要 30 分钟到几小时；少样本（GPT-SoVITS）1-5 分钟即可。

### 1.2 数据构造流水线

$$
\text{干声} \xrightarrow{\text{切句}} \xrightarrow{\text{ASR 转写}} \xrightarrow{\text{强制对齐}} \xrightarrow{\text{质量过滤}} \text{训练集}
$$

1. **切句**：按静音/标点切成 3-15 秒片段（太短无韵律、太长难对齐）；
2. **ASR 转写**：用你 Stage 2 的 SenseVoice 给音频打标签（技能复用！）；
3. **强制对齐（MFA）**：Montreal Forced Aligner 给出音素级时间戳，供需要时长标签的模型用；
4. **质量过滤**：去掉转写置信度低、时长异常、能量异常的样本。

```python
# 用 SenseVoice 自动转写（复用 Stage 2）——教学骨架，glob/duration 等按你的 Stage 2 工具补齐
import glob
import soundfile as sf
from funasr import AutoModel

asr = AutoModel(model="iic/SenseVoiceSmall")   # ModelScope 上的 SenseVoiceSmall

def clean_label(text: str) -> str:
    # SenseVoice 输出形如 <|zh|><|NEUTRAL|><|Speech|>文本
    for tag in ("<|zh|>", "<|en|>", "<|yue|>", "<|ja|>", "<|ko|>",
                "<|NEUTRAL|>", "<|Speech|>", "<|BGM|>", "<|Noise|>", "<|Emb|>"):
        text = text.replace(tag, "")
    return text.strip()

def transcribe_corpus(wav_dir):
    samples = []
    for wav_path in sorted(glob.glob(f"{wav_dir}/*.wav")):
        x, sr = sf.read(wav_path, dtype="float32")
        res = asr.generate(input=wav_path)
        text = clean_label(res[0]["text"])    # 去掉 <|zh|> 等特殊标签
        dur = len(x) / sr
        if 0.5 < dur < 15 and len(text) > 2:
            samples.append({"audio": wav_path, "text": text})
    return samples
```

**关键**：转写标签必须**人工抽查**——ASR 错误会直接教坏 TTS。

### 1.3 数据量与目标的关系

| 数据量 | 可达效果 | 适用路线 |
| --- | --- | --- |
| 5 秒 - 1 分钟 | 零样本/少样本克隆（音色接近） | GPT-SoVITS 零样本、CosyVoice 零样本 |
| 5 - 30 分钟 | 少样本微调（高贴合） | GPT-SoVITS 微调、F5-TTS LoRA（F5 建议十几分钟以上） |
| 30 分钟 - 数小时 | SFT（专业级贴合） | CosyVoice SFT |

---

## 2. 三条微调路线

### 2.1 GPT-SoVITS 微调（最低成本）

GPT-SoVITS 的微调是 **WebUI 驱动**的（`python webui.py` 启动，或 Windows 整合包双击 `go-webui.bat`），没有"一条命令微调"的官方 CLI：

```bash
# WebUI 流程（都在 "1-GPT-SoVITS-TTS" 页签里完成）：
# 1A 训练集格式化工具：上传音频 -> 切片 -> （可选 UVR5 降噪）-> ASR 转写 -> 人工校对转写
# 1B 微调 SoVITS 模块（音色），1C 微调 GPT 模块（韵律）——两个模块分别训练
# 1D 推理：填参考音频 + 目标文本即可试听
```

**优点**：上手最快、工具链全（切片/降噪/ASR/校对/训练/推理一站化）、低配 GPU 可跑。
**缺点**：工业化程度低，流式/服务化需自己包（训练产物是 SoVITS 的 `.pth` 与 GPT 的 `.ckpt` 两份权重，要用推理脚本自行封装）。

### 2.2 CosyVoice SFT（工业级贴合）

CosyVoice 仓库（`CosyVoice-300M-SFT` 系）提供训练示例脚本（`examples/libritts/cosyvoice/run.sh`），把 (音频, 转写) 对格式化后微调出专属音色模型，用 `inference_sft` 调用：

```python
# 数据格式：(音频路径, 转写) 对，数据量小时级起步更稳
# 训练流程见仓库 examples/ 目录（基于 run.sh，内部是数据准备 + LLM/flow 分阶段训练）
# 训练后得到专属音色模型，支持流式推理
```

**优点**：工业级、原生流式、贴合度高。
**缺点**：需要更大数据量与 GPU、训练链路比 WebUI 复杂。

### 2.3 F5-TTS LoRA（轻量）

F5-TTS 官方支持对 DiT 骨干做 **LoRA** 微调（`pip install f5-tts` 提供 `f5-tts_finetune-cli` 与 `f5-tts_finetune-gradio` 两个入口）：

```python
# pip install f5-tts 后，官方提供 4 个命令行入口：
#   f5-tts_infer-cli / f5-tts_infer-gradio      推理
#   f5-tts_finetune-cli / f5-tts_finetune-gradio 微调（CLI 或 Gradio 界面）
# 数据格式：(音频, 转写) 对
# 社区经验：十几分钟音频起步能看到效果，约 1 小时较稳，几小时更好
# LoRA 只训低秩增量，显存占用低于全参微调
```

**优点**：轻量、保留底模通用能力、可插拔。
**缺点**：无原生流式、服务化需自建。

### 2.4 选型建议

- **快速出可用音色** → GPT-SoVITS；
- **工业级 + 流式产品** → CosyVoice SFT；
- **轻量 + 想学 LoRA** → F5-TTS。

---

## 3. 微调实战（交付核心）

### 3.1 执行一次微调

以 GPT-SoVITS 为例（最低门槛）。注意：**GPT-SoVITS 没有 `webui.py finetune` 这类子命令**——所有数据处理与训练都在 WebUI 页签里点击完成：

```bash
# 1. 下载预训练模型（HuggingFace: lj1995/GPT-SoVITS）放到 GPT_SoVITS/pretrained_models
# 2. 启动 WebUI
python webui.py

# 3. 在 WebUI 中依次完成：
#    1-A 数据格式化（切片 -> ASR 转写 -> 人工校对 -> 生成 .list 训练清单）
#    1-B SoVITS 微调（填 .list 路径，点开始训练）
#    1-C GPT 微调（同上）
#    1-D 推理页签加载微调后的两个权重试听
```

（WebUI 版本迭代较快，具体页签布局以仓库最新文档为准；训练产物是 SoVITS 的 `*.pth` 与 GPT 的 `*.ckpt` 两份微调权重。）

### 3.2 盲评：微调前后的相似度

```python
# 用同一段目标文本，分别用 微调前（零样本）与 微调后 合成
# 盲评维度：说话人相似度、自然度、咬字准确度
# 组织 3 人盲评打分（1-5）
```

### 3.3 可懂度：合成语音喂 ASR 测 CER/WER

**客观指标**：把合成的语音再用 ASR 识别，看识别率——衡量"咬字是否清晰"。**中文文本要用 CER（Character Error Rate，字错误率）而不是 WER**——中文没有空格分词，按"字"算才合理；`jiwer` 两者都有：

```python
from jiwer import cer   # 英文场景用 from jiwer import wer

def normalize(t):        # 教学骨架：去标点、繁简统一、数字归一化
    return t

def intelligibility(tts_model, asr_model, texts):
    errors = []
    for text in texts:
        wav = tts_model.synthesize(text)
        hyp = asr_model.transcribe(wav)
        errors.append(cer(normalize(text), normalize(hyp)))
    return sum(errors) / len(errors)

cer_before = intelligibility(zero_shot_model, asr, test_texts)
cer_after = intelligibility(finetuned_model, asr, test_texts)
print(f"可懂度 CER: 微调前 {cer_before:.2%} -> 微调后 {cer_after:.2%}")
```

**预期**：微调后相似度主观分上升，可懂度（CER）持平或略降（咬字更清晰）。**两个指标都要报**——相似度升但可懂度崩 = 失败。

---

## 4. 推理服务化（交付）

### 4.1 FastAPI 流式服务

```python
import io
import wave
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
SR = 24000   # 与模型输出采样率一致（CosyVoice2 为 24 kHz）

def wav_header(n_samples: int) -> bytes:
    """生成 WAV 文件头（决定总样本数，让播放器知道时长）。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)             # int16
        w.setframerate(SR)
        w.writeframes(b"\x00\x00" * n_samples)   # 只为写出合法头部
    return buf.getvalue()

@app.post("/tts")
def tts(text: str, stream: bool = True):
    if not stream:
        wav = tts_model.synthesize(text)
        return StreamingResponse(io.BytesIO(encode_wav(wav)),
                                 media_type="audio/wav")

    def generate():
        # WAV 头部只发一次（先于音频数据），后面全部是裸 PCM：
        header_sent = False
        for chunk in tts_model.synthesize_stream(text):   # 每块为 float32 波形
            pcm = (np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes()
            if not header_sent:
                yield wav_header(estimate_total_samples(text))  # 教学骨架：estimate_total_samples 按语速估算字数×每字时长补齐
                header_sent = True
            yield pcm
    return StreamingResponse(generate(), media_type="audio/wav")
```

（非流式分支里的 `encode_wav` 同为教学骨架：用上面 `wave` 模块把完整波形写入 `BytesIO` 即可。）

**这段代码的两个真坑**（比"会不会写"更重要）：

1. **WAV 不是为流式设计的容器**：它的头部写死了"总样本数"。把多个"完整 wav chunk"直接拼接给客户端，得到的是**非法文件**——只有第一个头生效。上面代码的对策是"头只发一次、后面发裸 PCM"，但总样本数是估的（**简化做法**，多数播放器能容忍）；**工程做法**是换成真正为流式设计的容器（Ogg/Opus、WebM/Opus，浏览器 `<audio>` 原生支持）。
2. **同步端点阻塞**：`def tts` 里的模型推理是阻塞计算，FastAPI 会在线程池里跑同步生成器——demo 够用，并发一高就顶不住；生产上应把推理拆到独立进程/服务（如 vLLM 风格的推理服务），网关只做转发。

**流式的关键**：用 `StreamingResponse` 边生成边下发，客户端首包延迟低——这正是你结课助手（Stage 7）需要的"边想边说"。

### 4.2 并发压测

```python
# 用 locust 或 ab 压测并发 10 路
# 指标：吞吐（句/秒）、首包延迟、完整延迟、RTF
```

```python
import asyncio, aiohttp, time

async def one_request(session, text):
    t0 = time.time()
    first = None
    async with session.post(URL, params={"text": text, "stream": True}) as r:
        async for chunk in r.content.iter_any():
            if first is None:
                first = time.time() - t0
    total = time.time() - t0
    return first, total

async def bench(concurrency=10):
    async with aiohttp.ClientSession() as session:
        tasks = [one_request(session, "测试文本") for _ in range(concurrency)]
        results = await asyncio.gather(*tasks)
    firsts = [r[0] for r in results if r[0]]
    totals = [r[1] for r in results]
    print(f"并发 {concurrency}: 首包延迟均值 {sum(firsts)/len(firsts)*1000:.0f} ms")
    print(f"           完整延迟均值 {sum(totals)/len(totals):.2f} s")
```

**观察点**：并发上升时，单卡显存与算力被分摊，**首包延迟与 RTF 都会上升**。记录"并发 1/4/10"三档数据，画出延迟-并发曲线——这是容量规划的依据（联动性能优化方向）。

---

## 5. 工程权衡与失效模式

### 5.1 权衡

- **数据量**：多 → 贴合好但成本高；少 → 快但不够像。按产品需求定档；
- **过拟合**：微调过头会丢失底模的鲁棒性（韵律变差、只会说训练集风格）；
- **流式缓冲**：缓冲小延迟低但易卡顿；缓冲大反之。

### 5.2 失效模式

1. **标签错误教坏模型**：ASR 转写错 → 模型学会错误发音。修复：人工抽查标签。
2. **过拟合**：训练集外文本韵律崩。修复：早停、控制步数、混通用数据。
3. **录音质量差**：底噪/混响被学进去。修复：录音规范 + 前端降噪。
4. **服务并发崩**：单卡扛不住多路并发。修复：多副本、队列、扩容（联动性能优化）。

---

## 6. 延伸思考题（含解析）

**Q1**：为什么 TTS 数据要"强制对齐"？
**A**：需要时长/音素级标签的模型（如某些声学模型）要知道每个音素的起止时间。MFA 强制对齐给出这个标注，让模型学到"什么音素多长"。纯 flow/LM 路线对此依赖较小。

**Q2**：微调后相似度升但可懂度崩，问题出在哪？
**A**：可能是过拟合（咬字糊）、录音质量问题、或标签错误。需用可懂度（WER）指标守住底线——相似度与可懂度必须同时达标。

**Q3**：流式服务为什么要边生成边下发？
**A**：整句生成完再下发，首包延迟 = 整句时长，用户等太久。流式边生成边播，首包延迟只取决于第一个 chunk，体验大幅提升。

**Q4**：并发压测时延迟为什么会上升？
**A**：单卡算力/显存被多路请求分摊，每路分到的计算变少，生成变慢。这是容量规划的核心——需要多卡/多副本水平扩展。

**Q5**：你怎么防止微调过拟合？
**A**：早停、控制训练步数、混入通用数据、用验证集监控韵律指标、数据量足够（别用太少数据训太多步）。

---

## 本周交付清单

- [ ] 录制/准备 ≥30 分钟干声，完成 切句→转写→对齐→过滤 流水线。
- [ ] 完成一次微调（GPT-SoVITS / CosyVoice SFT / F5-TTS LoRA 任选）。
- [ ] 盲评微调前后相似度 + 可懂度（WER）双指标。
- [ ] FastAPI 流式服务 + 并发 1/4/10 压测，画延迟-并发曲线。

## Stage 5 总结

完成本阶段，你已能：

1. **原理**：声码器三代、RVQ 离散化、VALL-E、flow matching、FSQ 语义 token；
2. **拆解**：CosyVoice 2 / F5-TTS / GPT-SoVITS 三条路线的架构与权衡；
3. **实战**：零样本克隆、TTS 微调、流式服务压测。

对照学习计划自测清单核验后，进入 **Stage 6：语音大模型与 Omni 多模态**——把你已有的 VLM 知识与本阶段的语音能力焊在一起。
