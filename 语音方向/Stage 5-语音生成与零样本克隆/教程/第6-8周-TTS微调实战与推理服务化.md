# 第 6-8 周教程：TTS 微调实战与推理服务化

> **本周要回答的三个问题**
> 1. 30 分钟干声数据如何变成一份可微调的训练集？录音规范、转写、对齐、过滤各有什么坑？
> 2. CosyVoice SFT / GPT-SoVITS 微调 / F5-TTS LoRA 三条微调路线怎么选？
> 3. 流式 TTS 服务如何压测？并发 10 路时 RTF 与延迟怎么变？

对应学习计划：第 6-8 周。交付物：① 录制/准备 ≥30 分钟干声，完成一次 TTS 微调；② 盲评微调前后的说话人相似度 + 合成语音喂 ASR 测可懂度（WER）；③ FastAPI 流式服务 + 并发压测。

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
# 用 SenseVoice 自动转写（复用 Stage 2）
from funasr import AutoModel
asr = AutoModel(model="iic/SenseVoiceSmall")

def transcribe_corpus(wav_dir):
    samples = []
    for wav in sorted(glob(f"{wav_dir}/*.wav")):
        res = asr.generate(input=wav)
        text = clean_label(res[0]["text"])    # 去掉 <|zh|> 等特殊标签
        if 0.5 < duration(wav) < 15 and len(text) > 2:
            samples.append({"audio": wav, "text": text})
    return samples
```

**关键**：转写标签必须**人工抽查**——ASR 错误会直接教坏 TTS。

### 1.3 数据量与目标的关系

| 数据量 | 可达效果 | 适用路线 |
| --- | --- | --- |
| 5 秒 - 1 分钟 | 零样本/少样本克隆（音色接近） | GPT-SoVITS 零样本、CosyVoice 零样本 |
| 5 - 30 分钟 | 少样本微调（高贴合） | GPT-SoVITS 微调、F5-TTS LoRA |
| 30 分钟 - 数小时 | SFT（专业级贴合） | CosyVoice SFT |

---

## 2. 三条微调路线

### 2.1 GPT-SoVITS 微调（最低成本）

```bash
# WebUI 流程：
# 1. 上传 1-5 分钟参考音频 + 转写
# 2. 一键数据标注（ASR + 对齐）
# 3. 微调 GPT（语义）与 SoVITS（音色）两个模型
# 4. 推理测试
```

**优点**：上手最快、工具链全、几分钟出结果。
**缺点**：工业化程度低，流式/服务化需自己包。

### 2.2 CosyVoice SFT（工业级贴合）

```python
# CosyVoice SFT 模式：用格式化数据微调
# 数据格式：(音频, 转写) 对，几十小时级别
# 训练后得到专属音色模型，支持流式推理
```

**优点**：工业级、流式、贴合度高。
**缺点**：需要更大数据量（小时级）与 GPU。

### 2.3 F5-TTS LoRA（轻量）

```python
# 对 F5-TTS 的 DiT 骨干做 LoRA 微调
# 只需少量数据（5-30 分钟），训练快
# 保留底模能力，插入目标音色
```

**优点**：轻量、保留通用能力、可插拔。
**缺点**：无原生流式、服务化需自建。

### 2.4 选型建议

- **快速出可用音色** → GPT-SoVITS；
- **工业级 + 流式产品** → CosyVoice SFT；
- **轻量 + 想学 LoRA** → F5-TTS。

---

## 3. 微调实战（交付核心）

### 3.1 执行一次微调

以 GPT-SoVITS 为例（最低门槛）：

```bash
# 1. 数据准备（30 分钟干声 + SenseVoice 转写 + 人工抽查）
# 2. 启动微调（WebUI 或命令行）
python webui.py finetune \
    --data_path ./my_dataset \
    --pretrained_gpt ./pretrained/gpt-weights.ckpt \
    --pretrained_sovits ./pretrained/sovits-weights.ckpt

# 3. 训练完成后，用微调权重推理
python webui.py inference --gpt ./finetuned/gpt.ckpt --sovits ./finetuned/sovits.ckpt
```

（具体命令以仓库最新文档为准。）

### 3.2 盲评：微调前后的相似度

```python
# 用同一段目标文本，分别用 微调前（零样本）与 微调后 合成
# 盲评维度：说话人相似度、自然度、咬字准确度
# 组织 3 人盲评打分（1-5）
```

### 3.3 可懂度：合成语音喂 ASR 测 WER

**客观指标**：把合成的语音再用 ASR 识别，看识别率——衡量"咬字是否清晰"：

```python
from jiwer import wer

def intelligibility(tts_model, asr_model, texts):
    wers = []
    for text in texts:
        wav = tts_model.synthesize(text)
        hyp = asr_model.transcribe(wav)
        wers.append(wer(normalize(text), normalize(hyp)))
    return sum(wers) / len(wers)

wer_before = intelligibility(zero_shot_model, asr, test_texts)
wer_after = intelligibility(finetuned_model, asr, test_texts)
print(f"可懂度 WER: 微调前 {wer_before:.2%} -> 微调后 {wer_after:.2%}")
```

**预期**：微调后相似度主观分上升，可懂度（WER）持平或略降（咬字更清晰）。**两个指标都要报**——相似度升但可懂度崩 = 失败。

---

## 4. 推理服务化（交付）

### 4.1 FastAPI 流式服务

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import io

app = FastAPI()

@app.post("/tts")
def tts(text: str, stream: bool = True):
    def generate():
        for chunk in tts_model.synthesize_stream(text):
            buf = encode_wav_chunk(chunk)     # 编码为 wav 字节流
            yield buf
    if stream:
        return StreamingResponse(generate(), media_type="audio/wav")
    else:
        wav = tts_model.synthesize(text)
        return StreamingResponse(io.BytesIO(encode_wav(wav)),
                                 media_type="audio/wav")
```

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
