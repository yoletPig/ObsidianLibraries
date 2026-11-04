# 第 2 周教程：Whisper 深度拆解

> **本周要回答的四个问题**
> 1. 一段 30 秒音频，如何一步步变成 1500 个 Encoder Token？每一步的 Shape 是什么？
> 2. `<|transcribe|>`、`<|zh|>` 这些特殊 token 如何用一个模型统一 99 种语言 + 翻译 + 时间戳？
> 3. 手写推理循环里，KV cache 到底缓存了什么？为什么必须有它？
> 4. Whisper 的尺寸谱系（tiny→large）如何决定云/端的部署边界？

对应学习计划：第 2 周。交付物：不用 `pipeline`，手写完整 Whisper 推理循环，打印每步 Tensor Shape，与官方输出对齐，记录 CPU/GPU 的 RTF。

---

## 1. 第一性原理：用"弱监督大数据"暴力破解鲁棒性

### 1.1 前代瓶颈

Whisper 之前的 ASR 是"窄而精"：在干净、标注严格的域内数据上做到极致（如 LibriSpeech 2.7% WER），但**换域即崩**——口音、噪声、专业术语、低资源语言下性能暴跌。根因：训练数据太干净、太单一，模型学到了数据集的偏置而非"听觉能力"本身。

### 1.2 Whisper 的答案

OpenAI 收集了 **68 万小时**的多语言、带字幕互联网音频（弱监督：字幕不精确、有噪声；论文为 [Radford et al., 2022, arXiv:2212.04356](https://arxiv.org/abs/2212.04356)），不做精细清洗，而是用足够大的模型硬吃这些数据。核心洞察：

$$
\text{数据多样性} \times \text{数据规模} \;\gg\; \text{标注精度}
$$

**这个式子在说什么**：它不是数学等式，而是一句工程判断——对鲁棒性而言，数据的覆盖面（口音、噪声、领域）和总量，比每条标注的精确度重要得多。Whisper 团队有意放弃"清洗到 LibriSpeech 那么干净"，换来的是换域不崩。后缀事实：后续的 large-v3（2023-11 发布）更是用 100 万小时弱标注 + 400 万小时伪标注（用 large-v2 打的标签）继续验证这条路线。

结果是一个对噪声、口音、领域都**鲁棒**的通用模型。这个"数据工程优先于模型精巧"的思路，与 VLM 时代的大规模弱监督预训练一脉相承。

### 1.3 多任务统一：用特殊 token 编程

Whisper 不用多个模型，而是用**任务控制 token** 在解码起始处指定行为。解码总是从 `<|startoftranscript|>` 开始，随后依次给出：

$$
\langle|\text{startoftranscript}|\rangle \;\langle|\text{zh}|\rangle \;\langle|\text{transcribe}|\rangle \;\langle|\text{notimestamps}|\rangle \;\to\; \text{文字}
$$

**这行式子在干什么**：它是给解码器的"程序指令行"。Decoder 是自回归的，每一步都在预测下一个 token——把语言、任务、时间戳开关做成词表里的特殊 token，等于让"输出什么"变成生成序列的一部分，模型自己"读"到 `<|transcribe|>` 就走转写分支，读到 `<|translate|>` 就走翻译分支：

- `<|en|>` `<|zh|>` 等语言标签（99 种）→ 控制输出语言；
- `<|transcribe|>` vs `<|translate|>` → 转写原语言，还是翻译成英文；
- `<|notimestamps|>` 或省略它 → 是否输出 `<|0.00|>` 形式的时间戳 token。

一个模型、多种行为，全靠前缀 token 条件化——这就是"用 token 编程"。

---

## 2. 架构与数据流：逐步追踪 Shape

### 2.1 整体结构

Whisper 是标准的 **Encoder-Decoder Transformer**（不是纯 Decoder）：

$$
\text{波形} \xrightarrow{\text{Log-Mel}} \xrightarrow{\text{CNN 下采样}} \xrightarrow{\text{Encoder}} \xrightarrow{\text{Decoder (AR)}} \text{文字}
$$

### 2.2 特征提取：30 秒 → (80, 3000)

- 输入固定 **30 秒**（不足补零，超过切段）；
- 16 kHz × 30 s = 480000 样本；
- 25 ms 窗、10 ms 移（win_length=400、hop_length=160、n_fft=400）的 80 维 Log-Mel → $(80, 3000)$。

**Shape 怎么来的**：帧数 = $480000 / 160 = 3000$——特征提取器把"每 10 ms 一帧"拍平成除法；$80$ 是 Mel 滤波器组个数（第 1 周 Stage 1 的 Mel 频谱知识在此落地）。tiny 到 medium 都用 80 维；**large-v3 改用 128 维**，这是 v2→v3 少数的输入侧改动。所以完整的输入 shape 是 $(1, 80, 3000)$（batch=1），这就是官方 `WhisperFeatureExtractor` 输出的形状。

### 2.3 CNN 下采样 2×：(80, 3000) → (1500, d)

两层一维卷积 + GELU：**第一层步长 1（kernel 3），第二层步长 2（kernel 3）**，整体时间维减半：

$$
(80, 3000) \xrightarrow{\text{Conv1, stride 1}} (d_{\text{model}}, 3000) \xrightarrow{\text{Conv2, stride 2}} (1500, d_{\text{model}})
$$

**这两层卷积在干什么**：Transformer 的注意力是 $O(L^2)$ 的，3000 长度太贵；先用步长 2 的卷积把时间维砍半（3000→1500），同时把 80 维 Mel 通道混合升维到 $d_{\text{model}}$——相当于"免费"的一层局部特征提取（类比 Stage 1 学过的滤波器组+降采样）。$d_{\text{model}}$ 随尺寸变化（small=768, large=1280）。

随后加上**正弦（sinusoidal）位置编码**送入 Encoder——注意这是固定公式生成的正弦表，**不是可学习参数**（官方 `whisper/model.py` 中 Encoder 的 `positional_embedding` 是注册的 buffer；可学习位置编码在 Decoder 那一侧，易混淆，面试常考）。

### 2.4 Encoder 与 Decoder

- **Encoder**：标准 Transformer 块（Pre-LN 前置层归一化 + 末端 LayerNorm），输出 $(1500, d)$；
- **Decoder**：自回归生成，通过 **cross-attention 看 Encoder 输出**，自注意力带 **KV cache**。

**Shape 链小结**（以 small 为例，$d=768$）：

$$
\underbrace{480000}_{\text{样本}} \to \underbrace{(80, 3000)}_{\text{Log-Mel}} \to \underbrace{(1500, 768)}_{\text{Encoder 输入}} \to \underbrace{(1500, 768)}_{\text{Encoder 输出}} \to \text{Decoder}
$$

**自测题**（第 6 周验收原题）：30 秒音频经 2× 下采样后 Encoder 序列长度 = $3000/2 = 1500$。

---

## 3. 手写推理循环（交付核心）

### 3.1 为什么要手写

`pipeline` 把特征提取、解码、KV cache 全藏起来了。手写一遍你才能回答面试必问的"KV cache 怎么管理""解码每步算什么"。下面用 `transformers` 的模型权重，但**自己写循环**。

```python
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL = "openai/whisper-small"
device = "cpu"        # 本地无 GPU，CPU 推理；有云 GPU 时切 "cuda"

processor = WhisperProcessor.from_pretrained(MODEL)
model = WhisperForConditionalGeneration.from_pretrained(MODEL).to(device)
model.eval()

# ---- 1. 载入音频（16 kHz 单声道，≤30s）----
# librosa.ex("trumpet") 是 librosa 自带示例（音乐，需联网下载），
# 这里只用于验证 Shape 与推理链路——音乐没有中文语音，识别文本本身无意义。
# 想测真实效果，把路径换成你自己的 16 kHz 中文录音。
import librosa
wave, sr = librosa.load(librosa.ex("trumpet"), sr=16000)
wave = wave[:16000 * 30]
print(f"音频样本数: {len(wave)}")

# ---- 2. 特征提取：波形 -> (80, 3000) ----
inputs = processor(wave, sampling_rate=16000, return_tensors="pt")
mel = inputs.input_features.to(device)          # (1, 80, 3000)
print("Log-Mel shape:", tuple(mel.shape))      # (1, 80, 3000)

# ---- 3. Encoder 前向（一次）----
with torch.no_grad():
    enc_out = model.get_encoder()(mel, return_dict=True)
    enc_hidden = enc_out.last_hidden_state       # (1, 1500, 768)
print("Encoder 输出:", tuple(enc_hidden.shape)) # (1, 1500, 768)
assert enc_hidden.shape[1] == 1500, "下采样后应为 1500 帧"

# ---- 4. Decoder 自回归 + KV cache ----
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="zh", task="transcribe")

def generate_step(input_ids, past_kv, enc_hidden):
    """单步解码：返回 logits 与更新后的 KV cache。"""
    with torch.no_grad():
        out = model.model.decoder(
            input_ids=input_ids,
            encoder_hidden_states=enc_hidden,
            past_key_values=past_kv,
            use_cache=True,
        )
    logits = model.proj_out(out.last_hidden_state)   # -> 词表
    return logits, out.past_key_values

# 起始 token：<|startoftranscript|>
start_id = model.config.decoder_start_token_id
input_ids = torch.tensor([[start_id]], device=device)
past_kv = None
gen_ids = []

for step in range(220):                    # 最大解码步数
    logits, past_kv = generate_step(input_ids, past_kv, enc_hidden)
    next_logits = logits[:, -1, :]         # 取最后位置
    # 强制特殊 token（前几步按语言/任务约束）
    # forced_decoder_ids 是 (位置, token_id) 列表，位置从 1 开始计数
    for pos, tid in forced_decoder_ids:
        if step + 1 == pos:                # 第 step 个新 token 处在位置 step+1
            next_logits = torch.full_like(next_logits, -float("inf"))
            next_logits[0, tid] = 0
    next_id = int(next_logits.argmax(-1))
    gen_ids.append(next_id)
    if next_id == model.config.eos_token_id:
        break
    # 关键：有 KV cache 时，下一步只喂新 token（增量输入）
    input_ids = torch.tensor([[next_id]], device=device)

text = processor.decode(gen_ids, skip_special_tokens=True)
print("识别结果:", text)
```

**关键观察点**（务必在代码里打印验证）：

1. **KV cache 的作用**：第一次 `generate_step` 输入整段起始序列，之后每步 `input_ids` 只有 **1 个新 token**——因为历史 key/value 已缓存。若无 cache，每步要重新前向整个已生成序列，复杂度从 $O(U)$ 变 $O(U^2)$。
2. **cross-attention 的 KV**：每步都要看 Encoder 的 1500 帧输出，这部分也被缓存。
3. **强制解码**：前几步按 `forced_decoder_ids` 锁定语言/任务 token，保证输出是中文转写而非翻译。

### 3.2 与官方输出对齐验证

```python
# 官方 generate 作参照
# （新版 transformers 推荐直接传 language/task；上面那种
#   forced_decoder_ids 写法仍然有效，但会提示弃用警告）
ref = model.generate(mel, language="zh", task="transcribe")
ref_text = processor.batch_decode(ref, skip_special_tokens=True)[0]
print("官方结果:", ref_text)
print("手写结果:", text)
# 两者应高度一致（贪心下通常完全相同）
```

预期：贪心解码下手写与官方结果一致。若不一致，检查：① 是否漏了特殊 token 强制；② KV cache 是否正确传递；③ 终止条件。

### 3.3 RTF 测量（交付）

RTF（Real-Time Factor）= 处理时长 / 音频时长，< 1 表示比实时快：

```python
import time
t0 = time.time()
_ = model.generate(mel, forced_decoder_ids=forced_decoder_ids)
cpu_time = time.time() - t0
rtf = cpu_time / (len(wave) / 16000)
print(f"CPU 推理耗时 {cpu_time:.2f}s, RTF = {rtf:.2f}")
```

预期：CPU 上 small 模型 RTF 通常在 0.3–3 量级（强依赖 CPU 型号、线程数与 PyTorch 版本，**务必以自己实测为准**）。云 GPU 上可降到 0.05 以下。**记录这个对比**——它是你理解"为什么端侧要换非自回归"的定量依据：AR 模型在弱算力上太慢。

---

## 4. 尺寸谱系与部署边界

| 模型 | 参数量 | Mel 维 | Encoder 层/宽 | 典型场景 |
| --- | --- | --- | --- | --- |
| tiny | 39 M | 80 | 4 / 384 | 极端端侧、唤醒级 |
| base | 74 M | 80 | 6 / 512 | 轻量端侧 |
| small | 244 M | 80 | 12 / 768 | 端侧/边缘（本教程用） |
| medium | 769 M | 80 | 24 / 1024 | 云端高质量 |
| large-v3 | 1550 M | **128** | 32 / 1280 | 云端最高精度 |

（参数量与层数/宽度出自 Whisper 论文 [Radford et al., 2022] Table 及 [large-v3 模型卡](https://huggingface.co/openai/whisper-large-v3)。）

**large-v3（2023-11 发布）相对 large-v2 的三处变化**：① Mel 特征从 80 维换成 128 维；② 词表新增粤语（Cantonese）语言 token；③ 训练数据扩充到 100 万小时弱标注 + 400 万小时伪标注（large-v2 打标）。语言仍为 99 种。若你用 large-v3，`processor` 会自动用 128 维特征，**但自研前端代码要相应改 `n_mels`**——这是迁移时最常踩的坑。

**部署边界经验**：端侧 NPU/CPU 通常止于 small；medium 以上需要 GPU（2024-10 的 large-v3-turbo 把 decoder 从 32 层砍到 4 层、参数量降到 809M，是云端兼顾速度的新选择）。这也解释了你的结课项目为何选 SenseVoice-Small（NAR，端侧友好）而非 Whisper-medium。

---

## 5. 推理细节工程

### 5.1 长音频处理：VAD + 分段

Whisper 单次只吃 30 秒。长音频需：

1. **VAD 分段**：按静音切成 ≤30 秒片段；
2. **逐段识别 + 时间戳对齐**：每段带 `timestamps` 输出，拼回全局时间轴。

不分段直接滑窗会在窗口边界切断词语，产生拼接错误。

### 5.2 解码策略

- **贪心**：快，端侧用；
- **beam search**：官方 CLI 默认 `beam_size=5`（且仅在温度为 0 时启用；`transformers` 的 `generate` 默认贪心，需手动传 `num_beams`），更稳，云端用；
- **温度回退**：置信度低时升温重试，抑制幻觉。

### 5.3 时间戳头

Whisper 用专门的**时间戳 token**（`<|0.00|>`…`<|30.00|>`，0.02 s 粒度）而非回归头。开时间戳模式时解码在文字与时间戳 token 间交替。

---

## 6. 工程权衡与失效模式

### 6.1 权衡

- **30 秒窗口**：固定窗口简化训练与位置编码，但长音频必须分段，引入边界问题；
- **AR 解码**：质量高但延迟随长度增长——这正是 NAR（下周）要解决的；
- **模型尺寸**：精度随尺寸单调上升，但端侧算力/内存是硬约束。

### 6.2 失效模式

1. **静音段幻觉**：无语音输入时，模型"脑补"出重复文字或训练集片段。定位：看输入是否接近静音；修复：VAD 预过滤、温度回退、设置 `condition_on_previous_text=False`。
2. **长文本重复循环**：自回归陷入重复。修复：repetition penalty、no_repeat_ngram。
3. **跨语言误判**：中英混读时语言检测错误。修复：显式指定 `language`。
4. **时间戳漂移**：长音频分段拼接后时间戳错位。修复：以 VAD 全局时间为基准对齐。

---

## 7. 延伸思考题（含解析）

**Q1**：30 秒 16 kHz 音频，Encoder 输入序列长度是多少？怎么算？
**A**：$30\text{s} \times 100\text{帧/s} = 3000$ 帧（Log-Mel），经 2× 下采样 = **1500** 个 Encoder token。

**Q2**：KV cache 缓存了什么？没有它解码复杂度如何变化？
**A**：缓存已生成步的 Decoder 自注意力 key/value 与 cross-attention key/value。有 cache 每步只算新 token（$O(1)$ 增量）；没有则每步重算整个前缀，总复杂度 $O(U^2)$。

**Q3**：为什么 Whisper 用固定 30 秒窗口而不是变长？
**A**：固定长度简化批处理与位置编码、训练高效；代价是长音频需分段。工程上通过 VAD 分段弥补。

**Q4**：特殊 token 如何做到"一个模型多任务"？如果让你加一个"输出带标点"的任务，怎么设计？
**A**：在解码起始处增加一个任务控制 token（如 `<|punct|>`），用带标点的字幕数据微调，让模型学会该前缀下输出标点。本质是条件化生成。

**Q5**：为什么 Whisper 在噪声下比传统模型鲁棒？
**A**：68 万小时弱监督互联网数据覆盖了海量真实噪声、口音、领域，模型学到的是泛化的听觉-文本映射而非干净域偏置。数据多样性是鲁棒性的来源。

---

## 本周交付清单

- [ ] 手写推理循环跑通，打印 Log-Mel $(1,80,3000)$ → Encoder $(1,1500,768)$ 的 Shape。
- [ ] 手写结果与官方 `generate` 对齐。
- [ ] 测量并记录 CPU RTF（有云 GPU 则同时记录 GPU RTF 对比）。
- [ ] 能闭卷回答：1500 帧怎么来的、KV cache 缓存了什么。
