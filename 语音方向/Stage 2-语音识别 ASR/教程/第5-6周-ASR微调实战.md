# 第 5-6 周教程：ASR 微调实战——从数据到 bad case 归因

> **本周要回答的四个问题**
> 1. 一份 `{音频, 文本}` 的领域数据集，如何变成训练框架能吃的数据？音频列是怎么解码的？
> 2. LoRA 微调 Whisper/SenseVoice 时，为什么冻结 encoder、只训低秩适配层？
> 3. CER 怎么算？为什么必须逐条做 bad case 归因表而不只看平均值？
> 4. 微调后"聋了"（通用能力退化）怎么发现、怎么防？

对应学习计划：第 5-6 周。交付物：云端完成一次 500-1000 条领域数据的 LoRA 微调，输出微调前后 CER 对比、Loss 曲线、5 条典型 bad case 归因。

---

## 1. 第一性原理：为什么微调、微调什么

### 1.1 微调的动机

通用 ASR（Whisper/SenseVoice）在**开放域**鲁棒，但在**垂直域**（医疗术语、方言、特定口音、行业黑话）词表外错误率高。微调用少量领域数据把模型"拽"到目标分布上。

### 1.2 冻结策略的逻辑

全量微调大模型：显存爆炸 + 灾难性遗忘。标准配方是**分层冻结**。以 Whisper（encoder-decoder 结构）为例：

| 组件 | 冻结? | 原因 |
| --- | --- | --- |
| Audio Encoder | ✅ 冻结 | 声学表征已足够通用，动了易遗忘、参数多 |
| Decoder（LoRA） | ❌ LoRA | 低秩适配输出端的"语言分布"，参数效率高 |

这与你在 VLM 阶段学的"冻结视觉塔 + 训 projector + LoRA LLM"同构——Whisper 没有独立的 projector（encoder 输出直接进 decoder 的 cross-attention），所以少一层；SenseVoice/Paraformer 的微调用 FunASR 官方脚本（3.3 节），同样建议小学习率、少 epoch 防遗忘。**多模态微调方法论是通用的**。

### 1.3 LoRA 回顾（一句话）

在权重 $W$ 旁挂低秩增量 $\Delta W = BA$（$A \in \mathbb{R}^{r\times d}$, $B \in \mathbb{R}^{d\times r}$, $r \ll d$），只训 $A, B$。参数量从 $d^2$ 降到 $2rd$，显存与时间大幅下降，且可随时合并或插拔。

---

## 2. 数据准备：从原始音频到训练格式

### 2.1 数据格式

先统一维护一份最朴素的 jsonl（两个键：音频路径 + 文本），再按框架要求转换：

```json
{"audio": "/data/train/0001.wav", "text": "瑞芯微的RK3588有6TOPS算力"}
{"audio": "/data/train/0002.wav", "text": "今天把量化模型部署到昇腾910B"}
```

各框架的"包装"不同，别混用：

- **HF `transformers`（Whisper 微调，3.1 节）**：就用上面这份，自己写 `Dataset.from_list` 加载；
- **FunASR 官方微调（SenseVoice/Paraformer，3.3 节）**：jsonl，`source` 键放音频路径、`target` 键放文本，SenseVoice 还要加语言/情感/事件等前缀字段（官方示例见 [SenseVoice 仓库 `data/train_example.jsonl`](https://github.com/FunAudioLLM/SenseVoice/blob/main/data/train_example.jsonl)）；
- **ms-swift（Qwen2-Audio/Qwen3-ASR 等音频大模型）**：messages + `audios` 格式，如 `{"messages": [{"role": "user", "content": "<audio>转写这段音频"}, {"role": "assistant", "content": "瑞芯微的RK3588有6TOPS算力"}], "audios": ["/data/train/0001.wav"]}`。注意 ms-swift 当前版本**不再直接支持微调 openai/whisper**，Whisper 请走 3.1 节的 transformers 路线。

要点（所有框架通用）：

- **音频采样率统一**到 16 kHz 单声道（用你的 `AudioPipeline` 预处理一遍）；
- **文本是规范化后的转写**（去标点或与目标一致）；
- **时长过滤**：去掉 < 0.5 s 或 > 30 s 的极端样本（噪声或超长；Whisper 上限 30 秒）。

### 2.2 datasets 音频列的解码机制

HF `datasets` 的 `Audio` 特征**惰性解码**：读取时只存路径，访问时才解码为波形。批处理时由 collator 统一重采样 + 填充：

```python
from datasets import Dataset, Audio

ds = Dataset.from_list([
    {"audio": "data/0001.wav", "text": "瑞芯微的RK3588有6TOPS算力"},
    # ... 500-1000 条
])
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

# 访问时才解码
sample = ds[0]
print(type(sample["audio"]["array"]), sample["audio"]["sampling_rate"])
# <class 'numpy.ndarray'> 16000
```

**为什么惰性解码重要**：数据集可能有几百小时音频，全部预解码进内存会爆。惰性解码 + 流式 collator 是标准做法。

### 2.3 数据量与配比

- 500-1000 条领域数据足以让垂直域 CER 显著下降；
- **混入通用数据**（如 10-20% AISHELL）防止灾难性遗忘——这是第 4 问的答案之一；
- 文本标注用你 Stage 2 第 3 周的 SenseVoice 预标注 + 人工校对，半自动降成本。

---

## 3. 微调用例：transformers + PEFT 微调 Whisper

微调 Whisper 的主流做法是 Hugging Face 官方路线：`WhisperProcessor` 处理特征、`Seq2SeqTrainer` 跑训练、`jiwer`/`evaluate` 算 WER，LoRA 用 `peft` 挂在 Decoder 的注意力层上（官方博客与 [HF 任务页](https://huggingface.co/blog/fine-tune-whisper) 的标准配方，以下为适配本教程数据格式的骨架）。

```bash
pip install "transformers" datasets evaluate jiwer accelerate peft soundfile
```

### 3.1 训练脚本骨架（云端 GPU 运行）

```python
import torch, evaluate, re
from datasets import Dataset, Audio
from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

MODEL = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(MODEL, language="zh", task="transcribe")

# ---- 数据：2.1 节的 jsonl ----
import json
rows = [json.loads(l) for l in open("/data/asr_train.jsonl")]
ds = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=16000))

def prepare(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(audio["array"],
        sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

ds = ds.map(prepare, remove_columns=ds.column_names)
split = ds.train_test_split(test_size=0.1, seed=0)

def collate(features):
    import numpy as np
    feats = [{"input_features": f["input_features"]} for f in features]
    batch = processor.feature_extractor.pad(feats, return_tensors="pt")
    labels = processor.tokenizer.pad(
        {"input_ids": [f["labels"] for f in features]},
        return_tensors="pt", padding=True)
    batch["labels"] = labels["input_ids"].masked_fill(
        labels["attention_mask"].ne(1), -100)   # padding 位置不计损失
    return batch

# ---- LoRA：只训 Decoder 注意力（q/v），冻结 Encoder ----
# 注意用正则限定 decoder.*，否则 Encoder 里的 q/v 也会被挂上 LoRA
from peft import LoraConfig, get_peft_model
model = WhisperForConditionalGeneration.from_pretrained(MODEL)
model.config.forced_decoder_ids = None
model.generation_config.language = "zh"
model.generation_config.task = "transcribe"
lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules=r"decoder\..*(q_proj|v_proj)")
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()    # 观察：可训练参数只占总量的百分之几

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    ids = pred.label_ids
    ids[ids == -100] = processor.tokenizer.pad_token_id
    pred_ids = pred.predictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    ref_str = processor.batch_decode(ids, skip_special_tokens=True)
    norm = lambda s: re.sub(r"[，。！？、,.!?\s]", "", s)
    return {"wer": wer_metric.compute(
        references=[norm(s) for s in ref_str],
        predictions=[norm(s) for s in pred_str])}

args = Seq2SeqTrainingArguments(
    output_dir="output/whisper-lora",
    per_device_train_batch_size=8,
    learning_rate=1e-3,          # LoRA 学习率比全量微调高 1-2 个量级
    num_train_epochs=3,
    eval_strategy="epoch",
    predict_with_generate=True,  # 评测时真的跑解码，才能算 WER
    logging_steps=10,
    save_total_limit=2,
)

trainer = Seq2SeqTrainer(
    model=model, args=args, train_dataset=split["train"],
    eval_dataset=split["test"], data_collator=collate,
    compute_metrics=compute_metrics,
    processing_class=processor,   # 旧版 transformers 参数名叫 tokenizer=
)
trainer.train()
trainer.save_model("output/whisper-lora/final")
```

**关键点解读**：

- `target_modules=r"decoder\..*(q_proj|v_proj)"`：LoRA 只挂 Decoder 的注意力投影；**必须用正则限定 `decoder.*`**——Whisper 的 Encoder 里也有同名的 `q_proj/v_proj`，不限定的话 Encoder 会一起被微调（冻结 Encoder 的策略就失效了）；
- `lora_rank=8`：秩，8-16 常用；
- `learning_rate=1e-3`：LoRA 学习率通常比全量微调（~1e-5）高一到两个量级；
- `labels` 里 padding 位置填 `-100`：`CrossEntropyLoss` 自动忽略这些位置；
- 中文场景建议把 `compute_metrics` 里加一层归一化再算 WER/CER（4.1 节）；
- 脚本骨架 API 以你安装版本的[官方文档](https://huggingface.co/docs/transformers/model_doc/whisper)为准。

### 3.2 训练观察

用 WandB 或框架自带日志盯三条曲线：

1. **训练 Loss**：应平滑下降；若震荡 → 学习率过高；
2. **验证 WER/CER**：真正的目标指标（Loss 降 ≠ CER 降）；
3. **梯度范数**：异常飙升预示发散。

### 3.3 微调 SenseVoice / Paraformer：FunASR 官方脚本

阿里官方为这两个模型提供 FunASR 微调入口（[SenseVoice 仓库 `finetune.sh`](https://github.com/FunAudioLLM/SenseVoice/blob/main/finetune.sh)），核心是调用 FunASR 自带的 `train_ds.py` 训练器：

```bash
git clone https://github.com/FunAudioLLM/SenseVoice.git
cd SenseVoice
# 数据格式见 data/train_example.jsonl：source=音频路径, target=文本
bash finetune.sh    # 默认 iic/SenseVoiceSmall, lr=2e-4, 50 epoch
```

常见改法：小数据把 `train_conf.max_epoch` 调小（如 10-20）、`optim_conf.lr` 降到 1e-4 以下，避免过拟合。

### 3.4 常见故障（学习计划第 6 周要求掌握）

- **GPU OOM**：减 batch、开 gradient checkpointing、用 QLoRA；
- **Loss 不收敛**：学习率过大、数据文本未规范化、音频采样率不一致；
- **CER 先降后升**：过拟合，早停。

---

## 4. 评测闭环：CER 与 bad case 归因

### 4.1 CER 计算

CER（Character Error Rate，字错误率）定义为**编辑距离除以参考文本长度**——分母是**参考（人工标注）的字数**，不是识别结果的字数，这一点必须记牢（WER 同理，分母是参考词数）：

$$
\text{CER} = \frac{S + D + I}{N_{\text{ref}}}
$$

**每个符号的物理含义**：$S$（Substitution）替换——参考是"起"识别成"齐"；$D$（Deletion）删除——参考有、识别结果漏掉的"漏字"；$I$（Insertion）插入——识别结果多出来的"幻听字"；$N_{\text{ref}}$ 是参考文本总字数。三个数由把"参考 → 假设"对齐所需的最少编辑操作统计而来。

**算个小例子**：参考"量化感知训练"（6 字），识别成"量化感之训练"——1 个替换（知↔之），无删无插，CER = 1/6 ≈ 16.7%。若识别成"量化感训练"（漏了"知"），$D=1$，CER 同为 1/6——但根因完全不同（见 4.2 节归因分类学）。

```python
from jiwer import cer

def normalize_zh(s):
    import re
    s = re.sub(r"<\|.*?\|>", "", s)   # SenseVoice 的 <|...|> 标签
    return re.sub(r"[，。！？、,.!?\s\"'《》<>()\[\]]", "", s).lower()

refs = [normalize_zh(r) for r in ground_truths]
hyps = [normalize_zh(h) for h in predictions]
print(f"CER = {cer(refs, hyps)*100:.2f}%")
```

注意 `jiwer.cer` 传入**列表**时做的是**语料级聚合**：全部句子的编辑操作总数 ÷ 全部参考字数（本地实测：两句参考共 10 字、1 处插入 → 返回 0.1）。所以长句权重大于短句；若你想要"每句 CER 再平均"的宏平均，需自己逐句计算。中文按字错误率（character）算，`jiwer.cer` 直接支持；英文用 `jiwer.wer`（按词）。

三类错误要分开看——它们指向不同的根因（下一节的归因表）。

### 4.2 逐条归因表（交付核心）

只看平均 CER 会掩盖问题。必须建一张逐条表：

| 音频 | 参考 | 微调前 | 微调后 | 错误类型 | 归因 |
| --- | --- | --- | --- | --- | --- |
| 0012 | 昇腾910B | 升腾910B | 昇腾910B | 替换→修复 | 领域词，微调生效 |
| 0045 | RK3588 | RK3588 | RK 3588 | 插入 | 中英边界分词，待优化 |
| 0087 | 量化感知训练 | 量化感知训练 | 量化感知 | 删除 | 尾词丢失，端点问题 |

**归因分类学**（面试直接可用）：

1. **领域词表外**：专有名词错 → 微调有效或需热词；
2. **中英混读边界**：英文与中文交界处错 → 分词/对齐问题；
3. **口音/方言**：声母韵母混淆 → 需更多该口音数据；
4. **噪声鲁棒**：嘈杂段错误率飙升 → 需增强数据增广；
5. **端点截断**：句首/句尾字丢失 → VAD/解码边界问题。

这张表的价值：**把"平均提升 3%"变成"修复了领域词、但引入了边界插入"**——这才是工程师的汇报方式。

### 4.3 灾难性遗忘检测

微调后必须跑**通用集回归测试**（如 AISHELL 子集）：

```python
# 微调前后在同一通用集上测
cer_general_before = evaluate(model_base, general_set)
cer_general_after  = evaluate(model_lora, general_set)
print(f"通用集 CER: {cer_general_before:.2%} -> {cer_general_after:.2%}")
# 若上升 > 1-2 个百分点，说明发生遗忘，需加大通用数据配比
```

---

## 5. 工程权衡与失效模式

### 5.1 权衡

- **数据量**：领域数据越多越贴合，但标注成本高；500-1000 条是性价比甜点；
- **LoRA 秩**：高 → 拟合强、易过拟合；低 → 稳健、欠拟合。8-16 起步；
- **通用数据配比**：多 → 防遗忘但稀释领域效果；少 → 反之。10-30% 常用。

### 5.2 失效模式

1. **灾难性遗忘**：领域集大涨、通用集大跌。修复：混通用数据、降学习率、减训练步数。
2. **音频-文本错位**：数据制作时音频与标注对不上，模型学到错误映射。修复：抽样人工校验。
3. **过拟合**：验证 CER 先降后升、训练 Loss 持续降。修复：早停、增广、减秩。
4. **采样率不一致**：部分音频非 16 kHz，特征提取错乱。修复：预处理统一重采样并断言。

---

## 6. 延伸思考题（含解析）

**Q1**：为什么微调要冻结 Audio Encoder？动了会怎样？
**A**：Encoder 学的是通用声学表征，参数量大；微调它易过拟合小数据、灾难性遗忘，且显存开销大。冻结它、只训 projector + LoRA，既高效又保留通用能力。

**Q2**：为什么必须混入通用数据？比例怎么选？
**A**：纯领域数据会把模型拉离通用分布，遗忘开放域能力。混 10-30% 通用数据做"锚"。比例靠回归测试调：通用集 CER 上涨超阈值就加大配比。

**Q3**：CER 的替换/删除/插入三类错误，分别指向什么根因？
**A**：替换多为声学混淆或词表外；删除多为端点截断或漏检；插入多为解码幻觉或中英边界分词。分开统计才能定位。

**Q4**：LoRA 学习率为什么比全量微调高？
**A**：LoRA 初始增量为零（$B$ 初始化为 0），需要较大学习率才能让旁路产生有效更新；且只训低秩参数，高学习率不易破坏主干。

**Q5**：微调后领域集提升但线上效果差，可能为什么？
**A**：训练/线上分布不一致——如训练是干净朗读、线上是带噪对话；或文本规范化不一致。需按线上真实分布补数据、对齐前后处理。

---

## 本周交付清单

- [ ] 准备 500-1000 条领域数据集（预标注 + 人工校对），统一 16 kHz。
- [ ] 云端跑通一次 LoRA 微调（transformers+PEFT 或 FunASR 官方脚本），记录 Loss 曲线。
- [ ] 输出微调前后 CER 对比表 + 逐条 bad case 归因表（≥5 条典型）。
- [ ] 通用集回归测试，确认无灾难性遗忘。

## Stage 2 总结

完成本阶段，你已能：

1. **推导**：CTC 前向算法（含暴力枚举验证）、1500 帧的来源、CIF 长度损失；
2. **手写**：Whisper 完整推理循环（含 KV cache）、三模型横评脚本；
3. **工程**：流式转写、端点检测、LoRA 微调、bad case 归因。

对照学习计划第 6 周自测清单核验后，进入 **Stage 3：语音增强与前端信号处理**——你在 Stage 1 第 4 周写的谱减法将在那里升级为 DeepFilterNet。
