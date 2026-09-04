# 第 5-6 周教程：ASR 微调实战——从数据到 bad case 归因

> **本周要回答的四个问题**
> 1. 一份 `{音频, 文本}` 的领域数据集，如何变成训练框架能吃的数据？音频列是怎么解码的？
> 2. LoRA 微调 Whisper/SenseVoice 时，为什么冻结 encoder、只训 LoRA + projector？
> 3. CER 怎么算？为什么必须逐条做 bad case 归因表而不只看平均值？
> 4. 微调后"聋了"（通用能力退化）怎么发现、怎么防？

对应学习计划：第 5-6 周。交付物：云端完成一次 500-1000 条领域数据的 LoRA 微调，输出微调前后 CER 对比、Loss 曲线、5 条典型 bad case 归因。

---

## 1. 第一性原理：为什么微调、微调什么

### 1.1 微调的动机

通用 ASR（Whisper/SenseVoice）在**开放域**鲁棒，但在**垂直域**（医疗术语、方言、特定口音、行业黑话）词表外错误率高。微调用少量领域数据把模型"拽"到目标分布上。

### 1.2 冻结策略的逻辑

全量微调 7B 级声学模型：显存爆炸 + 灾难性遗忘。标准配方是**分层冻结**：

| 组件 | 冻结? | 原因 |
| --- | --- | --- |
| Audio Encoder | ✅ 冻结 | 声学表征已足够通用，动了易遗忘、参数多 |
| Projector / Adapter | ❌ 训练 | 模态对齐层小、最该适配目标域 |
| LLM/Decoder（LoRA） | ❌ LoRA | 低秩适配语言分布，参数效率高 |

这与你在 VLM 阶段学的"冻结视觉塔 + 训 projector + LoRA LLM"完全同构——**多模态微调方法论是通用的**。

### 1.3 LoRA 回顾（一句话）

在权重 $W$ 旁挂低秩增量 $\Delta W = BA$（$A \in \mathbb{R}^{r\times d}$, $B \in \mathbb{R}^{d\times r}$, $r \ll d$），只训 $A, B$。参数量从 $d^2$ 降到 $2rd$，显存与时间大幅下降，且可随时合并或插拔。

---

## 2. 数据准备：从原始音频到训练格式

### 2.1 数据格式

主流框架（`ms-swift`、HF `datasets`）接受的典型格式：

```json
{"audio": "/data/train/0001.wav", "text": "瑞芯微的RK3588有6TOPS算力"}
{"audio": "/data/train/0002.wav", "text": "今天把量化模型部署到昇腾910B"}
```

要点：

- **音频采样率统一**到 16 kHz 单声道（用你的 `AudioPipeline` 预处理一遍）；
- **文本是规范化后的转写**（去标点或与目标一致）；
- **时长过滤**：去掉 < 0.5 s 或 > 30 s 的极端样本（噪声或超长）。

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

## 3. 微调用例：ms-swift 一键流

`ms-swift` 是阿里开源的多模态微调框架，统一了 ASR/VLM/TTS 的微调接口。以 Whisper LoRA 为例：

```bash
# 安装
pip install 'ms-swift[all]'

# LoRA 微调（命令行示例，参数以官方最新文档为准）
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model openai/whisper-small \
    --train_type lora \
    --dataset /path/to/asr_train.jsonl \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --freeze_encoder true \
    --output_dir output/whisper-lora
```

**关键参数解读**：

- `--train_type lora`：启用 LoRA；
- `--freeze_encoder true`：冻结声学编码器（第 1 节策略）；
- `--lora_rank 8`：秩，8-16 常用；
- `--learning_rate 1e-4`：LoRA 学习率通常比全量高一个量级。

### 3.1 训练观察

用 WandB 或框架自带日志盯三条曲线：

1. **训练 Loss**：应平滑下降；若震荡 → 学习率过高；
2. **验证 CER**：真正的目标指标（Loss 降 ≠ CER 降）；
3. **梯度范数**：异常飙升预示发散。

**常见故障**（学习计划第 6 周要求掌握）：

- **GPU OOM**：减 batch、开 gradient checkpointing、用 QLoRA；
- **Loss 不收敛**：学习率过大、数据文本未规范化、音频采样率不一致；
- **CER 先降后升**：过拟合，早停。

---

## 4. 评测闭环：CER 与 bad case 归因

### 4.1 CER 计算

```python
from jiwer import cer

def normalize_zh(s):
    import re
    return re.sub(r"[，。！？、,.!?\s\"'《》<>()\[\]]", "", s).lower()

refs = [normalize_zh(r) for r in ground_truths]
hyps = [normalize_zh(h) for h in predictions]
print(f"CER = {cer(refs, hyps)*100:.2f}%")
```

CER = (替换 + 删除 + 插入) / 参考总字数。三类错误要分开看——它们指向不同的根因。

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
- [ ] 云端跑通一次 LoRA 微调（ms-swift 或官方脚本），记录 Loss 曲线。
- [ ] 输出微调前后 CER 对比表 + 逐条 bad case 归因表（≥5 条典型）。
- [ ] 通用集回归测试，确认无灾难性遗忘。

## Stage 2 总结

完成本阶段，你已能：

1. **推导**：CTC 前向算法（含暴力枚举验证）、1500 帧的来源、CIF 长度损失；
2. **手写**：Whisper 完整推理循环（含 KV cache）、三模型横评脚本；
3. **工程**：流式转写、端点检测、LoRA 微调、bad case 归因。

对照学习计划第 6 周自测清单核验后，进入 **Stage 3：语音增强与前端信号处理**——你在 Stage 1 第 4 周写的谱减法将在那里升级为 DeepFilterNet。
