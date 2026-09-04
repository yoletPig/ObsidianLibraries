# 第 3 周教程：SenseVoice 与 Paraformer——国产主力与 FunASR

> **本周要回答的四个问题**
> 1. CIF 的"积分发射"在代码里长什么样？长度预测如何参与训练？
> 2. SenseVoice 为什么能用 Whisper-large 1/15 的时间跑出结果，还附带情感与事件检测？
> 3. FunASR 的热词、标点恢复、时间戳是怎么挂上去的？
> 4. 三模型横评（SenseVoice / Paraformer / Whisper）怎么设计才公平？

对应学习计划：第 3 周。交付物：三模型在同一组测试音频上跑 CER 对比，输出准确率/RTF/显存三方表格与端侧选型结论。

---

## 1. 第一性原理：非自回归的两个支柱

上周讲了 CIF 的直觉（积分发射预测字数）。本周深入两个工程支柱：

1. **CIF 的训练监督**：开火次数必须等于真实字数 $U$，否则模型学不会"数数"。CIF 损失强制累积权重之和收敛到 $U$：

$$
\mathcal{L}_{\text{CIF}} = \left(\sum_{t=1}^{T} w_t - U\right)^2
$$

同时用"边界对齐"的交叉熵监督每个开火位置的输出字符。两个损失联合训练，让"长度预测"与"内容识别"同时收敛。

2. **声学特征到字符的直接映射**：NAR 模型没有自回归语言建模兜底，因此更依赖**强声学编码**（Paraformer 用 SAN-M 自注意力 + 大词表中文子词）。这也是为什么 NAR 模型在**中文**这种字粒度语言上特别能打——每个字的信息量大、边界清晰。

---

## 2. Paraformer 架构细节

### 2.1 结构组成

$$
\text{Fbank}(80) \xrightarrow{\text{Encoder (SAN-M)}} \xrightarrow{\text{Predictor (CIF)}} \text{开火位置} \xrightarrow{\text{Glancing 采样}} \xrightarrow{\text{Decoder}} \text{字序列}
$$

- **SAN-M Encoder**：Self-Attention with Non-autoregressive Memory，在标准自注意力外加入对历史上下文记忆的访问，增强长程建模；
- **Predictor**：两层 FFN 输出每帧发射权重 $w_t$，CIF 积分定位开火点；
- **Glancing Language Model 采样**：训练时随机把部分真实标签混入解码器输入（GLM 风格），弥合"训练看真实标签、推理看自己预测"的暴露偏差——这是 NAR 质量逼近 AR 的关键技巧。

### 2.2 与 CTC 的关系

Paraformer 可以看作"**连续化的 CTC**"：CTC 每帧硬选一个符号（离散对齐），CIF 允许"多帧攒一个字"（连续对齐）。前者受帧级条件独立限制，后者通过积分获得跨帧聚合能力。

---

## 3. SenseVoice：语音理解的多任务统一

### 3.1 设计定位

SenseVoice 不是"更快的 ASR"，而是**语音理解大模型**：一个模型同时输出

- 转写文字（中/英/日/韩/粤）；
- 情感标签（`<|HAPPY|>`、`<|SAD|>`…）；
- 音频事件（`<|BGM|>`、`<|Laughter|>`、`<|Applause|>`…）。

输出示例：

```
<|zh|><|NEUTRAL|><|Speech|>今天天气不错，我们出去走走吧。
```

### 3.2 为什么快 15 倍

三个因素叠加：

1. **非自回归主干**：一步并行解码（对比 Whisper 的逐字生成）；
2. **更短的声学下采样**：更高的压缩率减少序列长度；
3. **编码器-解码器规模精算**：SenseVoice-Small 的 Encoder 针对推理效率设计，无冗余层。

官方数据：SenseVoice-Small 在 A10 上 RTF 约 0.015（即处理 1 秒音频只要 15 毫秒），比 Whisper-large 快一个数量级以上。

### 3.3 数据配方

训练数据覆盖 40 万小时以上多语言音频，其中情感与事件标签由弱监督（分类模型预标注 + 清洗）获得——与你 VLM 阶段学过的"合成数据 + 执行验证"方法论同构。

---

## 4. FunASR 实战（交付核心）

### 4.1 安装与模型加载

```bash
pip install -U funasr modelscope
```

```python
from funasr import AutoModel

# 三个模型一次加载（内存紧张就分开跑）
model_paraformer = AutoModel(
    model="paraformer-zh",            # 非自回归主力
    vad_model="fsmn-vad",             # 前端 VAD
    punc_model="ct-punc",             # 标点恢复
)
model_sensevoice = AutoModel(model="iic/SenseVoiceSmall")
```

**注意**：`paraformer-zh` 的标准部署形态是 **VAD + ASR + 标点** 三件套级联——VAD 切段、ASR 转写、标点模型补标点。这本身就是"工业流水线"的示范：每个组件小而专，级联出完整能力。

### 4.2 三模型横评脚本

```python
import time
import numpy as np
import soundfile as sf
from funasr import AutoModel
from jiwer import cer

# ---- 1. 准备测试集：20 条中文 + 中英混读，带人工标注文本 ----
# 目录约定：test_set/{name}.wav + test_set/{name}.txt（正确转写）
import os, glob
pairs = []
for wav in sorted(glob.glob("test_set/*.wav"))[:20]:
    with open(wav.replace(".wav", ".txt")) as f:
        pairs.append((wav, f.read().strip()))
assert len(pairs) == 20, "请准备 20 条测试音频与对应文本"

def normalize_zh(s: str) -> str:
    """去标点、去空格、转小写——CER 前的标准清洗。"""
    import re
    s = re.sub(r"[，。！？、,.!?\s\"'《》<>()\[\]]", "", s)
    return s.lower()

def eval_model(model, pairs, use_hyp_key="text"):
    hyps, refs, rtf_list = [], [], []
    for wav, ref in pairs:
        t0 = time.time()
        res = model.generate(input=wav)
        dt = time.time() - t0
        audio_dur = len(sf.read(wav)[0]) / 16000
        rtf_list.append(dt / audio_dur)
        hyp = res[0][use_hyp_key] if isinstance(res[0], dict) else res[0]["text"]
        hyps.append(normalize_zh(hyp))
        refs.append(normalize_zh(ref))
    return cer(refs, hyps), float(np.mean(rtf_list))

results = {}
results["paraformer-zh"] = eval_model(model_paraformer, pairs)
results["SenseVoice-Small"] = eval_model(model_sensevoice, pairs)
print(f"{'模型':<20}{'CER':>8}{'RTF':>8}")
for name, (c, r) in results.items():
    print(f"{name:<20}{c*100:>7.2f}%{r:>8.3f}")
```

### 4.3 公平评测的四个要点

1. **文本归一化必须一致**：中文去标点去空格后再算 CER，否则标点模型会"作弊"或"吃亏"；
2. **RTF 统一硬件**：三个模型在同一台机器、同一精度下测；
3. **测试集平衡**：普通话、口音、中英混读、噪声各占比例，别全是干净朗读；
4. **CER 而非 WER**：中文按字错误率（character），`jiwer.cer` 直接支持。

**预期**：干净中文上三者 CER 都在 3-8% 区间，但 RTF 差异巨大——SenseVoice/Paraformer 约比 Whisper-large 快 10-20 倍。**把显存占用（`nvidia-smi` 或内存峰值）一并记录**，凑齐三维表。

### 4.4 热词（Hotword）机制

```python
res = model_paraformer.generate(
    input="test_set/sample.wav",
    hotword="瑞芯微 昇腾 量化感知训练")
```

热词通过**上下文偏置（contextual biasing）**注入解码：把热词编码成向量，在解码时提升匹配路径的分数。适用场景：专有名词、产品名、人名——这些是通用模型词表外的重灾区，也是面试常问的"领域适配手段"。

### 4.5 时间戳获取

```python
model_ts = AutoModel(model="paraformer-zh")
res = model_ts.generate(input="test_set/sample.wav",
                        output_dir=None, return_raw_text=False)
print(res[0]["timestamp"])   # [[start_ms, end_ms], ...] 每个字的时间区间
```

时间戳是第 4 周流式实验与第 5 周"句子级对齐"（分离方向预告）的基础数据。

---

## 5. 工程权衡与失效模式

### 5.1 权衡

- **级联三件套 vs 单模型**：VAD+ASR+标点的级联灵活可替换（换标点模型不动 ASR），但误差级联、部署组件多；
- **热词数量**：热词越多解码搜索空间越大，延迟上升；生产上通常限制在几十到几百词；
- **NAR 的长句风险**：句子越长，CIF 长度累积误差越大——所以工业部署一定先过 VAD 切短句。

### 5.2 失效模式

1. **中英混读语言误切**：SenseVoice 的语言标签打错（如把中文段标成 `<|en|>`）。定位：看输出的语言前缀；修复：显式指定 `language="zh"`。
2. **热词反噬**：热词权重过高导致"强行凑热词"，把普通词错纠成热词。修复：控制热词数量与置信度阈值。
3. **VAD 切分切词**：VAD 在词语中间误断句，造成半词识别。修复：调整 VAD 的端点延迟参数。
4. **标点模型幻觉**：在专有名词、数字串附近乱加标点。属于 ct-punc 已知弱点，评估时按场景分层统计。

---

## 6. 延伸思考题（含解析）

**Q1**：CIF 损失为什么是 $(\sum w_t - U)^2$ 而不是直接回归字数？
**A**：$\sum w_t$ 是可微的"软字数"，与逐帧特征绑定，能反传梯度到 Encoder；直接回归整数长度不可微且脱离帧级监督。软计数是连续化技巧的标准做法。

**Q2**：SenseVoice 的情感/事件标签对下游产品有什么用？举两个场景。
**A**：① 客服质检：自动标记愤怒来电优先人工接管；② 语音助手：根据用户情绪调整回复语气。多任务输出让一个模型覆盖原本需要独立分类器的能力。

**Q3**：为什么工业 ASR 部署普遍是 "VAD + ASR + 标点" 级联而不是端到端单模型？
**A**：① 各组件可独立升级替换；② VAD 前置过滤静音省算力；③ 短段识别精度与稳定性都优于长音频端到端；④ 单点故障影响面小。代价是链路复杂度与误差传递。

**Q4**：CER 评测前为什么要做文本归一化？不归一化会怎样？
**A**：标点、空格、大小写不是"语音识别"的正确性范畴。不归一化会把标点差异算成错误，系统性抬高 CER，且让带标点模型与不带标点模型不可比。

**Q5**：你的端侧选型结论是什么？依据哪三个数字？
**A**：选 SenseVoice-Small（或 Paraformer-Small）。依据：① RTF（端侧算力下能否实时）；② 中文测试集 CER（质量底线）；③ 内存/显存占用（板端硬约束）。三者构成端侧选型的铁三角。

---

## 本周交付清单

- [ ] 跑通 `paraformer-zh`（含 VAD+标点三件套）与 `SenseVoice-Small`。
- [ ] 完成 20 条测试集的三模型横评（含 Whisper 上周的推理脚本），输出 CER/RTF/显存表。
- [ ] 实验热词功能：加/不加热词对比 3 条含专有名词的音频。
- [ ] 写出端侧选型结论（引用你的实测数字）。
