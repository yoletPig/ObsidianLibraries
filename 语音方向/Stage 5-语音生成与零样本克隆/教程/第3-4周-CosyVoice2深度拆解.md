# 第 3-4 周教程：CosyVoice 2 深度拆解（主线）

> **本周要回答的四个问题**
> 1. CosyVoice 2 的"三件套"是什么？FSQ 语义 token 与 EnCodec 的声学 token 本质区别在哪？
> 2. 因果 Flow Matching（Causal Flow Matching）如何实现流式？首包延迟为什么能压到 150 ms？
> 3. 3 秒参考音频的零样本克隆，说话人信息是怎么注入的？
> 4. 中英跨语言与情感/笑声指令（`[laughter]`）是如何实现的？

对应学习计划：第 3-4 周。交付物：① 云部署 `CosyVoice2-0.5B`，用 10 秒你的声音完成零样本克隆；② 通读源码写出全链路 Shape 变化图；③ 测量流式首包延迟与 RTF。

---

## 1. 第一性原理：工业级流式克隆的三个设计决策

CosyVoice 2 是当前工业级零样本克隆的标杆（阿里通义）。它的架构是"**语义 token + 因果 flow matching**"配方的集大成者。理解它，就理解了当前工业 TTS 的主流设计。

三个核心决策：

1. **用监督语义 token**（FSQ）而非声学 codec token → 内容更干净、更可控；
2. **因果化 flow matching** → 支持流式、低延迟；
3. **大语言模型骨干** → 继承 LLM 的指令跟随与跨语言能力。

---

## 2. 三件套架构

$$
\text{文本} \xrightarrow{\text{LLM (AR)}} \text{语义 token} \xrightarrow{\text{Causal Flow Matching}} \text{梅尔谱} \xrightarrow{\text{声码器}} \text{波形}
$$

### 2.1 件一：监督语义离散 token（FSQ）

**来源**：不是 EnCodec 的自监督声学 codec，而是**从一个预训练的多语言 ASR encoder 提取语义特征**，再用 **FSQ（Finite Scalar Quantization）** 离散化。

**与 EnCodec 的本质区别**（本周自测题）：

| | EnCodec 声学 token | CosyVoice FSQ 语义 token |
| --- | --- | --- |
| 提取方式 | 自监督重建（压缩-重建波形） | 监督（ASR 任务）训练 |
| 承载信息 | 声学重建（内容+音色混合） | 语义/内容为主（音色剥离） |
| 用途 | VALL-E 式声学建模 | 干净的内容表示，利于可控 |

**为什么用语义 token**：ASR encoder 被训练来"识别说了什么"，其特征天然聚焦内容、剥离音色——这正是"内容可控、音色另注入"的理想解耦。

**FSQ 是什么**：有限标量量化——把每个特征维度量化到有限的离散电平（如 {-2,-1,0,1,2}），多维组合成码字。比 VQ 更稳定（无码本坍缩问题）。

### 2.2 件二：自回归 LLM 生成语义 token

把文本（字符/音素）和参考音频的语义 token 拼成序列，用 LLM **自回归**预测目标语义 token：

$$
[\text{参考音频语义 token} \;|\; \text{目标文本}] \;\xrightarrow{\text{AR LLM}}\; \text{目标语义 token}
$$

**说话人信息的注入**（本周自测题）：这里用的是 **prompt token 拼接**方式——参考音频不仅提供语义，其**声学信息通过一个说话人 embedding**（全局音色向量）注入。两者结合：语义 token 给内容示范，说话人 embedding 给音色。

### 2.3 件三：因果 Flow Matching 解码器

把语义 token 转成梅尔谱，用 **Conditional Flow Matching**（连续正则化流），并做了**因果化**：

- **Flow matching**：学习从噪声到梅尔谱的"速度场"，推理时沿速度场积分 ODE 得到梅尔谱；
- **因果化（Causal）**：每个时刻的梅尔谱只依赖当前及过去的语义 token，不看未来 → **可以边生成边解码，支持流式**。

**流式首包延迟 ~150 ms** 的来源：因果设计让模型在收到第一个语义 token chunk 后，就能立刻开始生成对应梅尔谱并出声，无需等整句生成完。

---

## 3. 流式机制深拆

### 3.1 Chunk-aware 因果流

CosyVoice 2 把生成按 **chunk**（如每 25 个语义 token 一块）推进：

```
收到 chunk 1 语义 token -> 生成 chunk 1 梅尔谱 -> 声码器出声
收到 chunk 2 语义 token -> 生成 chunk 2 梅尔谱 -> 声码器出声
...
```

每个 chunk 内部因果（只看过去），chunk 间连续。这样**边听边说**，首包延迟由第一个 chunk 的生成时间决定（~150 ms）。

### 3.2 有限状态 Transducer（FST）纠错

LM 生成的语义 token 可能有错（漏字、错字）。CosyVoice 用 **FST（有限状态转换器）** 做解码纠错——约束生成的语义 token 序列符合合法路径，抑制幻觉。

---

## 4. 跨语言与指令控制

### 4.1 中英跨语言

训练数据含大量中英混合，模型学会：
- 给定中文文本输出中文语音；
- 跨语言：中文参考音频 + 英文文本 → 用该音色说英文。

### 4.2 指令跟随（Instruct 版本）

CosyVoice-Instruct 支持自然语言指令控制：

```
"[laughter]哈哈太好笑了[laughter]"     -> 带笑声
"用四川话说这句话"                     -> 方言
"伤心地说"                            -> 情感
```

实现：训练数据里标注了这些指令-语音对，LLM 骨干继承了指令跟随能力。

---

## 5. 实现与验证（交付核心）

### 5.1 部署与零样本克隆

```bash
# 按官方仓库安装（以 FunAudioLLM/CosyVoice 最新文档为准）
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice && pip install -r requirements.txt
```

```python
from cosyvoice.cli.cosyvoice import CosyVoice2

model = CosyVoice2('CosyVoice2-0.5B')

# 零样本克隆：10 秒参考音频 -> 任意文本
prompt_wav = "my_10s_voice.wav"          # 录 10 秒你的声音
prompt_text = "这是参考音频里说的话。"      # 参考音频的转写

# 流式合成
for chunk in model.inference_zero_shot(
        "大家好，这是我用CosyVoice克隆的声音，听起来像不像我？",
        prompt_text, prompt_wav, stream=True):
    # chunk['tts_speech'] 是音频张量
    play_or_save(chunk['tts_speech'])     # 边生成边播
```

**交付**：用你的 10 秒声音克隆出一段任意文本，试听确认音色相似度。

### 5.2 全链路 Shape 追踪（沿用你的维度方法论）

通读源码（`cosyvoice/llm/llm.py` 与 flow matching 解码器），画出：

```
文本 "你好"
  -> 音素/字符 token: (L_text,)
  -> 参考音频语义 token: (L_ref,)
  -> LLM 输入序列: (L_ref + L_text,)
  -> AR 生成目标语义 token: (L_target,)
  -> Causal Flow Matching: (L_target,) -> 梅尔谱 (80, T_mel)
  -> 声码器: (80, T_mel) -> 波形 (N,)
```

**标注每一步的维度变化**——这是你 VLM 阶段练过的维度追踪法，迁移到语音生成。

### 5.3 测量首包延迟与 RTF

```python
import time

t0 = time.time()
first_chunk_time = None
total_audio = 0
gen_start = time.time()

for chunk in model.inference_zero_shot(text, prompt_text, prompt_wav, stream=True):
    if first_chunk_time is None:
        first_chunk_time = time.time() - gen_start
    total_audio += chunk['tts_speech'].shape[-1]

total_time = time.time() - t0
audio_dur = total_audio / model.sample_rate
print(f"首包延迟: {first_chunk_time*1000:.0f} ms")
print(f"RTF: {total_time / audio_dur:.3f}")
```

**预期**：首包延迟 ~100-300 ms（流式生效）；RTF < 1（GPU 上）。

---

## 6. 工程权衡与失效模式

### 6.1 权衡

- **流式因果性**：因果（流式）质量略低于非因果（离线看全句），但换来低延迟——实时交互必须因果；
- **语义 token 粒度**：粒度细表达强但序列长、算力高；粒度粗反之。25 Hz 是 CosyVoice 的甜点；
- **模型大小**：0.5B 端侧/云两用；更大模型音质更好但更重。

### 6.2 失效模式

1. **音色不像**：参考音频太短/太嘈杂。修复：用 5-10 秒干净干声、提升说话人 embedding 权重。
2. **流式首包慢**：非流式调用或 chunk 太大。修复：确认 `stream=True`、调小 chunk。
3. **内容错漏**：LM 生成语义 token 出错。修复：FST 纠错、温度调优。
4. **指令不生效**：用了非 Instruct 版本。修复：换 Instruct 模型或微调。

---

## 7. 延伸思考题（含解析）

**Q1**：CosyVoice 的 FSQ 语义 token 与 VALL-E 的 EnCodec token 本质区别？
**A**：EnCodec 是自监督声学重建导向，内容+音色混合；FSQ 语义 token 从监督训练的 ASR encoder 提取，聚焦内容、剥离音色。前者适合声学建模，后者内容更干净、更可控。

**Q2**：为什么 CosyVoice 选"语义 token + flow"两段式？
**A**：语义 token 干净表达内容，利于对齐与可控；因果 flow matching 生成声学细节，支持流式低延迟且质量稳。两段各司其职，是当前工业最优配方。

**Q3**：3 秒参考音频怎么让模型记住音色？
**A**：两条路：① 参考音频的语义 token 作为 prompt 拼进序列（in-context）；② 提取全局说话人 embedding 注入。前者示范内容风格，后者锁定音色。

**Q4**：CosyVoice 2 首包延迟为什么能压到 ~150 ms？
**A**：因果 flow matching + chunk-aware 流式：收到第一个语义 token chunk 就立即解码出声，不等整句。首包延迟 ≈ 第一个 chunk 的生成时间。

**Q5**：FST 纠错在防什么？
**A**：防 LM 生成语义 token 的错漏（幻觉、跳字）。FST 约束生成走合法路径，抑制不合理的序列，提升鲁棒性。

---

## 本周交付清单

- [ ] 云部署 CosyVoice2-0.5B，用你的 10 秒声音完成零样本克隆。
- [ ] 通读源码，画出全链路 Shape 变化图（文本 → 语义 token → 梅尔谱 → 波形）。
- [ ] 测流式首包延迟（预期 ~100-300 ms）与 RTF。
- [ ] 实验中英跨语言与（若用 Instruct）情感指令。
- [ ] 能解释：FSQ 语义 token、因果 flow matching 流式、说话人信息注入。
