# 第 3-4 周教程：CosyVoice 2 深度拆解（主线）

> **本周要回答的四个问题**
> 1. CosyVoice 2 的"三件套"是什么？FSQ 语义 token 与 EnCodec 的声学 token 本质区别在哪？
> 2. 因果 Flow Matching（Causal Flow Matching）如何实现流式？首包延迟为什么能压到 150 ms？
> 3. 3 秒参考音频的零样本克隆，说话人信息是怎么注入的？
> 4. 中英跨语言与情感/笑声指令（`[laughter]`）是如何实现的？

对应学习计划：第 3-4 周。交付物：① 云部署 `CosyVoice2-0.5B`，用 10 秒你的声音完成零样本克隆；② 通读源码写出全链路 Shape 变化图；③ 测量流式首包延迟与 RTF。

---

## 1. 第一性原理：工业级流式克隆的三个设计决策

CosyVoice 2 是当前工业级零样本克隆的标杆（阿里通义 FunAudioLLM 团队，2024 年 12 月，论文 arXiv:2412.10117，开源模型 `CosyVoice2-0.5B`）。它的架构是"**语义 token + 因果 flow matching**"配方的集大成者。理解它，就理解了当前工业 TTS 的主流设计。

三个核心决策：

1. **用监督语义 token**（FSQ）而非声学 codec token → 内容更干净、更可控；
2. **因果化 flow matching** → 支持流式、低延迟；
3. **大语言模型骨干** → 继承 LLM 的指令跟随与跨语言能力。

---

## 2. 三件套架构

$$
\text{文本} \xrightarrow{\text{LLM (AR)}} \text{语义 token (25 Hz)} \xrightarrow{\text{Causal Flow Matching}} \text{梅尔谱 (50 Hz, 24 kHz)} \xrightarrow{\text{HiFT 声码器}} \text{波形}
$$

### 2.1 件一：监督语义离散 token（FSQ）

**来源**：不是 EnCodec 的自监督声学 codec，而是把 **FSQ（Finite Scalar Quantization，有限标量量化）模块插入一个多语言 ASR 模型（SenseVoice-Large 系）的 encoder** 里训练得到——监督信号是"识别说了什么"，约 20 万小时中英文数据。**具体参数**：codebook 大小 **6561**（=$18^3$，3 个维度、每维 18 个电平），token 帧率 **25 Hz**（每秒 25 个语义 token）。前代 CosyVoice 1 用的是 VQ（codebook 4096），实际码字利用率只有约 23%，CosyVoice 2 换 FSQ 后利用率 100%——这就是换量的动机。

**与 EnCodec 的本质区别**（本周自测题）：

| | EnCodec 声学 token | CosyVoice FSQ 语义 token |
| --- | --- | --- |
| 训练信号 | 自监督重建（压缩-重建波形） | 监督（ASR 识别任务） |
| 承载信息 | 声学重建（内容+音色混合） | 语义/内容为主（音色剥离） |
| 量化方式 | 残差向量量化 RVQ（多层套娃） | 有限标量量化 FSQ（单层、多维电平组合） |
| 帧率 | 75 Hz | 25 Hz |
| 用途 | VALL-E 式声学建模 | 干净的内容表示，利于可控 |

**为什么用语义 token**：ASR encoder 被训练来"识别说了什么"，其特征天然聚焦内容、剥离音色——这正是"内容可控、音色另注入"的理想解耦。

**FSQ 是什么**：把每个特征维度独立量化到一组固定的离散电平（如 18 个电平），多维电平组合成一个码字（$18 \times 18 \times 18 = 6561$）。**与 VQ 的区别**：VQ 是"在一本 4096 页的字典里查最近条目"，字典太大训练时容易"翻烂前几页"（码本坍缩，很多码字永远用不到）；FSQ 没有"字典"这个实体——每个维度的电平是固定的、等距的，**天然每个码字都可能被用到**（利用率 100%），训练更稳。**手算小例**：3 维特征 $(1.2, -3.7, 0.4)$，每维就近吸附到 18 电平网格，比如量化成 $(1, -4, 0)$——三个数字组合就是码字 $(1,-4,0)$，不需要查任何码本。

### 2.2 件二：自回归 LLM 生成语义 token

CosyVoice 2 直接用预训练 LLM **Qwen2.5-0.5B** 做骨干：把文本（字符 token）和参考音频的语义 token 按固定比例交错拼成序列，自回归地预测目标语义 token：

$$
[\text{参考音频语义 token} \;|\; \text{目标文本}] \;\xrightarrow{\text{AR LLM}}\; \text{目标语义 token}
$$

**为什么用现成 LLM**：继承它的文本理解、指令跟随与上下文学习能力——这是"音色 in-context 学习"能工作的基础。

**说话人信息的注入**（本周自测题）：两条路一起用——① 参考音频的**语义 token 作为 speech prompt** 拼进序列（in-context，示范内容与韵律风格）；② 从参考音频提取**全局说话人 embedding**（基于 3D-Speaker 的 CAM++ 说话人验证模型）注入 flow matching 模型，锁定音色。语义 token 负责"说成什么样"，说话人 embedding 负责"用谁的嗓子说"。

### 2.3 件三：因果 Flow Matching 解码器

把语义 token 转成梅尔谱（50 Hz、24 kHz），用 **Conditional Flow Matching（条件流匹配，CFM）**，并做了**因果化**。先把 flow matching 本身讲透：

**flow matching 在学什么**：定义一条从"纯噪声 $\epsilon$"到"目标梅尔谱 $x_0$"的连续路径。最常用的是**直线（OT）路径**：

$$
x_t = (1-t)\,\epsilon + t\,x_0, \quad t \in [0, 1]
$$

$$
v_t = \frac{d x_t}{d t} = x_0 - \epsilon
$$

**这两个公式在干什么**：第一个公式把"生成"变成"走直线"——$t=0$ 时 $x_0$ 全是噪声，$t=1$ 时全是有声梅尔谱，中间是两者的线性混合（像调音台从"白噪声"推子慢慢推到"语音"推子）。第二个公式是这条直线的**速度**：恒定不变，方向由"噪声指向目标"。模型的任务就一个：**给定当前位置 $x_t$ 和时间 $t$，预测速度 $v$**（训练目标是 $x_0 - \epsilon$，即 MSE 回归，比扩散模型的预测目标更直白）。推理时解一个常微分方程（ODE）：$\frac{dx}{dt} = v_\theta(x_t, t)$，从噪声出发沿预测速度积分到 $t=1$，就得到梅尔谱。

**手算小例**（一维）：目标 $x_0 = 2$，噪声 $\epsilon = 10$。直线路径：$x_t = 10 - 8t$，速度恒为 $-8$。若模型在 $t=0$ 时就"看准"了速度 $-8$，一步积分就能到终点——**这就是 flow matching 路径直、少步数推理快的直觉来源**（对比扩散模型：预测的是"逐步去一点点噪声"，路径弯绕，需要更多步）。

**因果化（Causal）**：流匹配网络用因果卷积 + Transformer UNet 实现，每个时刻的梅尔谱只依赖当前及过去的语义 token（部分配置允许看有限帧未来）→ **可以边生成边解码，支持流式**。同一模型支持流式与非流式两种模式。

**声码器**：仓库使用 **HiFT 声码器**（HiFi-GAN 风格的生成器 + iSTFT 思路的混合结构）从梅尔谱恢复相位出波形。

**流式首包延迟 ~150 ms** 的来源（论文报告值）：因果设计让模型在收到第一个语义 token chunk 后，就能立刻开始生成对应梅尔谱并出声，无需等整句生成完。论文给出了延迟上界公式：

$$
L_{\text{TTS}} = M \cdot (d_{\text{lm}} + d_{\text{fm}} + d_{\text{voc}})
$$

其中 $M$ 是流式块数，$d_{\text{lm}} / d_{\text{fm}} / d_{\text{voc}}$ 分别是 LLM、流匹配、声码器处理一个块的耗时——**首包延迟 ≈ 一个块的"三段处理时间"**，块越小延迟越低（但质量略降），这就是"chunk 大小 = 延迟-质量折衷"的定量表达。

---

## 3. 流式机制深拆

### 3.1 Chunk-aware 因果流

CosyVoice 2 支持文本**流式输入**：文本按小块送进来，LLM 每收到一小块文本就生成一小块语义 token，flow matching 再把每个语义 token 块变成梅尔谱块，声码器随即出声：

```
收到文本块 1 -> LLM 生成语义 token 块 1 -> FM 生成梅尔谱块 1 -> 声码器出声
收到文本块 2 -> LLM 生成语义 token 块 2 -> FM 生成梅尔谱块 2 -> 声码器出声
...
```

**两个关键细节**：
- 语义 token 帧率 25 Hz、梅尔谱帧率 50 Hz，flow matching 前会先把 token **上采样 2 倍**（带前瞻卷积）对齐帧率；
- "chunk-aware" 的含义：训练时给流匹配模型喂了**多种因果配置**（非因果=离线、全因果=最低延迟、chunk 式=折衷），推理时可按延迟需求选——同一个模型既可流式也可非流式，不用训两版。

**块大小的权衡**：块越小首包越快，但单块上下文少、边界音质略差；块越大反之。这是延迟与质量的连续谱，工程上按场景选。

### 3.2 文本前端与一个常见误解

**纠个错**：CosyVoice 2 的论文与开源实现里，**没有**"用有限状态转换器（FST）纠正 LLM 生成的语义 token"这一机制——早期教程流传过这个说法，属张冠李戴。实际的两个相关机制是：

1. **文本正则化（TN, Text Normalization）**：文本前端把"12345.67 元"转成"一万两千三百四十五点六七元"这类读法、处理多音字与符号。TN 在工业界常用规则系统实现，FST（有限状态转换器）是这类规则系统的经典载体——FST 的作用位置在**文本侧**，不在 token 侧；
2. **LM 生成语义 token 的稳健性**：主要靠大规模训练 + 解码时的温度采样、重复惩罚等常规手段控制，而不是外挂的序列约束器。

---

## 4. 跨语言与指令控制

### 4.1 中英跨语言

训练数据含大量中英混合，模型学会：
- 给定中文文本输出中文语音；
- 跨语言：中文参考音频 + 英文文本 → 用该音色说英文。

### 4.2 指令跟随（Instruct 版本）

CosyVoice 2 提供两种控制接口（对应仓库两个推理方法）：

1. **自然语言指令**（`inference_instruct2`，用 CosyVoice2-Instruct 模型）：指令后用 `<|endofprompt|>` 与正文分隔：

```
"用四川话说这句话<|endofprompt|>今天天气真不错。"   -> 方言
"用伤心的语气说<|endofprompt|>我没事，真的。"      -> 情感
```

2. **细粒度标签**（`inference_cross_lingual` 等接口）：在文本里直接嵌标签：

```
"他突然[laughter]停下来。"     -> 带笑声
"这个<strong>非常</strong>重要" -> 重音强调
```

实现：训练数据里标注了这些指令/标签-语音对，LLM 骨干继承了指令跟随能力。注意 CosyVoice 1 的 Instruct 模型（`inference_instruct`）与 CosyVoice 2 的 `inference_instruct2` 是两代接口，别混用。

---

## 5. 实现与验证（交付核心）

### 5.1 部署与零样本克隆

```bash
# 按官方仓库安装（以 FunAudioLLM/CosyVoice 最新文档为准）
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice && pip install -r requirements.txt

# 下载模型（ModelScope 国内 / HuggingFace 海外）：iic/CosyVoice2-0.5B 或 FunAudioLLM/CosyVoice2-0.5B
# 放到 pretrained_models/CosyVoice2-0.5B
```

```python
from cosyvoice.cli.cosyvoice import CosyVoice2
import torchaudio

model = CosyVoice2('pretrained_models/CosyVoice2-0.5B')   # 参数是模型目录

# 零样本克隆：参考音频（要求 16 kHz 采样率的 prompt 语音）-> 任意文本
prompt_wav = "my_10s_voice_16k.wav"        # 录 10 秒你的声音并重采样到 16kHz
prompt_text = "这是参考音频里说的话。"        # 参考音频的转写

# 流式合成：返回生成器，逐块产出 dict
for chunk in model.inference_zero_shot(
        "大家好，这是我用CosyVoice克隆的声音，听起来像不像我？",
        prompt_text, prompt_wav, stream=True):
    # chunk['tts_speech'] 是音频张量（输出采样率见 model.sample_rate，CosyVoice2 为 24 kHz）
    torchaudio.save("clone_stream.wav", chunk['tts_speech'],
                    model.sample_rate, encoding="PCM_S", bits_per_sample=16)
```

**交付**：用你的 10 秒声音克隆出一段任意文本，试听确认音色相似度。

### 5.2 全链路 Shape 追踪（沿用你的维度方法论）

通读源码（`cosyvoice/llm/llm.py` 与 flow matching 解码器），画出：

```
文本 "你好"
  -> 字符 token: (L_text,)
  -> 参考音频语义 token: (L_ref,)        （speech prompt，随文本块交错拼入）
  -> 说话人 embedding: (D_spk,)          （CAM++ 提取，注入 flow matching，不进 LLM 序列）
  -> AR LLM 生成目标语义 token: (L_target,)
  -> 上采样 2 倍（带前瞻卷积）: (2·L_target,)   （25 Hz token -> 50 Hz 梅尔帧）
  -> Causal Flow Matching: -> 梅尔谱 (80, 2·L_target)
  -> HiFT 声码器: (80, 2·L_target) -> 波形 (N,)
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

1. **音色不像**：参考音频太短/太嘈杂。修复：用 5-10 秒干净干声；也可用 `add_zero_shot_spk` 把参考音频注册成固定音色复用。
2. **流式首包慢**：非流式调用或 chunk 太大。修复：确认 `stream=True`、文本传入改成生成器（分块送文本）、调小 chunk。
3. **内容错漏**：LM 生成语义 token 出错（漏字、复读）。修复：调低温度、加重复惩罚、检查文本前端（TN）是否正常。
4. **指令不生效**：用了基础版模型（不支持 instruct2）。修复：换 CosyVoice2-Instruct 模型或微调。

---

## 7. 延伸思考题（含解析）

**Q1**：CosyVoice 的 FSQ 语义 token 与 VALL-E 的 EnCodec token 本质区别？
**A**：EnCodec 是自监督声学重建导向（RVQ 多层、75 Hz），内容+音色混合；FSQ 语义 token 从监督训练的 ASR encoder（SenseVoice-Large 系）提取（单层、25 Hz、codebook 6561），聚焦内容、剥离音色。前者适合声学建模，后者内容更干净、序列更短、更可控。

**Q2**：为什么 CosyVoice 选"语义 token + flow"两段式？
**A**：语义 token 干净表达内容，利于对齐与可控；因果 flow matching 生成声学细节，支持流式低延迟且质量稳。两段各司其职，是当前工业主流配方。

**Q3**：3 秒参考音频怎么让模型记住音色？
**A**：两条路：① 参考音频的语义 token 作为 speech prompt 拼进序列（in-context）；② 用说话人验证模型（CAM++）提取全局说话人 embedding 注入 flow matching。前者示范内容风格，后者锁定音色。

**Q4**：CosyVoice 2 首包延迟为什么能压到 ~150 ms？
**A**：因果 + chunk-aware 的 flow matching：收到第一个语义 token 块就立即解码出声，不等整句。首包延迟 ≈ 第一个块的"LLM + 流匹配 + 声码器"三段处理时间（见 2.3 节延迟公式），块足够小、每段够快，就能压进 150 ms。

**Q5**：FST 在语音合成链路里的正确位置是哪里？
**A**：在**文本前端**，不在语义 token 侧。FST（有限状态转换器）是文本正则化（TN）规则系统的经典实现载体——把数字、符号、单位转成读法。它不参与"纠正 LLM 生成的语义 token"（那是早期教程的讹误）；LM 侧的稳健性靠训练与解码策略（温度、重复惩罚）保障。

---

## 本周交付清单

- [ ] 云部署 CosyVoice2-0.5B，用你的 10 秒声音完成零样本克隆。
- [ ] 通读源码，画出全链路 Shape 变化图（文本 → 语义 token → 梅尔谱 → 波形）。
- [ ] 测流式首包延迟（预期 ~100-300 ms）与 RTF。
- [ ] 实验中英跨语言与（若用 Instruct）情感指令。
- [ ] 能解释：FSQ 语义 token、因果 flow matching 流式、说话人信息注入。
