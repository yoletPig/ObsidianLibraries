# 第 2 周教程：Qwen2.5-Omni 深度拆解——Thinker-Talker 与 TMRoPE

> **本周要回答的四个问题**
> 1. Thinker-Talker 为什么要拆成两个模型？比单模型 Speech-to-Speech 稳在哪？
> 2. TMRoPE 解决了什么问题？它与你学过的 Qwen2-VL M-RoPE（2D-RoPE）是什么继承关系？
> 3. 全模态流式（边听边看边说）的双轨流水线怎么排布，延迟由哪一段决定？
> 4. 一个模型同时干 ASR、语音问答、TTS 三件事，训练上如何不打架？

对应学习计划：第 2 周。交付物：① 云部署 `Qwen2.5-Omni-7B`，跑通三种交互模式；② 用维度追踪法画出音频输入从 encoder 到 Thinker 的 Token 流全图。

参考文献：*Qwen2.5-Omni Technical Report*（arXiv:2503.20215）。

---

## 1. 第一性原理：全模态交互的两个矛盾

### 1.1 矛盾一：理解要深，响应要快

理解任务（视频+音频联合推理）需要大模型、深思考；实时对话要求首包延迟在数百毫秒内。单一大自回归模型同时背两个指标，必然顾此失彼。

### 1.2 矛盾二：语音输出既要「说得对」又要「说得像人」

文本侧的正确性由 LLM 的语言能力保证；语音侧的韵律、情感、音色是**连续信号属性**，无法用文字 token 表达。若强行用一个词表统一的模型同时生成文字和语音，两套目标会在同一组参数上互相拉扯。

**Qwen2.5-Omni 的答案**：结构上分工（Thinker/Talker），位置编码上对齐（TMRoPE），流水线上并行（双轨流式）。

---

## 2. Thinker-Talker 架构

### 2.1 分工

**这个流程图在干什么**：它描述一条「先想后说」的流水线。所有输入先被各模态编码器变成 token 拼进一条序列，交给 Thinker 处理；Thinker 一边输出文本（给人看、可审计），一边把内部隐藏状态喂给 Talker（只管「怎么说」）；Talker 自回归地生成语音 token，最后由声码器还原成波形。

$$
\text{音频+视频+文本} \xrightarrow{\text{编码器组}} \text{Thinker（大 LM）} \to
\begin{cases}
\text{文本 token} & \text{（可审计、可控）} \\
\text{隐藏状态} & \xrightarrow{\text{Talker（小 AR 模型）}} \text{语音 token} \xrightarrow{\text{流式声码器}} \text{波形}
\end{cases}
$$

- **Thinker**：多模态理解与文本生成主体，继承 Qwen2.5 LM 骨干，负责「想清楚」；
- **Talker**：较小的自回归模型（双轨设计：同时消费自己的文本轨和语音轨），**直接接收 Thinker 的隐藏状态**（而不是文本 token！）生成语音 token（约 12.5 Hz，即每个 token 约 80 ms）；
- **流式声码器**：用 Flow-Matching DiT 把语音 token 变成梅尔谱，再用改良的 BigVGAN 实时合成 **24 kHz** 波形。

### 2.2 为什么 Talker 吃隐藏状态而不是文本

这是架构里最妙的一笔：

1. **韵律不丢失**：文本是离散符号，「开心地说」这类信息一旦坍缩成文字就没了。隐藏状态连续地携带情感、语气、节奏线索，Talker 直接消费，语音才「像人」；
2. **延迟更低**：不必等文本 token 完整解码再转语音，Talker 与 Thinker 并行推进（第 4 节双轨流式）；
3. **分工清晰**：Thinker 保证内容正确（可审计、可接安全过滤），Talker 只管声学表现——两个目标不再在同一组参数上打架。

### 2.3 为什么比单模型 Speech-to-Speech 稳（本周交付核心论点）

| 维度 | 单模型端到端（如 Moshi） | Thinker-Talker |
| --- | --- | --- |
| 内容正确性 | 语音 token 直接承载语义，错了无法中途审计 | Thinker 先出文本，可审计可干预 |
| 训练数据 | 需要海量「语音问-语音答」对，极稀缺 | 理解侧复用文本 SFT，生成侧只需语音对齐数据 |
| 失败归因 | 黑盒，不知道是理解错还是发声错 | 文本对而语音错 → Talker 问题；文本就错 → Thinker 问题 |
| 能力继承 | 重训即失去文本能力 | 直接继承 Qwen2.5 全部文本能力 |

代价：两模型间要传隐藏状态、要做流式对齐，工程复杂度上升；且 Talker 依赖 Thinker，Thinker 慢则语音必慢。

---

## 3. TMRoPE：把你学过的 2D-RoPE 升级成「时间对齐」

### 3.1 复习：Qwen2-VL 的 M-RoPE

**这个公式在干什么**：RoPE（旋转位置编码）的本意是「位置不同 → 注意力里的相对角度不同」。标准 RoPE 把序列位置 $p$ 编成一个角度 $p\theta$；M-RoPE（Multimodal RoPE，Qwen2-VL 提出）把角度从 1 个升级成 3 个——**时间、高度、宽度**各转各的：

$$
\text{RoPE}(q, p) = q \cdot e^{i\,p\theta}, \quad p = (p_t, p_h, p_w)
$$

符号拆解：$q$ 是 query 向量的一个二维分量对（把它当复数看，乘 $e^{i p\theta}$ 就是旋转角 $p\theta$）；$\theta$ 是该维度对的基频；关键是位置从标量 $p$ 变成三元组 $p = (p_t, p_h, p_w)$——注意力头的不同维度对分别「看」时间、行、列位置。

- 纯文本：三个分量相等（退化为标准 1D RoPE）；
- 图像：时间分量相同，$(p_h, p_w)$ 编码 patch 的二维位置；
- 视频：$p_t$ 按帧推进（Qwen2-VL 按 2 fps 采样），$(p_h, p_w)$ 编码帧内位置。

**M-RoPE 的遗留问题**：视频有了时间轴，**音频没有**——Qwen2-VL 处理音频时只把它当普通序列，音频与视频在位置编码上互不知晓对方的时间位置。这对「视频里第 3 秒的爆炸声是什么」这类跨模态时间推理是致命的。

### 3.2 TMRoPE 的升级：给音频一个真实时间戳

TMRoPE（Time-aligned Multimodal RoPE，时间对齐多模态旋转位置编码）的做法：**音频每个 token 的时间分量 $p_t$ 直接取自它的物理时间戳**，与视频帧的时间戳共用同一时间轴，且统一规定「**1 个时间 ID 对应 40 ms**」：

$$
p_t^{\text{audio}}(k) = \left\lfloor \frac{t_k}{40\,\text{ms}} \right\rfloor, \qquad
p_t^{\text{video}}(j) = \left\lfloor \frac{\tau_j}{40\,\text{ms}} \right\rfloor
$$

**这个公式在干什么**：把「物理时刻」换算成「时间 ID」。$t_k$ 是第 $k$ 个音频 token 覆盖的时刻（音频塔输出恰好是 40 ms/帧，所以音频 token 的时间 ID 就是自然序号 $k$）；$\tau_j$ 是第 $j$ 个视频帧的时刻（视频是动态帧率，比如 2 fps 时相邻帧差 0.5 s，中间的时间 ID 就跳号）；分母 40 ms 是全模型统一的时间刻度。结果是：**同一物理时刻的音频和视频帧拿到相同/成比例的 $p_t$**——「视频里第 3 秒的爆炸声对应哪几帧画面」这个问题，被直接编码进了位置编码，所有注意力层、所有头都能免费使用，而不用靠数据去学。

工程实现上还有两个配套细节（论文原文）：① 音视频表征按实际时间**每 2 秒切成一个 chunk 交错排列**（每个 chunk 内视觉在前、音频在后），保证序列局部性；② 音频编码器的注意力从「整段全注意力」改成**按 2 秒分块**，以支持流式输入。

### 3.3 一句话总结继承关系

> M-RoPE 让图像有了空间感、视频有了时间感；TMRoPE 让**音频也上了同一条时间轴**——这是从「多模态拼接」到「多模态同步」的关键一步。

可运行的位置 ID 模拟见第 6.1 节。

---

## 4. 全模态流式：双轨流水线

### 4.1 三轨并行

```
输入轨：  音频块(2 秒注意力块，每帧 40ms) ─→ 音频编码器 ─→ Thinker 增量前向
          视频帧(动态帧率采样)          ─→ ViT ────────↗
          （音视频按实际时间每 2 秒一个 chunk 交错喂入，chunk 内视觉在前、音频在后）
Thinker轨：Thinker 逐块生成文本（边收边算）
Talker轨： Talker 消费 Thinker 已产出的隐藏状态，流式生成语音 token（~12.5 Hz）
输出轨：  流式声码器（Flow-Matching DiT + BigVGAN）把语音 token 合成 24 kHz 波形，边生成边播
```

### 4.2 首包延迟由什么决定

**这个公式在干什么**：它把「用户说完话」到「用户听到第一个字」之间的每一级流水线延迟列队点名——优化首包延迟时，先知道钱花在哪一段。

$$
T_{\text{首包}} \approx T_{\text{音频缓冲}} + T_{\text{Thinker 首 token}} + T_{\text{Talker 首语音块}} + T_{\text{声码器首块}}
$$

四个符号依次是：① 凑齐第一块输入所需的缓冲时间（流式编码器要求按块喂入）；② Thinker 吐出第一个文本 token 的时间（prefill + 首步解码）；③ Talker 凑齐第一组语音 token 的时间（约 80 ms/token，一组若干 token）；④ 声码器合成并播出第一块波形的时间。任何一段变慢都直接抬高首包——这就是为什么 Talker 必须不等整句。

关键工程点：Talker 不等整句文本，**Thinker 每产出一小段隐藏状态，Talker 就推进一次**。这与你 Stage 5 学的 CosyVoice chunk-aware 流式是同一个思想：**把「等整句」改成「等一个 chunk」**。

### 4.3 打断（barge-in）

流式输入轨持续开着：用户在模型说话时开口，输入轨立刻检测到新语音 → 中止 Talker/声码器输出 → Thinker 把新语音作为新指令处理。打断测试是本周 MVP 的第③项，状态机的完整实现放到 Stage 7 第 5-6 周教程。

---

## 5. 一模型三职：ASR + 语音问答 + TTS 如何不打架

多任务统一为同一序列格式，用任务指令区分：

- **ASR**：「请把这段语音转写成文字」→ 音频 → 文本；
- **语音问答**：「根据语音内容回答」→ 音频+问题 → 文本（+语音）；
- **TTS**：「把这句话读出来」→ 文本 → Talker 出语音。

训练上靠两件事防止互相干扰：① 任务指令模板严格区分（模型见过三种格式各自的大量样本）；② Thinker 的语言能力来自 Qwen2.5 预训练，微调阶段用混合配比保住通用能力——这与第 5-6 周要讲的灾难性遗忘监控是同一问题。

---

## 6. 实现与验证

### 6.1 TMRoPE 位置 ID 模拟（纯 Python，可运行）

按论文设定，时间刻度为「1 个时间 ID = 40 ms」。模拟 2 fps 视频 + 25 Hz 音频 token 在同一时间轴上换算：

```python
def tmrope_ids(video_timestamps, audio_timestamps, ms_per_id=40):
    """为视频帧与音频 token 分配统一时间轴上的位置 ID（1 ID = 40 ms）。
    video_timestamps: 每帧的物理时刻(秒)；audio_timestamps: 每个音频 token 的时刻。
    """
    def to_id(ts):
        return [round(t * 1000 / ms_per_id) for t in ts]
    return to_id(video_timestamps), to_id(audio_timestamps)

# 视频 2 fps（0, 0.5, 1.0, ... 秒，动态帧率下的最简情形），音频 token 25 Hz（40 ms/帧）
video_ts = [i * 0.5 for i in range(5)]            # 0 ~ 2.0 s
audio_ts = [k * 0.04 for k in range(51)]          # 0 ~ 2.0 s

v_ids, a_ids = tmrope_ids(video_ts, audio_ts)
print("视频 p_t:", v_ids)                          # [0, 12, 25, 38, 50] —— 帧间跳号

# 验证：1.0 秒时刻，视频帧与音频 token 的时间 ID 对齐
v_at_1s = v_ids[2]
a_at_1s = a_ids[25]
print(f"1.0s 处: 视频 p_t={v_at_1s}, 音频 p_t={a_at_1s}")
assert v_at_1s == a_at_1s == 25, "同一物理时刻的音视频位置 ID 必须对齐"
# 音频时间 ID 单调不减（允许同 ID 重复，对应同一时间桶）
assert all(a <= b for a, b in zip(a_ids, a_ids[1:])), "音频时间 ID 必须单调"
print("TMRoPE 时间对齐验证通过 ✓")
```

**预期输出**：`1.0s 处: 视频 p_t=25, 音频 p_t=25`，断言通过。注意视频帧的 ID 序列是 `[0, 12, 25, 38, 50]`——2 fps 的帧间隔 0.5 s = 12.5 个 ID，帧间「跳号」正是 M-RoPE/TMRoPE 与 1D RoPE 的本质区别：跳号空隙里没有任何 token，但物理时间照样流逝。真实实现还会把音视频按 2 秒 chunk 交错排列，这里只演示时间刻度对齐。

### 6.2 云部署与三种交互模式（本周 MVP）

```bash
pip install "transformers>=4.51" accelerate qwen-omni-utils soundfile
# flash-attention 2 可选（需 bf16/fp16 GPU）：pip install -U flash-attn --no-build-isolation
```

```python
import torch, soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B", torch_dtype=torch.bfloat16,
    device_map="auto", attn_implementation="flash_attention_2")
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

def run(conversation, use_audio_in_video=True):
    # 注意三处 use_audio_in_video 必须一致（apply_chat_template 的处理、processor、generate）
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True,
                       use_audio_in_video=use_audio_in_video)
    inputs = inputs.to(model.device).to(model.dtype)
    text_ids, audio = model.generate(**inputs, return_audio=True)  # speaker 默认 "Chelsie"，可选 "Ethan"
    if audio is not None:
        sf.write("reply.wav", audio.reshape(-1).float().detach().cpu().numpy(), samplerate=24000)
    return processor.batch_decode(text_ids, skip_special_tokens=True)

# 模式①：语音问答（输出文本+语音）
conv_qa = [{"role": "user", "content": [
    {"type": "audio", "audio": "question.wav"},
    {"type": "text",  "text": "请听这段语音并回答问题"}]}]

# 模式②：视频+音频联合理解（TMRoPE 的主战场）
conv_av = [{"role": "user", "content": [
    {"type": "video", "video": "clip.mp4"},
    {"type": "text",  "text": "视频里同时发生了什么画面和什么声音？"}]}]

# 模式③：打断测试——流式生成中途插入新语音，观察输出是否中止并响应新指令
# （自写推理循环：Talker 逐块生成 + 声码器逐块合成，边播边在输入轨喂入新音频块）
```

> **显存提示**：Qwen2.5-Omni-7B 的「7B」指 Thinker 主干；全模型（Thinker + Talker + 音频/视觉编码器）约 **10.7B 参数**，BF16 权重约 21 GB——建议 24 GB 显存以上的卡；不够时可量化（GPTQ/AWQ，性能优化方向的知识）或用 CPU 卸载。仅要文本输出时用 `model.generate(**inputs, return_audio=False)`，可省去 Talker 的开销。

**交付判据**：① 能听到语音回复且文本正确；② 对「画面与声音是否同步描述」抽查 5 条；③ 打断后 ≤1 s 内停止旧回复并开始响应。

### 6.3 Token 流全图（维度追踪法，本周交付）

以 10 秒语音输入为例，沿链路标注形状。音频编码器继承自 Qwen2-Audio：16 kHz 波形 → 128 通道梅尔谱（10 ms 步长）→ 卷积 stem 下采样到 50 Hz → **stride-2 池化**到 25 Hz（每帧对应 40 ms）：

```
16 kHz 波形           : (160000,)                      # 10 s
→ 梅尔谱              : (128, 1000)                    # 10 ms 步长 × 10 s
→ 卷积 stem（2 个 2× 卷积）: (1280, 500)                # 50 Hz × 10 s
→ stride-2 池化        : (250, 1280)                    # 25 Hz × 10 s
→ Thinker 嵌入         : (250, d_model)                 # 与文本 token 拼成一条序列
→ Thinker 输出         : 文本 token 流 (L_text,) + 隐藏状态流 (L, d_model)
→ Talker               : 语音 token 流 (~12.5 Hz × 回复时长)
→ 流式声码器            : 24 kHz 波形块流
```

**要点**：音频进 Thinker 是 25 Hz——10 秒占 250 个位置；对照第 1 周公式 $\text{token 数} = \text{时长} \times \text{帧率} \div \text{降采样倍率}$（这里帧率 50 Hz、池化 2×），30 秒语音就是 750 个 token，与文本共享同一上下文窗口。这个 25 Hz 也和 TMRoPE 的「1 ID = 40 ms」严格一致——位置刻度和 token 帧率是同一套设计。

### 6.4 Thinker-Talker 分工的可运行模拟

```python
import torch, torch.nn as nn

torch.manual_seed(0)
d = 128   # 玩具维度：Thinker 隐藏状态维度

class Thinker(nn.Module):
    def forward(self, text_ids, audio_feats):
        # 真实模型里音频经编码器+projector 后与文本拼接；此处直接拼玩具特征
        x = torch.cat([audio_feats, text_ids.float()], dim=0)  # (L, d)
        return torch.tanh(x)                                    # 隐藏状态流

class Talker(nn.Module):
    def __init__(self, d, vocab=64):
        super().__init__()
        self.head = nn.Linear(d, vocab)   # 语音 token 词表
    def step(self, hidden_state):
        logits = self.head(hidden_state)
        return int(logits.argmax(-1))     # 每个隐藏状态步产出一个语音 token

thinker, talker = Thinker(), Talker(d)
audio = torch.randn(25, d)                # 1 秒音频（25 Hz）
text = torch.randn(5, d)                  # 5 个文本位置
h_stream = thinker(text, audio)           # Thinker 边想边吐隐藏状态

speech_tokens = [talker.step(h) for h in h_stream]   # Talker 流式消费
assert len(speech_tokens) == h_stream.shape[0], "Talker 与 Thinker 逐步对齐"
assert all(0 <= t < 64 for t in speech_tokens)
print(f"Thinker 产出 {len(h_stream)} 个隐藏状态 -> Talker 产出 {len(speech_tokens)} 个语音 token")
print("Thinker-Talker 流式分工模拟 ✓（真实模型：Talker 自回归，还吃自己上一步的语音 token）")
```

**预期输出**：30 个隐藏状态对应 30 个语音 token，断言通过。注意真实 Talker 是**自回归**的（还以自己上一步的语音 token 为输入），这里只演示「消费 Thinker 隐藏状态」这条关键数据通路。

---

## 7. 工程权衡与失效模式

### 7.1 权衡

- **双模型通信成本**：传隐藏状态比传文本贵（连续向量、每步都传），换来韵律保真与低延迟；若部署资源紧张，可退化为「文本转语音」模式（Talker 只吃文本），音质略降但带宽骤减。
- **Thinker 大小**：7B 理解强但首包慢；端侧想抄这套架构必须砍 Thinker（这正是 Stage 7 用级联方案而不用 Omni 的原因之一）。
- **时间对齐精度**：TMRoPE 统一为「1 时间 ID = 40 ms」——刻度越细对齐越准但位置 ID 越大、外推压力越大；40 ms 恰好与音频塔的 25 Hz 输出帧率对齐，是精度与外推的折中。视频按动态帧率采样，帧间跳号由真实时间戳决定。

### 7.2 失效模式

1. **音画错位问答**：长视频中音频与视频时间戳处理不一致（如音频被重采样改了时长），TMRoPE 对齐失效。症状：「第 X 秒的声音」答非所问；定位：打印音视频各自时长与 token 数；修复：统一以解码器输出的真实时间戳为准，不信任文件头。
2. **语音音色/情感漂移**：长回复中 Talker 越说越「平」。根因：流式生成中隐藏状态的情感信号被文本内容信号稀释；修复：缩短单次回合、或在 Talker 侧加入参考音色/情感提示。
3. **打断失灵**：输入轨 VAD 阈值过高或缓冲过长，用户喊「停」无响应。定位：测「开口→输出停止」的实测延迟；修复：打断检测独立于 ASR，用低阈值能量/VAD 双通道。
4. **显存超限**：7B + 双模型 + 流式缓冲在 24G 卡上吃紧。修复：量化（GPTQ/AWQ——性能优化方向知识）或升卡；记录你的实际占用写进交付文档。

---

## 8. 延伸思考题（含解析）

**Q1**：如果去掉 Talker，让 Thinker 直接生成语音 token，会失去什么？
**A**：① 文本与语音两套目标挤进同一组参数，互相干扰，通用文本能力退化风险大；② 失去「先出文本可审计」的中间态，安全过滤与失败归因都变难；③ 流式并行收益减少——语音必须等文本解码完才能开始（除非改造成多任务并行头，工程上又绕回双模型）。

**Q2**：TMRoPE 相比「把音频时间戳写进文本提示」（如 `[3.2s] 有玻璃碎声`）好在哪？
**A**：提示法是显式符号，模型要先学会解析时间文本再做推理，误差依赖转写质量；TMRoPE 把时间直接编进位置编码，所有层、所有注意力头都能用，且对音频视频一视同仁，是归纳偏置级的对齐而非数据级的对齐。

**Q3**：为什么音频编码器输出要下采样到 25 Hz 而不是保留 50 Hz？
**A**：token 数与上下文占用、注意力 $O(L^2)$ 成本成正比；25 Hz 对语音语义足够（第 4 周会看到 Mimi 甚至压到 12.5 Hz），省下的一半预算给文本和视频。代价是时间分辨率减半，精确到几十毫秒的时间戳任务会受损。

**Q4**：Moshi 是单模型全双工，Qwen2.5-Omni 是双模型半双工——为什么说两者代表的是「不同的产品假设」？
**A**：Moshi 假设对话是连续的声学事件（抢话、附和都建模），适合电话/陪伴式场景；Omni 假设对话是回合制的任务交互（问-答-确认），先把内容做对，再做语音表现。架构选择由交互假设决定，不是纯技术优劣。

**Q5**：你 Stage 7 的端侧助手为什么不用 Qwen2.5-Omni，而是级联方案？
**A**：① 7B+Talker 双模型在 6 TOPS NPU 上实时性不达标；② 级联每个模块可单独量化/替换/调试（SenseVoice/llama.cpp 都是端侧验证过的）；③ 产品需要文本日志审计。Omni 是云端对照基线，不是端侧答案——这正是学习计划把两者分开训练的原因。

---

## 9. 面试一页纸：Qwen2.5-Omni 三连问

**问：Thinker-Talker 为什么拆两个模型？**
答：三个理由——① 目标解耦：内容正确性（文本）与声学表现（韵律情感）不在同一组参数上打架；② 数据效率：理解侧复用海量文本数据，生成侧只需语音对齐数据，避免端到端对「语音对话对」的强依赖；③ 可审计：Thinker 的文本输出是安全过滤与失败归因的中间态。代价是隐藏状态通信开销与工程复杂度。

**问：TMRoPE 解决什么问题？**
答：让音频和视频共享同一物理时间轴的位置编码。继承自 Qwen2-VL 的 M-RoPE（文本三轴相等、图像空间编码、视频时间轴），升级点是给音频 token 的 $p_t$ 赋真实时间戳——「第 3 秒的爆炸声对应哪几帧画面」这类跨模态时间推理由此成为位置编码级的归纳偏置，而非靠数据硬学。

**问：流式首包延迟由什么决定？**
答：$T_{\text{首包}} \approx T_{\text{音频缓冲}} + T_{\text{Thinker 首 token}} + T_{\text{Talker 首语音块}} + T_{\text{声码器首块}}$。关键设计是 Talker 不等整句文本，Thinker 每产出一小段隐藏状态 Talker 就推进一次——与 CosyVoice 的 chunk-aware 流式同一思想：把「等整句」换成「等一个 chunk」。

---

## 本周交付清单

- [ ] 云部署 Qwen2.5-Omni-7B，三种交互模式各留一条演示记录（文本+音频）。
- [ ] 完成打断测试，记录「开口→输出停止」实测延迟。
- [ ] 用维度追踪法画出 Token 流全图（6.3 为骨架，补上你实测的真实形状）。
- [ ] 跑通 6.1 的 TMRoPE 模拟，能口述 M-RoPE → TMRoPE 的继承与升级。
- [ ] 写一段 200 字结论：Thinker-Talker 双模型相比单模型 Speech-to-Speech 的三点优势。
