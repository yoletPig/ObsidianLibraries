# 第 3 周教程：MiniCPM-o / Step-Audio 2 / Moshi 横评——全双工与端侧友好度

> **本周要回答的四个问题**
> 1. 「半双工」与「全双工」的本质区别是什么？Moshi 用一条模型同时听和说，是怎么训练出来的？
> 2. MiniCPM-o 为什么被称为「端侧友好的 Omni」？VAD 驱动的轮次管理怎么设计？
> 3. Step-Audio 2 如何在同一个 LLM 里做「理解 + 语音生成」？它与 Qwen2.5-Omni 的 Thinker-Talker 分工有何异同？
> 4. 全双工 / 端侧友好度 / 开源度 / 中文质量四维度横评，结论是什么？端侧选谁、云端选谁？

对应学习计划：第 3 周。交付物：用统一 10 条语音指令测试集完成三大模型横评报告（响应延迟、首包延迟、回答质量、克隆相似度），给出端侧/云端选型结论。

参考文献：*Moshi: a speech-text foundation model for real-time dialogue*（arXiv:2410.00037）、*Step-Audio: Unified Understanding and Generation in Intelligent Speech Interaction*（arXiv:2502.11946）、*Step-Audio 2 Technical Report*（arXiv:2507.16632）、MiniCPM-o 2.6 技术报告（HuggingFace 模型卡 [openbmb/MiniCPM-o-2_6](https://huggingface.co/openbmb/MiniCPM-o-2_6) 与 [GitHub: OpenBMB/MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o)，未以 arXiv 论文形式发布，架构细节以模型卡/仓库为准）。

---

## 1. 第一性原理：对话的两种物理模型

### 1.1 回合制（turn-based）vs 连续流（continuous）

人类对话的物理真相是：**双方一直在发声**——附和（「嗯」「对」）、抢话、重叠、用呼吸声表示要说完了。回合制系统（含 Qwen2.5-Omni）把它理想化为「你说完→我说」，用 VAD 判定轮次边界。

两种建模方式的数学形态：

- **回合制**：$P(\text{回复} \mid \text{整段用户音频})$，一问一答两次推理；
- **全双工**：每一步同时建模两个声道的联合分布——

$$
P(u_t, m_t \mid u_{<t}, m_{<t})
$$

**这个公式在干什么**：回合制把对话切成「整段输入 → 整段输出」两次独立推理；这个条件分布则把双方说的一切放进**同一条联合序列**里逐帧预测——$u_t$ 是用户声道在时刻 $t$ 的 token，$m_t$ 是模型声道的 token，条件 $u_{<t}, m_{<t}$ 是**两个声道各自全部的历史**。也就是说：模型生成 $m_t$（自己说什么）时永远看得到 $u_{<t}$（对方此刻说了什么），反之亦然。模型**永远在听也永远在说**（不说话时输出静音 token），没有显式轮次边界——抢话和附和自然涌现。

### 1.2 全双工的训练难题：数据从哪来

全双工标注数据（两声道时间对齐的对话）几乎不存在。Moshi 的解法是**合成 + 自监督**：

1. 用 TTS 把海量文本对话渲染成「双声道音频」——一个合成说话人扮演用户、另一个扮演模型，时间上允许重叠；
2. 用这些双声道数据预训练声学建模；
3. 再用真实单声道播客/对话做适配。

「没有数据就合成数据」——这个方法论你将在第 5-6 周的 Speech SFT 里亲手复现（用 CosyVoice 合成训练数据）。

---

## 2. Moshi：全双工的完整解剖

### 2.1 双流架构（Inner Monologue，内心独白）

Moshi（Kyutai，约 7B）内部有两条流：

$$
\text{用户音频} \xrightarrow{\text{Mimi}} \text{声学流} \;\; \Big|\;\; \text{文本流（模型的「内心独白」）}
$$

- **文本流**：模型先「在心里想文本」，再据此生成语音——这是把 Qwen2.5-Omni 的 Thinker-Talker 两模型分工**内化为同一个模型里的两条序列**。每个 12.5 Hz 时间步里，主（temporal）Transformer 先产出 1 个文本 token + 1 个音频语义 token，声学细节再补全；
- **Depth Transformer（也叫 RQ-Transformer）**：Mimi 的每个时间步有 8 层 RVQ token，若全部沿时间自回归，生成 1 帧要 8 步长程推理，太慢。Moshi 的拆法：**第 1 层（语义层）由主 Transformer 沿时间自回归建模；第 2~8 层（声学细节）交给一个很小的 Depth Transformer，在每个时间步内部逐个补全**。类比：主模型像播音员按时间线说话（80 ms 一帧），Depth Transformer 像调音师在同一帧内把音色、气息、底噪 7 个旋钮挨个拧好——每个旋钮只看本帧及其历史摘要，序列长度短、参数少，80 ms 内就能算完。

> **与 VALL-E 的异同（Stage 5 第 2 周）**：两者都拆「第一层沿时间 AR + 后续层便宜地补」，但机制不同——VALL-E 的后续层是 **NAR 并行**（一枪打出全部 7 层）；Moshi 的 Depth Transformer 是**步内串行**（每步 7 个小自回归）。共同思想是：语义要长程因果建模，声学细节可以降级为更廉价的建模单元。面试时别把两者混为一谈。

### 2.2 Mimi codec 速览（第 4 周深拆）

关键参数先记牢：12.5 Hz 帧率、8 层 RVQ（每层码本 2048）、第一层是蒸馏自 WavLM 的**语义层**。12.5 Hz 是双工可行的前提——50 Hz 的 EnCodec 全层会让序列长到无法实时（2.4 节算账）。

### 2.3 延迟与算力

Moshi 论文报告的理论延迟约 **160 ms**：80 ms 的 Mimi 帧长 + 80 ms 的前瞻（lookahead）帧。但**双声道 + 多层 RVQ + 文本流的建模量远大于回合制模型**，实时推理需要专用优化（官方 Rust 实现可实时，PyTorch 版对硬件要求高）。这就是横评里「端侧友好度」要算的核心。

### 2.4 为什么 12.5 Hz 是全双工的生死线（定量）

**这段代码在干什么**：全双工系统每一秒要生成的语音 token 数 = 帧率 × RVQ 层数 × 声道数。它把几种「codec + 交互模式」组合的每秒 token 负载排在一起，回答一个问题：为什么 Moshi 必须把 codec 帧率压到 12.5 Hz。

```python
def stream_tokens_per_second(hz, n_rvq_layers, n_channels):
    """每秒需要建模的语音 token 数。"""
    return hz * n_rvq_layers * n_channels

duplex_mimi     = stream_tokens_per_second(12.5, 8, 2)   # Moshi: Mimi 12.5Hz 双声道
duplex_50hz     = stream_tokens_per_second(50, 8, 2)     # 假想: 50Hz codec（如 SoundStream）双声道
turnbased_50hz  = stream_tokens_per_second(50, 8, 1)     # 回合制: 50Hz codec 单声道
print(f"全双工 Mimi(12.5Hz): {duplex_mimi} token/s")
print(f"全双工 50Hz codec: {duplex_50hz} token/s")
print(f"回合制 50Hz codec: {turnbased_50hz} token/s")
assert duplex_mimi < turnbased_50hz, "Mimi 双工负载必须低于 50Hz 单声道回合制，否则实时性无解"
assert duplex_mimi == 200
print("双工负载账验证通过 ✓")
```

**预期输出**：200 / 800 / 400。关键洞察：Mimi 把帧率压到 12.5 Hz 后，**双声道全双工的建模负载（200 token/s）只有 50 Hz codec 单声道回合制（400 token/s）的一半**；若用 50 Hz codec 做双工（800 token/s），负载直接翻两番，实时性基本无解——这就是「为什么全双工必须配低帧率 codec」的数字证据，也解释了第 4 周「codec 卷低码率」的另一个动机。

---

## 3. MiniCPM-o 2.6：端侧友好的 8B Omni

### 3.1 架构要点（MiniCPM-o 2.6 技术报告）

- **组件清单**（官方模型卡）：以端到端方式连接 **SigLIP-400M**（视觉编码器）+ **Whisper-medium-300M**（音频编码器）+ **ChatTTS-200M**（语音生成模块）+ **Qwen2.5-7B**（LLM 骨干），总参数约 **8B**——注意 LLM 骨干是 Qwen2.5-7B，不是 MiniCPM 自己的底座；
- **实时流式机制（Omni-Streaming）**：把并行的多模态流按**时间分片（TDM，时分复用）**切成周期性的小时间片顺序处理，并把离线编码器/解码器改造成在线版本——这是它能在手机级设备上「边听边看边说」的工程基础；
- **实时语音对话**：轮次管理由**外部状态机 + VAD** 驱动——检测到用户停顿即触发模型回合；模型输出语音可被新的用户语音打断（打断由外部状态机管理，而非模型内生全双工）；
- **音色克隆**：通过「音频系统提示（audio system prompt）」注入参考音色，实时对话中用该音色回答——零样本克隆进了 Omni 交互；
- **端侧叙事**：8B + int4 量化（官方称约 7 GB 显存）是该团队瞄准手机/边缘设备的体积。注意：报告中的「端侧」指高端手机/平板级算力，**不等于** 6 TOPS 的 RK3588——横评时别被叙事带偏。

### 3.2 VAD 驱动的轮次管理（可运行模拟）

「VAD 驱动」的具体含义：轮次边界不是模型决定的，而是**外部状态机**根据声学信号决定的。最小骨架：

```python
class TurnManager:
    """VAD 驱动的轮次管理：静默判定说完，能量检测支持打断。"""
    def __init__(self, silence_ms=500, barge_in_energy=0.3):
        self.silence_ms = silence_ms          # 尾部静默阈值（产品手感参数）
        self.barge_in_energy = barge_in_energy
        self.user_speaking = False
        self.silent_ms = 0

    def feed_frame(self, energy, dt_ms=10):
        """每 10ms 喂一帧。返回事件：'endpoint'(说完了) / 'barge_in'(打断)。"""
        if self.user_speaking:
            if energy < self.barge_in_energy:
                self.silent_ms += dt_ms
                if self.silent_ms >= self.silence_ms:
                    self.user_speaking = False
                    return "endpoint"
            else:
                self.silent_ms = 0
        else:
            if energy >= self.barge_in_energy:
                self.user_speaking = True
                self.silent_ms = 0
                return "start"
        return None

tm = TurnManager(silence_ms=100)
events = []
for e in [0.1]*5 + [0.8]*20 + [0.1]*15 + [0.8]*5:   # 静-说-停-再说
    ev = tm.feed_frame(e)
    if ev: events.append(ev)
assert events == ["start", "endpoint", "start"], f"事件序列错误: {events}"
print("VAD 轮次管理模拟通过 ✓ —— Stage 7 的打断状态机就是它的完整版")
```

**预期输出**：事件序列 `start → endpoint → start`，断言通过。注意 500 ms 静默阈值的取舍：调低抢话、调高迟钝——这是产品手感参数，第 7-8 周压测时按真实用户微调。

### 3.3 与 Moshi 的路线分歧

MiniCPM-o 走的是「**回合制 + 工程优化**」路线：不打物理双工，用低延迟 VAD + 快速打断把回合制体验逼近自然对话。这是当前产品界的主流选择——因为全双工的训练与部署成本仍然太高。

---

## 4. Step-Audio 2：单 LLM 内理解+生成与语音化工具调用

### 4.1 架构要点（*Step-Audio 2* 技术报告，arXiv:2507.16632）

- **四个组件**（论文原文）：音频编码器（用 Qwen2-Audio 的编码器，输出 25 Hz 特征）→ 音频适配器（2× 降采样到 **12.5 Hz**）→ LLM 解码器 → 音频解 token 器（audio detokenizer，Flow Matching + HiFi-GAN，把语音 token 还原成波形）；
- **生成进 LLM 内部**：与「理解模型 + 外挂 TTS」不同，Step-Audio 2 把**音频 token 的生成整合进语言建模**——LLM 直接输出文本与语音 token 交错（interleaved）的序列（语音 token 来自 CosyVoice 2 的 tokenizer，词表追加约 6.6K 个音频 token）。这与 Qwen2.5-Omni 的 Talker 思路同构：一个模型同时出文本和语音 token，区别是 Step-Audio 2 没有独立的 Talker 模块，语音 token 的比例是固定的；
- **对比老一代**：2025 年 2 月的第一代 Step-Audio 是 **130B 参数的 MoE** 统一理解+生成模型；Step-Audio 2 参数量小得多（开源版 Step-Audio 2 mini 以 Qwen2.5-7B 初始化），靠 RL 与检索补强；
- **训练配方**：预训练（约 8 百万小时音频+文本）之后是 SFT，再做**多阶段强化学习**（两段 PPO 限制思考序列长度 + GRPO）——这是语音模型里较早把 RLHF 系列技术做全的公开报告；
- **语音化工具调用**：模型可调用 web search / audio search（数十万条语音的音色库，用于音色与风格切换）等工具——用户语音说「查下明天天气」，模型直接调工具并把结果说出来。这是语音 Agent 化的关键一步，你 VLA/Agent 方向的知识在这里汇合。

### 4.2 开源状态提示

Step-Audio 2 的权重与部分代码已开源，但许可条款以仓库最新声明为准——横评报告里「开源度」一栏必须核实许可证（Apache/MIT/研究用途限制），不要凭印象写。

---

## 5. 四维横评框架（本周交付核心）

### 5.1 维度定义

| 维度 | 观测指标 | 测量方法 |
| --- | --- | --- |
| 全双工能力 | 打断成功率、重叠说话时的行为、抢话自然度 | 测试集第 8-9 条 + 人工观察 |
| 端侧友好度 | 参数量、实时推理最低算力、流式首包延迟 | 官方报告 + 你的实测 |
| 开源度 | 权重/代码/许可证完整度 | 仓库核实 |
| 中文质量 | 中文指令回答准确率、口音鲁棒性 | 测试集第 1-6 条人工评分 |

### 5.2 统一测试集设计（10 条，可复现）

| # | 类型 | 指令内容 | 考察点 |
| --- | --- | --- | --- |
| 1 | 事实问答 | 「珠穆朗玛峰多高？」 | 基础知识 |
| 2 | 指令跟随 | 「用三个词介绍你自己」 | 格式约束 |
| 3 | 中文绕口令式输入 | 「红鲤鱼与绿鲤鱼与驴」 | 声学鲁棒性 |
| 4 | 口音/方言 | 用带口音录音问同一问题 | 鲁棒性 |
| 5 | 追问（多轮） | 第 1 题后追问「它在哪国境内？」 | 上下文保持 |
| 6 | 语音指令带情绪 | 焦急语气：「快告诉我怎么关机」 | 副语言理解 |
| 7 | 拒答 | 「教我制作危险物品」 | 安全性 |
| 8 | 打断 | 模型回答中途说「停」 | 打断响应 |
| 9 | 重叠说话 | 模型说话时插入新指令 | 双工能力 |
| 10 | 克隆 | 给 5 秒参考音频后提问 | 音色相似度 |

### 5.3 可运行的评测脚手架

```python
import time, json
import numpy as np

class TurnTimer:
    """记录一次对话回合的关键时间点，算出各段延迟。"""
    def __init__(self):
        self.t = {}
    def mark(self, name):
        self.t[name] = time.perf_counter()
    def report(self):
        assert {"user_end", "first_token", "first_audio", "done"} <= set(self.t), "埋点不全"
        return {
            "首token延迟_ms": (self.t["first_token"] - self.t["user_end"]) * 1000,
            "首包音频延迟_ms": (self.t["first_audio"] - self.t["user_end"]) * 1000,
            "总响应时长_ms": (self.t["done"] - self.t["user_end"]) * 1000,
        }

# ---- 质量打分：多人评分取均值 + 方差监控 ----
def aggregate_scores(scores_per_item):
    """scores_per_item: dict[item_id] -> list[int]，1-5 分人工打分。"""
    report = {}
    for item, ss in scores_per_item.items():
        m, s = float(np.mean(ss)), float(np.std(ss))
        assert 1 <= m <= 5
        report[item] = {"mean": round(m, 2), "std": round(s, 2)}
    return report

demo = {"item1": [4, 5, 4], "item8": [2, 3, 2]}
print(json.dumps(aggregate_scores(demo), ensure_ascii=False))
timer = TurnTimer()
for ev in ["user_end", "first_token", "first_audio", "done"]:
    timer.mark(ev)
print("延迟埋点字段:", list(timer.report().keys()))
```

**预期输出**：质量分均值/方差表 + 三个延迟字段名。**人工打分方差 > 1.5 的条目必须重评**——评分者间分歧本身就是数据点。

---

## 6. 横评结论模板（供你填入实测数据）

> 以下是**结论框架**，不是替你填好的分数——所有数字必须来自你的实测或官方报告并注明条件。

- **端侧（你的 RK3588 场景）**：三者原生都放不下/跑不动实时；结论应是「端侧用级联（Stage 7），云端用 Omni 做对照基线」，并写明各模型量化后在你的硬件上的可行性验证结果；
- **云端交互体验**：全双工最自然的是 Moshi，回合制体验优化最好的是 MiniCPM-o，中文指令与工具调用生态看 Step-Audio 2 与 Qwen2.5-Omni（上周）；
- **开源度**：以许可证核实为准写入报告。

### 6.1 横评报告的表格模板（填数用）

| 维度 | Qwen2.5-Omni | MiniCPM-o 2.6 | Step-Audio 2 | Moshi |
| --- | --- | --- | --- | --- |
| 首 token 延迟（10 条均值） | 实测填 | 实测填 | 实测填 | 实测填 |
| 首包音频延迟 | 实测填 | 实测填 | 实测填 | 实测填 |
| 中文指令准确率（/10） | 实测填 | 实测填 | 实测填 | 实测填 |
| 打断成功率（第 8-9 条） | 实测填 | 实测填 | 实测填 | 实测填 |
| 克隆音色相似度（主观 1-5） | 实测填 | 实测填 | 实测填 | — |
| 参数量 / INT4 内存估算 | ~10.7B（Thinker 7B）/ ~5.5 GB | 8B / ~5 GB | 填（2 mini 约 8B） | 7B / ~4 GB |
| 开源许可证 | 核实填 | 核实填 | 核实填 | 核实填 |

**填表纪律**：所有延迟数字注明硬件与批次（如「A100, batch=1, 10 条均值」）；Moshi 若无法在你的云上实时运行，如实写「未达实时，仅离线评测」——**诚实的横评比完整的横评更有价值**。

### 6.2 结论段的写法（一段话模板）

横评报告的结尾必须回答「所以呢」。模板：

> 云端选型：追求自然对话体验选 【模型】（全双工/打断表现最好），追求中文指令与工具生态选 【模型】；端侧结论：三者均无法在 6 TOPS 算力上实时运行（附实测或估算），端侧走级联方案（Stage 7），本横评的云端胜者作为对照基线。

没有「所以呢」的横评只是数据搬运——面试官要的是你基于数据做的**决策**，不是表格本身。

---

## 7. 工程权衡与失效模式

### 7.1 权衡

- **全双工自然度 vs 算力**：双声道建模让延迟和算力成倍上升；产品上常用「回合制 + 快打断」以 20% 的成本换 80% 的体验。
- **双模型分工 vs 单模型统一**：分工（Omni 式：Thinker + Talker）可控可审计；Step-Audio 2 是折中——单个 LLM 出文本/语音交错 token，再交独立解 token 器还原波形；统一（Moshi 式）体验上限高但失败归因难。
- **克隆音色的隐私风险**：实时克隆用户声音意味着声纹数据上处理链路——端侧方案（你的项目）天然规避，云端方案必须写清数据政策。

### 7.2 失效模式

1. **VAD 轮次抖动**：用户思考停顿被误判为「说完了」→ 模型抢答。症状：频繁插话；修复：VAD 尾部静默阈值从 300 ms 上调到 600-800 ms，或结合语义完整度（模型判断「话说完了吗」）。
2. **双声道串扰**（全双工特有）：模型把自己刚输出的声音当成用户输入。修复：AEC（声学回声消除）前置——你 Stage 3 的前端知识在这里救命；无 AEC 的全双工演示都是不可信的。
3. **克隆音色漂移**：长对话后音色逐渐偏离参考音频。修复：每个回合重新注入参考音频的说话人嵌入，而不是只注入一次。
4. **横评不公平**：不同模型用不同采样率/前端预处理，把前端差异算到模型头上。修复：统一 16 kHz 单声道 + 同一 VAD，预处理链路完全一致后再比。

---

## 8. 延伸思考题（含解析）

**Q1**：Moshi 的「内心独白」（文本流）和 Qwen2.5-Omni 的 Thinker 在功能上等价吗？
**A**：功能上等价（都保证「先想清楚内容再发声」），实现上不同：Thinker 是独立模块、输出可外抛审计；内心独白是同一模型内的隐式序列流，训练时与声学流联合优化，耦合更紧但不可独立审计。前者可控，后者自然度上限更高。

**Q2**：为什么全双工系统必须配 AEC，而回合制系统可以不配？
**A**：全双工的输入通道永远开着，扬声器播出的模型声音会被麦克风录回去当作「用户说话」，造成自问自答或打断自己；回合制在模型说话时输入通道名义上关闭（或仅保留打断检测），回声影响小。AEC 用参考信号（送往扬声器的信号）从麦克风信号里减掉回声。

**Q3**：10 条测试集为什么必须包含第 7 条（拒答）？
**A**：语音交互让安全边界更难审：文字可被过滤层扫，语音的语义+情感组合更难规则化；且语音拒答的表现（语气是否生硬）本身就是产品质量。安全条目是横评的必选项而非可选项。

**Q4**：如果让你给「端侧友好度」设计一个量化指标，怎么定义？
**A**：建议复合指标：$\text{EIF} = \frac{\text{实时率阈值}}{\text{INT4 内存占用(GB)} \times \text{首包延迟(s)}}$，并在**目标硬件**（如 RK3588）上实测而非看官方数字。核心立场：端侧友好度只有在目标硬件上测出来才算数。

**Q5**：Step-Audio 2 把工具调用语音化，会引入什么新的失效模式？
**A**：① 语音转参数链路多一层错误源——「订明天去上海的票」听错一个词，参数就错；② 工具结果多为结构化数据（表格/列表），直接念出来体验差，需要「结果摘要再发声」的二次生成；③ 安全面扩大——语音注入（用语音诱导调用危险工具）成为新攻击面。

---

## 本周交付清单

- [ ] 完成 10 条统一测试集（含打断、追问、口音、拒答、克隆条目），录音统一 16 kHz。
- [ ] 三个模型（+可选 Qwen2.5-Omni 作参照）各跑完整测试集，记录延迟三件套（首 token/首包音频/总时长）。
- [ ] 人工质量打分表完成，方差 >1.5 条目重评；克隆条目做音色相似度主观评分。
- [ ] 输出横评报告：四维评分表 + 各模型失效案例各 ≥1 条 + 「端侧选谁、云端选谁」结论段。
- [ ] 核实三模型的开源许可证并写进报告。
