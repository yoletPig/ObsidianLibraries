# Stage 6 教程索引：语音大模型与 Omni 多模态

本目录是《学习计划》的配套教程，按周组织。每篇教程遵循统一框架：**第一性原理与数学本质 → 系统架构与数据流 → 工程权衡与失效模式 → 实现与验证（可运行代码）→ MVP 交付指导 → 延伸思考题（含解析）**。

> 本阶段的主线问题：**如何让一个 LLM 直接"听懂"语音、"开口"说话？** 从级联包装到音频特征注入（类 LLaVA），再到 Speech-to-Speech 全双工，拆解 Qwen2.5-Omni / MiniCPM-o / Step-Audio 2 / Moshi 四大架构，最后用 TTS 合成数据完成一次 Speech SFT 微调闭环。你 VLM & VLA 阶段的全部架构知识，在这里逐项复用。

| 文档 | 对应周次 | 核心问题 |
| --- | --- | --- |
| [第 1 周：Speech-LLM 三种范式与 Qwen-Audio](第1周-Speech-LLM三种范式与Qwen-Audio.md) | 第 1 周 | 级联/特征注入/端到端三范式各有什么天花板？音频塔与 Vision Tower 如何逐项同构？ |
| [第 2 周：Qwen2.5-Omni 深度拆解](第2周-Qwen2.5-Omni深度拆解.md) | 第 2 周 | Thinker-Talker 为什么拆两个模型？TMRoPE 如何把音视频时间轴对齐进位置编码？ |
| [第 3 周：MiniCPM-o / Step-Audio / Moshi 横评](第3周-MiniCPM-o-StepAudio-Moshi横评.md) | 第 3 周 | 全双工怎么做？端侧友好度、开源度、中文质量四维横评谁赢？ |
| [第 4 周：音频离散化与语音 Codec 深挖](第4周-音频离散化与语音Codec深挖.md) | 第 4 周 | Semantic RVQ 为什么第一层是语义？12.5 Hz token 率如何决定 LLM 上下文压力？ |
| [第 5-6 周：Speech SFT 微调实战与阶段复盘](第5-6周-SpeechSFT微调实战与阶段复盘.md) | 第 5-6 周 | 没有语音指令数据怎么用 CosyVoice 造？冻结策略与 VLM LoRA 如何同构？微调后怎么防"变聋"？ |

## 贯穿全阶段的一条主线

$$
\text{波形} \xrightarrow{\text{音频塔}} \text{音频特征} \xrightarrow{\text{Projector}} \text{LLM 嵌入空间} \xrightarrow{\text{LLM}} \begin{cases} \text{文本} & \text{（理解）} \\ \text{语音 token} \xrightarrow{\text{声码器}} \text{波形} & \text{（生成）} \end{cases}
$$

理解侧与你学过的 VLM 完全同构（音频塔 ≈ Vision Tower，Projector 同构）；生成侧是本阶段新增的知识——LLM 的输出不再是文字，而是可被声码器还原成波形的离散语音 token。

## 前置依赖

- Stage 2 的 Whisper / SenseVoice 知识——多数音频塔就是 Whisper encoder 或其变体；
- Stage 5 的 RVQ / codec 知识——第 4 周会深挖并与量化方向联动；
- VLM & VLA 阶段的 LLaVA / Qwen2-VL（M-RoPE）/ LoRA 知识——本阶段大量逐项对照；
- 云 GPU（预算无上限）用于第 2 周部署 7B 与第 5-6 周微调；
- 依赖：`pip install transformers modelscope ms-swift`，`encodec`、`moshi`（含 Mimi）。

## 与其他方向的联动

- 第 4 周的 **RVQ** 与性能优化方向的 **GPTQ/AWQ 标量量化**互为镜像，是跨方向面试题的弹药；
- 第 5-6 周用 **Stage 5 的 CosyVoice** 合成训练数据，完成「语音生成 → 语音理解」的技能闭环；
- 本阶段的 Omni 架构认知，直接服务于 Stage 7 端侧助手的「云端对照」叙事。
