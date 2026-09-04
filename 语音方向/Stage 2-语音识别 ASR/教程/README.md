# Stage 2 教程索引：语音识别 ASR——从声学模型到端到端大模型

本目录是《学习计划》的配套教程，按周组织。每篇教程遵循统一框架：**第一性原理与数学本质 → 系统架构与数据流 → 工程权衡与失效模式 → 实现与验证（可运行代码）→ MVP 交付指导 → 延伸思考题（含解析）**。

> 本阶段的主线问题只有一个：**音频如何变成文字？** 三代技术（HMM 混合 → CTC/Attention → 大模型）都是对"音频与文字长度不一致、对齐未知"这一根本矛盾的不同回答。读完五篇，你应能闭卷推导 CTC 前向算法、手写 Whisper 推理循环、并解释为什么端侧要选非自回归模型。

| 文档 | 对应周次 | 核心问题 |
| --- | --- | --- |
| [第 1 周：ASR 技术史与对齐范式](第1周-ASR技术史与对齐范式.md) | 第 1 周 | 音频帧数与文字长度不一致，CTC 的空白符如何解决免对齐训练？ |
| [第 2 周：Whisper 深度拆解](第2周-Whisper深度拆解.md) | 第 2 周 | 30 秒音频窗口如何变成 1500 个 Encoder Token？特殊 token 如何统一 99 种语言？ |
| [第 3 周：SenseVoice 与 Paraformer（FunASR）](第3周-SenseVoice与Paraformer.md) | 第 3 周 | CIF 机制如何让模型自己决定"一帧对应几个字"？为什么它比 Whisper 快 15 倍？ |
| [第 4 周：流式 ASR 与低延迟解码](第4周-流式ASR与低延迟解码.md) | 第 4 周 | 实时转写的延迟预算怎么拆？chunk 多大才够低延迟又不伤精度？ |
| [第 5-6 周：ASR 微调实战](第5-6周-ASR微调实战.md) | 第 5-6 周 | 500 条领域数据如何微调 Whisper/SenseVoice？bad case 如何归因？ |

## 贯穿全阶段的一条主线

$$
\text{波形} \xrightarrow{\text{Mel 特征}} \text{Encoder} \xrightarrow{\text{对齐机制}} \text{文字序列}
$$

对齐机制是 ASR 的灵魂：CTC 用边缘化求和、Attention 用隐变量、CIF 用积分发射、LLM 用自回归。每一周都在回答"对齐怎么办"的不同侧面。

## 前置依赖

- Stage 1 的 `AudioPipeline`（第 5-6 周交付）是本阶段的数据入口；
- 微调实验需云 GPU（预算已确认），本地只做推理与代码研读；
- 依赖：`pip install funasr modelscope transformers jiwer datasets`。
