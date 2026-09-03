# Stage 4 教程索引：数据合成与数据工程

本目录是《Stage 4 学习计划》的配套教程，按周组织，延续前三阶段统一的四层框架：**第一性原理 → 系统架构与数据流 → 工程权衡与失效模式 → 实现与验证 → 延伸思考题**。

| 文档 | 对应周次 | 核心问题 |
| --- | --- | --- |
| [第 1 周：合成数据范式与 Self-Instruct 机制](第1周-合成数据范式与Self-Instruct.md) | 第 1 周 | 为什么要用合成数据？Depth/Breadth 演化到底怎么"演化"？多模态合成有哪两种模式？ |
| [第 2 周：细粒度视觉与空间数据自动化合成](第2周-细粒度视觉与空间数据合成.md) | 第 2 周 | 不靠人工标注，如何从一张普通图片自动生成带精准 BBox 的 Grounding 数据？ |
| [第 3 周：多步 Agent 轨迹与 Execution Trajectory 合成](第3周-Agent轨迹与执行验证合成.md) | 第 3 周 | ReAct 轨迹怎么造？为什么"环境执行验证"是 Agent 数据的灵魂？失败轨迹怎么用？ |
| [第 4 周：数据清洗、语义去重与多维过滤](第4周-数据清洗语义去重与过滤.md) | 第 4 周 | MinHash 和 Embedding 去重各解决什么问题？为什么语义去重比精确匹配重要？ |
| [第 5-6 周：高吞吐合成 Pipeline 工程化与闭环验证](第5-6周-高吞吐Pipeline与闭环验证.md) | 第 5-6 周 | asyncio + vLLM 的高并发架构怎么搭？Prefix Caching 省在哪？"合成→微调→验证"闭环怎么跑？ |
| [阶段四自测验收与复盘](阶段四自测验收与复盘.md) | 终极自测 | 四项自测的参考答案、合成数据质量检查清单、进入 Stage 5 的衔接 |

## 贯穿全阶段的一条主线

Stage 4 是 Stage 3 诊断结论的"生产线"：

$$
\underbrace{\text{归因报告的数据规格}}_{\text{Stage 3 输出}} \xrightarrow{\text{第 1-3 周: 三条合成产线}} \text{Raw 合成数据} \xrightarrow{\text{第 4 周: 清洗去重}} \text{高质量数据集} \xrightarrow{\text{第 5-6 周: 闭环验证}} \text{可量化的能力提升}
$$

- **文本/多模态 QA 线**（第 1 周）：Self-Instruct 演化出指令多样性；
- **空间/感知线**（第 2 周）：Grounding DINO + SAM 自动化产出几何真值——**这是唯一能给出"物理级精确真值"的合成线**，其数据可直接训练 Stage 3 诊断出的定位短板；
- **Agent 轨迹线**（第 3 周）：执行验证给出"过程级真值"，为 Stage 9（Agent 规划）提前铺路；
- **清洗与闭环**（第 4-6 周）把三条产线的产物变成可信数据集，并用 Stage 3 的 Benchmark 验证收益。

读完六篇后，你应该能搭出一条**"万级/日、带自动校验、分布可控"**的合成流水线——这是自测清单第 4 项的验收内核。

## 前置依赖与勘误说明

- 前置：[Stage 2 教程](../Stage%202-VLM%20训练实践与%20SFT%20微调/教程/README.md)（数据格式与 SFT 流程）、[Stage 3 教程](../Stage%203-VLM%20评测体系与幻觉分析/教程/README.md)（归因报告是本阶段的需求来源）。
- **论文勘误**：学习计划所引 TaskCraft 标题不完整。经核实（arXiv:2506.10055，OPPO Personal AI Lab），正确标题为 ***TaskCraft: Automated Generation of Agentic Tasks***（2025，含工具调用轨迹与执行反馈，约 4 万条任务，数据集结构为 pure_qa / atomic_trace / multihop_subtask_trace 三个 jsonl）。AgentTuning 标题核实无误（*AgentTuning: Enabling Generalized Agent Abilities for LLMs*, arXiv:2310.12823）。第 3 周教程按正确标题引用。
