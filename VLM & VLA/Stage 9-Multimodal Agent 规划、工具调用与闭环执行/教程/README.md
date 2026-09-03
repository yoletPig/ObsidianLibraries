# Stage 9 教程索引：Multimodal Agent 规划、工具调用与闭环执行

本目录是《Stage 9 学习计划》的配套教程，按周组织，延续前八阶段统一的四层框架：**第一性原理 → 系统架构与数据流 → 工程权衡与失效模式 → 实现与验证 → 延伸思考题**。

| 文档 | 对应周次 | 核心问题 |
| --- | --- | --- |
| [第 1 周：Agent 核心范式与 Tool Use / 约束解码](第1周-ReAct与工具调用约束解码.md) | 第 1 周 | ReAct 循环的工程化实现；约束解码如何做到 100% 合法 JSON？ |
| [第 2 周：任务规划、反思与长短期记忆](第2周-规划反思与记忆机制.md) | 第 2 周 | 子目标拆解/ToT 怎么做？`<reflection>` 自我纠错的工程化与局限？ |
| [第 3 周：视觉 GUI / Web / OS 操作控制 Agent](第3周-GUI-ScreenGrounding-Agent.md) | 第 3 周 | 截图→动作的映射；SoM 为什么把坐标回归变成选择题？ |
| [第 4 周：Agent 轨迹合成、Sanity Check 与清洗](第4周-轨迹合成与清洗.md) | 第 4 周 | 死循环/逻辑断层检测；Trajectory Pruning 为什么是必须的？ |
| [第 5-6 周：端到端 Multimodal Agent 系统集成](第5-6周-多Agent系统集成实战.md) | 第 5-6 周 | 多 Agent 协作架构、Guardrails 与容错，交付一个 Deep Research 系统 |
| [阶段九自测验收与复盘](阶段九自测验收与复盘.md) | 终极自测 | 四项自测的参考答案、Agent 可靠性检查清单、进入 Stage 10 的衔接 |

## 贯穿全阶段的一条主线

Stage 9 把模型从"回答者"升级为"执行者"。核心是一条**闭环控制回路**：

$$
\underbrace{\text{感知}}_{\text{图像/截图/工具返回}} \to \underbrace{\text{规划}}_{\text{Thought/子目标}} \to \underbrace{\text{行动}}_{\text{工具调用/点击}} \to \underbrace{\text{环境反馈}}_{\text{Observation}} \to \underbrace{\text{反思修正}}_{\text{Reflection}} \to \text{循环直至完成}
$$

与前序阶段的接口（大量复用，标注回链）：

- **Stage 7 第 4 周**已手写过 ReAct 轨迹与环境执行器——本阶段第 1 周把它升级为**生产级 Agent Loop**（约束解码、工具注册、超时控制）；
- **Stage 4 第 3 周**的轨迹合成与执行验证——本阶段第 4 周的清洗与剪枝是其质检端的深化；
- **Stage 3 的 Grounding/IoU 评测**与 **Stage 4 第 2 周的坐标数据**——本阶段第 3 周 GUI Grounding 的直接地基；
- **Stage 5 的难度/质量思维**——本阶段轨迹过滤的"两端截断"是它的镜像。

读完六篇后，你应该能独立交付一个"可观测、可校验、可恢复"的多模态 Agent 系统——自测清单第 4 项（轨迹工程）与实战交付（Deep Research 系统）的验收内核。

## 文献与框架说明

- **论文勘误**：学习计划第 4 周所引 TaskCraft 标题 *"TaskCraft: Structuring Execution Trajectories for Agentic Post-training"* 不准确。已核实正确标题为 ***TaskCraft: Automated Generation of Agentic Tasks***（arXiv:2506.10055，2025；核心是 depth/width 两个维度的任务扩展算子，约 3.6 万条难度可扩展、多工具、可验证的 Agentic 任务及执行轨迹）——第 4 周教程按正确标题与真实机制讲解。
- ReAct（*ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR 2023）与 Reflexion（*Reflexion: Language Agents with Verbal Reinforcement Learning*, NeurIPS 2023）所引无误。
- 工具链（Outlines、vLLM guided decoding、OmniParser、Playwright、langgraph 等）以所装版本文档为准，本教程聚焦不随版本漂移的机制。
