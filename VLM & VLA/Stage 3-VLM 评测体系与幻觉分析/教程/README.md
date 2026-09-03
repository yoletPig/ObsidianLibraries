# Stage 3 教程索引：VLM 评测体系与幻觉分析

本目录是《Stage 3 学习计划》的配套教程，按周组织，延续前两个 Stage 的四层框架：**第一性原理 → 系统架构与数据流 → 工程权衡与失效模式 → 实现与验证 → 延伸思考题**。

| 文档 | 对应周次 | 核心问题 |
| --- | --- | --- |
| [第 1 周：评测数据集体系与指标拆解](第1周-评测体系与指标拆解.md) | 第 1 周 | 选择题怎么出才防作弊？EM/BLEU 为什么不可靠？如何从带思维链的混乱输出中精准抽答案？ |
| [第 2 周：自动化评测框架 VLMEvalKit 实战](第2周-VLMEvalKit自动化评测实战.md) | 第 2 周 | 评测框架的架构怎么拆？如何用 vLLM 把几小时压到几分钟？缓存与断点续评怎么做？ |
| [第 3 周：视觉幻觉与空间能力深挖](第3周-视觉幻觉与空间能力深挖.md) | 第 3 周 | 幻觉分几类？POPE 的三种采样如何度量"先验诱导"？"没看见"和"瞎编"怎么区分？ |
| [第 4 周：自建 Benchmark 与错误根源诊断](第4周-自建Benchmark与失败归因.md) | 第 4 周 | 怎么为自己的场景建 mini 基准？失败案例如何归因到具体模块并反哺数据？ |
| [阶段三自测验收与复盘](阶段三自测验收与复盘.md) | 终极自测 | 彩票攻击、No-image baseline 消融、1 小时接入自定义模型——四项自测的参考答案 |

## 贯穿全阶段的一条主线

Stage 3 的全部内容服务于一个能力跃迁：**从"凭感觉看效果"到"定量化诊断"**。

$$
\text{直觉判断} \longrightarrow \underbrace{\text{标准化评测}}_{\text{第 1-2 周}} \longrightarrow \underbrace{\text{缺陷定位}}_{\text{第 3 周}} \longrightarrow \underbrace{\text{归因反哺}}_{\text{第 4 周}} \longrightarrow \text{数据/训练干预（Stage 4/5）}
$$

- 第 1 周解决**量什么**（Benchmark 体系与指标设计）；
- 第 2 周解决**怎么量**（自动化评测工程）；
- 第 3 周解决**量出来的失败意味着什么**（幻觉分类与归因实验）；
- 第 4 周解决**量完之后做什么**（自建基准与反哺闭环）。

读完五篇后，你应该能对一个陌生模型给出一份**有证据链的诊断报告**：每个结论都指向具体的评测数字与失败案例，每个数字都能复现——这正是 Stage 4/5 数据合成工作的输入。

## 前置依赖与勘误说明

- 建议先完成 [Stage 2 教程](../Stage%202-VLM%20训练实践与%20SFT%20微调/教程/README.md)：本阶段的"评测反哺训练"闭环直接对接 Stage 2 的数据配方方法论。
- **论文标题勘误**：学习计划所引 HallusionBench 标题有误，正确标题为 *HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models*（CVPR 2024，已核实 arXiv:2310.14566）。第 3 周教程按正确标题引用。
- VLMEvalKit 的接口与机制（`generate_inner()`、`use_vllm` 开关、`SPLIT_THINK` 解析等）已对照其 GitHub main 分支核实（2026 年仍在活跃维护），第 2 周教程按已核实事实编写。
