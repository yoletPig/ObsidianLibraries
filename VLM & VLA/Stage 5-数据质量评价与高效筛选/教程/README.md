# Stage 5 教程索引：数据质量评价与高效筛选

本目录是《Stage 5 学习计划》的配套教程，按周组织，延续前四阶段统一的四层框架：**第一性原理 → 系统架构与数据流 → 工程权衡与失效模式 → 实现与验证 → 延伸思考题**。

| 文档 | 对应周次 | 核心问题 |
| --- | --- | --- |
| [第 1 周：基础评价指标——从 NLL、PPL 到模态对齐得分](第1周-基础指标NLL与CLIPScore.md) | 第 1 周 | NLL 的"震惊程度"怎么算？为什么 NLL 两端都是坏样本？CLIP Score 过滤什么？ |
| [第 2 周：双向信息增益与模态依赖分析](第2周-双向信息增益与模态依赖.md) | 第 2 周 | "伪多模态数据"怎么定量识别？Image Necessity 的公式与实现逻辑是什么？ |
| [第 3 周：困难度矩阵、多样性与 Coreset 选择](第3周-困难度矩阵与Coreset选择.md) | 第 3 周 | Sweet Spot 为什么在中间？K-Means/FPS 如何兼顾困难度与多样性？ |
| [第 4 周：筛选算法验证与 "Less is More" 消融实验](第4周-LessIsMore消融验证.md) | 第 4 周 | 三组消融怎么设计才可信？"30% 数据达全量 98% 性能"如何证明？ |
| [阶段五自测验收与复盘](阶段五自测验收与复盘.md) | 终极自测 | 四项自测的参考答案、筛选算法选型速查、进入 Stage 6 的衔接 |

## 贯穿全阶段的一条主线

Stage 5 的全部内容服务于一个量化目标——**Less is More**：

$$
\underbrace{10\text{k 全量数据}}_{\text{Baseline}} \xrightarrow[\text{第 1-3 周}]{\text{打分: NLL + CLIP + }\Delta L_{\text{Img}} + \text{聚类}} \underbrace{30\%\text{ 精炼子集}}_{\text{剔除两端 + 覆盖多样性}} \xrightarrow{\text{第 4 周: 消融验证}} \geq 98\%\ \text{全量性能}
$$

三个层层递进的视角：

- **第 1 周（样本视角）**：单条数据本身的质量——模型对它的意外程度（NLL）、图文相关度（CLIP Score）、预测不确定性（Entropy）；
- **第 2 周（模态视角）**：图像与文本的**联动质量**——一条数据是否真的需要图才能学（Image Necessity），这是多模态数据筛选区别于纯文本筛选的核心增量；
- **第 3 周（集合视角）**：样本之间的关系——困难度处于 Sweet Spot、且在 Embedding 空间中互不冗余（Coreset）；
- **第 4 周（因果视角）**：以上全部假设的最终裁判——控制变量的消融实验。

读完五篇后，你应该能对一个 10k 级多模态数据集执行完整的"打分 → 剪枝 → 聚类 → 消融"流程，并用数据证明删掉的 70% 是冗余——这正是自测清单第 4 项的验收内核。

## 前置依赖与文献说明

- 前置：[Stage 2 教程](../Stage%202-VLM%20训练实践与%20SFT%20微调/教程/README.md)（SFT 流程与显存工程）、[Stage 3 教程](../Stage%203-VLM%20评测体系与幻觉分析/教程/README.md)（评测协议）、[Stage 4 教程](../Stage%204-数据合成与数据工程/教程/README.md)（本阶段的筛选对象即 Stage 4 的合成产出）。
- **文献标注**：DEITA 完整标题已核实为 *What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning*（ICLR 2024，arXiv:2312.15685，提出复杂度/质量/多样性三维度量，6K 样本即可匹敌基线）；LIMA（*Less Is More for Alignment*, NeurIPS 2023）无误。学习计划中的"Bidirectional Information Gain"（双向信息增益）作为专名未检索到唯一对应论文，但其机制——**以 PPL/Loss 差值度量图像对回答的因果贡献**——在 2025-2026 年文献中已被充分验证（如 Visual Information Gain, arXiv:2602.17186, ICML 2026；RAP 的因果差异估计器 arXiv:2506.04755 等）。第 2 周教程按"机制可靠、术语谨慎归因"的原则编写。
