# Stage 7 教程索引：RLVR 与 Multimodal Reasoning

本目录是《Stage 7 学习计划》的配套教程，按周组织，延续前六阶段统一的四层框架：**第一性原理 → 系统架构与数据流 → 工程权衡与失效模式 → 实现与验证 → 延伸思考题**。

| 文档 | 对应周次 | 核心问题 |
| --- | --- | --- |
| [第 1 周：RL 基础、PPO 痛点与 GRPO 算法推导](第1周-RL基础与GRPO推导.md) | 第 1 周 | GRPO 为什么不需要 Critic？组内相对优势的推导与 β/ε 的作用？ |
| [第 2 周：RLVR 可验证奖励与 Verifier Engine 设计](第2周-RLVR与Verifier引擎.md) | 第 2 周 | Rule-based Reward 怎么设计？防 Reward Hacking 的工程手段有哪些？ |
| [第 3 周：verl 框架架构与分布式 Rollout](第3周-verl框架与分布式Rollout.md) | 第 3 周 | HybridEngine 如何消除训练/生成切换开销？Ray 调度与配置节点怎么读？ |
| [第 4-5 周：多模态 Visual CoT 与 Vision-Language RLVR 实战](第4-5周-多模态RLVR与VisualCoT实战.md) | 第 4-5 周 | VLM+GRPO 的数据/Prompt 怎么构造？"慢思考涌现"怎么观测与验证？ |
| [第 6-7 周：轨迹过滤与 Post-Training 消融对比](第6-7周-轨迹过滤与三路消融.md) | 第 6-7 周 | 全同分组为什么无梯度？SFT/DPO/GRPO 三路消融怎么设计？失败模式怎么归因？ |
| [阶段七自测验收与复盘](阶段七自测验收与复盘.md) | 终极自测 | 四项自测的参考答案、算法选型速查、进入 Stage 8 的衔接 |

## 贯穿全阶段的一条主线

Stage 7 把 Stage 6 的"从偏好中学习"升级为"**从可验证的结果中学习**"：

$$
\underbrace{\text{偏好信号}}_{\text{Stage 6: 人类/AI 判断}} \longrightarrow \underbrace{\text{可验证奖励}}_{\text{RLVR: 程序化 Verifier}} \xrightarrow{\text{GRPO 组内相对优化}} \text{Visual CoT / 慢思考涌现} \xrightarrow{\text{第 6-7 周}} \text{SFT/DPO/GRPO 三路消融}
$$

- 第 1 周建立数学地基：**Critic 的本质是"基线"，而 GRPO 用组内均值当免费基线**——这是理解一切的关键；
- 第 2 周解决"奖励从哪来"：Verifier Engine 的模块化设计与防 Reward Hacking 工程学；
- 第 3 周解决"训练怎么跑"：verl 的 HybridEngine 与 Ray 分布式架构（RL 训练的算力形态与 SFT/DPO 完全不同——一半时间在生成，一半在更新）；
- 第 4-5 周在 VLM 上闭环：空间推理数据 + 视觉 Verifier + GRPO，观测慢思考涌现；
- 第 6-7 周用消融证明价值：三路对比 + 失败模式归因。

读完七篇后，你应该能独立完成"设计 Verifier → 搭 verl → 跑多模态 GRPO → 写消融报告"的全流程——这正是自测清单的验收内核，也是通向 Stage 8（训练系统与推理工程）的 RL 工程底座。

## 前置依赖与文献/框架说明

- 前置：[Stage 2](../Stage%202-VLM%20训练实践与%20SFT%20微调/教程/README.md)（训练工程与显存账本）、[Stage 3](../Stage%203-VLM%20评测体系与幻觉分析/教程/README.md)（评测协议）、[Stage 6](../Stage%206-Post-training%20偏好对齐与%20DPO%20实战/教程/README.md)（隐式奖励与 KL 正则——GRPO 的 $\beta D_{KL}$ 与 DPO 的 β 是同一正则思想）。
- **论文**：DeepSeekMath（arXiv:2402.03300，GRPO 的提出处）、DeepSeek-R1（arXiv:2501.12948，RLVR 规模化验证）——学习计划所引无误。
- **框架**（已核实 GitHub main 分支，2026-09 仍高度活跃，v0.9.0）：verl 已迁移至 `verl-project/verl`（原名 volcengine），是 **HybridFlow 论文**（arXiv:2409.19256）的开源实现；核心特性：FSDP/FSDP2/Megatron 训练后端 + vLLM/SGLang rollout 后端、**3D-HybridEngine**（官方描述："Efficient actor model resharding… Eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases"）、PPO/GRPO/GSPO/DAPO/RLOO/ReMax 等算法族、VLM 多模态 RL 支持（如 Qwen2.5-VL 的 GRPO 官方示例）、Reward 支持 model-based 与 function-based（可验证奖励）。生态参考：TinyZero、DAPO（AIME 2024 上 50 分的开源 SOTA，基于 verl）、Easy-R1（多模态 RL）、verl-vla（2026-08 发布的 VLA 后训练框架）等。
