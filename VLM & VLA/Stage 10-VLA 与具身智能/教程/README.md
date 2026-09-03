# Stage 10 教程索引：VLA 与具身智能（Embodied AI & Robot Policy）

本目录是《Stage 10 学习计划》的配套教程，也是整个 VLM & VLA 学习路径的收官篇，延续统一四层框架：**第一性原理 → 系统架构与数据流 → 工程权衡与失效模式 → 实现与验证 → 延伸思考题**。

| 文档 | 对应周次 | 核心问题 |
| --- | --- | --- |
| [第 1-2 周：VLA 核心架构与动作离散化](第1-2周-VLA架构与动作离散化.md) | 第 1-2 周 | RT 系列/Octo/OpenVLA 的架构谱系；连续动作如何变成 Token？ |
| [第 3-4 周：Open X-Embodiment 数据集与 RLDS 数据流](第3-4周-OXE与RLDS数据流.md) | 第 3-4 周 | 具身领域的"ImageNet"怎么读？Episode/Step 的数据结构？ |
| [第 5-6 周：SimplerEnv 仿真闭环评测](第5-6周-SimplerEnv闭环评测.md) | 第 5-6 周 | 开环 vs 闭环的本质差异；Covariate Shift 为什么是 VLA 的头号杀手？ |
| [第 7-9 周：VLA 微调与空间能力增强](第7-9周-VLA微调与空间增强.md) | 第 7-9 周 | LoRA 微调新任务；Co-finetuning 防遗忘；空间/深度信息融合 |
| [第 10 周：Sim-to-Real 与结课总结](第10周-SimToReal与结课总结.md) | 第 10 周 | 仿真到现实的鸿沟；控制频率约束下的推理加速；结课作品集 |
| [阶段十自测验收与复盘](阶段十自测验收与复盘.md) | 终极自测 | 四项自测参考答案、十大阶段能力全景、出师答辩要点 |

## 贯穿全阶段的一条主线：从"看懂世界"到"动手改变世界"

$$
\underbrace{\text{VLM}}_{\text{Stage 1-6: 看懂并回答}} \xrightarrow{\text{动作离散化}} \underbrace{\text{VLA}}_{\text{Stage 9-10: 看懂并行动}} \xrightarrow{\text{闭环评测+微调}} \underbrace{\text{可靠策略}}_{\text{仿真验证 → 现实部署}}
$$

Stage 10 的全部内容都在处理一个跨越：**语言的评价标准是"对不对"，而动作的评价标准是"成没成"**——这带来三重新约束：

1. **误差是累积的**：VQA 答错一题无后续影响；机器人第一步偏 2 厘米，后续所有动作都在错误状态上展开（Covariate Shift，第 5-6 周核心）；
2. **评价必须闭环**：静态数据集上的 loss 与真实成功率严重脱钩（Stage 7 的 rollout 思想在此变成刚需）；
3. **系统有实时硬约束**：10~20Hz 控制频率要求每次前向 <100ms/50ms——Stage 8 的全部推理工程在此从"省钱"变成"能不能用"。

## 前序阶段的最终汇合（本阶段大量复用）

| 复用点 | 来源 | 本阶段用途 |
| --- | --- | --- |
| SigLIP/DINOv2 视觉编码 | Stage 1 | OpenVLA 的双塔视觉编码器 |
| 跨模态拼接/占位符 | Stage 3 | VLA 的 VLM 主干机制 |
| LoRA 微调与灾难性遗忘 | Stage 2/6 | 第 7-9 周的策略微调 |
| 评测协议与归因纪律 | Stage 3 | 闭环评测与失败分析 |
| 数据合成与验证器 | Stage 4 | 空间数据融合、仿真 GT |
| 轨迹/动作执行 | Stage 7/9 | 闭环 rollout 与控制回路 |
| 推理加速 | Stage 8 | 满足控制频率的硬约束 |

读完六篇 + 完成结课作品，你应该拥有从"数据解析 → 模型微调 → 闭环评测 → 推理加速"的完整具身实践能力——这正是出师自测与求职作品集的内核。

## 文献核实说明

- **OpenVLA**（已核实 arXiv:2406.09246 及官网）：7B 参数开源 VLA，**Prismatic-7B VLM** 微调而来——视觉编码器为 **DINOv2 + SigLIP 双塔融合**，投影层接 **Llama 2 7B** 主干，输出 token 化动作；预训练数据为 **OXE 的 970k 真实机器人 episodes**（64×A100 训练 15 天）；论文口径：29 个任务上以 7×更少参数超过 RT-2-X (55B) **16.5%** 绝对成功率，超过 Diffusion Policy 20.4%；LoRA 微调仅调 **1.4% 参数**即匹敌全量微调，且**只调最后一层或冻结视觉塔效果差**（对第 7-9 周微调策略是关键证据）。学习计划所引标题无误。
- **RT-2**（*RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, 2023）与 **Octo**（*Octo: An Open-Source Generalist Robot Policy*, 2024）所引无误。
- OXE（Open X-Embodiment）、RLDS、SimplerEnv、ManiSkill 均为真实开源项目；具体数据集字段以所下载版本的文档为准。
