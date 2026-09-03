# Stage 8 教程索引：大模型与多模态训练系统及推理工程

本目录是《Stage 8 学习计划》的配套教程，按周组织，延续前七阶段统一的四层框架：**第一性原理 → 系统架构与数据流 → 工程权衡与失效模式 → 实现与验证 → 延伸思考题**。

| 文档 | 对应周次 | 核心问题 |
| --- | --- | --- |
| [第 1 周：显存瓶颈剖析与 PyTorch 分布式并行基础](第1周-显存剖析与分布式基础.md) | 第 1 周 | 显存账本的每一项怎么算？通信原语与 DDP 梯度重叠怎么工作？ |
| [第 2 周：分布式训练引擎——ZeRO 与 FlashAttention](第2周-ZeRO与FlashAttention.md) | 第 2 周 | ZeRO 1/2/3 各切分什么、通信代价如何？FlashAttention 的 IO 优化本质？ |
| [第 3 周：高吞吐推理引擎——vLLM 与 PagedAttention](第3周-vLLM与PagedAttention.md) | 第 3 周 | KV Cache 碎片从哪来？PagedAttention 与 Continuous Batching 如何工作？ |
| [第 4 周：前沿推理系统——SGLang 与 Prefix Caching](第4周-SGLang与PrefixCaching.md) | 第 4 周 | RadixAttention 怎么自动共享前缀？Chunked Prefill 混合调度省在哪？ |
| [第 5 周：性能诊断 Profiling 与量化/内核加速](第5周-Profiling与量化加速.md) | 第 5 周 | 怎么定位 Compute-Bound vs Memory-Bound？量化路线（AWQ/GPTQ/FP8）怎么选？ |
| [阶段八自测验收与复盘](阶段八自测验收与复盘.md) | 终极自测 | 四项自测的参考答案、系统选型速查、进入 Stage 9 的衔接 |

## 贯穿全阶段的一条主线

Stage 8 回答一个系统工程的元问题：**大模型的每一 TFLOPs、每一 GB 显存、每一条 HBM 带宽，到底花在了哪里，还能不能省？**

$$
\underbrace{\text{显存账本}}_{\text{第 1 周: 每项怎么算}} \longrightarrow \underbrace{\text{训练系统}}_{\text{第 2 周: ZeRO + FA}} \longrightarrow \underbrace{\text{推理系统}}_{\text{第 3-4 周: vLLM + SGLang}} \longrightarrow \underbrace{\text{诊断与压缩}}_{\text{第 5 周: Profiling + 量化}}
$$

- **第 1 周**是全阶段的"会计学"：权重/梯度/优化器/激活/KV Cache 的精确公式——后面所有技术的收益都要用这本账来量化；
- **第 2 周**治训练侧的两大瓶颈：显存冗余（ZeRO 切分）与注意力访存（FlashAttention）；
- **第 3-4 周**治推理侧：KV Cache 的碎片（PagedAttention）与调度（Continuous Batching），再到前缀复用（RadixAttention）；
- **第 5 周**收束为方法论：先 Profile 定位瓶颈（Compute-Bound vs Memory-Bound），再对症下药（量化/内核）。

本阶段与前序教程的深度衔接：Stage 2 第 4 周（显存账本与训练调优）、Stage 3 第 2 周（vLLM 评测加速）、Stage 7 第 3 周（verl 的 HybridEngine）已零散铺垫的机制，本阶段将系统性展开——**重复处不再推导，只标注回链**。

读完六篇后，你应该能对任一训练/推理任务：算清显存账、选对并行/引擎方案、用 Profiler 定位瓶颈、并用数据写出优化报告——这正是自测清单第 4 项"系统交付"的验收内核。

## 硬件与版本说明

性能数字（TFLOPS、加速比、显存缩减比）**强依赖硬件（A100/H100/4090 等）、CUDA/PyTorch/引擎版本与负载形态**。本教程所有引用数字均标注测试条件；凡标注"经验量级"的数字用于建立直觉，**你的环境必须实测**。论文出处（ZeRO、FlashAttention、vLLM/SOSP 2023、AWQ、GPTQ、SmoothQuant）均为领域标准文献，与学习计划所引一致，无需勘误。
