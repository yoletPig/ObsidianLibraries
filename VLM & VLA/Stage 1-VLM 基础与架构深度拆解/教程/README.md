# Stage 1 教程索引：VLM 基础与架构深度拆解

本目录是《学习计划》的配套教程，按周组织。每篇教程遵循统一框架：**第一性原理 → 系统架构与数据流 → 工程权衡与失效模式 → 实现与验证 → 延伸思考题**。

| 文档 | 对应周次 | 核心问题 |
| --- | --- | --- |
| [第 1 周：视觉编码器与表征基础](第1周-视觉编码器与表征基础.md) | 第 1 周 | 图片如何变成一串 Patch Token？CLIP/SigLIP 为什么是 VLM 视觉塔的事实标准？ |
| [第 2 周：模态对齐桥梁 Projector](第2周-模态对齐桥梁Projector.md) | 第 2 周 | 1024 维视觉特征如何映射到 4096 维 LLM 隐空间？压缩与保真如何取舍？ |
| [第 3 周：主流 VLM 架构与图像 Token 全流程](第3周-主流VLM架构与图像Token全流程.md) | 第 3 周 | `<image>` 占位符如何被替换？LLaVA-1.5 与 Qwen2-VL 前向传播有何本质差异？ |
| [第 4-5 周：SFT 与 LoRA 微调实战](第4-5周-SFT与LoRA微调实战.md) | 第 4-5 周 | LoRA 的数学原理是什么？如何用 LLaMA-Factory 完成一次 Qwen2-VL 微调？ |
| [第 6 周：复盘与自测验收](第6周-复盘与自测验收.md) | 第 6 周 | 进入 Stage 2 前，四项能力自测的参考答案与常见训练故障排查 |

## 贯穿全阶段的一条主线

$$
\text{图片} \xrightarrow{\text{Vision Encoder}} \text{视觉特征} \xrightarrow{\text{Projector}} \text{LLM 维度的 Embedding} \xrightarrow{\text{替换 } \langle image\rangle \text{ 占位符}} \text{LLM Backbone} \xrightarrow{\text{自回归}} \text{文本输出}
$$

每一周的教程都围绕这条链路的一个环节展开。读完五篇后，你应该能**徒手画出并推导出全链路每个节点的 Tensor 形状**——这正是第 6 周自测的验收标准。
