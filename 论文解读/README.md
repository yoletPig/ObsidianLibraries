# 论文解读系列：从"会读论文"到"读出为什么"

> 本系列精读近年（2023-2025）有**核心创新**且**开源可复现**的论文，覆盖大模型、端到端语音、世界模型、量化、加速五个方向。与教程系列的区别：教程回答"是什么、怎么用"，论文解读回答"**为什么这样设计、证据是什么、代价是什么**"。
> 每篇解读基于论文 PDF 原文（docling 解析提取图表），并强制包含：开源仓库导读、局限与争议、与本知识库其他方向的交叉衔接。

## 选题清单（第 1 批，6 篇）

| 篇 | 领域 | 论文 | 年份 | 核心创新 | 开源仓库 | 入选理由 |
| --- | --- | --- | --- | --- | --- | --- |
| [01](01-DeepSeek-V3/解读报告.md) | 大模型架构 | DeepSeek-V3 Technical Report (arXiv 2412.19437) | 2024-12 | MLA 多头潜在注意力、无辅助损失 MoE 负载均衡、FP8 混合精度训练、MTP 多 token 预测 | [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) | 2024 年底最有影响力的开源大模型技术报告；MoE+MLA 路线已被广泛复刻（Qwen3、Kimi 等），FP8 训练工程细节全网稀缺 |
| [02](02-DeepSeek-R1/解读报告.md) | 大模型 × 强化学习 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL (arXiv 2501.12948) | 2025-01 | 纯 RL（GRPO）无需 SFT 冷启动即涌现推理能力（R1-Zero），推理时"aha moment" | [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) | RLVR 范式的标志性工作，直接对应本库强化学习系列与 VLM Stage 7 |
| [03](03-Qwen2.5-Omni/解读报告.md) | 端到端语音 | Qwen2.5-Omni Technical Report (arXiv 2503.20215) | 2025-03 | Thinker-Talker 分脑架构（文本思考+流式语音合成并行）、TMRoPE 时间对齐多模态位置编码 | [QwenLM/Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) | 端到端全模态的代表：单一模型听/看/说，语音方向 Stage 6 的直系论文 |
| [04](04-NVIDIA-Cosmos/解读报告.md) | 世界模型 | Cosmos World Foundation Model Platform for Physical AI (arXiv 2501.03575) | 2025-01 | 世界基座模型平台：视频 tokenizer（连续/离散）+ 扩散/自回归两类 WFM + 后训练配方，服务物理 AI | [NVIDIA/Cosmos](https://github.com/NVIDIA/Cosmos) | 世界模型方向权重开放、生态最全的平台型工作；自动驾驶/具身智能的"视频生成式预训练"路线 |
| [05](05-BitNet-b1.58/解读报告.md) | 量化 | The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (arXiv 2402.17764) | 2024-02 | 权重三值化 {-1,0,1}（1.58 bit）、absmean 量化函数、矩阵乘变整数加法、从头训练而非 PTQ | [microsoft/BitNet](https://github.com/microsoft/BitNet) | 量化领域的"范式挑战"之作：不走 PTQ 缩位数路线，直接改数值范式；官方推理框架开源 |
| [06](06-vLLM-PagedAttention/解读报告.md) | 加速 | Efficient Memory Management for LLM Serving with PagedAttention (SOSP 2023, arXiv 2309.06180) | 2023-10 | KV Cache 虚拟分页管理（OS 分页思想移植到注意力缓存）、continuous batching、prefix 共享 | [vllm-project/vllm](https://github.com/vllm-project/vllm) | 推理加速的里程碑：吞吐提升 2-4×，已是事实标准；本库模型性能优化 Stage 5 的主题论文 |

**选题标准**（后续批次沿用）：① 近三年（可放宽到 2023）；② 有可清晰陈述的**核心创新**（一句话能说清"之前不行→现在行"）；③ 有**官方开源仓库**（代码或权重，能在本地跑通推理者优先）；④ 与本知识库至少一个方向强衔接。

## 解读报告统一模板

1. **论文档案**：标题 / 作者与机构 / 发表处 / arXiv / 开源仓库与许可证 / 模型权重可得性
2. **速览**：一句话 TL;DR + 问题/创新/证据三行卡
3. **背景与动机**：这篇论文出现之前，这个问题的主流做法与瓶颈
4. **核心创新拆解**：每条按"是什么 → 为什么之前不行 → 怎么做到 → 代价与权衡"
5. **方法细节**：关键公式（手写 LaTeX）、架构图（引用 PDF 提取图）
6. **实验证据复盘**：核心表格数字复述与解读，标注哪些结论被后续工作验证/推翻
7. **开源仓库导读**：目录结构、最小跑通路径、工程亮点
8. **局限与争议**：社区的真实批评与后续改进
9. **与本知识库的衔接**：指向库里具体章节文件
10. **面试视角**：3-5 个高频问题与答题要点

## 生产管线（可复现）

```bash
# PDF → markdown + 裁切图片（docling 2.x, --no-ocr 跳过OCR, 表格结构默认开启）
/home/yyl/papers_ws/run_docling.sh <论文.pdf> <输出目录>
```

- 解析产物（原始 md + 全部图片）存于 `/home/yyl/papers_ws/parsed/`（工作区，不入库）
- 精选关键图表复制到各论文目录 `assets/`，报告中以相对路径引用
- 公式由 docling 提取为 `formula-not-decoded` 占位，报告中一律手工重写为 LaTeX

## 与知识库的接口

- 大模型 → 《PyTorch与深度学习》《强化学习》；语音 → 《语音方向》Stage 5/6；世界模型 → 《自动驾驶》《VLM & VLA》Stage 10；量化/加速 → 《模型性能优化》Stage 1-5；每篇报告第 9 节给出具体文件级交叉引用。
- 同步到博客时，`论文解读` 作为新方向加入 `import_tutorials.py` 的 `DIRECTIONS`（需为其两层目录结构增加适配）。

## 约定

- 所有图表来自 arXiv 原文 PDF，版权归原论文作者，仅用于学习解读（合理引用）。
- 关键数字一律与 PDF 原文核对；解读中的推断明确标注"解读视角"。
