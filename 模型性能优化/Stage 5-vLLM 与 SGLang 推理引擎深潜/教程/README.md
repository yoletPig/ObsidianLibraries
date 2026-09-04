# Stage 5 教程索引：vLLM 与 SGLang 推理引擎深潜

本目录是《学习计划》的配套教程，按周组织。每篇教程遵循统一框架：**第一性原理与数学本质 → 系统架构与数据流 → 工程权衡与失效模式 → 实现与验证（可运行代码）→ MVP 交付指导 → 延伸思考题（含解析）**。

> 你在 VLM Stage 8 已经"会用" vLLM/SGLang。本阶段的目标是下潜到引擎内部，回答一个更硬的问题：**每一个延迟数字从哪来？** 读完五篇，你应能手算任意模型的 KV 显存、讲清调度器每个队列的迁移规则、推导投机解码的加速比公式，并做一次完整的引擎级调优实验。

| 文档 | 对应周次 | 核心问题 |
| --- | --- | --- |
| [第 1 周：KV Cache 与 PagedAttention 实现层](第1周-KVCache与PagedAttention实现层.md) | 第 1 周 | 物理块表怎么管理？7B 模型 8k 上下文的 KV 显存怎么手算？ |
| [第 2 周：调度器与连续批处理状态机](第2周-调度器与连续批处理状态机.md) | 第 2 周 | waiting/running 队列怎么迁移？chunked prefill 的预算怎么分？ |
| [第 3 周：投机解码与解码加速全家桶](第3周-投机解码与解码加速全家桶.md) | 第 3 周 | 接受率 α 与加速比的关系式怎么推？CUDA Graph 消掉了什么开销？ |
| [第 4 周：量化接入与多模态推理](第4周-量化接入与多模态推理与多LoRA服务.md) | 第 4 周 | GPTQ/AWQ 在引擎里映射到哪个 kernel？音频输入怎么进引擎？ |
| [第 5-6 周：单引擎调优方法论与复盘](第5-6周-单引擎调优方法论与复盘.md) | 第 5-6 周 | `gpu-memory-utilization` 调高调低的真实影响？排队爆炸怎么排查？ |

## 贯穿全阶段的一条主线

$$
\text{请求} \xrightarrow{\text{调度器}} \text{Prefill（算力瓶颈）} \xrightarrow{\text{KV Cache}} \text{Decode（访存瓶颈）} \xrightarrow{\text{输出}}
$$

推理引擎的所有设计——PagedAttention、连续批处理、chunked prefill、投机解码、PD 分离——都是围绕这两个阶段的**不同瓶颈特性**展开的。抓住"prefill 算力瓶颈、decode 访存瓶颈"这条主线，所有优化动机都能推出来。

## 前置依赖

- VLM Stage 8 的使用层知识（PagedAttention 概念、Prefix Caching 用法）——不重讲；
- 云 GPU（预算已确认，用完即释放）；
- 依赖：`pip install vllm sglang`，benchmark 脚本在各自仓库的 `benchmarks/` 目录。
