# Stage 4 教程索引：GGML / llama.cpp 与端侧推理——CPU 与 NPU 上的大模型

本目录是《学习计划》的配套教程，按周组织。每篇教程遵循统一框架：**第一性原理与数学本质 → 系统架构与数据流 → 工程权衡与失效模式 → 实现与验证（可运行代码/命令链）→ 延伸思考题（含解析）→ 交付清单**。

> 🔧 **本阶段无 GPU 依赖，全部本地可执行**：x86 机器跑 GGML CPU 推理，RK3588/RK3576 与 MTK 板做 NPU 部署（两板都要实操）。这是整个方向里唯一"不用上云"的阶段——也是你语音结课项目端侧组件的直接来源。

> 本阶段的主线问题只有一个：**没有 Tensor Core、没有大显存，大模型怎么在 CPU/NPU 上活下来？** 答案是三件套：① 为 CPU 而生的量化格式（Q4_K 这类"超块"结构）；② 吃满内存带宽的 SIMD 内核（AVX2/NEON）；③ 把特殊算子送上 NPU 的工具链（RKNN/NeuroPilot）。四周后你应能手算 Q4_K 的 4.5 bpw、手写 Q4_0 解量化、并在两块板上各跑通一个真实模型。

| 文档 | 对应周次 | 核心问题 |
| --- | --- | --- |
| [第 1 周：GGUF 格式与 llama.cpp 架构解剖](第1周-GGUF格式与llama.cpp架构解剖.md) | 第 1 周 | GGUF 二进制布局是什么？Q4_K 的 4.5 bpw 怎么手算？ |
| [第 2 周：CPU 量化内核与采样器](第2周-CPU量化内核与采样器.md) | 第 2 周 | AVX2/NEON 如何让 GEMV 吃满内存带宽？imatrix 为什么让 2-3 bit 可用？ |
| [第 3 周：异构卸载与 RKNN 部署实战](第3周-异构卸载与RKNN部署实战.md) | 第 3-4 周 | -ngl / RPC / ggml-backend 怎么把层拆到别的设备？语音模型怎么上 RK3588？ |
| [第 4 周：RKNN 自定义算子与 MTK-NeuroPilot](第4周-RKNN自定义算子与MTK-NeuroPilot.md) | 第 5 周 | NPU 不支持的算子怎么办？CPU 自定义算子 + NPU 主体怎么混跑？ |
| [第 5-6 周：端侧推理全家桶整合与选型报告复盘](第5-6周-端侧推理全家桶整合与选型报告复盘.md) | 第 6 周 | CPU(GGML)/NPU(RKNN)/APU(NeuroPilot) 怎么选？功耗与热管理的天花板在哪？ |

## 贯穿全阶段的一条主线

$$
\text{HF 权重} \xrightarrow{\text{GGUF/量化}} \text{CPU 可执行的块格式} \xrightarrow{\text{SIMD 内核}} \text{tokens/s} \xrightarrow{\text{NPU 工具链}} \text{端侧实时}
$$

第 1-2 周把 llama.cpp 的"格式 + 内核"拆透（x86 与 ARM 都能打的原因），第 3 周把模型送上你手上的异构硬件（含语音方向成果的板端首秀），第 4 周解决 NPU 的算子缺口与第二块板，第 5-6 周把四平台数据汇总成《端侧 LLM 部署选型报告》——它同时是语音结课项目的技术决策书。

## 前置依赖与环境

- Stage 2 的量化知识（per-group、码本）直接复用：Q4_K 就是"超块版 per-group"；
- 本地环境：x86 机器（GCC/Clang + cmake）、RK3588 板（Debian/Ubuntu + rknn-toolkit2）、MTK 板（NeuroPilot SDK）；
- 依赖：`git`、`cmake`、`python3`、`pip install gguf numpy`；llama.cpp 源码编译（纯 CPU 即可：`cmake -B build && cmake --build build -j`）。
