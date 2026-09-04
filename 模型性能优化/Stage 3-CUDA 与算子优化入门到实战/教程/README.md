# Stage 3 教程索引：CUDA 与算子优化——从读懂到写出第一个高性能 Kernel

本目录是《学习计划》的配套教程，按周组织。每篇教程遵循统一框架：**第一性原理与数学本质 → 系统架构与数据流 → 工程权衡与失效模式 → 实现与验证（完整可编译代码 + nvcc 命令）→ 延伸思考题（含解析）→ 交付清单**。

> 📌 **学习者适配声明**：你 Python 为主、C++ 只能读懂、没写过 CUDA——本阶段刻意采用**「先读懂再仿写」的爬坡路径**：所有 CUDA 代码都给到逐行中文注释，先读懂骨架、改参数跑通，再仿写变体。不要求你成为 C++ 高手，但要求六周后**亲手写出 5 个能跑、能测、能对比性能的 kernel**。

> 📌 **算力安排**：本地无 GPU，全部 kernel 编写与压测在云 GPU 上进行（建议按小时租 RTX 4090/A10G 级别即可，学习阶段不需要 H100）。每个实验写好启动脚本，用完即释放。

> 本阶段的主线问题只有一个：**同一个计算，为什么换个写法就能快 10 倍？** 答案永远在访存层级与并行组织里：Roofline 决定天花板，合并访存与共享内存决定你能吃到天花板的几成。四周后你应能手算 7B decode 的带宽上限、写出三阶 MatMul、推导在线 softmax，并用 Triton 把这一切重写成 Python。

| 文档 | 对应周次 | 核心问题 |
| --- | --- | --- |
| [第 1 周：GPU 硬件模型与 CUDA 编程模型](第1周-GPU硬件模型与CUDA编程模型.md) | 第 1 周 | SM/warp/访存层级是什么？7B decode 每秒最多出多少 token？ |
| [第 2 周：MatMul 优化三阶](第2周-MatMul优化三阶.md) | 第 2 周 | 朴素 → 共享内存分块 → Tensor Core，每一级的性能从哪来？ |
| [第 3 周：融合算子与 FlashAttention 拆解](第3周-融合算子与FlashAttention拆解.md) | 第 3 周 | 在线 softmax 的递推式怎么推？fused RMSNorm 为什么快？ |
| [第 4 周：Triton 入门与 Profiling 驱动优化复盘](第4周-Triton入门与Profiling驱动优化复盘.md) | 第 4-6 周 | Triton 的 block 级模型省了什么？vLLM 的真实瓶颈在哪？ |

## 贯穿全阶段的一条主线

$$
\text{Roofline 定位瓶颈} \xrightarrow{\text{访存合并/共享内存}} \text{逼近带宽上限} \xrightarrow{\text{Tensor Core/融合}} \text{逼近算力上限} \xrightarrow{\text{Profiling}} \text{下一个瓶颈}
$$

第 1 周建立硬件心智模型与两个入门 kernel；第 2 周用 MatMul 三阶把"访存优化"练成肌肉记忆；第 3 周转向"融合"这个推理引擎的速度之源；第 4 周用 Triton 把开发效率换回来，并用 nsys/ncu 对真实推理系统做端到端诊断——INT4 GEMV kernel 会把 Stage 2 的量化知识与本阶段焊在一起。

## 前置依赖与环境

- Stage 2 的量化知识（per-group、反量化）会在第 4-6 周的 INT4 GEMV 中直接复用；
- 云 GPU 环境：CUDA Toolkit ≥ 12.0（`nvcc --version` 验证）、`nsight-compute`/`nsight-systems`（云上通常预装，CLI 为 `ncu`/`nsys`）；
- 依赖：`pip install torch triton vllm`（Triton 随 torch 安装）。
