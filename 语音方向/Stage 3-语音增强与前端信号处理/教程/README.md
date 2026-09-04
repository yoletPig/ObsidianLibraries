# Stage 3 教程索引：语音增强与前端信号处理——降噪 / AEC / AGC / ANC

本目录是《学习计划》的配套教程，按周组织。每篇教程遵循统一框架：**第一性原理与数学本质 → 系统架构与数据流 → 工程权衡与失效模式 → 实现与验证（可运行代码）→ MVP 交付指导 → 延伸思考题（含解析）**。

> 本阶段的主线问题：**如何把一段"脏"音频变"干净"？** 你在 Stage 1 第 4 周手写的谱减法是古典基线，本阶段把它升级为深度降噪（DeepFilterNet），并补全实时通信的三大件：回声消除（AEC）、自动增益（AGC）、主动降噪（ANC），最后整合成工业标准 WebRTC APM 全链路。

| 文档 | 对应周次 | 核心问题 |
| --- | --- | --- |
| [第 1 周：噪声模型与经典增强](第1周-噪声模型与经典增强方法.md) | 第 1 周 | 维纳滤波的最优增益怎么推？先验/后验信噪比如何估计？ |
| [第 2 周：AI 深度降噪——DeepFilterNet](第2周-AI深度降噪DeepFilterNet.md) | 第 2 周 | Deep Filtering 为什么比纯掩码好？实时约束怎么做到？ |
| [第 3 周：回声消除（AEC）](第3周-回声消除AEC.md) | 第 3 周 | 自适应滤波器如何估计回声路径？双讲时为何要冻结？ |
| [第 4 周：AGC、VAD 与主动降噪（ANC）](第4周-AGC-VAD与主动降噪ANC.md) | 第 4 周 | AGC 如何追踪目标响度？FXLMS 为什么需要次级路径估计？ |
| [第 5-6 周：WebRTC APM 全链路整合](第5-6周-WebRTC-APM全链路整合与复盘.md) | 第 5-6 周 | AEC/NS/AGC/VAD 的调用顺序为何不能乱？前端对 ASR 提升多少？ |

## 贯穿全阶段的一条主线

$$
\text{麦克风原始信号} \xrightarrow{\text{AEC 去回声}} \xrightarrow{\text{NS 降噪}} \xrightarrow{\text{AGC 稳音量}} \xrightarrow{\text{VAD 判活}} \text{干净语音} \xrightarrow{\text{ASR}}
$$

**顺序是硬约束**：AEC 必须在 NS 之前（降噪会破坏回声路径的线性结构），这个顺序问题是本周自测的工程直觉题。

## 前置依赖

- Stage 1 第 4 周的 `mix_at_snr()`、`spectral_subtraction()`、`evaluate()`（PESQ/STOI）会被直接复用；
- 依赖：`pip install deepfilternet speechbrain asteroid onnxruntime`，以及 WebRTC APM 的 Python 绑定。
