# Stage 1 教程索引：数字音频与信号处理基础

本目录是《学习计划》的配套教程，按周组织。每篇教程遵循统一框架：**第一性原理与数学本质 → 系统架构与数据流 → 工程权衡与失效模式 → 实现与验证（可运行代码）→ MVP 交付指导 → 延伸思考题（含解析）**。

> 你是零基础（不了解采样定理、STFT、Mel 滤波器组），本阶段是整个语音方向的地基。教程刻意从"声音是什么"讲起，每个概念都给出推导 + 直觉 + 代码三重解释。**强烈建议每周的代码亲手敲一遍**，而不是只读。

| 文档 | 对应周次 | 核心问题 |
| --- | --- | --- |
| [第 1 周：声音数字化——采样、量化与波形表示](第1周-声音数字化采样与量化波形表示.md) | 第 1 周 | 连续声波如何变成一串离散数字？为什么欠采样会混叠？ |
| [第 2 周：频域分析——FFT 与短时傅里叶变换](第2周-频域分析FFT与短时傅里叶变换.md) | 第 2 周 | 波形如何变成频谱图？n_fft/窗长/帧移如何互相制约？ |
| [第 3 周：听觉特征——Mel 滤波器组、MFCC 与 Fbank](第3周-听觉特征Mel滤波器组MFCC与Fbank.md) | 第 3 周 | 为什么 ASR 用 Mel 域？现代模型为何弃用 MFCC？ |
| [第 4 周：数字滤波器与信噪比——增强方向入场券](第4周-数字滤波器与信噪比增强入场券.md) | 第 4 周 | 谱减法怎么去噪？如何按指定 SNR 混合噪声？ |
| [第 5-6 周：工具箱整合与 AudioPipeline 构建](第5-6周-工具箱整合与AudioPipeline构建.md) | 第 5-6 周 | 如何用 torchaudio 搭一条可复用的音频流水线？ |

## 贯穿全阶段的一条主线

$$
\text{连续声波} \xrightarrow{\text{采样}} \text{离散序列} \xrightarrow{\text{STFT}} \text{频谱图} \xrightarrow{\text{Mel 滤波 + log}} \text{听觉特征} \xrightarrow{\text{模型}} \cdots
$$

Stage 1 只走到"听觉特征"这一步——把它喂给模型是 Stage 2 起的事。读完五篇后，你应该能**徒手推导一段音频从波形到 80 维 Log-Mel 特征每一步的 Shape 变化**，这正是第 6 周自测的验收标准。

## 环境准备

所有代码只依赖 `numpy`、`scipy`、`librosa`、`soundfile`、`matplotlib`。建议建一个虚拟环境：

```bash
python -m venv audio_env && source audio_env/bin/activate
pip install numpy scipy librosa soundfile matplotlib torchaudio
```

每篇教程的代码块都标注了预期输出，跑通后与预期不符时先检查采样率与数组维度，90% 的问题出在这两处。
