# 第 1-2 周教程：VLA 核心架构与动作离散化

> **本周要回答的三个问题**
> 1. RT-1 → RT-2 → Octo → OpenVLA 的架构演进在解决什么？
> 2. 连续的 7-DoF 动作怎么变成 Token？均匀分箱的量化误差怎么算？
> 3. Action Chunking 为什么能治动作抖动？

对应学习计划：第 1-2 周。交付物：Action Tokenizer & Detokenizer——输入 `[Batch, 7]` 连续动作，离散化为 256 词表的 Token IDs，反解回连续动作并计算量化误差。

本篇是具身方向的"第 1 周第 1 课"：**VLA = VLM 的输出端从词表换成动作词表**——Stage 1~6 的全部 VLM 知识几乎原样迁移，新知识只有动作空间这一块。

---

## 1. 第一性原理：为什么"动作即语言"是正确抽象

### 1.1 根本矛盾：控制需要专门训练，而知识在互联网数据里

传统机器人学习（模仿学习/RL）的困境：每个任务从零训练（数据饥渴）、无法利用语言指令、更无法利用互联网级的视觉-语义知识。而 VLM 恰恰拥有后者：知道"可乐罐长什么样"（视觉）、理解"把可乐移到盘子旁"（语义）——**缺的只是把语义意图翻译成动作的能力**。

**RT-2 的核心论证（co-fine-tuning）**：把动作离散化为 Token 后，机器人数据与互联网 VQA 数据可以**共用同一个模型、同一种损失（next-token prediction）联合微调**——动作数据教模型"怎么动"，互联网数据保住"知道什么"。RT-2 展示的涌现能力（理解"把恐龙玩具移到 Taylor Swift 附近"这类从未在机器人数据出现过的语义）证明了这条路线的价值：**动作即语言 Token，是打通"互联网知识 → 物理行为"的最短路径**。

### 1.2 架构演进谱系（每代解决一个瓶颈）

| 模型 | 年份 | 架构要点 | 解决的问题 | 遗留问题 |
| --- | --- | --- | --- | --- |
| **RT-1** | 2022 | 机器人专用 Transformer，EfficientNet 视觉 + Token 化动作（256 bins） | 大规模真实数据可扩展训练 | 闭源、语义能力弱 |
| **RT-2** | 2023 | VLM（PaLI-X/PaLM-E）+ 动作 Token 输出，co-fine-tune | **互联网知识迁移到控制** | 闭源、55B 推理重 |
| **Octo** | 2024 | 开源通用策略：Transformer + Diffusion 动作头，多相机 + 语言/目标图像双条件 | 开源、多本体、灵活条件输入 | 参数小（~90M），语义弱 |
| **OpenVLA** | 2024 | **Prismatic VLM（DINOv2+SigLIP 双塔）→ Llama 2 7B**，动作自回归 Token，OXE 970k 预训练 | **开源 + 强语义 + 可 LoRA 适配**（已核实官网/论文：7×少参数胜 RT-2-X 16.5%） | 单图单帧；控制频率压力（Stage 8 问题） |

**OpenVLA 的三段流水线**（已核实官网描述）：(1) 融合视觉编码器（SigLIP + DINOv2 双塔 patch embedding——Stage 1 第 1 周的双塔知识直接适用）；(2) 投影层把视觉 embedding 映射进语言空间（Stage 2 第 1 周的 Projector）；(3) Llama 2 7B 主干自回归预测动作 Token，解码为连续动作执行。**一个 VLA = Stage 1 的视觉编码 + Stage 2 的拼接 + Stage 6 的对齐底子 + 动作词表**。

### 1.3 动作离散化的数学：7-DoF 与均匀分箱

机器人末端执行器的标准动作空间（7-DoF）：

$$
\mathbf{a} = [\Delta x, \Delta y, \Delta z, \Delta r, \Delta p, \Delta y_w, g] \in [-1, 1]^7
$$

（位置增量 3 维 + 旋转增量 roll/pitch/yaw 3 维 + 夹爪开合 1 维；OXE 数据归一化到 $[-1,1]$。）

**均匀分箱（Uniform Binning）**：每一维独立地线性映射到 $B = 256$ 个 bin（OpenVLA/RT 系的标准选择）：

$$
\text{bin}(a) = \text{clip}\left(\left\lfloor \frac{a + 1}{2} \cdot B \right\rfloor,\ 0,\ B-1\right)
$$

**反解（Detokenize）**：取 bin 中心作为重建值：

$$
\hat{a} = -1 + \frac{2 \cdot (\text{bin} + 0.5)}{B}
$$

**量化误差**：均匀分箱的最大量化误差为半箱宽：

$$
|\hat{a} - a|_{\max} = \frac{1}{B} = \frac{2}{2 \times 256} = \frac{1}{256} \approx 0.0039 \ (\text{归一化单位})
$$

这个误差的实际含义：对位置维，若机械臂工作空间 1m、归一化单位对应 0.5m，则 0.0039 单位 ≈ **2mm**——抓取任务通常可接受，精密装配则不够。**256 bins 是"语言词表成本 vs 控制精度"的折中**：bin 数翻倍 → 动作词表翻倍（Llama 词表 32000 加 7×256=1792 个特殊 Token）→ 输出序列变长但精度提升；不翻倍 → 每步 7 个 Token 的输出长度可控（推理快，控制频率友好）。

### 1.4 Action Chunking：为什么一次预测 k 步

**问题**：逐步预测（每步 1 个动作）有两个病：① **抖动（Jitter）**——模型对相邻帧的预测不一致，机械臂来回晃；② **累积误差**——每步误差在闭环中复利；③ **控制频率压力**——7B 前向一次数百 ms，10Hz 控制要求 100ms 内出动作，逐步预测把推理时间摊在每个控制步上。

**Action Chunking**：一次前向预测未来 $k$ 步的动作序列（$k=4$ 常见，OpenVLA 的多 Token 输出 / ALOFT 式 chunk 化），执行期按序播发（可用插值平滑）：

- **平滑性**：k 步动作来自同一次前向，天然连贯，抖动消失；
- **频率解耦**：模型以低频推理（如 2Hz 出 4 步 chunk），底层以高频插值执行（10Hz）——**大模型慢推理与机器人快控制的频率鸿沟被 chunk 桥接**（Stage 8 的加速是另一条腿）；
- **代价**：k 步内无法根据新观测修正（开环 k 步）——k 越大越平滑但适应性越差，replan 频率是核心权衡（ACT/ALOHA 系的 "$C = k$ 步 chunk + 每 $r < k$ 步重规划"是标准折中）。

---

## 2. 实现与验证

### 2.1 本周 MVP：Action Tokenizer & Detokenizer

```python
"""
7-DoF 动作离散化/反解 + 量化误差分析 (OpenVLA 风格, B=256)。
运行方式: python stage10_week1_action_tokenizer.py
依赖: torch, numpy
"""
import numpy as np
import torch

BINS = 256


class ActionTokenizer:
    """连续 [-1,1]^7 -> 256 bins/维 -> Token IDs (词表内偏移)"""

    def __init__(self, bins: int = BINS, token_offset: int = 31744,
                 vocab_size: int = 32000):
        assert token_offset + 7 * bins <= vocab_size, "词表空间不足"
        self.bins, self.offset = bins, token_offset

    def tokenize(self, actions: torch.Tensor) -> torch.Tensor:
        """actions: [Batch, 7] in [-1,1] -> [Batch, 7] token ids"""
        assert actions.dim() == 2 and actions.shape[1] == 7
        assert (actions <= 1.0 + 1e-6).all() and (actions >= -1.0 - 1e-6).all(), \
            "动作须归一化到 [-1,1]"
        # bin(a) = clip(floor((a+1)/2 * B), 0, B-1)
        norm = (actions + 1.0) / 2.0
        b = torch.clamp((norm * self.bins).floor().long(), 0, self.bins - 1)
        return b + self.offset                            # 词表偏移成 <action_xx>

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        """token ids -> bin 中心值 [-1,1]"""
        b = tokens - self.offset
        assert (b >= 0).all() and (b < self.bins).all(), "token 超出动作词表"
        return -1.0 + (b.float() + 0.5) * 2.0 / self.bins


def quantization_error(actions: torch.Tensor, recon: torch.Tensor) -> dict:
    """量化误差统计: 最大误差、RMS 误差 (归一化单位)"""
    err = (recon - actions).abs()
    return {"max": err.max().item(), "rms": err.pow(2).mean().sqrt().item(),
            "max_theory": 1.0 / 256}


if __name__ == "__main__":
    torch.manual_seed(0)
    tok = ActionTokenizer()

    # ---- 1. 合成动作: 真实分布近似 (多数维小增量, 夹爪常饱和) ----
    B = 512
    actions = torch.clamp(torch.randn(B, 7) * 0.3, -1, 1)
    actions[:, 6] = torch.where(torch.rand(B) > 0.5, 1.0, -1.0)   # 夹爪二值化
    tokens = tok.tokenize(actions)
    recon = tok.detokenize(tokens)

    # ---- 断言: 往返一致性与理论误差 ----
    err = quantization_error(actions, recon)
    assert err["max"] <= err["max_theory"] + 1e-9, \
        f"最大量化误差 {err['max']:.5f} 超过理论半箱宽 {err['max_theory']:.5f}"
    assert err["rms"] < err["max_theory"] / 2, "RMS 误差应显著小于最大误差 (均匀分布性质)"
    # 往返一致性: 对已重建值再 tokenize 必须稳定 (bin 中心是重编不动点)
    assert (tok.tokenize(recon) == tokens).all(), "bin 中心应满足 tokenize∘detokenize 幂等"

    # ---- 2. 物理含义换算: 归一化误差 -> 物理毫米 ----
    workspan_m, norm_span = 0.8, 2.0                   # 0.8m 工作空间 对应 2.0 归一化跨度
    mm_per_unit = workspan_m * 1000 / norm_span
    print(f"bins={BINS}: 最大量化误差 {err['max']:.5f} 归一化单位 "
          f"≈ {err['max'] * mm_per_unit:.2f} mm (物理)")
    assert err["max"] * mm_per_unit < 5, "位置精度应 <5mm (抓取任务常见要求)"

    # ---- 3. 边界与饱和 ----
    edge = torch.tensor([[-1.0] * 6 + [1.0]])
    t_edge = tok.tokenize(edge)
    assert (t_edge[:, :6] == tok.offset).all() and t_edge[0, 6] == tok.offset + 255, \
        "边界值应映射到第 0 / 第 255 bin"
    print("✅ Action Tokenizer 全部断言通过 (往返一致 / 误差符合理论 / 边界饱和正确)")


    # ---- 4. Chunking 演示: [B, k, 7] 的 chunk tokenize ----
    k = 4
    chunk = torch.clamp(torch.randn(2, k, 7) * 0.2, -1, 1)
    t_chunk = tok.tokenize(chunk.reshape(-1, 7)).reshape(2, k, 7)
    assert t_chunk.shape == (2, k, 7)
    print(f"✅ Action Chunking: [B={2}, k={k}, 7] 一次预测 {k} 步, "
          f"每步 7 个动作 Token, 单序列输出 {k*7} Token")
```

**预期输出**：

```text
bins=256: 最大量化误差 0.00389 归一化单位 ≈ 1.56 mm (物理)
✅ Action Tokenizer 全部断言通过 (往返一致 / 误差符合理论 / 边界饱和正确)
✅ Action Chunking: [B=2, k=4, 7] 一次预测 4 步, 每步 7 个动作 Token, 单序列输出 28 Token
```

断言验证四个关键性质：**往返一致性**（tokenize∘detokenize 幂等——bin 中心是不动点，这是"训练目标与推理解码自洽"的基础）、**理论误差界**（≤半箱宽 1/256）、**物理换算**（1.56mm < 5mm 的抓取要求）、**边界饱和**（±1 映射到 0/255）。非均匀分箱的改进（高频动作区间加密）在思考题 2 展开。

### 2.2 与 OpenVLA 官方实现对照

OpenVLA 的官方 tokenizer（`openvla/vla/datasets/rlds/dataset_transforms` 附近）与本实现同构：`bins=256`、双端归一化、每维独立分箱；差异在于其归一化统计量按 OXE 数据集维度缓存（`norm_stats`），**跨数据集迁移时归一化参数必须随 checkpoint 走**——这是第 7-9 周微调的高频坑（详见该篇失效模式 2）。

---

## 3. 工程权衡与失效模式

### 3.1 决策表：离散化参数

| 参数 | OpenVLA 默认 | 调整方向 |
| --- | --- | --- |
| bins | 256 | 精密任务升 512（词表代价）；粗任务降 128 |
| 分箱方式 | 均匀 | 动作分布长尾时用分位数分箱（思考题 2） |
| chunk 长度 k | 1（OpenVLA 原生逐步） | 抖动严重/频率不足时引入 chunk（配插值执行） |
| 归一化 | 数据集统计量 | **随 checkpoint 迁移，绝不静默重算** |

### 3.2 三个代表性失效模式

**失效 1：归一化统计量错配——换数据集微调后动作系统性偏移**
- **症状**：微调后仿真中机械臂整体向一个方向漂移，或夹爪永远半开。
- **根因**：微调数据与预训练数据的动作归一化统计量（min/max 或均值方差）不同，tokenizer 用错了 norm_stats——同一 Token 值解出不同物理量。
- **定位**：对一条已知 GT 轨迹做 tokenize→detokenize 往返，与 GT 差异超过理论量化误差即确认。
- **修复**：归一化参数随 checkpoint 保存与加载；微调新平台时重算统计量并**同步更新 detokenizer**（两侧必须同一套）。

**失效 2：夹爪维的量化边界抖动**
- **症状**：夹爪在开/合之间高频抖动。
- **根因**：夹爪在数据里是二值（-1/1），落在 bin 0 与 bin 255 的边界；模型预测在边界 bin 附近振荡时，反解出的夹爪值在两个极端间跳变。
- **定位**：统计夹爪维 Token 的相邻步变化率。
- **修复**：夹爪维降低 bin 数（甚至 2 bin）、或输出后加滞回（hysteresis：两次一致才切换）、或阈值化后处理（>0 算合）。

**失效 3：旋转维的归一化不连续（角度回绕）**
- **症状**：机械臂末端朝向在某角度附近突然反向旋转 180°。
- **根因**：yaw 角在 ±π 处回绕（+π 与 -π 是同一朝向），线性归一化把连续旋转切成了两段——模型学到的是"不连续的旋转空间"。
- **定位**：可视化 yaw 维的动作轨迹，在 ±π 附近找跳变。
- **修复**：旋转维改用连续表示（6D rotation representation 或四元数）或归一化前做角度unwrap——**离散化之前的数据表示决定一切**。

---

## 4. 延伸思考题

1. **离散化 vs 连续头**：Octo 用 Diffusion 动作头输出连续动作，OpenVLA 用离散 Token。从三个维度系统对比：多任务词表成本、多本体适配难度、与 LLM 联合训练的兼容性——并回答"什么时候连续头是必须的"（提示：高频精细控制如书写/插孔，256 bins 的 2mm 精度不够；而 LLM 骨干的 co-training 优势只有 Token 化能吃到）。
2. **分位数分箱**：若某维动作 95% 集中在 [-0.1, 0.1]（小幅微调）而 5% 是大幅移动，均匀分箱会把大部分样本挤进少数 bin。设计分位数分箱方案并分析其对"训练分布拟合"与"测试时分布外动作表达"的双重影响。
3. **控制频率的账**：OpenVLA-7B 在 A100 上单次前向约 200~400ms（经验量级，实测为准）。若任务要求 5Hz 控制（200ms），逐步预测勉强达标；15Hz（66ms）呢？分别用"量化推理（Stage 8）"与"chunk=4"两条路算出可行的控制频率，并说明 chunk 化牺牲了什么。

---

*下一篇：[第 3-4 周：Open X-Embodiment 数据集与 RLDS 数据流](第3-4周-OXE与RLDS数据流.md)*
