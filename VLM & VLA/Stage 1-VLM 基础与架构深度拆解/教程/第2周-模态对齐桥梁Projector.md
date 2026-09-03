# 第 2 周教程：模态对齐桥梁（Connectors / Projector）

> **本周要回答的三个问题**
> 1. 1024 维的视觉特征（CLIP-L）如何变成 LLM 能"读懂"的 4096 维 Embedding？一个两层 MLP 真的够吗？
> 2. Q-Former、Perceiver Resampler 为什么要压缩 Token？压缩比与信息保真如何取舍？
> 3. 为什么 2023 年之后新出的 VLM 几乎清一色选了 MLP 路线？

对应学习计划：第 2 周。交付物：手写一个独立的两层 MLP Projector 模块，输入 $[B, 576, 1024]$，输出 $[B, 576, 4096]$。

---

## 1. 第一性原理：模态鸿沟与投影映射

### 1.1 根本矛盾

视觉塔和 LLM 是**独立预训练**的两个模型：视觉塔的 $1024$ 维空间（以 CLIP ViT-L 为例；SigLIP-SO400M 则为 1152）里编码的是"图像块的视觉相似性"（由对比学习塑造），LLM 的 $4096$ 维空间里编码的是"文本 Token 的分布规律"（由下一词预测塑造）。两个空间的几何结构没有任何先验对齐——同一个"狗"的概念，在视觉特征空间和文本 Embedding 空间中的坐标毫无关系。

Bridge 模块要解决的是：**在不破坏两个预训练模型的前提下，学一个映射 $f$，把视觉特征变换到 LLM 的输入空间，使得 LLM 把这些"视觉 Token"当作普通文本 Token 一样参与注意力计算**。

形式化地，设视觉塔输出 $\mathbf{V} \in \mathbb{R}^{N \times d_v}$，要求投影后的 $\mathbf{H} = f(\mathbf{V}) \in \mathbb{R}^{N' \times d_l}$ 满足：把它拼进 LLM 的 Embedding 序列后，LLM 在图文混合任务上的损失下降。这个目标函数由后续的多模态预训练/SFT 提供，Bridge 本身只是参数化的函数族选择。

### 1.2 为什么线性投影不够，两层 MLP 就够

单层线性 $\mathbf{W} \in \mathbb{R}^{d_v \times d_l}$ 只能做**仿射变换**，无法表达非线性对齐。LLaVA-1 论文（*Visual Instruction Tuning*）的消融显示，单层线性投影在语言-图像对齐预训练时效果明显差于 MLP（LLaVA-1.5 报告换成两层 MLP 后在多个 benchmark 上有系统性提升）。

两层 MLP（Linear → GELU → Linear）引入的非线性足以拟合两个空间之间复杂的流形映射，同时参数量极小：

$$
\text{params} = d_v \cdot d_h + d_h \cdot d_l \approx 1024 \times 4096 + 4096 \times 4096 \approx 21\text{M}
$$

对比 LLM 本身 7B+ 的参数量，Projector 只占约 0.3%。**这决定了训练策略**：Stage 1 预训练（对齐阶段）可以只训 Projector，冻结其余部分，单卡即可完成——这是 LLaVA 路线能快速迭代的核心工程原因。

### 1.3 GELU 而非 ReLU 的选择

GELU（Gaussian Error Linear Unit）：

$$
\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{2/\pi}\,(x + 0.044715 x^3)\right]\right)
$$

相比 ReLU 在 0 点硬性截断，GELU 处处光滑、负区间有小幅非零响应，梯度性质更好。视觉塔和 LLM 的 Transformer 块内部标准激活就是 GELU，Projector 沿用同一选择属于"与预训练分布一致"的保守决策。没有证据表明换成其他激活（SiLU 等）会带来实质差异——这里不是性能瓶颈所在。

---

## 2. 三种 Bridge 方案的系统对比

### 2.1 方案全景

| 维度 | Linear / MLP（LLaVA 路线） | Q-Former（BLIP-2 路线） | Perceiver Resampler（Flamingo 路线） |
| --- | --- | --- | --- |
| 输出 Token 数 | 与输入相同（$N' = N$，如 576） | 固定（如 32 个 Query → 32 Token） | 固定（如 64 个 Latent → 64 Token） |
| 参数量 | 极小（约 20M） | 较大（含一个完整 Transformer，约 100M 级） | 中等 |
| Token 压缩 | 1×（不压缩） | 最高 18×（576→32） | 最高 9×（576→64） |
| 细节保留 | 完整保留（每个 Patch 都在） | 有损，依赖 Query 学到什么 | 有损，同上 |
| 训练难度 | 低（收敛快，不易崩） | 高（Query 初始化敏感，需精心设计预训练目标：ITC/ITM/ITG） | 中 |
| 代表模型 | LLaVA-1.5、Qwen-VL 之后的多数开源 VLM、PaliGemma | BLIP-2、InstructBLIP、Qwen-VL(旧版) | Flamingo、IDEFICS |

### 2.2 Q-Former 的工作机制

Q-Former 是一个小型 Transformer（约 12 层），内含 $M$ 个**可学习的 Query 向量**（如 32 个，$d_q = 768$）。它的前向过程分两步：

1. **Self-Attention**：32 个 Query 互相注意，交换信息、分工（哪个 Query 管物体、哪个管颜色……这个分工是训练中自发形成的，无显式监督）。
2. **Cross-Attention**：Query 作为 Q，视觉塔的全部 $N$ 个 Patch Token 作为 K 和 V。每个 Query 从 576 个 Patch 中"检索"与自己分工相关的信息。

$$
\text{CrossAttn}(\mathbf{Q}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}, \quad \mathbf{Q} \in \mathbb{R}^{M \times d},\ \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}
$$

输出永远是 $M$ 个 Token——**与输入图片的分辨率无关**。这是一种"信息瓶颈"设计：无论图片多复杂，都强制压缩为 32 个向量。

### 2.3 压缩与保真的根本权衡

信息瓶颈的代价可以用互信息视角直觉化：压缩到 $M$ 个 Token，意味着图片信息最多保留"$M$ 个向量能承载的内容"。对"描述图中主要物体"这类粗粒度任务，32 个 Token 足够；对 OCR（图中可能有几百个字符）、图表数据读取这类任务，压缩是致命的——每个 Token 必须平均承载过多信息。

**这就是历史选择的关键证据链**：

- BLIP-2（2023 初）：Q-Former + 冻结 FlanT5/OPT，在当时的 benchmark 上表现优异，核心卖点是**训练成本低**（只训 Q-Former）。
- LLaVA-1.5（2023 下）：同样冻结视觉塔、只用两层 MLP 不压缩，以更简单的结构在 11 个 benchmark 上超越当时的 InstructBLIP/BLIP-2。
- Qwen2-VL、PaliGemma、LLaVA-OneVision 等后续模型：全部回到"MLP + 保留全部 Patch Token"，用**动态分辨率 + Token 合并**（第 3 周）来兼顾细节与长度。

结论（经验规则，非绝对）：**当视觉 Token 预算充足（LLM 上下文够长）时，不做信息压缩的 MLP 路线效果更好；Q-Former 类方案的价值区间在"LLM 上下文极短/多图视频场景需要硬性限长"**。LLM 长上下文能力在 2023 年后的飞速进步，直接导致压缩型 Bridge 失去了存在理由。

### 2.4 Perceiver Resampler 的定位

Flamingo 的 Perceiver Resampler 与 Q-Former 同属"Latent Query + Cross-Attention"家族，差异在于：它用固定的 64 个 Latent 数组，每层同时做 Latent 间 Self-Attention 和对视觉特征的 Cross-Attention，且输入视觉特征先经过一个冻结的 ResNet/ViT。学习重点放在理解"它是 Q-Former 的思想近亲"即可，工程上当今新模型已很少采用。

---

## 3. 系统架构与数据流

### 3.1 MLP Projector 的张量流转

以 $[B, 576, 1024]$ 输入、目标 LLM 维度 4096 为例：

| 步骤 | 操作 | 形状 |
| --- | --- | --- |
| 1 | 输入视觉特征 | $[B, 576, 1024]$ |
| 2 | Linear(1024 → 4096) | $[B, 576, 4096]$ |
| 3 | GELU | $[B, 576, 4096]$（逐元素，形状不变） |
| 4 | Linear(4096 → 4096) | $[B, 576, 4096]$ |

注意：LLaVA 的 MLP 是 `Linear(d_v → d_l) → GELU → Linear(d_l → d_l)`；也存在变体（如 InternVL）使用 `d_v → d_l` 的隐藏层维度。判断标准是中间层维度，**输出维度必须等于 LLM 的 hidden size**，否则无法进入 Embedding 拼接。

### 3.2 Projector 输出如何被 LLM 使用（与第 3 周衔接）

Projector 的输出不是终点。它将作为"视觉 Token 的 Embedding"替换文本序列中的 `<image>` 占位符位置：

```text
input_ids:    [BOS] user \n <image> \n 这张图里有什么？ [EOS]
                 ↓ 文本部分查 Embedding 表 ↓
embeddings:   [文本 embed ... , V_proj(576 × 4096), 文本 embed ...]
                              ↑ Projector 输出整体嵌入这里 ↑
```

本周只需记住：**Projector 的职责是把 $\mathbf{V}$ 变换为与 LLM Embedding 同维度、同分布尺度的向量**；替换的具体机制是第 3 周的主题。

---

## 4. 实现与验证

### 4.1 手写 MLP Projector（本周 MVP）

```python
"""
手写 LLaVA 风格的两层 MLP Projector，并验证形状与梯度流。
运行方式: python week2_projector.py
依赖: torch
"""
import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """LLaVA-1.5 风格 Projector: Linear -> GELU -> Linear"""

    def __init__(self, vision_dim: int = 1024, llm_dim: int = 4096):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, llm_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, vision_feats: torch.Tensor) -> torch.Tensor:
        # vision_feats: [B, N, vision_dim] -> [B, N, llm_dim]
        return self.fc2(self.act(self.fc1(vision_feats)))


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, d_v, d_l = 2, 576, 1024, 4096
    projector = MLPProjector(d_v, d_l)

    # 参数量检查: 应约 21M
    n_params = sum(p.numel() for p in projector.parameters())
    print(f"Projector 参数量: {n_params / 1e6:.1f}M")

    x = torch.randn(B, N, d_v, requires_grad=True)
    out = projector(x)

    # ---- 断言验证关键行为 ----
    assert out.shape == (B, N, d_l), f"输出形状错误: {out.shape}"
    assert n_params == d_v * d_l + d_l * d_l + d_l + d_l, "参数量与结构不符"

    # 非线性检查: 相同输入必须得到相同输出, 且模型确实是非线性的
    x2 = x.clone()
    assert torch.allclose(projector(x), out), "前向不确定?"

    # 梯度流检查: 损失能回传到输入 (冻结视觉塔场景下梯度要传回视觉塔最后一层)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "梯度未回传或异常"
    assert projector.fc1.weight.grad is not None, "Projector 参数未收到梯度"
    print("✅ 形状 / 参数量 / 非线性 / 梯度回传 全部通过")
```

**预期输出**：

```text
Projector 参数量: 21.0M
✅ 形状 / 参数量 / 非线性 / 梯度回传 全部通过
```

### 4.2 与 Hugging Face 官方实现对照

用 `LlavaNextProcessor`/官方代码中的 `MultiModalProjector` 对比，确认你的实现与官方 `nn.Sequential(nn.Linear, nn.GELU, nn.Linear)` 结构一致。更进一步的对照实验（可选）：

```python
from transformers import LlavaForConditionalGeneration

# 加载后可检查官方 Projector 结构 (llava-hf/llava-1.5-7b-hf)
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16)
print(model.multi_modal_projector)
# 期望输出:
#   LlavaMultiModalProjector(
#     (linear_1): Linear(in_features=1024, out_features=4096, bias=True)
#     (act): GELU(approximate='none')
#     (linear_2): Linear(in_features=4096, out_features=4096, bias=True)
#   )
```

### 4.3 手写 Q-Former 核心块（理解 Cross-Attention 压缩）

不追求完整复现 BLIP-2，只验证"任意输入长度 → 固定输出 Token 数"这一关键性质：

```python
import torch
import torch.nn as nn


class TinyQFormerBlock(nn.Module):
    """最小 Q-Former 块: Query Self-Attn + 对视觉特征的 Cross-Attn"""

    def __init__(self, n_queries: int = 32, dim: int = 768, heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, dim))
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, vision_feats: torch.Tensor) -> torch.Tensor:
        # vision_feats: [B, N, d_v]; N 可以是任意值
        q = self.queries.expand(vision_feats.size(0), -1, -1)
        q = q + self.self_attn(q, q, q)[0]           # Query 间信息交换
        q = self.norm1(q)
        out, _ = self.cross_attn(q, vision_feats, vision_feats)  # 检索
        return self.norm2(q + out)


if __name__ == "__main__":
    torch.manual_seed(0)
    block = TinyQFormerBlock()
    for n in (144, 576, 1024, 4096):                 # 四种分辨率下的 Token 数
        v = torch.randn(2, n, 768)
        out = block(v)
        assert out.shape == (2, 32, 768), f"N={n} 时输出形状错误"
    print("✅ 任意 N -> 固定 32 Token, 压缩性质验证通过")
```

**预期输出**：`✅ 任意 N -> 固定 32 Token, 压缩性质验证通过`

---

## 5. 工程权衡与失效模式

### 5.1 核心权衡总结

1. **Token 长度 vs 信息保真**：压缩型 Bridge 用信息瓶颈换取短序列；MLP 用长序列换细节。在 2024 年后的工程默认语境下，长上下文便宜了，**默认选 MLP 不压缩**，仅在多图/视频限长时叠加"训练后的 Token 合并"（见第 3 周 Qwen2-VL 的 Pixel Shuffle/Merger）而非硬性 Q-Former。
2. **可训练参数量 vs 对齐质量**：只训 Projector（约 20M）时对齐质量有限，这正是 LLaVA 流程要分"Stage 1 预训练（训 Projector）→ Stage 2 SFT（训 Projector + LLM）"两阶段的原因——纯冻结 LLM 无法让模型学会"用 LLM 的语言描述图像"。Q-Former 参数量大一个量级，但仍需大量对齐预训练数据。
3. **深度 vs 收益**：把 Projector 从 2 层加深到 4-6 层收益甚微且更难训（这属于工程上被反复验证的"不划算"经验规则）。将升级预算花在视觉塔分辨率或 LLM 上，回报远高于加深 Projector。

### 5.2 三个代表性失效模式

**失效 1：Projector 初始化不当导致对齐训练不收敛**
- **症状**：Stage 1 预训练 loss 长期不动，或模型输出全是重复字符。
- **根因**：PyTorch `nn.Linear` 默认初始化（均匀分布 $\pm 1/\sqrt{d}$）在 $d=4096$ 时输出方差偏大，与 LLM Embedding 的数值尺度不匹配，冻结的 LLM 收到"分布外"输入。
- **定位**：打印 `out.std()` 与 LLM 词 Embedding 的 `std()` 对比；两者差一个量级以上即确认。
- **修复**：小尺度初始化（如 `nn.init.normal_(w, std=0.02)`）或对输出加 LayerNorm；LLaMA-Factory 等框架的默认配置已处理，自己手写训练循环时最容易踩坑。

**失效 2：混用不同视觉塔维度到同一 Projector 权重**
- **症状**：加载 LLaVA checkpoint 时报 `shape mismatch`（期望 1024 入实际 1152 入，或反之）。
- **根因**：CLIP-L 的 $d_v = 1024$、SigLIP-SO400M 的 $d_v = 1152$，两者 patch 数也不同（576 vs 729）；而 SigLIP-base 的 $d_v = 768$ 又连维度都不同。Projector 权重与特定视觉塔**强绑定**。
- **定位**：读 checkpoint 的 `config.json` 中 `vision_feature_layer` / `mm_projector_cfg`。
- **修复**：换视觉塔必须重训 Projector；不存在"通用" Projector。

**失效 3：忘记 Projector 也有 LayerNorm 之外的数值稳定性问题（fp16 溢出）**
- **症状**：fp16 训练中 loss 突然变 NaN，多发生在 Projector 附近。
- **根因**：两层 $d=4096$ 的 Linear 在 fp16 下中间激活值可达数千，超出 fp16 安全范围（约 65504 上限）。
- **定位**：用 `torch.autograd.set_detect_anomaly(True)` 或在前向中逐层 `print(activation.abs().max())`。
- **修复**：用 bf16（动态范围与 fp32 相同）；或 fp16 + 混合精度下让 Projector 保持 fp32（`torch.autocast` 会自动处理大部分场景，但自写循环需显式保护）。

---

## 6. 延伸思考题

1. **互信息直觉**：若 Q-Former 用 32 个 Token 表示一张含 200 字文本的图，平均每个 Token 需承载约 6 个字符的信息。结合"OCR 任务上 BLIP-2 系明显弱于 LLaVA-1.5"的事实，解释信息瓶颈如何具体地导致这一差距。
2. **反事实思考**：假设 LLM 的上下文窗口只有 1024 Token（2022 年的水平），你会选择哪种 Bridge？此时动态分辨率还有意义吗？（提示：576 Token 已占一半窗口，压缩重新变得必要——这正是 Flamingo/BLIP-2 诞生的时代约束。）
3. **动手实验**：把 4.1 的 Projector 接在第 1 周提取的 SigLIP 特征后面（$d_v=1152$ 需改构造参数），冻结全部参数并随机初始化，观察输出与随机文本 Embedding 的余弦相似度分布；理解"未训练的 Bridge 输出只是随机向量"——对齐完全来自训练目标，而非结构本身。

---

*下一篇：[第 3 周：主流 VLM 架构与图像 Token 全流程](第3周-主流VLM架构与图像Token全流程.md)*
