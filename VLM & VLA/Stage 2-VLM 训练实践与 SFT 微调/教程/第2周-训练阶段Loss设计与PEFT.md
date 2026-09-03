# 第 2 周教程：训练阶段划分、Loss 设计与参数效率微调

> **本周要回答的三个问题**
> 1. 两阶段训练（对齐预训练 → 指令微调）各自在优化什么？为什么不能一步到位？
> 2. Token 级交叉熵的梯度具体怎么流？为什么 Vision Token 不设 label？
> 3. LoRA 该挂在 LLM 的 Attention 上，还是也要挂到 Projector 和 Vision Tower？

对应学习计划：第 2 周。交付物：同一小批数据上的对照实验——实验 A（仅训 Projector）vs 实验 B（Projector + LLM LoRA），对比 Loss 降幅与输出连贯度。

---

## 1. 第一性原理：两个阶段在解两个不同的优化问题

### 1.1 根本矛盾

"让 LLM 看懂图"实际上是**两个性质不同的任务**：

1. **表示对齐**：把视觉特征空间平移旋转到 LLM 的输入空间——这是一个**几何/分布匹配**问题，改动量小，需要大量图文对（百万级）来覆盖视觉语义的多样性，但对 LLM 本身的知识零要求。
2. **指令泛化**：教会模型"看图回答问题"的行为范式——这是一个**行为学习**问题，需要的是高质量的指令-回答对（几万条即可），且要求 LLM 保留其语言能力与知识。

用一个优化目标同时解两个任务，会互相拖累：指令数据的视觉多样性不够支撑对齐，对齐数据的语言质量不足以支撑指令泛化。**两阶段划分的本质是把一个病态的联合优化问题拆成两个良态的子问题。**

### 1.2 两个阶段的优化问题形式化

**Stage 1：Feature Alignment（对齐预训练）**

$$
\min_{\theta_{\text{proj}}} \ \mathbb{E}_{(I, T)} \left[ -\sum_{i} \log P_{\theta_{\text{llm}}}\left( x_i \mid x_{<i}, f_{\text{proj}}(g(I)) ; \theta_{\text{llm}} \right) \right]
$$

可训练参数只有 Projector 的 $\theta_{\text{proj}}$（约 20M）。Vision Tower $g$ 与 LLM $\theta_{\text{llm}}$ 完全冻结。数据形态通常是**密集描述式 caption**（LLaVA 用 GPT 生成的详细图注，558K 条）而非问答——目的是让 Projector 学会把视觉特征变成"LLM 读了会觉得通顺"的向量，caption 这种连续描述文本比问答更适合度量对齐质量。

**Stage 2：Visual Instruction Tuning（SFT）**

$$
\min_{\theta_{\text{proj}},\ \Delta\theta_{\text{llm}}} \ \mathbb{E}_{(I, Q, A)} \left[ -\sum_{i \in A} \log P\left( a_i \mid q, x_{<i}, g(I) \right) \right]
$$

解冻 Projector + LLM（全量或 LoRA 的 $\Delta\theta_{\text{llm}}$），Vision Tower 通常继续冻结。注意 loss 只对回答 Token $i \in A$ 求和（第 1 周的掩码机制）。数据形态是多任务指令集（对话、VQA、推理、定位混合）。

**为什么 Stage 1 不能省**：跳过对齐直接 SFT（相当于随机初始化 Projector 后立刻训指令），Projector 需要在"学会对齐"和"拟合指令行为"之间分摊有限的梯度信号，小数据下两头都做不好。LLaVA 论文的消融显示：仅 558K caption 数据的 Stage 1 之后，即使不经过大规模 SFT，模型已能产生连贯的图像描述——对齐是"听懂语言"的前提，指令是"学会应答"的叠加。

### 1.3 Token 级交叉熵与梯度流

SFT 的损失在 Token 级展开（结合第 1 周的掩码）：

$$
\mathcal{L} = -\frac{1}{|A|} \sum_{i \in A} \log \frac{\exp(h_i^\top W_{\text{lm}}[y_i])}{\sum_{v \in V} \exp(h_i^\top W_{\text{lm}}[v])}
$$

其中 $h_i$ 是第 $i$ 位置的 LLM 隐状态，$W_{\text{lm}}$ 是 LM head（词表投影），$A$ 是回答 Token 位置集合，归一化在**整个词表**上进行。

**梯度回传路径**（这正是"训练谁"的物理含义）：

$$
\mathcal{L} \to W_{\text{lm}} \to \text{LLM 各层} \to \text{视觉 Token 位置的隐状态} \to \text{Projector} \to \text{Vision Tower}
$$

三个推论：

1. **Projector 有梯度，当且仅当回答的预测依赖了视觉信息**。若一个样本的回答纯靠语言先验（如"天空是什么颜色"不需要看图），梯度对 Projector 的贡献接近零——这解释了为什么 Stage 1 要用"必须看图才能写对"的密集 caption。
2. **Vision Token 位置的 label 为 -100，不影响 Projector 接收梯度**：loss 在回答位置计算，但回答位置的注意力学虑了视觉 Token，梯度沿注意力路径流回视觉 embedding，再到 Projector。**"不算 loss"不等于"不传梯度"**——这两件事经常被混淆。
3. **Vision Tower 是否有梯度，取决于它是否被冻结**：冻结时 `requires_grad=False`，梯度到它的输入 embedding 就停了；解冻时继续向 ViT 内部回传。

### 1.4 为什么 Vision Token 默认不算 Loss，何时开

**默认不算的三层原因**：

1. **任务定义**：SFT 的目标是 $P(\text{文本} \mid \text{图文上下文})$，视觉 Token 是条件不是目标。模型的自回归解码只生成文本 Token，视觉 Token 永远不会出现在输出分布的采样空间里。
2. **预训练与 SFT 的差异**（自测清单考点）：纯文本 LLM 预训练对**所有** Token 计算 loss（next token prediction），因为整条序列都是"语言"，模型需要学会生成语言的每个部分。VLM 的 SFT 中，视觉 Token 是**外来的条件信号**——它不来自词表、无法被 LM head 参数化、也没有"预测下一个视觉 patch"的任务需求。
3. **信号稀释**：一张图的视觉 Token 数（数百）常超过回答长度（数十）。若参与 loss，交叉熵会被"预测视觉 Token"这种低质量目标淹没，文本能力反而退化。

**什么时候会开视觉 Token 的 loss**（加分量论点）：

- **视觉生成模型**（如统一理解+生成的 Chameleon/Emu 系）：输出侧本身包含图像 Token，自然参与 loss——那是另一个任务范式；
- **MEGAFLOW 式的视觉-文本联合预训练**：某些多模态预训练会对视觉 Token 做辅助预测任务（如 patch 顺序、掩码重建），作为表征学习的辅助目标；
- **SFT 阶段几乎从不开启**——工程上的默认共识。

---

## 2. 参数效率微调：显存账本与 LoRA 挂载位置

### 2.1 三种方案的显存账本（7B 为例）

延续 Stage 1 的核算，这里补全"为什么"的每一项：

| 组成项 | 公式 | Full FT | LoRA | QLoRA |
| --- | --- | --- | --- | --- |
| 权重 | 参数量 × 精度字节 | 14 GB (fp16) | 14 GB (fp16) | **3.5 GB (NF4)** |
| 权重梯度 | 同权重精度 | 14 GB | ≈ 0（冻结） | ≈ 0 |
| Adam 一阶矩 $m$ | 参数量 × 4 B (fp32) | 28 GB | 适配器量级 | 适配器量级 |
| Adam 二阶矩 $v$ | 参数量 × 4 B (fp32) | 28 GB | 适配器量级 | 适配器量级 |
| **合计（不含激活）** | | **84 GB** | **≈ 14.5 GB** | **≈ 4 GB** |

要点：

- **优化器状态是大头**：AdamW 每个可训练参数要维护 $m, v$ 两个 fp32 副本（8 字节/参数）。84GB 里优化器占了 56GB。这解释了为什么"减少可训练参数"（LoRA）比"压缩权重精度"（量化）对显存更敏感——两者兼得就是 QLoRA。
- **梯度精度**：混合精度训练下梯度通常与权重同精度（fp16/bf16），但 LoRA 场景下常直接以 fp32 维护 A/B 矩阵（量小，不在乎），数值更稳。
- **激活显存另算**：与 batch × 序列长 × 隐藏维 × 层数成正比，gradient checkpointing 可将其压到约 $\sqrt{L_{\text{depth}}}$ 量级（第 4 周展开）。

### 2.2 LoRA 挂载位置的系统分析

学习计划提出的思考题值得展开成一个决策表。LoRA 注射点的选择本质是问：**"这个任务需要的权重改动，分布在哪些模块的低秩子空间里？"**

| 挂载位置 | 参数量 | 效果 | 何时选 |
| --- | --- | --- | --- |
| 仅 LLM Attention 的 q_proj, v_proj（原版 LoRA 默认） | 最小 | 指令跟随足够 | 数据 <1k，任务偏语言风格 |
| **LLM 全部 Linear（`lora_target: all`，含 MLP up/down/gate）** | 中 | **SFT 标准选择** | 绝大多数 VLM SFT |
| + Projector | 极小增量 | 对齐强化 | 视觉任务定制强（如坐标输出） |
| + Vision Tower 顶层 | 中 | 视觉表征适配 | 域差距大（医学/遥感影像） |

三个关键论断：

1. **LLM 的 MLP 值得挂**：早期 LoRA 论文只挂 q/v 是在 GLUE 小任务上的结论；后续工作（如 QLoRA 论文的消融）显示对 SFT 类任务，挂上 MLP 的收益显著。现代框架默认 `lora_target: all` 不是偷懒，是消融后的共识。
2. **Projector 要不要挂**：SFT 阶段 Projector 本来就是解冻全训的（20M 参数直接全量更新，不值得用 LoRA 包裹）。"给 Projector 挂 LoRA"是个伪需求——除非你的 Pipeline 把 Projector 也冻结了（某些只调 LLM 的场景，此时挂上 LoRA 以小幅校准对齐）。
3. **Vision Tower 的默认答案是不挂**（Stage 1 第 6 周已论证冻结理由），例外是**域差距大**时挂顶部若干层的 LoRA——注意是"顶层"，因为 ViT 浅层是通用的边缘/纹理特征，深层才是语义特征；域适配只需要动语义层。

### 2.3 学习率的适配原则

$$
\text{lr}_{\text{LoRA}} \approx 10 \times \text{lr}_{\text{Full}} \approx 100 \times \text{lr}_{\text{预训练}}
$$

- 全量 SFT：$1 \sim 2 \times 10^{-5}$；LoRA：$1 \times 10^{-4}$；QLoRA 论文发现 4-bit 下可稍大（$1 \sim 2 \times 10^{-4}$）。
- 不同模块可异构：Projector 是从零/随机初始化的模块（若 Stage 1 未做），需要比 LoRA 更大的 lr（$1 \times 10^{-4} \sim 10^{-3}$）；LLM LoRA 用 $10^{-4}$。LLaMA-Factory 支持 `learning_rate` 分组配置，混冻训练时建议分组。

---

## 3. 实现与验证：对照实验 A/B 设计

### 3.1 实验矩阵

同一数据集（建议 200~500 条图文指令，构造方法见第 1 周教程）、同一超参（lr $10^{-4}$、3 epoch、bf16、batch 8），仅改可训练参数集合：

| 实验 | 可训练参数 | 实现方式（LLaMA-Factory） |
| --- | --- | --- |
| A：仅 Projector | Projector（约 20M） | `finetuning_type: freeze` + `freeze_trainable_modules: multi_modal_projector` + `freeze_vision_tower: true` |
| B：Projector + LLM LoRA | Projector + LoRA(r=8)（合计约 60M） | `finetuning_type: lora` + `lora_target: all`（Projector 默认解冻） |

### 3.2 观测指标与判读

**训练动力学对比**（记录到 WandB）：

| 指标 | 实验 A 预期 | 实验 B 预期 | 原因 |
| --- | --- | --- | --- |
| 初始 loss | 明显偏高（如 3~5） | 相近 | Projector 未对齐时 LLM 读到"乱码"视觉 Token |
| loss 下降斜率 | 前 100 步快降后**迅速平台** | 持续下降到更低水位 | A 只有 20M 参数可调，拟合容量是瓶颈 |
| 收敛 loss | 停在较高水位（如 1.5+） | 0.5~0.8 | LLM 的语言适配能力在 B 中被释放 |
| 梯度范数 | 波动大（小参数量+大 lr） | 平稳 | 参数越多，优化景观越平滑 |

**推理输出对比**（同 5 张测试图、同 prompt）：

- **A 的典型表现**：输出"通顺但空洞"——语法正确、风格正常（LLM 没动，语言能力完好），但对图的描述泛泛（"一张图片，上面有一些东西"），细节问题（数量、位置、颜色）常答错。**Projector 独训只能让 LLM"大概感觉到图里有什么"，改变不了 LLM 的应答行为。**
- **B 的典型表现**：输出"具体且可辨"——能说出物体数量/颜色/空间关系，回答风格向训练数据的规范收敛。代价是需要监控域外能力（灾难性遗忘，第 3 周展开）。

### 3.3 最小验证脚本：确认两实验的可训练参数边界

在启动昂贵的真实训练前，先用 1 分钟验证参数冻结配置是否真的生效：

```python
"""
验证两种微调策略的可训练参数集合与显存差异。
运行方式: python stage2_week2_freeze_check.py
依赖: torch, transformers (仅加载 config, 不下权重)
"""
import torch
from transformers import AutoModelForVision2Seq, AutoConfig


def report(model, tag):
    trainable, total = 0, 0
    groups = {}
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            key = n.split(".")[1] if "." in n else n      # 顶层模块名
            groups[key] = groups.get(key, 0) + p.numel()
    print(f"[{tag}] 可训练 {trainable/1e6:.1f}M / 全部 {total/1e6:.0f}M "
          f"({trainable/total:.2%})  模块分布: { {k: f'{v/1e6:.1f}M' for k, v in groups.items()} }")
    return trainable


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# ---- 实验 A: 仅 Projector ----
model = AutoModelForVision2Seq.from_config(AutoConfig.from_pretrained(MODEL_ID))
for p in model.parameters():
    p.requires_grad_(False)
for n, p in model.named_parameters():
    if "visual" not in n and "model.layers" not in n and "lm_head" not in n:
        p.requires_grad_(True)     # 剩余即 Projector/Merger
ta = report(model, "A: 仅 Projector")

# ---- 实验 B: Projector + LLM LoRA (用 r=8 估算) ----
for p in model.parameters():
    p.requires_grad_(False)
for n, p in model.named_parameters():
    if "visual" not in n and "merger" in n or "mlp" in n and "visual" in n:
        p.requires_grad_(True)     # Projector 部分
# LoRA 参数量估算: 每个被注入的 Linear 增加两个 r×d 矩阵
r = 8
lora_params = 0
for name, module in model.named_modules():
    if "model.layers" in name and isinstance(module, torch.nn.Linear):
        d_in, d_out = module.in_features, module.out_features
        lora_params += r * (d_in + d_out)
print(f"[B: Projector + LoRA] LoRA 注入量估算 ≈ {lora_params/1e6:.1f}M "
      f"(r={r}, LLM 侧全部 Linear)")
assert ta < 30e6, "实验 A 的可训练参数应只有 Projector 量级 (~20M)"
print("✅ 两种策略的参数边界符合预期")
```

**预期输出**（估算值，用于规模核对）：

```text
[A: 仅 Projector] 可训练 ~23M / 全部 ~2210M (1.05%)  模块分布: {'visual.merger': ...}
[B: Projector + LoRA] LoRA 注入量估算 ≈ 42M (r=8, LLM 侧全部 Linear)
✅ 两种策略的参数边界符合预期
```

关键断言是 `ta < 30M`：如果实验 A 的可训练参数达到几十亿，说明冻结配置写错了（这是对照实验最常见的静默失败——A/B 实际训了同样的东西，对比结论作废）。**做任何消融实验前，先打印可训练参数清单**，这条纪律能省掉大部分无效 GPU 小时。

### 3.4 结论模板（写进你的实验笔记）

> 仅训 Projector（实验 A）：loss 快速平台、输出通顺但视觉细节差——它只解决"LLM 能读到视觉信息"，解决不了"LLM 会用视觉信息回答"。
> Projector + LLM LoRA（实验 B）：loss 更低、回答具体且风格对齐——指令行为由 LLM 侧的适配承载。
> 两阶段设计（Stage 1 = A 的大数据版 → Stage 2 = B）正是把两个正交的子任务分配给各自最经济的参数集合。

---

## 4. 工程权衡与失效模式

### 4.1 何时偏离标准两阶段

| 场景 | 策略调整 | 理由 |
| --- | --- | --- |
| 底座 VLM 已对齐（如 Qwen2-VL 官方权重） | **跳过 Stage 1**，直接 SFT | 别人已替你完成对齐，重做浪费且可能退步 |
| 视觉域差距大（医学/遥感） | Stage 1 升级：解冻 Vision Tower 顶部 + 大规模域内 caption | 通用视觉特征失效，需先适配表征 |
| 数据 <1k 且任务为风格/格式 | 仅实验 A 或轻量 LoRA，防遗忘 | 动 LLM 的收益低于遗忘风险 |
| 需要 OCR/坐标等精细输出 | Stage 2 中挂 MLP + 提高 r | 格式约束需要更高秩的权重改动空间 |

### 4.2 三个代表性失效模式

**失效 1：Stage 1 数据用了问答而非 caption，对齐质量差**
- **症状**：Stage 1 loss 收敛良好，但 Stage 2 初期 loss 异常高、视觉相关问题仍差。
- **根因**：短答案的问答数据监督信号太少（一条样本十几个 Token），Projector 拿到的对齐梯度不足；LLM 还学会了走捷径（靠语言先验答题，视觉通道形同虚设——"语言先验捷径"是多模态训练的经典陷阱，hallucination 的重要来源，Stage 3 会正面处理）。
- **定位**：做一个探针测试：输入纯噪声图片问"描述图片"，若输出依然流畅具体，说明模型在靠先验答题。
- **修复**：Stage 1 换密集 caption 数据（每条回答 50+ Token，且必须看图才能写对）。

**失效 2：LoRA 的 `modules_to_save` 漏配导致 Projector 没训**
- **症状**：实验 B 的效果与预期差距大，接近实验 A。
- **根因**：某些框架版本中 `finetuning_type: lora` 默认**冻结除 LoRA 外的一切**，Projector 若未被加入 `additional_trainable_parameters`，则实际只训了 LLM 适配器，视觉通道的适配没发生。
- **定位**：训练日志开头会打印 trainable params 数量——对照 3.3 节的估算值（60M 量级 vs 42M 量级），差一个 Projector 就能看出来。
- **修复**：显式配置 Projector 可训（LLaMA-Factory 的 `freeze_multi_modal_projector: false`）。

**失效 3：两阶段学习率没重置，Stage 2 沿用过大的 lr**
- **症状**：Stage 2 初期 loss 剧烈震荡甚至短暂上升。
- **根因**：Stage 1 常用较大 lr（$10^{-3}$ 量级，小参数量需要大步长），Stage 2 涉及 LLM 权重（全量或 LoRA），需要小一个量级以上的 lr。沿用旧配置等于对 7B 权重做激进更新。
- **定位**：对比两阶段的 lr 配置与 loss 曲线开局形态。
- **修复**：阶段切换时显式重置 lr 并保留 warmup；用 cosine scheduler 时确认 `warmup_ratio` 在新阶段重新生效。

---

## 5. 延伸思考题

1. **梯度流向推演**：假设把 Vision Token 的 label 打开（不设 -100），用 2.1 节的显存账本推演：可训练参数量不变的前提下，显存会变吗？梯度范数统计会发生什么变化？（提示：激活不变、loss 计算的中间量不变，显存几乎不变；但视觉 Token 的 loss 梯度会通过 Projector 反向涌入，Projector 梯度范数可能放大一个量级，需重调 lr——"loss 口径"不只是监督问题，还是优化动力学问题。）
2. **反事实实验设计**：如果让你设计实验证明"Stage 1 是必要的"，你会控制哪些变量？至少给出四组对照（提示：{Stage1 有/无} × {Stage1 数据量 1%/100%} × {评估域内/域外}，并注意两组实验要共享同一个 Stage 2 配置与随机种子）。
3. **秩的分配问题**：总 LoRA 参数预算固定为 50M，方案一是 LLM 全层挂 r=8，方案二是只挂 q/v 但 r=64。对"图表数据提取"任务，你赌哪个好？说明你的优化景观直觉（开放题：MLP 参与格式/知识型改动的证据 vs 高秩单模块的表达力，两者都有文献支持；关键是训练前想清楚任务改动的"分布维度"）。

---

*下一篇：[第 3 周：数据配方与多任务消融实验](第3周-数据配方与消融实验.md)*
