# 第 4-5 周教程：SFT 机制与 LoRA 微调实战

> **本周要回答的三个问题**
> 1. LoRA 为什么能用 0.1% 的参数达到接近全量微调的效果？数学上发生了什么？
> 2. VLM 的 SFT 数据格式长什么样？Loss 是在所有 Token 上算的吗？
> 3. 如何用 LLaMA-Factory 在单卡上完成一次 Qwen2-VL 的 LoRA 微调并验证效果？

对应学习计划：第 4-5 周。交付物：200~500 条多模态数据集 + LLaMA-Factory 完成一次 LoRA/QLoRA 微调 + WandB Loss 曲线 + 微调前后推理对比。

---

## 1. 第一性原理：LoRA 的低秩假设

### 1.1 根本矛盾

全量微调一个 7B VLM 需要：权重 14GB（fp16）+ 梯度 14GB + Adam 优化器状态 56GB（fp32 的动量与二阶矩）≈ **84GB 起步**，再加激活值，单张消费级显卡完全不可能。适配器方案的动机：**微调带来的权重变化量 $\Delta W$ 的"本征维度"远低于 $W$ 本身**——任务适配不需要移动所有方向的权重，只需要在少数几个"任务子空间"里调整。

### 1.2 LoRA 的数学形式

对预训练权重 $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$，LoRA 不直接更新 $\mathbf{W}_0$，而是学一个**低秩分解**的增量：

$$
\mathbf{W} = \mathbf{W}_0 + \Delta \mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B}\mathbf{A}, \quad \mathbf{A} \in \mathbb{R}^{r \times k},\ \mathbf{B} \in \mathbb{R}^{d \times r},\ r \ll \min(d, k)
$$

前向传播变为：

$$
\mathbf{h} = \mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r} \mathbf{B}(\mathbf{A}\mathbf{x})
$$

关键设计点：

- **$\mathbf{A}$ 用高斯初始化，$\mathbf{B}$ 初始化为零** → 训练起点 $\Delta W = 0$，模型行为与预训练完全一致，不会一上来就破坏预训练能力。
- **缩放因子 $\alpha/r$**：$\alpha$ 固定（常取 $2r$），调 $r$ 时学习率尺度保持稳定。
- **梯度只流向 $\mathbf{A}, \mathbf{B}$**，$\mathbf{W}_0$ 冻结。可训练参数量从 $d \times k$ 降为 $r \times (d + k)$。以 $r=8$、$d=k=4096$ 为例：从 16.7M 降到 65K，约 **0.4%**。

### 1.3 为什么"低秩"是合理的假设

LoRA 论文（*LoRA: Low-Rank Adaptation of Large Language Models*）给出的证据：对微调前后的权重差 $\Delta W$ 做奇异值分解，其奇异值谱高度集中——前几个奇异方向承载了绝大部分变化。不同任务学到的 $\Delta W$ 的 top 奇异向量高度重叠，说明**任务适配共享一个低维子空间**。$r = 8$ 或 $r = 16$ 时，LoRA 在多数任务上能与全量微调打平；$r$ 过大反而过拟合。

经验规则：指令跟随类任务 $r=8{\sim}16$ 足够；领域风格迁移（如让它学会输出特定 JSON schema）$r=32$ 起步；涉及大量新知识注入时 LoRA 本身就不是好选择（应考虑继续预训练或 RAG）。

### 1.4 显存核算（为什么 QLoRA 能上单卡）

| 方案 | 权重 | 梯度 | 优化器 | 7B 模型合计（不含激活） |
| --- | --- | --- | --- | --- |
| 全量微调（AdamW） | fp16 14GB | fp16 14GB | fp32 56GB | ≈ 84GB |
| LoRA（冻结底座） | fp16 14GB | 仅适配器 <0.1GB | 仅适配器 <0.3GB | ≈ 14.5GB |
| QLoRA | **4-bit NF4 ≈ 3.5GB** | 仅适配器 | 仅适配器 | **≈ 4GB** |

QLoRA（*QLoRA: Efficient Finetuning of Quantized LLMs*）的三件套：4-bit NormalFloat（NF4）量化底座权重、双重量化（量化常数本身再量化）、分页优化器。计算仍在 bf16 下进行（4-bit 存储与 bf16 计算解耦），因此精度损失可控——QLoRA 论文显示在多数评测上与 16-bit LoRA 打平。

---

## 2. VLM SFT 的机制细节

### 2.1 数据格式：多轮对话 + 视觉占位

LLaMA-Factory 的 `sharegpt` 多模态格式示例（`dataset_info.json` 注册后使用）：

```json
{
  "conversations": [
    {"from": "human", "value": "<image>这张图里的动物是什么品种？"},
    {"from": "gpt", "value": "这是一只橘色的美国短毛猫，特征是银灰色虎斑纹。"}
  ],
  "images": ["images/cat_001.jpg"]
}
```

要点：

- `<image>` 占位符与 `images` 字段按顺序一一对应（多图时写多个 `<image>`）；
- 图像路径相对数据集根目录；
- `from: human/gpt` 定义轮次，系统提示可另配。

### 2.2 Loss Mask：只对回答部分计算损失

SFT 的核心机制：**把整条对话拼成一个序列，但只在 assistant 回答的 Token 上计算交叉熵**，用户输入与图片 Token 的 label 置为 -100（PyTorch `CrossEntropyLoss` 的 ignore_index）：

```text
序列:    [系统提示] [用户问题] [图片 Token ×576] [回答 Token] [EOS]
label:   -100      -100      -100               [回答 Token] [EOS]
                                            ↑ 只有这里算 loss
```

为什么要 mask？若对用户问题也算 loss，模型会浪费容量去学"如何复述问题"，且用户输入风格各异会引入噪声。视觉 Token 数量大（576+），若不 mask 会严重稀释有效梯度信号。

数据集构造时按样本长度分组打包（group by length）可减少 padding 浪费；`cutoff_len` 要预留视觉 Token 的空间（如文本只占 1024 但图片占 576，`cutoff_len` 至少要 2048）。

### 2.3 训练哪些模块：VLM 的冻结策略

| 策略 | 训练部分 | 显存 | 适用场景 |
| --- | --- | --- | --- |
| 全冻结 + 只训 Projector | Projector（约 20M） | 极低 | 对齐预训练阶段 |
| LoRA on LLM + 冻结视觉塔 | Projector + LLM 的 LoRA | 低（单卡 24GB 可跑 7B QLoRA） | **SFT 标准做法，本周实战采用** |
| 视觉塔也开 LoRA | 上者 + ViT LoRA | 中 | 视觉域差异大时（如医疗影像） |
| 全量微调 | 全部 | 极高（多机） | 大规模 SFT |

**为什么默认冻结视觉塔**（对应第 6 周自测题）：视觉塔参数量大（CLIP-L 约 300M），开训练会显著增加显存与优化器状态；更重要的是，SFT 数据量小（几百到几万条），更新视觉塔会**破坏其在大规模预训练中习得的通用视觉表征**，导致灾难性遗忘——模型在训练任务上过拟合、在泛化任务上变差。而 LoRA 的低秩更新只作用于语言侧的任务适配，视觉表征保持稳定。

---

## 3. 实战：LLaMA-Factory 完成 Qwen2-VL-2B LoRA 微调

### 3.1 环境与数据准备

```bash
# 依赖（建议 Python 3.10+，CUDA 12.x）
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# 数据集：准备 200~500 条 "图片+问答" 样本
# 目录结构:
#   data/my_vqa/
#     ├── images/*.jpg
#     └── train.json
```

`data/my_vqa/train.json` 格式（对应 2.1 节）：

```json
[
  {
    "conversations": [
      {"from": "human", "value": "<image>用一句话描述这张图。"},
      {"from": "gpt", "value": "一只金毛犬在草地上追飞盘。"}
    ],
    "images": ["images/0001.jpg"]
  }
]
```

在 `data/dataset_info.json` 注册：

```json
{
  "my_vqa": {
    "file_name": "my_vqa/train.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "images": "images"}
  }
}
```

**数据质量优先于数量**：500 条风格一致、答案规范的样本，效果远好于 5000 条杂乱样本。答案长度分布要均衡（全是短句会导致模型只会短句）。

### 3.2 训练配置（单卡 24GB 显存参考）

`qwen2vl_lora_sft.yaml`：

```yaml
### 模型
model_name_or_path: Qwen/Qwen2-VL-2B-Instruct
trust_remote_code: true

### 方法
stage: sft
do_train: true
finetuning_type: lora
lora_target: all          # 对所有 Linear 层注入 LoRA
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05

### 数据集
dataset: my_vqa
template: qwen2_vl
cutoff_len: 2048          # 必须容纳视觉 Token!
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 8

### 输出
output_dir: saves/qwen2vl-2b/lora-sft
logging_steps: 10
save_steps: 100
plot_loss: true
report_to: wandb          # WandB 记录

### 训练超参
per_device_train_batch_size: 1
gradient_accumulation_steps: 8    # 等效 batch = 8
learning_rate: 1.0e-4             # LoRA 用大学习率 (全量的 10~100 倍)
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
gradient_checkpointing: true      # 省激活显存的关键开关

### 显存不足时的降级选项 (QLoRA)
# quantization_bit: 4
# quantization_method: bitsandbytes
```

启动训练：

```bash
llamafactory-cli train qwen2vl_lora_sft.yaml
```

**显存参考**（经验值，实际随序列长度波动）：Qwen2-VL-2B + LoRA + bf16 + gradient checkpointing，batch=1、cutoff 2048，约需 12~16GB；加上 4-bit 量化可压到 8GB 内。7B 模型同配置约需 20~24GB，超限时开启 `quantization_bit: 4`。

### 3.3 训练超参的第一性解释

- **学习率 1e-4**：LoRA 参数是**从零初始化的新参数**（$\mathbf{B}=0$），不像全量微调是在预训练权重附近微移，所以可以用大一个数量级的学习率。用 1e-5 跑 LoRA 是新手最常见错误（表现为 loss 几乎不动）。
- **gradient_checkpointing**：用"反向时重算前向"换显存，激活显存降约一个量级，速度慢约 20~30%。24GB 卡跑 7B 的必要开关。
- **等效 batch size = 1 × 8 = 8**：小 batch + 梯度累积在 VLM 场景是常态（视觉 Token 使单样本显存波动大）。500 条样本 × 3 epoch = 1500 步 ÷ 8 ≈ 187 个优化步，loss 应在前 50 步内明显下降。

### 3.4 推理验证与前后对比

```bash
# 合并 LoRA 权重并导出
llamafactory-cli export --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
  --adapter_name_or_path saves/qwen2vl-2b/lora-sft \
  --template qwen2_vl --export_dir saves/qwen2vl-2b/merged

# 交互式推理对比
llamafactory-cli chat --model_name_or_path saves/qwen2vl-2b/merged \
  --template qwen2_vl
```

设计**可区分的对比集**：微调前后各跑一遍，每类 10 条。例如：

| 测试类型 | 示例问题 | 预期差异 |
| --- | --- | --- |
| 训练域内 | 用训练集同风格问题问新图 | 微调后输出风格/格式明显对齐 |
| 训练域外 | 完全无关的通用图片问题 | 两者都正常 → 无灾难性遗忘 |
| 格式遵循 | "用 JSON 格式回答图中物体" | 微调后格式符合度提升（若训练数据含 JSON 答案） |
| 对抗样本 | 训练中从未出现的极端构图 | 观察是否暴露过拟合 |

**MVP 验收标准**：域内回答肉眼可见地改善了（风格、格式或事实），域外能力未退化，且 WandB 上 loss 从约 1.x 收敛到 0.3~0.6 区间并趋平。

### 3.5 训练过程最小验证脚本

不依赖 LLaMA-Factory，验证 LoRA 模块本身的行为（理解 1.2 节的机制）：

```python
"""
最小 LoRA 实现 + 行为验证。
运行方式: python week4_lora_min.py
依赖: torch
"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():        # 冻结底座
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # 关键: B=0
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T


torch.manual_seed(0)
base = nn.Linear(4096, 4096)
torch.nn.init.normal_(base.weight, std=0.02)
layer = LoRALinear(base, r=8)

x = torch.randn(2, 4096)
y_before = layer(x).detach().clone()

# ---- 断言 1: 初始状态 B=0, LoRA 层输出与底座完全一致 ----
assert torch.allclose(y_before, base(x), atol=1e-6), "初始化后应等价于底座"

# ---- 断言 2: 只有 A/B 可训练 ----
trainable = [n for n, p in layer.named_parameters() if p.requires_grad]
assert trainable == ["A", "B"], f"可训练参数应为 A/B: {trainable}"

# ---- 断言 3: 训练一步后 delta 只来自低秩项 ----
opt = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=1e-3)
loss = layer(x).pow(2).mean()
loss.backward()
opt.step()
y_after = layer(x).detach()
delta = (y_after - y_before)
# delta 的秩不应超过 r=8 (低秩约束的数值验证)
delta_mat = delta.reshape(-1, 4096).float()
rank = torch.linalg.matrix_rank(delta_mat[:64]).max().item()
assert rank <= 8, f"更新量秩 {rank} 超过 r=8, LoRA 结构被破坏"
n_trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
print(f"✅ B=0 初始化等价 | 仅 A/B 可训练 ({n_trainable} 参数) | 更新秩 ≤ 8")
```

**预期输出**：`✅ B=0 初始化等价 | 仅 A/B 可训练 (65600 参数) | 更新秩 ≤ 8`。三个断言分别验证 LoRA 的三个关键行为：零起点安全、参数隔离、低秩更新。

---

## 4. 工程权衡与失效模式

### 4.1 超参权衡速查

| 参数 | 保守取值 | 激进取值 | 取舍 |
| --- | --- | --- | --- |
| `lora_rank` | 8 | 64 | 大 r 拟合力强但过拟合风险高、显存多 |
| `learning_rate` | 5e-5 | 2e-4 | 大 lr 收敛快但易震荡；小 lr 稳但可能欠拟合 |
| `epochs` | 2 | 5+ | 小数据多轮必过拟合，看 eval loss 拐点 |
| `cutoff_len` | 1024 | 4096 | 越长显存越高，VLM 记得预留视觉 Token |

### 4.2 四个代表性失效模式

**失效 1：Loss 不降（最常见）**
- **症状**：训练几十步，loss 停在初始值附近抖动。
- **根因排查顺序**：① 学习率过小（LoRA 用了全量的 1e-5）② 数据格式错，`<image>` 与 `images` 数量不匹配导致样本被静默丢弃 ③ `lora_target` 没有命中任何模块 ④ mask 配置错误导致有效 label 全是 -100。
- **定位**：先跑 `llamafactory-cli train ... --max_samples 10 --num_train_epochs 1` 十分钟冒烟测试；打印一条预处理后的样本确认 label 非全 -100。
- **修复**：按排查顺序逐项修正；lr 建议从 1e-4 起调。

**失效 2：GPU OOM 在训练中途而非开局**
- **症状**：前 100 步正常，遇到某张高分辨率大图突然 CUDA OOM。
- **根因**：动态分辨率下视觉 Token 数随图片面积变化，单样本峰值显存不可预估；`group_by_length` 只按文本长度分组，不管图片。
- **定位**：记录 OOM 样本的图片尺寸，验证 Token 计数（用第 3 周 4.3 节的公式）。
- **修复**：预处理阶段统一限制图片最大边或最大面积；降低 batch 至 1；开启 `quantization_bit: 4`；或设置 `visual_max_tokens` 类参数限制视觉 Token 上限（框架支持各异，LLaMA-Factory 中可通过自定义图像处理参数控制）。

**失效 3：过拟合——微调后"变笨"**
- **症状**：域内任务表现完美，但通用问答能力明显退化（灾难性遗忘）。
- **根因**：小数据 + 大 r + 多 epoch + 高 lr 的组合，让 LoRA 增量覆盖了 LLM 的通用能力方向。
- **定位**：域外测试集（3.4 节对比集的第 2 类）精度大幅下降即确认；观察 WandB 中 eval loss 是否先降后升。
- **修复**：降 r 到 8、epoch 降到 2、lr 减半；混入 10~20% 通用多模态指令数据（数据配比是治本手段）。

**失效 4：Loss 收敛但推理结果与训练时完全不同**
- **症状**：训练时 eval 输出正常，合并导出后推理输出退化。
- **根因**：导出/推理时 template 不一致（如训练用 `qwen2_vl`，推理用默认模板），chat 模板错位导致提示词格式不同；或漏了 `--adapter_name_or_path` 导致用的是裸底座。
- **定位**：对同一条样本打印训练框架内部的最终 prompt 字符串，与推理时手动构造的 prompt 逐字符对比。
- **修复**：训练与推理**始终显式指定同一 template**；合并导出后先跑一条训练集内样本回归验证。

---

## 5. 延伸思考题

1. **秩的选择**：假设你的任务是"让模型输出严格遵循某个复杂 JSON schema"，猜猜需要多大的 r？为什么结构化输出这种"格式约束"任务需要比普通问答更高的秩？（提示：格式约束要求改变输出分布的许多独立维度，等效于更高的目标函数秩；实验上 r=32~64 常见。）
2. **LoRA 注入位置**：`lora_target: all` 会把 LoRA 加到所有 Linear（包括 MLP 的 up/down projection）。设计消融实验比较"只加 attention 的 q/v"与"全加"在 500 条小样本上的差异，猜测哪个更容易过拟合、为什么。
3. **视觉塔微调的边界**：如果你要做一个"医学影像报告生成"模型，通用图片与医学影像的域差距极大，此时还该冻结视觉塔吗？给出你的冻结策略和数据量估算依据（提示：域差距大到 SigLIP 特征本身失效时，需要解冻视觉塔顶部若干层或用更大规模域内数据先做视觉塔继续预训练；几百条 SFT 数据远不够支撑）。

---

*下一篇：[第 6 周：复盘与自测验收](第6周-复盘与自测验收.md)*
