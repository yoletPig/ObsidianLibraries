# 08 - Transformer 从零实现

## 本篇目标

读完本篇，你应该能做到：

1. 逐块手写 Transformer 的每个组件：多头注意力、位置编码、FFN、LayerNorm、残差、causal mask——每个张量的 shape 都写得出来。
2. 把组件组装成一个**可训练的字符级 miniGPT**（nanoGPT 路线），CPU 上几百步就能看到 loss 明显下降。
3. 理解 GPT 风格的几个工程细节：pre-LN、weight tying、初始化 std=0.02。
4. 用模型自回归地生成文本，理解训练目标"预测下一个 token"如何变成生成能力。

本篇路线与 Andrej Karpathy 的 nanoGPT/"Let's build GPT"一致（Karpathy, *nanoGPT*: https://github.com/karpathy/nanoGPT ；视频教程 *Let's build GPT from scratch*, 2023）——那是全社区公认的最好入门材料，本篇用中文+逐 shape 注释重走一遍。学完本篇再去看《VLM & VLA Stage 1》的 ViT/LLM 架构拆解，会非常顺。

## 1. 蓝图：GPT 长什么样

### 1.1 训练目标

语言模型 = 对"下一个 token"的条件分布建模：

$$P(x_1, ..., x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, ..., x_{t-1})$$

训练就是拿整段文本，**每个位置同时**做"看前文猜下一个"的多分类（交叉熵，3 章）。一条数据当 $T$ 条训练样本用，这就是自回归语言模型的便宜之处。

### 1.2 结构总览（形状记号 $B$=批, $T$=序列长, $C$=通道）

```text
token ids (B, T)
   │  tok_emb: 查表 (vocab, C)  → (B, T, C)
   │  + pos_emb: 位置 (T, C)    → (B, T, C)      ← 让模型知道"词序"
   ▼
┌─ Block × n_layer ──────────────────────────┐
│  x = x + Attn(LN(x))     ← 注意力混合 token 间信息
│  x = x + MLP(LN(x))      ← FFN 逐位置加工
└────────────────────────────────────────────┘
   │  LN → Linear(C, vocab)                   → logits (B, T, vocab)
   ▼
cross_entropy(logits.view(B*T, V), targets.view(B*T))
```

两个思想来源：**注意力**提供 token 间的混合（7 章），**残差**提供无衰减梯度通路（6 章）。Transformer = "注意力做混合的残差网络"。

## 2. 逐块手写

### 2.1 CausalSelfAttention（带 causal mask 的多头注意力）

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """多头自注意力 + 因果掩码。d_model 必须能被 n_head 整除。"""
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head          # 每个头的维度 d_k = C/nh
        self.qkv = nn.Linear(d_model, 3 * d_model)   # 一次投影出 Q,K,V（省两次 GEMM）
        self.proj = nn.Linear(d_model, d_model)      # 拼接后的输出投影
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape                            # (B, T, C)
        q, k, v = self.qkv(x).chunk(3, dim=2)        # 各 (B, T, C)
        # 拆头: (B, T, C) -> (B, T, nh, dh) -> (B, nh, T, dh)
        # 每个头独立做注意力 —— 在 nh 维上并行, 等效 batch 增大
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)   # (B, nh, T, dh)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)   # (B, nh, T, dh)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)   # (B, nh, T, dh)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)     # (B, nh, T, T)
        # causal mask: 位置 t 只准看 <= t 的位置 (语言模型不许偷看未来)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device),
                          diagonal=1)                # 上三角(不含对角)为 True
        att = att.masked_fill(mask, float("-inf"))   # 未来位置 -> -inf
        att = torch.softmax(att, dim=-1)             # (B, nh, T, T), 每行和为 1
        att = self.attn_drop(att)
        y = att @ v                                  # (B, nh, T, dh)
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # 合头: (B, T, C)
        return self.resid_drop(self.proj(y))         # 输出投影 (B, T, C)
```

shape 推演（拿 $B=2, T=4, C=64, nh=4, dh=16$ 走一遍）：

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `qkv(x)` | (2, 4, 192) | 3C=192 |
| `q/k/v` | (2, 4, 64) | chunk 三等分 |
| 拆头后 | (2, 4, 4, 16) | (B, T, nh, dh) |
| transpose 后 | (2, 4, 4, 16) | (B, nh, T, dh)——前两维恰好同值，别混淆 |
| `q @ k^T` | (2, 4, 4, 4) | (B, nh, T, T) 注意力矩阵 |
| `att @ v` | (2, 4, 4, 16) | (B, nh, T, dh) |
| 合头后 | (2, 4, 64) | 回到 (B, T, C) |

**多头为什么有效**：单个注意力头只能学一种"查询-键"匹配模式；8 个头让模型**同时**跟踪"指代关系""相邻性""语义类别"等不同模式。$d_k = C/nh$ 变小后总计算量不变（$\sum_h T^2 d_k = T^2 C$），等于把一个大注意力切成几份做不同的"视角"。原论文多头设定与消融见 Vaswani et al. 2017 §3.2.2。

**causal mask 的实现要点**：先加 $-\infty$ 再 softmax，softmax 后未来位置权重恰为 0；**不要在 softmax 之后乘 0**——那样每行权重和不再为 1，语义错误。`float("-inf")` 在整行被 mask 时会产生 NaN，但 causal mask 每行至少保留对角线，不会整行全 mask。

### 2.2 位置编码

注意力本身是**置换等变**的——把输入顺序打乱，输出按同样方式打乱，模型根本不知道谁在前谁在后。必须显式注入位置信息。原论文（Vaswani et al. 2017 §3.5）用正弦位置编码：

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

直觉：不同维度 $i$ 对应不同"频率的钟摆"——低维摆得快（区分相邻位置），高维摆得慢（区分远距离），组合起来每个位置拿到唯一"指纹"，且相对位置可由线性变换表达（$PE_{pos+k}$ 是 $PE_{pos}$ 的线性函数，三角恒等式保证）。

```python
def sinusoidal_pe(T, d):
    """返回 (T, d) 的正弦位置编码矩阵"""
    pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)      # (T, 1)
    i = torch.arange(d // 2, dtype=torch.float32).unsqueeze(0)   # (1, d/2)
    div = torch.exp(-math.log(10000.0) * 2 * i / d)              # 10000^{-2i/d}
    pe = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(pos * div)     # 偶数维 sin
    pe[:, 1::2] = torch.cos(pos * div)     # 奇数维 cos
    return pe

pe = sinusoidal_pe(4, 8)
print(pe[0])    # [0, 1, 0, 1, 0, 1, 0, 1] —— pos=0: sin=0, cos=1
print(pe[3])    # [0.141, -0.990, 0.296, 0.955, 0.030, 1.000, 0.003, 1.000]
```

工程现状（GPT 路线的选择）：GPT-2/GPT-3 与 nanoGPT 用**可学习位置嵌入**（一个 `(T_max, C)` 的 Embedding 表，随训练更新）；LLaMA 等现代模型用 **RoPE（旋转位置编码）**，它把相对位置信息注入 Q/K，外推性更好。本篇 miniGPT 沿用 nanoGPT 的可学习方案（8 行代码，见 2.4），正弦版作为理解基础。RoPE 见 RoFormer 论文（Su et al., 2021, arXiv:2104.09864），此处仅预告。

### 2.3 FFN 与 Block

FFN（前馈网络）：对**每个位置独立**做两层升维-降维变换：

$$\mathrm{FFN}(x) = W_2\,\sigma(W_1 x + b_1) + b_2, \qquad \text{中间维 } d_{ff} = 4C$$

分工记忆法：**注意力负责"跨位置混合"，FFN 负责"逐位置深加工"**（把注意力检回的信息做非线性变换）。GeLU 激活（比 ReLU 平滑，GPT/BERT 均用）。

把注意力、FFN、LayerNorm、残差组装成标准 Block。**用 pre-LN**（LayerNorm 放在子层之前）而不是原论文的 post-LN：

```python
class Block(nn.Module):
    """Transformer block: pre-LN 残差结构 (GPT-2 风格)"""
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))    # 残差 1: 注意力支路
        x = x + self.mlp(self.ln2(x))     # 残差 2: FFN 支路
        return x
```

**pre-LN vs post-LN**：post-LN（原始论文）里 LN 在残差**内部**，深层训练需要学习率 warmup 精细呵护才能不发散；pre-LN 把 LN 移到支路里，残差主干 $x$ 保持"纯加法通路"，梯度像 ResNet 一样直通——深层 Transformer 训练稳定性的关键改动（出处：Xiong et al., *On Layer Normalization in the Transformer Architecture*, ICML 2020；GPT-2 即采用 pre-LN）。LN 本身的数学与 BN/LN/RMSNorm 家族对比在 9 章。

### 2.4 组装 MiniGPT

```python
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_head=2, n_layer=2,
                 block_size=32, dropout=0.0):
        super().__init__()
        self.block_size = block_size                       # 上下文窗口
        self.tok_emb = nn.Embedding(vocab_size, d_model)   # 词表 -> 向量
        self.pos_emb = nn.Embedding(block_size, d_model)   # 可学习位置嵌入
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(                       # 必须用 ModuleList(03 章)!
            [Block(d_model, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)                  # 最后一个 LN (pre-LN 配套)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight             # weight tying(见下)
        self.apply(self._init_weights)                     # 递归初始化所有子模块

    def _init_weights(self, m):
        """GPT-2 惯例: 线性层/嵌入 normal(0, 0.02), 偏置清零"""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape                                   # idx: (B, T) 的 token id
        assert T <= self.block_size, "序列超出上下文窗口"
        pos = torch.arange(T, device=idx.device)           # (T,)
        x = self.tok_emb(idx) + self.pos_emb(pos)          # (B,T,C) + (T,C) 广播 -> (B,T,C)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)                                   # (B, T, C) 不变形状地穿过
        x = self.ln_f(x)
        logits = self.head(x)                              # (B, T, vocab)
        return logits
```

**weight tying**（输出层与嵌入层共享权重）：`head.weight = tok_emb.weight`。词嵌入矩阵是 `(vocab, C)`，输出投影天然要一个 `(C, vocab)` 的矩阵——互为转置，共享后省一大块参数（以本篇 demo 为例 30×64=1920 个）且小数据上困惑度更好（Press & Wolf, *Using the Output Embedding to Improve Language Models*, EACL 2017；GPT-2 论文亦采用）。陷阱： tying 后不要对该权重重复初始化两次（`apply` 在共享后执行会同时改两处——本例中顺序是先 tying 再 apply，两处本来就是同一张量，无害；若把 tying 放在 apply 之后也一样，因为赋值让两者永远指向同一存储）。

## 3. 训练演示：字符级语料

### 3.1 数据与 batch

字符级 tokenize（每字符一个 token）——vocab 只有几十，免去分词器依赖，专注机制本身。语料用一段 Shakespeare 文本（nanoGPT 的 tiny shakespeare 数据集即此风格；离线可直接用下方内嵌文本）：

```python
text = ("We are accounted poor citizens, the patricians good.\n"
        "What authority surfeits on would relieve us: if they\n"
        "would yield us but the superfluity, while it were\n"
        "wholesome, we might guess they relieved us humanely;\n"
        "but they think we are too dear.") * 60          # 重复 60 份凑足训练量
chars = sorted(set(text))                                # 字符表
stoi = {c: i for i, c in enumerate(chars)}
V = len(chars)                                           # 本例 V = 30
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)   # (14400,)

def get_batch(batch_size=8, T=32):
    """随机采 batch_size 段长度 T 的窗口; 目标是输入右移一位"""
    ix = torch.randint(len(data) - T - 1, (batch_size,))
    x = torch.stack([data[i : i + T] for i in ix])       # (B, T)
    y = torch.stack([data[i + 1 : i + T + 1] for i in ix])  # (B, T) 右移一位
    return x, y
```

`y` 是 `x` 右移一位——同一窗口里每个位置都自带"下一字符"标签，一次前向等于 $B \times T$ 个分类样本。

### 3.2 训练循环（CPU 几分钟）

```python
torch.manual_seed(42)
model = MiniGPT(vocab_size=V, d_model=64, n_head=2, n_layer=2, block_size=32)
print("参数量:", sum(p.numel() for p in model.parameters()))   # 104064
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)     # 9 章: 为何是 AdamW+3e-4

for step in range(301):
    x, y = get_batch(8, 32)
    logits = model(x)                                          # (8, 32, 30)
    loss = F.cross_entropy(logits.view(-1, V), y.view(-1))     # 摊平成 (256, 30)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    # 9 章: 防爆炸
    optimizer.step()
    if step % 100 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")
```

实测输出（CPU、seed=42，本章代码可逐行复现）：

```text
参数量: 104064
step   0  loss 3.4292     ← 起点约等于 ln(30) = 3.401(均匀猜测的损失)
step 100  loss 2.2912
step 200  loss 1.7738
step 300  loss 1.4208
```

**看懂这三行数字**：初始 loss 必然接近 $\ln V$（随机参数下模型输出近似均匀分布，交叉熵 = $\ln V$）——如果你的初始 loss 远大于 $\ln V$，说明初始化有问题（9 章）。随后单调下降，300 步从 3.43 到 1.42，等价于每字符困惑度从 30 降到 $e^{1.42} \approx 4.1$。

### 3.3 自回归生成

```python
model.eval()
idx = torch.zeros(1, 1, dtype=torch.long)          # 从 <空字符>(id 0) 开始
for _ in range(60):                                # 生成 60 个字符
    idx_cond = idx[:, -32:]                        # 截断到上下文窗口(滑窗)
    logits = model(idx_cond)                       # (1, t, V)
    probs = torch.softmax(logits[:, -1], dim=-1)   # 只取最后一个位置的分布 (1, V)
    next_id = torch.multinomial(probs, num_samples=1)   # 按概率采样
    idx = torch.cat([idx, next_id], dim=1)
print("".join(chars[i] for i in idx[0].tolist()))
```

实测输出示例（temperature=1 的采样，每次运行不同）：

```text
areitk :
czites d.
whe
nwhelianhe guwouepe, tcit ghe zeessho
```

初具英文单词的"形状"（空格分布、常见字母组合）——300 步 CPU 训练的合理产出。要更像样：加数据（完整 tiny shakespeare 约 1.1MB）、加模型（d_model=384, n_layer=6, 正是 nanoGPT 的 char-level 配置）、GPU 上训半小时（[GPU]）；CPU 上 d_model=128、训练 1-2 万步也能到 loss≈1.3 附近（经验做法）。

### 3.4 每步 tensor shape 总账

把整个前向的 shape 变化列成一张表（$B=8,T=32,C=64,V=30$）：

| 阶段 | 形状 |
| --- | --- |
| 输入 idx | (8, 32) |
| tok_emb + pos_emb | (8, 32, 64) |
| Block 内 attn 输入 LN 后 | (8, 32, 64) |
| 注意力矩阵 | (8, 2, 32, 32) |
| Block 输出（残差不变形状） | (8, 32, 64) |
| 最后 LN 后 head 输出 logits | (8, 32, 30) |
| cross_entropy 摊平 | (256, 30) vs (256,) |

**残差结构让 $C$ 维贯穿全程**——这也是它被称为"网络主干道"的原因。

## 4. 常见陷阱

1. **用 `nn.Sequential` 堆 Block**：`self.blocks = nn.Sequential(*[Block(...)])` 其实也能注册（Sequential 本身是 Module），但中间想取中间特征/加自定义逻辑时不灵活；标准做法是 `nn.ModuleList` + 循环（03 章）。
2. **mask 忘记 / 加错方向**：语言模型必须 causal；`torch.triu(..., diagonal=1)` 的 `diagonal=1` 保证对角线（自己看自己）不被 mask。mask 加反会把"看过去"堵死，loss 完全不降。
3. **`float("-inf")` 与全 mask 行**：某些任务（如 padding 位置）可能整行 mask，softmax 出 NaN；本篇 causal mask 无此问题，但写 padding mask 时要给每行至少留一个可达位置。
4. **softmax 维度**：注意力权重在最后一维（key 维）归一化，`dim=-1`；写成 `dim=-2` 形状不变但语义全错——用 2.1 的手算例子对拍。
5. **拆头 reshape 顺序**：`view(B, T, nh, dh)` 错写成 `view(B, nh, T, dh)`——后者把 head 维切在了序列维上，不报错、loss 不降，是最阴险的 shape bug。记忆法：先 `(B, T, nh, dh)` 再 transpose，**数据在内存里仍是按 (B, T, ...) 排列的连续块**。
6. **生成时忘截断窗口**：序列长度超过 `block_size` 后 `pos_emb` 越界报错；`idx_cond = idx[:, -block_size:]` 滑窗处理（3.3 已示范）。
7. **权重 tying 后重复 `apply` 初始化**：会同时覆盖两处（同一存储），若初始化策略不同会互相污染——把 tying 放在初始化之后最稳（经验做法）。

## 本篇小结

- GPT = token/位置嵌入 + n×(attention 残差 + FFN 残差) + 最后 LN + 输出投影；训练目标是逐位置"预测下一个 token"（交叉熵），初始 loss 基准线是 $\ln V$。
- 多头注意力：`(B,T,C) → (B,nh,T,dh) → (B,nh,T,T) → (B,nh,T,dh) → (B,T,C)`，causal mask 用 $-\infty$ 加在 softmax 之前。
- 位置编码解决注意力的置换等变问题：正弦版是理解基础，GPT 系用可学习嵌入，现代模型用 RoPE。
- pre-LN + 残差让深层训练稳定（对比原论文 post-LN）；weight tying 省参数且小数据更优。
- 104K 参数、2 层、d=64 的 miniGPT 在 CPU 上 300 步 loss 3.43→1.42，能生成有"词形"的文本——Transformer 的全部机制已在你手里。

## 练习

1. 把 `n_head` 从 2 改成 1 和 8（其余不变）各训 300 步，对比 loss 曲线，解释单头为什么吃亏。
2. 删掉 causal mask（改成双向注意力）训练，观察 loss 是否反而更低、生成质量如何崩溃（提示：训练时看了未来，推理时没有未来）。
3. 把正弦位置编码换成 `nn.Embedding` 方案与"完全没有位置编码"方案，三者对比：无位置编码时 loss 明显更高且生成的文本有重复倾向，验证"注意力需要位置信息"。
4. 在 Block 中加一个打印钩子（`register_forward_hook`），跑一个 batch，确认每个 Block 的输入输出形状都是 (8, 32, 64)。
5. （进阶）给 miniGPT 加 temperature 采样：`probs = softmax(logits[:, -1] / temperature)`，取 temperature=0.5 与 1.5 各生成一段，描述风格差异。

## 参考来源

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017: https://arxiv.org/abs/1706.03762
- Karpathy, *nanoGPT*（本篇代码结构与训练循环的参照）：https://github.com/karpathy/nanoGPT
- Karpathy, *Let's build GPT: from scratch, in code, spelled out*, 2023: https://www.youtube.com/watch?v=kCc8FmEb1nY
- Radford et al., *Language Models are Unsupervised Multitask Learners*（GPT-2，pre-LN、std=0.02 初始化、weight tying、可学习位置嵌入）: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Press & Wolf, *Using the Output Embedding to Improve Language Models*, EACL 2017: https://arxiv.org/abs/1608.05859
- Xiong et al., *On Layer Normalization in the Transformer Architecture*, ICML 2020: https://arxiv.org/abs/2002.04745
- RoPE: Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021: https://arxiv.org/abs/2104.09864
