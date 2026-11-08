# 07 - 序列建模：RNN 到注意力

## 本篇目标

读完本篇，你应该能做到：

1. 把"变长输入"形式化为序列建模问题，说清它和定长分类的本质差别。
2. 写出 RNN 的递推公式，并用梯度连乘的账本解释**它为什么记不住长距离依赖**。
3. 用"门"的直觉讲清 LSTM/GRU 各自怎么缓解这个问题。
4. 推导注意力机制（Q/K/V）的直觉与公式，并说清**注意力凭什么取代循环**——这是 08 章 Transformer 的最后一跳。

本篇是全系列承上启下的一章：向上接 5 章（梯度连乘）、6 章（残差），向下直通 08 章 Transformer 与《语音方向 Stage 2》的 Whisper（encoder-decoder 结构）。

## 1. 序列问题建模

### 1.1 为什么需要它

前面的模型都吃**定长输入**：MLP 吃拉平的向量，CNN 吃固定分辨率的图。语言、音频、股价天然是**变长**的：一句话 3 个词或 300 个词；音频 0.5 秒或 30 秒。另外序列还有两个新性质：

- **时序依赖**：顺序有语义（"狗咬人"≠"人咬狗"）；
- **长度不定但结构相似**：同一个语法结构套不同长度。

### 1.2 建模方式

把第 $t$ 步的输入记作 $x_t \in \mathbb{R}^d$，任务三类：

| 任务 | 输入 → 输出 | 例子 |
| --- | --- | --- |
| 序列分类 | $(x_1..x_T) \to 1$ 个标签 | 情感分类 |
| 逐位置标注 | $(x_1..x_T) \to (y_1..y_T)$ | 词性标注、ASR 帧级标签 |
| 序列生成 | $(x_1..x_T) \to (y_1..y_M)$，$M \neq T$ | 翻译、语音识别、GPT |

"读完整个序列再输出"需要一个能**累积信息**的状态——这就是 RNN。

## 2. RNN：用循环状态记住历史

### 2.1 公式与实现

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$$

$h_t \in \mathbb{R}^m$ 是隐状态（"到目前为止的记忆摘要"），每步同套参数（权重共享，同 6 章卷积的思想）。最小实现：

```python
import torch
import torch.nn as nn

class MyRNN(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.Wx = nn.Linear(d_in, d_hidden, bias=False)
        self.Wh = nn.Linear(d_hidden, d_hidden, bias=False)
        self.b = nn.Parameter(torch.zeros(d_hidden))

    def forward(self, x):
        # x: (B, T, d_in)，时间维上逐步循环
        h = torch.zeros(x.size(0), self.Wh.weight.size(1), dtype=x.dtype)
        hs = []
        for t in range(x.size(1)):        # T 步串行 —— 注意力将消灭这个 for
            h = torch.tanh(self.Wx(x[:, t]) + self.Wh(h) + self.b)
            hs.append(h)
        return torch.stack(hs, dim=1)     # (B, T, d_hidden)

rnn = MyRNN(4, 8)
x = torch.randn(2, 5, 4)                  # batch=2, 序列长 5, 特征 4
print(rnn(x).shape)                       # torch.Size([2, 5, 8])
```

PyTorch 内置 `nn.RNN(input_size, hidden_size, batch_first=True)` 可直接用，输出同形状。

### 2.2 RNN 的梯度消失：一笔账

把 BPTT（沿时间反向传播）展开，$h_t$ 对 $h_k$ 的梯度是**雅可比连乘**（5 章的连乘，时间换成了层数）：

$$\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}, \qquad \frac{\partial h_i}{\partial h_{i-1}} = \mathrm{diag}\!\left(1 - h_i^2\right) W_h^\top$$

两个致命因子：

1. $\tanh$ 的局部导数 $\le 1$（饱和时趋近 0）；
2. 每一步都乘同一个 $W_h^\top$——它的谱半径（最大奇异值）$\rho(W_h)$ 决定连乘的放大率：

$$\left\|\frac{\partial h_t}{\partial h_k}\right\| \lesssim \rho(W_h)^{\,t-k} \cdot \prod (1-h_i^2)$$

$\rho(W_h) < 1$ 或 $\tanh$ 饱和 → 指数消失（第 $k$ 步的信息传不到第 $t$ 步的学习信号）；$>1$ → 指数爆炸（靠裁剪续命）。数字感受一下：$\rho=0.8$，走 50 步后 $0.8^{50} \approx 1.4\times10^{-5}$——**50 步之前发生了什么，梯度几乎归零，"记忆"名存实亡**（Hochreiter 1991 年的 diploma 论文首次系统分析了这个问题；英文综述见 Pascanu et al. 2013）。

这就是"RNN 记不住长距离依赖"的数学本质：不是参数不够，是**梯度通路上每一步都要过同一个衰减器**。

## 3. LSTM / GRU：用门控保护信息通路

### 3.1 LSTM 的门控直觉

LSTM（Hochreiter & Schmidhuber, *Long Short-Term Memory*, Neural Computation 1997）的核心是一独立于 $h$ 的**细胞状态** $c_t$，与三个门：

$$f_t = \sigma(W_f[h_{t-1}, x_t] + b_f) \quad \text{遗忘门：旧记忆保留多少}$$
$$i_t = \sigma(W_i[h_{t-1}, x_t] + b_i) \quad \text{输入门：新内容写入多少}$$
$$o_t = \sigma(W_o[h_{t-1}, x_t] + b_o) \quad \text{输出门：记忆透露多少}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \qquad\qquad \text{更新细胞状态}$$
$$h_t = o_t \odot \tanh(c_t)$$

**关键设计在 $c_t$ 的更新式**：加法！旧记忆 $c_{t-1}$ 乘一个 $f_t$ 直接**加**上新内容，而不是像 RNN 那样被矩阵变换+非线性反复加工。于是梯度沿 $c$ 的通路是：

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t \quad(\text{对角}), \qquad \frac{\partial c_t}{\partial c_k} = \prod f_i + (\text{其他路径})$$

当网络学会把 $f_t \approx 1$（"这条记忆别忘"），梯度就**近似恒等地直通**——6 章 ResNet 的 $+x$ 与这里是同一个思想：**给梯度留一条不衰减的加法通路**（加法通路即"常数误差.carousel"，LSTM 论文的原话）。$\sigma$ 门输出在 0~1，是"每个维度上的软开关"：$f_t$ 一维接近 0 = 抹掉该维度记忆，接近 1 = 完整保留。

### 3.2 GRU：LSTM 的瘦身版

GRU（Cho et al., 2014）合并出两个门：更新门 $z_t$（同时管遗忘与输入，$h_t = (1-z_t)\odot h_{t-1} + z_t \odot \tilde h_t$）与重置门 $r_t$。参数少 1/4，多数任务效果与 LSTM 相当（经验做法：小数据/要快用 GRU，需要细胞状态记忆的任务用 LSTM）。

### 3.3 用起来

```python
lstm = nn.LSTM(input_size=4, hidden_size=16, num_layers=1, batch_first=True)
x = torch.randn(2, 5, 4)                 # (B, T, d_in)
out, (h_n, c_n) = lstm(x)
print(out.shape, h_n.shape, c_n.shape)   # (2,5,16) (1,2,16) (1,2,16)
# out: 每个时间步的 h; h_n/c_n: 最后一步的 h 与 c
```

## 4. seq2seq 与 teacher forcing

### 4.1 encoder-decoder 结构

翻译类任务输入输出长度不等，标准解法（Sutskever et al., *Sequence to Sequence Learning with Neural Networks*, NeurIPS 2014）：

```text
encoder: x1 x2 ... xT  ──RNN──> 上下文向量 c（末步隐状态）
decoder: 用 c 初始化，逐步生成 y1, y2, ..., 每步的输出作为下一步的输入
```

**瓶颈问题**：任意长的句子都要压进一个**固定维向量** $c$，长句信息必然丢失。

### 4.2 teacher forcing

decoder 训练时怎么定"上一步的输入"？两种选择：

- 用模型自己上一步的预测（可能早已错到天边，误差滚雪球）；
- **teacher forcing**：直接喂**真实答案** $y_{t-1}^{\text{gold}}$——训练快且稳（Williams & Zipser, 1989 提出该术语）。

```python
# decoder 训练步（示意）:
# decoder_input = torch.cat([bos_id, gold[:-1]])   # 右移一位: <bos>, y1..y_{T-1}
# logits = decoder(decoder_input, c)               # 一步并行算完所有位置（比逐步快）
# loss = F.cross_entropy(logits, gold)
```

代价：训练分布（永远看真值）与推理分布（看自己的预测）不一致，即 **exposure bias**——推理时一步错步步错。缓解：scheduled sampling（训练中按概率混入自己的预测）、或干脆换成注意力架构。经验做法：现代 seq2seq（Transformer 时代）多靠大训练量 + beam search 缓解，scheduled sampling 反而少用。

## 5. 注意力机制：让 decoder 回头看 encoder

### 5.1 直觉：检索式记忆

翻译"I love cats"第 3 步要输出"猫"时，decoder 最需要的信息其实在 encoder 的第 2、3 步隐状态里——但 seq2seq 把它们全压进了 $c$。注意力（Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015）的解法：**decoder 每生成一步，都回头"查"一遍 encoder 的全部位置，按相关性加权取用**。

用检索打比方：

| 角色 | 含义 | 直觉 |
| --- | --- | --- |
| Query $Q$ | 当前步的"查询向量" | "我现在需要什么样的信息？" |
| Key $K$ | 每个源位置的"索引向量" | "我这里有什么信息？" |
| Value $V$ | 每个源位置的"内容向量" | "真要选我，我交出这些内容" |

Query 与每个 Key 算相似度 → softmax 成权重 → 按 weight 加权求和 Value。类比数据库：$Q$ 是查询语句，$K$ 是索引，$V$ 是记录本体。

### 5.2 公式（scaled dot-product attention）

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

（Vaswani et al., *Attention Is All You Need*, NeurIPS 2017，第 3.2.1 节。）为什么除 $\sqrt{d_k}$：设 $q, k$ 的分量独立同分布、均值 0 方差 1，则点积 $q\cdot k = \sum_i q_i k_i$ 的方差是 $d_k$——维度越高点积越大，softmax 被推向 one-hot，梯度趋近 0（softmax 饱和）。除以 $\sqrt{d_k}$ 把点积方差拉回 1。

**跑一个能用手按计算器验算的最小例子**（数值已实测）：

```python
import torch, math

q = torch.tensor([[1.0, 0.0]])            # (1, 2)  一个查询
K = torch.tensor([[1.0, 0.0],             # (2, 2)  两个位置的 key
                  [0.0, 1.0]])
V = torch.tensor([[10.0, 0.0],            # (2, 2)  两个位置的内容
                  [0.0, 10.0]])

scores = q @ K.T / math.sqrt(2)           # (1, 2) = [[0.7071, 0.0]]
w = torch.softmax(scores, dim=-1)         # [[0.6698, 0.3302]] —— q 与第1个key更"像"
out = w @ V                                # [[6.6976, 3.3024]]
print(scores, w, out)
# 输出是 V 的加权平均: 0.6698*[10,0] + 0.3302*[0,10] = [6.70, 3.30]
```

注意结果的语义：**注意力输出永远是 value 的凸组合**（权重非负、和为 1），落在 value 的"凸包"里——它做的是"检索+混合"，不是"变换出没见过的东西"；变换的活交给后续 FFN（8 章）。

### 5.3 为什么注意力取代循环

三个维度的碾压（Vaswani et al. 2017, Table 1 的核心论点）：

| 维度 | 循环（RNN/LSTM） | 自注意力 |
| --- | --- | --- |
| 每步计算量 | $O(d^2 T)$ 串行 T 步 | $O(T^2 d)$ 完全并行 |
| 最小串行步数 | $O(T)$——必须逐时间步 | $O(1)$——所有位置同时算 |
| 任意两位置的信息路径 | $O(T)$ 步（隔 100 词要传 100 次） | $O(1)$ 步（一次注意力直达） |

第三条最根本：**长距离依赖不再需要穿过 $T$ 个衰减器**，5.2 的梯度消失账本从架构上被销账。代价是 $T\times T$ 的注意力矩阵（显存 $O(T^2)$，长文本成为新的瓶颈——这是《VLM & VLA》《模型性能优化》系列里 FlashAttention、KV cache 等优化的由来，12 章会预告 kv cache）。

并行性还带来工程红利：训练时 GPU 能把整个序列一口气算完，而 RNN 的 for 循环让 GPU 闲着——这是 2017 年后 Transformer 全面取代 RNN 的直接推手。

### 5.4 自注意力 vs 交叉注意力

- **自注意力**：$Q,K,V$ 都来自**同一个**序列——序列内部互相看（Transformer encoder/decoder 内部用）；
- **交叉注意力**：$Q$ 来自 decoder，$K,V$ 来自 encoder——"看着源句生成目标句"（seq2seq 的注意力升级版，Whisper 的 decoder 就用它；VLM 里图文对齐也常是这种结构）。

## 6. 常见陷阱

1. **忘记 `batch_first`**：`nn.RNN/LSTM/GRU` 默认输入 $(T, B, d)$，传成 $(B, T, d)$ 不报错但结果全错——本系列所有代码统一 `batch_first=True` 并在注释里标 shape。
2. **对 padded 位置不 mask**：变长 batch（4 章 collate 补 0）在 RNN/注意力里直接污染统计；RNN 用 `pack_padded_sequence`，注意力用加 $-\infty$ 的 mask。
3. **注意力后忘归一化的权重语义**：softmax 在哪一维做（key 维）搞反，输出形状对但语义错——用手算例子对拍（5.2）。
4. **把 seq2seq 的 $c$ 当万能**：仍拿"固定向量"做长文本任务而不加注意力，指标上限锁死。
5. **teacher forcing 推理也照用**：推理时没有真值，必须自回归；训练/推理的输入来源不同导致的分布差异要有意识（4.2 节 exposure bias）。

## 本篇小结

- 序列建模的核心是"可累积的状态"；RNN 用递推 $h_t=\tanh(W_xx_t+W_hh_{t-1}+b)$ 实现，但梯度是雅可比连乘，长距离依赖训练不动。
- LSTM/GRU 用门控 + **加法式**记忆更新保护梯度通路（$c_t = f_t\odot c_{t-1} + i_t\odot\tilde c_t$），与 ResNet 的恒等通路同一思想。
- seq2seq 把编码与解码解耦，teacher forcing 加速训练但引入 exposure bias；固定上下文向量是瓶颈。
- 注意力 = 用 $Q$ 查 $K$ 算相似度、softmax 加权取 $V$；$\sqrt{d_k}$ 缩放防 softmax 饱和；$O(1)$ 的信息路径与并行性使它取代循环——下一章把它组装成 Transformer。

## 练习

1. 把 2.1 的 `MyRNN` 在一个"序列末位求和是否超过阈值"的二分类任务上训练，对比 $T=10$ 与 $T=50$ 的收敛速度，体会长序列学习困难。
2. 手算验证 5.2 的例子：把 $q$ 改成 $[0,1]$，写出新的注意力权重与输出。
3. 用 `nn.LSTM` 在同一个任务上对比 $T=50$ 的表现，解释细胞状态带来的改善。
4. 实现 scaled-dot-product attention 函数（输入 $(B,T,d)$ 三元组），构造 $q\cdot k$ 很大（比如把 q 乘 10）的情况，观察 softmax 输出与梯度（对 q 求 `.grad`）的变化，验证 $\sqrt{d_k}$ 缩放的必要性。
5. 思考题：自注意力的显存是 $O(T^2)$，为什么训练 8K 上下文比训练 1K 贵远不止 8 倍？这个瓶颈在推理时如何被 KV cache 部分缓解？（可先查证后作答，为 12 章《模型保存、部署与推理》铺垫。）

## 参考来源

- Hochreiter & Schmidhuber, *Long Short-Term Memory*, Neural Computation 9(8), 1997: https://www.bioinf.jku.at/publications/older/2604.pdf
- 梯度消失/爆炸在 RNN 的系统分析：Pascanu, Mikolov & Bengio, ICML 2013: https://arxiv.org/abs/1211.5063
- Sutskever et al., *Sequence to Sequence Learning with Neural Networks*, NeurIPS 2014: https://arxiv.org/abs/1409.3215
- Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015: https://arxiv.org/abs/1409.0473
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017: https://arxiv.org/abs/1706.03762（$\sqrt{d_k}$ 方差论证见 §3.2.1 与附录；计算复杂度对比见 Table 1）
- GRU：Cho et al., EMNLP 2014: https://arxiv.org/abs/1406.1078
- teacher forcing 术语出处：Williams & Zipser, *A Learning Algorithm for Continually Running Fully Recurrent Neural Networks*, Neural Computation 1989
