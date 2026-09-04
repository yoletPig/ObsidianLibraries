# 第 1 周教程：GPTQ——逐层最优二阶量化

> **本周要回答的问题**
> 1. 为什么 RTN（直接舍入）在 4-bit 以下崩得厉害，而 GPTQ 能救回来？
> 2. 量化一列后，其余列的最优补偿 $\Delta w = -H_{qq}^{-1}(w_q - \hat w_q)e_q$ 是怎么推出来的？
> 3. Hessian 为什么可以用 $2XX^\top$ 的激活外积累积，而不需要反向传播？
> 4. act-order、block size、dampening 各自在防什么病？

对应学习计划：第 1 周。交付物：① `llm-compressor` 对 Qwen2.5-7B 做 GPTQ-4bit（group=128）并测 PPL/速度；② 手写单层 GPTQ 循环（50 行内），与官方输出对齐。

---

## 1. 第一性原理：把量化写成"逐列最小二乘"

### 1.1 目标：层输出重构误差最小

对线性层 $Y = XW^\top$（$X$ 为校准激活，$d \times n$；$W$ 为 $m \times n$），GPTQ 要解：

$$
\hat W = \arg\min_{\hat W} \ \big\| XW^\top - X\hat W^\top \big\|_F^2 \quad \text{s.t.} \quad \hat W_{ij} \in \text{量化网格}
$$

直接逐元素舍入（RTN, Round-To-Nearest）完全忽略约束之间的耦合：舍入 $w_{ij}$ 造成的输出误差本可以由同行其他列部分抵消。GPTQ 的核心动作就是**显式地把这个抵消算出来**。

### 1.2 OBS 的钥匙：单列量化的最优补偿

把目标函数对 $W$ 的某一行做二阶泰勒展开（展开点在原始 $W$，一阶项为 0——预训练模型处于极小点附近）：

$$
\|XW^\top - X\hat W^\top\|_F^2 \approx (w - \hat w)\, H\, (w - \hat w)^\top, \qquad H = 2X^\top X
$$

其中 $H$ 是该行重构误差的 Hessian（$n \times n$）。OBS（Optimal Brain Surgeon, Hassibi 1993）告诉我们：**把第 $q$ 个元素强行改成 $\hat w_q$ 后，其余元素的最优调整是**

$$
\Delta w = -\, H_{qq}^{-1}\,(w_q - \hat w_q)\, e_q^{\top} \cdot H_{\cdot q} \quad\Longleftrightarrow\quad w_{-q} \mathrel{+}= -\frac{w_q - \hat w_q}{[H^{-1}]_{qq}}\, [H^{-1}]_{\cdot q,\,-q}
$$

配套的最小误差增量为

$$
\Delta E = \frac{(w_q - \hat w_q)^2}{2\,[H^{-1}]_{qq}}
$$

**直觉**：$[H^{-1}]_{qq}$ 衡量"第 $q$ 列方向上的曲率平坦度"——越平坦，强行改动的代价越小；补偿项沿 $H^{-1}$ 的第 $q$ 列方向摊还误差，恰好抵消输出变化。这就是"为什么 GPTQ 比 RTN 精度好这么多"的标准答案：**RTN 只做了舍入，GPTQ 还做了最优补偿**。

### 1.3 Hessian 累积：$H = 2XX^\top$ 不需要反向传播

$H = 2X^\top X$ 是激活的**外积和**，可以按 batch 在线累积：

$$
H \mathrel{+}= 2\, X_b^\top X_b \quad \text{（每个校准 batch 累加一次）}
$$

原因：重构误差是 $W$ 的二次型，其系数矩阵只依赖输入的二阶矩——与损失函数、反向传播无关。这与第 1 周（Stage 1 的剪枝周）里 SparseGPT 用的是同一个 $H$：剪枝是"把列置零"，量化是"把列舍入到网格"，OBS 框架下两者同构。

### 1.4 三个工程开关

- **block size（默认 128）**：一次只处理 128 列（$H$ 按块求逆），控制内存并避免误差长程累积；块间误差不补偿——这是精度与速度的折中。
- **act-order（desc）**：按 $1/[H^{-1}]_{ii}$ 降序处理列，即**先量化"最疼"的列**。先处理的列拿到完整补偿空间，误差敏感度低的列放最后牺牲。通常带来 0.1~0.3 PPL 的收益，代价是权重列序重排（推理内核必须支持）。
- **dampening**：$H \mathrel{+}= \lambda \cdot \mathrm{mean}(\mathrm{diag}(H)) \cdot I$（典型 $\lambda = 0.01$）。校准数据少时 $H$ 病态（近奇异），$H^{-1}$ 数值爆炸；加对角阻尼换稳定性——**失效模式兜底，不是免费午餐**。

---

## 2. 系统架构与数据流

```
校准集 128 条
  │ ① 逐层前向：收集该层输入激活，累积 H = 2 Σ XᵇᵀXᵇ
  ▼
per-group 拆分（g=128：每 128 列一组独立量化）
  │ ② 块循环（每块 128 列）：
  │     对块内每列 q：
  │       ŵ_q = Quantize(w_q)                 ← 舍入到网格
  │       err = (w_q − ŵ_q) / [H⁻¹]_qq        ← 归一化误差
  │       w_后续列 −= err · [H⁻¹]_{q,后续列}   ← OBS 补偿
  ▼
量化权重 + scale + （act-order 时的列置换表）→ 存为 GPTQ checkpoint
```

注意逐列顺序依赖：补偿只影响**未处理**的列，所以这是一个不可并行的扫描——GPTQ 离线跑一次（7B 约 10-30 分钟），推理时完全无感。

---

## 3. 实现与验证（本周交付核心）

### 3.1 手写单层 GPTQ（50 行内，完整可运行）

```python
import torch

def gptq_quantize_layer(W: torch.Tensor, X: torch.Tensor, bits: int = 4,
                        group: int = 128, blocksize: int = 128,
                        damp: float = 0.01) -> torch.Tensor:
    """
    对单层权重 W (m, n) 做 GPTQ INT4 per-group 量化，返回反量化后的 Ŵ。
    X: 校准激活 (samples, n)，Hessian 由它累积。
    """
    m, n = W.shape
    W, X = W.float().clone(), X.float()
    qmax = 2 ** (bits - 1) - 1

    # ① Hessian 累积 + 阻尼
    H = 2 * X.t() @ X / X.shape[0]
    H += damp * H.diagonal().mean() * torch.eye(n, device=W.device)

    # ② per-group：每组独立做 scale 与 OBS 扫描（此处演示 g = n 的逐组循环）
    W_hat = torch.zeros_like(W)
    for g0 in range(0, n, group):
        g1 = min(g0 + group, n)
        Wg, Hg = W[:, g0:g1].clone(), H[g0:g1, g0:g1].clone()
        Hinv = torch.linalg.inv(Hg)          # 演示用直接求逆；官方实现用 Cholesky
        scale = Wg.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax  # per-row
        for b0 in range(0, g1 - g0, blocksize):
            b1 = min(b0 + blocksize, g1 - g0)
            for q in range(b0, b1):
                wq = Wg[:, q]
                wq_hat = torch.round(wq / scale.squeeze()).clamp(-qmax-1, qmax) * scale.squeeze()
                err = (wq - wq_hat) / Hinv[q, q]            # 归一化误差
                Wg[:, q] = wq_hat
                Wg[:, q+1:] -= err.unsqueeze(1) * Hinv[q, q+1:].unsqueeze(0)  # OBS 补偿
        W_hat[:, g0:g1] = Wg
    return W_hat

# ---- 单元自检：GPTQ 的重构误差必须 ≤ RTN ----
torch.manual_seed(0)
m, n = 256, 512
W = torch.randn(m, n) * 0.02
X = torch.randn(256, n)                     # 校准激活
Y = X @ W.t()

def rtn(W, bits=4):
    qmax = 2 ** (bits - 1) - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / s).clamp(-qmax-1, qmax) * s

err_rtn = (Y - X @ rtn(W).t()).pow(2).mean().item()
err_gptq = (Y - X @ gptq_quantize_layer(W, X).t()).pow(2).mean().item()
print(f"重构 MSE：RTN = {err_rtn:.3e}   GPTQ = {err_gptq:.3e}")
assert err_gptq < err_rtn, "GPTQ 重构误差应小于 RTN"
print("GPTQ 单元自检通过 ✓")
```

**预期输出**：`err_gptq` 比 `err_rtn` 低约 2~5 倍——补偿项在数值上兑现了 §1.2 的理论。

### 3.2 官方路线：llm-compressor 量化 7B（云 GPU）

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),      # 可选：先平滑离群激活
    GPTQModifier(targets="Linear", scheme="W4A16",
                 ignore=["lm_head"], block_size=128),
]
oneshot(model="Qwen/Qwen2.5-7B", dataset="ultrachat_200k",
        recipe=recipe, max_seq_length=2048, num_calibration_samples=512,
        output_dir="Qwen2.5-7B-GPTQ-Int4")
```

随后用 Stage 1 的评测脚本测 PPL，并与 3.1 的手写版在**同一个单层**上对比：手写 `gptq_quantize_layer` 的输入输出与该层官方量化结果对齐（逐元素差 < 1e-3，差异来自 act-order 与 Cholesky 数值路径——对齐前先关掉 act-order）。

### 3.3 Hessian 累积的可运行演示（$H = 2XX^\top$ 不需要反向传播）

```python
import torch

def hessian_from_calibration(W_shape_n, calib_batches):
    """逐 batch 累积 H = 2 Σ XᵀX / 总样本数——与损失函数无关。"""
    H = torch.zeros(n, n)
    total = 0
    for X in calib_batches:                 # X: (batch, n) 该层输入激活
        H += 2 * X.float().t() @ X.float()
        total += X.shape[0]
    return H / total

torch.manual_seed(0)
n = 128
Xs = [torch.randn(32, n) for _ in range(8)]          # 模拟 8 个校准 batch
H_inc = hessian_from_calibration(n, Xs)                # 增量式
X_all = torch.cat(Xs)
H_ref = 2 * X_all.t() @ X_all / X_all.shape[0]         # 一次性参考
assert (H_inc - H_ref).abs().max() < 1e-5, "增量累积应等于一次性计算"
# 特征值检查：病态程度决定要不要加阻尼
eig = torch.linalg.eigvalsh(H_inc)
print(f"Hessian 条件数 = {eig.max()/max(eig.min(),1e-12):.1e}")
# 条件数 > 1e6 → 必须加 dampening；这也解释了 §1.4 阻尼开关的存在意义
print("Hessian 累积自检通过 ✓")
```

**预期输出**：断言通过；随机校准数据下条件数通常在 10~100 量级（良性），真实模型某几层可达 1e4+——把它记下来，与量化后 PPL 对照，你会看到"病态层 = 敏感层"。

---

## 4. 工程权衡与失效模式

### 4.1 权衡

- **精度**：GPTQ-4bit(g128) 在 7B 上 PPL 损失通常 < 0.3；3-bit 开始明显（0.5~1.5），必须配 act-order。
- **速度**：离线量化 7B 约 10-30 分钟（单卡），比 AWQ 慢（AWQ 只搜 scale），比 RTN 慢两个量级——但只付一次。
- **act-order 的代价**：列置换要求推理内核支持（vLLM/Marlin 支持）；不支持的引擎要么拒绝加载，要么回退慢路径。
- **校准集敏感度**：512 条是常见值；过少（< 64）→ $H$ 病态 → 补偿方向噪声大。

### 4.2 失效模式

1. **$H$ 奇异导致补偿爆炸**：症状——量化后权重出现 inf/NaN；根因——校准样本线性相关、$H$ 近奇异；修复——加/调大 dampening，或增补校准数据。
2. **act-order 与引擎不匹配**：症状——模型加载成功但输出乱码；根因——引擎未应用列置换表；修复——确认引擎声明支持 `desc_act`，或关掉 act-order 重量化。
3. **group size 与内核约束冲突**：症状——某些引擎只支持 g=128，g=32 的 checkpoint 拒载；根因——内核硬编码组大小；修复——按目标引擎倒推 group 选择（先选引擎再选格式）。
4. **逐层独立最优 ≠ 全局最优**：症状——每层重构误差都小，整模型仍掉点；根因——层间误差复合（Stage 1 第 2 周同款问题）；修复——敏感层保留高精度、或上 QAT 收尾。

---

## 5. 延伸思考题（含解析）

**Q1**：为什么 GPTQ 的 Hessian 用 $2XX^\top$ 而不是损失函数的真 Hessian？
**A**：GPTQ 优化的是**层输出重构**这个代理目标（二次型），其 Hessian 恰好是输入二阶矩，免反向传播、可逐层独立求解。真损失 Hessian 需要全模型二阶信息，代价不可承受；逐层重构误差小 → 整模型损失增量小，是经验上成立的近似。

**Q2**：act-order 为什么"先处理敏感列"？顺序反过来会怎样？
**A**：先处理的列在补偿时还有全部未量化列可动用；后处理的列只能吃掉残余补偿空间。敏感列（$[H^{-1}]_{ii}$ 小、曲率陡）放后面会被迫接受大误差。反过来（asc）等价于把最疼的伤口留到最后缝，PPL 明显变差。

**Q3**：per-group(128) 在 GPTQ 里除了隔离离群值，还有什么作用？
**A**：它限制了**补偿的作用范围**——误差只在 128 列内摊还，避免长程补偿把数值噪声传到远处，同时把 $H^{-1}$ 的求逆规模控制在 128×128。组越小越局部、越稳，但 scale 开销与内核约束上升。

**Q4**：为什么 GPTQ 量化过程不能并行处理多列？
**A**：第 $q$ 列的补偿项要写入所有 $q' > q$ 的列——存在顺序数据依赖。工程上靠"块内串行 + 层间/样本并行 + 分块求逆"摊薄成本，但列扫描本身是串行的，这是它比 AWQ 慢的根本原因。

**Q5**：什么情况下 GPTQ-3bit 是可接受的选择？
**A**：显存/带宽预算硬约束（如 24G 卡跑 14B）、且任务对长文本连贯性容忍度高（短问答、分类抽取）时；必须配 act-order + g=64/128 + 足量校准，并在评测里加长输出项——3-bit 的崩坏往往先出现在长生成而非 PPL。

---

## 6. GPTQ 超参数敏感性指南（实战调参表）

| 超参 | 默认 | 调大的影响 | 调小的影响 | 建议搜索范围 |
| --- | --- | --- | --- | --- |
| group_size | 128 | scale 开销↓、精度↓ | 精度↑、存储↑ | {128, 64, 32} |
| blocksize | 128 | 补偿范围↑、内存↑ | 内存↓、长程补偿缺失 | 128 固定 |
| dampening | 0.01 | 数值更稳、补偿变弱 | 补偿更强、可能爆炸 | {0, 0.01, 0.1} |
| act_order | False | 精度↑、内核要求↑ | 兼容性最好 | {False, True} |
| 校准条数 | 512 | H 更准、时间↑ | H 病态、精度↓ | {128, 512} |

**一条经验法则**：先把校准条数与 act_order 定住（收益最确定），再扫 group_size（存储与精度的权衡），最后才动 dampening（通常是兜底项）。调参顺序按"收益/成本比"从高到低。

### 6.1 GPTQ 与 AWQ 的选型一句话

- 要极致精度（3-bit 或精度红线紧）→ GPTQ（有补偿机制）；
- 要速度（量化耗时敏感、校准数据稀缺）→ AWQ（免二阶信息）；
- 两者都可接受时，先跑 AWQ（快），不达标再上 GPTQ。

---

## 7. 面试快问快答（本周内容的压缩版）

**Q：GPTQ 一句话原理？**
逐列量化，用剩余列按 $H^{-1}$ 二阶补偿输出误差——把量化写成带约束的逐层最小二乘。

**Q：为什么 RTN 在低比特崩、GPTQ 不崩？**
RTN 各列误差独立累积、互不抵消；GPTQ 每列的误差被主动摊还给未量化列，层输出重构误差被压到二阶最优。

**Q：Hessian 从哪来？**
重构目标的二阶导 = 输入二阶矩 $2XX^\top$，按校准 batch 外积累积，不需要反向传播。

**Q：act-order 的收益与代价？**
收益：敏感列先处理拿到完整补偿空间（0.1-0.3 PPL）；代价：列置换表要求推理内核支持 `desc_act`。

**Q：block size 为什么是 128？**
块内求逆的内存与数值稳定性、块间不补偿的精度损失、内核效率三方平衡的经验值。

**Q：手写 50 行 GPTQ 与官方实现的差异在哪？**
官方：Cholesky 分解求逆（数值更稳）、act-order 列重排、死组检测、并行块处理；教学版：直接求逆、固定列序——数值语义一致，工程鲁棒性不同。

**本周知识在后续阶段的位置**：第 2 周 AWQ 的"免二阶"路线是本周方法的速度对照；第 5-6 周横评里 GPTQ 行的所有数字由本周产出；Stage 3 的 INT4 GEMV 内核将按本周产出的 per-group 布局读取权重。

---

## 本周交付清单

- [ ] 闭卷推导 OBS 补偿式 $\Delta w$ 与误差增量 $\Delta E$，写出 $H = 2XX^\top$ 的累积式。
- [ ] 跑通手写 `gptq_quantize_layer`，断言 `err_gptq < err_rtn`（记录倍数）。
- [ ] 云上完成 Qwen2.5-7B GPTQ-4bit(g128)，填 PPL 与推理速度进基线表。
- [ ] 手写版与官方单层输出对齐（关 act-order 后逐元素差 < 1e-3）。
- [ ] 用自己的话解释 act-order 与 dampening 各防什么病（≤ 100 字）。
