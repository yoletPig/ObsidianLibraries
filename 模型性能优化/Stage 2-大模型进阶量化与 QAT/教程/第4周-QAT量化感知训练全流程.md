# 第 4 周教程：QAT——量化感知训练全流程

> **本周要回答的问题**
> 1. `round()` 处处不可导，QAT 凭什么能反向传播？STE（Straight-Through Estimator）的数学借口是什么？
> 2. Fake Quant 插在哪、前向与反向各做什么？
> 3. OmniQuant / SpinQuant / EfficientQAT 各自在可学习的"旋钮"上做了什么文章？
> 4. 什么精度损失阈值下，值得付出 ~10× 的 QAT 训练成本？（QAT vs PTQ 决策树）

对应学习计划：第 4 周。交付物：`llm-compressor` QAT 模式对 Qwen2.5-1.5B 做 INT4 QAT，对比同位宽 GPTQ 的 PPL，画训练曲线并解释"前 5% step 损失最大"。

---

## 1. 第一性原理：把量化噪声变成训练目标的一部分

### 1.1 PTQ 的天花板

PTQ 只能**事后补救**（补偿、迁移、保护），因为它不改变权重分布本身。当位宽 ≤ 4、或模型本身很小（容量冗余少）时，事后补救的上限不够——此时唯一的出路是**让训练过程看见量化噪声**，让权重自己学会"站在网格上也不疼"。这就是 QAT。

### 1.2 Fake Quant：前向模拟、反向直通

在权重（或激活）上插入**伪量化（fake quantization）**算子：

$$
\mathrm{FQ}(w) = \mathrm{dequant}\big(\mathrm{quant}(w)\big) = s \cdot \mathrm{clamp}\!\Big(\mathrm{round}\Big(\frac{w}{s} + z\Big),\ q_{\min},\ q_{\max}\Big)
$$

前向：输出被吸附到量化网格上（真实模拟推理时的数值）；反向：`round` 与 `clamp` 几乎处处导数为 0/未定义，**STE 直接把梯度透传**：

$$
\frac{\partial \mathcal{L}}{\partial w} \approx \frac{\partial \mathcal{L}}{\partial\, \mathrm{FQ}(w)} \quad \text{（在 } |w| \text{ 未裁剪区域内）}
$$

STE 的借口：`FQ(w)` 是 $w$ 的"近似恒等映射"（误差 ≤ $s/2$），局部导数近似为 1——粗糙但对梯度下降足够有效。被 `clamp` 饱和的区域梯度置零（权重要么自己走回来，要么放弃）。

**工程细节**：① 训练初期可用**学习率预热**与渐进收紧网格，防止噪声过大撕裂收敛；② 随机舍入（stochastic rounding）在 QAT 里比确定性舍入更常用——期望无偏，避免系统性偏置。

### 1.3 三个前沿：各自可学习的"旋钮"

| 方法 | 可学习对象 | 核心思想 |
| --- | --- | --- |
| **OmniQuant** | 权重截断阈值（LWC）+ 等价变换（LET） | 裁剪点不必是 min/max，让它可学习；等价变换把激活难度搬给权重（SmoothQuant 的可学习版） |
| **SpinQuant** | 旋转矩阵 $R$ | 对权重/激活做**学习的正交旋转**（Hadamard 初始化），把离群值摊匀再量化——QUIP# 的在线版 |
| **EfficientQAT** | 无（块级重建） | 把每层当独立小块，用生成的伪输入做免真实数据重建——QAT 的成本降到接近 PTQ |

三者的共同趋势：**把 PTQ 时代的"手工超参"（裁剪点、迁移系数、旋转矩阵）变成梯度优化的变量**——这是 QAT 2.0 的方法论。

### 1.4 工程配方（实战背下来）

1. **先 PTQ 再 QAT**：用 GPTQ/AWQ 的产物做初始化，QAT 从"已经很接近"的网格起步，收敛快且稳；从零开始 QAT 容易先经历一段量化噪声主导的高原期。
2. **数据量 1-10k 条足够**：QAT 是微调不是重训；通用语料 + 目标领域混合。
3. **冻结策略**：小预算下冻结浅层（量化不敏感）与嵌入层，只训中间层；或全员只训 scale/zero-point（最便宜）。
4. **学习率**：比预训练低 1~2 个量级（1e-5 ~ 5e-5），余弦衰减。

### 1.5 QAT vs PTQ 决策树

```
PTQ（GPTQ/AWQ, g128）先跑一遍，测 PPL 增量 Δ：
├─ Δ < 0.3        → 收工，不需要 QAT
├─ 0.3 ≤ Δ < 1.0  → 试 OmniQuant/EfficientQAT（低成本 QAT）
│      └─ 仍不达标 → 全量 QAT（PTQ 初始化 + 10k 数据）
└─ Δ ≥ 1.0（或任务崩坏）→ 直接全量 QAT；若位宽 ≤ 3，考虑先升回 4-bit
```

成本坐标：PTQ 分钟~小时级；QAT 是"一次小规模微调"（10× 起步）。**决策依据永远是实测 Δ，不是直觉**。

---

## 2. 系统架构与数据流

```
QAT 训练步：
输入 batch ──► 每层权重 w ──► FakeQuant(w) ──► 前向（网格上的数值）
                                     │
损失 ◄── 输出 ◄──────────────────────┘
  │ 反向：梯度经 STE 直通 FakeQuant ──► 更新 w（FP16 主权重）
  ▼
收尾：导出真实量化权重 (q, s, z) + config ──► vLLM/TRT 加载
```

---

## 3. 实现与验证（本周交付核心）

### 3.1 手写 Fake Quant + STE（完整可运行，CPU 可验证）

```python
import torch
import torch.nn as nn

class FakeQuantize(torch.autograd.Function):
    """对称 INT 伪量化：前向吸附网格，反向 STE 直通。"""
    @staticmethod
    def forward(ctx, w, scale, qmax):
        q = torch.round(w / scale).clamp(-qmax, qmax)
        ctx.save_for_backward(w, scale, qmax)
        return q * scale

    @staticmethod
    def backward(ctx, grad_out):
        w, scale, qmax = ctx.saved_tensors
        # STE：未饱和区域梯度透传，饱和区域置零
        inside = ((w / scale) >= -qmax) & ((w / scale) <= qmax)
        return grad_out * inside, None, None

def fake_quant(w, bits=4):
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax() / qmax
    return FakeQuantize.apply(w, scale, qmax)

# ---- 单元自检 ----
w = torch.randn(64, requires_grad=True)
wq = fake_quant(w, bits=4)
loss = (wq ** 2).sum()
loss.backward()
# 1) 前向必须落在网格上
qmax = 7
s = w.detach().abs().amax() / qmax
resid = (wq.detach() / s).round() * s - wq.detach()
assert resid.abs().max() < 1e-6, "前向输出必须落在量化网格上"
# 2) 反向梯度应与直接对 w² 求导一致（STE 透传）
assert torch.allclose(w.grad, 2 * w, atol=1e-6), "STE 应透传梯度"
print("Fake Quant + STE 单元自检通过 ✓")
```

### 3.2 端到端：llm-compressor QAT 模式（云 GPU）

```python
# 云环境：A10-24G 可跑 1.5B QAT（bf16 + 梯度）
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.utils.rotation import RotationModifier  # 可选

# QAT 配方：先定网格（PTQ 初始化思想），再进入训练循环
recipe = [QuantizationModifier(targets="Linear", scheme="W4A16",
                               ignore=["lm_head"])]
oneshot(model="Qwen/Qwen2.5-1.5B", dataset="ultrachat_200k",
        recipe=recipe, num_calibration_samples=512,
        output_dir="Qwen2.5-1.5B-W4A16-init")     # ← PTQ 初始化产物
# 随后进入 QAT 训练循环（transformers Trainer + 上述初始化权重），
# 学习率 2e-5、500-2000 步、1k-10k 条数据，记录每 50 步的 PPL
```

**交付图表**：QAT 训练曲线（step vs WikiText-2 PPL）。预期形态：前 ~5% step PPL 最高（量化噪声刚注入、权重尚未适配），随后快速回落并低于"同位宽 GPTQ 的 PPL 横线"。解释写进报告：**初始高损是网格强约束的"休克期"，STE 梯度驱动权重向网格点迁移，容量冗余被重新组织**。

对比表（实测填写）：

| 方案 | PPL(WikiText-2) | 相对 FP16 |
| --- | --- | --- |
| FP16 基线 | _实测_ | — |
| GPTQ INT4 (g128) | _实测_ | _Δ_PTQ_ |
| QAT INT4（GPTQ 初始化，1000 步） | _实测（预期最优）_ | _Δ_QAT < Δ_PTQ_ |

### 3.3 可学习裁剪（OmniQuant 的 LWC 最小可运行版）

OmniQuant 的核心旋钮之一是"让裁剪点可学习"。把 §1.3 的思想压缩到 15 行：

```python
import torch, torch.nn as nn

class LearnableClipQuant(nn.Module):
    """对称量化，裁剪上界 clip 可训练（LWC 的最小演示）。"""
    def __init__(self, bits=4):
        super().__init__()
        self.qmax = 2 ** (bits - 1) - 1
        self.clip = nn.Parameter(torch.tensor(1.0))   # 可学习裁剪点（初始化=幅度最大）

    def forward(self, w):
        c = self.clip.abs()                            # 保证为正
        w_clip = w.clamp(-c, c)                        # 前向裁剪
        s = c / self.qmax
        # 假量化：round + STE（第 1 周已实现同款）
        q = torch.round(w_clip / s)
        w_q = (q + (w_clip / s - q).detach()) * s      # STE 技巧的等价写法
        return w_q

m = LearnableClipQuant()
w = torch.randn(256, requires_grad=True)
loss = (m(w) - w.detach()).pow(2).mean()   # 重建误差目标
loss.backward()
assert m.clip.grad is not None and m.clip.grad.abs() > 0, "裁剪点应收到梯度"
print(f"clip 梯度 = {m.clip.grad:.4e} ✓（裁剪点可被梯度下降优化）")
# 训练若干步后，clip 会收敛到比 amax 更优的位置——这就是 LWC 的全部秘密
```

**要点**：真实权重以高精度保存，`clip` 是唯一新增的可训练参数——所以"让裁剪点可学习"几乎零显存成本。OmniQuant 在此之上加 LET（可学习等价变换），SpinQuant 则把"旋转矩阵"变成可学习对象——万变不离其宗：**把超参变成梯度变量**。

---

## 6. QAT 实验排期表（云资源规划）

| 实验 | 目的 | 预估资源 | 输出 |
| --- | --- | --- | --- |
| GPTQ INT4 初始化 | QAT 起点 | 1×A10, ~30 min | 初始化 checkpoint |
| QAT 500 步 | 休克期观察 | 1×A10, ~2 h | 训练曲线前半段 |
| QAT 1000 步 | 收敛对比 | 续上 | 对比表数字 |
| 同位宽 GPTQ 复测 | 对照 | 1×A10, ~30 min | Δ_PTQ |

**曲线解读检查点**：
- 0-5% step：损失最高（休克期）——必须出现在图上，否则怀疑初始化或配置错误；
- 5-30%：快速回落，斜率最大——权重向网格迁移的主阶段；
- 30% 后：缓降趋平——此时可评估早停，省下的云时就是钱。

### 6.1 一句话决策模板（面试背诵）

"先跑一遍 PTQ 拿 Δ：< 0.3 收工；0.3~1.0 试低成本 QAT；≥ 1.0 上全量 QAT 或升位宽。所有判断基于实测，不凭直觉。"

---

## 7. 面试快问快答（本周内容压缩版）

**Q：STE 为什么能让不可导的 round 训练起来？**
前向输出落在网格上模拟推理数值；反向把 round 当恒等映射透传梯度。借口是 `FQ(w)≈w`（误差有界 $s/2$），局部导数近似 1。

**Q：STE 的失效边界？**
网格极粗（2-bit）时 $s/2$ 与权重幅度同量级，"近似恒等"不成立，梯度方向噪声大——这就是低位宽需要 OmniQuant/SpinQuant 的原因。

**Q：为什么先 PTQ 再 QAT？**
GPTQ 已把权重放在二阶最优点附近，QAT 只需小幅适配端到端损失——起点好、步数少、不掉坏盆地。从零 QAT 要先付"休克期"学费。

**Q：OmniQuant 的 LET 和 SmoothQuant 什么关系？**
LET 是可学习的 SmoothQuant：迁移系数不再是固定 α，而是逐层可训练的等价变换参数，由梯度决定难度迁移量。

**Q：EfficientQAT 的"免数据"数据从哪来？**
用每层输入的统计量（均值/协方差）合成伪输入，做逐层块级重建——省真实语料与全模型前向，代价是合成残差。

**Q：为什么前 5% step 损失最大？**
量化噪声刚注入、权重尚未适配网格的"休克期"；STE 梯度驱动权重向网格点迁移后快速回落——这条曲线形态是 QAT 跑对了的标志。

---

## 4. 工程权衡与失效模式

### 4.1 权衡

- **成本**：QAT ≈ 一次微调（数据 1-10k、数百~数千步），比 PTQ 贵 10-100×；只在实测 Δ 超标时使用（§1.5 决策树）。
- **QAT 产物的引擎匹配**：与 GPTQ/AWQ 一样，导出格式必须与目标引擎对齐（llm-compressor 产出的 compressed-tensors 可被 vLLM 直载）。
- **学到的网格锁死校准分布**：QAT 权重是为训练时的数据分布优化的，部署分布漂移时可能比 PTQ 更敏感——领域数据要混进去。

### 4.2 失效模式

1. **训练初期发散**：症状——注入 Fake Quant 后 loss 爆炸；根因——网格太粗 + 学习率太大；修复——预热、渐进量化（先 8-bit 热身再降到 4-bit）、或 PTQ 初始化。
2. **STE 饱和梯度死亡**：症状——大量权重卡在裁剪边界不动；根因——`clamp` 区域梯度为零且权重方向一直向外推；修复——放宽初始裁剪范围、LWC 可学习截断（OmniQuant）。
3. **只训 scale 不训权重的"假 QAT"**：症状——训了几千步 PPL 纹丝不动；根因——可训练参数太少，优化自由度不够；修复——放开主权重（至少中间层）的梯度。
4. **评测与训练网格不一致**：症状——训练时指标正常、部署后崩坏；根因——推理引擎用的量化参数（如激活量化、算子融合顺序）与训练时 Fake Quant 配置不同；修复——导出后在目标引擎上重测全指标。

---

## 5. 延伸思考题（含解析）

**Q1**：STE 的梯度近似为什么"够用"？它的失效边界在哪？
**A**：`FQ(w) ≈ w`（误差有界 $s/2$），所以用恒等映射的导数 1 近似真实次梯度在大部分区域内方向正确，足以驱动梯度下降。失效边界：网格极粗（2-bit）时 $s/2$ 与权重幅度同量级，"近似恒等"不再成立，STE 梯度方向噪声大——这正是低位宽需要 OmniQuant/SpinQuant 等更强结构的原因。

**Q2**：为什么"先 PTQ 初始化再 QAT"几乎总是更好？
**A**：GPTQ 已把权重放在"二阶最优的网格点"附近，QAT 只需小幅微调适配端到端损失——优化起点好、需要的步数少、不容易掉进量化噪声主导的坏盆地。从零 QAT 则要先付一段"休克期"学费。

**Q3**：OmniQuant 的 LET 与 SmoothQuant 是什么关系？
**A**：LET（Learnable Equivalent Transformation）就是"可学习的 SmoothQuant"：迁移系数不再是固定超参 $\alpha$，而是逐层可训练的等价变换参数，由梯度决定难度该迁多少、迁到哪——把经验规则升级为数据驱动。

**Q4**：EfficientQAT 声称"免数据"，它的数据从哪来？
**A**：用每层输入分布的统计量（均值/协方差）**合成**伪输入，对每层做独立块级重建训练——省掉真实语料与全模型前向，代价是合成输入与真实激活分布的残差。它是"块重建"（GPTQ 的近亲）与"训练"的杂交。

**Q5**：什么时候应该放弃 QAT、改升位宽？
**A**：当 Δ ≥ 1.0 且任务对精度敏感时，3-bit QAT 的工程成本与不确定性通常高于 4-bit PTQ 的体积成本（差 ~25% 存储）。面试答题模板：**"先算账——体积预算是否真的卡死在 3-bit，不是就别冒低位宽的险"**。

---

## 本周交付清单

- [ ] 闭卷写出 Fake Quant 前向公式与 STE 反向规则。
- [ ] 跑通手写 `FakeQuantize`：网格吸附与梯度透传两个断言通过。
- [ ] 云上完成 Qwen2.5-1.5B INT4 QAT（GPTQ 初始化），填三行对比表。
- [ ] 画 QAT 训练曲线，标注"休克期"并解释前 5% step 损失最大的原因。
- [ ] 背下 §1.5 决策树与四条工程配方（面试直接可用）。
