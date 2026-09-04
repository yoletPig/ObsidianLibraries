# 第 2 周教程：AWQ 与 SmoothQuant——激活离群值的两派解法

> **本周要回答的问题**
> 1. SmoothQuant 的等价变换 $Y = (X/s)(s \cdot W)$ 凭什么能把激活的量化难度"迁移"给权重？$\alpha$ 迁移系数怎么选？
> 2. AWQ 为什么只保护 1% 的显著通道？"按激活幅度加权"和 GPTQ 的二阶方法哲学上差在哪？
> 3. W4A16 与 W8A8 的适用边界是什么？为什么云端服务偏爱 W8A8、端侧偏爱 W4A16？
> 4. QUIP# / SqueezeLLM 的码本路线在解决什么 GPTQ 解决不了的问题？

对应学习计划：第 2 周。交付物：① 同一 7B 模型分别跑 SmoothQuant（W8A8）与 AWQ（W4），对比 PPL/显存/吞吐；② 手写 AWQ scale 搜索核心循环，可视化 1% 显著通道的激活幅度分布。

---

## 1. 第一性原理：离群值在激活里，解法却有两条路

### 1.1 共同的问题陈述

Stage 1 已实证：LLM 激活的离群值**按通道固定出现**、幅度可达主体 10~100 倍。权重量化（W4A16）绕开了激活这个雷区，但代价是计算仍在 FP16——**算力没省**。要同时量化激活（W8A8 走 INT8 Tensor Core），就必须正面处理激活离群值。两条路线：

| 路线 | 核心动作 | 位宽形态 | 目标场景 |
| --- | --- | --- | --- |
| SmoothQuant | 把激活难度**迁移**给权重 | W8A8 | 云端大 batch 服务 |
| AWQ | **保护**显著权重通道，激活保持 FP16 | W4A16 | 端侧/显存受限 |

### 1.2 SmoothQuant：难度迁移的数学

对线性层 $Y = XW$，引入 per-channel 正缩放 $s$（作用在输入通道维）：

$$
Y = XW = \underbrace{(X \cdot \mathrm{diag}(s)^{-1})}_{\hat X} \cdot \underbrace{(\mathrm{diag}(s) \cdot W)}_{\hat W}
$$

这是**数学恒等变换**：输出逐比特不变，但量化难度在 $\hat X$ 与 $\hat W$ 之间重新分配。$s_j$ 取大 → 第 $j$ 个激活通道被压小（离群值被"平滑"掉），代价是对应权重列被放大。迁移系数 $\alpha$ 的定义（逐通道）：

$$
s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}, \qquad \alpha \in [0, 1]
$$

- $\alpha = 0$：难度全留给激活（等价不迁移）；$\alpha = 1$：难度全推给权重；
- 经验值 $\alpha \approx 0.5 \sim 0.8$：权重的量化容忍度比激活高（权重静态、可用 per-channel 细粒度），所以默认多迁一些给权重；
- 落地位置：$s$ 被**吸收进前一个算子**（LayerNorm 的 $\gamma$ 或前一层权重），推理图结构不变——这是它能直接进 TensorRT-LLM 的关键。

### 1.3 AWQ：1% 显著通道保护

AWQ（Lin et al., 2023）的两个观察：

1. **权重并非同等重要**：仅约 1% 的权重通道对应着大激活幅度的输入通道，剪掉/量化坏它们会造成显著精度损失（与 Wanda 的 $|W|\cdot|X|$ 观察同源）；
2. **保护不必保留全精度**：对这 1% 通道做 per-channel 缩放 $s$，让它们的数值幅度变大再量化——等价变换后这些通道的相对量化误差被压小，**不需要混合精度、不需要额外存储原始权重**。

形式化：对选定通道集 $\mathcal{S}$（按 $\bar{|X_j|}$ 取 top-1%），

$$
Y = XW = (X \cdot s^{-1})(s \cdot W), \qquad s_j = \frac{\bar{|X_j|}^{\,\alpha}}{\max(|W_j|)^{1-\alpha}},\ j \in \mathcal{S}
$$

然后对 $s \cdot W$ 做普通的 per-group 量化。注意与 SmoothQuant 的微妙差异：**SmoothQuant 迁的是激活难度（目标是让激活动态范围变小从而可量化）**；**AWQ 放的是显著权重（目标是让重要通道的量化误差变小，激活根本不量化）**。方向相反，哲学不同。

### 1.4 AWQ vs GPTQ：一阶保护 vs 二阶补偿

- GPTQ：接受量化误差，然后用 $H^{-1}$ 把它**摊还**给其他列——误差仍在，只是被最优分配；
- AWQ：从源头**减少**重要通道的误差产生（缩放后网格相对更细），不做列间补偿——因此免校准数据（只需激活统计），速度远快于 GPTQ。

### 1.5 W4A16 vs W8A8 适用边界

| 维度 | W4A16（AWQ/GPTQ） | W8A8（SmoothQuant） |
| --- | --- | --- |
| 收益来源 | 权重体积/带宽 4× ↓ | 算力 + 带宽双省（INT8 Tensor Core） |
| batch=1 decode | 快（访存瓶颈，带宽即收益） | 几乎无收益（瓶颈不在算力） |
| 大 batch prefill/服务 | 算力没省，吞吐受限 | 吞吐显著提升 |
| 典型引擎 | vLLM + Marlin、llama.cpp | TensorRT-LLM |

**一句话**：显存/带宽受限选 W4A16，算力/吞吐受限选 W8A8。

### 1.6 速览：QUIP# 与 SqueezeLLM

两者都走**码本（codebook）路线**：不用均匀网格，而是学一组非均匀码点。

- **QUIP#**：随机正交旋转（incoherence processing）把权重/激活的离群值"打散"成近似高斯，再用向量量化码本——旋转后分布规整，2-bit 也能活；
- **SqueezeLLM**：k-means 学码本 + 把离群值单独拎出来稀疏存储（呼应 LLM.int8() 的混合精度思想）。

它们的意义：证明 4-bit 以下要活得靠**非均匀码点 + 分布整形**，为 Stage 1 讲的 NF4 与 imatrix 提供同族背景。

---

## 2. 系统架构与数据流

```
SmoothQuant 流水线：
校准集 ──► 逐层收集激活通道极值 max|X_j| ──► s_j（α 迁移）
  ──► s 吸收进 LayerNorm γ / 上一层权重 ──► 激活与权重都可 INT8（W8A8）

AWQ 流水线：
校准集 ──► 逐层收集激活通道均值 |X̄_j| ──► 选 top-1% 通道 𝒮
  ──► 网格搜索 α（0~1，按该层输出重构误差）──► 应用 s 到 𝒮
  ──► 普通 per-group W4 量化（激活不量化）
```

---

## 3. 实现与验证（本周交付核心）

### 3.1 手写 AWQ scale 搜索（完整可运行）

```python
import torch

def awq_search_scale(W: torch.Tensor, X: torch.Tensor, bits: int = 4,
                     topk_ratio: float = 0.01, alpha_grid: int = 20):
    """
    AWQ 核心循环：选显著通道，网格搜索 α，返回最优 s。
    W: (out, in)；X: (samples, in)。
    """
    qmax = 2 ** (bits - 1) - 1
    act_mean = X.float().abs().mean(dim=0)                  # 每输入通道激活幅度
    k = max(1, int(W.shape[1] * topk_ratio))
    salient = act_mean.topk(k).indices                      # 1% 显著通道

    w_absmax = W.float().abs().amax(dim=0).clamp(min=1e-8)  # 每输入通道权重极值
    best_alpha, best_err = 0.0, float("inf")
    Y = X.float() @ W.float().t()

    for i in range(alpha_grid + 1):
        a = i / alpha_grid
        s = torch.ones(W.shape[1], device=W.device)
        s[salient] = act_mean[salient].pow(a) / w_absmax[salient].pow(1 - a)
        # 等价变换后量化权重（per-tensor 演示）
        Ws = W.float() * s[None, :]
        sc = Ws.abs().amax() / qmax
        Wq = torch.round(Ws / sc).clamp(-qmax - 1, qmax) * sc
        Y_hat = (X.float() / s[None, :]) @ Wq.t()
        err = (Y - Y_hat).pow(2).mean().item()
        if err < best_err:
            best_err, best_alpha = err, a
    s = torch.ones(W.shape[1])
    s[salient] = act_mean[salient].pow(best_alpha) / w_absmax[salient].pow(1 - best_alpha)
    return s, best_alpha, salient

# ---- 单元自检 ----
torch.manual_seed(0)
W = torch.randn(256, 512) * 0.02
X = torch.randn(128, 512)
X[:, [7, 42]] *= 30.0                    # 人造两个离群通道
s, a, salient = awq_search_scale(W, X)
assert 7 in salient and 42 in salient, "离群通道应被选为显著通道"
assert (s[salient] > 1.0).any(), "显著通道的 scale 应放大权重"
print(f"最优 alpha = {a:.2f}；显著通道数 = {len(salient)}")
# 可视化（交付要求）：显著通道的激活幅度分布
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
act = X.abs().mean(dim=0).numpy()
plt.figure(figsize=(8, 3))
plt.bar(range(512), act); plt.title("activation magnitude per channel")
plt.savefig("awq_salient_channels.png", dpi=120)
print("已输出 awq_salient_channels.png：两根高柱即 1% 显著通道 ✓")
```

**预期输出**：两根高柱（通道 7、42）远超其余通道——论文"1% 通道贡献大部分精度"的可视化复现。

### 3.2 双方案对比（云 GPU）

```python
# AWQ：AutoAWQ 官方
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
model.quantize(model.tokenizer, quant_config={"w_bit": 4, "q_group_size": 128})
model.save_quantized("Qwen2.5-7B-AWQ")

# SmoothQuant（W8A8）：llm-compressor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
oneshot(model="Qwen/Qwen2.5-7B", dataset="ultrachat_200k",
        recipe=[SmoothQuantModifier(smoothing_strength=0.8),
                QuantizationModifier(scheme="W8A8_INT8", targets="Linear",
                                     ignore=["lm_head"])],
        num_calibration_samples=512, output_dir="Qwen2.5-7B-SQ-W8A8")
```

对比表（实测填写）：

| 方案 | PPL(WikiText-2) | 模型体积 | 显存 | 吞吐(tokens/s, b=32) |
| --- | --- | --- | --- | --- |
| FP16 基线 | _实测_ | ~14.5 GB | _实测_ | _实测_ |
| AWQ W4A16 | _实测（预期 ≈ 基线 +0.1）_ | ~4.3 GB | _实测_ | _实测（b=1 快，b=32 受限）_ |
| SmoothQuant W8A8 | _实测（预期 ≈ 基线 +0.2）_ | ~7.5 GB | _实测_ | _实测（b=32 显著快）_ |

**预期结论**：b=1 时 AWQ 吞吐反超 W8A8（带宽收益 > 算力收益）；b=32 时 W8A8 反超——这就是 §1.5 适用边界的实测版。

---

## 4. 工程权衡与失效模式

### 4.1 权衡

- **AWQ 免校准二阶信息**：只需激活统计 → 量化速度快、对校准集不敏感；代价是它不做误差摊还，极端低位宽（≤3bit）下不如 GPTQ。
- **SmoothQuant 的 $\alpha$ 是全局超参**：不同层的最优 $\alpha$ 其实不同，逐层搜索成本高；0.5~0.8 是安全区间。
- **等价变换的位置约束**：$s$ 必须能被吸收进相邻算子（LayerNorm/前一层），对非标准结构（某些 MoE、共享嵌入）要逐层检查。

### 4.2 失效模式

1. **AWQ 显著通道选错统计量**：症状——保护后仍掉点；根因——用 max 而非 mean 统计激活（max 又被离群样本污染）；修复——AWQ 论文用均值 + 跨样本聚合，照做。
2. **SmoothQuant 后权重反而难量化**：症状——激活平滑了、权重侧误差暴涨；根因——$\alpha$ 过大，权重被放大出新的离群列；修复——调低 $\alpha$，或对权重用 per-channel 量化兜底。
3. **scale 吸收位置错误**：症状——数学等价但输出变了；根因——$s$ 被同时应用于变换两侧（抵消错误）或漏吸收；修复——单元级数值对齐测试（变换前后输出差 < 1e-4）。
4. **W8A8 在小 batch 下"负优化"**：症状——上线 W8A8 后延迟反升；根因——b=1 时瓶颈是访存，INT8 算力收益为零，反量化与格式转换白付开销；修复——按 §1.5 边界回退 W4A16。

---

## 5. 延伸思考题（含解析）

**Q1**：SmoothQuant 的等价变换为什么"免费"？它的隐藏成本是什么？
**A**：$s$ 离线吸收进相邻参数，推理图与算子数不变，无运行时开销。隐藏成本：被放大的权重列可能恶化**权重侧**量化、且吸收位置受模型结构约束——等价是数学的，工程落地要逐层核对。

**Q2**：AWQ 为什么用激活的**均值**幅度而不是最大值选显著通道？
**A**：最大值对单个异常样本敏感（又引入了离群值问题），均值反映通道的**持续性**重要程度——AWQ 保护的是"总是重要"的通道，不是"偶尔巨大"的通道。

**Q3**：为什么说 AWQ 与 Wanda 共享同一个观察？两者的动作差在哪？
**A**：都观察到 ~1% 的输入通道激活幅度显著大、与之相连的权重"更重要"（$|W|\cdot|X|$）。Wanda 据此**决定剪谁**（保留显著连接），AWQ 据此**决定缩放谁**（放大显著通道再量化）——同一观察，剪枝与量化两种用法。

**Q4**：QUIP# 的随机旋转为什么能救 2-bit 量化？
**A**：旋转（正交矩阵）保持模长不变但把能量在维度间"摊匀"（incoherence）：原本集中在少数通道的离群幅度被打散成近似高斯的均匀分布——网格/码本对规整分布的表示效率最高。代价是推理时要付旋转变换（可与相邻线性层合并）。

**Q5**：同一个 7B 模型，端侧（你的 RK3588）和云端服务各选什么？
**A**：端侧：AWQ W4A16（或直接 Stage 4 的 GGUF Q4_K）——内存是硬约束、batch=1 访存瓶颈，带宽收益直接兑现；云端：SmoothQuant W8A8 + TensorRT-LLM——大 batch 下 INT8 算力翻倍才是真金白银。选型的本质是**认清当前负载的瓶颈资源**（Stage 3 Roofline 的预演）。

---

## 6. 两派解法的统一视角：等价变换家族

### 6.1 一个公式包住三种方法

SmoothQuant、AWQ、OmniQuant-LET、SpinQuant 的旋转，本质都是同一个动作：**在相邻算子之间插入互逆的变换对**，数学上输出不变，量化难度重新分配：

$$
Y = XW = X \cdot T \cdot T^{-1} \cdot W = \hat X \hat W, \qquad T = \mathrm{diag}(s)\ \text{或正交矩阵 } R
$$

| 方法 | $T$ 的形式 | 优化目标 | 求解方式 |
| --- | --- | --- | --- |
| SmoothQuant | 对角，按极值比 | 激活动态范围↓ | 解析式（$\alpha$ 超参） |
| AWQ | 对角，仅 1% 通道 | 显著通道误差↓ | 网格搜索 $\alpha$ |
| OmniQuant-LET | 对角，逐层 | 层重构误差↓ | 梯度学习 |
| SpinQuant/QUIP# | 正交旋转 | 分布各向同性化 | 随机/学习的旋转 |

**记忆锚点**：对角变换调"幅度分配"，正交旋转变"分布形状"——前者便宜（可离线吸收），后者更强（可救 2-bit）但要在推理图里保留旋转开销。

### 6.2 等价变换的验证纪律（可运行）

任何等价变换落地前，必须过数值对齐测试：

```python
import torch

def verify_equivalence(X, W, transform_fn, atol=1e-4):
    """变换前后输出必须一致——等价变换的验收测试。"""
    Y = X @ W
    X_hat, W_hat = transform_fn(X, W)
    Y_hat = X_hat @ W_hat
    err = (Y - Y_hat).abs().max().item()
    assert err < atol, f"等价变换破坏数值：最大误差 {err}"
    return err

torch.manual_seed(0)
X, W = torch.randn(64, 128), torch.randn(128, 64)
s = torch.rand(128) * 2 + 0.5
err = verify_equivalence(X, W, lambda x, w: (x / s, s[:, None] * w))
print(f"SmoothQuant 式变换验收通过 ✓（最大误差 {err:.2e}）")
```

真实模型里误差来源是 FP16 舍入与"吸收进相邻算子"时的重排——验收阈值放宽到 1e-3，但**必须测**，这是等价变换类方法的第一纪律。

---

## 7. 本周实验记录模板（填完并入基线表）

| 实验 | 配置 | PPL | 体积 | b=1 吞吐 | b=32 吞吐 |
| --- | --- | --- | --- | --- | --- |
| FP16 基线 | 7B | _实测_ | 14.5G | _实测_ | _实测_ |
| AWQ | w_bit=4, g=128 | _实测_ | ~3.9G | _实测_ | _实测_ |
| SmoothQuant | W8A8, α=0.8 | _实测_ | ~7.5G | _实测_ | _实测_ |
| AWQ α 扫描（选做） | α ∈ {0, .25, .5, .75, 1} | _曲线_ | — | — | — |

**三个必须写进结论的观察**：① α 扫描曲线的最优点及其解释；② 1% 显著通道的激活幅度可视化（§3.1 的图）；③ b=1 与 b=32 的排名翻转（若有）。这三条就是面试时"我亲手做过 AWQ 与 SmoothQuant 对比"的完整证据链。

### 7.1 两分钟口头复述稿（自测用）

> 激活离群值集中在约 1% 的固定通道，幅度可达主体百倍。要量化激活，有两条路：SmoothQuant 做**难度迁移**——用等价变换 $Y=(X/s)(sW)$ 把激活通道的极端幅度搬给权重，迁移比例由 α 控制，结果是激活变得可量化，走 W8A8 INT8 Tensor Core，适合云端大 batch。AWQ 做**显著通道保护**——不动激活（保持 FP16），只对那 1% 重要权重通道做缩放放大再量化，让重要通道的相对误差变小，走 W4A16，适合端侧与显存受限场景。两者本质都是"等价变换 + 难度重分配"，方向相反：一个迁激活，一个保权重。

---

## 本周交付清单

- [ ] 闭卷推导 SmoothQuant 等价变换与 $s_j$ 公式，解释 $\alpha$ 两端（0 与 1）的含义。
- [ ] 跑通手写 `awq_search_scale`，断言通过并截图 `awq_salient_channels.png`。
- [ ] 云上完成 AWQ-4bit 与 SmoothQuant-W8A8 双量化，填写三指标对比表。
- [ ] 实测验证"b=1 AWQ 快、b=32 W8A8 快"的边界翻转。
- [ ] 用自己的话区分两派：难度迁移（SmoothQuant）vs 显著通道保护（AWQ）（≤ 100 字）。
