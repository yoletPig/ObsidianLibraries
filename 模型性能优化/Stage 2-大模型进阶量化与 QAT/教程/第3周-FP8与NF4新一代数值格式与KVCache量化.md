# 第 3 周教程：FP8 与 NF4——新一代数值格式与 KV Cache 量化

> **本周要回答的问题**
> 1. FP8 E4M3/E5M2 的位级构成是什么？为什么前向用前者、反向用后者？
> 2. Transformer Engine 的延迟缩放（Delayed Scaling）在解决什么问题？
> 3. NF4 码本为什么取正态分布的分位数中点？"双重量化"省了多少显存？
> 4. KV Cache 为什么是长上下文的最后战场？KIVI 的 INT2 凭什么可行？

对应学习计划：第 3 周。交付物：① 手写 NF4 码本生成器，与 `bitsandbytes` 的 `create_normal_map()` 数值对齐；② 云上 `bitsandbytes` 4-bit 加载 + LoRA 微调，记录显存与收敛曲线，验证"7B 单卡 24G 可训"。

---

## 1. 第一性原理：当均匀网格不够用，就换数值格式本身

INT4 均匀网格的 16 个码点是"平均分配"的——但 LLM 权重分布是**中间密、两端疏**的近似正态。两条出路：① 换**浮点**格式（FP8：指数位自动给两端稀疏、中间密集的相对精度）；② 换**定制非均匀**格式（NF4：码点直接按权重分布的分位数摆放）。

### 1.1 FP8 位级解析

FP8 两种格式（IEEE-like，含符号位）：

| 格式 | 指数/尾数 | 最大正规值 | 特点 |
| --- | --- | --- | --- |
| E4M3 | 4 / 3 | **448** | 尾数多 → 相对精度高；范围小 |
| E5M2 | 5 / 2 | **57344** | 指数多 → 范围大；尾数粗 |

手工解码一个 E4M3 数（符号 $s$、指数 $e$、尾数 $m$）：

$$
v = (-1)^s \cdot 2^{e-7} \cdot \left(1 + \frac{m}{8}\right) \quad (e \ge 1), \qquad
v = (-1)^s \cdot 2^{-6} \cdot \frac{m}{8} \quad (e = 0,\ \text{次正规})
$$

例：E4M3 `0 1000 010` → $e=8, m=2$ → $2^{8-7}\cdot(1+2/8) = 2 \times 1.25 = 2.5$。

**为什么分工是"前向 E4M3、反向 E5M2"**：权重与前向激活幅度集中在 ±448 内，需要尾数精度；梯度跨多个数量级且偶发极大值，需要 E5M2 的动态范围。这是 *FP8 Formats for Deep Learning*（NVIDIA 白皮书）确立的标准配方。

### 1.2 Transformer Engine 的延迟缩放

FP8 GEMM 需要把张量缩放到格式的可表示范围（否则上溢/下溢）。**延迟缩放（Delayed Scaling）**：用**前几个迭代记录到的 amax 历史**来估计当前迭代的缩放因子，而不是当场全量统计——

$$
\text{scale}_t = f\big(\max(\text{amax}_{t-1}, \text{amax}_{t-2}, \dots)\big)
$$

理由：① 当场统计要多一次全张量归约（抵消 FP8 的速度收益）；② 相邻迭代间激活分布变化平缓，历史 amax 是良好估计。风险：分布突变时首步溢出 → loss spike，需要 amax 历史窗口与梯度裁剪兜底。**FP8 训练配方**：主权重（master weights）与优化器状态仍保 FP32，FP8 只出现在 GEMM 的前向/反向矩阵上。

### 1.3 NF4：信息论最优的 4-bit 码本

QLoRA（Dettmers et al., 2023）的 NormalFloat 4 构造：

1. **假设**权重 $w \sim \mathcal{N}(0, \sigma^2)$（实践上先做分块归一化——每块除以该块绝对均值，强制满足）；
2. 取标准正态的 $2^4 + 1 = 17$ 个**等概率分位点** $q_i = \Phi^{-1}\!\left(\frac{i}{17}\right)$（把概率质量等分成 16 份）；
3. 相邻分位点**取中点**作为码点，并显式包含 0；归一化到 $[-1, 1]$。

$$
c_i = \frac{1}{2}\left(\Phi^{-1}\!\left(\tfrac{i}{17}\right) + \Phi^{-1}\!\left(\tfrac{i+1}{17}\right)\right), \quad i = 0,\dots,15
$$

**为什么这是"信息论最优"**：对已知分布 $p$，使期望量化误差 $\mathbb{E}[(w - Q(w))^2]$ 最小的码点，正是使每个码点覆盖等概率区间的分位数中点（标量量化的经典结论，Lloyd-Max 在高斯下的特例）。0 附近概率密度最高 → 码点最密；尾部稀疏 → 码点稀——恰好匹配正态形状。

### 1.4 双重量化（Double Quantization）

NF4 量化后每 64 个权重配一个 FP32 scale（常数块大小）。QLoRA 再把这个 scale 本身量化成 8-bit（256 个块共享一个 FP32 二级 scale）：

$$
\text{额外开销}:\ \frac{32\ \text{bit}}{64} = 0.5\ \text{bpw} \ \xrightarrow{\text{双重量化}}\ \frac{8}{64} + \frac{32}{64 \times 256} \approx 0.127\ \text{bpw}
$$

7B 模型省下约 $(0.5 - 0.127) \times 7\times10^9 / 8 \approx 0.4$ GB——不多，但**足以决定 7B 能否塞进 24G 卡**（配合优化器状态的 CPU offload）。

### 1.5 KV Cache 量化：长上下文的最后战场

decode 时 KV cache 体积 = $2 \times n_{layers} \times n_{heads} \times d_{head} \times seq \times \text{bytes}$。7B、4K 上下文、FP16 ≈ 4 GB——与权重本身同量级，且**随序列长度线性增长**。两条路线：

- **FP8 KV cache**：vLLM/TRT-LLM 已支持，体积减半、精度损失通常 < 0.1 PPL；
- **KIVI**（Liu et al., 2024）：Key 沿**通道维**、Value 沿 **token 维**做 2-bit 量化（两者离群结构方向不同：Key 离群在通道、Value 离群在早期 token），把 KV 压到 2.6× 以下，支持 4× 长上下文。

---

## 2. 系统架构与数据流

```
NF4/QLoRA 数据流：
HF 权重 ──► 分块(64) ──► 每块除以 |mean| ──► NF4 码点查表 ──► (q4, scale_fp32)
                                                     │
                             双重量化：scale_fp32 ──► q8 + 二级 scale
推理/训练时：q4 ──► 码点查表 ──► × 块 scale ──► FP16 前向
训练梯度：只更新 LoRA 分支（主权重冻结在 NF4）
```

---

## 3. 实现与验证（本周交付核心）

### 3.1 手写 NF4 码本生成器（与 bitsandbytes 对齐）

```python
import torch
from scipy.stats import norm

def nf4_codebook():
    """正态分位数中点码本，返回 16 个码点（升序，含 0，归一化到 [-1, 1]）。"""
    # 等概率分界点：把概率质量 17 等分
    quantiles = norm.ppf(torch.linspace(0, 1, 18)[1:-1].numpy())  # 15 个内部分界
    # 码点 = 相邻分界中点；首尾取分布截断代表值
    mids = [(quantiles[i] + quantiles[i+1]) / 2 for i in range(len(quantiles)-1)]
    code = [quantiles[0] - 1.0] + mids + [quantiles[-1] + 1.0]   # 16 个
    code = torch.tensor(code)
    code[code.abs().argmin()] = 0.0                 # 显式包含 0（QLoRA 做法）
    return code / code.abs().max()                  # 归一化到 [-1, 1]

mine = nf4_codebook()

# ---- 与 bitsandbytes 官方数值对齐 ----
import bitsandbytes.functional as F
bnb = F.create_normal_map(offset=0.9677083, use_extra_value=False)  # 官方 16 码点
print("mine:", [f"{v:.4f}" for v in mine.tolist()])
print("bnb :", [f"{v:.4f}" for v in bnb.tolist()])
diff = (mine.sort().values - bnb.sort().values).abs().max().item()
assert diff < 2e-2, f"码本应对齐，实际最大偏差 {diff}"
print(f"NF4 码本对齐 ✓（最大偏差 {diff:.2e}）")
```

**预期输出**：两套码点排序后逐位偏差 < 2e-2（bitsandbytes 用数值积分 + 对称微调，中点法与其在第三位小数上略有出入属正常——把偏差来源写进实验记录）。

### 3.2 QLoRA 云端微调（验证"7B 单卡 24G 可训"）

```python
# 云环境：单卡 RTX 4090/A10G-24G 即可
import torch, bitsandbytes as bnb
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                  # ← 本周主角
    bnb_4bit_use_double_quant=True,             # ← 双重量化
    bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B", quantization_config=bnb_cfg)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
                    target_modules="all-linear", task_type="CAUSAL_LM"))
model.print_trainable_parameters()
# 预期：trainable ≈ 0.3%（LoRA 分支），冻结主体 99.7%

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
data = load_dataset("tatsu-lab/alpaca", split="train[:5000]")
# （tokenizer 编码、Trainer 配置从略——复用 Stage 1 蒸馏周的 collate）
```

**记录三件事**（交付）：① `nvidia-smi` 峰值显存（预期 ~20 GB，< 24G 断言成立）；② loss 曲线前 500 步收敛趋势；③ `print_trainable_parameters()` 的百分比。三者合起来就是对"7B 单卡 24G 可训"的完整实证。

### 3.3 FP8 位级解码小练习（可运行）

```python
def e4m3_decode(bits: int) -> float:
    s = (bits >> 7) & 1; e = (bits >> 3) & 0xF; m = bits & 0x7
    if e == 0:
        return (-1)**s * 2**(-6) * (m / 8)
    return (-1)**s * 2**(e - 7) * (1 + m / 8)

assert abs(e4m3_decode(0b0_1000_010) - 2.5) < 1e-9
assert e4m3_decode(0b0_1111_111) == 448          # E4M3 最大值
print("E4M3 位级解码 ✓：0b0_1000_010 → 2.5，全 1 → 448")
```

---

## 4. 工程权衡与失效模式

### 4.1 权衡

- **FP8 训练 vs BF16**：Hopper/Ada 上 GEMM 吞吐 ~2×，但需要 Transformer Engine 生态与缩放调参；老硬件无 Tensor Core 支持就完全没收益。
- **NF4 vs INT4**：NF4 免校准、分布匹配好，但**没有配套的高速推理内核**（bitsandbytes 推理偏慢）——它是**训练/微调格式**，部署时通常转 GPTQ/AWQ/GGUF。
- **KV 量化位宽**：FP8 稳妥（几乎无损）；INT2（KIVI）激进，需按任务验收长文本质量。

### 4.2 失效模式

1. **FP8 loss spike**：症状——训练中途 loss 突跳；根因——延迟缩放估计滞后于分布突变；修复——加大 amax 历史窗口、梯度裁剪、必要时回退该层 BF16。
2. **NF4 假设失配**：症状——某些层（如稀疏化后、特定蒸馏模型）NF4 掉点比 INT4 还多；根因——权重偏离正态；修复——检查分块归一化是否生效、对该层换格式。
3. **4-bit 加载后首层乱码**：症状——生成乱码但 loss 正常；根因——`compute_dtype` 与设备不匹配（如老卡不支持 bf16）；修复——改 `float16`。
4. **KV INT2 长文本崩坏**：症状——短问答正常、超过某长度后幻觉激增；根因——早期 token 的 KV 离群值被 2-bit 截断（Value 侧尤甚）；修复——回退 FP8 或对敏感层保留高精度（KIVI 的通道/Token 维选择正是为此设计）。

---

## 5. 延伸思考题（含解析）

**Q1**：为什么 FP8 不能只用一种格式？
**A**：指数位与尾数位是零和的：给范围（E5M2）就丢精度，给精度（E4M3）就丢范围。前向数值幅度集中 → 要精度；反向梯度跨数量级 → 要范围。一种格式无法同时满足两个分布。

**Q2**：延迟缩放相比"当场统计缩放"省了什么、赌了什么？
**A**：省了一次全张量归约（否则吃掉 FP8 GEMM 的大部分收益）；赌的是"相邻迭代分布变化平缓"。分布平稳时免费，突变时首步溢出——所以要有 amax 历史与裁剪兜底。

**Q3**：NF4 码本为什么必须显式包含 0？
**A**：正态分布在 0 处概率密度最高，大量权重会被量化到 0 附近；若码点不含 0，最接近 0 的两个码点 ±ε 会引入系统性偏置（符号随机化噪声），含 0 码点让"该是零的权重就是零"。

**Q4**：双重量化的收益为什么在 7B/24G 场景特别关键？
**A**：省下的 ~0.4 GB 看似小，但 24G 卡的预算表里：权重 3.5G + KV + 激活 + LoRA 梯度 + 优化器已逼近上限——0.4G 正是"能训"与"OOM"的分界线。收益大小取决于离预算边界的距离。

**Q5**：Key 和 Value 为什么用不同维度做超细量化？
**A**：离群结构不同：Key 的大幅度值集中在固定**通道**（跨 token 稳定），Value 的离群集中在**早期/特殊 token**（跨通道）。沿各自离群方向做分组量化才能隔离极端值——KIVI 的通道维（Key）/token 维（Value）选择是数据观察的结果，不是对称美学。

---

## 6. KV Cache 预算计算器（可运行，长上下文必备）

```python
def kv_cache_gb(n_layers, n_kv_heads, d_head, seq_len, batch,
                dtype_bytes=2.0):
    """KV cache 体积：2(K+V) × 层数 × 头数 × 头维 × 序列长 × batch × 字节。"""
    return 2 * n_layers * n_kv_heads * d_head * seq_len * batch * dtype_bytes / 1e9

# Qwen2.5-7B：28 层、28 个 KV 头、头维 128
for seq in (4096, 32768):
    for fmt, b in [("FP16", 2), ("FP8", 1), ("KIVI-INT2", 0.25)]:
        gb = kv_cache_gb(28, 28, 128, seq, batch=1, dtype_bytes=b)
        print(f"seq={seq:>6}, {fmt:>9}: KV = {gb:5.2f} GB")
# 预期：4K FP16 ≈ 2.1 GB → FP8 减半 → INT2 再减 4×
# 断言：FP8 恰好是 FP16 的一半
fp16 = kv_cache_gb(28, 28, 128, 4096, 1, 2)
fp8 = kv_cache_gb(28, 28, 128, 4096, 1, 1)
assert abs(fp16 / fp8 - 2.0) < 1e-9
print("KV 预算计算自检通过 ✓")
```

**结论用法**：长上下文服务里，KV 很快超过权重本身成为显存大头——这就是为什么 vLLM 的 PagedAttention 管 KV、KIVI/FP8 压 KV，两者解决的是同一个资源问题（分页管"碎片"，量化管"总量"）。

### 6.1 数值格式全景一页表（复习锚点）

| 格式 | 类别 | 码点分布 | 校准需求 | 主战场 |
| --- | --- | --- | --- | --- |
| FP8 E4M3/E5M2 | 浮点 | 对数稀疏（指数位） | 缩放策略（延迟缩放） | 训练 + H 卡推理 |
| INT8/INT4 | 均匀整数 | 等距 | 需要（定 scale） | 通用 PTQ |
| NF4 | 非均匀码本 | 正态分位数 | 免（分布假设） | 训练/微调（QLoRA） |
| IQ 系列 | 非均匀码本 | imatrix 加权码点 | 需要（imatrix） | CPU 极限压缩（Stage 4） |

四个家族的共同主线：**码点布局匹配数值分布**。分布越规整（旋转后/正态假设后），同样的位宽能承载越多的信息——这是贯穿 Stage 1-2 的一条暗线。

---

## 7. FP8 训练配方速查（配合 Transformer Engine）

| 组件 | 精度 | 理由 |
| --- | --- | --- |
| 主权重（master） | FP32 | 优化器状态与权重更新要高精度 |
| 前向 GEMM 输入 | E4M3 | 幅度集中，精度优先 |
| 反向梯度 | E5M2 | 跨数量级，范围优先 |
| 优化器状态 | FP32（或 ZeRO 下分片） | 数值稳定底线 |
| 缩放策略 | 延迟缩放 + amax 历史窗口 | 避免当场统计开销 |

**loss spike 应对清单**（出现时按序排查）：① amax 历史窗口是否太短；② 梯度裁剪阈值是否过松；③ 该层是否应回退 BF16（混合粒度到层）；④ 数据批是否有异常样本。

### 7.1 本周交付的实验记录模板

| 实验 | 关键数字 | 断言/结论 |
| --- | --- | --- |
| NF4 码本对齐 | 最大偏差 _实测_ | < 2e-2 |
| E4M3 解码练习 | 2.5 与 448 | 断言通过 |
| QLoRA 7B 微调 | 峰值显存 _实测_ | < 24 GB |
| KV 预算计算 | 4K/32K 三档 | FP8 恰为 FP16 一半 |

---

## 8. 面试快问快答（本周内容压缩版）

**Q：E4M3 和 E5M2 怎么选？**
前向/权重用 E4M3（尾数 3 位、精度优先、范围 448 够）；反向梯度用 E5M2（指数 5 位、范围 57344、跨数量级）。一种格式无法同时满足两个分布。

**Q：延迟缩放省了什么、赌了什么？**
省一次全张量归约（否则吃掉 FP8 GEMM 收益）；赌相邻迭代分布平稳。突变时首步溢出，用 amax 历史 + 梯度裁剪兜底。

**Q：NF4 为什么信息论最优？**
码点取正态分布的等概率分位数中点，使期望量化误差最小——前提是权重近似正态（QLoRA 用分块归一化强制满足）。

**Q：双重量化省多少？**
每 64 权重的 FP32 scale 从 0.5 bpw 压到 ~0.127 bpw，7B 省 ~0.4 GB——恰是 7B 能否塞进 24G 卡的分界线。

**Q：KIVI 的 Key 和 Value 为什么用不同维度量化？**
Key 离群在固定通道 → 通道维分组；Value 离群在早期/特殊 token → token 维分组。沿各自离群方向隔离极端值。

**Q：为什么 FP8 KV 比 INT2 KV 更常用？**
FP8 几乎无损（< 0.1 PPL）且引擎原生支持；INT2 省得多但长文本崩坏风险高，需逐任务验收。稳妥与激进的取舍。

---

## 本周交付清单

- [ ] 手算解码一个 E4M3 数（写出 $2^{e-7}(1+m/8)$ 的代入过程）。
- [ ] 跑通 NF4 码本生成器，与 `bitsandbytes.create_normal_map()` 对齐（偏差 < 2e-2）。
- [ ] 云上跑通 QLoRA：记录峰值显存（< 24G）、可训练参数比例、500 步 loss 曲线。
- [ ] 写出双重量化的显存节省推导（0.5 → 0.127 bpw）。
- [ ] 用自己的话回答：NF4 为什么是"训练格式"而不是"部署格式"？
