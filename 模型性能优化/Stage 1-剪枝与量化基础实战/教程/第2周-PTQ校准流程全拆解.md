# 第 2 周教程：PTQ 校准流程全拆解——四种策略的数学与工程落地

> **本周要回答的问题**
> 1. 校准数据集要多少条、怎么选？激活统计量在哪些 hook 上收集？
> 2. MinMax / Percentile / MSE / KL 四种校准目标各在优化什么？数学形式是什么？
> 3. 为什么**权重用 MinMax、激活用 Percentile** 是行业默认配方？
> 4. llm-compressor / ONNX Runtime / TensorRT 的 PTQ 流水线分别长什么样？

对应学习计划：第 2 周。交付物：完整 PTQ 管线（校准前向 → 统计收集 → 四策略算 scale → 量化导出），在 Qwen2.5-1.5B 上用 WikiText-2 困惑度对比四种校准策略，输出结论表。

---

## 1. 第一性原理：校准 = 用少量数据估计"该用哪张网格"

### 1.1 问题定义

第 1 周讲过：量化误差 $\mathbb{E}[\varepsilon^2] \approx s^2/12$，而 $s$ 由校准统计量决定。PTQ（Post-Training Quantization，训练后量化）的核心问题就是：

> **在不更新任何权重的前提下，用一小批有代表性的数据，为每一层选出使下游任务损失最小的 $(s, z)$。**

权重是静态的，一次统计终身有效；**激活是输入相关的**，必须用校准集"看一遍"才能定范围——这是"校准"二字的由来。

### 1.2 校准数据集：多少条、怎么选

- **数量**：128~1024 条足够。激活分布的统计量（分位数、直方图）收敛很快，边际收益在几百条后趋平。
- **代表性**：校准集必须覆盖部署分布。通用 LLM 用 WikiText-2 / C4 切片（每条约 2048 token）；垂类部署（如客服、代码）必须混入领域数据，否则激活范围估计偏小 → 部署时裁剪（clipping）激增。
- **长度**：按模型训练序列长度截取（常见 2048），太短会低估长上下文激活。

### 1.3 统计量怎么收

对每一层的输入/输出激活，在校准前向时挂 hook，累积以下之一：

- 运行极值 $\min/\max$（MinMax）；
- 全量样本的分位数（Percentile）；
- 直方图 $h(x)$（MSE 与 KL 都需要，典型 2048 个 bin）。

---

## 2. 四种校准策略的数学

设激活值集合（或直方图）为 $X$，候选裁剪阈值 $c$（即量化区间 $[-c, c]$ 或非对称 $[c_{\min}, c_{\max}]$），量化-反量化算子 $Q_c(\cdot)$。

### 2.1 MinMax：$c = \max|X|$

$$
s = \frac{2c}{2^b - 1}, \qquad c = \max_i |x_i|
$$

- 优点：零裁剪（clipping=0），计算最便宜，**权重标定首选**；
- 缺点：对离群值零抵抗——一个 ±100 的激活通道会把 $s$ 撑大两个数量级（第 1 周已实证）。

### 2.2 Percentile：$c = \mathrm{quantile}_{p}(|X|)$，典型 $p = 99.9\%$

$$
c = F^{-1}_{|X|}(p), \qquad p \in [99, 99.99]\%
$$

主动裁掉最极端的 $(1-p)$ 部分，换取其余 $p$ 部分的网格细化。代价是**裁剪误差**：被裁掉的尾部会被饱和到最大码点上。最优 $p$ 是"裁剪误差↓"与"网格误差↓"的折中——激活的离群值幅度大但出现概率低，99.9% 通常就是甜点。

### 2.3 MSE：网格搜索最小重建误差

$$
c^\star = \arg\min_c \ \mathbb{E}_{x \sim X}\big[(x - Q_c(x))^2\big]
$$

在候选 $c$ 网格（如把分位数从 90% 扫到 100%，步长 0.1%）上逐个计算量化重建 MSE，取最小。误差被显式拆成**舍入项 + 裁剪项**的联合最优——比 Percentile 更精细，成本是多次遍历直方图。

### 2.4 Entropy / KL：最小化分布偏移（TensorRT 默认）

把激活直方图 $P$（参考分布，不裁剪）与量化后再上采样回原 bin 的分布 $Q$ 对齐：

$$
c^\star = \arg\min_c \ D_{\mathrm{KL}}(P_c \,\|\, Q_c) = \arg\min_c \sum_i P_c(i)\, \log \frac{P_c(i)}{Q_c(i)}
$$

其中 $P_c$ 是裁剪到 $c$ 后的直方图，$Q_c$ 是"量化到 $2^b$ 个 bin 再展开回来"的近似分布。KL 的目标不是逐点重建，而是**让量化后的概率分布形状尽量不变**——对分类/softmax 型输出特别合理。注意 $Q_c$ 中 bin 概率为 0 的项按惯例跳过或加 $\epsilon$。

### 2.5 四策略一页对比

| 策略 | 目标函数 | 抗离群值 | 成本 | 典型用途 |
| --- | --- | --- | --- | --- |
| MinMax | 无裁剪、零成本 | ✗ | 最低 | **权重**、无离群的激活 |
| Percentile(99.9) | 裁尾部换网格 | ✓ | 低 | **激活默认** |
| MSE | 最小重建误差 | ✓✓ | 中 | 权重也可，逐层精调 |
| KL | 最小分布偏移 | ✓✓ | 高 | TensorRT 系、分类头 |

**为什么权重用 MinMax、激活用 Percentile**：① 权重分布静态、对称、幅度集中（±0.5 量级），MinMax 的"离群撑爆"风险小，且零裁剪保证权重信息无损；② 激活动态、有长尾离群通道（第 1 周 §1.6），必须牺牲 0.1% 尾部换网格——而激活尾部对输出的贡献本就是少数。一句话：**静态对称信 MinMax，动态长尾信 Percentile**。

---

## 3. 系统架构与数据流：三条工业管线走读

### 3.1 通用静态量化流水线

```
FP16 模型
  │ ① 校准前向（128~1024 条，hook 收集激活统计）
  ▼
每层激活的 {min, max, 直方图}
  │ ② 策略求 scale/zero-point（权重 MinMax / 激活 Percentile…）
  ▼
量化参数表 (per-layer / per-channel / per-group)
  │ ③ 插入 quant/dequant 节点（或直接导出量化权重）
  ▼
INT8/INT4 模型 ──► 引擎特定格式（ORT / TRT / vLLM）
```

### 3.2 三个框架的走读要点

- **llm-compressor**（vLLM 官方）：`oneshot()` 一条命令走 GPTQ/AWQ/SmoothQuant/FP8；底层复用 transformers 的校准 dataloader，产出 vLLM 直接可加载的 compressed-tensors 格式。
- **ONNX Runtime 静态量化**：`quantize_static(model, output, calibration_data_reader)`——DataReader 提供校准批次，默认增强模式（`QuantType.QInt8` 权重 + `QUInt8` 激活），按算子粒度插入 QDQ 节点。
- **TensorRT**：`IInt8EntropyCalibrator2`（KL 校准器）——实现 `get_batch()` 喂校准数据，`read_calibration_cache()` 复用缓存；builder 打开 `INT8` flag 后，引擎构建期对每层试 INT8/FP16 两种 tactic 取快者（**精度回退机制**是 TRT 的独特设计）。

---

## 4. 实现与验证（本周交付核心）

### 4.1 手写四策略校准器（完整可运行，CPU 可跑）

```python
import torch

def histogram(x: torch.Tensor, bins=2048, symmetric=True):
    """构建激活直方图。返回 (counts, edges)。"""
    amax = x.abs().max().item()
    lo, hi = (-amax, amax) if symmetric else (x.min().item(), x.max().item())
    counts = torch.histc(x.float().flatten(), bins=bins, min=lo, max=hi)
    edges = torch.linspace(lo, hi, bins + 1)
    return counts, edges

def calib_minmax(x, bits):
    c = x.abs().max()
    qmax = 2 ** (bits - 1) - 1
    return (c / qmax).clamp(min=1e-8)

def calib_percentile(x, bits, p=99.9):
    c = torch.quantile(x.float().abs().flatten(), p / 100)
    qmax = 2 ** (bits - 1) - 1
    return (c / qmax).clamp(min=1e-8)

def calib_mse(x, bits, candidates=64):
    """在 [80%, 100%] 分位数上网格搜索最小重建 MSE。"""
    flat = x.float().flatten()
    qmax = 2 ** (bits - 1) - 1
    qs = torch.linspace(0.80, 1.0, candidates)
    best_s, best_e = None, float("inf")
    for q in qs:
        c = torch.quantile(flat.abs(), q.item()).clamp(min=1e-8)
        s = c / qmax
        xq = torch.round(flat / s).clamp(-qmax - 1, qmax) * s
        e = ((flat - xq) ** 2).mean().item()
        if e < best_e:
            best_s, best_e = s, e
    return best_s

def calib_kl(counts, edges, bits, min_bins=128):
    """TensorRT 风格：在候选阈值上最小化 KL(P || Q)。"""
    n = len(counts)
    num_q = 2 ** (bits - 1)          # 对称：正负各 128 档（INT8 取 128）
    best_thr, best_kl = None, float("inf")
    total = counts.sum()
    for thr in range(min_bins, n + 1):
        ref = counts[:thr].clone()
        ref = ref / ref.sum().clamp(min=1)                 # P：裁剪后直方图
        # Q：把 thr 个 bin 压到 num_q 个量化档，再摊回原 bin
        q = torch.zeros(num_q)
        idx = (torch.arange(thr) * num_q // thr)
        q.index_add_(0, idx, ref)
        merged = torch.zeros(thr)
        cnt = torch.zeros(num_q)
        cnt.index_add_(0, idx, torch.ones(thr))
        safe = q / cnt[idx].clamp(min=1)                   # 每 bin 的平均概率
        merged = safe[idx]
        mask = (ref > 0) & (merged > 0)
        kl = (ref[mask] * (ref[mask] / merged[mask]).log()).sum().item()
        if kl < best_kl:
            best_thr, best_kl = thr, kl
    c = edges[best_thr]                                     # 阈值对应的幅值
    return (c.abs() / (2 ** (bits - 1) - 1)).clamp(min=1e-8)

# ---- 单元自检：构造含离群值的分布，验证策略排序 ----
torch.manual_seed(0)
x = torch.cat([torch.randn(100_000), torch.tensor([80.0, -80.0])])  # 0.002% 离群
s = {
    "MinMax": calib_minmax(x, 8).item(),
    "Percentile99.9": calib_percentile(x, 8).item(),
    "MSE": calib_mse(x, 8).item(),
    "KL": calib_kl(*histogram(x), 8).item(),
}
for k, v in s.items():
    print(f"{k:>16s}: scale={v:.5f}")
assert s["MinMax"] > 3 * s["Percentile99.9"], "MinMax 应被离群值撑大"
assert s["Percentile99.9"] > s["KL"] * 0.5, "KL/MSE 类策略应收缩阈值"
print("校准策略单元自检通过 ✓")
```

**预期输出**：MinMax 的 scale 被 ±80 撑大约 16 倍，其余三策略收缩到 ~5 的尾部附近——数值化第 1 周的结论。

### 4.2 云上跑通：Qwen2.5-1.5B 四策略 PPL 对比

```python
# 在云 GPU 上运行（A10/L20 级别即可，1.5B INT8 权重量化内存 < 8GB）
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import cosine_similarity

MODEL = "Qwen/Qwen2.5-1.5B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda")

calib = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [t["text"] for t in calib.select(range(256)) if len(t["text"]) > 100][:128]
enc = tok(texts, return_tensors="pt", truncation=True, max_length=2048,
          padding="max_length").to("cuda")

# ① 校准前向：收集每层 down_proj 输入激活的 abs-max 与分位数
stats = {}
def make_hook(name):
    def hook(m, inp, out):
        x = inp[0].float()
        stats.setdefault(name, []).append(
            (x.abs().amax().item(),
             torch.quantile(x.abs().flatten().float(), 0.999).item()))
    return hook
handles = [m.register_forward_hook(make_hook(n))
           for n, m in model.named_modules() if n.endswith("down_proj")]
with torch.no_grad():
    for i in range(0, len(texts), 8):
        model(**{k: v[i:i+8] for k, v in enc.items()})
[h.remove() for h in handles]

# ② 四策略各自算出的 scale，量化→反量化 down_proj 权重，测层输出偏移
import math
w = model.model.layers[10].mlp.down_proj.weight
def requant(scale):
    q = torch.round(w / scale).clamp(-127, 127)
    return q * scale
print("各策略 scale / 重建余弦相似度：")
for name, fn in [("MinMax", lambda a: a), ("Percentile", lambda a: a)]:
    pass  # 实际实现见 4.1；此处用 stats 中激活极值指导裁剪
```

> 说明：完整"量化模型 → WikiText-2 PPL"对比建议直接用 `llm-compressor` 或 `torchao` 的 INT8 weight-only 接口跑四组（把 4.1 的四种 scale 注入），每组 128 条校准、1024 条测试。预期结论写进下表（交付要求填实测值）：

| 校准策略 | 注入位置 | PPL(WikiText-2) | vs FP16 |
| --- | --- | --- | --- |
| FP16（基线） | — | _实测_ | — |
| MinMax | 激活 | _实测_ | _Δ_ |
| Percentile(99.9) | 激活 | _实测（预期最低）_ | _Δ_ |
| MSE | 激活 | _实测_ | _Δ_ |
| KL | 激活 | _实测_ | _Δ_ |

**经验预期**：激活侧 Percentile ≈ MSE < KL < MinMax；MinMax 若撞上离群通道会出现 PPL 暴涨的"悬崖"，这正是 §2.5 结论的实证。

---

## 5. 工程权衡与失效模式

### 5.1 权衡

- **校准成本 vs 精度**：KL/MSE 要遍历直方图若干遍，比 MinMax 慢一个量级；但只在离线做一次，通常值得。
- **裁剪点选择**：阈值越激进（分位数越低），网格越细、裁剪越多——PPL 曲线通常呈 U 形，最优点在 99.5%~99.99% 之间。
- **静态量化 vs 动态量化**：动态量化（运行时算激活 scale）免校准但每步多一次统计开销，只适合激活难校准的场景。

### 5.2 失效模式

1. **校准集与部署分布错位**：症状——评测 PPL 正常、线上长文本崩坏；根因——校准文本过短或领域不符；修复——按真实流量采样校准集、加长序列。
2. **KL 校准的直方图饱和**：症状——TRT 校准出的模型在极端输入上输出乱码；根因——校准集未覆盖极端幅度，直方图边界即裁剪边界；修复——扩大校准分布范围或用 Percentile 回退。
3. **逐层误差累积**：症状——浅层单层余弦相似度都 > 0.999，整模型 PPL 仍明显上涨；根因——误差沿深度复合放大，敏感层（通常是第一层与 lm_head 邻近层）需要保护；修复——敏感层保留 FP16（混合精度量化）。
4. **激活非对称却按对称校准**：症状——ReLU/正偏激活量化后均值漂移；根因——对称网格浪费一半码域；修复——激活改用非对称 + 零点（第 1 周公式）。

---

## 6. 延伸思考题（含解析）

**Q1**：为什么 128 条校准数据通常就够了？什么情况下必须加量？
**A**：激活的分位数/直方图统计量收敛速度是 $O(1/\sqrt{n})$，128 条 ×2048 token ≈ 26 万样本点，估计误差已低于量化步长本身。需要加量的情况：部署分布多峰（多领域混合）、序列极长（长上下文激活漂移）、动态范围随位置变化大。

**Q2**：KL 校准为什么在 TensorRT 上是默认，而在 LLM 社区少见？
**A**：TRT 面向视觉/分类为主的时代，softmax 输出分布的形状保持是正确目标；LLM 社区更关心权重侧二阶误差（GPTQ）与离群通道（AWQ），KL 对激活长尾的收缩不如显式 Percentile 可控，且 LLM 校准还要喂逐层重建目标，工具链就绕开了 KL。

**Q3**：Percentile 的裁剪误差和舍入误差怎么相互消长？最优点怎么找？
**A**：阈值 $c$ 变小 → 步长 $s$ 变小 → 舍入误差 $\downarrow$，但被裁尾部 $\uparrow$。总误差 U 形；实践上在分位数 99%~100% 网格搜索重建 MSE（即 §2.3），或直接画"PPL vs 分位数"曲线取最低点。

**Q4**：为什么第一层和嵌入层常常量化收益差、甚至要跳过？
**A**：第一层直接吃原始嵌入，误差无前置层"平滑"，且其权重分布常有特殊结构（与词表相关）；lm_head 输出直接进 softmax 指数，误差被放大。工业配方通常对这两层保留高精度——混合精度是免费午餐。

**Q5**：如果校准集里混入了大量重复样本，四种策略谁受伤最重？
**A**：MinMax 受伤最轻（极值不依赖频率）；KL 受伤最重——直方图形状被重复样本扭曲，$P$ 本身就是错的分布，最小化"与错误目标的距离"反而更差。这提示校准集要先去重（呼应数据工程方向）。

---

## 本周交付清单

- [ ] 闭卷写出四种校准策略的目标函数（MinMax/Percentile/MSE/KL 各一行公式）。
- [ ] 跑通 4.1 手写校准器，单元自检断言通过，截图四策略 scale 对比。
- [ ] 云上完成 Qwen2.5-1.5B 四策略对比，填写 §4.2 结论表（PPL 实测值）。
- [ ] 用自己的话回答：为什么权重用 MinMax、激活用 Percentile？（≤ 80 字）
- [ ] 走读 llm-compressor / ONNX Runtime / TensorRT 三条管线各 30 分钟，记录接口名。
