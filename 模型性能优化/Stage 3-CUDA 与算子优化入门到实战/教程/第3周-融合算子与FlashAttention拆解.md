# 第 3 周教程：融合算子与 FlashAttention 拆解

> **本周要回答的问题**
> 1. 为什么"多个小算子融合成一个大算子"能快数倍？Kernel launch 与显存往返的账怎么算？
> 2. FlashAttention 的在线 softmax 递推式怎么推？为什么它不实例化 $N \times N$ 矩阵还不损失精度？
> 3. fused add + RMSNorm 的 CUDA kernel 长什么样？reduction 在块内怎么做？
> 4. CUTLASS 的 epilogue 概念与上周的 GEMM 是什么关系？

对应学习计划：第 3 周。交付物：① 手写 fused add+rmsnorm CUDA kernel，与 PyTorch 两步实现对比延迟（bs=32, seq=4096, hidden=4096）；② 手绘 FlashAttention 的 tiling 数据流图。

---

## 1. 第一性原理：融合 = 把"显存往返"压缩成"寄存器内流动"

### 1.1 不融合的代价

PyTorch 里 `y = rmsnorm(x + residual)` 是两个算子：每个算子都要 ① 启动一次 kernel（launch 开销 ~5-10 μs）；② 从显存读入、写回中间结果。对 $32 \times 4096 \times 4096$ 的张量（~2 GB @FP16），一次读+写 ≈ 1-2 ms——**纯搬运时间，零计算价值**。

融合后：数据读进来一次，在寄存器/共享内存里依次完成加、平方和归约、归一化，写出去一次：

$$
t_{\text{fused}} \approx \frac{2 \cdot \text{bytes}}{BW} \quad \text{vs} \quad
t_{\text{unfused}} \approx \frac{4 \cdot \text{bytes}}{BW} + 2 \times t_{\text{launch}}
$$

elementwise 类融合的理论收益就是 ~2× 起步（搬运减半），再叠加 launch 节省。这与第 1 周 Roofline 一脉相承：融合提高了算术强度，把算子从带宽墙左侧往右推。

### 1.2 RMSNorm 与其 reduction

RMSNorm（LLaMA/Qwen 系标配）：

$$
\hat{x}_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}} \cdot g_i
$$

注意它比 LayerNorm 少一个均值中心化——但**仍有一次跨 $d$ 维的归约**（平方和）。融合实现的关键：一行（一个 $d$ 维向量）交给一个块（或一个 warp），块内归约用共享内存或 warp shuffle 完成。

### 1.3 FlashAttention：在线 softmax 推导（必须闭卷）

标准 attention 要算 $\mathrm{softmax}(QK^\top/\sqrt{d})V$，朴素实现先物化 $N \times N$ 的分数矩阵——显存 $O(N^2)$、访存 $O(N^2)$。FlashAttention（Dao et al., 2022）的突破：**softmax 可以分块流式计算**。

设按块处理，已处理部分的局部最大值为 $m^{(t)}$、未归一化指数和为 $l^{(t)}$、未归一化输出为 $O^{(t)}$。新来一块分数 $S_{t+1}$ 时，递推：

$$
m^{(t+1)} = \max\big(m^{(t)},\ \mathrm{rowmax}(S_{t+1})\big)
$$

$$
l^{(t+1)} = e^{m^{(t)} - m^{(t+1)}}\, l^{(t)} + \mathrm{rowsum}\big(e^{S_{t+1} - m^{(t+1)}}\big)
$$

$$
O^{(t+1)} = \frac{1}{l^{(t+1)}}\Big( e^{m^{(t)} - m^{(t+1)}}\, l^{(t)}\, O^{(t)} + e^{S_{t+1} - m^{(t+1)}} V_{t+1} \Big)
$$

直觉：当新块里出现更大的 logit 时，旧的指数和与输出都要乘以修正因子 $e^{m^{(t)} - m^{(t+1)}} < 1$ "缩小"——这正是数值稳定 softmax 减去行最大值的在线版本。**全程不需要知道整行的最大值**，因此无需物化 $N \times N$ 矩阵：显存 $O(N)$，且结果与标准实现逐比特等价（非近似）。

### 1.4 FlashAttention 的 SRAM 数据流（本周交付图）

```
对 Q 的每个块（驻留 SRAM）：
  对 K、V 的每个块（流式载入 SRAM）：
    ① SRAM 内算 S = QᵢKⱼᵀ/√d        ← 分数矩阵只存在于 SRAM，不落 HBM
    ② 在线 softmax 递推（§1.3 三式）
    ③ 累加 O ← O + softmax块 × Vⱼ
  K/V 块循环结束 → O 归一化，写回 HBM
```

标注要点：Q 块常驻、K/V 流式、S 只在 SRAM 存活——画这张图并解释 $O(N^2) \to O(N)$ 的来源，是本周的口头答辩题。

---

## 2. 系统架构与数据流：推理引擎的融合清单

LLaMA 系一个 Transformer 块的标准融合配方：

```
x ─► [fused: add + RMSNorm] ─► QKV(一个 GEMM) ─► Attention(FlashAttn)
  ─► [fused: add + RMSNorm] ─► gate/up(GEMM) ─► [fused: SiLU×gate (SwiGLU)]
  ─► down(GEMM) ─► 残差 ─► 下一层
```

每一处方括号都是一次"读一次、写一次"的合并——这就是"推理引擎的速度之源"的具体清单。

---

## 3. 实现与验证（本周交付核心）

### 3.1 fused add + RMSNorm kernel（完整可编译）

```cuda
// fused_add_rmsnorm.cu —— 一行 = 一个 block；hidden 维归约走共享内存
#include <cstdio>
#include <cuda_runtime.h>

__global__ void fused_add_rmsnorm(float* x, const float* resid,
                                  const float* g, int d, float eps) {
    extern __shared__ float smem[];          // 动态共享内存：存一行 + 部分和
    float* row = smem;                       // 前 d 个浮点存残差加结果
    float* part = smem + d;                  // 块内部分和树
    int r = blockIdx.x;                      // 本块负责第 r 行
    int tid = threadIdx.x, bs = blockDim.x;

    // ① 残差加 + 平方和（网格跨步循环，兼容 hidden > blockDim）
    float ss = 0.f;
    for (int i = tid; i < d; i += bs) {
        float v = x[r * d + i] + resid[r * d + i];
        row[i] = v; ss += v * v;
    }
    // ② 块内归约（共享内存树）
    part[tid] = ss; __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) part[tid] += part[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(part[0] / d + eps);   // 1 / sqrt(mean square + eps)
    // ③ 归一化 × 增益，写回（唯一一次写出）
    for (int i = tid; i < d; i += bs)
        x[r * d + i] = row[i] * rms * g[i];
}
```

### 3.2 封装、对比与断言

```python
# fused_add_rmsnorm.py —— 用 torch cpp_extension 即时编译（云环境跑）
import torch, time
from torch.utils.cpp_extension import load_inline

cuda_src = open("fused_add_rmsnorm.cu").read().replace("#include <cstdio>\n", "")
mod = load_inline(name="fused_rms", cpp_sources="""
#include <torch/extension.h>
void launch(torch::Tensor x, torch::Tensor r, torch::Tensor g, double eps);
""", cuda_sources=cuda_src + """
void launch(torch::Tensor x, torch::Tensor r, torch::Tensor g, double eps) {
    int rows = x.size(0), d = x.size(1);
    int bs = 256;
    size_t smem = (d + bs) * sizeof(float);
    fused_add_rmsnorm<<<rows, bs, smem>>>(
        x.data_ptr<float>(), r.data_ptr<float>(), g.data_ptr<float>(), d, eps);
}
""", functions=["launch"], verbose=False)

B, S, H = 32, 4096, 4096          # 学习计划指定尺寸
x = torch.randn(B * S, H, device="cuda")
r = torch.randn_like(x); g = torch.randn(H, device="cuda")

def ref(x, r, g, eps=1e-6):       # PyTorch 两步基线
    y = x + r
    return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps) * g

y_ref = ref(x, r, g)
y_fused = x.clone(); mod.launch(y_fused, r, g, 1e-6)
assert torch.allclose(y_ref, y_fused, atol=1e-4, rtol=1e-3), "数值不一致"

def bench(fn, iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.time() - t) / iters * 1e3

t_ref = bench(lambda: ref(x, r, g))
t_fus = bench(lambda: mod.launch(y_fused, r, g))
print(f"PyTorch 两步: {t_ref:.3f} ms | fused: {t_fus:.3f} ms | 加速 {t_ref/t_fus:.2f}x")
# 预期：加速 1.5-2.5×（纯访存算子，融合收益接近理论 2× 上限）
```

```bash
# load_inline 会自动调 nvcc；手工编译命令等价于：
# nvcc -O3 -arch=sm_86 -Xcompiler -fPIC -shared fused.cu -o fused.so
```

**交付**：记录加速比与 `ncu` 抓的带宽利用率；融合版应更接近显存带宽峰值。

---

## 4. 工程权衡与失效模式

### 4.1 权衡

- **融合度的边界**：融合越多越省搬运，但单 kernel 变复杂、寄存器压力上升、调试困难；工业界把"高频 + 纯访存"的算子融合（norm/激活/残差），复杂逻辑仍独立成算子。
- **FlashAttention 的算力换访存**：在线递推让 FLOPs 略增（重复的缩放修正），但访存从 $O(N^2)$ 降到 $O(N)$——在访存瓶颈的硬件上稳赚；长序列上收益更大。
- **torch.compile 的自动融合**：PyTorch 2.x 的 Inductor 后端自动生成 Triton 融合内核，覆盖大部分 elementwise 组合——手写融合的价值在于它覆盖不了的（跨 GEMM 边界的 epilogue、定制 reduction）。

### 4.2 失效模式

1. **归约精度漂移**：症状——fused norm 与 PyTorch 结果差 > 1e-3；根因——FP16 累加平方和溢出/精度不足；修复——累加器用 FP32（本周代码 `float ss` 已如此）。
2. **共享内存超限**：症状——大 hidden（如 8192）时启动失败；根因——动态共享内存超每 SM 配额；修复——改网格跨步 + 寄存器累加、或分两遍（先算平方和再归一化）。
3. **FlashAttention 变体数值不一致**：症状——换实现后生成结果变了；根因——某些"快注意力"用近似 softmax（如省略修正项）；修复——确认是 exact 版本（递推式 §1.3 完整）。
4. **融合后无法单独调试**：症状——结果错但不知道错在加、归约还是缩放；修复——保留逐段开关（模板参数），出问题时降级为分步对照。

---

## 5. 延伸思考题（含解析）

**Q1**：FlashAttention"不损失精度"的关键是什么？它和近似注意力（Linear Attention 等）的本质区别？
**A**：在线递推式是 softmax 的**恒等代数变换**（结合律 + 最大值修正），无任何近似，结果与朴素实现数值等价（浮点顺序误差量级）。近似注意力则是改了数学（如核近似），是另一类方法，有真实精度代价。

**Q2**：为什么融合收益的理论上限约 2×（对两步算子）？什么时候能超过？
**A**：两步 = 两次读两次写 ≈ 4 份搬运；融合 = 一读一写 = 2 份 → 2×。超过 2× 的情况：① 省下的 kernel launch 在小张量上占比大；② 融合让中间结果留在寄存器（连共享内存都省）；③ 三步以上算子融合（SwiGLU 等）。

**Q3**：RMSNorm 相比 LayerNorm 省了什么？为什么 LLaMA 系选它？
**A**：省了均值中心化（一次归约 + 一次减法）。经验上归一化质量相当，但少一趟统计、融合更简单——对推理引擎这种"每一微秒都抠"的场景，免费的简化就是收益。

**Q4**：epilogue 融合与本周的算子融合是同一件事吗？
**A**：同一思想、不同位置：epilogue 特指"GEMM 结束后、结果还在寄存器时"顺手做后处理（bias/激活/残差），是融合的**最优时机**（搬运成本为零）；本周的 fused norm 是独立算子之间的融合。CUTLASS 把前者做成了模板抽象。

**Q5**：decode 阶段（单 token）的 attention 还需要 FlashAttention 吗？
**A**：收益变小但仍需要——decode 时 Q 只有一行，"不物化 $N^2$"的收益只剩 KV 侧的访存组织；真正的长上下文瓶颈转向 KV cache（Stage 2 第 3 周的 KIVI/FP8 KV 正是为此）。FlashAttention 在 prefill 阶段才是主英雄。

---

## 6. 搬运账定量练习：融合收益上界计算器（可运行）

```python
def fusion_bound(n_steps, bytes_per_tensor, bw_gbs=900, launch_us=7.0):
    """n 步独立算子 → 1 个融合算子的理论上界。"""
    t_unfused = n_steps * 2 * bytes_per_tensor / (bw_gbs * 1e9) * 1e3 + n_steps * launch_us / 1e3
    t_fused = 2 * bytes_per_tensor / (bw_gbs * 1e9) * 1e3 + launch_us / 1e3
    return t_unfused / t_fused

B, S, H = 32, 4096, 4096
bytes_ = B * S * H * 2                        # FP16 张量 ~2.1 GB
print(f"两步融合上界: {fusion_bound(2, bytes_):.2f}x")
print(f"三步融合上界（add+norm+scale）: {fusion_bound(3, bytes_):.2f}x")
# 小张量场景：launch 占比飙升
small = 1024 * 4096 * 2                        # ~8 MB
print(f"小张量两步融合上界: {fusion_bound(2, small):.2f}x（launch 主导）")
assert fusion_bound(3, bytes_) > fusion_bound(2, bytes_) > 1.5
print("融合收益账自检通过 ✓：步数越多、张量越小，融合越赚")
```

**读法**：大张量时收益趋近 $n$（搬运主导）；小张量时即使只有两步也可能有 3× 以上（launch 主导）——这就是"decode 碎片化场景里，连两个小算子都值得融合"的定量依据。

### 6.1 延伸阅读清单（本周之后）

1. `flash-attn` v2 论文与 `csrc/` 导读：关注"Q 常驻、K/V 流式"的循环交换（比 v1 少一次 HBM 往返）；
2. vLLM 的 `attention/backends/flash_attn.py`：FlashAttention 在推理引擎里的接线方式（与 PagedAttention 的组合）；
3. `torch.compile` 生成的 Triton 融合内核（`TORCH_LOGS="output_code"` 查看）：对照第 4 周的 Triton 语法，看自动融合的真实产物。

---

## 本周交付清单

- [ ] 闭卷推导在线 softmax 三式（$m$、$l$、$O$ 的递推）。
- [ ] 手绘 FlashAttention tiling 数据流图，标注各块 SRAM 用途与 $O(N^2) \to O(N)$。
- [ ] fused add+rmsnorm 编译运行：数值断言通过 + 记录加速比（预期 1.5-2.5×）。
- [ ] 列出 LLaMA 一个 Transformer 块的融合清单（§2 的图能闭卷画出）。
- [ ] 读 `Dao-AILab/flash-attention` 的 `csrc/flash_attn/` 入口 30 分钟，记录调用链（Python → C++ → kernel）。
