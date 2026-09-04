# 第 4 周教程：Triton 入门与 Profiling 驱动优化复盘

> **本周要回答的问题**
> 1. Triton 的 block 级编程模型替你管了什么？为什么它比原生 CUDA 快写好几个量级？
> 2. `@triton.autotune` 在搜什么？配置空间怎么定？
> 3. nsys 与 ncu 各看什么？怎么从一次 vLLM 推理 trace 里找出真瓶颈？
> 4. INT4 GEMV（权重 × FP16 激活）怎么设计，才能把 Stage 2 的量化知识变成真实带宽收益？

对应学习计划：第 4-6 周。交付物：① Triton 三连（vector_add / MatMul+autotune / fused norm）；② 手写 INT4 GEMV kernel；③ 《7B 模型 vLLM 推理 Profiling 报告》。

---

## 1. 第一性原理：抽象层次换开发效率

### 1.1 Triton 替你管了什么

原生 CUDA 你要手管：线程到数据的映射、warp 级原语、共享内存分配与同步、访存合并、bank conflict。Triton 把并行单位从**线程**抬到**block（tile）**：

- 你写的是"这个 program 处理哪一块数据"（`tl.program_id` + 块偏移），块内的线程映射、合并访存、共享内存使用由编译器自动生成；
- 循环、掩码（`tl.load(..., mask=...)` 处理越界）、归约（`tl.sum`）都是块级操作。

代价：极限性能略低于手写 CUDA（编译器不如专家手工）；收益：开发效率高一个量级——**推理引擎的新算子大量用 Triton 写**（vLLM 自带若干），因为"够快 + 快速迭代"是工程上的更优解。

### 1.2 autotune：把"调参"交给网格搜索

`@triton.autotune(configs=[...], key=["M","N","K"])`：对每种形状，首次调用时把所有配置（块大小、warp 数、流水级数）各跑一遍取最快者，之后缓存。你只需给出合理的候选网格——这相当于把第 2 周手工调块大小的过程自动化。

### 1.3 Profiling 的两把尺子

| 工具 | 视角 | 回答的问题 |
| --- | --- | --- |
| `nsys`（Nsight Systems） | **系统时间线**：CPU/GPU 全流水 | 瓶颈在哪个阶段？GPU 有没有在等 CPU？kernel 之间有没有空隙？ |
| `ncu`（Nsight Compute） | **单 kernel 解剖** | 这个 kernel 的带宽/算力利用率、occupancy、瓶颈原因？ |

口诀：**先 nsys 找"谁慢"，再 ncu 问"为什么慢"**。

### 1.4 INT4 GEMV 的设计要点（呼应 Stage 2）

decode 的 Linear 是 GEMV（$1 \times n$ 乘 $n \times m$），纯访存瓶颈。INT4 per-group(128) 权重的设计清单：

1. **打包**：两个 4-bit 权重挤一个字节（低 4 位 + 高 4 位），加载后移位解包；
2. **scale 的组织**：每 128 个权重一个 FP16 scale，与权重块同步加载，解包后立即反量化；
3. **并行映射**：一个输出元素 = 权重的一列（GEMV 中即一行）——每线程/每块负责一段行，FP16 累加，最后归约；
4. **目标指标**：带宽利用率 ≥ 80%（`ncu` 的 `dram__throughput`）——INT4 GEMV 没有别的活，就是拼搬运。

---

## 2. 实现与验证（本周交付核心）

### 2.1 Triton 三连（完整可运行）

```python
import torch, triton, triton.language as tl

# ---- 题 1：vector_add ----
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)      # 本 program 负责的元素区间
    mask = offs < n                                # 越界掩码（替代 CUDA 的 if）
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def t_add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out

x, y = torch.randn(1 << 24, device="cuda"), torch.randn(1 << 24, device="cuda")
assert torch.allclose(t_add(x, y), x + y), "Triton vector_add 错误"
print("Triton vector_add PASS")

# ---- 题 2：MatMul + autotune ----
@triton.autotune(
    configs=[triton.Config({"BM": 64, "BN": 64, "BK": 32}, num_warps=4),
             triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8)],
    key=["M", "N", "K"])
@triton.jit
def mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    # 把一维 pid 映射到 (row_block, col_block)
    nm = tl.cdiv(M, BM); pid_m = pid % nm; pid_n = pid // nm
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):                     # K 维分块循环
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < K) & (offs_m[:, None] < M), other=0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0)
        acc += tl.dot(a, b)                       # 编译器自动映射到 Tensor Core
        a_ptrs += BK; b_ptrs += BK * N
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def t_mm(a, b):
    M, K = a.shape; K2, N = b.shape; assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)
    mm_kernel[grid](a, b, c, M, N, K)
    return c

a, b = torch.randn(1024, 1024, device="cuda"), torch.randn(1024, 1024, device="cuda")
err = (t_mm(a, b) - a @ b).abs().max().item()
assert err < 1e-2, f"Triton MatMul 误差过大: {err}"
print(f"Triton MatMul PASS（最大误差 {err:.2e}）")

# ---- 题 3：fused RMSNorm（对照第 3 周的 CUDA 版） ----
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, d, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < d
    x = tl.load(x_ptr + row * d + offs, mask=mask, other=0.0)
    ms = tl.sum(x * x, axis=0) / d                # 块级归约，一行搞定
    x_hat = x * tl.rsqrt(ms + eps)
    w = tl.load(w_ptr + offs, mask=mask)
    tl.store(out_ptr + row * d + offs, x_hat * w, mask=mask)

def t_rmsnorm(x, w, eps=1e-6):
    out = torch.empty_like(x)
    M, d = x.shape
    rmsnorm_kernel[(M,)](x, w, out, d, eps, BLOCK=triton.next_power_of_2(d))
    return out

x = torch.randn(512, 4096, device="cuda"); w = torch.randn(4096, device="cuda")
ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
assert torch.allclose(t_rmsnorm(x, w), ref, atol=1e-3), "Triton RMSNorm 错误"
print("Triton fused RMSNorm PASS")
```

**交付**：三题的 benchmark 表（各自与 `torch` 对应算子计时对比）；MatMul 记录 autotune 选中的配置。

### 2.2 vLLM Profiling 工作流

```bash
# ① nsys 抓系统级 trace（云上跑，模型用 Qwen2.5-7B）
nsys profile -o qwen7b_vllm --trace=cuda,nvtx \
    python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B &
# 发请求（一条长 prompt + 一条短 prompt 各若干），然后停止采集
# ② ncu 解剖热点 kernel（对 nsys 里找到的最耗时 kernel）
ncu --set full --kernel-name regex:gemm --launch-skip 10 --launch-count 3 \
    <你的推理脚本>
```

《7B 模型 vLLM 推理 Profiling 报告》模板（交付）：

| 章节 | 内容要求 |
| --- | --- |
| 1. 实验设置 | 硬件型号、vLLM 版本、prompt/输出长度、batch 配置 |
| 2. prefill Top10 算子 | nsys 时间线截图 + 耗时表（预期：GEMM/Attention 主导） |
| 3. decode Top10 算子 | 同上（预期：GEMV/norm/采样碎片化，空隙变多） |
| 4. 利用率 | GPU 利用率、显存带宽利用率（ncu）、KV cache 占用 |
| 5. 三条优化建议 | 必须基于你自己的数据（如：decode 碎片化 → 连续批处理；norm 未融合 → 融合算子） |

### 2.3 INT4 GEMV：设计先行（实现按 §1.4 清单）

```python
# 设计验收（可运行）：先用 PyTorch 模拟"打包-解包-反量化"确认数值语义
import torch
def pack_int4(w_int4: torch.Tensor) -> torch.Tensor:
    """两个 4-bit 打包进一个 uint8：低 4 位 = 偶数索引，高 4 位 = 奇数索引。"""
    lo, hi = w_int4[0::2] & 0xF, w_int4[1::2] & 0xF
    return (hi << 4 | lo).to(torch.uint8)

def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    lo = (packed & 0x0F).to(torch.int8)
    hi = (packed >> 4).to(torch.int8)
    lo = torch.where(lo > 7, lo - 16, lo)      # 有符号还原
    hi = torch.where(hi > 7, hi - 16, hi)
    out = torch.empty(packed.numel() * 2, dtype=torch.int8)
    out[0::2], out[1::2] = lo, hi
    return out

w = torch.randint(-8, 8, (256,), dtype=torch.int32)
assert (unpack_int4(pack_int4(w)) == w.to(torch.int8)).all()
print("INT4 打包/解包语义验证 ✓ —— 下一步：在 CUDA/Triton kernel 里做同样的事")
```

> 实现路线（第 5-6 周）：用 CUDA 或 Triton 写 GEMV：每 program 负责一段输出，`tl.load` 打包字节 → 位运算解包 → 乘组内 scale 反量化 → 与激活点积累加。与 `torch.matmul` FP16 版比 batch=1 延迟与带宽利用率（预期：显存占用 1/4，延迟接近 1/3~1/4）。

---

## 3. 工程权衡与失效模式

### 3.1 权衡

- **Triton vs 原生 CUDA**：开发效率换极限性能。规则：先 Triton 出活测收益，确有必要（差 20%+ 且在关键路径）才下沉到 CUDA；极限场景（Marlin 级 GEMV）仍需手写。
- **autotune 的运行时成本**：首次调用要跑全部配置（秒级卡顿），服务启动时要预热；配置空间过大会拖慢冷启动。
- **profiling 的代表性**：单次短请求的 trace 不代表生产负载——要用真实长度分布的请求压一遍。

### 3.2 失效模式

1. **Triton mask 漏写**：症状——输出尾部/边界错乱；根因——非整块尺寸时越界 `tl.load` 读到脏内存；修复——所有 load/store 带 `mask`（§2.1 已示范）。
2. **autotune key 不全**：症状——某些形状性能奇差；根因——key 没覆盖该维度，复用了不适用的配置；修复——把影响性能的维度都放进 `key`。
3. **nsys 里 GPU 大段空白**：症状——GPU 利用率低但 kernel 本身不慢；根因——CPU 侧预处理/采样/调度成为瓶颈（Python 阻塞、同步点）；修复——异步化、连续批处理（Stage 5 主题）。
4. **ncu 测的不是目标 kernel**：症状——剖析了预热期的特殊 kernel；根因——没跳过前几次启动；修复——`--launch-skip` + `--kernel-name` 过滤。

---

## 4. 延伸思考题（含解析）

**Q1**：Triton 编译器自动做对了哪三件你第 1-2 周手工做的事？
**A**：① 线程到块内元素的映射与访存合并；② 共享内存的分配、加载与同步（`tl.dot` 背后自动 tiling）；③ Tensor Core 指令选择（`tl.dot` 自动用 WMMA 类指令）。你保留的决策是：块大小、循环结构、数据布局——这正是"专家知识"剩下的部分。

**Q2**：为什么 decode 阶段的 nsys 时间线上 kernel 之间空隙比 prefill 多？
**A**：decode 每步的算子又多又小（单 token），launch 与 CPU 调度的固定开销占比飙升；prefill 是少数大 kernel。这就是"碎片化"，解法是连续批处理攒大 batch、CUDA Graph 固化流水（Stage 5 深挖）。

**Q3**：INT4 GEMV 为什么"带宽利用率 80%"就是好结果，而不是追求 100%？
**A**：它已是纯访存算子，剩余 20% 损失来自：scale 的额外搬运、解包指令占用的发射槽、尾部不满块的掩码——访存峰值本身也留有余量（刷新、争用）。80% 已逼近硬件实际可达上限。

**Q4**：什么时候该"写算子"、什么时候该"改调度"？
**A**：ncu 显示单 kernel 远离硬件上限 → 写/换算子；nsys 显示 GPU 大量空闲、瓶颈在流水组织 → 改调度（批处理、图模式、异步）。算子级优化救不了系统级瓶颈，反之亦然——这是本阶段与 Stage 5 的分界认知。

**Q5**：本阶段四周一共手写了几个 kernel？按"访存优化 → 计算优化 → 融合"分类。
**A**：访存：vector_add、transpose（合并/共享内存两版）；计算：GEMM 三阶（朴素/分块/WMMA）；融合：fused add+RMSNorm（CUDA 与 Triton 各一）、Triton MatMul、Triton RMSNorm、INT4 GEMV。面试问"手写过什么 kernel"，就报这份清单 + 每个的加速数字。

---

## 5. ncu 指标速查表（写报告时直接引用）

| 你想知道 | ncu 指标 | 健康值参考 |
| --- | --- | --- |
| 带宽吃满没 | `dram__throughput.avg.pct_of_peak` | 访存算子 > 70% |
| 算力吃满没 | `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak`（或对应 pipe） | GEMM > 50% |
| 占用率 | `sm__warps_active.avg.pct_of_peak` | 不是越高越好，看瓶颈类型 |
| 访存合并质量 | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` 对比理想扇区数 | 越接近理想越好 |
| kernel 间空隙 | nsys 时间线（不是 ncu） | decode 空隙占比是优化线索 |

**报告纪律**：每个结论附指标名与数值——"带宽利用率 83%"比"基本吃满"有力一百倍，这也是你与"只会调 API"的候选人的分界线。

### 5.1 Triton → CUDA 的下沉判据（工程决策模板）

```
该算子值得下沉到原生 CUDA 吗？
├─ Triton 版已达硬件上限的 85%+？ → 不下沉（收益空间太小）
├─ 在关键路径（占端到端 > 5%）？
│    ├─ 否 → 不下沉
│    └─ 是 → 差距 > 20% 且有明确的手工优化点（如寄存器级流水）？
│         ├─ 是 → 下沉（参考 Marlin 的实现手法）
│         └─ 否 → 先试 autotune 扩大配置空间，再评估
```

这个判据对应学习计划"算子级与系统级的边界"：**写算子是手段，端到端指标是目的**——Triton 把"试错成本"降到了分钟级，让下沉决策可以基于数据而不是冲动。

---

## 本周交付清单

- [ ] Triton 三连全部运行通过（三个断言），填三题的基准表。
- [ ] 记录 autotune 为 1024³ MatMul 选中的配置与对应 TFLOPS。
- [ ] INT4 打包/解包语义断言通过；完成 INT4 GEMV kernel 并记录带宽利用率。
- [ ] 云上用 nsys 抓一次 vLLM 推理 trace，ncu 剖析 Top1 kernel。
- [ ] 完成《7B 模型 vLLM 推理 Profiling 报告》（按 §2.2 模板五章节）。
- [ ] 核对学习计划自测清单：5 个手写 kernel 均有 benchmark 存档。
