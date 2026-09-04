# 第 2 周教程：MatMul 优化三阶——GPU 编程的试金石

> **本周要回答的问题**
> 1. 朴素 GEMM 为什么是性能灾难？每个输出元素重复读了多少数据？
> 2. 共享内存分块（tiling）为什么能带来 5-10× 提升？`__syncthreads()` 在防什么？
> 3. WMMA Tensor Core 的编程接口长什么样？为什么 GEMM 必须吃 Tensor Core？
> 4. cuBLAS / CUTLASS 的工业级 GEMM 比你的版本还快在哪？

对应学习计划：第 2 周。交付物：三版 MatMul 完整可编译代码 + 与 `cublasGemmEx` 的 TFLOPS 对比表；第二版目标达到 cuBLAS 的 50%。

---

## 1. 第一性原理：GEMM 的性能 = 数据复用率

### 1.1 计算与访存的账

$C = AB$，$A, B, C$ 均为 $N \times N$。FLOPs $= 2N^3$（每个输出 $N$ 次乘加）。朴素实现中，**每个输出元素**都要独立读 $A$ 的一行 + $B$ 的一列——整个 $A$、$B$ 被完整读了 $N$ 遍：

$$
\text{朴素总搬运} \approx 2N^3 \times \text{字节}, \qquad \text{AI} \approx \frac{2N^3}{2N^3 \cdot \text{bytes}} \sim O(1) \quad \text{（访存瓶颈，浪费）}
$$

而理论最优只需读 $A$、$B$ 各一遍（$2N^2$ 字节），AI $= 2N^3 / 2N^2 = N$——**复用率的分母就是性能上限**。分块的意义：把 $N$ 拆成小块，让每块数据在被搬进共享内存后被块内所有输出复用。

### 1.2 三阶路径总览

| 版本 | 核心手段 | 数据复用发生在 | 预期（vs cuBLAS） |
| --- | --- | --- | --- |
| v1 朴素 | 无 | 无（L2 缓存兜底） | ~1-5% |
| v2 共享内存分块 | tiling + `__syncthreads` | 共享内存（块内复用） | 30-60% |
| v3 WMMA | Tensor Core 指令 | 寄存器 + Tensor Core 单元 | 60-90% |

### 1.3 Tensor Core 的账

Tensor Core 一条 `wmma::mma_sync` 指令完成 $16 \times 16 \times 16$ 的矩阵乘累加（FP16 输入、FP32 累加）。A100 上 FP16 Tensor Core 峰值 312 TFLOPS，是 CUDA Core 的 ~16×——**GEMM 不吃 Tensor Core，等于自动放弃 90% 的算力**。

---

## 2. 系统架构与数据流：分块 GEMM 的一轮迭代

```
for 每个块步 k_tile：
  ① 全体线程协作：把 A 的 (BM×BK) 子块、B 的 (BK×BN) 子块搬进共享内存
  ② __syncthreads()         ← 确保整块就绪，防止读到半成品
  ③ 每个线程用寄存器累加器，遍历两个子块算自己负责的输出部分
  ④ __syncthreads()         ← 确保所有人读完，才能覆盖加载下一块
最终：寄存器累加器写回 C
```

两个 `__syncthreads()` 缺一不可：第一个防"读未写完"，第二个防"写覆盖未读完"——这是本周必须画出来的数据流图。

---

## 3. 实现与验证（本周交付核心）

### 3.1 v1 朴素版（完整可编译）

```cuda
// gemm_v1.cu —— 朴素版：每线程一个输出元素，直接读 global memory
__global__ void gemm_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float acc = 0.f;
        for (int k = 0; k < N; k++)          // 每元素独立扫全行全列
            acc += A[row * N + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}
```

### 3.2 v2 共享内存分块版（完整可编译骨架）

```cuda
// gemm_v2.cu —— tiling：BM=BN=BK=32，每线程一个输出
#define BM 32
#define BN 32
#define BK 32

__global__ void gemm_shared(const float* A, const float* B, float* C, int N) {
    // ① 两块共享内存：A 的行块 (BM×BK)、B 的列块 (BK×BN)
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int row = blockIdx.y * BM + threadIdx.y;   // 本线程负责的行
    int col = blockIdx.x * BN + threadIdx.x;   // 本线程负责的列
    float acc = 0.f;                           // ② 寄存器累加器

    for (int t = 0; t < (N + BK - 1) / BK; t++) {
        // ③ 协作加载：每线程搬一个元素进共享内存（越界补 0）
        int aCol = t * BK + threadIdx.x, aRow = blockIdx.y * BM + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (aRow < N && aCol < N) ? A[aRow * N + aCol] : 0.f;
        int bCol = blockIdx.x * BN + threadIdx.x, bRow = t * BK + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] =
            (bRow < N && bCol < N) ? B[bRow * N + bCol] : 0.f;
        __syncthreads();                       // ④ 等整块就绪（防读半成品）

        #pragma unroll
        for (int k = 0; k < BK; k++)           // ⑤ 在共享内存上累加
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();                       // ⑥ 等读完，再覆盖下一块
    }
    if (row < N && col < N) C[row * N + col] = acc;
}
```

### 3.3 v3 WMMA Tensor Core 版（骨架 + 关键调用）

```cuda
// gemm_v3.cu —— WMMA：16×16×16 分块，FP16 输入 FP32 累加
#include <mma.h>
using namespace nvcuda;

__global__ void gemm_wmma(const __half* A, const __half* B, float* C, int N) {
    // 每 warp 负责一个 16×16 的输出块；声明 fragment（硬件寄存器布局）
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // 计算本 warp 负责的输出块起点（需要与共享内存分块配合，此处简化为直接加载）
    int row = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * 16;
    int col = blockIdx.x * 16;
    for (int k = 0; k < N; k += 16) {
        wmma::load_matrix_sync(a_frag, A + row * N + k, N);   // 加载 A 块
        wmma::load_matrix_sync(b_frag, B + k * N + col, N);   // 加载 B 块
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);       // 一条指令 = 16³ 乘累加
    }
    wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
}
// 完整生产版需：共享内存双缓冲 + 每线程多块 + swizzle，参考 CUTLASS examples/00-03
```

### 3.4 编译、基准与对齐

```bash
nvcc -O3 -arch=sm_86 gemm_v1.cu -o gemm_v1
nvcc -O3 -arch=sm_86 gemm_v2.cu -o gemm_v2
nvcc -O3 -arch=sm_86 gemm_v3.cu -o gemm_v3 -lcuda   # WMMA 需要 sm_70+
```

```python
# bench.py —— 统一基准：N=2048，10 次预热 + 100 次计时，对齐用 cuBLAS 结果
import torch, time
N = 2048
A, B = torch.randn(N, N, device="cuda"), torch.randn(N, N, device="cuda")
ref = A @ B                                     # cuBLAS 结果与计时基线
def bench(fn, *args, iters=100):
    for _ in range(10): fn(*args)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(iters): fn(*args)
    torch.cuda.synchronize()
    return (time.time() - t) / iters
ms = bench(lambda: A @ B)
tflops = 2 * N**3 / ms / 1e9                    # FLOPs/s → TFLOPS
print(f"cuBLAS: {ms*1e3:.2f} ms, {tflops:.1f} TFLOPS")
# 你的三版通过 pybind11/torch cpp_extension 封装后用同一函数计时；
# 正确性断言：
#   assert (your_C - ref).abs().max() < 1e-2 * ref.abs().max()
```

对比表（实测填写）：

| 版本 | TFLOPS | vs cuBLAS | 备注 |
| --- | --- | --- | --- |
| v1 朴素 | _实测_ | ~1-5% | 纯访存灾难 |
| v2 共享内存 | _实测_ | **目标 ≥ 50%** | 不达 → 查合并访存与 unroll |
| v3 WMMA | _实测_ | 60-90% | 骨架版；完整版需双缓冲 |
| cuBLAS | _实测_ | 100% | 对照组 |

---

## 4. 工程权衡与失效模式

### 4.1 权衡

- **块大小**：BM/BN 越大复用越高，但共享内存与寄存器压力上升 → occupancy 下降。32×32 是入门甜点，工业级用 128×128/256×128 + 每线程多输出。
- **FP32 vs FP16 Tensor Core**：精度换速度；推理/训练主流都是"FP16 计算 + FP32 累加"，累加器必须高精度（数值稳定）。
- **自己写 vs cuBLAS**：标准 GEMM 永远用 cuBLAS/CUTLASS；自己写的价值在于**融合定制**（第 3 周的 epilogue）与特殊形状（INT4 GEMV）。

### 4.2 失效模式

1. **缺 `__syncthreads()`**：症状——结果偶发错误（有时对有时错）；根因——读到未写完的共享内存（竞态）；修复——补齐两个同步点（§3.2 ④⑥）。
2. **共享内存溢出**：症状——`invalid argument at launch`；根因——块尺寸 × 数据类型超每 SM 配额；修复——减块大小或 `cudaFuncSetAttribute` 提升上限。
3. **bank conflict 拖慢 v2**：症状——v2 提升不到 3×；根因——按列访问 `[BK][BN]` 时同列落同 bank；修复——padding 或 swizzle 布局（CUTLASS 的核心技巧之一）。
4. **WMMA 结果对不上**：症状——数值全错；根因——fragment 的 row/col-major 声明与实际内存布局不符；修复——核对 `load_matrix_sync` 的 leading dimension 与 major 参数。

---

## 5. 延伸思考题（含解析）

**Q1**：v2 相比 v1 的加速本质是什么？用复用率定量说。
**A**：v1 中每个输出元素从 global 读 $2N$ 个元素；v2 中 $BK$ 宽的数据被共享内存里的 $BM$ 个行复用——global 搬运量除以块尺寸（~32×），AI 从 $O(1)$ 升到 $O(32)$，于是从访存地狱爬向算力区间。加速上限 ≈ 复用倍数与带宽裕量的较小者。

**Q2**：为什么需要两个 `__syncthreads()`，一个行不行？
**A**：不行。第一个保证"本轮块加载完整"（才能安全计算）；第二个保证"上轮计算全部读完"（才能覆盖写下一轮）。少第二个 → 快线程加载新块时覆盖慢线程正在读的旧块——经典竞态，且错误率随负载变化，极难复现。

**Q3**：CUTLASS 的"Epilogue 融合"是什么？为什么它是推理引擎的命门？
**A**：GEMM 算完后，结果还在寄存器里——此刻顺手做偏置加、激活函数（GELU/SiLU）、残差加，省掉一次"写回显存再读出来"的往返。SwiGLU、Add+RMSNorm 的融合（第 3 周）本质都是 epilogue 思想的推广。

**Q4**：你的 v2 没到 cuBLAS 的 50%，按什么顺序排查？
**A**：① `ncu` 看带宽利用率与 bank conflict；② 检查加载是否合并（连续线程读连续地址）；③ 加 `#pragma unroll` 与寄存器累加；④ 考虑每线程算 2×2 输出（提高指令级复用）。顺序就是"先访存、后计算"。

**Q5**：为什么 decode 阶段的 Linear 不叫 GEMM 而叫 GEMV？它对本周知识意味着什么？
**A**：batch=1 时输入是向量（$1 \times n$），矩阵乘退化为矩阵-向量乘（GEMV）：无复用可言（AI ≈ 2），Tensor Core 吃不满，纯拼带宽——这正是第 4-6 周手写 INT4 GEMV 的背景：decode 的主战场不是本周的 tiling，而是"怎么把权重搬得更快"。

---

## 6. 性能账本：三版 kernel 的 Roofline 对账（可运行）

```python
# roofline_check.py —— 用第 1 周的 Roofline 给本周三版 MatMul 对账
def roofline(ai, peak_tflops=19.5, bw_gbs=900.0):
    """A10G 量级示例：FP32 19.5 TFLOPS、带宽 900 GB/s。返回可达 TFLOPS。"""
    return min(peak_tflops, bw_gbs * ai / 1000)

N = 2048
bytes_moved = 3 * N * N * 4          # A、B、C 各一遍（理想下限）
flops = 2 * N ** 3
ai_ideal = flops / bytes_moved       # ≈ N/6 ≈ 341（远超拐点，算力瓶颈区）
print(f"GEMM 理论 AI = {ai_ideal:.0f} FLOP/Byte → Roofline 上限 {roofline(ai_ideal):.1f} TFLOPS")

# 朴素版的"实际" AI：每个输出都重读全行全列 → 搬运 ≈ 2N³ 次
ai_naive = flops / (2 * N**3 * 4)
print(f"朴素版实际 AI = {ai_naive:.1f} → 上限 {roofline(ai_naive):.2f} TFLOPS（差百倍）")
assert roofline(ai_ideal) > 10 * roofline(ai_naive), "分块应带来数量级的上限提升"
print("Roofline 对账 ✓：tiling 的本质 = 把算子从带宽区搬回算力区")
```

**对账方法**：把你实测的三版 TFLOPS 标到这张图上——v1 应贴在朴素 AI 的上限附近，v2/v3 逐步逼近硬件拐点右侧。任何版本"远低于其所在区域的理论线"，就说明还有访存或指令级的问题没解决。

### 6.1 双缓冲（double buffering）：v2 → v3 之间的下一站

本周的 v2 每轮迭代都要"加载→等待→计算"串行。工业级的下一步是**双缓冲**：用两份共享内存，加载第 $k+1$ 块的同时计算第 $k$ 块——访存延迟被计算完全掩盖。CUTLASS 的 pipeline 抽象就是在做这件事的多级版本（multi-stage）。读懂它，你就拿到了阅读 CUTLASS 的钥匙。

---

## 本周交付清单

- [ ] 手绘 v2 的双同步数据流图（标注两个 `__syncthreads` 各防什么）。
- [ ] 三版代码全部编译运行，正确性断言通过（与 cuBLAS 结果对齐）。
- [ ] 填完四行 TFLOPS 对比表；v2 不达 50% 则按 §5 Q4 顺序排查重跑。
- [ ] 读懂 CUTLASS examples/00（basic_gemm）的模板参数，记录 5 个看不懂的名词（下阶段消化）。
- [ ] 用自己的话回答：为什么 GEMM 必须吃 Tensor Core？（≤ 60 字）
