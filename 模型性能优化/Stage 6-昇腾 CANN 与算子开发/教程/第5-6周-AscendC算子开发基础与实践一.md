# 第 5-6 周教程：Ascend C 算子开发基础与实践一（fused Add + RMSNorm）

> **本周要回答的三个问题**
> 1. Ascend C 程序的三要素——Kernel、Tiling、流水——各自是什么、怎么协作？
> 2. double buffer 流水为什么能隐藏访存延迟？时序图长什么样？
> 3. 如何把你在 CUDA 里写过的 fused Add+RMSNorm 用 Ascend C 重写，并验证数值正确？

对应学习计划：第 5-6 周。交付物：用 Ascend C 实现第一个自定义算子 **fused Add + RMSNorm**，完成 Kernel + Tiling + 单算子精度测试，输出与 PyTorch 参考实现的误差对比表。

---

## 1. 第一性原理：为什么要"显式流水"编程

### 1.1 GPU 与 NPU 编程模型的差异

CUDA 里，数据在显存，线程自己去读，硬件帮你调度访存。Ascend C 更**显式**：你要亲自安排数据在内存层级间的搬运（HBM → L1/UB → 计算单元），并让"搬运"与"计算"**流水并行**——这是性能的关键，也是难点。

### 1.2 内存层级

```
HBM（片外，大）
   │  DataCopy（DMA 搬运）
UB / Unified Buffer（片上，小，类比共享内存）
   │
计算单元（Cube / Vector / Scalar）
```

计算单元不能直接高效吃 HBM 数据，要先搬到 UB；搬完才能算。若"搬一块→算一块→搬下一块→算下一块"串行做，搬运和计算互相等待。**double buffer 流水**就是来解决这个问题的。

---

## 2. 三要素：Kernel / Tiling / 流水

### 2.1 Kernel：设备侧的计算逻辑

Kernel 是跑在 AI Core 上的函数，描述"每个核怎么算"。Ascend C 用 C++ 类组织，典型结构：

```cpp
// 伪代码骨架（完整工程以官方 example 为准）
class AddRmsNorm {
public:
    __aicore__ inline AddRmsNorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
                                uint32_t totalRows, uint32_t cols, ...);
    __aicore__ inline void Process();          // 主流程：循环分块 + 流水
private:
    __aicore__ inline void CopyIn(int32_t row);   // HBM -> UB
    __aicore__ inline void Compute(int32_t row);  // 在 UB 上算
    __aicore__ inline void CopyOut(int32_t row);  // UB -> HBM
    // 双 buffer
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueue;     // 输入双缓冲
    TQue<QuePosition::VECOUT, 2> outQueue;   // 输出双缓冲
};
```

### 2.2 Tiling：任务怎么分给多个核

一块芯片有多个 AI Core。Tiling 决定"总任务怎么切、每个核算哪部分、每块多大"。Tiling 参数在 **host 侧**算好，传给 kernel。

对 `(R, C)` 的矩阵：
- 按行切：每核分 `rows_per_core` 行；
- 每核内再按 `tile_rows` 分块，逐块流水处理。

```cpp
// Tiling 结构（示意）
struct AddRmsNormTiling {
    uint32_t totalRows;
    uint32_t cols;
    uint32_t rowsPerCore;    // 每核分到的行数
    uint32_t tileRows;       // 每次流水块的大小
    // ...
};
```

**Tiling 的取舍**：块太小 → 流水启动/收尾开销占比高；块太大 → UB 装不下。要满足 `块大小 ≤ UB 容量` 且让每个核都有足够活干。

### 2.3 Double Buffer 流水

用**两块缓冲**轮流：搬第 2 块的同时算第 1 块，算第 1 块的结果往外搬的同时搬第 3 块……搬运与计算重叠。

时序图（每格一个阶段）：

```
        t0   t1   t2   t3   t4   t5
CopyIn : C0   C1   C2   C3   C4
Compute:      V0   V1   V2   V3   V4
CopyOut:           O0   O1   O2   O3
```

`C`=搬入，`V`=计算，`O`=搬出。稳定后每步同时有三件事在进行——**用空间（双缓冲）换时间（隐藏访存延迟）**。这正是学习计划要求你画的流水时序图。

---

## 3. fused Add + RMSNorm 算法回顾

输入 $x$、残差 $r$、权重 $\gamma$，目标：

$$
h = x + r, \qquad
y = \frac{h}{\sqrt{\frac{1}{C}\sum_c h_c^2 + \epsilon}} \cdot \gamma
$$

三步：① 逐元素加；② 求每行的均方根（归约）；③ 逐元素乘缩放与 $\gamma$。对应到硬件：①③ 是 Vector 逐元素，② 是 Vector 归约。

你在 Stage 3 用 CUDA 写过它（全局内存版），现在用 Ascend C 在 UB 上重写——**同一算法、两种硬件编程模型**，对照是最好的学习方式。

---

## 4. Kernel 实现（交付核心）

### 4.1 Process 主循环

```cpp
__aicore__ inline void AddRmsNorm::Process() {
    // 每个核处理自己分到的行
    for (int32_t i = 0; i < rowsPerCore / tileRows; ++i) {
        CopyIn(i);      // 申请 inQueue buffer，HBM->UB
        Compute(i);     // 在 UB 上算
        CopyOut(i);     // 结果搬回
    }
}
```

### 4.2 CopyIn / Compute / CopyOut（伪代码，突出逻辑）

```cpp
__aicore__ inline void AddRmsNorm::CopyIn(int32_t idx) {
    LocalTensor<half> xLocal = inQueue.AllocTensor<half>();
    // DataCopy 从全局内存搬 x 与 residual 到 UB
    DataCopy(xLocal, xGm + offset, tileElems);
    // residual 同理（或用第二 buffer）
    inQueue.EnQue(xLocal);               // 入队，交给 Compute
}

__aicore__ inline void AddRmsNorm::Compute(int32_t idx) {
    LocalTensor<half> xLocal = inQueue.DeQue<half>();
    LocalTensor<half> yLocal = outQueue.AllocTensor<half>();

    // ① h = x + r   （Vector Add）
    Add(yLocal, xLocal, rLocal, tileElems);

    // ② 求每行均方根（Vector 归约）：
    //    sum_sq = ReduceSum(yLocal * yLocal, per_row)
    //    scale = rsqrt(sum_sq / cols + eps)
    // ③ y = yLocal * scale * gamma  （Vector Mul）

    outQueue.EnQue(yLocal);              // 交给 CopyOut
    inQueue.FreeTensor(xLocal);
}

__aicore__ inline void AddRmsNorm::CopyOut(int32_t idx) {
    LocalTensor<half> yLocal = outQueue.DeQue<half>();
    DataCopy(yGm + offset, yLocal, tileElems);   // UB -> HBM
    outQueue.FreeTensor(yLocal);
}
```

**注意**：具体 API 名（`Add`、归约接口、`DataCopy` 参数）以官方 Ascend C 文档与 example 为准——向量算子接口随版本演进。上面给的是**逻辑骨架**，你要照着官方样例补全真实调用。

### 4.3 Host 侧：Tiling 计算与算子注册

```cpp
// Host：计算 tiling，填入结构体，传给 kernel
// 并实现算子注册（op type、输入输出、tiling 函数）
// 具体注册宏与接口以自定义算子开发指南为准
```

开发流程按官方：**建算子工程 → 写 Kernel + Tiling → 编译（算子包）→ 单算子测试**。

---

## 5. 精度测试（交付）

### 5.1 参考实现（PyTorch）

```python
import torch

def add_rmsnorm_ref(x, residual, gamma, eps=1e-6):
    h = x + residual
    var = h.pow(2).mean(dim=-1, keepdim=True)
    return h * torch.rsqrt(var + eps) * gamma

# 生成测试数据
torch.manual_seed(0)
R, C = 1024, 4096
x = torch.randn(R, C, dtype=torch.float16)
r = torch.randn(R, C, dtype=torch.float16)
gamma = torch.randn(C, dtype=torch.float16)
ref = add_rmsnorm_ref(x, r, gamma)
```

### 5.2 与算子输出对比

```python
import numpy as np

op_out = run_your_ascend_op(x, r, gamma)     # 通过单算子测试框架调用
ref_np, out_np = ref.numpy(), op_out.numpy()

max_err = np.abs(ref_np - out_np).max()
cos_sim = (ref_np * out_np).sum() / (
    np.linalg.norm(ref_np) * np.linalg.norm(out_np) + 1e-9)

print(f"最大绝对误差: {max_err:.4e}")
print(f"余弦相似度:  {cos_sim:.6f}")
assert cos_sim > 0.9999, "算子精度不达标"
```

**误差预期**：FP16 下最大绝对误差在 1e-3 量级、余弦相似度 > 0.9999 属正常。误差明显更大要查归约实现与累加精度（是否用 FP32 累加）。

### 5.3 误差对比表（交付）

```
| 实现 | 相对参考误差 | 余弦相似度 | 备注 |
| CUDA fused (Stage3) | | | 基线 |
| Ascend C fused | | | 本交付 |
```

---

## 6. 工程权衡与失效模式

### 6.1 权衡

- **块大小**：受 UB 容量约束，太小流水开销大、太大装不下；
- **累加精度**：归约用 FP32 累加防误差累积，但略慢；
- **半精度**：FP16 省带宽但需注意范围。

### 6.2 失效模式

1. **UB 溢出**：块太大装不下。修复：减块大小、核对 UB 容量。
2. **流水没起来**：串行等待、未用双缓冲。修复：核对 EnQue/DeQue 顺序，让搬与算重叠。
3. **归约误差大**：FP16 累加溢出。修复：中间用 FP32 累加。
4. **Tiling 分块不均**：某核没活干/越界。修复：处理不能整除的余数行。

---

## 7. 延伸思考题（含解析）

**Q1**：Ascend C 的 Kernel、Tiling、流水各管什么？
**A**：Kernel 是设备侧每核的计算逻辑；Tiling 在 host 侧决定任务怎么切给多核、每块多大；流水（double buffer）让数据搬运与计算重叠。三者协作：Tiling 定分工，Kernel 定算法，流水定执行节奏。

**Q2**：double buffer 为什么能隐藏访存延迟？
**A**：用两块缓冲，搬第 2 块的同时算第 1 块、搬出第 0 块，三个阶段重叠。稳定后每步同时有搬入/计算/搬出在进行，访存延迟被计算掩盖，硬件利用率提高。

**Q3**：Tiling 的块大小怎么定？
**A**：上界是 UB 容量（块必须装得下）；下界是让流水启动/收尾开销占比可接受。在两者间选能整除或余数最少、且每核有足够活的值。

**Q4**：fused Add+RMSNorm 里哪步是归约？为什么它比逐元素难？
**A**：求每行均方根是归约（把一行压成一个数）。它需要跨元素累加、涉及累加精度（要 FP32 防溢出）和分块时的跨块合并，比纯逐元素复杂。

**Q5**：为什么归约要用 FP32 累加，即使输入是 FP16？
**A**：FP16 范围小，长累加易溢出/精度丢失。用 FP32 累加器累计平方和更稳，最后再转回。这是数值稳定性的常见实践。

---

## 本周交付清单

- [ ] 建算子工程，实现 fused Add+RMSNorm 的 Kernel（CopyIn/Compute/CopyOut）。
- [ ] 实现 Tiling 参数计算。
- [ ] 编译算子包，跑单算子测试。
- [ ] 输出与 PyTorch 参考的误差对比表（余弦相似度 > 0.9999）。
- [ ] 画 double buffer 流水时序图，解释各阶段。
