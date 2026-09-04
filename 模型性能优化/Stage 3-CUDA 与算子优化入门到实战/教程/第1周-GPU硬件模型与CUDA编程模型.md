# 第 1 周教程：GPU 硬件模型与 CUDA 编程模型

> **本周要回答的问题**
> 1. GPU → SM → warp → thread 的层级是什么？为什么"SM 数 × 每 SM 线程数"决定并行度？
> 2. 访存合并（Coalescing）是什么？为什么 90% 的 kernel 慢在访存而不是计算？
> 3. Roofline 模型怎么判断算子是算力瓶颈还是访存瓶颈？
> 4. 已知权重 14 GB、显存带宽 2 TB/s，7B decode 的理论上限是多少 tokens/s？

对应学习计划：第 1 周。交付物：① 手写 `vector_add` 与 `matrix_transpose`（合并/非合并对比）；② 用 Roofline 手算 7B decode 上限。

> **爬坡提示**：本周所有代码先**逐行读懂注释**，改参数（块大小、矩阵尺寸）重跑看性能变化，最后再合上文档仿写。这是整个 Stage 3 的学习姿势。

---

## 1. 第一性原理：GPU 是一台"用并行换延迟"的机器

### 1.1 硬件层级

$$
\text{GPU} \supset \text{GPC} \supset \text{TPC} \supset \underbrace{\text{SM（流多处理器）}}_{\text{基本调度单位}} \supset \{\text{CUDA Core},\ \text{Tensor Core},\ \text{寄存器堆},\ \text{共享内存}\}
$$

- **SM（Streaming Multiprocessor）**：GPU 的基本计算单元，如 A100 有 108 个、4090 有 128 个。每个 SM 独立调度自己名下的线程；
- **warp**：32 个线程的**锁步执行组**，是硬件真正的调度粒度——同 warp 内所有线程执行同一条指令（SIMT），分支发散会串行化；
- **线程层级**：`grid`（整个任务）→ `block`（一组线程，同 block 可共享内存 + 同步）→ `warp` → `thread`。

**并行度 = SM 数 × 每 SM 并发线程数**（A100：108 × 2048 ≈ 22 万线程）。GPU 不靠"快"取胜——单核频率低于 CPU——它靠**海量线程掩盖访存延迟**：一个 warp 等显存时，调度器立刻切到另一个就绪的 warp（零开销切换）。这就是"延迟隐藏（latency hiding）"，理解它才理解后面所有优化。

### 1.2 访存层级（必须背下的数字级直觉）

| 层级 | 容量（A100 量级） | 延迟 | 谁管理 |
| --- | --- | --- | --- |
| 寄存器 | 每线程 ~256 个 32-bit | ~1 cycle | 编译器 |
| 共享内存（shared） | 每 SM ~100-164 KB | ~20-30 cycles | **程序员显式** |
| L2 | 40 MB | ~200 cycles | 硬件 |
| 显存（HBM） | 40/80 GB | ~500 cycles | 硬件 |

关键结论：**共享内存比显存快 ~20 倍但小 ~100 万倍**——所有"分块（tiling）"优化的本质，都是把反复使用的数据搬进共享内存再用。

### 1.3 访存合并（Coalescing）

一个 warp 的 32 个线程**同时**发出访存请求。若它们访问的 32 个地址落在**一段连续内存**里，硬件合并成一次事务（32 线程 × 4 字节 = 128 字节，恰好一条缓存行）；若地址散乱（如按列遍历行主序矩阵），每个线程各触发一次独立事务——**带宽利用率可能掉到 1/32**。

$$
\text{合并}:\ [a_0, a_1, \dots, a_{31}] \text{ 连续} \to 1 \text{ 次事务} \qquad
\text{非合并}:\ [a_{0}, a_{N}, a_{2N}, \dots] \to \text{最多 } 32 \text{ 次事务}
$$

### 1.4 Roofline 模型：先判断瓶颈再动手

算子的**算术强度（Arithmetic Intensity, AI）** = 浮点运算数 / 搬运字节数。硬件有两道墙：算力上限 $\pi$（FLOPS）与带宽上限 $\beta$（Byte/s）：

$$
\text{可达性能} = \min\big(\pi,\ \beta \times \text{AI}\big), \qquad \text{拐点 AI}^* = \pi / \beta
$$

- AI < AI\*：**访存瓶颈**（性能 = β × AI，优化方向：减少搬运/提高复用）；
- AI > AI\*：算力瓶颈（优化方向：Tensor Core、指令级并行）。

例（A100）：$\pi \approx 312$ TFLOPS(FP16)，$\beta = 2$ TB/s → AI\* ≈ 156 FLOP/Byte。**LLM 的绝大多数算子（elementwise、norm、decode GEMV）AI 远低于 156，全是访存瓶颈**——这就是"理解访存才理解推理优化"的含义。

### 1.5 手算 7B decode 上限（本周交付之一）

decode 每生成一个 token，需把全部权重（≈ 14 GB @FP16）过一遍，计算量相对搬运可忽略（batch=1 时 AI ≈ 2 FLOP/Byte）：

$$
\text{tokens/s} \le \frac{\beta}{\text{权重字节数}} = \frac{2 \times 10^{12}\ \text{B/s}}{14 \times 10^9\ \text{B}} \approx 143\ \text{tokens/s}
$$

推论三连：① 这是 batch=1 的天花板，任何框架都超不过（KV cache 搬运还会再扣一点）；② batch 增大 → 同一份权重摊给更多序列 → 每 token 成本下降（连续批处理的收益来源）；③ INT4 量化把权重缩到 ~3.7 GB → 上限升 ~3.8×（Stage 2 第 5-6 周已算过同一笔账）。

---

## 2. 系统架构与数据流：一个 kernel 的生命周期

```
CPU 侧：分配显存 → H2D 拷贝数据 → 配置 grid/block → 启动 kernel（异步）
GPU 侧：grid 拆给各 SM → block 内线程协作（共享内存 + __syncthreads）
        → warp 锁步执行 → 访存（合并与否决定效率）
CPU 侧：同步等待 → D2H 拷回结果
```

---

## 3. 实现与验证（本周交付核心）

### 3.1 vector_add：你的第一个 kernel（完整可编译）

```cuda
// vector_add.cu —— 逐行注释版
#include <cstdio>
#include <cuda_runtime.h>

// 在 GPU 上执行的函数；每个线程算一个元素
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // ① 全局线程编号：blockIdx.x（第几个块）× blockDim.x（块宽）+ 块内偏移
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // ② 越界保护：线程总数通常向上取整到块宽的倍数
    if (i < n) {
        c[i] = a[i] + b[i];   // ③ 相邻线程读相邻地址 → 完美合并访存
    }
}

int main() {
    int n = 1 << 24;                       // 16M 个元素
    size_t bytes = n * sizeof(float);
    float *ha = new float[n], *hb = new float[n], *hc = new float[n];
    for (int i = 0; i < n; i++) { ha[i] = 1.0f; hb[i] = 2.0f; }

    float *da, *db, *dc;                   // d 前缀 = device（显存）
    cudaMalloc(&da, bytes); cudaMalloc(&db, bytes); cudaMalloc(&dc, bytes);
    cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);   // H2D
    cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);

    int block = 256;                                  // 块宽：128/256/512 常见
    int grid = (n + block - 1) / block;               // 块数：向上取整
    vector_add<<<grid, block>>>(da, db, dc, n);       // <<<>>> 启动 kernel

    cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost); // D2H
    for (int i = 0; i < n; i++)                       // 断言式验证
        if (hc[i] != 3.0f) { printf("FAIL at %d\n", i); return 1; }
    printf("vector_add PASS: %d elements, all == 3.0\n", n);
    return 0;
}
```

编译与运行：

```bash
nvcc -O3 -arch=sm_86 vector_add.cu -o vector_add   # sm_86=30/40系; A100 用 sm_80
./vector_add
# 预期输出：vector_add PASS: 16777216 elements, all == 3.0
```

### 3.2 matrix_transpose：合并访存的对照组（完整可编译）

```cuda
// transpose.cu —— 同一份数据，两种访存模式，性能差 3-10 倍
#include <cstdio>
#include <cuda_runtime.h>
#define N 4096   // N×N 矩阵

// 非合并版：按行读（合并）、按列写（跨行，非合并！）
__global__ void transpose_naive(const float* in, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 列
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 行
    if (x < N && y < N)
        out[x * N + y] = in[y * N + x];  // 写侧相邻线程地址跨 N → 非合并
}

// 合并版：用共享内存中转，读写两侧都合并
__global__ void transpose_shared(const float* in, float* out) {
    __shared__ float tile[32][33];       // +1 padding 避免 bank conflict
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (x < N && y < N) tile[threadIdx.y][threadIdx.x] = in[y * N + x]; // 合并读
    __syncthreads();                     // 等整个块写完共享内存
    x = blockIdx.y * 32 + threadIdx.x;   // 读写块的坐标互换
    y = blockIdx.x * 32 + threadIdx.y;
    if (x < N && y < N) out[y * N + x] = tile[threadIdx.x][threadIdx.y]; // 合并写
}

int main() {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h = new float[N*N];  for (int i = 0; i < N*N; i++) h[i] = i % 7919;
    float *di, *do_; cudaMalloc(&di, bytes); cudaMalloc(&do_, bytes);
    cudaMemcpy(di, h, bytes, cudaMemcpyHostToDevice);
    dim3 block(32, 32), grid(N/32, N/32);

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    for (int v = 0; v < 2; v++) {
        cudaEventRecord(s);
        for (int it = 0; it < 20; it++)
            v == 0 ? transpose_naive<<<grid, block>>>(di, do_)
                   : transpose_shared<<<grid, block>>>(di, do_);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        double gbps = 2.0 * bytes * 20 / (ms * 1e6);   // 读+写，GB/s
        printf("%s: %.3f ms/20iter, %.1f GB/s\n",
               v == 0 ? "naive  " : "shared ", ms, gbps);
    }
    // 验证：转置两次应还原
    transpose_shared<<<grid, block>>>(di, do_);
    transpose_shared<<<grid, block>>>(do_, di);
    cudaMemcpy(h, di, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N*N; i++) if (h[i] != i % 7919) { printf("FAIL\n"); return 1; }
    printf("transpose PASS（两次转置还原）\n");
    return 0;
}
```

```bash
nvcc -O3 -arch=sm_86 transpose.cu -o transpose && ./transpose
# 预期：shared 版带宽为 naive 版的 3-10 倍（云卡型号不同倍数不同）
```

**交付要点**：把两版的 `GB/s` 抄进实验记录；用 `ncu --metrics dram__throughput.avg.pct_of_peak ./transpose` 抓显存带宽利用率，确认合并版接近峰值、非合并版远低于峰值——这就是"90% kernel 慢在访存"的亲眼见证。

---

## 4. 工程权衡与失效模式

### 4.1 权衡

- **块大小选择**：太小（< 128）→ warp 数不足、延迟隐藏差；太大 → 寄存器/共享内存摊薄、occupancy 反降。128~512 是安全区，用 `ncu` 的 occupancy 报告校准。
- **occupancy 不是越高越好**：它只是延迟隐藏的**手段**；访存已饱和时，更高的 occupancy 只是更多线程抢同一条带宽。
- **CPU-GPU 同步是隐形杀手**：每次 `cudaMemcpy` D2H 或 `cudaDeviceSynchronize` 都会打断流水——推理框架都在拼命减少同步点。

### 4.2 失效模式

1. **越界写显存**：症状——偶发崩溃/结果错乱；根因——忘记 `if (i < n)` 保护，多余线程写脏相邻内存；修复——边界检查（§3.1 ②）+ `compute-sanitizer` 体检。
2. **warp 分支发散**：症状——`if/else` 两侧都慢；根因——同 warp 内线程走了不同分支，硬件只能串行执行两侧；修复——让分支按 warp 对齐（如 `if (threadIdx.x < 32)` 整 warp 一致）。
3. **共享内存 bank conflict**：症状——共享内存版没达到预期加速；根因——同 warp 多线程访问同一 bank 的不同地址被串行化；修复——`+1` padding（§3.2 的 `[32][33]`）。
4. **用 CPU 思维写循环**：症状——在线程里写 for 循环遍历整个数组；根因——没有把迭代映射到线程；修复——"一个元素一个线程"的映射思维。

---

## 5. 延伸思考题（含解析）

**Q1**：GPU 单核比 CPU 慢，为什么深度学习快几十倍？
**A**：深度学习的并行度天然巨大（百万级元素级操作），GPU 用 22 万级并发线程 + 零开销切换掩盖访存延迟，总吞吐 = 海量线程 × 单线程慢速，净效果远超 CPU 的少量快核。前提是负载有足够的并行度与规整访存。

**Q2**：warp 发散为什么代价是"串行"而不是"报错"？
**A**：SIMT 硬件只有一套取指/执行单元，同 warp 必须执行同一条指令；遇到分支时先执行 if 分支（else 侧线程被掩码屏蔽），再执行 else 侧——两条路径的时间相加。

**Q3**：Roofline 拐点 AI\* = π/β 在 H100 上约是多少？对"该不该写融合算子"意味着什么？
**A**：H100：~990 TFLOPS(FP16) / 3.35 TB/s ≈ 295 FLOP/Byte。常见 elementwise 算子 AI ≈ 0.25~2，远在拐点左侧 → 融合（提高 AI：一次搬运做多步计算）几乎总是赚。这是第 3 周的理论预告。

**Q4**：transpose_shared 里 `tile[32][33]` 的 +1 是什么？不加会怎样？
**A**：共享内存分 32 个 bank（4 字节/bank），32×32 矩阵按列访问时 32 个线程打同一 bank → 32 路冲突（串行 32 倍）。+1 列 padding 让列访问错开到不同 bank。

**Q5**：decode 上限 143 tokens/s 是"权重 14GB ÷ 带宽"，那 prefill 呢？
**A**：prefill 一次前向处理整个 prompt（seq 个 token 共享同一份权重搬运），是算力+带宽混合瓶颈，吞吐用 tokens/s 计远高于 decode，但首 token 延迟（TTFT）正比于 prompt 长度——"prefill 看算力、decode 看带宽"是推理优化的第一口诀。

---

## 6. 实操附录：云 GPU 环境速查（每次开实验先过一遍）

```bash
# ① 确认硬件与算力（sm_xx 决定 nvcc 的 -arch 参数）
nvidia-smi --query-gpu=name,memory.total --format=csv
# 用 deviceQuery 或：
python -c "import torch; p=torch.cuda.get_device_properties(0); \
print(p.name, f'sm_{p.major}{p.minor}', p.multi_processor_count, 'SMs')"

# ② 工具链版本
nvcc --version && ncu --version && nsys --version

# ③ 编译模板（按卡的算力改 -arch）
nvcc -O3 -arch=sm_86 kernel.cu -o kernel && ./kernel
#   A100 → sm_80 | A10/L20 → sm_86 | 4090 → sm_89 | H100 → sm_90

# ④ 体检三件套
compute-sanitizer ./kernel          # 越界/竞态检查（交付前必跑）
ncu ./kernel                        # 单 kernel 剖析
nsys profile ./kernel               # 系统时间线
```

**常见坑**：`-arch` 与实卡不符 → 编译通过但运行报 `no kernel image`；云上容器缺 `compute-sanitizer` → 用 `apt`/`conda` 补 CUDA Toolkit 完整组件。把这些写进你的实验启动脚本，每次开机 30 秒自检。

### 6.2 本周两个实验的预期数字锚点（对不上就排查）

| 实验 | 预期 | 偏差时查什么 |
| --- | --- | --- |
| vector_add 带宽 | 达峰值带宽 60-80%（简单算子上限） | 块大小、n 是否太小 |
| transpose naive | 峰值带宽 10-30% | 非合并写是预期内的慢 |
| transpose shared | 峰值带宽 50-80%，为 naive 的 3-10× | < 3× → 查 bank conflict |
| 7B decode 手算 | ~143 tokens/s @2TB/s | 代入你租的卡的真实带宽重算 |

**爬坡验收标准**：合上文档，仅凭"一个元素一个线程 + 越界保护 + H2D/D2H"三个提示，独立写出 `vector_add` 并编译通过——这是从"读懂"到"会写"的第一道门。

### 6.1 占用率（occupancy）快速心算法

$$
\text{occupancy} = \frac{\text{每 SM 活跃 warp 数}}{\text{每 SM 最大 warp 数（如 64）}}
$$

三个限制源取最小：① 每线程寄存器数 × 线程数 ≤ 寄存器堆；② 每块共享内存 × 块数 ≤ 配额；③ 线程总数上限。用 `nvcc --ptxas-options=-v` 看每 kernel 的寄存器/共享内存用量，用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` API 算实际值——第 2 周调 MatMul 时会反复用到这套流程。

---

## 本周交付清单

- [ ] 背下访存层级表与 Roofline 公式（含 A100 的 AI\* ≈ 156）。
- [ ] 手算 7B decode 上限：14 GB / 2 TB/s ≈ 143 tokens/s（闭卷）。
- [ ] 云 GPU 上编译运行 `vector_add`（PASS 输出）与 `transpose`（两版带宽对比）。
- [ ] 用 `ncu` 抓 transpose 两版的带宽利用率，截图存档。
- [ ] 合上文档仿写一遍 `vector_add`（爬坡验收：编译通过 + 输出正确）。
