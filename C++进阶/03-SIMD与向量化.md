# 03 - SIMD 与向量化

## 本篇目标

- 建立"数据并行"思维：一条指令处理多个数据，理解它为什么快、什么情况下快不起来。
- 认识 x86 指令集族（SSE/AVX/AVX2/AVX-512）与 ARM NEON 的事实参数，会用 AVX2 intrinsics 手写 float 求和并实测加速比。
- 掌握对齐（`alignas`/`aligned_alloc`）与未对齐访问的崩溃现场，学会用 `-fopt-info-vec` 验证自动向量化。
- 知道跨平台的工程出路（Highway/Xsimd、运行时派发），以及"什么时候 CPU SIMD 还够用、该上 GPU"。

前置：《计算机系统与网络》03（缓存层级与局部性）。

## 1. 数据并行思想

普通代码是 SISD：一条指令处理一个数据。SIMD（Single Instruction, Multiple Data）让一条指令同时对 N 个数据做同样的运算——CPU 里叫向量化，GPU 里这个思想被推到极致（几千个"线程"锁步执行同一条指令，见《模型性能优化》Stage 3）。

对 `a[i] = b[i] + c[i]` 这样的循环，标量执行每轮要做：取指 → 一次加法 → 下一轮。向量化后一次加 8 个 float（256 位），取指和分支开销摊薄到 1/8，浮点吞吐理论上 ×8。这就是 GEMM、卷积、量化、音频重采样这类"规整的批量算术"能被加速的本质。

两个先决条件同样重要：**数据在内存里要连续**（一次 load 取到的是一整段连续地址），**控制流要规整**（`if` 分支会让不同 lane 走不同路，得靠掩码模拟）。这一章的所有技巧本质上都在伺候这两条。

## 2. x86 指令集族事实表

| 指令集 | 年代 | 向量宽度 | 寄存器 | 代表能力 |
| --- | --- | --- | --- | --- |
| MMX | 1996 | 64 bit | 复用 FPU 栈 8 个 | 整数打包运算，已淘汰 |
| SSE | 1999 | 128 bit | XMM0–XMM7（64 位模式 16 个） | float 打包、`movaps/movups` |
| AVX | 2011 | 256 bit | 16 个 YMM（XMM 是低 128 位） | float/double 打包，VEX 编码 |
| AVX2 | 2013 | 256 bit | 同上 | 整数打包用满 256 位，gather |
| AVX-512 | 2017 | 512 bit | 32 个 ZMM + 8 个掩码寄存器 | 掩码掩蔽、逐 lane 条件 |
| NEON（ARM） | 2008– | 128 bit | 32 个 V 寄存器 | ARM 服务器/手机基线，RK3588 即 NEON |

事实要点：**x86-64 基线自带 SSE2**（所以 `#include <immintrin.h>` 的 SSE2 代码无需任何编译选项）；AVX2 于 2013 年随 Haswell 普及，2020 年后的 x86 服务器/桌面几乎都有；AVX-512 在消费级上仍不齐（Intel 12/13 代酷睿小核甚至砍掉），这也是"运行时派发"存在的理由。本机（i5-12600K）实测 `/proc/cpuinfo` 含 `avx2`、不含 `avx512f`。宽度不等于免费加速：AVX-512 降频问题、寄存器数量、load 端口压力都会咬掉理论值。

## 3. intrinsics 入门：AVX2 求和实测

intrinsics 是"看着像函数调用的汇编助记符"：`_mm256_add_ps(a, b)` 编译成一条 `vaddps`。命名规则：`_mm` 前缀 + 寄存器宽度（`128`=SSE、`256`=AVX、`512`=AVX-512）+ 操作 + 数据类型后缀（`ps`=packed single 精度，`epi32`=packed int32）。

```cpp
// simdsum.cpp —— 标量 vs AVX2 的 float 数组求和
#include <immintrin.h>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

float sum_scalar(const float* a, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) s += a[i];   // 注意：依赖链，一轮等上一轮
    return s;
}

float sum_avx2(const float* a, size_t n) {
    __m256 acc = _mm256_setzero_ps();            // 8 个 float 全 0
    for (size_t i = 0; i < n; i += 8) {
        __m256 v = _mm256_loadu_ps(a + i);       // u = unaligned：不要求 32 字节对齐
        acc = _mm256_add_ps(acc, v);             // 8 个加法一条指令
    }
    // 水平归约：把 8 个 lane 加成一个数（4 步）
    __m128 lo = _mm256_castps256_ps128(acc);     // 低 128 位（零代价重解释）
    __m128 hi = _mm256_extractf128_ps(acc, 1);   // 高 128 位
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);                       // 4+4 → 2+2
    s = _mm_hadd_ps(s, s);                       // → 1
    return _mm_cvtss_f32(s);
}

int main() {
    const size_t n = 8 * 1024 * 1024;            // 8M float = 32 MB
    std::vector<float> a(n, 1.0f);
    using clk = std::chrono::steady_clock;
    float s1 = 0, s2 = 0;
    auto t0 = clk::now();
    for (int r = 0; r < 20; ++r) s1 += sum_scalar(a.data(), n);
    auto t1 = clk::now();
    for (int r = 0; r < 20; ++r) s2 += sum_avx2(a.data(), n);
    auto t2 = clk::now();
    double t_s = std::chrono::duration<double, std::milli>(t1 - t0).count() / 20;
    double t_v = std::chrono::duration<double, std::milli>(t2 - t1).count() / 20;
    std::cout << "标量: " << s1 << "  " << t_s << " ms/轮\n";
    std::cout << "AVX2: " << s2 << "  " << t_v << " ms/轮\n";
    std::cout << "加速比: " << t_s / t_v << "x\n";
}
```

```bash
g++ -std=c++20 -O2 -mavx2 simdsum.cpp -o simdsum && ./simdsum
# 输出（i5-12600K 实测）：
# 标量: 1.67772e+08  4.43545 ms/轮
# AVX2: 1.67772e+08  1.22329 ms/轮
# 加速比: 3.62584x
```

三个观察：加速比 3.6x 而不是 8x——标量版浮点加法约 4 周期延迟形成依赖链，AVX2 版同样有链，双方都被延迟而非吞吐卡住；**真正的大加速要拆多条独立累加链**（4 个 acc 寄存器轮转，可逼近带宽极限）；`-mavx2` 让编译器"可以生成 AVX2 指令"，本例手写 intrinsics 的部分总是 AVX2，但主程序其他代码也受益。`loadu`（不对齐 load）在 AVX 之后基本无惩罚（拆成两条对齐 load 的代价已被硬件吸收），所以新代码不必强求 `load` 的对齐——但对齐仍是硬需求的场合见下一节。

## 4. 对齐：alignas、aligned_alloc 与段错误现场

SIMD 数据类型有硬性对齐要求的历史（SSE 时代 `movaps` 对未对齐地址直接 #GP 异常）。现代写法用 `alignas` 声明、用对齐分配器拿内存：

```cpp
// align_demo.cpp
#include <aligned_alloc>  // C++17 起在 <cstdlib>，这里示意
#include <cstdlib>
#include <iostream>

struct alignas(32) Vec8 {      // 强制 32 字节对齐（一个 YMM 寄存器宽）
    float v[8];
};
static_assert(alignof(Vec8) == 32 && sizeof(Vec8) == 32);

int main() {
    // malloc 保证 max_align_t（一般 16）对齐；要 32 对齐得用 aligned_alloc
    float* p = static_cast<float*>(std::aligned_alloc(32, 32 * sizeof(float)));
    std::cout << "地址 % 32 = " << reinterpret_cast<uintptr_t>(p) % 32 << '\n';  // 0
    std::free(p);
}
```

```bash
g++ -std=c++20 -Wall align_demo.cpp -o align_demo && ./align_demo
# 输出：地址 % 32 = 0
```

段错误现场（真实复现方法，别在生产环境试）：**用 `new`/`malloc` 的默认 16 字节对齐拿内存，再喂给要求 32/64 字节对齐的 `movaps/vmovaps`**。典型症状是 SIGSEGV 且 `%rip` 指向一条 `vmovaps`/`movaps` 指令，`addr2line` 定位后多半是 `load` 写成了 `load`（aligned 版）而不是 `loadu`，或结构体忘加 `alignas` 导致数组里隔元素跨了边界。另一个日常坑：`std::vector<Vec8>` 在 C++17 前 allocator 不感知对齐、会静默降级，C++17 起标准库按 `alignof` 正确对齐——老代码迁移时注意。

## 5. 自动向量化：怎么写编译器才会向量化

手写 intrinsics 是重武器；第一选择永远是**让编译器自动向量化**（vectorizer），它生成的代码可移植、可维护。条件归纳成一句话：**循环次数可数、无循环间依赖、内存访问可证明不重叠、分支可化为掩码**。

```cpp
// av.cpp
void scale(float* __restrict a, const float* __restrict b, float k, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] = b[i] * k;
}
```

```bash
g++ -std=c++20 -O2 -ftree-vectorize -fopt-info-vec-optimized -c av.cpp -o av.o
# 输出：
# av.cpp:3:26: optimized: loop vectorized using 16 byte vectors
# av.cpp:3:26: optimized: loop vectorized using 8 byte vectors
```

（GCC 11 的 `-O2` 默认不开向量化，需要 `-O3` 或显式 `-ftree-vectorize`；GCC 12 起 `-O2` 启用了保守版向量化——同一份代码跨编译器版本性能会自己变，这也是性能测试要锁工具链版本的原因。）反过来，**写出会阻止向量化的循环**，让 `-fopt-info-vec-missed` 告诉你原因：

```cpp
// av2.cpp —— 前缀和式依赖
void accum(float* a, size_t n) {
    for (size_t i = 1; i < n; ++i) a[i] += a[i-1] * 0.5f;
}
```

```bash
g++ -std=c++20 -O3 -fopt-info-vec-missed -c av2.cpp -o av2.o
# 输出：
# av2.cpp:3:26: missed: couldn't vectorize loop
# av2.cpp:3:49: missed: not vectorized, possible dependence between data-refs *_5 and *_2
```

编译器直接点名"data-refs 之间存在依赖"。让自动向量化更常成功的清单（经验做法）：数组参数加 `__restrict` 声明不重叠（否则编译器必须为 a、b 可能互指生成保护代码）；内层循环用 `size_t i` 从 0 到 n（计数可证明）；循环体保持 `a[i] = f(b[i])` 形状，把分支改成 `select`（`d[i] = c ? x : y` 或 `std::min/max`）；热点循环的步长固定、避免函数调用（内联后向量器才看得见整体）。

## 6. gather / scatter：不连续访问怎么办

规整连续内存一 load 搞定；**步长不为 1**（取 `a[i*4]`）或**随机下标**（`a[idx[i]]`）就得 gather/scatter：

```cpp
// AVX2 gather：按 8 个 int32 下标从 float 数组各取一个
__m256i idx = _mm256_loadu_si256(...);        // 8 个下标
__m256 v = _mm256_i32gather_ps(a, idx, 4);    // 4 = sizeof(float)
```

gather 把 8 次 load 压成一条指令，但**硬件逐 lane 执行**，吞吐远低于连续 load（早期实现一个周期只处理 1–2 个 lane）；AVX-512 才有对应 scatter。经验做法：**下标规整（固定步长）→ 手动拆成多个连续 load + 重排（shuffle）更快；真随机 → 先考虑数据重排（排序/分桶）把随机变局部**。GPU 上 gather/scatter 是一等公民（显存随机访问天然支持），这是"该上 GPU"的信号之一。

## 7. 跨平台怎么办：库与运行时派发

三重困境：RK3588 是 ARM NEON，x86 服务器是 AVX2，老机器只有 SSE2。出路按投入排序：

1. **先靠自动向量化**（第 5 节）：源代码不写任何 ISA，`-march` 交给部署方。这是 80% 场景的正解。
2. **运行时 CPU 派发**：同一算法写 2–3 个实现，启动时探测 CPU 支持，选一条。GCC 的函数多版本（FMV）可以自动生成派发壳：

```cpp
__attribute__((target_clones("avx2", "sse4.2", "default")))
float dot(const float* a, const float* b, size_t n) { /* 朴素循环 */ }
// 链接器为每个 ISA 生成一份 + ifunc 派发；运行时首次调用自动选最优实现
```

   手写探测用 `__builtin_cpu_supports("avx2")` 或 cpuid。分发判断放程序启动一次完成，别在热循环里每次判断（经验做法）。
3. **抽象库**：Google Highway、Xsimd 用一套 API 编译到 x86/ARM 各 ISA，派发与 fallback 内建。代价是依赖与构建复杂度；收益是"写一次、处处向量化"。simdjson、llama.cpp 内部正是这一路（llama.cpp 的 ggml 对 x86 手写 AVX2/AVX-512 分支 + ARM NEON 分支 + 运行时选择，可作为"派发 + 手写热点"的活标本）。

## 8. 与 GPU 的关系：什么时候 CPU SIMD 还够用

CPU SIMD 与 GPU 都是数据并行，但分工不同（衔接《模型性能优化》Stage 3）：

| 维度 | CPU SIMD | GPU |
| --- | --- | --- |
| 数据规模 | 小批量（KB–MB） | 大批量（MB–GB） |
| 启动开销 | 无（函数调用级） | 有（kernel launch，μs 级） |
| 适合 | 延迟敏感、分支多、吞吐低 | 规整稠密、吞吐优先 |
| 典型 | 预处理、量化/反量化、重采样、CRF | 矩阵乘、注意力、训练 |

经验做法：**单次计算量在几十 μs 以下、数据在 L2/L3 里、或输入输出需要频繁与标量控制流交织——CPU SIMD 是对的选择**；一次要算几百 MB 的稠密矩阵/整层注意力，别用 AVX2 硬磨。llama.cpp 的 CPU 路径把每个算子都 SIMD 化，但跑大模型时 GPU 版本仍快一个数量级——分工明确。端侧（RK3588）则是第三条路：CPU NEON 做预处理/后处理，NPU 做推理，CPU SIMD 与 NPU 各管一段（见第 10 章）。

## 常见误区

1. **理论加速比 8x 当成实际值**：被内存带宽、依赖链、shuffle 开销稀释。先测，再谈优化（本章实测 3.6x）。
2. **`-mavx2` 发布通用二进制**：在无 AVX2 机器上直接 SIGILL（非法指令）。分发二进制用 `-march=x86-64-v2`（SSE4.2 基线）或运行时派发；AVX2 专属构建单独出包。
3. **以为标量循环编译器总会向量化**：依赖链、可能别名、内层分支都会静默阻止。**必须用 `-fopt-info-vec`（GCC）/ `-Rpass=loop-vectorize`（Clang）验证**，"应该向量化了"不算数。
4. **对齐随机踩坑**：结构体没 `alignas`、`new` 默认 16 字节对齐、`movaps` 炸 SIGSEGV。所有 SIMD 类型成员数组统一 `alignas(32/64)`，分配用 `aligned_alloc`/C++17 对齐感知 allocator。
5. **gather 当连续 load 用**：随机 gather 吞吐只有连续 load 的几分之一，热点路径的随机访问先想数据布局。
6. **微基准忘了看 `sum` 结果**：整个循环被优化器删掉，测出 0.0001ms 的"超光速"。结果必须被消费（打印/累加到全局），或用 benchmark 库的 `DoNotOptimize`（第 6 章展开）。

## 本篇小结

- SIMD = 一条指令 N 个数据；收益来自规整控制流 + 连续内存，两者缺一不可。
- x86 事实：x86-64 基线 SSE2、AVX2（256bit）已普及、AVX-512 不齐；ARM 是 NEON（128bit，RK3588）。
- intrinsics 入门套路：`loadu → add → storeu` + 水平归约；实测 3.6x，想更高拆独立累加链。
- 对齐：`alignas` 声明、`aligned_alloc` 分配；`movaps` 对未对齐地址是 SIGSEGV 现场而不是警告。
- 自动向量化优先于手写：`__restrict` + 可数循环 + 无依赖 + 无分支，用 `-fopt-info-vec` 验证；missed 信息会直接点名原因。
- 跨平台：自动向量化 → 运行时派发（FMV/cpuid）→ Highway/Xsimd；与 GPU 的分工看数据规模与启动开销。

## 练习

1. 把第 3 节 `sum_avx2` 改成 4 条独立累加链（`acc0..acc3` 轮转），实测加速比从 3.6x 提升到多少？再解释为什么没有到 8x（提示：内存带宽 ~40 GB/s vs 数据量）。
2. 用 `-fopt-info-vec-missed` 找出下面循环不能向量化的原因并修好：`for (size_t i = 0; i < n; ++i) a[i] = (b[i] > 0 ? b[i] : 0) * c[i];`
3. 写一个 `struct alignas(64) Counter { long v; };` 数组，用 `_mm256_load_ps` 读第一个元素验证对齐；再把数组换成普通 `new`，用 gdb（或直接看 core）复现 SIGSEGV。
4. `target_clones("avx2","default")` 编译一个点积函数，用 `objdump -d` 找出派发桩（`*ABS*+0x...@plt`/resolver），说明运行时怎么选实现。
5. 在 RK3588（或任意 ARM）上把第 3 节程序改用 NEON intrinsics（`vld1q_f32`/`vaddq_f32`/`vaddvq_f32`）重写，对比 x86 上的 API 差异；没有 ARM 机器时，说出两者的三处不同即可。
