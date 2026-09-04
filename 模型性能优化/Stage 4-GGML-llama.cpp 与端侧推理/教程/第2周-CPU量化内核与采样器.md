# 第 2 周教程：CPU 量化内核与采样器——llama.cpp 为什么在 CPU 上也能打

> **本周要回答的问题**
> 1. AVX2/NEON 在量化 kernel 里到底干了什么？为什么量化对 CPU 是"雪中送炭"而不是"锦上添花"？
> 2. decode 阶段的 GEMV 怎么吃满内存带宽？prefetch/unroll/SIMD 打包各管什么？
> 3. 采样器链（temp → top-k → top-p → min-p）的顺序与实现在哪？
> 4. 从 HF 权重到可服务的 GGUF，完整命令链是什么？

对应学习计划：第 2 周。交付物：① x86 机器上完成「HF → convert_hf_to_gguf → imatrix → Q4_K_M/Q3_K_S 双版本量化」全流程；② 对比两档量化的 tokens/s、内存与生成质量；③ `perf` 火焰图确认 GEMV 占时。

---

## 1. 第一性原理：CPU 推理的唯一出路 = 带宽 × SIMD

### 1.1 为什么量化对 CPU 是雪中送炭

decode 是 GEMV：每 token 搬全部权重一遍。CPU 内存带宽 ~50-100 GB/s（比 HBM 低 20-40 倍），若不量化，7B FP16（14 GB）上限只有 ~5-10 tokens/s。**量化把搬运量压到 1/4，上限直接 ×4**——CPU 没有 Tensor Core 可挖，带宽就是唯一的矿。

同时量化让 SIMD 更划算：一条 `vpmaddubsw`（AVX2）一周期处理 32 个 INT8 乘累加；数据位宽越小，同一条指令吞的有效权重越多。

### 1.2 SIMD 指令集分工

| 指令集 | 宽度 | 在量化内核中的角色 |
| --- | --- | --- |
| AVX2（x86） | 256-bit | INT8 打包乘累加、4-bit 移位解包（`vpshufb` 查表解包是 Q4 内核的招牌动作） |
| AVX-512 | 512-bit | 同上加宽，但部分 CPU 降频——收益需实测 |
| NEON（ARM） | 128-bit | RK3588 A76 核上的对应物：`vmlaq` 乘累加、`vtbl` 查表解包 |

Q4_K 内核的通用骨架：加载 16 字节打包权重 → 查表/移位解出 32 个 4-bit 值 → 乘子块 scale → 与激活（Q8_0 格式）做点积 → 累加。**量化内核的性能 = 解包吞吐 × 访存效率**，两者都要顶到硬件上限。

### 1.3 GEMV 吃满带宽的三板斧

1. **prefetch**：提前把后续块的权重发进缓存，掩盖内存延迟（`__builtin_prefetch` 或硬件预取器友好的顺序访问）；
2. **unroll**：循环展开 4-8 次，摊薄循环开销、暴露指令级并行，让乱序执行器有活干；
3. **多线程分块**：行维度切给各核，每核顺序扫自己那段——保证每个核的访存都是连续的（合并的 CPU 版）。

$$
\text{tokens/s} \approx \frac{BW_{\text{mem}} \times \eta}{\text{权重字节数}}, \qquad \eta = \text{带宽利用率（好内核 > 80\%）}
$$

### 1.4 采样器链：生成质量的手术刀

llama.cpp 的采样按固定顺序作用在 logits 上：

```
logits ─► repetition_penalty ─► temperature ─► top-k ─► top-p/nucleus ─► min-p ─► 归一化 ─► 采样
```

- **temperature**：$p_i \propto \exp(z_i / T)$，$T < 1$ 收紧、$T > 1$ 摊平；
- **top-k**：只保留概率前 $k$；**top-p**：保留累积概率达到 $p$ 的最小集合；
- **min-p**：保留 $p_i \ge p_{\max} \times p_{\text{threshold}}$ 的 token——比 top-p 更抗"全分布都很平"的病态场景；
- 顺序敏感：温度先改变分布形状，截断才有意义；截断在归一化之前。

### 1.5 KV Cache 量化

长上下文的 KV cache 同样吃内存。llama.cpp 支持把 KV 存成 Q8_0（`--cache-type-k q8_0`）：体积约 1/2，对质量影响通常 < 0.1 PPL——端侧长上下文的关键开关。

---

## 2. 实现与验证（本周交付核心）

### 2.1 完整命令链：HF → imatrix → 双版本量化

```bash
# 0) 准备（x86 机器，一次性）
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build -j --config Release
pip install gguf numpy sentencepiece

# 1) HF 权重 → GGUF（FP16 存储）
python convert_hf_to_gguf.py Qwen/Qwen2.5-1.5B-Instruct \
    --outfile qwen2.5-1.5b-f16.gguf --outtype f16

# 2) imatrix：用校准数据统计每通道重要性（低比特必需）
./build/bin/llama-imatrix -m qwen2.5-1.5b-f16.gguf \
    -f calibration_text.txt --chunks 100 -o imatrix.dat

# 3) 双版本量化
./build/bin/llama-quantize --imatrix imatrix.dat \
    qwen2.5-1.5b-f16.gguf qwen2.5-1.5b-Q4_K_M.gguf Q4_K_M
./build/bin/llama-quantize --imatrix imatrix.dat \
    qwen2.5-1.5b-f16.gguf qwen2.5-1.5b-Q3_K_S.gguf Q3_K_S
```

### 2.2 对比评测脚本（完整可运行）

```bash
#!/bin/bash
# bench_q4k_vs_q3ks.sh —— 对两档量化跑同样的基准
for tag in Q4_K_M Q3_K_S; do
  echo "===== $tag ====="
  ./build/bin/llama-cli -m qwen2.5-1.5b-$tag.gguf \
      -p "请用 300 字介绍量子计算的基本原理。" \
      -n 256 --temp 0.7 --top-p 0.8 --min-p 0.05 -t 8 | tee out_$tag.txt
  # llama-cli 末尾会打印 eval time：记录 tokens/s 与每 token 毫秒
  grep -E "eval time|load time" out_$tag.txt
done
# 记录三列：tokens/s、进程峰值内存（另开终端 /usr/bin/time -v 或 top）、
# 生成质量（长文本连贯性人工打分 1-5）
```

对比表（实测填写，预期形态）：

| 版本 | bpw | 文件体积 | tokens/s | 峰值内存 | 连贯性(1-5) |
| --- | --- | --- | --- | --- | --- |
| Q4_K_M | 4.85 | _实测_ | _实测_ | _实测_ | 预期 4-5 |
| Q3_K_S | 3.5 | _实测_ | _实测（更快）_ | _实测_ | 预期 3-4（长文本先崩） |

### 2.3 perf 火焰图：确认 GEMV 占时

```bash
perf record -g ./build/bin/llama-cli -m qwen2.5-1.5b-Q4_K_M.gguf -p "你好" -n 128
perf report                      # 或生成火焰图：
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
# 预期：dequantize/gemm 相关的 ggml 函数合计占 60-90% 的 CPU 时间
```

**交付断言**：火焰图里 `ggml_vec_dot_q4_K_q8_K`（或同族点积函数）必须是最热的单个函数——它直接证明"decode 就是搬权重 + 解包点积"这个模型。

---

## 3. 工程权衡与失效模式

### 3.1 权衡

- **Q4_K_M vs Q3_K_S**：体积差 ~30%，速度差 ~20-30%，但长文本连贯性差距不成比例地大（误差复合）——端侧内存不是硬到极限，别轻易降到 3-bit。
- **线程数**：`-t` 不是越大越好——超过物理大核数后，内存带宽争用与同步开销反噬。RK3588 上通常绑 4 个 A76 大核（`taskset`）。
- **KV 量化**：q8_0 KV 是免费午餐级；更激进的 KV 量化（q4）在长上下文下要重测质量。

### 3.2 失效模式

1. **imatrix 缺失 + 低比特**：症状——Q3 以下模型胡说八道；根因——误差预算无加权分配；修复——命令链不可跳过第 2 步（§2.1）。
2. **线程数过量反而变慢**：症状——`-t 16` 比 `-t 8` 慢；根因——小核拖后腿 + 带宽争用；修复——`perf` 确认，绑大核。
3. **生成重复循环**：症状——同一句话复读；根因——采样链缺 repetition penalty 或温度过低；修复——加 `--repeat-penalty 1.1`。
4. **量化后"变快但变笨"的错觉**：症状——短回答都正常、任务完成率下降；根因——只测了 tokens/s 没测任务；修复——固定一组提示词做质量回归（本周连贯性打分就是雏形）。

---

## 4. 延伸思考题（含解析）

**Q1**：为什么 Q8_0 是激活的常用格式（配合 Q4 权重）？
**A**：GEMV 点积两侧都要量化才能走整数乘累加快路径；激活动态范围大但只需"够用"的精度，Q8_0（8-bit + 块 scale）几乎无损且与 AVX2/NEON 的 INT8 指令完美匹配。权重侧才是需要精打细算的那一侧。

**Q2**：`vpshufb` 查表解包为什么比移位+掩码快？
**A**：一条 `vpshufb` 把 32 个 4-bit 索引直接查成 8-bit 值（并行查表），等价于"移位+掩码+符号扩展"好几条指令的合体——指令数减少 = 发射端口压力减少 = 解包不成为点积的瓶颈。

**Q3**：采样器为什么把 temperature 放在 top-k/top-p 之前？
**A**：截断是对"概率分布形状"的操作——温度先定形状（尖或平），截断才有稳定的语义。若先截断再降温，被裁掉的尾部在低温下本可能抬头，行为不可预测。

**Q4**：min-p 相比 top-p 解决了什么问题？
**A**：top-p 的阈值是累积概率——当模型很不确定时（分布平坦），它可能放进几十个低质量 token。min-p 用"相对最大概率"做门槛，分布再平也只放行接近头部的那些，抗"全平"病态更稳。

**Q5**：同一模型，为什么你的 x86 机器可能比 RK3588 快 5-10 倍？差在哪？
**A**：内存带宽（DDR4/5 双通道 ~50-100 GB/s vs LPDDR4x 共享带宽 ~30 GB/s 且 NPU/GPU 共享）+ 单核性能 + SIMD 宽度（256 vs 128 bit）。GEMV 的天花板 ≈ 带宽 × 利用率，三项全输就全输——这正是第 5-6 周选型报告要量化的事。

---

## 5. 亲手验证：用 Python 模拟 Q4_K 内核的解包-点积流水

下面把 §1.2 的内核骨架翻译成 Python 数值语义（验证你对解包流水的理解，非性能代码）：

```python
import numpy as np

def q4_kernel_simulate(w_int4: np.ndarray, x_int8: np.ndarray,
                       w_scale: float, x_scale: float) -> float:
    """
    模拟一次 Q4 权重 × Q8 激活的点积：
    ① 解包 4-bit 权重；② 乘各自块 scale；③ 整数域点积后统一缩放。
    与 AVX2 内核的数值语义一致（顺序：先整数累加、后乘 scale）。
    """
    assert w_int4.min() >= -8 and w_int4.max() <= 7
    assert x_int8.dtype == np.int8
    # 关键：整数域点积（对应 vpmaddubsw 一族指令），最后才乘 scale
    acc_int = int(np.dot(w_int4.astype(np.int32), x_int8.astype(np.int32)))
    return acc_int * w_scale * x_scale

np.random.seed(0)
n = 32                                   # 一个块的宽度
w = np.random.randint(-8, 8, n).astype(np.int8)
x = np.random.randint(-64, 64, n).astype(np.int8)
ws, xs = 0.031, 0.11

y_kernel = q4_kernel_simulate(w, x, ws, xs)
y_float = (w.astype(np.float32) * ws) @ (x.astype(np.float32) * xs)
assert abs(y_kernel - y_float) < 1e-3, "整数流水与浮点参考不一致"
print(f"整数域点积 = {y_kernel:.4f}，浮点参考 = {y_float:.4f} ✓")
# 结论：量化内核"快"的秘密 = 点积主体全在整数域完成，
# 浮点乘法只在每个块结束时做一次（n 个乘累加摊 1 次缩放）
```

**交付联动**：把这段语义与 `ggml-quants.c` 里 `ggml_vec_dot_q4_K_q8_K` 的注释对照读一遍——C 代码里的 `vpshufb`/移位就是在做这里的"解包"，`vpmaddubsw` 就是在做这里的 `np.dot` 整数版。

---

## 6. 采样器实战：温度与截断的数值直觉（可运行）

```python
import numpy as np

def sample_chain(logits, T=0.7, top_k=40, top_p=0.8, min_p=0.05):
    """按 llama.cpp 顺序复现采样链，返回候选 token 集合大小。"""
    z = np.asarray(logits, dtype=np.float64)
    p = np.exp(z / T); p /= p.sum()          # ① temperature
    idx = np.argsort(p)[::-1]
    keep = idx[:top_k]                        # ② top-k
    cum = np.cumsum(p[keep])
    keep = keep[:int(np.searchsorted(cum, top_p)) + 1]   # ③ top-p
    keep = keep[p[keep] >= p.max() * min_p]   # ④ min-p
    return keep

np.random.seed(0)
logits = np.random.randn(32000) * 2          # 模拟词表
logits[7] = 12.0; logits[42] = 11.0          # 两个头部
for T in (0.3, 0.7, 1.5):
    cand = sample_chain(np.full(32000, 0.0) + np.random.randn(32000)*0.1
                        + np.eye(32000)[7]*T*8, T=T)
    print(f"T={T}: 采样后候选数 = {len(cand)}")
# 预期：温度越低候选越少（分布越尖，截断后剩得越少）
# 思考：为什么低温 + top-p 小会导致复读？（尾部被裁光，概率集中循环）
```

---

## 本周交付清单

- [ ] 跑通 §2.1 完整命令链（convert → imatrix → 双量化），五个命令无文档复现。
- [ ] 填完 Q4_K_M/Q3_K_S 对比表（tokens/s、内存、连贯性三列）。
- [ ] `perf` 火焰图截图，标注最热的点积函数与其占时百分比。
- [ ] 验证 `--cache-type-k q8_0` 的内存节省（长上下文 2048+）。
- [ ] 用自己的话回答：imatrix 是什么？不做会怎样？（≤ 80 字）
