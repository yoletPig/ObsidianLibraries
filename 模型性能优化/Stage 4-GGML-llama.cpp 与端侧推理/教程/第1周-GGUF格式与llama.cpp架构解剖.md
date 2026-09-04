# 第 1 周教程：GGUF 格式与 llama.cpp 架构解剖

> **本周要回答的问题**
> 1. GGUF 文件的二进制布局是什么？KV 元数据区与 tensor 目录各存什么？
> 2. Q4_0 / Q4_K / IQ 系列的位宽构成分别是什么？Q4_K 的 4.5 bpw 怎么手算？
> 3. imatrix（重要性矩阵）为什么能让 2-3 bit 量化可用？
> 4. llama.cpp 的三层架构（ggml → llama → 应用）调用链长什么样？

对应学习计划：第 1 周。交付物：① `gguf-py` 解析脚本（打印元数据与每个 tensor 的量化类型/形状）；② 手写 `dequantize_row_q4_0` 与官方对齐；③ 图解 Q4_K 超块结构。

---

## 1. 第一性原理：端侧的量化格式必须"为 CPU 而设计"

### 1.1 为什么不能直接搬 GPU 格式

GPU 量化（GPTQ/AWQ）假设：有并行反量化内核、有显存带宽余量、按 group=128 查 scale。CPU 的约束完全不同：**内存带宽小一个量级（~50-100 GB/s）、SIMD 单元按固定宽度处理、随机访存昂贵**。因此 GGML 的格式设计目标是：

1. **块状自包含**：每 32 个权重一个块，scale 就在块内——解码时顺序读、零随机访存；
2. **位打包友好**：4-bit 权重两个挤一字节，SIMD 移位解包；
3. **精度换结构的阶梯**：Q4_0（最简单）→ Q4_K（二级缩放）→ IQ（imatrix 加权的激进压缩）。

### 1.2 量化类型全家福（位宽构成）

| 格式 | 块结构 | 位宽构成 | bpw |
| --- | --- | --- | --- |
| Q4_0 | 32 权重/块：1 个 FP16 scale + 32×4-bit | $16 + 32\times4 = 144$ bit / 32 | **4.5** |
| Q4_1 | 32 权重/块：1 FP16 scale + 1 FP16 min + 32×4-bit | $16+16+128 = 160$ bit / 32 | 5.0 |
| **Q4_K** | **256 权重/超块：8 子块 × 32**（见 §1.3） | 见下 | **4.5** |
| Q8_0 | 32 权重/块：1 FP16 scale + 32×8-bit | $16 + 256 = 272$ bit / 32 | 8.5 |

**Q4_K 超块手算（面试必考）**：一个超块 256 个权重 = 8 个子块 × 32。存储内容：

- 权重体：$256 \times 4\text{-bit} = 128$ 字节；
- 子块 scale 与 min：8 个子块各一个 6-bit scale + 6-bit min = $8 \times 12 = 96$ bit = 12 字节；
- 超块级：1 个 FP16 总 scale + 1 个 FP16 总 min = 4 字节。

$$
\text{合计} = 128 + 12 + 4 = 144 \text{ 字节} / 256 \text{ 权重} = \frac{144 \times 8}{256} = 4.5 \text{ bpw}
$$

**K-quants 比 Q4_0 精度好的原因**：两级缩放（超块 FP16 scale × 子块 6-bit scale）让每 32 个权重拥有独立的有效步长——等价于"更细的 per-group"，离群子块被隔离，而平均位宽仍是 4.5。

**IQ 系列（imatrix 量化）**：在 2-3 bit 下，均匀网格必然崩。IQ2/IQ3 用**码本 + 超级块**：把权重分组后映射到预训练的离散码点集，并用 imatrix 加权选择最优码点组合——本质是 Stage 2 讲过的"非均匀码点"路线在 CPU 上的实现。

### 1.3 imatrix：用校准数据给每个通道发"重要性权重"

$$
\text{imatrix}_j = \sum_{\text{calib}} \bar{x}_j^2 \quad (\text{第 } j \text{ 个输入通道的激活平方均值})
$$

量化某层时，误差目标从"逐元素均等"变成 **imatrix 加权**：重要通道（激活幅度大、对输出影响大）的量化误差被优先压低，不重要通道允许更粗。效果：2-3 bit 的 IQ 格式从"不可用"变"勉强可用"。**不做 imatrix 直接量化低比特 = 精度悬崖**——这是第 2 周命令链里 `--imatrix` 步骤存在的原因。

### 1.4 GGUF 二进制布局

```
┌─ Magic "GGUF" + 版本号
├─ KV 元数据区：N 个 (key, type, value)
│     如 tokenizer.ggml.tokens、llama.context_length、general.architecture
├─ Tensor 目录：M 个 tensor 的描述
│     (name, dims, type[Q4_K…], offset) —— 只有描述，没有数据
└─ 数据区：按目录 offset 排布的张量原始字节（按对齐填充）
```

要点：① 元数据自包含（分词器、配置都在文件里）→ 单文件分发；② 目录与数据分离 → 可先读目录再按需 mmap 数据（冷启动快的原因）。

### 1.5 llama.cpp 三层架构调用链

```
应用层  llama-cli / llama-server        ← 采样循环、HTTP API
模型层  llama.h (llama_model/context)   ← 层堆叠、KV cache、分词
张量层  ggml + ggml-backend             ← 算子、量化内核、多后端调度
```

一次生成的调用链：`llama_decode()` → 构建 ggml 计算图（embedding → attn → FFN × N 层）→ `ggml_backend_graph_compute` 分派到 CPU/CUDA/Vulkan 后端 → 采样器从末层 logits 取 token。

---

## 2. 实现与验证（本周交付核心）

### 2.1 gguf-py 解析脚本（完整可运行）

```python
# inspect_gguf.py —— 用法：python inspect_gguf.py model.gguf
import sys
from gguf import GGUFReader

path = sys.argv[1]
r = GGUFReader(path, "r")

print(f"版本: {r.version} | tensors: {len(r.tensors)}")
print("---- KV 元数据（前 15 项）----")
for f in list(r.fields.values())[:15]:
    v = f.parts[f.data[0]] if len(f.data) == 1 else f.parts[f.data[0]:][:4]
    print(f"  {f.name} = {bytes(v).decode(errors='ignore')[:60] if v.dtype == 'B' else v}")

print("---- tensor 目录（按量化类型汇总）----")
stats = {}
for t in r.tensors:
    stats.setdefault(str(t.tensor_type), []).append(t.name)
for k, names in sorted(stats.items()):
    print(f"  {k:>10s}: {len(names)} 个，如 {names[0]}")

# ---- 断言：架构与词表元数据必须存在 ----
assert "general.architecture" in r.fields, "缺少架构元数据"
assert any("tokens" in n for n in r.fields), "缺少分词器元数据"
print("GGUF 结构自检通过 ✓（单文件自包含：配置+分词器+权重）")
```

```bash
pip install gguf
python inspect_gguf.py Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
# 预期：打印版本号、元数据项、各量化类型的张量计数
```

### 2.2 手写 dequantize_row_q4_0（与官方对齐）

```python
# q4_0_dequant.py —— Q4_0：每 32 权重一块 = [FP16 scale][16 字节 nibble 对]
import numpy as np, struct

def dequantize_row_q4_0(data: bytes, n: int) -> np.ndarray:
    """data: 连续块字节流；n: 权重个数（32 的倍数）。返回 float32 数组。"""
    out = np.empty(n, dtype=np.float32)
    BLK = 18                       # 2(scale) + 16(nibbles) 字节/块
    for b in range(n // 32):
        blk = data[b * BLK: (b + 1) * BLK]
        scale = np.frombuffer(blk[:2], dtype=np.float16).astype(np.float32)[0]
        qs = np.frombuffer(blk[2:], dtype=np.uint8)      # 16 字节
        lo = (qs & 0x0F).astype(np.int8) - 8             # 低 4 位，偏移 -8
        hi = (qs >> 4).astype(np.int8) - 8               # 高 4 位
        w = np.empty(32, dtype=np.float32)
        w[0::2], w[1::2] = lo, hi                        # 交错布局
        out[b*32:(b+1)*32] = w * scale
    return out

# ---- 自构造数据验证：量化→解量化闭环 ----
np.random.seed(0)
w = np.random.randn(64).astype(np.float32)               # 2 块
def quantize_row_q4_0(w):                                # 官方同款公式
    blocks = []
    for b in range(0, len(w), 32):
        chunk = w[b:b+32]
        amax = np.abs(chunk).max(); scale = amax / -8 if amax > 0 else 1.0
        q = np.clip(np.round(chunk / scale) + 8, 0, 15).astype(np.uint8)
        packed = (q[0::2] & 0x0F) | ((q[1::2] & 0x0F) << 4)
        blocks.append(np.frombuffer(np.float16(scale).tobytes(), np.uint8).tobytes()
                      + packed.tobytes())
    return b"".join(blocks)

raw = quantize_row_q4_0(w)
w_hat = dequantize_row_q4_0(raw, 64)
err = np.abs(w_hat - w).max()
assert err <= np.abs(w).max() / 8 + 1e-6, "解量化误差超过半格"
print(f"Q4_0 闭环验证 ✓（最大误差 {err:.4f}，半格上限 {np.abs(w).max()/8:.4f}）")
# 与官方对齐：llama.cpp 的 ggml-quants.c 中 dequantize_row_q4_0 逻辑同构，
# 可用 GGUF 文件中的真实张量字节喂给本函数，与 ggml 加载结果逐元素对比。
```

**预期**：闭环误差 ≤ 半个量化格（$\text{scale}/2$）。进阶交付：从真实 GGUF 里抽一个张量的原始字节，本函数输出与 `llama.cpp` 加载后的权重对比，误差为 0。

### 2.3 Q4_K 超块结构图（本周交付图）

```
Q4_K 超块（144 字节 = 4.5 bpw）
├─ d     : FP16 超块总 scale        ┐
├─ dmin  : FP16 超块总 min          ┤ 4 字节
├─ scales: 8 子块 × (6-bit scale + 6-bit min) = 12 字节
└─ qs    : 256 个 4-bit 权重 = 128 字节
      └─ 子块 i 的有效值 = d × scale_i × q − dmin × min_i
```

---

## 3. 工程权衡与失效模式

### 3.1 权衡

- **格式阶梯的选择**：Q4_K_M 是通用甜点（精度/体积平衡）；内存紧选 Q3_K_S/IQ3；追求质量选 Q5_K_M 或 Q6_K——命名里 M = medium（混合：重要层用更高位宽）。
- **mmap 与加载速度**：GGUF 靠 mmap 秒级"加载"（按需缺页）；但首次遍历权重会触发真实读盘——长上下文服务启动后先预热一轮。
- **单文件自包含的代价**：分词器/配置锁死在文件里，换分词器要重新转换。

### 3.2 失效模式

1. **低比特不做 imatrix**：症状——IQ2/IQ3 模型输出乱码；根因——无重要性加权的 2-bit 网格无法分配误差预算；修复——转换链必须带 `--imatrix`（第 2 周）。
2. **格式与内存预算错配**：症状——板端 OOM；根因——按"参数量×4.5/8"估内存时忘了 KV cache 与运行时开销；修复——体积 + ~20% 余量 + KV 预算单独算。
3. **GGUF 版本不匹配**：症状——新版 llama.cpp 读旧文件报警告、或反之直接拒绝；根因——格式版本演进；修复——转换工具与运行时同版本更新。
4. **张量对齐踩坑**：症状——自己写解析器读出错位；根因——数据区有对齐填充（默认 32 字节），目录里的 offset 才是真相；修复——永远信目录 offset，不信累加。

---

## 4. 延伸思考题（含解析）

**Q1**：Q4_0 和 Q4_K 都是 4.5 bpw，为什么后者精度好？
**A**：Q4_0 每 32 权重只有一个 FP16 scale；Q4_K 用"超块 FP16 × 子块 6-bit"两级 scale，等效每 32 权重一个独立步长（且总开销不变，因为子块 scale 只用 6 bit）——同样的位宽，更细的粒度，离群子块被隔离。

**Q2**：GGUF 把分词器塞进权重文件，是优点还是缺点？
**A**：优点是单文件分发、绝不出现"权重与分词器版本错配"这类部署事故；缺点是文件更大、换分词器要重转。对端侧分发场景，防错配的价值远大于体积代价。

**Q3**：imatrix 与 Stage 2 的 AWQ 在思想上是什么关系？
**A**：同源：都用"激活幅度统计"识别重要通道。AWQ 用它决定**缩放保护谁**；imatrix 用它决定**误差预算偏向谁**（加权量化目标）。一个是防御（保护），一个是进攻（分配）。

**Q4**：llama.cpp 为什么能在 x86 和 ARM 上都打得动？
**A**：① 量化格式块状自包含，天然适配两种 ISA 的 SIMD；② 内核按 AVX2/AVX-512/NEON 分别手写特化；③ 推理主力是 GEMV（访存瓶颈），吃满各自的内存带宽即可，不依赖特殊硬件单元。第 2 周展开内核细节。

**Q5**：为什么 tensor 目录与数据区分离对端侧特别重要？
**A**：端侧启动时间敏感：目录极小可瞬间读完（知道有什么、在哪），数据区用 mmap 按需缺页——用户看到"首字"之前，只需真正读进当前层所需的那部分权重。冷启动从"读全文件"变"读热路径"。

---

## 本周交付清单

- [ ] 闭卷手算 Q4_K 超块：144 字节 / 256 权重 = 4.5 bpw（逐项列出）。
- [ ] 跑通 `inspect_gguf.py`，截图元数据与量化类型汇总。
- [ ] 手写 `dequantize_row_q4_0` 闭环断言通过；进阶：与真实 GGUF 张量对齐。
- [ ] 画 Q4_K 超块结构图（§2.3 能闭卷重画）。
- [ ] 编译 llama.cpp（纯 CPU）并跑通一次 `llama-cli` 生成，记录 tokens/s。
