# 第 1 周教程：KV Cache 与 PagedAttention 实现层

> **本周要回答的三个问题**
> 1. PagedAttention 的物理块表（block table）到底管什么？块分配器与 copy-on-write 如何协作？
> 2. 7B 模型、8k 上下文、batch=32 的 KV cache 占多少显存？怎么一步步手算出来？
> 3. Prefix Caching 的哈希键怎么构造？它和 SGLang 的 RadixAttention 差在哪？

对应学习计划：第 1 周。交付物：① 手写迷你版 PagedAttention 块管理器（200 行内），跑模拟请求流画块占用时间线；② 手算 + 实测 Qwen2.5-7B、ctx=8k、FP8 KV 的单卡最大并发。

---

## 1. 第一性原理：KV cache 为什么是显存第一杀手

### 1.1 KV cache 的由来

自回归解码时，每生成一个新 token，注意力都要看所有历史 token。若每步都重算历史，复杂度是 $O(U^2)$。KV cache 把已算过的 key/value 存下来，增量计算，把每步降到 $O(1)$——**用显存换计算**。

问题：KV cache 随序列长度**线性增长**，且在请求结束前不能释放。长上下文时代，它比模型权重还吃显存。

### 1.2 手算 KV cache（本周自测题，必须闭卷）

对单个请求、序列长度 $L$，KV cache 大小：

$$
\text{KV bytes} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times L \times \text{bytes\_per\_elem}
$$

#### 深入拆解：这个公式在干什么

> 这个公式回答一个问题："每多存一个 token 的历史，我要在显存里多花多少字节？"它是后面一切并发/上下文容量估算的地基。

- **为什么需要它**：规划并发与上下文长度时，你必须先知道 KV 的"单价"（每 token 多少字节），才能算出给定显存能同时养多少请求（交付物②就靠它）；
- **系数 2**：注意力里 K 和 V 是两组独立的张量，各存一份；
- **$n_{\text{layers}}$**：每一层都有独立的注意力、都要存自己的 K/V，层数越多 KV 越大；
- **$n_{\text{kv\_heads}} \times d_{\text{head}}$**：K/V 每层的"宽度"。GQA（Grouped Query Attention）下 KV 头数远小于 Q 头数——多个 Q 头共享一组 KV 头，这正是省显存的关键（Q 不进 cache，不用管）；
- **$L$**：序列长度，KV 与它**线性增长**；
- **`bytes_per_elem`**：存储精度，FP16=2 字节、FP8=1 字节；
- **结果拿来做什么**：乘上 batch 与序列长度就是总 KV 显存 → 反推"这块卡最多扛多少并发/多长上下文"。

**手算小例**：假设一个只有 2 层、2 个 KV 头、head_dim=2 的小玩具模型，存 1 个 token 的 FP16 KV：$2 \times 2 \times 2 \times 2 \times 2 = 64$ 字节。真实模型只是把这个乘法换成大数字而已。

**实例：Qwen2.5-7B**（以官方 `config.json` 为准：**28 层、GQA 4 个 KV 头、head_dim=128**，由 hidden_size 3584 / 28 个 Q 头 = 128 得出）：

$$
\text{每 token KV 元素数} = 2 \times 28 \times 4 \times 128 = 28672 \text{ 元素/token}
$$

FP16 下 $28672 \times 2 = 57344$ 字节 $= 56$ KB/token。

**8k 上下文、单请求**：$56 \text{ KB} \times 8192 = 448$ MB。

**batch=32、8k**：$32 \times 448 \text{ MB} = 14$ GB。

对比：7B 权重本身（FP16）约 14 GB。**batch=32、8k 时 KV cache（14 GB）已与权重（14 GB）相当**；上下文再翻倍、或并发再翻倍，KV 就会**超过权重**——这就是"长上下文时代 KV 是显存第一杀手"的定量直觉：权重是固定开销，KV 随 batch × 序列长度线性膨胀。若换 FP8 KV（元素减半），并发上限直接翻倍，决定单卡能扛多少请求。

> **动手验证**：`python -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('Qwen/Qwen2.5-7B-Instruct'); print(c.num_hidden_layers, c.num_key_value_heads, c.hidden_size // c.num_attention_heads)"` 输出 `28 4 128`。**算 KV 必须以目标模型 config 为准**——层数、KV 头数随模型型号差异巨大，不要背某个模型的数字。

> **一句话总结**：KV 单价 = 2（K、V）× 层数 × KV 头数 × 头维度 × 字节数；Qwen2.5-7B FP16 下每 token 约 56 KB，8k 上下文单请求 448 MB，batch=32 时 KV 已追平权重——并发与上下文任一翻倍，KV 就反超权重成为显存大头。

---

## 2. PagedAttention 实现层

### 2.1 问题：预分配的浪费

朴素做法：每个请求预留**整个最大上下文**的连续显存。若请求只用了 200/8192，剩下 97.6% 浪费。高并发下这种"内部碎片"直接拖垮吞吐。

### 2.2 PagedAttention：把 KV cache 分页

借鉴操作系统虚拟内存：

- 把 KV cache 切成固定大小的**块（block）**（如 16 token/块）；
- 每请求按需分配块，**不连续**也行；
- 用**块表（block table）**记录"逻辑块 → 物理块"映射，类似页表。

$$
\text{逻辑 token 位置} \xrightarrow{\text{block table}} \text{物理显存块地址}
$$

注意力计算时通过块表查物理地址。碎片从"整段预留"降到"最多一个块的尾部浪费"，显存利用率逼近 100%。

### 2.3 块分配器与引用计数

块分配器维护一个空闲块池。分配时取一块、引用计数 +1；释放时 -1，归零则回收。

**copy-on-write（写时复制）**：并行采样（同一 prompt 出多个回复）或 beam search 时，多个序列**共享**同一段 KV 块——只复制块表指针，不复制物理数据；只有当某个分支要写入新内容时才真正复制那块。这把并行采样的显存开销从 $O(k)$ 降到接近 $O(1)$。

### 2.4 手写迷你块管理器（交付核心）

```python
from dataclasses import dataclass, field
from typing import Dict, List, Set

BLOCK_TOKENS = 16      # 每块容纳的 token 数

@dataclass
class Block:
    block_id: int
    ref_count: int = 0

class BlockAllocator:
    """模拟 PagedAttention 的物理块池 + 引用计数。"""
    def __init__(self, num_blocks: int):
        self.free: Set[int] = set(range(num_blocks))
        self.blocks: Dict[int, Block] = {
            i: Block(i) for i in range(num_blocks)}

    def allocate(self) -> Block:
        if not self.free:
            raise MemoryError("KV cache 块耗尽 -> 实际即 OOM/排队")
        bid = self.free.pop()
        b = self.blocks[bid]
        b.ref_count = 1
        return b

    def free_block(self, b: Block):
        b.ref_count -= 1
        if b.ref_count == 0:
            self.free.add(b.block_id)

    def fork(self, b: Block) -> Block:
        """写时复制：共享引用，不复制物理数据。"""
        b.ref_count += 1
        return b

class SeqBlockManager:
    """每个请求维护一个逻辑块表（逻辑块 -> 物理块）。"""
    def __init__(self, allocator: BlockAllocator):
        self.alloc = allocator
        self.table: List[Block] = []     # block table
        self.num_tokens = 0

    def append_token(self):
        """新增一个 token，必要时分配新块。"""
        self.num_tokens += 1
        needed = (self.num_tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS
        while len(self.table) < needed:
            self.table.append(self.alloc.allocate())

    def release(self):
        for b in self.table:
            self.alloc.free_block(b)
        self.table = []

# ---- 验证 ----
alloc = BlockAllocator(num_blocks=8)        # 8 块 × 16 token = 128 token 容量
req = SeqBlockManager(alloc)

for _ in range(20):                          # 写入 20 个 token
    req.append_token()
assert len(req.table) == 2                   # 20 token -> ceil(20/16)=2 块
assert len(alloc.free) == 6                  # 8-2=6 空闲
print(f"20 token 用 {len(req.table)} 块，空闲 {len(alloc.free)} 块 ✓")

# 写时复制：并行采样共享
forked = SeqBlockManager(alloc)
forked.table = [alloc.fork(b) for b in req.table]
assert req.table[0].ref_count == 2           # 共享，引用计数 +1
assert len(alloc.free) == 6                  # 未新分配物理块
print("写时复制共享生效，无额外物理块分配 ✓")

req.release()
assert req.table[0].ref_count == 1           # fork 仍持有
print("释放后引用计数正确 ✓")
```

**预期**：20 token 恰好占 2 块；fork 后引用计数为 2 且不分配新块。这就是 PagedAttention 块管理的最小可验证实现。

### 2.5 块占用时间线可视化

跑一段模拟请求流（请求到达→生成→结束），记录每时刻空闲块数，画时间线：

```python
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(0)
alloc = BlockAllocator(num_blocks=64)
timeline = []
active = {}                                  # req_id -> manager
for step in range(200):
    # 随机到达新请求
    if random.random() < 0.3 and len(active) < 8:
        rid = step
        active[rid] = SeqBlockManager(alloc)
        active[rid].append_token()
    # 已有请求继续生成
    done = []
    for rid, m in active.items():
        if len(m.table) >= 6 or random.random() < 0.1:
            done.append(rid)
        else:
            m.append_token()
    for rid in done:
        active[rid].release(); del active[rid]
    timeline.append(len(alloc.free))

plt.figure(figsize=(12, 4))
plt.plot(timeline)
plt.xlabel("时间步"); plt.ylabel("空闲块数")
plt.title("KV cache 块占用时间线（波谷 = 高并发峰值）")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("kv_block_timeline.png", dpi=110)
print("已保存 kv_block_timeline.png")
```

观察波谷：空闲块最少的时刻就是并发峰值、最接近触发排队/抢占的时刻。

---

## 3. Prefix Caching 与 RadixAttention

### 3.1 Prefix Caching（vLLM）

多个请求共享同一前缀（如系统提示词）。vLLM 用**块内容哈希**做键：

$$
\text{hash}(\text{block}) = H(\text{父块 hash}, \text{本块 token ids})
$$

新请求的前缀块若能命中已有哈希，就直接复用物理块（引用计数 +1），跳过重复的 prefill 计算。逐出用 LRU。

> **版本注意**：V1 引擎（约 v0.8.x 起）中 prefix caching（官方叫 Automatic Prefix Caching，APC）**默认开启**——不再需要 `--enable-prefix-caching`；要关掉用 `--no-enable-prefix-caching`。部分模型（如 Mamba/混合架构）不支持，默认仍关闭。

### 3.2 RadixAttention（SGLang）

SGLang 用 **Radix Tree（基数树）** 组织所有缓存的前缀：

- 树节点 = 一段 token 前缀，边 = token；
- 共享前缀天然在树的公共路径上；
- **cache-aware 调度**：把共享前缀的请求排在一起，最大化命中。

**区别**：vLLM 的 prefix caching 是"块级哈希命中"，被动复用；SGLang 的 RadixAttention 是"树结构主动管理 + 调度感知"，对多轮对话/共享前缀场景命中率更高。这是两个引擎选型的重要差异点（第 2 周压测会体现）。

---

## 4. KV 量化与压缩

### 4.1 为什么压 KV

第 1.2 节算过，长上下文下 KV 会逼近甚至反超权重。压缩 KV 直接放大并发：

- **FP8 KV**：元素 2 字节 → 1 字节，KV 减半，精度损失小（KV 数值范围比权重窄）；
- **KIVI**：Key 用 2-bit、Value 用 2-bit 的极端量化（Key 沿通道量化、Value 沿时间量化）；
- **Sliding Window + Sink（StreamingLLM）**：只保留最近窗口 + 开头几个"attention sink" token，支持无限长流式。

### 4.2 在引擎中的位置

这些都作用在"写 KV / 读 KV"的算子上（`attention/backends/`），对调度器透明。开启方式通常是启动参数（如 `--kv-cache-dtype fp8`）。

---

## 5. 工程权衡与失效模式

### 5.1 权衡

- **块大小**：小块碎片少但块表开销大；大块反之。16 token 是常见折中；
- **FP8 KV**：并发翻倍但精度略降，需验证目标任务；
- **prefix caching**：命中省算力，但维护哈希/树有开销，前缀高度重复时才划算。

### 5.2 失效模式

1. **块耗尽（OOM-like）**：并发高 + 长序列，空闲块归零 → 新请求排队或旧请求被抢占。定位：看块占用时间线波谷；修复：降 `max-num-seqs`、开 KV 量化、扩容。
2. **内部碎片误判**：以为显存够却排队——其实是块粒度导致的尾部浪费叠加。修复：理解块大小，必要时调整。
3. **prefix caching 命中率低**：请求前缀各不相同，缓存形同虚设还添开销。修复：评估前缀重复率再开。
4. **引用计数泄漏**：fork 后未正确释放 → 块永不回收，显存缓慢泄漏。定位：长跑监控空闲块单调下降。

---

## 6. 延伸思考题（含解析）

**Q1**：手算 7B、8k、batch=32 的 KV 显存，并说明它如何逼近权重。
**A**：以 Qwen2.5-7B 官方 config（28 层、4 KV 头、head_dim=128）为例：每 token KV = 2×28×4×128 = 28672 元素，FP16 约 56 KB；8k 上下文单请求 448 MB，×32 = 约 14 GB，与 7B 权重（约 14 GB）相当。KV 随序列长度与并发线性增长，权重固定，故上下文或并发再翻倍，KV 即反超权重成为显存大头。

**Q2**：PagedAttention 相比预分配，显存利用率为什么能逼近 100%？
**A**：预分配按最大上下文整段预留，短请求浪费巨大（内部碎片）；PagedAttention 按需分页，浪费最多是最后一块的尾部（<16 token），碎片从"整段"降到"一块尾部"。

**Q3**：copy-on-write 在并行采样中省了什么？
**A**：多个分支共享同一段 KV 物理块，只复制块表指针、引用计数 +1，不复制物理数据；直到某分支要写新内容才真正复制那一块。把并行采样的额外显存从 O(k) 降到接近 O(1)。

**Q4**：prefix caching 的哈希键怎么构造？为什么含父块哈希？
**A**：hash = H(父块哈希, 本块 token)。含父块哈希保证"相同后缀但不同前缀"不会误命中——只有从头开始的完整前缀一致才算命中。

**Q5**：vLLM prefix caching 与 SGLang RadixAttention 的本质区别？
**A**：vLLM 是块级哈希的被动复用；SGLang 用基数树主动管理所有前缀，且调度器 cache-aware 地把共享前缀请求排一起，命中率与多轮对话场景更优。

---

## 本周交付清单

- [ ] 跑通迷你块管理器，验证分配/写时复制/引用计数断言。
- [ ] 画块占用时间线，标注并发峰值波谷。
- [ ] 闭卷手算 7B/8k/batch=32 的 KV 显存（Qwen2.5-7B 约 14 GB），与权重对比。
- [ ] 用 `AutoConfig` 验证你所用模型的层数/KV 头数/头维度，确认手算输入无误。
- [ ] 云上实测 `--max-num-seqs` 与手算并发对照（误差 < 20%）。
