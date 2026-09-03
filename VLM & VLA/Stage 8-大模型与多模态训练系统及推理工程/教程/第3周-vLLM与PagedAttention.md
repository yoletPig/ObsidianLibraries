# 第 3 周教程：高吞吐推理引擎核心——vLLM 与 PagedAttention

> **本周要回答的三个问题**
> 1. 传统 KV Cache 管理的显存浪费（碎片/过度预分配）到底有多少？从哪来？
> 2. PagedAttention 的分页机制怎么运作？Continuous Batching 与 Static Batching 差在哪？
> 3. MHA/MQA/GQA 对推理显存与吞吐意味着什么？
> 4. （实战）QPS / TTFT / ITL 三个指标怎么测？

对应学习计划：第 3 周。交付物：部署 vLLM 推理服务（Qwen2.5-VL/LLaVA），asyncio 压测 50 并发，对比 HF `generate()` 与 vLLM 的 QPS、TTFT、ITL。

**论文**：vLLM（*Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023）——学习计划所引无误。Stage 3 第 2 周（评测加速）与本篇的关系：那里是"用"，这里是"懂"。

---

## 1. 第一性原理：KV Cache 是推理的显存主角

### 1.1 根本矛盾：KV Cache 的需求是动态的，显存分配是静态的

自回归生成时，已生成 Token 的 K/V 必须缓存（否则每步重算全前缀）。公式（第 1 周 1.3 节）：单序列 KV = $2 \times L \times n_{kv} \times d_{head} \times S \times b$。7B（GQA-8）8k 上下文约 2GB——**一个并发请求就吃掉 GB 级**，服务吞吐的上限直接由"能同时装下多少条 KV"决定。

传统实现（HF 风格）为每个请求**按 max_length 预分配一块连续显存**存 KV。论文实测口径（vLLM SOSP 2023，实验条件为其测试的模型与负载）：这种管理方式下**实际有效利用率常仅 20~40%**，浪费来自三处：

1. **内部碎片**：预留 max_length，实际生成 300 Token——预留的 8000 位置大部分闲置；
2. **预留浪费**：为"可能的最大长度"过度保守预留；
3. **外部碎片**：连续分配的要求让显存被切成碎片，即使总量够也找不到连续块 → 新请求被拒绝（论文里称之为 memory waste 与 premature preemption 的来源）。

**这解释了 Stage 3/4 阶段"vLLM 快 3~8 倍"的深层原因**：不只 batching 更聪明，而是同样显存能装下数倍的并发序列（KV 利用率从 ~30% 提到 ~90%+）。

### 1.2 PagedAttention：把操作系统的虚拟内存搬进 KV Cache

vLLM 的方案逐字借鉴 OS 分页：

| OS 虚拟内存 | PagedAttention |
| --- | --- |
| 虚拟地址空间 | 每请求的逻辑 KV 序列 |
| 物理页框（固定 4KB） | 物理 KV Block（固定 16 个 Token/块） |
| 页表（虚拟→物理映射） | Block Table（每请求一张逻辑块→物理块表） |
| 按需分页 | KV 按块惰性分配（生成一个块才占一个物理块） |
| 共享内存页 | **多请求共享相同前缀的物理块**（Copy-on-Write 语义，第 4 周 RadixAttention 的先声） |

收益链条：**碎片消除**（块内浪费 <4%，块粒度 16 Token）→ 同显存可容纳并发数 ×4 以上（论文口径）→ batch 变大 → 吞吐提升。外加两个推论能力：**前缀共享**（同 System Prompt 的请求复用物理块）与**按需抢占/换出**（显存紧张时把低优先级序列的块换出到 CPU）。

### 1.3 Continuous Batching：调度粒度从"批"到"迭代"

Static Batching（HF generate 传统形态）：凑一批请求 → 全部一起生成 → **最长的那条结束时整批才释放**——短请求陪跑几十倍的空转（Stage 2 评测加速篇已给过数值）。

Continuous Batching（vLLM/SGLang 的默认）：**每次迭代（decode 一步）层面动态调度**——某序列生成完 EOS，立即退出；排队的新请求立即补进 batch；prefill 与 decode 混合调度（第 4 周 Chunked Prefill 展开）。GPU 的 decode 永远满载。

两个引擎机制的关系：**PagedAttention 解决"装得下"（显存利用率），Continuous Batching 解决"转得快"（调度利用率）**——两者叠加才有 vLLM 的量级优势。

### 1.4 MHA / MQA / GQA：注意力结构对 KV 的影响

| 结构 | KV 头数 | KV Cache（相对 MHA） | 质量 |
| --- | --- | --- | --- |
| MHA（Multi-Head） | = 查询头数（如 32） | 1×（基准） | 最好 |
| MQA（Multi-Query） | 1 | **1/32** | 略降 |
| **GQA**（Grouped-Query） | 分组（如 8） | **1/4** | 接近 MHA |

第 1 周思考题 3 的答案展开：7B 模型 MHA 下 32k 上下文 batch 16 的 KV = 2×32×32×128×32768×16×2B ≈ **171GB**；GQA（8 头）= **43GB**——差 4 倍。GQA 用 1/4 的显存拿到接近 MHA 的质量，是"KV Cache 是推理瓶颈"这一论断在**模型结构侧**的回应（PagedAttention 是系统侧的回应）。现代模型（Qwen2/2.5、Llama-3）几乎清一色 GQA。

---

## 2. 实现与验证

### 2.1 部署与压测

```bash
# 部署 VLM 推理服务 (OpenAI 协议)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64
```

```python
"""
asyncio 压测: vLLM 服务 vs HF generate 的 QPS/TTFT/ITL 对比。
运行方式: python stage8_week3_bench.py --backend vllm --url http://localhost:8000/v1 \
            --concurrency 50 --n 50 --mode chat
          python stage8_week3_bench.py --backend hf --model Qwen/Qwen2.5-VL-7B-Instruct ...
依赖: openai, aiohttp, asyncio, time, statistics
"""
import argparse
import asyncio
import time
import statistics as st
from openai import AsyncOpenAI

PROMPT = ("这张图片里的主要物体是什么？请用三句话详细描述画面内容、"
          "颜色构成，以及它们可能的使用场景。")


async def one_request(client, sem, img_url, max_tokens, stats):
    async with sem:
        t0 = time.perf_counter()
        first = None
        stream = await client.chat.completions.create(
            model="served-model",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "text", "text": PROMPT}]}],
            max_tokens=max_tokens, temperature=0.0, stream=True)
        n = 0
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if first is None:
                    first = time.perf_counter() - t0        # TTFT
                n += 1
        total = time.perf_counter() - t0
        itl = (total - first) / max(1, n - 1)               # ITL (流式 decode 间隔)
        stats.append({"ttft": first, "itl": itl, "total": total})


async def bench(args):
    client = AsyncOpenAI(base_url=args.url, api_key="EMPTY", timeout=300)
    sem = asyncio.Semaphore(args.concurrency)
    stats = []
    img = "https://placehold.co/448x448.png"                # 实测换成真实图片 URL/base64
    t0 = time.perf_counter()
    await asyncio.gather(*[one_request(client, sem, img, 256, stats) for _ in range(args.n)])
    wall = time.perf_counter() - t0
    qps = args.n / wall
    ttfts = [s["ttft"] for s in stats]
    itls = [s["itl"] for s in stats]
    print(f"[vLLM] QPS={qps:.1f}  TTFT p50={st.median(ttfts)*1000:.0f}ms "
          f"p95={sorted(ttfts)[int(.95*len(ttfts))]*1000:.0f}ms  "
          f"ITL p50={st.median(itls)*1000:.1f}ms")
    return qps, ttfts, itls


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.set_defaults(url="http://localhost:8000/v1", concurrency=50, n=50)
    ap.add_argument("--url"); ap.add_argument("--concurrency", type=int); ap.add_argument("--n", type=int)
    asyncio.run(bench(ap.parse_args()))
```

HF 对照组（同 prompt 集，串行/静态批）：记录相同三指标。**预期形态**（7B、50 并发、256 输出 Token、A100 量级，务必以实测为准）：vLLM QPS 常为 HF generate 的 **5~20 倍**；TTFT 相近或略优；差距主要在吞吐（Continuous Batching 吃满 decode）。压测纪律：预热 5 条再计数；三指标分开报告（**TTFT 对交互体验、ITL 对流式体验、QPS 对成本**——一个数字讲不清推理性能）。

---

## 3. 工程权衡与失效模式

### 3.1 决策表：vLLM 服务的关键旋钮

| 参数 | 起点 | 权衡 |
| --- | --- | --- |
| `gpu_memory_utilization` | 0.85~0.92 | 高 → KV 多、并发强；留太少给视觉编码器/激活会 OOM（VLM 场景注意！） |
| `max_num_seqs` | 64~256 | 并发上限；过高加剧调度与 cache 压力 |
| `max_model_len` | 按业务 | 直接决定 KV 预算（8k → 数万块） |
| block_size | 16（默认） | 大块省块表、小块减碎片，默认即可 |

### 3.2 三个代表性失效模式

**失效 1：VLM 的视觉编码显存没算进预算**
- **症状**：`gpu_memory_utilization=0.9` 下启动即 OOM，或高并发下周期性 OOM。
- **根因**：vLLM 的预算给 KV+权重，VLM 的视觉编码器（ViT）激活与图像 embedding 在峰值时另需数 GB——预算把它们挤掉了。
- **定位**：OOM 栈在 vision tower / multimodal projector 附近。
- **修复**：降 `gpu_memory_utilization`（0.85→0.75 试起）；限制输入图像分辨率；`max_num_batched_tokens` 调低（vLLM 对多模态的 token 预算参数）。

**失效 2：长请求被抢占/重算——尾部延迟尖刺**
- **症状**：P99 TTFT/总延迟远高于 P50（数倍），日志出现 preemption/swap。
- **根因**：显存被长序列占满后，新长请求被抢占（KV 换出），恢复时全量重算——Continuous Batching 的公平性代价。
- **定位**：vLLM 日志的 preemption 计数；按请求长度分桶看延迟分布。
- **修复**：限制 `max_model_len`；长短请求分池部署；或接受抢占但监控 SLO 违约率。

**失效 3：把"吞吐数字"跨引擎跨负载乱比**
- **症状**：报告声称"我们引擎比 vLLM 快 3 倍"，复现不出来。
- **根因**：负载形态不同（短问答 vs 长生成、共享前缀比例不同——共享前缀场景 SGLang 的 Radix 占尽便宜，第 4 周）、并发压力不同、指标口径不同（QPS vs tokens/s）。
- **定位**：审计对比双方的 prompt 集与调度参数是否一致。
- **修复**：性能报告四要素——硬件/引擎版本/负载描述/指标口径；跨引擎对比用同一压测脚本。

---

## 4. 延伸思考题

1. **块大小的设计空间**：PagedAttention 默认 16 Token/块。推导块大小对"块内碎片率"（期望浪费 = 块大小/2）与"块表开销"（每块一条表项）的双向影响，并讨论为何 16 是工程甜点（提示：块太小→块表访存与调度开销大；太大→内部碎片回升至接近连续分配）。
2. **Prefix 共享与安全**：多请求共享物理前缀块在多租户服务里有信息泄漏面吗？设计一个攻击场景与防御方案。（提示：CoW 语义下共享块只读，理论上安全；风险在共享判定逻辑与缓存淘汰的边界 case——这是系统安全的经典"共享即攻击面"命题。）
3. **实测 GQA 的价值**：找一个同时提供 MHA 与 GQA 权重的模型对（或用配置改造），实测 32k 上下文下的 KV 显存与吞吐差，验证 1.4 节的 4 倍推算——把第 1 周思考题变成实测报告。

---

*下一篇：[第 4 周：前沿推理系统——SGLang 与 Prefix Caching](第4周-SGLang与PrefixCaching.md)*
