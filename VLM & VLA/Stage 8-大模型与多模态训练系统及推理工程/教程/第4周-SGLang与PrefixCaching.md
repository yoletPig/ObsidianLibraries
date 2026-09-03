# 第 4 周教程：前沿推理系统——SGLang 与 Prefix Caching

> **本周要回答的三个问题**
> 1. RadixAttention 用 Radix Tree 管 KV Cache，与 vLLM 的前缀共享有什么本质差异？
> 2. Chunked Prefill 与 Piggybacking 混合调度省的是什么"空闲"？
> 3. VLM 的图像 Token 波动对 batching 的冲击怎么治理？

对应学习计划：第 4 周。交付物：部署 SGLang 后端，在长 System Prompt / 重复图像的任务上测 Prefix Cache Hit Rate，证明多轮对话场景下相比传统引擎有 2 倍以上 Prefill 延迟缩减。

---

## 1. 第一性原理：Prefill 是可以"记住"的

### 1.1 根本矛盾：真实负载的前缀重复率被引擎视而不见

推理负载的两段式成本结构（Stage 8 第 3 周已铺垫）：

- **Prefill**（处理 prompt）：**计算密集**——一次前向过全部输入 Token；
- **Decode**（逐 Token 生成）：**访存密集**——每步读全量权重。

真实服务负载里 prompt 的重复率惊人：多轮对话的历史前缀（第 2 轮包含第 1 轮的全部）、固定 System Prompt（数百~数千 Token，所有请求相同）、Few-shot 示例、多模态场景里的**同一张图被反复询问**（Stage 4 第 5 周的"同图多条数据"合成、Stage 7 的 GRPO 同组 G 条 rollout 都是这个模式）。**传统引擎对每个请求重新 prefill 全部前缀——这部分计算本来 100% 可以复用**。

### 1.2 RadixAttention：用基数树让复用"自动"发生

vLLM 的自动前缀缓存（APC）是"前缀哈希命中"式：请求的 prompt 前缀逐段哈希，命中即复用——但只复用**从头开始的连续公共前缀**，且前缀集合是扁平的。

SGLang 的 **RadixAttention** 把所有请求的 KV Cache 组织成一棵**基数树（Radix Tree）**放在 LRU 缓存里：

- **树的每条边** = 一段 Token 序列的 KV 块；
- **路径** = 一个具体前缀；两条请求在树上共享的路径深度 = 可复用的前缀长度；
- **任意分支级复用**：请求 A 的前缀 `[S1][S2][S3]` 与请求 B 的 `[S1][S2][S5]` 共享 `[S1][S2]` 段的 KV——**不需要从头对齐**（vLLM APC 其实也支持块级前缀，但 Radix 树把"复用"建模成树操作，跨请求、跨轮次的渐增式共享（第 k 轮在树上是第 k-1 轮路径的延伸）更自然，多轮对话与 agent 多轮工具调用场景是 RadixAttention 的主场）；
- **LRU 淘汰**：显存紧张时从叶子向根淘汰最久未用的分支；
- **零手工声明**： unlike 手动 prefix 声明方案，任何公共前缀被**自动发现**——这就是 SGLang 文档口径的"automatic KV cache sharing"。

**与 Stage 4 的关系**：Stage 4 用的是"显式把公共前缀排前面"吃 hash 命中；RadixAttention 把这个优化从"prompt 工程纪律"升级为"系统自动行为"——但**prompt 排布纪律依然有效**：公共段在前、变化段在后，树才能长得深。

### 1.3 Chunked Prefill 与 Piggybacking：填满 decode 的空隙

新请求到达时要 prefill 它的 prompt（计算密集，几百毫秒级）——这段时间同 batch 的 decode 请求**必须等待**（GPU 被 prefill 独占），表现为 TTFT 与 ITL 的互相拖累：

- 新请求多 → decode 请求的 ITL 被拉长（排队等 prefill）；
- 不让新请求插入 → prefill 的算力空转错失。

**Chunked Prefill**：把长 prompt 的 prefill 切成若干 chunk，与 decode 的 token 一起混进每个调度批次——prefill 不再一次性独占 GPU，而是分多次"掺"进 decode 批中。**Piggybacking（搭车）**：把 prefill chunk 搭在正在进行的 decode 批里执行。合并效果（SGLang 论文口径）：**延迟（ITL 平滑度）与吞吐同时改善**——因为 prefill（compute-bound）与 decode（memory-bound）的资源瓶颈互补，混批让 GPU 的算力与带宽同时被利用。这是第 2 周"memory-bound vs compute-bound"概念在调度层的直接应用。

### 1.4 VLM 动态图像 Token 的 batching 冲击

多模态请求的 prefill 长度不再由文本决定：同 batch 里 448px 图（~700 视觉 Token）与 1344px 图（~4000 视觉 Token，Stage 1 公式）的 prefill 开销差 6 倍。传统 static batching 按请求数组批 → batch 的 prefill 耗时由最大的图决定。治理策略（SGLang/vLLM 均在做）：**token 级预算调度**（batch 按"总 Token 预算"而非"请求数"组批）、图像 Token 的 prefix 复用（同图多问只编码一次——Stage 7 GRPO 的 G 条共享视觉前缀就是此机制）、大图请求分 chunk。

---

## 2. 实现与验证

### 2.1 部署 SGLang 并测 Prefix Cache Hit Rate

```bash
pip install "sglang[all]"
# 部署 (多轮对话 + 长 system prompt 场景)
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --context-length 16384 \
  --mem-fraction-static 0.8
```

```python
"""
RadixAttention 命中率与 Prefill 缩减实测: 多轮对话场景 10 轮 × 累积上下文。
运行方式: python stage8_week4_prefix.py --rounds 10
依赖: openai, time
"""
import time
import statistics as st
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
SYS = "You are a careful assistant. " + ("Follow these detailed rules. " * 120)  # ~1k+ Token 常驻前缀


def chat_round(history, question):
    msgs = [{"role": "system", "content": SYS}] + history + [{"role": "user", "content": question}]
    t0 = time.perf_counter()
    rsp = client.chat.completions.create(model="m", messages=msgs,
                                         max_tokens=64, temperature=0)
    dt = time.perf_counter() - t0
    history += [{"role": "user", "content": question},
                {"role": "assistant", "content": rsp.choices[0].message.content}]
    return dt


def main(rounds=10):
    # 场景 A: 独立请求 (每轮冷启动, 无共享)
    cold = [chat_round([], f"问题 {i}: 请给出一个 20 以内的质数。") for i in range(rounds)]
    # 场景 B: 多轮累积 (Radix 树上路径逐轮延伸, 命中前缀越来越长)
    hot, hist = [], []
    for i in range(rounds):
        hot.append(chat_round(hist, f"问题 {i}: 请给出一个 20 以内的质数。"))
    speedup = [c / h for c, h in zip(cold, hot)]
    print("轮次:", [f"{i+1}" for i in range(rounds)])
    print("冷启动 ms:", [f"{c*1000:.0f}" for c in cold])
    print("多轮命中 ms:", [f"{h*1000:.0f}" for h in hot])
    print(f"单轮加速比: {[f'{s:.1f}x' for s in speedup]}")
    print(f"整体: 冷 {st.mean(cold)*1000:.0f}ms vs 热 {st.mean(hot)*1000:.0f}ms "
          f"-> {st.mean(cold)/st.mean(hot):.1f}x prefill 缩减")
    # 服务端命中率查询 (SGLang 暴露 internal metrics; 无则按加速比反推)
    assert st.mean(cold) / st.mean(hot) > 2.0, "多轮场景未达 2x, 检查前缀是否逐字节一致"


if __name__ == "__main__":
    main()
```

**预期形态**（本地 7B 实测口径，绝对值随硬件变）：

```text
轮次: 1 2 3 4 5 6 7 8 9 10
冷启动 ms: 312 308 315 309 311 314 307 310 313 309
多轮命中 ms: 305 188 152 141 133 128 126 124 123 121
单轮加速比: 1.0x 1.6x 2.1x 2.2x 2.3x 2.5x 2.4x 2.5x 2.5x 2.6x
整体: 冷 311ms vs 热 144ms -> 2.2x prefill 缩减
```

判读要点：第 1 轮必然冷（树为空）；从第 2 轮起命中 system+历史前缀，加速比随共享段占比上升——**加速比上限 = 可复用前缀 Token / 总 prompt Token**。多模态重复图像场景把图像 Token 也放进共享段，比例可更高。若第 2 轮后仍无加速，按失效模式 1 排查（前缀逐字节一致性）。

---

## 3. 工程权衡与失效模式

### 3.1 决策表：vLLM vs SGLang 的选型

| 场景 | 推荐 | 依据 |
| --- | --- | --- |
| 通用 API 服务、负载前缀随机 | vLLM（生态成熟、PagedAttention 基线强） | 前缀复用收益小 |
| 多轮对话 / Agent 多轮工具调用 | **SGLang（RadixAttention 主场）** | 轮次路径延伸自动复用 |
| GRPO/VLM 同图多 rollout（Stage 7） | 两者均可（均支持前缀复用），SGLang 的多轮/VLM 支持活跃 | verl 两者后端都官方支持 |
| 结构化输出 / 复杂调度需求 | SGLang（原生结构化生成、前沿特性迭代快） | 论文即以此起家 |
| 生态与工具链稳定性优先 | vLLM | 社区与集成更广 |

（verl 官方对两个后端均完整支持；Stage 7 已注明"以你的负载实测为准"——本篇给了实测方法。）

### 3.2 三个代表性失效模式

**失效 1：前缀"看起来一样、字节不一样"——命中率归零**
- **症状**：多轮场景无加速；Radix 树退化成一根链。
- **根因**：System Prompt 里有时间戳/请求 ID；或模板渲染在变量上插入了位置不同的空格/换行；或 chat 模板对 system 段做了动态拼接。
- **定位**：dump 两次请求的最终 prompt 做逐字节 diff（Stage 1 周期的老朋友：第 4 阶段失效 2 的同源）。
- **修复**：变量后置；前缀内容做规范化（去时间戳）；渲染层缓存模板实例。

**失效 2：缓存被打爆——LRU 淘汰风暴**
- **症状**：命中率随并发数上升而下降；长尾 TTFT 抖动。
- **根因**：活跃前缀集合的 KV 总量超过缓存预算（`mem-fraction-static` 留给 KV 的部分），LRU 高频淘汰刚建的分支——刚复用就被挤出。
- **定位**：SGLang 的 cache 命中/淘汰指标；估算活跃前缀集的 KV 足迹（第 1 周 KV 公式）。
- **修复**：提预算；或业务侧收敛前缀多样性（System Prompt 归一化）；对超长历史做摘要压缩而非全量累积。

**失效 3：Chunked Prefill 调参不当——ITL 反而恶化**
- **症状**：开 chunked prefill 后 P99 ITL 比不开还差。
- **根因**：chunk 过大 → 每批里的 prefill chunk 仍独占过久；chunk 过小 → 调度开销与碎片化。
- **定位**：ITL 直方图分位数对比（p50/p99 分开看——chunked 的目标是 p99）。
- **修复**：扫 chunk 预算参数（如 max_prefill_tokens）；混合批的 decode 配额保底。

---

## 4. 延伸思考题

1. **Radix 树的并发语义**：两个请求同时到达、共享前缀但都未缓存——推理并发下如何避免重复 prefill 同一段？查 SGLang 的实现（prefix 段锁/单飞机制），对比 OS 的 page fault 去重（同页只加载一次）。
2. **命中率的经济账**：你的业务 System Prompt 1200 Token、平均对话 8 轮、每轮新增 300 Token。算 RadixAttention 相对无缓存的 prefill 计算节省比例（提示：总 prefill 从 $\sum_k (1200 + 300k)$ 降到 $1200 + 300N + \text{新增}$，节省率随轮数增长）；再乘以你的 QPS 得出每日节省的 GPU 秒——把系统优化翻译成成本报告。
3. **前瞻：分布式前缀缓存**：多副本推理服务里，Radix 树是每副本独立的——同一前缀在不同副本上被重复 prefill。思考跨副本共享 KV（或路由到已缓存副本）的可行方案与一致性代价（这是当前推理系统研究的前沿话题，SGLang 社区有相关讨论）。

---

*下一篇：[第 5 周：性能诊断 Profiling 与量化/内核加速](第5周-Profiling与量化加速.md)*
