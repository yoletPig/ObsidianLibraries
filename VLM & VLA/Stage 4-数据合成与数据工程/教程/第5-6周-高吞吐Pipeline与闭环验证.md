# 第 5-6 周教程：高吞吐合成 Pipeline 工程化与闭环验证

> **本周要回答的三个问题**
> 1. asyncio + vLLM/SGLang 的高并发架构怎么组织？每分钟上百条产出的瓶颈在哪一侧？
> 2. Prefix Caching 省的是什么开销？什么样的 prompt 结构能吃到这份红利？
> 3. "合成 → 微调 → 验证"的闭环怎么设计才能证明合成数据有效？

对应学习计划：第 5-6 周。交付物：① asyncio + vLLM/SGLang 高并发流水线（每分钟上百条）；② 3,000 条高质量、已去重、可验证的多模态/Agent SFT 数据集（HF Dataset 格式）；③ Baseline 模型训练与评测得分对比。

本篇属于**系统与基础设施**主题（高吞吐生成引擎），分析重点是吞吐、并发调度与 I/O 复用，方法论直接复用 Stage 2 第 4 周的训练工程框架——生成与训练是同一台 GPU 在两个时间片上的两种负载。

---

## 1. 第一性原理：生成吞吐的账本

### 1.1 根本矛盾：串行 API 调用的等待时间 vs GPU 的批处理能力

一条合成数据的产出时间拆解（教师 VLM 生成一条含 CoT 的 QA）：

- **Prefill**：处理 prompt（含图片 Token）的一次前向，计算密集；
- **Decode**：逐 Token 生成答案（几百 Token），**访存受限**（每步都要读全部权重）；
- **串行调用的空窗**：单请求 decode 时 GPU 利用率只有个位数百分比——权重读取的带宽被浪费在"一次只推进一个 Token"上。

串行 100 条 ≈ 100 × (prefill + decode) ≈ 数十分钟到小时级。而 decode 的访存受限特性决定了：**一个 batch 推进 64 个 Token 与推进 1 个 Token 的耗时几乎相同**（权重读取是大头）。把并发打满，吞吐可提升 1~2 个数量级——这就是"每分钟上百条"的物理来源。

### 1.2 两级并发的职责划分

高吞吐流水线是**客户端并发 × 服务端 batching**的乘积：

| 层 | 职责 | 关键参数 |
| --- | --- | --- |
| **服务端**（vLLM/SGLang） | continuous batching、KV Cache 管理、Prefix Caching | `max_num_seqs`（并发序列数上限）、`gpu_memory_utilization` |
| **客户端**（asyncio） | 发起并发请求、限速、重试、结果落盘 | 并发信号量（≈ 服务端 `max_num_seqs`）、指数退避 |

常见误区是只调一边：客户端开 1000 并发但服务端 `max_num_seqs=8`，请求在服务端排队，客户端并发是假象；反之服务端能容 64 并发而客户端只发 4 个请求，GPU 喂不饱。**经验起点：客户端并发 = 服务端 max_num_seqs = 单卡 32~64（7B 教师、图片输入）**，然后按 GPU 利用率微调。

vLLM 启动参考（教师模型服务）：

```bash
# 7B 教师模型, 单卡 A100 40G
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --max-model-len 8192 \
  --max-num-seqs 48 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching          # 打开前缀缓存 (见 1.3)
```

SGLang 的差异化优势：**RadixAttention 前缀树缓存**（自动发现并复用任意公共前缀的 KV Cache，无需手动声明），且官方基准中在多轮/结构化生成场景吞吐常优于 vLLM（不同版本互有胜负，选型以自己的负载实测为准）。

### 1.3 Prefix Caching：省的是 Prefill

合成流水线的 prompt 结构高度模板化：

```text
[系统提示: 你是多模态数据合成专家…约 800 Token (所有请求相同)]
[演化算子说明: …约 300 Token (同类请求相同)]
[图片 Token: …约 600 Token (每图不同)]
[种子描述与具体要求: …约 200 Token (每条不同)]
```

前两段在数千条请求间**完全一致**。无缓存时每条请求都要重复 prefill 这 ~1100 Token；开启 Prefix Caching 后，相同前缀的 KV Cache 被复用，prefill 只需计算新增部分。**收益 = 公共前缀占比**：本例 prefill 计算量降约 60%；对"系统提示极长 + 输出较短"的合成任务（如分类打分），收益更极端。工程纪律：**把不变的内容放在 prompt 最前面，把每条不同的内容放最后**——前缀匹配是从头开始的逐段比较，中间插一个不同 Token 就断了后续复用。

（进阶：SGLang 的 RadixAttention 还能复用"图片 Token 相同"的情况——同一张种子图生成多条数据时，把该图的 Token 段也放进公共前缀。）

### 1.4 端到端数据流

```text
种子图/种子任务池
   ▼
asyncio 生产者 (构造 prompt, 公共前缀前置)
   ▼  Semaphore(48) 限并发
OpenAI 协议 client (异步) ──► vLLM/SGLang 服务 (continuous batching + prefix cache)
   ▼  流式回收
结果校验闸 (schema/长度/一致性, 第4周闸1/2 前置到这里)
   ▼
暂存 JSONL (逐条 append, 崩溃不丢)
   ▼
第 4 周清洗五闸 ──► HF Dataset ──► Stage 2 SFT ──► Stage 3 评测对比
```

两条可靠性纪律：**逐条落盘**（断点续跑：启动时读已有文件按去重键跳过）与**限速退避**（`asyncio.Semaphore` 控并发 + 429/超时指数退避重试）。

---

## 2. 实现与验证

### 2.1 高并发合成引擎（asyncio 客户端骨架）

```python
"""
asyncio 高并发多模态合成客户端 (对接 vLLM/SGLang 的 OpenAI 协议服务)。
运行方式: python stage4_week5_pipeline.py --images ./seed_images/ --out raw.jsonl \
           --api-base http://localhost:8000/v1 --model Qwen2.5-VL-7B-Instruct \
           --concurrency 48 --target 3000
依赖: openai, aiofiles(可选)
"""
import argparse
import asyncio
import hashlib
import json
import time

PREFIX = ("你是多模态数据合成专家。基于图片生成含 CoT 的高质量问答数据。"
          "严格输出 JSON: {\"question\":..., \"answer\":..., \"evolution\":...}。"
          "回答需包含 观察->推断->结论 三段结构。")          # ~公共前缀, 越靠前越好

OPS = ["补充细粒度细节并三步推理", "构造两步计数问题并作答",
       "分析两物体空间关系", "跨域改写为巡检视角", "转化为结构化 JSON 描述"]


def make_prompt(op: str, seed_desc: str) -> str:
    """公共前缀在前, 每条变化在后 (吃到 prefix cache 的关键)"""
    return f"{PREFIX}\n演化算子: {op}\n种子描述: {seed_desc}"


async def synth_one(client, sem, model, image_b64, op, out_f, lock, stats, retry=3):
    key = hashlib.md5(f"{image_b64[:32]}{op}".encode()).hexdigest()
    async with sem:                                   # 并发闸门
        payload = {"model": model, "temperature": 0.9,
                   "messages": [{"role": "user", "content": [
                       {"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                       {"type": "text", "text": make_prompt(op, "种子描述")}]}]}
        for attempt in range(retry):
            try:
                rsp = await client.chat.completions.create(**payload)
                text = rsp.choices[0].message.content
                rec = json.loads(text[text.index("{"): text.rindex("}") + 1])
                if not all(rec.get(k) for k in ("question", "answer")):
                    raise ValueError("schema")
                rec.update({"op": op, "key": key})
                async with lock:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["ok"] += 1
                return
            except Exception as e:
                if attempt == retry - 1:
                    async with lock:
                        stats["fail"] += 1
                else:
                    await asyncio.sleep(2 ** attempt)   # 指数退避


async def run(args):
    import base64
    from pathlib import Path
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=args.api_base, api_key="EMPTY", timeout=120)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = {"ok": 0, "fail": 0}
    out_f = open(args.out, "a", encoding="utf-8")

    images = [base64.b64encode(open(p, "rb").read()).decode()
              for p in sorted(Path(args.images).glob("*.jpg"))][: 30]
    jobs = [(img, op) for img in images for op in OPS]     # 30 图 × 5 算子
    t0 = time.time()
    await asyncio.gather(*[synth_one(client, sem, args.model, img, op,
                                     out_f, lock, stats) for img, op in jobs])
    out_f.close()
    dt = time.time() - t0
    print(f"完成: ok={stats['ok']} fail={stats['fail']} "
          f"吞吐={stats['ok'] / dt * 60:.0f} 条/分钟 (并发={args.concurrency})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.set_defaults(images="seed_images", out="raw.jsonl", concurrency=48,
                    target=3000, model="Qwen2.5-VL-7B-Instruct",
                    api_base="http://localhost:8000/v1")
    ap.add_argument("--images"); ap.add_argument("--out"); ap.add_argument("--concurrency", type=int)
    ap.add_argument("--target", type=int); ap.add_argument("--model"); ap.add_argument("--api-base")
    asyncio.run(run(ap.parse_args()))
```

**性能验收口径**（学习计划要求"每分钟上百条"）：7B 教师、单卡 A100、输出约 300 Token/条时，48 并发下吞吐的合理量级是 100~300 条/分钟（**务必以自己环境实测为准**）。若实测远低于此，按 Stage 2 第 4 周 3.3 节的决策树排查——生成场景的瓶颈几乎都在"客户端并发没打满"或"输出长度失控"。

**扩到 3000 条**：30 图 × 5 算子只有 150 组合，需要组合扩展：多轮演化（对已生成数据再施加算子）+ 第 2/3 周的空间/轨迹数据混合，凑足目标量后进清洗五闸，最终按质量分截取 3000 条。

### 2.2 交付 HF Dataset 格式

```python
"""把清洗后的 JSONL 转为 HF Dataset 并保存/推送"""
from datasets import Dataset, Features, Value, Image as HFImage

def to_hf_dataset(clean_jsonl: str, images_dir: str) -> Dataset:
    rows = [json.loads(l) for l in open(clean_jsonl)]
    ds = Dataset.from_dict({
        "image": [f"{images_dir}/{r['image']}" for r in rows],
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "quality": [r.get("quality", 0) for r in rows],
    }, features=Features({
        "image": HFImage(), "question": Value("string"),
        "answer": Value("string"), "quality": Value("float32")}))
    return ds

if __name__ == "__main__":
    ds = to_hf_dataset("clean.jsonl", "./seed_images")
    assert len(ds) >= 3000, f"交付量不足: {len(ds)}"
    ds.save_to_disk("sft_3000_final")            # ds.push_to_hub("you/dsname")
    print(ds)
```

### 2.3 闭环验证：合成数据到底有没有用

**实验设计**（这是 MVP 第 3 项的核心，也是自测"工程交付"的实验部分）：

| 组 | 训练数据 | 目的 |
| --- | --- | --- |
| Baseline | 仅公开基础数据（如 LLaVA 指令集等量抽样） | 参照系 |
| +Synthetic | 同量公开数据 + 3000 合成数据 | 合成数据增益 |

**控制变量纪律**（Stage 3 第 3 周方法论复用）：两组的底座、超参、epoch、种子完全一致；唯一变量是数据；评测用 Stage 3 的固定协议（POPE/MME/自建 mini-Bench，锁定框架版本与提取器）。

**判读框架**：

- 预期增益方向与 Stage 3 归因报告对齐：合成数据按归因规格定点合成（空间线补定位短板、CoT 线补推理），则对应维度应显著提升，而 MME 总分可能仅小幅变化——**定点提升 > 全面上浮**是合成数据有效性的健康信号；
- 若全维度无差异：查合成数据的实际入训量（可能被清洗筛掉了大半）、以及与公开数据的同质度（语义去重后两池重叠高则无增量信息）；
- 若部分维度下降：查合成数据占比是否过高引发分布倾斜（Stage 2 第 3 周的配比问题），回补通用数据。

**报告模板一行摘要**：`底座/数据量/训练配置 + [Baseline vs +Synthetic 在各维度得分] + 归因报告链接`——能被复现的增益才叫增益。

---

## 3. 工程权衡与失效模式

### 3.1 决策表：生成引擎参数

| 参数 | 起点 | 调整方向 |
| --- | --- | --- |
| 客户端并发 = max_num_seqs | 48（7B 单卡） | GPU util <80% 加并发；出现排队延迟（TPOT 上升）减并发 |
| temperature | 0.9 | 多样性任务高；事实抽取类降到 0.2 |
| max_tokens | 400~600 | 合成 CoT 常见长度 + 30% 余量；失控输出是吞吐杀手 |
| 重试 | 3 次指数退避 | 配合限速（429）识别；重试预算烧完即记录失败不阻塞全局 |

### 3.2 三个代表性失效模式

**失效 1：输出长度失控，吞吐暴跌**
- **症状**：并发正常但吞吐从 200 条/分钟掉到 30 条/分钟。
- **根因**：某些演化算子诱发教师"无限展开"（如"详细分析"没有上限约束），decode 步数 10 倍膨胀——decode 是吞吐的倒数项。
- **定位**：日志统计输出 Token 分布的 P99；抽查超长样本内容。
- **修复**：`max_tokens` 硬上限 + prompt 明确字数约束 + 对超长样本截断重写；把"输出长度"加入闸 1 校验。

**失效 2：Prefix Cache 命中率极低，优化白做**
- **症状**：开了 `--enable-prefix-caching` 但 prefill 耗时不降。
- **根因**：prompt 里把日期戳/随机 ID 放在了开头（每条都不同，前缀在第一段就断）；或图片 Token 段被放在系统提示之前。
- **定位**：vLLM 日志有 prefix cache 命中统计（或用 SGLang 的 radix cache 指标）；命中率 <30% 即结构错误。
- **修复**：重排 prompt——"全局不变 → 任务类不变 → 图片 → 逐条变化"，严格从通用到特异。

**失效 3：闭环对比"无效"——两组数据根本不同质**
- **症状**：+Synthetic 组反而更差，得出"合成数据无用"的结论。
- **根因**：两组训练总量不同（加合成数据时总量变了，epoch 效应混淆）；或合成数据的分布与公开数据严重错位（第 1 周失效 3），加入即偏移。
- **定位**：核对两组的总 Token 数与步数是否一致；对合成池与公开池做 CLIP 语义分布对比（第 4 周工具）。
- **修复**：等总量重配（Baseline 组补抽样）；合成池按业务分布再配额；或退一步先做"少量合成数据 + 低占比混合"的消融梯度。

---

## 4. 延伸思考题

1. **吞吐建模**：设教师 7B、输出 300 Token、单卡 decode 单序列约 30ms/Token、连续 batching 下每步 60ms 推进 48 序列。估算理论吞吐（条/分钟），再乘一个 0.4~0.6 的工程折扣解释你实测值的差距来源（prefill、调度、失败重试、长度方差）。
2. **成本-质量前沿**：闭源 API（高质量、$X/千条）vs 本地 7B（质量 85 分、边际成本趋零）vs 本地 72B（质量 95 分、需要多卡）。给定"1 周内交付 3 万条、质量分均值 ≥4/5"的约束，设计三层混合策略并写出每层的量与验收闸。
3. **前瞻 Stage 5**：本阶段的清洗闸是基于"样本级"质量的；Stage 5（数据质量评价与高效筛选）会把视角升级到"数据集级分布"（与目标任务分布的匹配度、information density）。思考：一个样本 5 分但整个数据集分布偏斜的场景，样本级闸为什么无能为力？（提示：回到 Stage 3 的"配比"与"分布画笔"——样本级满分数据堆成一个偏斜分布，仍然训出偏科模型。）

---

*下一篇：[阶段四自测验收与复盘](阶段四自测验收与复盘.md)*
