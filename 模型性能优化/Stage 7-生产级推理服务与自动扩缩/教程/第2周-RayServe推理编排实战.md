# 第 2 周教程：Ray Serve 推理编排实战

> **本周要回答的三个问题**
> 1. Ray Serve 的 Deployment Graph 相比"一个服务一个模型"省了什么、多了什么？
> 2. 动态批处理（`@serve.batch`）如何把零散请求攒成一批喂给引擎？
> 3. 语音方向的级联链路（VAD→ASR→LLM→TTS）怎么用 Ray Serve 组织成 DAG？

对应学习计划：第 2 周。交付物：Docker/本地起 Ray Serve 集群，部署"vLLM 后端 + 预处理 Deployment"的推理图，开动态批处理，用 `locust` 压测，画副本数与吞吐/延迟的关系曲线，验证自动扩缩。

---

## 1. 第一性原理：编排层解决"模型之间、请求之间"的两个问题

### 1.1 问题一：多模型如何协作

真实业务很少只有一个模型。语音助手就是级联：**VAD 判活 → ASR 转文字 → LLM 思考 → TTS 合成**。若每个模型各自起服务、客户端串起来，会产生多次网络往返与客户端编排复杂度。

### 1.2 问题二：请求如何高效利用算力

零散到达的请求逐个处理，算力利用率低。**攒批（batching）**能显著提升吞吐——但"攒多久"是延迟与吞吐的权衡，需要框架帮你管。

Ray Serve 对两个问题都有答案：**Deployment Graph**（编排）+ **`@serve.batch`**（动态批处理）。

---

## 2. Ray Serve 核心概念

### 2.1 Deployment 与 Replica

- **Deployment**：一个可部署单元（一个模型、一段逻辑）；
- **Replica**：Deployment 的运行副本，可以有多个（并发能力 = 副本数 × 单副本并发）。

```python
from ray import serve
import torch

@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})
class LLMDeployment:
    def __init__(self):
        # 加载 vLLM 或任何推理后端
        self.engine = load_vllm_engine("Qwen/Qwen2.5-7B-Instruct")

    async def __call__(self, req):
        return await self.engine.generate(req["prompt"])
```

### 2.2 Deployment Graph：声明式流水线

把多个 Deployment 用 `@serve.ingress` 或 handle 调用串成 DAG：

```python
@serve.deployment
class Preprocessor:
    def __call__(self, raw):
        # 清洗、格式化、构造 prompt
        return {"prompt": build_prompt(raw)}

@serve.deployment
class InferencePipeline:
    def __init__(self, preprocessor, llm):
        self.pre = preprocessor
        self.llm = llm

    async def __call__(self, raw):
        processed = await self.pre.remote(raw)      # 调用预处理
        return await self.llm.remote(processed)      # 调用 LLM

app = InferencePipeline.bind(Preprocessor.bind(), LLMDeployment.bind())
serve.run(app)
```

**价值**：客户端只调一个入口；模型间的调用在 Ray 内部，低开销；每个环节可独立扩缩副本数。

### 2.3 动态批处理

```python
from ray.serve import Request

@serve.deployment
class BatchedLLM:
    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.1)
    async def __call__(self, requests: list):
        # requests 是攒好的一批（最多 16 个，或等 100ms）
        prompts = [r["prompt"] for r in requests]
        results = self.engine.batch_generate(prompts)
        return results   # 按顺序返回，框架分发给各请求

    def __init__(self):
        self.engine = load_vllm_engine("Qwen/Qwen2.5-7B-Instruct")
```

**参数权衡**：
- `max_batch_size` 大 → 吞吐高，但攒批久、延迟升；
- `batch_wait_timeout_s` 短 → 延迟低，但可能攒不满。

### 2.4 自动扩缩

```python
@serve.deployment(
    num_replicas=1,
    autoscaling_config={
        "target_ongoing_requests": 4,   # 每副本目标在途请求
        "min_replicas": 1,
        "max_replicas": 8,
    },
)
```

当在途请求超过目标，Ray Serve 自动加副本；负载下降再缩回。

---

## 3. 语音级联的 DAG 组织（跨方向联动）

把语音助手的云端链路组织成图：

```
用户音频 → [VAD] → [ASR] → [LLM] → [TTS] → 音频响应
```

```python
@serve.deployment
class VoiceAssistantGraph:
    def __init__(self, vad, asr, llm, tts):
        self.vad, self.asr, self.llm, self.tts = vad, asr, llm, tts

    async def __call__(self, audio):
        if not await self.vad.remote(audio):       # 判活
            return None
        text = await self.asr.remote(audio)        # 转写
        reply = await self.llm.remote(text)        # 思考
        return await self.tts.remote(reply)        # 合成

app = VoiceAssistantGraph.bind(
    VAD.bind(), ASR.bind(), LLMDeployment.bind(), TTS.bind())
```

**每个环节的副本数独立配**：ASR 快、LLM 慢，就给 LLM 更多副本。这正是编排的价值——**按瓶颈分配资源**。

---

## 4. 实战与压测（交付）

### 4.1 启动

```python
# deploy.py
import ray
from ray import serve
# ...（上面的 Deployment 定义）
serve.run(app, host="0.0.0.0", port=8000)
```

```bash
python deploy.py
```

### 4.2 locust 压测

```python
# locustfile.py
from locust import HttpUser, task, between

class User(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def infer(self):
        self.client.post("/", json={"prompt": "讲个笑话"})
```

```bash
locust -f locustfile.py --host http://localhost:8000 \
       --users 50 --spawn-rate 10 --run-time 3m --headless
```

### 4.3 副本数-吞吐/延迟曲线（交付）

改变 `num_replicas`（1/2/4），各压一轮，记录吞吐与延迟：

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

replicas = [1, 2, 4]
throughput = [...]      # 填实测
p99_latency = [...]     # 填实测

fig, ax1 = plt.subplots()
ax1.plot(replicas, throughput, 'b-o', label='吞吐')
ax2 = ax1.twinx()
ax2.plot(replicas, p99_latency, 'r-s', label='P99 延迟')
ax1.set_xlabel("副本数"); ax1.set_ylabel("吞吐", color='b')
ax2.set_ylabel("P99 延迟 (ms)", color='r')
plt.title("副本数与吞吐/延迟")
fig.tight_layout(); plt.savefig("replica_curve.png", dpi=110)
```

**预期**：吞吐随副本近似线性上升，直到某个瓶颈（CPU/网络/后端）后趋平；延迟在副本不足时高、加副本后降。**找到趋平点**——那就是下一步扩容不再划算的拐点。

---

## 5. 工程权衡与失效模式

### 5.1 权衡

- **攒批参数**：大批高吞吐但高延迟，小批反之；按在线/离线分场景配；
- **副本粒度**：每模型独立副本灵活，但模型多时资源碎片；
- **DAG vs 微服务**：DAG 低延迟紧耦合，微服务独立但多次网络往返。

### 5.2 失效模式

1. **攒批等太久**：`batch_wait_timeout` 太大，低负载下延迟高。修复：按负载调小，或动态批处理。
2. **副本加不动**：达到 `max_replicas` 或资源不足。修复：调上限、扩集群。
3. **DAG 单点**：图里某环节成为瓶颈拖累全链。修复：给瓶颈环节加副本、优化该环节。
4. **背压失控**：下游慢、上游持续灌，队列爆。修复：限流、背压机制。

---

## 6. 延伸思考题（含解析）

**Q1**：Ray Serve 相比"每模型一个独立服务"好在哪？
**A**：① 多模型编排在框架内部完成，客户端只调一个入口、减少网络往返；② 每环节独立扩缩副本，按瓶颈分配资源；③ 动态批处理、自动扩缩开箱即用。

**Q2**：`@serve.batch` 的 max_batch_size 和 wait_timeout 怎么权衡？
**A**：max_batch_size 大批吞吐高但攒批久延迟升；wait_timeout 短延迟低但可能攒不满批。在线低延迟场景用小批短超时，离线吞吐场景用大批长超时。

**Q3**：语音级联里为什么给 LLM 的副本通常最多？
**A**：LLM 是链路中最慢的环节（生成式、逐 token），吞吐瓶颈在它那。给瓶颈环节更多副本才能让整条链不被卡住。资源应按瓶颈分配。

**Q4**：自动扩缩的触发指标为什么用"在途请求"而不是 CPU？
**A**：推理服务负载特征是请求级并发，在途请求直接反映排队压力；CPU 利用率对 GPU 推理不敏感。按在途请求扩缩更贴合推理负载。

**Q5**：压测曲线里吞吐不再随副本上升，可能卡在哪？
**A**：常见瓶颈：单卡/后端吞吐上限、CPU 预处理、网络、或 locust 客户端自身压不上去。需逐一排查（看 GPU 利用率、CPU、网络、客户端负载）。

---

## 本周交付清单

- [ ] 起 Ray Serve，部署"预处理 + LLM"推理图。
- [ ] 开 `@serve.batch` 动态批处理，验证攒批生效。
- [ ] 配置自动扩缩，观察负载变化时副本增减。
- [ ] locust 压测，画副本数-吞吐/延迟曲线，找趋平拐点。
- [ ] 设计语音级联（VAD→ASR→LLM→TTS）的 DAG 结构。
