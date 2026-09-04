# 第 3 周教程：MindIE 大模型推理服务与 vLLM 对照

> **本周要回答的三个问题**
> 1. MindIE 相对 vLLM，哪些概念完全相通、哪些是昇腾特有的？
> 2. 一个 7B 模型怎么在昇腾上部署成 OpenAI 兼容服务并压测？
> 3. 昇腾与 NVIDIA 同档硬件比吞吐，方法上要注意什么才公平？

对应学习计划：第 3 周。交付物：用 MindIE 部署 7B 模型，跑通 OpenAI 兼容 API 压测，输出吞吐/延迟报告 + 「vLLM ↔ MindIE 配置项」对照表。

---

## 1. 第一性原理：推理引擎的核心问题在哪个平台都一样

### 1.1 不变的问题

无论 NVIDIA 还是昇腾，LLM 推理引擎都要解决同一组问题：

- **KV cache 管理**（显存有限、随长度增长）→ PagedAttention；
- **连续批处理**（请求动态进出）→ 调度器；
- **prefill/decode 两阶段特性**（算力瓶颈 / 访存瓶颈）。

所以你在 Stage 5 学的**概念全部适用**，变的只是实现载体（CUDA kernel → 昇腾算子）。这是"概念迁移"学习的典范。

### 1.2 变的部分

- **算子实现**：Attention/GEMM 跑在 Cube/Vector 上，由 CANN 算子库提供；
- **内存模型**：HBM + UB，管理 API 不同；
- **量化支持谱系**：昇腾当前以 W8A8 / W8A16 为主，4-bit 生态在追赶。

---

## 2. MindIE 概念对照表（本周交付核心）

### 2.1 定位

MindIE（MindSpore Inference Engine）是昇腾的 LLM 推理引擎，提供：
- 连续批处理 + PagedAttention（昇腾实现）；
- OpenAI 兼容 API；
- MindIE-LLM（大模型推理）与 MindIE-Service（服务层）。

### 2.2 配置项对照

| 概念 | vLLM | MindIE |
| --- | --- | --- |
| 启动服务 | `vllm serve <model>` | MindIE-Service 配置启动 |
| 最大并发 | `--max-num-seqs` | 配置项（如 `maxBatchSize`） |
| KV cache 显存占比 | `--gpu-memory-utilization` | 对应显存配置项 |
| 最大序列长度 | `--max-model-len` | `maxSequenceLength` 类 |
| 量化 | `--quantization` | 权重格式/量化配置 |
| 张量并行 | `--tensor-parallel-size` | TP 配置（多卡） |
| OpenAI API | `/v1/chat/completions` | 兼容接口 |

（具体配置名以昇腾社区《MindIE 安装与配置指南》最新版为准——版本迭代快，别死记名称，记**概念**。）

### 2.3 概念验证清单

对照你在 vLLM 上的理解，逐项确认在 MindIE 中：
1. ✅ 连续批处理存在（观察并发下的吞吐）；
2. ✅ KV cache 管理存在（观察长序列显存）；
3. ✅ OpenAI 接口可用；
4. ⚠️ 量化谱系不同（确认支持的格式）。

---

## 3. 部署实操

### 3.1 准备权重

```bash
# HF 权重 → MindIE 支持的格式（按官方工具转换）
# 若是昇腾适配版模型（如 Qwen/Llama 的昇腾版），按其说明加载
```

**注意**：并非所有 HF 模型都能在昇腾直接跑——需确认该模型架构有昇腾适配（官方模型库或社区适配）。选型时优先选官方支持的模型。

### 3.2 启动服务

```bash
# MindIE-Service 启动（示意，实际以官方配置文件为准）
# 配置：模型路径、TP 数、最大并发、显存占比、端口
mindie-service-start --config config.json
```

```json
{
  "model_path": "/path/to/model",
  "tensor_parallel": 1,
  "max_batch_size": 32,
  "max_sequence_length": 4096,
  "port": 8080
}
```

### 3.3 OpenAI 兼容调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://<昇腾主机>:8080/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "介绍一下昇腾芯片"}],
    max_tokens=256,
)
print(resp.choices[0].message.content)
```

---

## 4. 压测与对比方法学（交付核心）

### 4.1 压测

```python
# 用 vLLM 的 benchmark_serving（它支持 OpenAI 兼容后端）
# 指向 MindIE 的端点
```

```bash
python benchmark_serving.py \
    --backend openai-chat \
    --base-url http://<昇腾主机>:8080/v1 \
    --model qwen \
    --dataset-name sharegpt \
    --num-prompts 100 --request-rate 4
```

记录：吞吐（tokens/s）、TTFT P50/P99、TPOT P50/P99。

### 4.2 昇腾 vs NVIDIA 对比的公平性（重要）

对比两平台吞吐，必须控制变量：
1. **同模型、同量化精度**（都 W8A8，或都 FP16）；
2. **同负载**（同数据集、同到达率、同并发）；
3. **同延迟约束**（在相同 SLO 下比 goodput，而非裸吞吐）；
4. **注明硬件代际**（如 910B vs A100，算力/带宽各多少）。

**不控制这些的对比都是耍流氓**——这也是面试里"你怎么测性能"的正确答法。

### 4.3 报告模板

```
《MindIE 7B 部署压测报告》
硬件：910B × 1（或 310P，注明内存限制）
模型：XX-7B，量化：XX
负载：ShareGPT 100 条，request-rate 4
结果：吞吐 __，TTFT P50/P99 __，TPOT __
对比（如有）：vs vLLM on A100（同精度同负载）
结论：__
```

**310P 注意**：若 310P 内存装不下 7B 的 KV，要么减小 `max_sequence_length`/并发，要么切云端 910B。如实记录约束。

---

## 5. 工程权衡与失效模式

### 5.1 权衡

- **模型适配成熟度**：昇腾支持的模型谱系在扩展，选型先看适配情况；
- **量化谱系**：当前以 W8A8/W8A16 为主，4-bit 选择少于 NVIDIA；
- **本地推理卡内存**：310P 内存小，长上下文/高并发受限。

### 5.2 失效模式

1. **模型不支持**：架构无昇腾适配。修复：换官方支持模型或等适配。
2. **显存不足**：7B + 长上下文超内存。修复：降并发/序列长度，或切云端。
3. **精度异常**：量化或算子问题（沿用第 2 周排查法）。
4. **吞吐远低于预期**：量化没生效或配置不当。修复：核对量化配置与实测算子。

---

## 6. 延伸思考题（含解析）

**Q1**：MindIE 和 vLLM 哪些概念相通、哪些不同？
**A**：相通：连续批处理、PagedAttention、KV cache 管理、OpenAI 接口这些**概念**。不同：实现载体（CUDA→昇腾算子）、量化谱系（昇腾以 W8A8/16 为主）、配置接口名称。概念可迁移，名称和细节要查昇腾文档。

**Q2**：为什么对比昇腾和 NVIDIA 性能要控制这么多变量？
**A**：吞吐受模型、量化精度、负载、延迟约束、硬件代际共同影响。不控制变量，差异无法归因到"芯片本身"。公平对比要同模型同精度同负载同 SLO，并注明硬件规格。

**Q3**：310P 部署 7B 最可能遇到什么限制？
**A**：内存。7B 权重 + 长上下文 KV 可能超过 310P 显存。对策：量化降权重体积、限制序列长度与并发、或切 910B。

**Q4**：为什么选型要先看昇腾的模型适配情况？
**A**：不是所有架构都有昇腾算子支持。选未适配的模型会卡在算子层。优先选官方/社区已适配的模型，避免大量自定义算子开发成本。

**Q5**：如果让你评估"某业务要不要迁到昇腾"，你测什么？
**A**：① 该模型在昇腾是否有适配/需多少自定义算子；② 同精度同负载下吞吐/延迟对比（goodput）；③ 精度是否达标；④ 迁移工作量与生态风险。给出量化的性能+成本+工作量评估。

---

## 本周交付清单

- [ ] MindIE 部署 7B 模型，跑通 OpenAI 兼容调用。
- [ ] 压测输出吞吐/延迟报告。
- [ ] 完成「vLLM ↔ MindIE 配置项」对照表（记概念不记死名称）。
- [ ] 掌握跨平台公平对比的方法学。
