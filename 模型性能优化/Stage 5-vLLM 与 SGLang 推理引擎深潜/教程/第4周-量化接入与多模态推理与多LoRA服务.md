# 第 4 周教程：量化接入与多模态推理与多 LoRA 服务

> **本周要回答的三个问题**
> 1. Stage 2 量化的 {GPTQ, AWQ, FP8, INT8-SQ} 在 vLLM 里分别映射到哪个 kernel？`--quantization` 如何与模型匹配验证？
> 2. 音频/图像这类多模态输入，在进入引擎前经过怎样的预处理流水线？
> 3. 多 LoRA 如何共享同一个底座权重？热插拔是怎么实现的？

对应学习计划：第 4 周。交付物：① 把 GPTQ/AWQ/FP8 三个量化模型逐个接入 vLLM，验证引擎内量化与离线量化等价；② 部署多模态模型（Qwen2-Audio），走通"音频输入 → 引擎 → 文本输出"。

---

## 1. 第一性原理：量化格式必须与"算子实现"匹配

### 1.1 量化不只是"存得小"

Stage 2 你学会了把权重压成 4-bit/8-bit。但要在引擎里真正**加速**，压好的权重必须被**专门的 kernel** 消费——否则引擎只会把它反量化回高精度再算，白压。

所以核心命题是：

$$
\text{量化格式} \;\longleftrightarrow\; \text{对应的加速 kernel}
$$

格式与 kernel 不匹配，量化只剩"省显存"而失去"提速"。

### 1.2 权重存储与计算的分离

量化模型在引擎里的典型形态：

- **权重**：低比特存储（省显存、降带宽）；
- **计算**：GEMM 时用反量化或专用 kernel，把低比特权重与激活结合。

这就是为什么"权重 4-bit、激活 16-bit"（W4A16）常见——权重是访存大头，激活保持精度。

---

## 2. 各量化格式在 vLLM 中的通路

### 2.1 映射表（本周核心表）

| 量化格式 | 位宽 | vLLM kernel / 后端 | 启动参数 | 典型场景 |
| --- | --- | --- | --- | --- |
| GPTQ | W4A16 | **Marlin**（或 exllama） | `--quantization gptq` | 显存受限的 4-bit |
| AWQ | W4A16 | **Marlin** | `--quantization awq` | 同上，激活感知更稳 |
| FP8 (W8A8) | 8-bit 浮点 | **cutlass scaled_mm** | `--quantization fp8` | Hopper/Ada，精度好 |
| INT8-SQ (SmoothQuant) | W8A8 | cutlass int8 gemm | `--quantization compressed-tensors` 等 | INT8 服务部署 |
| bitsandbytes NF4 | 4-bit | （vLLM 支持有限） | — | 训练侧为主 |

**关键理解**：
- **Marlin** 是为 4-bit 权重 × 16-bit 激活优化的 GEMM kernel，batch=1 下能比 FP16 还快（带宽收益 > 反量化开销）；
- **cutlass scaled_mm** 处理带缩放因子的低比特/浮点 GEMM（FP8、INT8）。

### 2.2 `--quantization` 的匹配验证

启动时引擎会检查：

1. 模型 `config.json` 里的 `quantization_config` 声明的格式；
2. 与 `--quantization` 参数是否一致；
3. 当前 GPU 是否支持对应 kernel（如 FP8 需要 Hopper/Ada）。

不匹配会直接报错——这是好事，避免"量化了却没走加速路径"的静默退化。

### 2.3 接入验证实验（交付）

```bash
# 三个量化模型逐个启动，确认走对 kernel
vllm serve <模型>-GPTQ --quantization gptq
vllm serve <模型>-AWQ  --quantization awq
vllm serve <模型>-FP8  --quantization fp8
```

对每个：
1. 启动日志确认加载了量化权重、未报"falling back to dequantize"；
2. 压测吞吐与延迟，与 Stage 2 的横评数据对照；
3. 用相同提示词比对输出质量，确认量化精度与离线一致。

```python
# 等价性验证：同一批提示词，量化版与 FP16 版输出对比
# 记录 ROUGE/精确匹配，确认引擎内量化没有引入额外退化
```

**预期**：三者吞吐 > FP16（尤其 4-bit 在显存受限时能开更大 batch）；输出质量与 Stage 2 离线评测一致。若明显退化，检查是否走错 kernel 或量化参数不匹配。

---

## 3. 低比特新前沿

### 3.1 W4A4：权重激活都 4-bit

比 W4A16 更激进，激活也量化。难点：激活的离群值比权重更难压（回想 SmoothQuant 的动机）。当前仍在探索，精度损失需仔细评估。

### 3.2 KV 量化进生产

第 1 周讲过：FP8 KV 已基本可用，KIVI 2-bit 更激进。生产中 FP8 KV 因"精度损失小 + 并发翻倍"被越来越多采用；2-bit KV 仍多用于研究。

---

## 4. 多模态推理：音频输入怎么进引擎

### 4.1 多模态输入的预处理流水线

与纯文本不同，音频/图像输入要先**编码成 token/特征**，再进 LLM：

$$
\text{音频} \xrightarrow{\text{重采样到 16kHz}} \xrightarrow{\text{Mel/特征提取}} \xrightarrow{\text{Audio Encoder}} \text{音频特征} \xrightarrow{\text{Projector}} \text{LLM embedding}
$$

这正是你语音方向 + VLM 方向的交汇点——音频塔 ≈ Vision Tower，Projector 同构。

### 4.2 引擎侧的处理

vLLM 对多模态模型：

1. **接收**：请求里带音频（URL/base64/路径）；
2. **预处理**：按模型的 preprocessor 配置提取特征（采样率、窗口、维度都要与训练一致）；
3. **占位替换**：把音频特征投影后替换输入序列中的音频占位符；
4. **前向**：进 LLM 生成。

### 4.3 部署与调用（交付）

```bash
# 部署音频多模态模型（以 Qwen2-Audio 类为例）
vllm serve <audio-multimodal-model>
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="<audio-multimodal-model>",
    messages=[{
        "role": "user",
        "content": [
            {"type": "input_audio", "input_audio": {"data": "<base64或URL>", "format": "wav"}},
            {"type": "text", "text": "这段音频说了什么？"},
        ],
    }],
)
print(resp.choices[0].message.content)
```

**验证点**：音频采样率是否正确重采样到模型要求（通常 16 kHz）；音频占位符是否被正确替换；输出文本是否合理。

### 4.4 Qwen2.5-Omni 类模型的特殊性

全模态模型（音频+视频+图像）的 serving 更复杂：多模态时间轴对齐、流式音频输入输出。这些是生产级多模态服务的进阶，本周先走通单音频→文本。

---

## 5. 多 LoRA 服务（S-LoRA）

### 5.1 问题：一个底座、很多适配器

实际场景常需同时服务几十上千个 LoRA（每客户/每任务一个）。若每加载一个就复制整个底座，显存爆炸。

### 5.2 S-LoRA 的解法：共享底座 + 动态挂载

$$
\text{一份底座权重} \;+\; \text{多个轻量 LoRA} \;\xrightarrow{\text{按请求动态挂载}}\; \text{各自适配的输出}
$$

- **底座**只存一份，所有请求共享；
- **LoRA 权重**很小（低秩），按需加载到显存、可驻留可换出；
- **batch 内混合**：同一批里不同请求用不同 LoRA，通过分块计算或 padding 处理。

### 5.3 热插拔

新 LoRA 上传 → 注册进引擎 → 后续请求即可引用；下线则卸载。无需重启服务、无需动底座。

```python
# vLLM 动态加载 LoRA 的概念接口
# 启动时：--enable-lora --lora-modules adapter_a=/path/a adapter_b=/path/b
# 运行时通过 API 动态增删适配器（以引擎实际接口为准）
```

**关键收益**：把"每适配器一个服务实例"降为"一个实例 + 动态适配器"，显存与运维成本大幅下降。

---

## 6. 工程权衡与失效模式

### 6.1 权衡

- **4-bit（Marlin）**：显存省、能开大 batch，但精度略降；
- **FP8**：精度好、需较新硬件；
- **多模态预处理**：特征配置必须与训练严格一致，否则输出乱。

### 6.2 失效模式

1. **量化没走加速路径**：静默反量化回高精度，只省显存不提速。诊断：看启动日志与实测吞吐。
2. **多模态采样率错配**：音频未按模型要求重采样，识别/理解全错。修复：预处理强制重采样。
3. **多 LoRA 显存超限**：驻留适配器太多。修复：LRU 换出、控制同时活跃数。
4. **batch 内混合 LoRA 低效**：适配器太碎导致计算碎片化。修复：按适配器聚批。

---

## 7. 延伸思考题（含解析）

**Q1**：为什么 4-bit 量化（GPTQ/AWQ）在 batch=1 时可能比 FP16 还快？
**A**：decode 是访存瓶颈，4-bit 权重体积是 FP16 的 1/4，加载时间大幅缩短；这个带宽收益超过了反量化的计算开销，故单请求延迟反而更低。大 batch 下计算占比上升，优势缩小。

**Q2**：FP8 量化为什么需要 Hopper/Ada 架构？
**A**：FP8 的硬件加速依赖 Tensor Core 对 FP8 数据类型的原生支持，这是 Hopper/Ada 及以后才有的。老架构无原生支持，无法走加速路径。

**Q3**：多模态输入进引擎前为什么要"重采样到模型要求"？
**A**：模型的音频编码器在特定采样率（通常 16 kHz）上训练，特征提取（窗口、维度）都基于该采样率。输入采样率不符会导致时间轴与特征全错，输出失真。

**Q4**：多 LoRA 共享底座为什么能大幅省显存？
**A**：底座权重是大头，只存一份被所有请求共享；每个 LoRA 只是低秩增量、极小。相比"每适配器复制整个底座"，显存从 O(底座×N) 降到 O(底座 + N×小增量)。

**Q5**：LoRA 热插拔如何做到不重启服务？
**A**：底座与适配器解耦。新适配器只是注册一组小的低秩权重并挂载到底座的前向路径，卸载则移除。底座不变、服务不重启，适配器可动态增删。

---

## 本周交付清单

- [ ] GPTQ/AWQ/FP8 三模型逐个接入，确认走对 kernel、未静默反量化。
- [ ] 压测吞吐并验证输出质量与离线量化一致。
- [ ] 部署音频多模态模型，走通"音频 → 文本"调用。
- [ ] 能解释各格式的 kernel 映射与多 LoRA 共享机制。
