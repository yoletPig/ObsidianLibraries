# 第 4 周教程：流式 ASR 与低延迟解码

> **本周要回答的四个问题**
> 1. 离线识别（看完整句）与流式识别（边听边出字）的本质区别是什么？
> 2. chunk 多大？look-ahead 几步？延迟与精度的权衡公式是什么？
> 3. 端点检测（Endpointing）如何决定"这句话说完了"？判早了判晚了分别什么后果？
> 4. 端到端首字延迟（First Token Latency）怎么拆预算？你的 200 ms 花在哪？

对应学习计划：第 4 周。交付物：用流式模型实现实时转写脚本，按 60 ms chunk 喂入，打印增量识别结果与首字延迟。

---

## 1. 第一性原理：流式 = 因果约束下的在线决策

### 1.1 根本矛盾

离线 ASR 看完整句再输出，可以利用全局上下文（后文帮前文消歧）。流式 ASR 必须在**只看到当前及过去音频**时就输出当前字——这是**因果约束（causal constraint）**：

$$
\hat{y}_t \text{ 只能依赖 } x_1, x_2, \dots, x_t
$$

代价：看不到未来 → 消歧能力下降 → 同模型下流式精度通常比离线低 10-30%。所有流式设计的核心都是：**在"看多少未来"与"等多久才出字"之间找平衡**。

### 1.2 三种流式架构

| 方案 | 机制 | 延迟 | 精度 | 代表 |
| --- | --- | --- | --- | --- |
| **Chunk-wise** | 每攒够一个 chunk 才编码该块（块内可见、块间因果） | 中（≈chunk 长） | 中 | E-Branchformer streaming |
| **Causal conv** | 卷积只看过去（右填充为 0），逐帧推进 | 低 | 较低 | 全因果 encoder |
| **Look-ahead** | 允许偷看未来 $k$ 帧，换取消歧 | 中（+$k$ 帧） | 较高 | Paraformer-streaming |

**核心权衡式**（直觉）：

$$
\text{端到端延迟} \approx \underbrace{\text{chunk 缓冲}}_{\text{攒够才算}} + \underbrace{\text{look-ahead}}_{\text{偷看未来}} + \underbrace{\text{计算耗时}}_{\text{模型前向}} + \underbrace{\text{端点检测}}_{\text{确认说完}}
$$

每一项都是"精度杠杆"——加大它精度升、延迟也升。

### 1.3 Chunk 大小的量化感觉

以 16 kHz、帧移 10 ms 计：

- chunk = 10 帧 = 100 ms：低延迟，但上下文短、精度差；
- chunk = 20 帧 = 200 ms：折中；
- chunk = 60 帧 = 600 ms：接近离线质量，但延迟高。

端侧语音助手通常选 100-200 ms chunk + 少量 look-ahead，把算法延迟控制在 200-300 ms。

---

## 2. 流式解码的状态管理

### 2.1 增量解码的缓存

流式模型每来一个 chunk 就前向一次，但必须缓存**跨 chunk 的状态**：

1. **编码器缓存**：卷积/注意力的历史上下文（否则块边界处信息断裂）；
2. **解码器状态**：已生成字、KV cache；
3. **partial hypothesis**：当前未确认的候选，新 chunk 到来时可能被修正。

### 2.2 partial result 与 revision

流式输出分两级：

- **partial（临时）**：快速给出当前猜测，允许后续改写——用户看到的是"边打字边修正"；
- **final（确认）**：端点检测到句子结束，输出不再变更。

**工程要点**：partial 的修正幅度要控制。频繁大改会让字幕"跳动"，体验差；常用手段是限制修订窗口、或只在置信度足够时才刷新显示。

---

## 3. 端点检测（Endpointing）：何时算"说完"

### 3.1 判据

端点检测回答"用户这句话结束了吗"，典型判据：

1. **静音时长**：检测到连续静音超过阈值（如 500-800 ms）→ 判定结束；
2. **解码稳定性**：长时间无新字输出；
3. **语义完整性**（进阶）：用语言模型判断句子是否完整。

### 3.2 判早/判晚的代价

- **判早**（过早截断）：把还在说的用户切断，漏掉后半句——体验灾难；
- **判晚**（等太久）：响应延迟拉长，用户说完还要干等——同样糟糕。

这是一个**直接面向用户体验**的权衡，生产上通常 500-700 ms 静音阈值 + 可配置。你的结课助手会在这里反复调参。

---

## 4. 流式实战（交付核心）

### 4.1 用 FunASR 流式模型

```python
from funasr import AutoModel

chunk_size = [5, 10, 5]   # [左看, 当前块, 右看(look-ahead)]，单位：帧(10ms)
model = AutoModel(
    model="paraformer-zh-streaming",
    disable_update=True,
)

cache = {}                 # 跨 chunk 的状态缓存
```

`chunk_size=[5,10,5]` 的含义：当前块 10 帧，向右偷看 5 帧（look-ahead），左侧 5 帧是历史缓存。这是延迟/精度的可调旋钮。

### 4.2 实时转写循环

```python
import numpy as np
import soundfile as sf
import time

wave, sr = sf.read("test_set/sample.wav", dtype="float32")
if wave.ndim > 1:
    wave = wave.mean(axis=1)
assert sr == 16000, "流式模型要求 16kHz"

# 每帧 10 ms = 160 样本（16 kHz 下 10ms × 16 样本/ms）。
# 一个 10 帧的 chunk = 10 × 160 = 1600 样本 = 100 ms。
chunk_stride = 1600                   # 100 ms 一块
n_chunks = int(np.ceil(len(wave) / chunk_stride))

cache = {}
first_token_time = None
t_start = time.time()

for i in range(n_chunks):
    chunk = wave[i*chunk_stride : (i+1)*chunk_stride]
    is_final = (i == n_chunks - 1)
    res = model.generate(
        input=chunk,
        cache=cache,
        is_final=is_final,
        chunk_size=chunk_size,
    )
    text = res[0]["text"] if res else ""
    if text and first_token_time is None:
        first_token_time = time.time() - t_start
    if text:
        print(f"[chunk {i:3d}] {text}")
    if is_final:
        print("【最终】", res[0].get("text", ""))

if first_token_time is not None:
    print(f"首字延迟（相对处理起点）: {first_token_time*1000:.0f} ms")
```

**预期**：每 100 ms 一个 chunk 打印增量识别结果，句子末尾输出最终文本，并显示首字延迟。

**注意**：上面的延迟是"从开始喂第一个 chunk 到出第一个字"的**算法延迟**（含攒块 + 前向）。真实麦克风采集还要叠加**采集缓冲**与**播放延迟**，端到端预算要单独核算。

### 4.3 首字延迟预算拆解（交付）

针对你的结课助手（Stage 7），端到端延迟预算示例：

| 环节 | 预算 | 说明 |
| --- | --- | --- |
| 麦克风采集缓冲 | ~30 ms | 几帧采集延迟 |
| VAD 端点确认 | ~200 ms | 静音判定的必要等待 |
| ASR 流式（攒块+前向） | ~150 ms | chunk 100ms + 计算 |
| LLM 首 token | ~400 ms | 端侧量化模型 |
| TTS 首包 | ~200 ms | 流式合成 |
| **合计** | **~1.0 s** | 目标 < 1.5 s |

这张表就是你结课项目系统设计文档的延迟预算雏形——**本周把它建起来**。

---

## 5. 工程权衡与失效模式

### 5.1 权衡

- **chunk 大小**：小 → 低延迟、高精度损失；大 → 反之。100-200 ms 是甜点；
- **look-ahead**：每多看 1 帧 = 多 10 ms 延迟，换一点消歧。通常 3-5 帧；
- **partial 修订**：频繁修订体验差，但不修订会显示错误中间结果。控制修订窗口。

### 5.2 失效模式

1. **块边界吞字**：chunk 切分恰好在词中，模型看不到完整词 → 识别错。修复：重叠缓冲、look-ahead。
2. **流式精度骤降**：chunk 太小导致上下文不足。修复：加大 chunk 或换更大流式模型。
3. **端点误判**：判早漏句、判晚延迟。修复：调静音阈值、结合解码稳定性双判据。
4. **首字延迟抖动**：计算耗时不稳定（CPU 降频、缓存未命中）。修复：预热模型、固定线程数。

---

## 6. 延伸思考题（含解析）

**Q1**：为什么流式 ASR 天然比离线精度低？差距能完全消除吗？
**A**：因果约束使其无法利用未来上下文消歧，信息量客观更少。不能消除，只能靠 look-ahead、更强模型、更大训练数据缩小差距。

**Q2**：`chunk_size=[5,10,5]` 中三个数分别是什么？把第三个数改成 0 会怎样？
**A**：[左缓存, 当前块, 右 look-ahead]。第三个改 0 = 不看未来，延迟最低但消歧能力下降，精度降低。

**Q3**：端点检测判早与判晚，哪个对语音助手体验更致命？为什么？
**A**：判早更致命——它直接切断用户正在说的话，造成漏识别且用户无法补救；判晚只是多等一点，用户可感知但信息完整。

**Q4**：partial 结果为什么需要"修订"机制？不修订会怎样？
**A**：流式早期上下文不足，初判可能错；新音频到来可修正。不修订则错误被固化为最终结果。修订机制用"允许改"换"最终对"。

**Q5**：给出一个 200 ms 首字延迟的拆解，并指出最大可优化项。
**A**：攒块 100 ms + 前向计算 60 ms + 调度开销 40 ms。最大可优化项是攒块——减小 chunk 或用因果卷积逐帧推进可压低，代价是精度。

---

## 本周交付清单

- [ ] 跑通 `paraformer-zh-streaming`，按 100 ms chunk 增量转写并打印。
- [ ] 测量并记录首字延迟（算法延迟），与离线识别的 CER 对比流式精度损失。
- [ ] 实验 `chunk_size` 第三项（look-ahead）= 0 / 5 的精度-延迟差异。
- [ ] 建立端到端延迟预算表（结课项目雏形）。
