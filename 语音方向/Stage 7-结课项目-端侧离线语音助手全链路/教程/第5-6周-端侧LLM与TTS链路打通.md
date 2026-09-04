# 第 5-6 周教程：端侧 LLM 与 TTS 链路打通——唤醒、思考、开口、被打断

> **本周要回答的四个问题**
> 1. Qwen2.5-1.5B INT4 的权重与 KV cache 各占多少内存？上下文长度如何被内存反推？
> 2. TTS 怎么「边生成边播」？首包 300 ms 预算靠什么机制达成？
> 3. 唤醒词引擎常驻系统里，误唤醒与漏唤醒怎么平衡？
> 4. 语音打断的状态机有几个状态？「开口→TTS 静音」≤200 ms 怎么保证？

对应学习计划：第 5-6 周。交付物：板端跑通「唤醒 → 听 → 想 → 说」闭环，100 次对话延迟分布（P50/P95）；实现语音打断并测量截断延迟。

---

## 1. 第一性原理：端侧 LLM 是内存问题，不是算力问题

### 1.1 INT4 权重内存

1.5B 参数 INT4（每参数 0.5 字节 + 分组量化的 scale 开销 ~10%）：

$$
M_{\text{weight}} \approx 1.5 \times 10^9 \times 0.5 \times 1.1 \approx 0.83\,\text{GB}
$$

3B 则约 1.7 GB——都在 8 GB 板子的承受范围内，**真正的变量是 KV cache**。

### 1.2 KV cache：每多一个上下文位置都要付费

自回归生成时，每个历史 token 的 K/V 向量要缓存。Qwen2.5-1.5B 的结构参数（28 层、GQA 2 个 KV 头、head dim 128、fp16）：

$$
M_{\text{KV}} = 2 \times n_{\text{layer}} \times n_{\text{kv\_head}} \times d_{\text{head}} \times L \times 2\,\text{bytes}
$$

代入：每位置 $2 \times 28 \times 2 \times 128 \times 2 = 28\,672$ 字节 ≈ **28 KB/token**。这就是本周代码要验证的核心数字——上下文长度是被内存**反推**出来的：

$$
L_{\max} = \frac{M_{\text{KV 预算}}}{28\,\text{KB}}
$$

给 KV cache 0.5 GB → $L_{\max} \approx 18000$，绰绰有余；但 4 GB 板子或 3B 模型（每位置 ~56 KB）时必须精算。

### 1.3 prefill 与首 token 延迟

首 token 延迟 ≈ prefill（并行处理全部输入 token）时间。prefill 计算量 $\propto L_{\text{输入}}$，所以：

- 会话历史必须截断（第 1-2 周已定 4 轮滑窗）；
- 系统提示词能短则短——每个常驻字符都在每次请求里重复付费。

---

## 2. 端侧 LLM 部署：llama.cpp 与 rknn-llm

### 2.1 两条路径

| 路径 | 适用 | 说明 |
| --- | --- | --- |
| llama.cpp（GGUF INT4） | CPU 先行，立刻可用 | aarch64 编译成熟、NEON 加速、量化方案全（Q4_K_M 是质量甜点） |
| rknn-llm | NPU 回填，性能优化方向联动 | 官方支持 Qwen2 系，需按其量化要求转模型 |

策略与 ASR 周相同：**llama.cpp 先打通，rknn-llm 后回填**，接口层封装成「文本进、文本流出」的黑盒。

### 2.2 部署骨架

```bash
# llama.cpp：转换 + 量化 + 编译后的板端运行
llama-cli -m qwen2.5-1.5b-instruct-q4_k_m.gguf \
    --ctx-size 4096 --threads 4 --affinity 0xF00 \   # 绑 4 个大核
    -cnv                                                 # 会话模式
```

Python 侧用 `llama-cpp-python` 的流式接口消费文本流——LLM 每产出一段文本就喂给 TTS 切句，这是「想」与「说」并行的关键。

---

## 3. TTS 边生成边播

### 3.1 分段策略

整句合成必破 300 ms 首包预算。拆法：

$$
\text{LLM 文本流} \xrightarrow{\text{按标点切句}} [\text{句}_1, \text{句}_2, \dots] \xrightarrow{\text{逐句合成}} \text{PCM 块流} \xrightarrow{\text{队列}} \text{播放}
$$

- **第一句特殊对待**：哪怕只有半句（「好的，」）也先出声——用户听到任何响应，容忍度立刻上升；
- 句间用双缓冲：播句 1 时合成句 2，消除句间停顿；
- 每块带序列号：打断时按序列号清空队列（第 1-2 周的接口契约在此兑现）。

### 3.2 首包延迟构成

$$
T_{\text{TTS 首包}} = T_{\text{等第一句}} + T_{\text{首句合成}} + T_{\text{播放缓冲}}
$$

三项都要压：等第一句靠 LLM 流式输出（别等整段）、首句合成靠小模型、播放缓冲 1~2 块（20~40 ms）。

---

## 4. 唤醒词与打断状态机

### 4.1 唤醒词的权衡

误唤醒率（每天误触发次数）与漏唤醒率此消彼长：阈值调高 → 误唤醒↓ 但用户喊不应。产品经验：**每天 ≤1 次误唤醒**可接受，漏唤醒不可接受（用户直接放弃使用）。双唤醒词（「小 X 小 X」）比单词误唤醒率低一个量级。

### 4.2 状态机（本周交付核心）

```
        唤醒词命中            VAD 端点             LLM/TTS 就绪
IDLE ──────────────> LISTENING ──────────> THINKING ──────────> SPEAKING
  ^                     │ 超时30s              │                    │
  │                     └──────────────────────┴──> IDLE（无响应）  │
  │                                                                  │
  └──────────────────── SPEAKING 中检测到用户语音（打断）<───────────┘
```

打断的 200 ms 预算拆解：VAD 检测（≤100 ms，低阈值快速通道）+ TTS 停止与队列清空（≤50 ms）+ 回 LISTENING（≤50 ms）。**打断检测必须用独立的低阈值能量+VAD 通道**，不能等 ASR——等识别出「停」字就晚了。

---

## 5. 实现与验证

### 5.1 KV cache 内存计算器（验证第 1.2 节的数字）

```python
def kv_cache_bytes(n_layers, n_kv_heads, head_dim, ctx_len, bytes_per_elem=2):
    return 2 * n_layers * n_kv_heads * head_dim * ctx_len * bytes_per_elem

# Qwen2.5-1.5B：28 层，GQA 2 KV 头，head dim 128，fp16
per_token = kv_cache_bytes(28, 2, 128, 1)
print(f"1.5B 每位置 KV: {per_token} 字节 = {per_token/1024:.1f} KB")
assert per_token == 28672, "Qwen2.5-1.5B 每位置应为 28 KB"

budget_mb = 512
max_ctx = budget_mb * 1024 * 1024 // per_token
print(f"0.5 GB KV 预算 -> 最大上下文 {max_ctx} 位置")
assert max_ctx >= 4096, "预算应能容纳 4K 上下文"

# 3B 对照：36 层，同样 2 KV 头
per_token_3b = kv_cache_bytes(36, 2, 128, 1)
assert per_token_3b > per_token, "3B 的 KV 开销应更高"
print(f"3B 每位置 KV: {per_token_3b/1024:.1f} KB")
print("KV cache 预算验证通过 ✓")
```

**预期输出**：28.0 KB/位置、0.5 GB 容纳 18724 位置、3B 约 36 KB/位置。**交付**：用你实际部署的模型结构参数重跑一遍，写进内存预算表。

### 5.2 打断状态机（可运行、可测试）

```python
import time
from enum import Enum

class State(Enum):
    IDLE, LISTENING, THINKING, SPEAKING = range(4)

class DialogStateMachine:
    def __init__(self, vad_threshold_bargein=0.3, listen_timeout_s=30.0):
        self.state = State.IDLE
        self.vad_th = vad_threshold_bargein
        self.timeout = listen_timeout_s
        self.t_enter = time.monotonic()
        self.bargein_latency_ms = None

    def _goto(self, s):
        self.state = s
        self.t_enter = time.monotonic()

    def on_wakeword(self):
        if self.state is State.IDLE:
            self._goto(State.LISTENING)

    def on_vad_endpoint(self):
        if self.state is State.LISTENING:
            self._goto(State.THINKING)

    def on_response_ready(self):
        if self.state is State.THINKING:
            self._goto(State.SPEAKING)

    def on_user_voice(self, energy):
        """SPEAKING 中检测用户开口 -> 打断。energy: 快速通道能量/VAD 分数。"""
        if self.state is State.SPEAKING and energy >= self.vad_th:
            self.bargein_latency_ms = (time.monotonic() - self.t_enter) * 1000
            self._goto(State.LISTENING)          # 直接回去听新指令
            return True                          # 调用方：停 TTS + 清播放队列
        return False

    def tick(self):
        if self.state is State.LISTENING and time.monotonic() - self.t_enter > self.timeout:
            self._goto(State.IDLE)               # 唤醒后不说话 -> 回待机

    def on_finish_speaking(self):
        if self.state is State.SPEAKING:
            self._goto(State.IDLE)

# ---- 单元测试：把状态机跑一遍 ----
sm = DialogStateMachine()
assert sm.state is State.IDLE
sm.on_wakeword();        assert sm.state is State.LISTENING
sm.on_vad_endpoint();    assert sm.state is State.THINKING
sm.on_response_ready();  assert sm.state is State.SPEAKING
assert sm.on_user_voice(0.1) is False           # 低能量不触发打断
assert sm.on_user_voice(0.6) is True            # 打断！
assert sm.state is State.LISTENING
sm.on_vad_endpoint(); sm.on_response_ready()
sm.on_finish_speaking(); assert sm.state is State.IDLE
print("状态机全路径测试通过 ✓（交付：在板端把 on_user_voice 接到快速 VAD 通道）")
```

### 5.3 100 次对话延迟分布（MVP 验收脚本）

```python
import numpy as np

def latency_report(samples_ms, target_p95_ms=1500):
    a = np.asarray(samples_ms, dtype=float)
    assert len(a) >= 100, "至少 100 次对话"
    p50, p95, p99 = np.percentile(a, [50, 95, 99])
    print(f"P50={p50:.0f} ms  P95={p95:.0f} ms  P99={p99:.0f} ms  max={a.max():.0f} ms")
    assert p95 <= target_p95_ms, f"P95 {p95:.0f} ms 超目标，按模块埋点定位超标段"
    return p50, p95

# 示例：交付时替换为板端真实采集的 100 个端到端延迟
rng = np.random.default_rng(0)
demo = rng.normal(1100, 180, 100).clip(600, 2200)
latency_report(demo, target_p95_ms=1500)
print("延迟分布验收框架就绪 ✓")
```

**预期输出**：P50 ≈ 1100、P95 ≈ 1400 左右，断言通过。**验收口径**：每次对话记录 5 段埋点（唤醒→VAD→ASR→LLM 首 token→TTS 首包），P95 超标时直接看哪一段的分布右尾长——这就是第 7-8 周火焰图的数据源。

### 5.4 TTS 分句与边生成边播骨架

```python
import re

def split_sentences(text_stream_chunk: str, pending: str):
    """LLM 流式输出的文本块 -> 可合成的完整句子列表 + 残留。"""
    buf = pending + text_stream_chunk
    parts = re.split(r"(?<=[。！？；，,\.\!\?;])", buf)
    complete, tail = parts[:-1], parts[-1]
    complete = [s for s in complete if s.strip()]
    return complete, tail

# 播放循环（伪码中的真实逻辑）：
# for chunk in llm.stream(prompt):
#     sentences, pending = split_sentences(chunk, pending)
#     for s in sentences:
#         pcm = tts.synthesize(s)                 # 蒸馏小模型，逐句
#         play_queue.put((seq, pcm)); seq += 1    # 打断时按 seq 清空
# 打断回调：play_queue.clear(); tts.stop()

sentences, tail = split_sentences("好的，今天天气晴", "")
sentences2, tail2 = split_sentences("朗，适合出门。", tail)
assert sentences == ["好的，"] and sentences2 == ["今天天气晴朗，适合出门。"]
print("分句逻辑验证通过 ✓")
```

---

## 6. 工程权衡与失效模式

### 6.1 权衡

- **1.5B vs 3B 的决策标准**：不看跑分，看你的 100 条测试集上「答非所问率」差多少——差距 <5 个点就选 1.5B 保延迟；
- **KV cache fp16 vs INT8**：INT8 缓存省一半内存，但长上下文下质量有可见损失——端侧上下文本来就短（<2K），默认 fp16，内存告急再降；
- **打断灵敏度**：阈值低 → 电视声误打断；阈值高 → 用户喊不停。用「打断通道」与「正常识别通道」分离的双阈值：打断通道宁误触发（代价只是停一下），识别通道宁保守；
- **TTS 音色**：蒸馏小模型音色不如原版——在「像真人」与「300 ms 首包」之间，端侧选后者，音色留作后续升级项。

### 6.2 失效模式

1. **OOM 于长对话**：历史不截断，KV cache 线性膨胀。症状：第 6~8 轮崩溃；修复：4 轮滑窗 + 硬上限（4.1 节预算表），写入接口契约并加监控告警。
2. **打断后残留语音**：播放队列清空了但音频驱动缓冲还有半秒数据。症状：喊「停」后还吐出半个词；修复：清空队列后调用声卡 `flush`，并把这 50 ms 算进打断预算。
3. **唤醒后立即说话被截头**：唤醒词引擎切到 ASR 需要几十毫秒，用户紧接着说「今天天气」丢了「今」字。修复：唤醒触发时刻起持续录音缓冲，ASR 从缓冲区起点（而非就绪时刻）开始解码。
4. **prefill 抖动**：同样问题有时 300 ms 有时 700 ms。根因：输入长度差异（历史轮数不同）+ 大核被抢占。修复：历史截断固定长度 + 绑核 + 埋点区分「输入长度」与「单位 token 耗时」两个变量。

---

## 7. 延伸思考题（含解析）

**Q1**：为什么 KV cache 用 GQA（2 个 KV 头）能省内存？对端侧意味着什么？
**A**：GQA 让多个 query 头共享一组 K/V，KV cache 与 KV 头数成正比——Qwen2.5 用 2 个 KV 头而非 28 个，KV 内存直接降到 1/14。对端侧：同等内存预算下上下文长 14 倍，这就是「端侧选模型先看 KV 结构」的原因。

**Q2**：TTS 为什么不能等 LLM 全部生成完再合成？两种延迟差多少？
**A**：整句等待时用户沉默时间 = LLM 全部生成 + TTS 全句合成；流式时 ≈ 第一句生成 + 第一句合成。一句 50 字的回答，前者约 2~4 s，后者约 400~600 ms——差的就是「听助手开口」的生死线。代价是句间韵律可能不连贯（靠双缓冲与轻量拼接缓解）。

**Q3**：打断检测为什么不能复用 ASR？
**A**：ASR 要等一段话说完或至少几百毫秒缓冲才能识别，打断要的是「开口即停」（≤200 ms）。所以打断用低阈值能量+VAD 快速通道：它不需要知道用户说了什么，只需要知道用户在说话。误触发的代价（停一下）远小于慢触发的代价（喊不停）。

**Q4**：llama.cpp 的 Q4_K_M 与纯 INT4（Q4_0）差在哪？端侧怎么选？
**A**：Q4_K_M 是分块混合量化（敏感层高精度 + 4bit 主体 + 量化 scale 精细化），质量损失显著小于 Q4_0，体积只大 ~10%。端侧内存够就选 Q4_K_M——质量-内存曲线上它几乎总是帕累托点。

**Q5**：100 次对话测出 P95 达标但 max 超过 3 s，要处理吗？
**A**：要。长尾的 max 往往对应具体可定位的事件（温度降频、GC、历史轮数最满的一次）。处理路径：埋点找出最慢那 5 次的共同特征，归入第 7-8 周的压测清单。产品上用户对「偶尔一次特别慢」的记忆远强于平均水平。

---

## 本周交付清单

- [ ] 板端跑通「唤醒 → 听 → 想 → 说」完整闭环，录一段 3 轮对话演示。
- [ ] 100 次对话延迟分布表：P50/P95/P99/max，P95 ≤1.5 s 达标或给出超标段的归因。
- [ ] 打断功能上线：实测「开口 → TTS 静音」截断延迟 ≤200 ms（附 5.2 状态机测试记录）。
- [ ] KV cache 实测内存与 5.1 计算值对账（误差 <10%）。
- [ ] 1.5B vs 3B 决策记录：同测试集答非所问率 + 延迟对比，写明最终选型。
