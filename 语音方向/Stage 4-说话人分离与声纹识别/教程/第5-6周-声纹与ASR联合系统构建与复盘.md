# 第 5-6 周教程：声纹 + ASR 联合系统——构建会议转写系统

> **本周要回答的三个问题**
> 1. ASR 的句子时间戳与分离的说话人标签如何对齐成"带说话人的逐句文本"？
> 2. 声纹库（注册 + 检索）如何在百万级底库上做到毫秒级检索？
> 3. 在线/流式分离与离线分离的差别是什么？增量聚类怎么做？

对应学习计划：第 5-6 周。交付物：构建「会议转写系统」——输入多人录音，输出带说话人标签的逐句文本；支持注册已知说话人声纹库，未知人标为 "Speaker X"；评估句级说话人准确率，并接入 Stage 3 前端验证增强收益。

---

## 1. 第一性原理：把"说了什么"和"谁说的"焊在一起

### 1.1 两条独立信息流

- **ASR（Stage 2）**：输出 `(文字, 起始时间, 结束时间)` 的句子序列；
- **分离（Stage 4）**：输出 `(说话人, 起始时间, 结束时间)` 的标签序列。

两者**时间轴对齐但来源不同**。联合系统的核心就是**按时间把句子归给说话人**。

### 1.2 对齐策略：多数投票

对每个 ASR 句子，看它的时间区间 `[s, e]` 落在哪些分离区间里，用**时间加权多数投票**决定句子的说话人：

$$
\text{speaker}(句) = \arg\max_{spk} \; \text{overlap}(句, spk\text{ 的区间})
$$

即：哪个说话人的分离区间与这个句子的重叠时间最长，句子就归谁。

**为什么用时间加权而非简单计数**：一句话可能跨两个分离段（如中途换人），按重叠时长加权比按段数计数更鲁棒。

---

## 2. 系统架构

$$
\text{录音} \xrightarrow{\text{Stage 3 前端}} \xrightarrow{\text{ASR（时间戳）}} \xrightarrow{\text{分离}} \xrightarrow{\text{对齐}} \xrightarrow{\text{声纹库匹配}} \text{带说话人文本}
$$

五个组件：

1. **前端**（Stage 3）：AEC+NS+AGC+VAD，产出干净语音；
2. **ASR**（Stage 2）：SenseVoice/Paraformer，输出带时间戳的句子；
3. **分离**（Stage 4）：pyannote，输出说话人区间；
4. **对齐**：时间加权投票，句子 ↔ 说话人；
5. **声纹库**：把匿名的 "speaker_0" 映射到已知姓名（若有注册），否则保留 "Speaker X"。

---

## 3. 实现与验证（交付核心）

### 3.1 对齐函数

```python
import numpy as np

def align_sentences_to_speakers(sentences, diar_turns):
    """
    sentences: [{"text", "start", "end"}, ...]  ASR 输出
    diar_turns: [(start, end, speaker), ...]     分离输出
    返回每句的说话人（时间加权多数投票）。
    """
    result = []
    for sent in sentences:
        s, e = sent["start"], sent["end"]
        vote = {}
        for d_start, d_end, spk in diar_turns:
            # 计算句子区间与分离区间的重叠时长
            overlap = max(0.0, min(e, d_end) - max(s, d_start))
            if overlap > 0:
                vote[spk] = vote.get(spk, 0.0) + overlap
        if not vote:
            speaker = "UNKNOWN"
        else:
            speaker = max(vote, key=vote.get)
        result.append({**sent, "speaker": speaker})
    return result
```

### 3.2 声纹库：注册与检索

```python
import numpy as np

class SpeakerRegistry:
    """注册已知说话人声纹，把匿名标签映射到姓名。"""
    def __init__(self, threshold=0.70):
        self.db = {}            # name -> 声纹向量
        self.threshold = threshold

    def register(self, name, embedding):
        self.db[name] = embedding / (np.linalg.norm(embedding) + 1e-9)

    def identify(self, embedding):
        """返回最匹配的已知名，或 None（未知人）。"""
        e = embedding / (np.linalg.norm(embedding) + 1e-9)
        best_name, best_sim = None, -1
        for name, ref in self.db.items():
            sim = float(np.dot(e, ref))
            if sim > best_sim:
                best_sim, best_name = sim, name
        if best_sim >= self.threshold:
            return best_name
        return None
```

**生产升级**：底库上百万时用 **FAISS**（近似最近邻，ANN），把线性扫描换成毫秒级检索——这正是学习计划点名的技术。

### 3.3 完整会议转写系统

```python
from funasr import AutoModel
from pyannote.audio import Pipeline

class MeetingTranscriber:
    def __init__(self, hf_token, registry=None):
        self.asr = AutoModel(model="iic/SenseVoiceSmall")
        self.diar = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        self.registry = registry

    def transcribe(self, wav_path):
        # 1. ASR（带时间戳）
        asr_res = self.asr.generate(input=wav_path, output_timestamp=True)
        sentences = self._parse_asr(asr_res)   # -> [{text, start, end}]

        # 2. 分离
        diar = self.diar(wav_path)
        turns = [(t.start, t.end, spk)
                 for t, _, spk in diar.itertracks(yield_label=True)]

        # 3. 对齐
        tagged = align_sentences_to_speakers(sentences, turns)

        # 4. 声纹库映射（匿名 -> 姓名）
        if self.registry:
            for item in tagged:
                emb = self._segment_embed(wav_path, item["start"], item["end"])
                name = self.registry.identify(emb)
                if name:
                    item["speaker"] = name
        return tagged

    def _parse_asr(self, res):
        # 把 FunASR 的时间戳输出解析成 [{text, start, end}]
        # 具体字段以实际返回为准
        return []

    def _segment_embed(self, wav_path, start, end):
        # 抽取该句片段，用 ECAPA 提声纹（复用 Stage 4 第 1 周代码）
        return None
```

### 3.4 句级说话人准确率评估

```python
def sentence_speaker_accuracy(tagged, ground_truth):
    """
    tagged: 系统输出（每句带 speaker）
    ground_truth: 人工标注（每句的正确说话人）
    """
    correct = sum(1 for t, g in zip(tagged, ground_truth)
                  if t["speaker"] == g["speaker"])
    return correct / len(ground_truth)
```

**交付指标**：句级说话人准确率（sentence-level speaker accuracy）。目标 ≥ 85%（干净双人会议）。

### 3.5 接入前端验证增益（联动 Stage 3）

```python
# 对比：原始录音 vs Stage 3 前端增强后，转写系统的准确率
acc_raw = evaluate(raw_wav)
acc_enh = evaluate(frontend_enhanced_wav)
print(f"原始: {acc_raw:.2%}  前端增强后: {acc_enh:.2%}")
# 预期：增强后准确率上升（降噪让声纹更稳、ASR 更准）
```

这张消融表回答学习计划的问题："**前端对分离/转写质量的提升**"。

---

## 4. 流式分离与在线聚类

### 4.1 离线 vs 在线

- **离线**：看完整段音频再聚类（本周方案）；
- **在线/流式**：音频边来边处理，实时输出说话人标签——会议实时字幕需要。

### 4.2 增量聚类

在线时不能等全部数据再聚类，用**增量聚类**：

```
对每个新片段的声纹 e：
    与已有簇中心比相似度
    if 最高相似度 > 阈值:
        归入该簇，更新簇中心
    else:
        新建一个簇（新说话人）
```

**难点**：新说话人何时建簇（阈值）、簇中心如何在线更新、说话人离开后簇是否保留。这是实时系统的核心工程。

---

## 5. 隐私合规（面试加分项）

声纹是**生物特征**，受《个人信息保护法》/ GDPR 严格约束：

- **采集需明示同意**：注册声纹前必须获得授权；
- **存储加密**：声纹向量加密存储，不可逆推原始语音；
- **注销权**：用户可要求删除其声纹；
- **最小化**：只存必要数据，定期清理。

面试中被问"声纹系统怎么合规"，答到这四点就是加分项。

---

## 6. 工程权衡与失效模式

### 6.1 权衡

- **对齐粒度**：句子级投票简单稳定；逐帧对齐更准但复杂。句子级够用；
- **声纹库阈值**：高 → 少误识但多"未知人"；低 → 反之。按业务定（联动第 2 周校准）；
- **在线聚类阈值**：影响"新建说话人"的灵敏度。

### 6.2 失效模式

1. **时间戳错位**：ASR 与分离的时钟基准不一致，对齐全错。修复：统一时间轴。
2. **短句归错**：一句话太短，落在分离边界上被归错人。修复：时间加权 + 上下文平滑。
3. **声纹库误识**：阈值太低把陌生人认成已知人。修复：提高阈值、加校准。
4. **重叠句丢失**：两人同时说话的句子被归给一个人。修复：标注"多说话人"，或端到端。

---

## 7. 延伸思考题（含解析）

**Q1**：ASR 时间戳与分离标签如何对齐？为什么用时间加权投票？
**A**：对每个句子，计算它与各说话人分离区间的重叠时长，归给重叠最长者。用时间加权是因为句子可能跨分离段，按时长比按段数更鲁棒。

**Q2**：百万级声纹库怎么做到毫秒检索？
**A**：用 FAISS 的近似最近邻（如 IVF + PQ 量化），把线性扫描降为亚线性。先粗筛候选簇，再精排——牺牲极小精度换几个数量级的速度。

**Q3**：你做过最难的分离 case 是什么？（面试叙事题）
**A**：（填你的实测）典型难题：① 频繁短发言的说话人被漏；② 重叠密集段混淆；③ 信道差导致同人声纹漂移。叙述时给出诊断（DER 分解）与你的应对。

**Q4**：在线分离为什么比离线难？
**A**：离线能看全局做最优聚类；在线只能增量决策——新说话人何时建簇、簇中心如何更新、无法回看修正。每个决策都不可逆，误差会累积。

**Q5**：声纹系统上线要考虑哪些合规点？
**A**：采集明示同意、存储加密、用户注销权、数据最小化与定期清理。生物特征是高敏感数据，合规是产品能否上线的前提。

---

## 本周交付清单

- [ ] 实现 `align_sentences_to_speakers()`，完成句子 ↔ 说话人对齐。
- [ ] 实现 `SpeakerRegistry`，注册 3 个已知人，未知人标 "Speaker X"。
- [ ] 组装 `MeetingTranscriber`，跑通多人录音 → 带说话人文本。
- [ ] 评估句级说话人准确率，并接入 Stage 3 前端验证增益。
- [ ] 能解释在线增量聚类与隐私合规四要点。

## Stage 4 总结

完成本阶段，你已能：

1. **原理**：池化定长表征、ArcFace 度量学习、EER/minDCF、PIT；
2. **实现**：ECAPA 声纹、DET 曲线、谱聚类分离、pyannote 流水线、会议转写系统；
3. **系统**：声纹 + ASR 联合，含声纹库与合规意识。

对照学习计划自测清单核验后，进入 **Stage 5：语音生成与零样本克隆**——方向从"听懂"转向"说出来"。
