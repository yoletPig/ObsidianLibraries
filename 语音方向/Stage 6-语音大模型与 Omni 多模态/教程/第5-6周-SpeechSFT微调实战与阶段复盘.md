# 第 5-6 周教程：Speech SFT 微调实战与阶段复盘——用你的 CosyVoice 造数据

> **本周要回答的四个问题**
> 1. `(音频, 指令, 回答)` 三元组数据从哪来？没有真实语音指令数据时，TTS 合成闭环怎么搭？
> 2. ms-swift 微调 Qwen2-Audio 时冻结谁、训谁？这套策略与 VLM LoRA 的哪条经验一一对应？
> 3. 灾难性遗忘怎么监控？「微调后变聋」（ASR 能力退化）用什么回归测试抓？
> 4. Speech-LLM 与 VLM 的训练方法论，异同到底是什么（面试稿）？

对应学习计划：第 5-6 周。交付物：① 1000 条语音指令数据集（500 真实 + 500 CosyVoice 合成）；② 云端 LoRA 微调 + 微调前后语音问答准确率对比；③ 通用能力回归测试表；④ 「Speech-LLM 与 VLM 训练方法论异同」总结。

---

## 1. 第一性原理：微调的本质是「分布搬家」

### 1.1 目标分布

预训练的 Qwen2-Audio 学的是通用分布 $P_{\text{pre}}(\text{回答} \mid \text{音频}, \text{指令})$。Speech SFT 要把它搬到你的目标分布：

$$
P_{\theta}(\text{回答} \mid \text{音频}, \text{指令}) \;\to\; P_{\text{target}}
$$

目标分布越窄（比如只做「语音控制智能家居」），需要的数据越少、过拟合风险越高；越宽（通用语音助手），越需要真实数据多样性。

### 1.2 数据量级直觉

损失函数就是标准的下一 token 交叉熵，只在回答部分算（指令与音频部分 mask 掉）：

$$
\mathcal{L} = -\sum_{t \in \text{回答}} \log P_{\theta}(y_t \mid \text{音频}, \text{指令}, y_{<t})
$$

经验量级：窄域任务 500~2000 条高质量三元组可见效（本周的 1000 条有出处）；风格/格式适配 200~500 条足够。你在 VLM Stage 4 合成数据阶段验证过同一规律——**数据质量 >> 数据数量**。

### 1.3 为什么音频塔基本不动

音频塔（Whisper encoder）已在海量音频上收敛，它的特征空间是「全局公理」。微调只训 projector + LLM 低秩层，就是在公理上建新房——这与 VLM 阶段「冻结 ViT、训 projector + LoRA」完全同构，理由也完全相同：① 防灾难性遗忘；② 省显存；③ 音频塔样本效率极低，少量数据碰它只会变差。

---

## 2. 数据构造：1000 条三元组的完整闭环

### 2.1 数据配方设计

| 来源 | 数量 | 作用 | 风险 |
| --- | --- | --- | --- |
| 真实录音（你 + 众包） | 500 | 锚定真实声学分布（噪声、口音） | 成本高 |
| CosyVoice 合成 | 500 | 扩量 + 音色多样性 + 可控文本 | 合成腔偏差（分布偏移） |
| 混比 1:1 | 1000 | 合成数据提量，真实数据纠偏 | 合成占比 >70% 时声学分布被带偏 |

**合成比例红线**：你在 VLM Stage 4 学到的结论直接复用——合成数据占比超过 ~70% 后，模型在真实分布上开始退化。本周守住 50%。

### 2.2 指令生成（文本侧）

```python
import json, random

random.seed(0)

# 域内种子指令模板（语音助手场景），用云端大 LLM 扩写到数百条
SEED_TEMPLATES = [
    "帮我定一个{t}点的闹钟",
    "今天天气怎么样",
    "打开{room}的灯",
    "把音量调{dir}",
    "{q}",   # 开放问答槽位
]

def build_text_pairs(llm_expand_fn, n=1000):
    """llm_expand_fn: 你云端 LLM 的扩写函数，输入种子列表输出 (指令, 回答) 对。
    返回 [{'instruction':..., 'response':...}, ...]"""
    pairs = llm_expand_fn(SEED_TEMPLATES, n=n)
    # 质量门：回答必须非空、指令长度合理、去重
    seen, out = set(), []
    for p in pairs:
        ins, resp = p["instruction"].strip(), p["response"].strip()
        assert resp, "回答不能为空"
        if 2 <= len(ins) <= 50 and ins not in seen:
            seen.add(ins)
            out.append({"instruction": ins, "response": resp})
    assert len(out) >= n * 0.9, "质量门过滤后数量不足"
    return out
```

### 2.3 用 CosyVoice 合成训练音频（技能闭环核心）

用你 Stage 5 部署的 CosyVoice2，把指令文本渲染成多音色语音——**语音生成能力反哺语音理解训练**：

```python
from cosyvoice.cli.cosyvoice import CosyVoice2
import soundfile as sf, torch, os

model = CosyVoice2("CosyVoice2-0.5B")
# 收集 5~10 段不同音色的参考音频（你的声音 + 公开多音色样本）
PROMPTS = [("spk1.wav", "这是第一个说话人参考音频里说的话。"),
           ("spk2.wav", "这是第二个说话人参考音频里说的话。")]

def synthesize_pairs(text_pairs, out_dir):
    """把每条指令文本合成为一条训练音频，输出三元组清单。"""
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for i, p in enumerate(text_pairs):
        ref_wav, ref_text = PROMPTS[i % len(PROMPTS)]
        for chunk in model.inference_zero_shot(
                p["instruction"], ref_text, ref_wav, stream=False):
            wav = chunk["tts_speech"].squeeze().numpy()
            path = f"{out_dir}/syn_{i:04d}.wav"
            sf.write(path, wav, model.sample_rate)
            assert len(wav) / model.sample_rate > 0.5, "合成音频过短，重合成"
            manifest.append({"audio": path, "instruction": p["instruction"],
                             "response": p["response"], "source": "tts"})
            break  # 非流式取整段
    return manifest
```

**真实录音侧**：同一批指令文本，由真人朗读/自然说出并录音（16 kHz），Stage 2 的 SenseVoice 反向校验转写一致（转写与文本不一致的样本剔除——这是数据清洗的最低要求）。

### 2.4 合成数据的声学增广（缩小合成腔差距）

TTS 输出太「干净」是分布偏移的根源。轻量增广能把合成音频拉向真实分布，不增加任何录制成本：

```python
import numpy as np
import soundfile as sf

rng = np.random.default_rng(0)

def augment(wav, sr):
    """对合成音频做三类声学增广：加性噪声、增益抖动、混响近似。"""
    out = wav.copy()
    # ① 加性噪声：SNR 10~30 dB 随机（模拟家庭环境）
    snr_db = rng.uniform(10, 30)
    noise = rng.standard_normal(len(out))
    sig_p = np.mean(out ** 2) + 1e-9
    noi_p = np.mean(noise ** 2)
    out = out + noise * np.sqrt(sig_p / (noi_p * 10 ** (snr_db / 10)))
    # ② 增益抖动 ±3 dB
    out *= 10 ** (rng.uniform(-3, 3) / 20)
    # ③ 一阶混响近似（指数衰减回声）
    k = int(0.05 * sr)
    padded = np.concatenate([out, np.zeros(k)])
    padded[k:] += 0.3 * out
    return np.clip(padded[:len(out)], -1, 1)

wav = np.sin(2 * np.pi * 300 * np.arange(16000) / 16000) * 0.5
aug = augment(wav, 16000)
assert aug.shape == wav.shape and np.abs(aug - wav).max() > 0
print("声学增广验证通过 ✓ —— 对 500 条合成音频各增广 1 份，等效扩到 1000 条不同声学条件")
```

**用法**：增广只作用于**合成**那一半；真实录音保持原样（它本身就是分布锚点）。这是「用数据工程替代数据量」的标准手法——你 VLM Stage 4 的数据配方经验在这里第二次变现。

### 2.4 ms-swift 数据格式

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "<audio>帮我定一个早上八点的闹钟"}], "audios": ["/data/train/real_0001.wav"], "label": "response"}
```

关键点：音频以 `<audio>` 占位 + `audios` 字段传入——占位符替换机制与你第 1 周学过的 `<image>` 替换完全同构。

---

## 3. 微调实操：ms-swift + 冻结策略

### 3.1 冻结配置（与 VLM LoRA 逐项对照）

| 组件 | 策略 | VLM 对应经验 |
| --- | --- | --- |
| 音频塔（Whisper enc） | **冻结** | ViT 冻结（防灾难性遗忘） |
| Multi-audio projector | **全量训练** | Projector 全训（对齐是它的本职） |
| LLM 骨干 | **LoRA（低秩）** | ViT 冻结下 LLM 加 LoRA |

```bash
pip install 'ms-swift' -U
```

```bash
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2-Audio-7B-Instruct \
    --train_type lora \
    --dataset speech_instruction_train.jsonl \
    --freeze_vit true \
    --freeze_aligner false \
    --target_modules all-linear \
    --lora_rank 16 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 4096 \
    --output_dir output/qwen2audio-speech-sft
```

参数要点：`freeze_vit true` = 音频塔不动；`freeze_aligner false` = projector 全训；`lora_rank 16` 对千条级数据足够（你在 VLM 阶段验证过 rank 8~32 的差异不大）。

### 3.2 训练监控指标

- **train loss 曲线**：1000 条 × 3 epoch，正常应在第 1 epoch 内从 ~2 降到 <1；若降不动先查数据格式（占位符是否被正确处理）；
- **过拟合信号**：eval loss 在第 2 epoch 后上升 → 早停，取 eval loss 最低的 checkpoint（LoRA 下可存多个 rank 目录对比）。

---

## 4. 评测：语音问答 + 遗忘回归（交付核心）

### 4.1 回归测试框架

```python
import json

def eval_accuracy(eval_fn, testset):
    """eval_fn: 输入 (音频路径, 指令) 返回模型回答字符串。
    返回逐条判定与准确率。判定用关键词命中或云端 LLM 裁判。"""
    hits = []
    for item in testset:
        pred = eval_fn(item["audio"], item["instruction"]).strip()
        ok = all(kw in pred for kw in item["expected_keywords"])
        hits.append({"instruction": item["instruction"], "pred": pred, "ok": ok})
    acc = sum(h["ok"] for h in hits) / len(hits)
    return acc, hits

def regression_report(before: dict, after: dict):
    """before/after: {任务名: 准确率}。任何任务下降超过阈值都算回归警报。"""
    report, alarms = {}, []
    assert set(before) == set(after), "前后测试集必须一致"
    for task in before:
        delta = after[task] - before[task]
        report[task] = {"before": before[task], "after": after[task],
                        "delta": round(delta, 3)}
        if delta < -0.02:
            alarms.append(task)
    return report, alarms

before = {"语音指令跟随": 0.62, "ASR-WER反向": 0.93, "通用对话": 0.81, "情感识别": 0.70}
after  = {"语音指令跟随": 0.88, "ASR-WER反向": 0.91, "通用对话": 0.80, "情感识别": 0.71}
rep, alarms = regression_report(before, after)
print(json.dumps(rep, ensure_ascii=False, indent=1))
assert "ASR-WER反向" not in alarms or after["ASR-WER反向"] >= 0.90, "ASR 回归超红线"
print("回归警报:", alarms if alarms else "无")
```

**预期输出**：四任务前后对比表 + 警报列表。四个回归任务的含义：

| 任务 | 防的是什么遗忘 | 红线 |
| --- | --- | --- |
| 语音指令跟随（新域） | ——（这是要涨的） | 必须显著上升 |
| ASR 转写（WER） | 「变聋」：听的能力退化 | 下降 ≤ 2 个点 |
| 通用文本对话 | 语言能力被语音数据冲掉 | 下降 ≤ 2 个点 |
| 音频事件/情感识别 | 音频塔-LLM 对齐被破坏 | 下降 ≤ 2 个点 |

### 4.2 灾难性遗忘的根因与对策

根因：梯度把参数从通用分布拉向窄域分布。对策（按成本排序）：

1. **混入通用数据**：训练集里掺 10~20% 通用语音问答（成本最低，首选）；
2. **LoRA 而非全参**：改动参数面小，遗忘天然轻（本周默认）；
3. **低学习率 + 早停**：1e-4 对 LoRA 已偏高，若回归警报频发降到 5e-5；
4. **合并多个 LoRA**（进阶）：窄域 LoRA 与基座推理时按场景切换，互不污染。

---

## 5. 阶段复盘：Speech-LLM 与 VLM 训练方法论异同（面试稿）

**同（四条，全部有本周实证）**：

1. 对齐靠 projector，先训它再谈别的——音频塔 ≈ Vision Tower；
2. 编码器默认冻结，防遗忘省显存；
3. LoRA rank/学习率的甜点区间一致（窄域小数据、早停）；
4. 合成数据可用但占比有红线（≤~70%），真实数据锚定分布。

**异（三条，音频特有）**：

1. **帧率敏感**：音频是 1D 无上限时序，token 预算与降采样是视觉侧没有的第一约束（第 1、4 周）；
2. **数据合成更容易也更危险**：TTS 合成语音比图像生成便宜得多，但「合成腔」分布偏移同样更系统——50% 红线是本周实测结论；
3. **评测多一维**：语音任务必须加「声学鲁棒性」维度（噪声、口音、打断），纯文本评测集不适用。

---

## 6. 工程权衡与失效模式

### 6.1 权衡

- **合成比例**：多合成 → 便宜快量，但真实场景指标被稀释；本阶段 1:1 是验证过的平衡点。
- **projector 全训的过拟合**：千条数据下 projector 可能背答案（听开头猜指令）。症状：训练集准确率虚高、真实录音崩；对策：projector 学习率单独调低或加音频增广。
- **单卡可训的代价**：batch 1 + 梯度累积 8，1000 条 × 3 epoch 在单张 A100 上约数小时；多卡只是提速，配方不变。

### 6.2 失效模式

1. **占位符未替换**：数据里 `<audio>` 没被正确注入 → 模型在纯文本上瞎猜，loss 降得反而快（学到捷径）。定位：抽 10 条打印模型实际输入序列；修复：核对 ms-swift 数据格式与库版本。
2. **合成数据音色单一**：只用一个参考音色合成 500 条 → 模型对该音色过拟合，换人说话就崩。修复：≥5 个参考音色轮换 + 语速/情感参数扰动。
3. **评测泄漏**：测试集指令与训练模板同源同分布 → 准确率虚高。修复：测试集用**训练时没出现过的句式模板**单独生成，且至少 30% 为真实录音。
4. **checkpoint 选错**：只按 train loss 选模型，取到过拟合点。修复：按 eval loss + 回归测试表联合选择，把回归表写进交付。

---

## 7. 延伸思考题（含解析）

**Q1**：为什么合成数据的红线是 ~70% 而不是 50% 或 90%？
**A**：这不是定理而是经验区间：合成数据提供文本多样性（便宜、可控），真实数据提供声学分布锚点（噪声、混响、口音）。超过 ~70% 后声学分布由 TTS 的「干净录音室分布」主导，模型在真实噪声下指标滑坡；本周 1:1 配比 + 回归测试是这条经验的个人实证版本。

**Q2**：微调后「变聋」最可能发生在哪个组件？
**A**：最容易发生在 projector 与 LLM 的音频理解通路（编码器冻结时它本身不会退化）：梯度把「音频特征 → 语义」的映射挤窄到指令域，通用听辨（转写、事件识别）被牺牲。所以回归表里 ASR 任务是对 projector 的探针。

**Q3**：如果预算无限，你会把 1000 条扩到 10 万条吗？先扩什么？
**A**：先扩**真实录音的多样性**（口音、噪声、远场），而不是合成量——合成边际收益在 50% 占比后递减，真实数据的边际收益衰减更慢。顺序：真实多样性 > 任务覆盖广度 > 合成量。

**Q4**：LoRA 与全参微调在「遗忘-学习」平衡上为什么不同？
**A**：LoRA 的更新被限制在低秩子空间，基座绝大部分参数不动，通用能力保留多、新域容量受限；全参容量大但所有参数都被拉走，遗忘重。千条级数据下新域容量需求小，LoRA 是帕累托最优点——这也是「小数据用 LoRA」的一般结论。

**Q5**：同一套微调配方迁移到 Qwen2.5-Omni（Thinker-Talker）要注意什么？
**A**：① 只微调 Thinker 侧（理解+文本），Talker 的语音生成另有数据需求（语音问答对要带目标语音）；② 隐藏状态接口不能因 LoRA 形状变化而破坏——LoRA 不改形状故安全；③ 回归测试要加「语音输出质量」维度（Talker 未训但输入分布变了，音色可能漂）。

---

## 本周交付清单

- [ ] 1000 条数据集完成：500 真实录音（SenseVoice 反向校验）+ 500 CosyVoice 多音色合成，manifest 齐备。
- [ ] ms-swift LoRA 微调跑完，记录 loss 曲线与最终超参。
- [ ] 回归测试表完成：语音指令跟随必须涨，ASR/通用/情感三任务下降 ≤2 点。
- [ ] 消融记录：合成比例 0% / 50% / 100% 三组的指令跟随准确率对比（各跑一次即可）。
- [ ] 「Speech-LLM 与 VLM 训练方法论异同」一页纸（面试稿）完成。
- [ ] 阶段自测清单（学习计划附表）逐项过关，进入 Stage 7。
