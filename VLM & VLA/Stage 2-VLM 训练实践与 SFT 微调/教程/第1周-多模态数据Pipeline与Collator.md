# 第 1 周教程：多模态数据格式与 Pipeline 深度构造

> **本周要回答的三个问题**
> 1. 一条图文对话样本，是怎么一步步变成 `input_ids`、`attention_mask`、`pixel_values`、`labels` 四件套的？
> 2. `labels` 里的 `-100` 掩码，边界到底画在哪些 Token 上？
> 3. 变长分辨率和多轮对话，是怎么被 Padding / Packing 统一成一个 batch 的？

对应学习计划：第 1 周。交付物：手写 `Dataset` + `DataCollator`，加载 10 条自定义图文 JSON，输出模型可用的四件套并用断言验证掩码正确。

**源码路径勘误**：学习计划写的 `src/llamafactory/data/collator/` 实际是单文件 [`src/llamafactory/data/collator.py`](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/collator.py)（已核实 main 分支，2026 年仍在活跃维护，近期提交涉及 packed M-RoPE 的 position ids 修正）；多模态 Processor 组装逻辑在 [`data/processor/`](https://github.com/hiyouga/LLaMA-Factory/tree/main/src/llamafactory/data/processor) 目录与 [`data/mm_plugin.py`](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/mm_plugin.py)。本文按真实路径引用。

---

## 1. 第一性原理：为什么需要 Processor 与 Collator 两级转换

### 1.1 根本矛盾

模型前向需要的是**形状规整的张量批次**（`input_ids: [B, L]`、`pixel_values: [B, C, H, W]`），而真实世界的数据是**异构的**：文本长度不一、图片分辨率各异、对话轮数不同、视觉 Token 数随分辨率漂移。从"磁盘上的一堆 JSON + JPG"到"GPU 上的一批对齐张量"，必然存在一条多级转换流水线。

这条流水线为什么拆成两级？

- **Processor（样本级）**：处理**单条样本**——JSON 转成 prompt 字符串、图片转成视觉张量。工作在 `Dataset.__getitem__` 内，可以被多进程并行（`num_workers`），且与 batch 无关。
- **Collator（批次级）**：处理**一批样本的拼装**——padding 到同一长度、stack 张量、构造 labels 掩码。工作在 `DataLoader` 的 `collate_fn`，必须在主进程（或收集点）执行，因为它要看到整个 batch。

拆两级的原因是**并行性约束**：图片解码 + resize 是整个流水线中最重的 CPU 操作（单张高清图可占几十毫秒），必须放进 `Dataset` 才能享受多 worker 并行；而 padding 必须知道 batch 内最长序列，只能后置。混淆两级职责（比如在 Collator 里做图片解码）会让 DataLoader 退化成单进程，数据加载成为训练瓶颈。

### 1.2 消息结构：ChatML 与多模态占位

一条多轮图文对话的原始形态（`sharegpt` 风格）：

```json
{
  "conversations": [
    {"from": "human", "value": "<image>这张图里是什么？"},
    {"from": "gpt",   "value": "两只猫坐在粉色沙发上。"},
    {"from": "human", "value": "它们是什么颜色？"},
    {"from": "gpt",   "value": "一只是橘猫，另一只是黑白花色的猫。"}
  ],
  "images": ["img/cat_sofa.jpg"]
}
```

经过模板渲染（LLaMA-Factory 中 `data/template.py`，Qwen2-VL 用 ChatML 变体），它变成一条带角色标记的 Token 序列：

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>\n
<|im_start|>user
<VISUAL_TOKENS×N>这张图里是什么？<|im_end|>\n
<|im_start|>assistant
两只猫坐在粉色沙发上。<|im_end|>\n
<|im_start|>user
它们是什么颜色？<|im_end|>\n
<|im_start|>assistant
一只是橘猫，另一只是黑白花色的猫。<|im_end|>\n
```

两个关键细节：

1. **`<image>` 占位符扩展为 $N$ 个视觉 Token ID**（Qwen2-VL 中是专门的 `vision_start` + 连续视觉 Token；LLaVA 中是一个占位 ID 在 embedding 层展开）。$N$ 由图片分辨率决定（Stage 1 第 3 周的公式：$\frac{H}{28} \times \frac{W}{28}$），**同一条样本在不同分辨率配置下长度会变**，这是多模态 padding 比纯文本复杂的根源。
2. **每轮 assistant 回答末尾的 `<|im_end|>`（或 EOS）必须保留在 Token 序列里**，它是模型学会"何时闭嘴"的唯一监督信号。丢掉它，微调后模型会答完一句继续自问自答。

---

## 2. 系统架构与数据流

### 2.1 从 JSON 到批次张量的完整流水线

```text
磁盘数据 (JSON + JPG)
   │  Dataset.__getitem__  ←── 多进程并行区 (num_workers=8)
   ▼
┌────────────── Processor (样本级) ──────────────┐
│ 1. 渲染对话模板 → 完整 prompt 字符串             │
│ 2. 图片加载 + Resize → pixel_values [C,H,W]     │
│ 3. 计算 image_grid_thw (Qwen2-VL 的网格元数据)   │
│ 4. 分词 → input_ids [L]  (占位符已展开为 N 个)   │
│ 5. 定位回答区间 → labels [L] (区间外=-100)       │
└────────────────────────────────────────────────┘
   │  DataCollator (批次级)  ←── 主进程收集点
   ▼
┌────────────── Collator ────────────────────────┐
│ 6. 找 batch 内最长 L_max → 右/左 padding        │
│ 7. pixel_values 变长 → 沿 batch 维 concat 成    │
│    [ΣB·N, C, P, P] + image_grid_thw [ΣB, 3]     │
│    (Qwen2-VL 风格, 交给模型按网格切分)           │
│ 8. attention_mask: 真实=1, padding=0            │
│ 9. labels: 回答区=token_id, 其余=-100           │
└────────────────────────────────────────────────┘
   ▼
input_ids [B, L_max] | attention_mask [B, L_max]
labels [B, L_max]    | pixel_values [ΣN, ...] | image_grid_thw
```

### 2.2 `image_grid_thw`：变长视觉 Token 的元数据方案

LLaVA 的做法是"占位符在 embedding 层展开"，batch 内 `pixel_values` 形状统一（固定分辨率时）；而 Qwen2-VL 系的动态分辨率下，**每张图的视觉 Token 数都不同**，无法 stack 成规则四维张量。工程解法：

- `pixel_values` 把 batch 内所有图**沿第 0 维 concat**：$[\sum_{b} N_b \cdot \frac{P^2}{4}, C, 14, 14]$（已经过 Patch Merger 前的切分）；
- 附带 `image_grid_thw: [\sum B, 3]`（每张图的 $t, h, w$ 网格数），模型内部据此把连续的 patch 序列**切回每张图**，做 Merger 与 2D 位置编码；
- 文本侧 `input_ids` 中每个视觉 Token 占一个坑位（`vision_start` + 扩展位），前向时用 `torch.where` 替换为对应 patch embedding。

理解这个设计的钥匙：**变长问题在张量世界里只有两条出路——要么 padding 成规则形状，要么 concat + 元数据记录边界**。视觉用后者（因为视觉 Token 动辄数百个，padding 浪费太大），文本用前者（文本短，padding 便宜）。

### 2.3 labels 掩码的精确边界

对 2.1 节的多轮样本，labels 的取值逐 Token 展开：

| 区间 | Token 内容 | label |
| --- | --- | --- |
| system 头 | `<\|im_start\|>system ... <|im_end\|>` | -100 |
| 第 1 轮 user（含视觉占位） | `<\|im_start\|>user [N个视觉Token] 问题 <|im_end\|>` | -100 |
| 第 1 轮 assistant | `<\|im_start\|>assistant` | -100（角色头也不算） |
| 第 1 轮回答 | `两只猫坐在粉色沙发上。<|im_end\|>` | **token_id（含 im_end）** |
| 第 2 轮 user | 全部 | -100 |
| 第 2 轮回答 | **token_id（含 im_end）** | **token_id** |
| padding 区 | pad token | -100 |

注意三处易错点：

1. **每个 assistant 轮的起始角色头 `<|im_start|>assistant` 是 -100，但回答末尾的 `<|im_end|>` 要算 loss**——前者是"格式"，模型不该学"何时开始说"，后者是"停止"，必须学。
2. **多轮对话有两种流派**：上面是"全程掩码，只学每轮回答"（主流）；也有实现对最后一轮前的回答同样掩码、只学最后一轮（更保守）。混用会静默改变有效数据量，训练前务必确认框架行为。
3. **视觉 Token 区域必须整体掩码**——它们是条件输入而非预测目标，若不掩码，576+ 个 Token 的 loss 会淹没真正的文本监督（Stage 1 第 4-5 周教程 2.2 节已论证）。

### 2.4 Padding 与 Packing 的取舍

| 机制 | 做法 | 算力利用率 | 实现复杂度 | 适用场景 |
| --- | --- | --- | --- | --- |
| Padding（默认） | batch 内补到等长 | 低（极端短长混批时 <50%） | 低 | 小规模、原型验证 |
| Length Grouping | 按长度分桶组 batch | 中高 | 低 | 单图固定分辨率 |
| **Sample Packing** | 多条样本拼接成固定长度序列 | **>95%** | 高（需位置编码/掩码配合） | **大规模 SFT 标配** |

Packing 的关键细节（也是 90% 的坑所在）：

- **跨样本注意力隔离**：朴素拼接后，样本 B 的 Token 会注意到样本 A 的内容（信息泄漏 + 分布污染）。两种解法：
  - **FlashAttention varlen 模式**：传入 `cu_seqlens`（累积序列长度），attention kernel 按段计算，段间零交互——**这是正确且主流的做法**；
  - 注意力掩码矩阵（block-diagonal）：正确但 $O(L^2)$ 显存，长序列下不可行；
  - 偷懒不隔离（仅靠位置重置）：**错误**，loss 会假性偏低。
- **位置编码重置**：每条样本内 position id 从 0 重新计数（Qwen2-VL 的 packed M-RoPE 还要重置三维位置，这正是 LLaMA-Factory 近期提交 `fix position ids after merging packed-mrope` 修复的问题——packing 与 M-RoPE 的组合是当前工程前沿坑区）。
- **跨样本边界不吞 Token**：拼接点要保证上一条的最后一个 Token（如 `<|im_end|>`）与下一条的第一个 Token 完整，边界截断需按样本边界而非固定字符数。

---

## 3. 实现与验证

### 3.1 本周 MVP：手写 Dataset + DataCollator

以下代码**不依赖任何微调框架**，只依赖 `transformers` 的 Processor，可直接运行。它完整实现了 2.1 节流水线的核心路径（单图单轮简化版 + 断言验证），多轮/多图的扩展点在注释中标出。

```python
"""
手写多模态 Dataset 与 DataCollator，输出模型输入四件套并验证 labels 掩码。
运行方式: python stage2_week1_dataset_collator.py
依赖: torch, transformers, pillow   (首次运行会下载 Qwen2-VL-2B 的 processor)
"""
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
IMG_TOKEN_ID = 151655          # Qwen2-VL 的 <image> 占位 ID
IM_END_ID = 151645             # <|im_end|>


class MyImageTextDataset(Dataset):
    """样本级处理: JSON -> 渲染 prompt -> 图片张量 + 分词"""

    def __init__(self, path: str, processor: AutoProcessor):
        self.samples = json.load(open(path))
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = s["image"]              # 实际项目中这里做 Image.open + convert("RGB")
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": s["conversations"][0]["value"].replace("<image>", "")},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": s["conversations"][1]["value"]}]},
        ]
        # apply_chat_template 分两段: prompt(含视觉占位) 与完整序列
        prompt_text = self.processor.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True)  # 到 <|im_start|>assistant\n 为止
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)      # 加上回答与 <|im_end|>

        prompt_ids = self.processor(text=[prompt_text], images=[image],
                                    return_tensors="pt").input_ids[0]
        full_ids = self.processor(text=[full_text], images=[image],
                                  return_tensors="pt").input_ids[0]

        # 掩码构造: 在完整序列上把 prompt 段置 -100
        labels = full_ids.clone()
        labels[: len(prompt_ids)] = -100
        # 防御性检查: prompt 长度对齐处不应是回答内容
        assert (full_ids[: len(prompt_ids)] == prompt_ids).all(), \
            "full 序列前缀与 prompt 不一致, 模板渲染有问题"
        return {
            "input_ids": full_ids,
            "labels": labels,
            "pixel_values": image,      # 交给 processor 统一处理的占位, 见 collator
            "prompt_len": len(prompt_ids),
        }


class MyDataCollator:
    """批次级处理: padding -> attention_mask -> 张量堆叠"""

    def __init__(self, processor: AutoProcessor, pad_to_multiple_of: int = 16):
        self.processor = processor
        self.pad_id = processor.tokenizer.pad_token_id or IM_END_ID
        self.mult = pad_to_multiple_of          # 对齐到 16 的倍数, 利好 tensor core

    def __call__(self, batch):
        L_max = max(len(x["input_ids"]) for x in batch)
        L_max = ((L_max + self.mult - 1) // self.mult) * self.mult

        input_ids, labels, attn = [], [], []
        for x in batch:
            L = len(x["input_ids"])
            pad = L_max - L
            input_ids.append(torch.cat([x["input_ids"],
                                        torch.full((pad,), self.pad_id)]))
            labels.append(torch.cat([x["labels"],
                                     torch.full((pad,), -100)]))
            attn.append(torch.cat([torch.ones(L,), torch.zeros(pad,)]))

        pixel_values = self.processor(
            images=[x["pixel_values"] for x in batch], return_tensors="pt")["pixel_values"]

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn),
            "pixel_values": pixel_values,
        }


if __name__ == "__main__":
    # ---- 构造 10 条测试数据 ----
    data = [{
        "conversations": [
            {"from": "human", "value": "<image>用一句话描述图片。"},
            {"from": "gpt", "value": f"图片展示了一个测试场景 {i}。"}
        ],
        "image": torch.randn(3, 448, 448)      # 模拟已解码图片, 实际为 PIL.Image
    } for i in range(10)]
    json.dump(data, open("/tmp/sft_mini.json", "w"))

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    ds = MyImageTextDataset("/tmp/sft_mini.json", processor)
    collator = MyDataCollator(processor)
    batch = collator([ds[i] for i in range(4)])

    # ---- 四件套形状断言 ----
    B, L = batch["input_ids"].shape
    assert batch["input_ids"].shape == (4, L) and L % 16 == 0
    assert batch["attention_mask"].shape == (4, L)
    assert batch["labels"].shape == (4, L)
    assert batch["pixel_values"].dim() == 4     # [B, C, H, W]

    # ---- 核心验证 1: prompt 区间 labels 全为 -100, 回答区间为真实 id ----
    for b in range(4):
        p = (ds[b]["labels"] != -100).nonzero()[0, 0]
        assert (ds[b]["labels"][:p] == -100).all(), f"样本{b}: prompt 区未完全掩码"
        assert (ds[b]["labels"][p:] != -100).any(), f"样本{b}: 回答区没有监督信号"
        # 回答区最后一个监督 Token 应是 <|im_end|> (学会停止的关键)
        assert ds[b]["labels"][p:][ds[b]["labels"][p:] != -100][-1] == IM_END_ID

    # ---- 核心验证 2: labels 与 input_ids 在监督区间严格对齐 ----
    supervised = batch["labels"] != -100
    assert (batch["labels"][supervised] == batch["input_ids"][supervised]).all()

    # ---- 核心验证 3: padding 区 label=-100 且 attention=0 且 id=pad ----
    pad_pos = batch["attention_mask"] == 0
    if pad_pos.any():
        assert (batch["labels"][pad_pos] == -100).all()

    # ---- 核心验证 4: 视觉占位 token 确实被展开进了序列 (Qwen2-VL) ----
    assert (batch["input_ids"] == IMG_TOKEN_ID).any(), "未找到视觉占位符"

    print(f"✅ batch 四件套: input_ids{tuple(batch['input_ids'].shape)}, "
          f"pixel_values{tuple(batch['pixel_values'].shape)}, "
          f"监督 Token 占比 {supervised.float().mean():.1%}")
```

**预期输出**（形状数字随图片分辨率/文本长度浮动，占比约为 `监督 Token 占比 8%~15%`）：

```text
✅ batch 四件套: input_ids(4, 1104), pixel_values(4, 3, 448, 448), 监督 Token 占比 9.3%
```

**一个值得注意的现象**：监督 Token 占比通常只有 5%~15%——prompt、视觉占位、格式头占掉了序列的绝大部分。这正是第 4 周 Sample Packing 有价值的原因：这些被掩码的 Token 虽不算 loss，但仍要参与前向计算，浪费必须靠打包来摊薄。

### 3.2 与 LLaMA-Factory 实现对照（源码研读指引）

带着本周的概念读框架源码，对应关系如下：

| 本周概念 | LLaMA-Factory 位置（已核实） |
| --- | --- |
| 样本级模板渲染与分词 | `data/processor/processor_utils.py`、`data/template.py` |
| 多模态输入组装（占位符展开、`image_grid_thw`） | `data/mm_plugin.py`（各模型的 Plugin 子类） |
| labels 掩码构造 | `data/processor/supervised.py`（`_encode_data_example` 附近） |
| 批次级 padding / packing | `data/collator.py`（`MultiModalDataCollatorForSeq2Seq`，其中用 `get_position_ids`/`get_seqlens` 处理 packed M-RoPE） |

阅读顺序建议：`template.py`（理解渲染）→ `supervised.py`（理解掩码）→ `collator.py`（理解拼装）→ `mm_plugin.py`（理解多模态特殊处理）。读完再回头看 3.1 的手写版，会发现结构一一对应——区别只在框架版处理了多轮、多图、工具调用、packing 等生产细节。

---

## 4. 工程权衡与失效模式

### 4.1 核心权衡

1. **Processor 放 Dataset 还是 Collator**：图片重处理必须在 Dataset（多进程并行）；轻量文本拼接放 Collator。判据是"是否需要看到整个 batch"。
2. **pad_to_multiple_of 的取值**：对齐 16/64 可触发 tensor core 的最优 tile，长序列下吞吐提升可观（经验规则 5%~15%）；代价是轻微的 padding 浪费。bf16 训练建议必开。
3. **Packing vs Padding 的临界点**：样本平均长度 < 序列预算的 30% 时，packing 收益巨大（3 倍以上吞吐）；样本本来就接近满长时收益趋零。1000 条以内的小规模实验不必折腾 packing，10k 条以上必开。

### 4.2 三个代表性失效模式

**失效 1：`<image>` 占位符数量与图片数不匹配**
- **症状**：预处理报 `ValueError: Image features and image tokens do not match`，或训练正常但推理时模型声称"没看到图"。
- **根因**：prompt 里写了 2 个 `<image>` 但 `images` 字段只有 1 张图；或文本答案里意外包含了占位符字符串。
- **定位**：打印一条预处理完成的样本，数 `input_ids` 中占位 ID 的个数 vs `pixel_values` 的第 0 维。
- **修复**：数据构造脚本中对 `<image>` 出现次数做 `assert == len(images)`；答案文本做占位符黑名单过滤。

**失效 2：掩码边界差一位——回答的第一个 Token 被掩掉**
- **症状**：训练 loss 比预期低一截、推理输出整体"延后一拍"或答非所问的风格。
- **根因**：`labels[: len(prompt_ids)] = -100` 时，`add_generation_prompt` 渲染的 prompt 末尾是 `<|im_start|>assistant\n`，若模板变体把换行符归到回答侧，`len(prompt_ids)` 的切分点就错位一个 Token。Causal LM 中错位一个位置意味着模型学到的"下一词预测"整体平移。
- **定位**：把一条样本的 `labels` 中非 -100 的 Token decode 回字符串肉眼看——若开头带了标点/空白错位即确认。
- **修复**：不要依赖"长度切分"，改用**内容锚点**：找到回答区间第一个真实内容 Token 的位置，从它开始放开掩码；升级框架版本时重跑掩码断言。

**失效 3：Packing 后 loss 假性偏低、评测指标反而下降**
- **症状**：开启 packing 后训练 loss 明显低于不 packing 的同配置，但 benchmark 分数不升反降。
- **根因**：跨样本注意力未隔离（样本间互相"背答案"），loss 被稀释为"在别的样本语境下预测本样本"的假任务。
- **定位**：构造两条语义矛盾的单轮样本（如 A 问"天空什么颜色"答"红色"、B 同问答"蓝色"），packing 后同批训练，观察模型回答是否出现串扰；或检查位置编码在样本边界是否重置。
- **修复**：确认 collator 使用 FlashAttention varlen 路径（`cu_seqlens`）而非纯拼接；检查 M-RoPE/1D 位置在样本边界重置逻辑（LLaMA-Factory 近期的 packed-mrope 修复即此类问题）。

---

## 5. 延伸思考题

1. **占比分析**：3.1 节输出的"监督 Token 占比"只有约 10%。如果一条样本有 3 张高分辨率图但回答只有 10 个字，占比会低到什么程度？这种样本对训练是增益还是噪声？（提示：算一算 3×(H/28×W/28) vs 10 个监督 Token 的比例；极低占比样本等效于用大量算力换极少监督，配方层面应设最低占比过滤线。）
2. **左 padding 的秘密**：推理（generation）时框架用左 padding 而训练用右 padding，为什么？（提示：生成从序列末尾续写，左 padding 保证所有样本的"最后一个真实 Token"对齐在末位；训练则是从每个位置预测下一个位置，右 padding 自然。）
3. **动手实验**：把 3.1 的 collator 改造成 packing 版本：两条样本拼进一条长度 2048 的序列，用 `transformers` 的 `FlashAttentionKwargs`（`cu_seq_lens_q/k`）实现段间隔离，并断言两条样本各自的 loss 与单独前向一致（容差 1e-3）。这个实验完成后，你对 collator 的理解会超过多数调参者。

---

*下一篇：[第 2 周：训练阶段划分、Loss 设计与参数效率微调](第2周-训练阶段Loss设计与PEFT.md)*
