# 第 4 周教程：VLM 偏好对齐训练实战与抑幻验证

> **本周要回答的三个问题**
> 1. TRL / LLaMA-Factory 的 DPO/SimPO 训练怎么配置？超参（β、lr、梯度裁剪）的第一性依据是什么？
> 2. WandB 上的 `rewards/margins`、`rewards/accuracies`、`logps/*` 各自在报告什么训练状态？
> 3. "DPO 后幻觉率显著下降"如何在 POPE/HallusionBench 上被严谨地证明？

对应学习计划：第 4 周。交付物：用第 3 周偏好数据对 Qwen2-VL/LLaVA 做一次 DPO/SimPO 微调，导出 WandB 的 rewards/margins 曲线，POPE 前后对比证明幻觉率下降（目标：POPE 准确率提升 ≥3%）。

---

## 1. 第一性原理：对齐训练的动力学监控

### 1.1 根本矛盾：损失下降 ≠ 对齐成功

DPO 的损失可以"完美"下降（margin 拉开、acc 到 1.0），同时模型的生成质量在退化（第 1 周失效 1、第 2 周失效 1 的共同机制：策略学会了"拉开分差的捷径"——压低一切/变长/模板化）。**对齐训练的监控必须同时看三个层面**：

1. **偏好判别层**（DPO 的直接优化目标）：margins、accuracies——模型能否区分好坏；
2. **生成质量层**（对齐的不可妥协底线）：`logps/chosen` 的绝对轨迹、生成样例的人工抽检——模型是否还在正常说话；
3. **下游能力层**（最终验收）：POPE/HallusionBench/通用基准——幻觉率降了多少、通用能力掉没掉。

只看第 1 层就宣布成功，是 DPO 实战中最常见的自欺。三层齐全才算训练完成。

### 1.2 超参的第一性依据

| 超参 | 常用值 | 第一性依据 |
| --- | --- | --- |
| **β** | 0.1（DPO）；SimPO 用 β=2~2.5 配 γ=0.5~1.0 | KL 锚定强度（第 1 周 1.4 节）：抑幻场景偏好稳定更新，默认 0.1 |
| **lr** | **SFT 的 1/10 或更低**（如 5e-7 ~ 5e-6） | 对齐是在"已会说话"的模型上做微小行为修正——大幅权重移动会破坏 SFT 能力（Stage 2 的遗忘机制）；DPO 的梯度天然小（sigmoid 饱和），也容不下大 lr |
| **epochs** | 1（最多 2） | 偏好数据量小（数百~数千条），多轮必过拟合（第 3 周数据上限决定的） |
| **gradient clipping** | max_grad_norm = 1.0（标配） | DPO 初期梯度分布与 SFT 不同（rejected 的强推低梯度），裁剪防单 batch 拉飞 |
| **batch（等效）** | 32~128 | DPO 是对比型损失，大 batch 让 margin 统计更稳、训练更平滑 |
| **LoRA 配置** | r=16~32，`lora_target: all` | 抑幻需要改变输出分布的多个维度；Stage 2 的结论直接沿用 |

**β-lr 联动**（第 1 周失效 1 的对策落地）：β 提高会压低奖励尺度与梯度，需要 lr 微升补偿；反之降 β 时必须降 lr。单独动 β 是最常见的调参事故。

### 1.3 WandB 曲线的判读手册

| 指标 | 健康形态 | 危险形态 |
| --- | --- | --- |
| `rewards/chosen` | 初期约 0（policy≈ref），随后缓升或持稳 | 持续下降（chosen 被牺牲，似然坍塌前兆） |
| `rewards/rejected` | 持续下降（压低坏回答） | 快速暴跌（rejected 被推向零概率，过拟合噪声标签） |
| **`rewards/margins`** | **稳步增大后趋平** | 暴涨后平台（过拟合）/ 始终平坦（没在学，查 lr/β/数据） |
| `rewards/accuracies` | 稳步爬升到 **0.7~0.9** | 冲到 ≥0.98（数据可分性过强=表面捷径）；长期 <0.6（数据无梯度或超参失配） |
| `logps/chosen` | 缓慢小幅下降 | **异常暴跌（模型崩溃信号，学习计划点名的监控项）** |
| `nll_loss`（若有 SFT 混合项） | 正常下降 | 发散 |

一个重要的读图细节：`rewards/*` 是 **β 缩放后的隐式奖励在 batch 上的均值**——它的绝对数值没有跨 β 配置的可比性（β 变了尺度就变），**只有同一次运行内的趋势有意义**。

---

## 2. 实现与验证

### 2.1 LLaMA-Factory 的 DPO 实战配置

`qwen2vl_dpo.yaml`（承接 Stage 2 第 4 周的训练工程配置，DPO 阶段复用其全部纪律）：

```yaml
### 模型 (DPO 起点是 SFT 后的模型, 不是基座!)
model_name_or_path: saves/qwen2vl-2b/sft_merged   # Stage 2 的 SFT 产物
trust_remote_code: true

### 方法
stage: dpo                    # 换 simpo 则改此字段 (LLaMA-Factory: stage: simpo? 以版本文档为准)
do_train: true
finetuning_type: lora
pref_beta: 0.1                # DPO 的 β
pref_loss: sigmoid            # dpo 默认; 可选 ipo/kto_pair 等变体
lora_target: all
lora_rank: 16
lora_alpha: 32

### 数据 (第 3 周产物, sharegpt 风格偏好注册)
dataset: halluc_pref_500
template: qwen2_vl
cutoff_len: 2048
preprocessing_num_workers: 8

### 训练超参
per_device_train_batch_size: 2
gradient_accumulation_steps: 16     # 等效 batch = 32 (对比损失的稳定性)
learning_rate: 5.0e-7               # SFT 的 1/10 量级, DPO 关键纪律
num_train_epochs: 1.0               # 小数据单轮, 防过拟合
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true
max_grad_norm: 1.0                  # 梯度裁剪

### 输出
output_dir: saves/qwen2vl-2b/dpo_halluc
logging_steps: 5                    # DPO 曲线信息密度高, 密采样
save_steps: 50
plot_loss: true
report_to: wandb
```

```bash
# 偏好数据注册 (data/dataset_info.json):
# "halluc_pref_500": {
#   "file_name": "pref_500.jsonl",
#   "ranking": true,                       # 关键: 声明这是偏好数据 (chosen/rejected)
#   "columns": { "prompt": "prompt", "query": "question",
#                "chosen": "chosen", "rejected": "rejected", "images": "image" } }
FORCE_TORCHRUN=1 llamafactory-cli train qwen2vl_dpo.yaml
```

**显存参考**（经验量级）：2B 模型 + LoRA + bf16 + ref 前向，等效 batch 32、cutoff 2048——约 16~22GB；SimPO 少 ref 前向可再省。7B 模型建议 QLoRA 或多卡。

### 2.2 TRL 的等价路径（自定义数据列时用）

```python
"""TRL DPOTrainer 骨架 (数据字段更自由时的选择)"""
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForVision2Seq, AutoProcessor
from datasets import load_dataset

MID = "saves/qwen2vl-2b/sft_merged"
model = AutoModelForVision2Seq.from_pretrained(MID, torch_dtype="bfloat16")
ref = AutoModelForVision2Seq.from_pretrained(MID, torch_dtype="bfloat16")  # ref 可省略
processor = AutoProcessor.from_pretrained(MID)
ds = load_dataset("json", data_files="pref_500.jsonl", split="train")

cfg = DPOConfig(output_dir="out_dpo", per_device_train_batch_size=2,
                gradient_accumulation_steps=16, learning_rate=5e-7,
                beta=0.1, max_prompt_length=1024, max_completion_length=1024,
                num_train_epochs=1, bf16=True, logging_steps=5,
                max_grad_norm=1.0, report_to="wandb")
trainer = DPOTrainer(model=model, ref_model=ref, args=cfg,
                     train_dataset=ds, processing_class=processor)
trainer.train()
```

（多模态列的映射与 `ref_model=None` 的细节随 TRL 版本演进，以所装版本文档为准；两个引擎选一个深度用即可，原理层完全同构。）

### 2.3 抑幻验证：POPE 前后消融

**实验矩阵**（Stage 3 第 4 周的消融纪律 + Stage 5 的 B 组思想）：

| 组 | 模型 | 目的 |
| --- | --- | --- |
| P0 | SFT 底座（DPO 起点） | 基线 |
| P1 | SFT + DPO（第 3 周数据） | 抑幻效果 |
| P2 | SFT + DPO（**随机文本对**对照，非视觉特异） | 证明增益来自视觉偏好而非"DPO 本身" |

P2 是严谨性的关键：如果 P1 比 P0 好，可能是 DPO 训练本身带来的通用改善；只有 P1 > P2 才能归因到"视觉幻觉偏好对"的内容。

```bash
# 三组模型分别评测 (协议冻结: VLMEvalKit 固定 commit, POPE 三采样全跑)
for g in sft_base dpo_halluc dpo_randompair; do
  python run.py --model saves/$g --data POPE --work-dir evals/$g
done
```

**报告模板**（达成标准：POPE 准确率提升 ≥3%）：

| 组 | POPE Random F1 | Popular F1 | **Adversarial F1** | FP 数 | 通用(自定抽查) |
| --- | --- | --- | --- | --- | --- |
| P0 SFT | 0.842 | 0.821 | 0.788 | 202 | 基准 |
| P1 DPO-幻觉 | **0.881 (+3.9)** | **0.856** | **0.831 (+4.3)** | **141 (-30%)** | 持平±1 |
| P2 DPO-随机对 | 0.849 | 0.827 | 0.796 | 194 | 持平 |

判读要点：① **Adversarial 档的提升幅度应 ≥ Random 档**（幻觉偏好对主要治"先验诱导"）；② FP 绝对数下降比 F1 微涨更有说服力（幻觉直接减少）；③ 通用能力抽查持平才允许宣称"无害的抑幻"。三者齐备，"显著下降"才成立。

### 2.4 HallusionBench 补充验证（可选加强）

POPE 只测物体存在幻觉；用 Stage 3 第 3 周的完整协议补测属性/空间维度（HallusionBench 或自建 mini 基准），并复跑 No-image 探针——一个漂亮的闭环证据是：**DPO 后模型的 No-image 与有图回答一致率下降**（模型更依赖视觉证据了，Stage 3 第 2 周判读矩阵的良性方向）。

---

## 3. 工程权衡与失效模式

### 3.1 决策表：抑幻对齐的引擎与算法选择

| 场景 | 推荐 | 理由 |
| --- | --- | --- |
| 快速验证（数百条数据） | LLaMA-Factory + DPO（LoRA, β=0.1） | 配置面小，纪律文件成熟 |
| 长度敏感/答案精炼类抑幻 | SimPO（γ=0.5 起） | 长度归一化 + margin |
| 数据是单边标签（点赞/审核） | KTO | 免配对 |
| 幻觉片段级精确纠正 | 段级 dense DPO（RLHF-V 式） | 专一性最大化，但需片段标注 |

### 3.2 三个代表性失效模式

**失效 1：DPO 后通用对话能力下降（对齐税）**
- **症状**：POPE 涨了，但通用 VQA/对话变笨（回答变短、模板化、拒绝率上升）。
- **根因**：偏好数据太窄（全是幻觉对）+ β 太小/lr 偏大 → 模型把"输出分布"整体向"拒答与简短"方向移动（幻觉对的 rejected 常是长而详尽的错误描述，模型泛化出"长=坏"）。
- **定位**：通用抽查集前后对比 + `logps/chosen` 轨迹（持续下降即确认）；检查回答平均长度漂移。
- **修复**：偏好对里混入 20~30% "双好对"（chosen 略好于 rejected 的通用样本，KTO 论文式的正则）拉平分布；升 β 或降 lr；或缩短训练（1 epoch 内提前停）。

**失效 2：`rewards/accuracies` 虚高与表面捷径**
- **症状**：acc 训练 50 步内冲到 0.98，margin 暴涨，实际抑幻效果为零。
- **根因**：偏好对可分性过强——rejected 带着表面特征（如全是"图中"开头跑题、或系统性更短），模型学会按表面特征分类而非视觉语义（第 3 周失效 1 在训练侧的显影）。
- **定位**：margin 分布形状 + 抽 20 条"模型高置信判对"的样本人工看它依据什么判别。
- **修复**：回数据侧（对比度审计、规则熵控制）；训练侧无可救药——**垃圾数据调不出好 DPO**。

**失效 3：评测协议漂移导致"提升"不可比**
- **症状**：POPE 从 0.84 涨到 0.88，但同事复现只有 0.845。
- **根因**：前后评测用了不同 VLMEvalKit 版本（PR #1175 改过 MCQ 提取路由）/不同 work-dir 缓存/不同 prompt 模板——Stage 3 第 2 周失效 3 的缓存复用与协议漂移在这里兑现为假提升。
- **定位**：核对两次评测的框架 commit、work-dir 新旧、预测文件 hash。
- **修复**：前后评测**必须同环境同协议同日跑完**；报告里永久绑定协议四要素（框架版本/裁判/提取器/失败率）。

---

## 4. 延伸思考题

1. **DPO vs RLHF-V 式 dense DPO 的对比实验**：整句级偏好对（本周）与段级纠正（RLHF-V）在同等标注预算下（如 500 整句对 vs 1.4k 段级对）哪个抑幻效率高？设计实验并预测结果的方向（提示：段级的专一性优势 vs 整句的数据规模优势；RLHF-V 论文的 1.4k vs 10k 证据提示方向）。
2. **β 的抑幻特调**：幻觉抑制要求模型"对视觉证据敏感"，这与 KL 锚定（不远离 SFT）存在张力——锚太紧则视觉依赖提不起来。设计一个"双 β"实验：对视觉相关 Token 与文本 Token 施加不同 β 的可行性分析（提示：Token 级隐式奖励加权，接 Stage 5 思考题的 VIG 思路；讨论实现成本与理论自洽性）。
3. **完整的阶段闭环审计**：把 Stage 3→6 的抑幻产线写成一份"审计链"文档：归因报告（证据）→ 偏好对规格（设计）→ 合成脚本与校验（制造）→ DPO 配置（加工）→ POPE 消融（质检）。每个环节标注：输入物证、质量闸、责任人可复现命令。这份文档本身就是进入 Stage 7（RLVR）的最佳门票——RLVR 会把"偏好"升级为"可验证奖励"，而审计链的纪律完全通用。

---

*下一篇：[阶段六自测验收与复盘](阶段六自测验收与复盘.md)*
