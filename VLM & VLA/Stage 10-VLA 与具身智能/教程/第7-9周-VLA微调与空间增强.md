# 第 7-9 周教程：VLA 微调与空间能力增强

> **本周要回答的三个问题**
> 1. OpenVLA 适配新任务/新本体，LoRA 该怎么配？为什么"只调最后一层/冻结视觉塔"效果差？
> 2. Co-finetuning（混入通用数据）防遗忘的机制与配比怎么定？
> 3. 如何把空间/深度信息融合进 VLA 训练，提升抓取成功率？

对应学习计划：第 7-9 周。交付物：500 条新物体抓取仿真数据 → LoRA 微调 OpenVLA（混 10% 通用 VQA 数据）→ SimplerEnv A/B 对比，证明新物体抓取成功率提升 ≥30%。

---

## 1. 第一性原理：微调 VLA = 在"保留通用性"与"适配新域"之间走钢丝

### 1.1 根本矛盾：新域数据少，而底座能力怕被覆盖

VLA 微调面对的是 Stage 2/6 反复出现的老矛盾在具身语境的升级版：

- **新域数据稀缺**：500 条轨迹对"学会新物体抓取"够吗？——取决于预训练分布与新域的距离（Stage 5 的分布思维）；
- **灾难性遗忘**：纯动作数据微调会把 7B 模型的语言能力与通用视觉表征挤掉（自测考点）——OpenVLA 官网明确指出其简化训练（只在机器人数据上微调 VLM）导致 **RT-2-X 在"互联网概念泛化"任务（如把可乐移到 Taylor Swift 旁）上更好**，这正是遗忘的公开证据（已核实）；
- **闭环放大一切**：微调引入的任何行为怪癖（抖动/迟疑）都会被误差累积放大（第 5-6 周）。

因此第 7-9 周的三件事——LoRA 策略、Co-finetuning、空间增强——本质是**同一枚硬币的三面：如何用最少的破坏换最大的适配**。

### 1.2 LoRA 策略：OpenVLA 的实证结论

OpenVLA 论文对微调策略做过系统消融（已核实官网/论文），结论直接指导实践：

| 策略 | 结论（论文口径） |
| --- | --- |
| **只调最后一层** | 效果差 |
| **冻结视觉编码器** | 效果差 |
| **LoRA（r=16~32）** | **最佳性价比：仅 1.4% 参数，匹敌全量微调** |
| 全量微调 | 效果好但显存/算力高 |

**为什么"冻结视觉塔"在 VLA 上不行**（与 Stage 2 的结论表面冲突！）：Stage 2 说"VLM SFT 冻结视觉塔是对的"；但那是在**视觉域与预训练分布接近**的前提下。具身场景的图像分布（腕部相机、机械臂视角、桌面杂乱场景）与互联网图像差异大——**域差距大到通用视觉特征失效时，视觉塔必须参与适配**（Stage 1 第 6 周的"何时解冻"论点的具身版）。LoRA 的答案优雅：**视觉塔也挂 LoRA（低秩适配）而非冻结/全调**——既适配域差又防遗忘。

**Stage 6 的 LoRA 挂载决策表在具身的更新版**：

| 位置 | 通用 VLM SFT | VLA 微调（域差距大） |
| --- | --- | --- |
| 视觉塔 | 冻结 | **挂 LoRA** |
| LLM 主干 | LoRA (all) | LoRA (all) |
| Projector | 全训 | 全训（量小） |

### 1.3 Co-finetuning：防遗忘的机制与配比

**机制**：混入通用数据（LLaVA VQA / OXE 原始混合）的目的是**给"通用能力方向"保留梯度锚点**——Stage 6 第 3 周"通用数据是防遗忘的锚"的直接移植。配比的经验起点：

$$
\text{微调集} : \text{通用回放} = 90\% : 10\%
$$

（学习计划口径：混 10% 通用 VQA。）配比的调节信号：**通用能力抽查集**（VQA/对话 50 题）在微调中的表现——掉分 → 提回放比例；新任务学不动 → 降回放比例。与 Stage 3 配比消融方法完全同构。

**回放数据的形态细节**：回放的 VQA 数据必须转成与动作微调**相同的输入格式**（图像 + 文本 prompt + 文本答案 Token——VQA 的"答案"就在动作词表之外，天然走文本输出路径；这正是"动作即语言"设计的红利：同一模型同一损失无需特殊处理）。

### 1.4 空间/3D 能力增强：把"知道在哪"喂给策略

VLA 的抓取失败多数源于空间精度（Stage 3 的老结论在机械臂上更严酷：2mm 级要求）。增强路径三条：

1. **Prompt 注入 3D 信息**：把深度图/3D Bounding Box 的文本描述（如 "target at (0.32, -0.15, 0.05) m"）写进 prompt——零训练成本的注入（依赖模型的数值理解力，效果有上限）；
2. **额外模态通道**：深度图作为第二个"图像"输入（视觉塔加一路 depth encoder）——效果强、改动大；
3. **数据融合（学习计划指定路径）**：用 Stage 4 的高质量空间 GT 合成"指令 + 3D 坐标 + 动作"的训练样本，教模型把 3D 推理内化进动作预测——**与预训练分布的兼容性最好**（仍是 next-token prediction）。

三条路线的共同前提：空间 GT 的质量（Stage 4 空间产线的真值可信度）——垃圾空间标注会教出自信的错位抓取。

---

## 2. 实现与验证

### 2.1 微调数据准备（500 条 + 10% 回放）

```python
"""
微调数据构造: 新物体抓取仿真轨迹 (500) + 通用回放 (50), 转 OpenVLA 训练格式。
运行方式: python stage10_week7_make_ftdata.py
依赖: 标准库 (真实现接 SimplerEnv 数据采集与 LLaVA VQA 集)
"""
import json
import random

def make_ft_dataset(n_target=500, replay_ratio=0.1):
    # ---- 主体: 新物体抓取轨迹 (来自 SimplerEnv 采集/遥操作, 每条带成功标记) ----
    rows = []
    rng = random.Random(0)
    new_objects = ["yellow duck", "metal wrench", "red dice", "sponge", "tape roll"]
    while len(rows) < n_target:
        obj = rng.choice(new_objects)
        # 真实来源: SimplerEnv rollout (成功) 或遥操作; 此处为格式样例
        rows.append({
            "image": f"ft/episode_{len(rows):03d}/front.png",
            "wrist_image": f"ft/episode_{len(rows):03d}/wrist.png",
            "prompt": f"In: What action should the robot take to pick up the {obj}?",
            "action_token_ids": f"[episode {len(rows):03d} 的 7-DoF 序列已离散化]",  # 占位说明
            "meta": {"source": "sim_ft", "success": True, "object": obj},
        })
    # ---- 回放: 通用 VQA (LLaVA 子集), 保持语言/通用视觉能力 ----
    n_replay = int(n_target * replay_ratio)
    for i in range(n_replay):
        rows.append({
            "image": f"replay/vqa_{i:03d}.png",
            "prompt": f"USER: 描述这张图片的主要内容。 ASSISTANT: 图片中展示了场景 {i} 的物体与布局。",
            "meta": {"source": "replay_vqa", "success": None},
        })
    rng.shuffle(rows)
    # ---- 断言: Co-finetuning 的配比与格式 ----
    n_ft = sum(r["meta"]["source"] == "sim_ft" for r in rows)
    n_rp = len(rows) - n_ft
    assert abs(n_rp / len(rows) - replay_ratio) < 0.02, "回放比例偏离设计值"
    json.dump(rows, open("ft_dataset.json", "w"), ensure_ascii=False, indent=1)
    print(f"微调集 {n_ft} + 回放 {n_rp} (共 {len(rows)}) -> ft_dataset.json")
    # 防遗忘抽查集 (独立于训练!) —— 微调前后通用能力对比的载体
    json.dump([{"q": f"天空通常是什么颜色? #{i}", "a": "蓝色"} for i in range(50)],
              open("replay_probe.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    make_ft_dataset()
```

### 2.2 LoRA 微调（OpenVLA 官方脚本路线）

```bash
# OpenVLA 官方提供微调脚本 (vla-scripts/finetune.py), LoRA 关键参数:
python vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir <数据目录> --dataset_name <你的 RLDS 微调集> \
  --run_root_dir runs/ft_newobj \
  --lora_rank 32 \                       # 官方结论: LoRA 1.4% 参数匹敌全量
  --batch_size 16 --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \                 # LoRA lr 量级 (Stage 6 纪律: 新参数用大 lr)
  --image_aug true \                     # 图像增广 (域随机化的数据侧对应物)
  --wandb_project vla-ft-newobj
# 混入回放数据: 通过数据配比实现 (2.1 的 ft_dataset 已混合), 或官方的多数据集混合支持
```

（官方还支持量化 LoRA 微调以进一步省显存——Stage 8 技术的复用；参数以所装 openvla 版本为准。）

### 2.3 A/B 评测与交付图表

```python
"""微调前后 A/B: SimplerEnv 新物体任务成功率对比 (含通用能力防遗忘抽查)"""
import numpy as np
import matplotlib.pyplot as plt

def run_eval(model_path, tasks, eps_per_task=10):
    """逐任务跑闭环 (复用第 5-6 周的 rollout), 返回 {task: sr}"""
    # ... 加载模型 -> 逐任务 rollout -> 成功率 ...
    return {t: np.random.RandomState(hash(t) % 2**31).uniform(0.3, 0.9) for t in tasks}

if __name__ == "__main__":
    tasks = ["pick yellow duck", "pick metal wrench", "pick red dice",
             "pick sponge", "pick tape roll"]
    base = run_eval("openvla/openvla-7b", tasks)                 # 微调前
    ft = run_eval("runs/ft_newobj/last", tasks)                  # 微调后
    delta = {t: ft[t] - base[t] for t in tasks}
    mean_gain = np.mean(list(delta.values()))

    x = np.arange(len(tasks))
    plt.figure(figsize=(9, 4))
    plt.bar(x - 0.2, [base[t] * 100 for t in tasks], 0.4, label="OpenVLA base")
    plt.bar(x + 0.2, [ft[t] * 100 for t in tasks], 0.4, label="+LoRA ft (10% replay)")
    plt.xticks(x, [t.replace("pick ", "") for t in tasks], rotation=15)
    plt.ylabel("Success Rate (%)"); plt.legend(); plt.tight_layout()
    plt.savefig("stage10_ab.png", dpi=150)

    print(f"各任务提升: " + ", ".join(f"{t.replace('pick ','')}={delta[t]:+.0%}" for t in tasks))
    print(f"平均提升: {mean_gain:+.0%}")
    assert mean_gain >= 0.30, f"未达 MVP 门槛 (+30%): {mean_gain:+.0%}"
    # 防遗忘抽查 (replay_probe.json): 微调后通用问答 50 题准确率不应显著下降
    # ... 通用抽查实现 ...
    print("✅ A/B 达标; 记得附上通用能力抽查结果 (防遗忘证据)")
```

**验收口径**：平均提升 ≥30%（学习计划门槛）+ **通用抽查不掉分**（Co-finetuning 生效证据）。两个都过才算达标——只涨新任务但通用崩了，是"用遗忘换适配"的假成功。A/B 的公平性纪律（同 episode 数、同 variation、同种子，Stage 5 消融纪律）全部适用。

---

## 3. 工程权衡与失效模式

### 3.1 决策表：微调规模的阶梯

| 数据量 | 策略 | 预期 |
| --- | --- | --- |
| <100 条 | 只调 Projector + 强增广；或遥操作补数据 | 适配有限 |
| 100~1k | **LoRA (r=16~32) + 视觉塔 LoRA + 10% 回放** | 官方结论的甜点区 |
| 1k~10k | LoRA r↑ 或全量（多卡） | 新平台深度适配 |
| >10k | 考虑把新数据并入预训练混合（大工程） | 通用能力升级 |

### 3.2 三个代表性失效模式

**失效 1：归一化统计量未随 checkpoint 迁移（第 1-2 周失效 1 的微调版）**
- **症状**：微调后 SR 反而低于基线；或动作幅度异常。
- **根因**：微调用的新数据集重算了 `norm_stats`，但评测时 detokenizer 仍用预训练的键（或反之）——两侧行为不一致。
- **定位**：取一条微调集样本做 tokenize→detokenize 往返，对比 GT。
- **修复**：norm_stats 随 checkpoint 打包；评测脚本显式加载对应键；跨数据集评测时明确声明用哪套统计量。

**失效 2：Co-finetuning 的回放数据格式不一致**
- **症状**：混入回放数据后训练 loss 出现两个平台（或回放子集 loss 恒高），通用能力没保住。
- **根因**：回放的 VQA 样本没有转成与动作样本一致的对话模板/图像预处理——模型把它们当 OOD 输入，梯度行为异常。
- **定位**：分数据源打印 loss 曲线（wandb 分组）——两组曲线形态迥异即确认。
- **修复**：回放数据走同一 Processor/模板管线（"动作即语言"的格式红利要求统一格式才能兑现）。

**失效 3：A/B 评测的"变相作弊"——微调集泄漏进评测集**
- **症状**：提升远超 30%（如 +80%），但换成全新物体列表立刻回落。
- **根因**：微调的 500 条与评测的新物体同源同分布（同批采集/同场景 ID）——评测测的是"背题"。
- **定位**：把评测物体换成**从未在任何数据中出现过的全新物体**复测（held-out 物体）。
- **修复**：数据采集期就预留 held-out 物体池（Stage 3 第 4 周 mini-Bench 的隔离纪律）；报告必须包含"训练物体 / held-out 物体"两组成功率。

---

## 4. 延伸思考题

1. **LoRA 挂载位置的具身消融设计**：OpenVLA 论文说"冻结视觉塔效果差"，但没细说"视觉塔 LoRA 的秩"该多大。设计消融：视觉塔 LoRA r ∈ {4, 16, 64} × LLM LoRA r=32，在 SR 与通用抽查两轴上画权衡曲线——预测视觉塔秩对"域适配 vs 遗忘"的敏感度高于 LLM 侧的原因。
2. **深度注入的对比实验**：路线一（prompt 文本注入 3D 坐标）与路线三（数据融合教模型内化）在"遮挡物体抓取"上的预期差异？设计一个 100 条遮挡场景的对照评测（提示：prompt 注入依赖模型把文本坐标与视觉对齐——跨模态对齐恰是 VLM 弱项；数据融合把对齐学进权重，但需要足够的量）。
3. **防遗忘的度量设计**："通用能力没掉"怎么量化？设计一个 5 分钟可跑的通用抽查协议（题目构成/评分方式/通过线），并论证为什么它必须独立于训练数据。（提示：50 题覆盖 VQA/对话/常识；通过线相对底座掉幅 <2%；独立性是防"抽查题被回放数据泄漏"——Stage 5 的隔离纪律。）

---

*下一篇：[第 10 周：Sim-to-Real 与结课总结](第10周-SimToReal与结课总结.md)*
