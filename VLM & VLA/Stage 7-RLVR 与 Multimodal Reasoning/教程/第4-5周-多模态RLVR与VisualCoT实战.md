# 第 4-5 周教程：多模态 Visual CoT 与 Vision-Language RLVR 实战

> **本周要回答的三个问题**
> 1. VLM + GRPO 的 Prompt 结构与数据集怎么构造？"可验证"的多模态任务怎么选？
> 2. "慢思考涌现"（`<think>` 里自主的视觉推理）是真实能力还是模式复读？怎么区分与验证？
> 3. 多模态 Rollout 的显存突刺（多 Rollout × 图像 Token）怎么治？

对应学习计划：第 4-5 周。交付物：① 500~1,000 条空间推理/几何定位图像数据集；② verl + Qwen2-VL + 第 2 周 Verifier 跑 GRPO；③ 记录第 0/100/500 步的同图 `<think>` 思考链长度与准确率演变。

---

## 1. 第一性原理：RL 能放大什么，不能发明什么

### 1.1 根本矛盾：探索空间里必须有"对的东西"

RLVR 的机制是**放大已存在于策略采样分布中的行为**：Verifier 给正确的推理轨迹高分，策略梯度把"采样分布中已有的正确模式"的概率推高。这带来一个经常被误解的推论：

> **RL 不能教模型"不会的东西"，只能让"偶尔做对的"变成"稳定做对的"。**

（DeepSeek-R1 论文的结论方向一致：基座的推理先验决定 RL 的天花板。）由此导出本周两个最重要的工程推论：

1. **冷启动决定成败**：若 SFT 底座在目标任务的 pass@G（G 条采样里至少一条对）接近 0，GRPO 无从放大——要么换更强的底座、要么先用小规模 CoT SFT 热身（R1 的 cold-start 阶段）、要么降低任务难度建立梯度。**训练前先测 pass@G 是 RLVR 的第一纪律**（对应 Stage 5 "两端截断"思想的 RL 版：太易全对无区分、太难全错无信号）。
2. **数据不需要答案的推理过程，只需要可验证的答案**：与 SFT 需要 CoT 标注不同，RLVR 只要求 $(image, question, verifiable\_gt)$——GT 是框坐标（IoU 可算）、数值（可比较）、或选项字母。**这把数据构造成本从"标注推理链"降到"标注结果"**，是 RLVR 数据的可扩展性来源。

### 1.2 Visual CoT 涌现的机制假说与验证方法

"慢思考涌现"的现象描述：GRPO 数百步后，`<think>` 从模板填充变成结构化的视觉推理（"先定位左下角物体 → 估算相对距离 → 换算坐标"），且回答变长、变准。

机制解释（主流假说）：格式奖励 + 正确性奖励的组合下，**所有能提高答对率的行为模式都被正优势放大**——其中就包括"生成更长的探索性推理链"（多想几步更容易算对）。长 CoT 不是被直接奖励的（奖励只看结果），而是作为"答对的路径"被间接选择。

**涌现 vs 复读的判别**（本周交付的核心鉴别点）：

| 观察维度 | 真涌现 | 复读/表演 |
| --- | --- | --- |
| CoT 内容 | 随图片变化，引用图中的具体空间关系 | 不同图高度雷同的套话 |
| 长度与正确率 | 长度增长伴随准确率同步提升 | 长度涨、准确率平（纯耗时） |
| 消融 | 打乱图片后 CoT 内容显著变化 | 打乱图片 CoT 不变（不依赖视觉！） |
| 中间结论 | CoT 里的中间量与最终答案数值连贯 | 中间量与答案脱节 |

其中"打乱图片消融"（换噪声图看 CoT 是否变化）正是 Stage 3 No-image 探针的复用——**第 4-5 周的"涌现验证"就是 Stage 3 幻觉分析工具的反向应用**。

### 1.3 多模态 Rollout 的显存问题

GRPO 的 G 倍采样与图像 Token 的碰撞：G 条生成各自维护 KV cache，而 prompt 里的视觉 Token（高分辨率下数千，Stage 1 公式）在**每条**回答的 cache 里都占位。prompt 视觉 Token 512、G=8、回答 512 Token 时，仅该 prompt 组的 KV 就是单条的 ~8 倍。vLLM 的 prefix cache 在此处是救星：**同组 G 条共享同一份视觉前缀 KV**（1 份 prompt cache + G 份生成 cache，而非 8 份全量）——这就是学习计划说的"Vision Feature 缓存优化"的机制层。工程配套：限制输入分辨率（Stage 2 管线纪律）、`gpu_memory_utilization` 预算（第 3 周失效 1）。

---

## 2. 实战：全流程搭建

### 2.1 数据集构造（500~1,000 条，三种来源混合）

| 来源 | 构造方式 | 规模 | GT 形态 |
| --- | --- | --- | --- |
| **合成几何图**（主打，真值免费） | 程序绘制：随机布点/多边形/线段 + 已知坐标 | 400~600 | 精确坐标/数量/角度 |
| **公开数据集改造** | CV-Bench / Spatial457 类空间基准抽子集 | 200~300 | 官方 GT |
| **Stage 4 空间产线复用** | Grounded-SAM 标注的真实图片（provenance 里有 verified 标记） | 100~200 | 检测框 IoU 可验 |

合成几何示例（真值即构造参数，零标注成本）：

```python
# gen_geo.py 片段: 程序化生成"两物体相对方位+距离"样本
import json, random
from PIL import Image, ImageDraw

def make_sample(i, rng):
    W = H = 640
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
    x1, y1 = rng.randint(40, 300), rng.randint(40, 300)      # 物体A (红方块)
    x2, y2 = rng.randint(340, 600), rng.randint(40, 600)     # 物体B (蓝圆)
    s = rng.randint(40, 80)
    d.rectangle([x1, y1, x1+s, y1+s], fill="red")
    d.ellipse([x2, y2, x2+s, y2+s], fill="blue")
    dist = round(((x2-x1)**2 + (y2-y1)**2) ** 0.5)
    qa = {"image": f"geo_{i:03d}.jpg", "question":
          "红方块与蓝圆的水平距离是多少像素（取整）？", "gt": dist,
          "meta": {"type": "distance", "gt_box": [x1, y1, x1+s, y1+s]}}
    img.save(f"images/geo_{i:03d}.jpg")
    return qa
```

### 2.2 Prompt 结构与 Verifier 挂载

按学习计划的模板（写进数据集的 prompt 字段或 verl 的 chat template）：

```text
System: You are a visual reasoning assistant. Think step by step inside
        <think>...</think> and provide the final precise answer in <answer>...</answer>.
User: <image>
      Question: 红方块与蓝圆的水平距离是多少像素（取整）？
```

Verifier Engine（第 2 周）新增距离/计数判定路由，挂载为 verl 的 custom reward function：

```python
# my_verifier.py — verl 的 function-based reward 入口签名
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """verl 每条 rollout 调一次; ground_truth 来自 parquet 的 gt 列"""
    engine = RewardEngine()                      # 第2周的引擎 (模块级单例)
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    total, breakdown = engine.reward(solution_str, gt["answer"] if isinstance(gt, dict) else gt)
    return total
```

### 2.3 verl 训练配置要点（多模态差异项）

```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=geo_vlm/train.parquet \
  data.max_prompt_length=2048 \            # 图像 Token 已计入! (第3周失效3)
  data.max_response_length=768 \
  actor_rollout_ref.model.path=Qwen/Qwen2-VL-2B-Instruct \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.multi_turn.enable=false \
  trainer.total_epochs=1 ...               # 其余沿用第3周Demo
```

（官方提供 `run_qwen2_5_vl_*.sh` 多模态 GRPO 示例，字段以其为准；上面的差异项——prompt 预算、VLM 路径、rollout 设置——是跨版本稳定的思考维度。）

### 2.4 训练前体检（比训练本身更重要）

```python
"""rollout 体检: GRPO 开跑前确认数据/模型/验证器三方兼容"""
def preflight(model, processor, samples, engine, G=8):
    import torch
    n_solvable, n_fmt = 0, 0
    for s in samples[:30]:
        outs = generate_g(model, processor, s, G)          # G 条采样
        rewards = [engine.reward(o, s["gt"])[0] for o in outs]
        n_solvable += (max(rewards) > 0.5)                 # pass@G > 0.5 分
        n_fmt += sum(r for r in rewards if r > 0) / max(1, sum(1 for _ in outs))
    print(f"pass@G≈{n_solvable/30:.0%}  平均奖励={n_fmt/30:.2f}")
    assert n_solvable / 30 >= 0.3, "pass@G 过低: 换底座/降难度/先做冷启动 SFT"
    print("✅ 体检通过, 可以开训")
```

**pass@G ≥ 30%~50% 是开训门槛**：低于此值时 RLVR 只会浪费 GPU（第 1 周失效 1 的预防性体检）。

### 2.5 演变记录：第 0 / 100 / 500 步的三张快照

训练中每隔 N 步固化一份"同题快照"（固定 20 张测试图 + 温度采样 4 条），最终交付对比表：

| 步数 | `<think>` 均长 (Token) | 是否引用图中具体关系 | 任务准确率 | 复读占比 |
| --- | --- | --- | --- | --- |
| 0（SFT 底座） | 18 | 12% | 31% | 4% |
| 100 | 85 | 54% | 47% | 9% |
| 500 | 162 | 78% | 63% | 3% |

（示意数字；你的实测填入。）每行配 1 条完整样例（原图 + `<think>` 全文 + 答案对错），人工标注"涌现/复读"判定——这张表 + 样例就是 MVP 交付物 3。第 6-7 周会把判别升级为定量协议（打乱图片消融 + 中间量一致性检查）。

---

## 3. 工程权衡与失效模式

### 3.1 决策表：任务与奖励的匹配

| 任务 | GT 形态 | 奖励 | 难点 |
| --- | --- | --- | --- |
| 定位（Grounding） | 框坐标 | 软 IoU | 坐标格式解析鲁棒性 |
| 几何计算（距离/角度） | 数值 | 数值容差 | 中间量的格式自由度 |
| 计数 | 整数 | 精确匹配 | pass@G 低（计数是 VLM 弱项，需难度分层） |
| 空间关系（左右/上下） | 类别 | 精确匹配 | 真值视角约定（Stage 3 教训） |

### 3.2 三个代表性失效模式

**失效 1：坐标格式奖励主导——模型只会写框不会定位**
- **症状**：`R_format` 早早满分，`R_acc` 缓慢爬升但 IoU 一直低；输出的框是"语法正确、位置荒谬"的。
- **根因**：格式分太好刷（第 2 周失效 1 的多模态版）；或软 IoU 的 0.3 门槛太高（早期几乎拿不到 acc 分，格式分成了唯一梯度来源）。
- **定位**：reward breakdown 两项轨迹分叉 + 抽样看框的位置质量。
- **修复**：IoU 门槛降为 0.1~0.2 起步（课程化收紧）；$w_{\text{fmt}}$ 降到 0.1；或格式奖励改衰减制。

**失效 2：长度爆炸——"想得长"被误当"想得好"**
- **症状**：`response_length` 单调暴涨，reward 微涨，算力消耗翻倍，最终输出开始复读。
- **根因**：长 CoT 与答对率弱相关时（模型用更长探索碰对），长度被间接正选择；无长度惩罚时失控。
- **定位**：长度分布 P99 + "长度 vs 正确率"的分桶曲线（若长桶正确率不更高，长度是纯浪费）。
- **修复**：超长惩罚（第 2 周引擎已有，阈值按任务调）；或按步数截断采样；DAPO 式 overlong 缓冲带设计（截断惩罚只施加于越界部分）。

**失效 3：视觉 Prefix Cache 未生效——多 Rollout OOM**
- **症状**：训练正常跑但 rollout 阶段周期性 OOM，且同组 G 条的采样速度远低于预期。
- **根因**：同组 G 条的 prompt 图像 Token 未命中前缀缓存（prompt 文本里有随机元素导致前缀断裂——第 4 阶段 Prefix Caching 失效 2 的 RL 版）。
- **定位**：vLLM/SGLang 日志的 cache 命中率；对比"同组采样耗时 vs 单条 × G"。
- **修复**：保证同组 G 条的 prompt 前缀逐字节一致（随机性只放在采样温度上）；图像预处理输出确定性张量；`gpu_memory_utilization` 给足 KV 空间。

---

## 4. 延伸思考题

1. **涌现的因果检验**：设计一个"能力剥离实验"区分"RL 教会了新推理"与"RL 只提高了采样选择"——对比 RL 前后模型的 pass@1 与 pass@16。若 pass@16 几乎不变而 pass@1 大涨，说明什么？（提示：RL 没创造新能力，只是把"分布中已有的正确模式"变成高概率输出——这恰是 1.1 节机制假说的直接检验，也是当前 RLVR 研究的核心争论之一。）
2. **多模态奖励的放大器风险**：数学 RLVR 的验证器没有歧义，但空间推理的 GT（视角约定、距离定义）有构造自由度。分析一个案例：训练数据的"水平距离"定义为像素欧氏距离，部署后用户问的是"实际米数"——模型会怎么错？如何在数据构造期预防？（提示：GT 语义入 prompt；训练分布的语义广度决定 RL 后的泛化边界。）
3. **课程化 RL**：把 Stage 5 的难度分层思想移植到 RLVR——用 pass@G 把数据池分成"易中难"三档，按训练进度动态调整采样配比（易:中:难 从 5:4:1 演化到 1:4:5）。写出这个自适应课程的状态机设计，并说明它同时解决哪两个失效模式。（提示：太易组全对无信号、太难组全错无信号——课程化让有效组占比最大化，等价于把 Stage 5 的 Sweet Spot 变成动态追踪。）

---

*下一篇：[第 6-7 周：轨迹过滤与 Post-Training 三路消融](第6-7周-轨迹过滤与三路消融.md)*
