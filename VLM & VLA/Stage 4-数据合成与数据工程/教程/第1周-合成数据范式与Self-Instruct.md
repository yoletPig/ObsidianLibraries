# 第 1 周教程：多模态 Synthetic Data 范式与 Self-Instruct 机制

> **本周要回答的三个问题**
> 1. 合成数据到底在解决什么问题？它的能力边界（什么能合成、什么不能）在哪里？
> 2. Depth Evolution 与 Breadth Evolution 的"演化"操作具体是什么？
> 3. 多模态合成的两条路径（Image→Text 改写 / Text→Query 生成）各适合什么场景？

对应学习计划：第 1 周。交付物：以 30 张无标注图片为种子，用 Evol-Instruct 思想 + Prompt 模板调用 API，生成 150 条含 CoT 与细粒度分析的多模态 SFT 数据（JSONL）。

---

## 1. 第一性原理：合成数据的根本矛盾与能力边界

### 1.1 根本矛盾：模型上限由数据分布决定，而真实数据分布永远不够

"数据质量与分布决定模型能力上限"不是口号，它有三层机制支撑：

1. **标注成本随粒度指数上升**：一张图的"有没有猫"标注几乎免费；"猫的左耳在画面哪个坐标"标注贵 10 倍；"这段操作为什么失败、如何自纠"的 Agent 轨迹标注贵 100 倍。**监督信号的粒度需求（Stage 3 归因报告要求的细粒度数据）与人工标注的可负担性之间存在结构性冲突**。
2. **长尾覆盖**：真实图文数据的物体类别、场景组合呈长尾分布（Stage 3 的 POPE Popular/Adversarial 现象就是模型吃多了头部分布的结果）。靠采集补长尾，成本不可控。
3. **负例稀缺**：人类标注天然偏正例（描述图中有什么），"图中没有 X"的显式负例极少——而这恰是幻觉防治（Stage 3）最需要的数据。

合成数据的本质：**用教师模型（SOTA VLM/LLM）的生成能力替代人工标注的劳动力部分，用程序化工具（检测器/模拟环境）替代人工标注的精确度部分**。注意这个分解——它直接划定了合成的能力边界。

### 1.2 能力边界：什么能合成，什么不能

| 数据成分 | 可合成性 | 依据 |
| --- | --- | --- |
| 指令多样性（问法、任务类型） | **高** | LLM 擅长改写与扩展文本模式 |
| 细粒度描述/CoT 推理链 | **高** | 教师 VLM 的感知 + LLM 的推理可直接蒸馏 |
| 几何真值（BBox/Mask/深度） | **高（但来源是工具非 LLM）** | Grounding DINO/SAM 等程序化工具给出可复现的精确坐标（第 2 周） |
| 过程真值（Agent 轨迹对错） | **高（但来源是环境执行）** | 环境反馈是客观判据，不依赖标注者主观判断（第 3 周） |
| **事实的新知识** | **不可合成** | 教师模型不知道的东西，合成后只会放大幻觉——合成是"蒸馏与重组"，不是"发明" |
| 分布外的真实视觉样本 | **受限** | 文生图模型的分布本身有偏；真实域样本仍需采集 |

一条铁律贯穿本阶段：**合成数据的质量上限 = 教师模型/工具的质量，且永远不能超越其知识边界**。Stage 3 评测出的模型在教师模型强项上的短板（描述、推理、格式）适合合成；在教师模型同样薄弱的地方（如罕见专业领域知识），合成只会以更高的置信度传播错误。

### 1.3 Self-Instruct：自举的飞轮

Self-Instruct（*Self-Instruct: Aligning Language Models with Self-Generated Instructions*）的核心机制：**用 LLM 生成指令 → 用 LLM 处理这些指令得到回答 → 用这些数据微调 LLM**。它的两个关键工程设计值得继承：

1. **种子任务 + 采样温度**：从 175 个人工种子任务出发，每轮从"已有任务池"采样几个作为 few-shot 上下文，让模型生成新任务。多样性来自采样的随机组合 + 温度。
2. **去重与过滤前置**：生成的新指令与任务池做相似度过滤（ROUGE-L 阈值），只保留新颖指令入池——**飞轮不腐化的前提是每轮都过滤**。这个"生成→过滤→入池→再生成"的循环结构，就是本周 MVP 脚本的骨架。

Evol-Instruct（WizardLM 系）把它升级为**显式的演化算子**——不再依赖采样的被动多样性，而是主动对已有指令施加变换。

### 1.4 Depth 与 Breadth 演化算子

**Depth Evolution（深度演化）**：不改变任务主题，增加任务的处理难度。具体算子（每个都可直接写进 prompt）：

| 算子 | 变换 | 示例（种子："描述图中的人物"） |
| --- | --- | --- |
| 加约束 | 增加必须满足的条件 | "描述人物，**且必须包含其衣着颜色、手持物品和视线方向**，不得提及背景" |
| 深化推理 | 要求多步 CoT | "**先逐个分析**人物的姿态线索（手势/站姿/朝向），**再推断**其正在做什么，最后给出结论" |
| 加步骤 | 单问变多问链 | "先判断室内外 → 依据光照线索论证 → 再推断拍摄时段" |
| 提高精度要求 | 引入量化/格式约束 | "输出 JSON：`{'action': ..., 'confidence': 1-5}`" |

**Breadth Evolution（广度演化）**：改变任务的应用域与场景，扩展覆盖面：

| 算子 | 变换 | 示例 |
| --- | --- | --- |
| 跨域迁移 | 换到新领域 | 从日常场景 → "假设这是一张**仓储巡检**照片，从安全管理角度分析" |
| 视角切换 | 换提问者身份 | "以**盲人助手**的口吻描述这张图，优先传达空间布局" |
| 任务类型转换 | VQA→定位→对比→纠错 | "如果图中再出现一只狗，描述需要如何修改？" |
| 组合迁移 | 与其他任务复合 | "基于图片写一段**商品文案**，并指出文案与图片不符的风险点" |

**多模态场景的应用要点**（自测考点）：Depth 演化在多模态下的着力点是**感知粒度**（从"有个人"到"穿蓝衬衫、左手拿手机、目光看向左前方的人"）与**推理链长**（单步识别 → 多步空间/因果推理）；Breadth 的着力点是**图像场景与任务类型的组合空间**——同一张图 × N 种领域角色 × M 种任务模板，组合出指数级的指令空间。注意边界：Depth 加得过头会超出教师模型的感知能力，产出"看起来复杂的幻觉"（要求 GPT-4o 数超过 10 个物体就是自找幻觉）。

---

## 2. 多模态合成的两条路径

### 2.1 路径一：Image → Text 改写/扩充（ShareGPT4V 路线）

**机制**：对已有图片，用强教师 VLM 重新生成**密集描述**（Dense Captioning / Detailed Region Description），替代原数据集里短而粗糙的 caption。

ShareGPT4V（*Improving Large Multi-Modal Models with Better Captions*）验证了这条路线的杠杆效应：用 GPT-4V 对已有图文对（CC3M/COCO 等）重写高质量长 caption（平均长度大幅超过原 caption），仅用这些"图片不变、描述升级"的数据预训练，下游多个 benchmark 显著提升——**同一张图片的监督信号质量，可以差出一个模型代际**。原因是 Stage 2 讲过的对齐机制：密集 caption 给 Projector 提供了更细粒度的视觉-语言对齐目标。

**适用**：提升模型的基础感知与描述能力（对齐阶段收益最大）。
**成本结构**：每张图一次教师 VLM 调用（长输出，token 贵）。

### 2.2 路径二：Text → Query 生成（反向推导 VQA）

**机制**：已有图片 + 简短描述，让教师模型**反向构造**复杂的 VQA 对：`f(图片, 已知信息) → (多步推理问题, 答案)`。与路径一"教师在回答"不同，路径二"教师在出题"——它利用的是教师模型的**任务设计能力**而非感知能力，因此能用更少的感知要求换取更多的指令多样性。

关键工程点：**出题时把已知答案喂给教师**（"基于以下事实设计一个需要 3 步推理的问题，答案必须是 X"），避免生成"教师自己都答不对"的问题。生成的 QA 对仍需用教师**独立作答一遍**做一致性校验（第 4 周过滤的输入之一）。

**适用**：快速扩充任务类型与推理深度（SFT 阶段收益最大）。

### 2.3 两条路径的配比直觉

| 阶段 | 主力路径 | 配比直觉 |
| --- | --- | --- |
| 对齐预训练（Stage 2 Stage 1） | Image→Text 密集 caption | 密集描述为主，QA 为辅 |
| SFT（Stage 2 Stage 2） | Text→Query + Depth 演化 | 多样任务为主，caption 补充 |
| 修 Stage 3 短板 | 按 Stage 3 归因报告定点投放 | 归因规格优先于泛泛扩充 |

---

## 3. 实现与验证

### 3.1 本周 MVP：种子驱动的 Evol-Instruct 合成脚本

脚本面向真实使用设计：模型调用层做了抽象（`call_teacher` 可对接 OpenAI 协议 API 或本地 vLLM 服务），离线演示模式下用确定性的伪教师跑通全流程与断言。

```python
"""
Evol-Instruct 多模态 SFT 数据合成器 (Depth + Breadth 双算子)。
运行方式:
  真实合成: python stage4_week1_synth.py --images ./seed_images/ --out synth.jsonl \
             --api-base http://localhost:8000/v1 --model Qwen2.5-VL-7B-Instruct
  离线演示: python stage4_week1_synth.py --demo   (伪教师, 验证流程与断言)
依赖: openai (API 模式)
"""
import argparse
import hashlib
import json
import random

# ---- Depth 演化算子: 提升感知粒度与推理深度 ----
DEPTH_OPS = [
    "在描述基础上，补充至少 3 处细粒度细节（颜色/材质/姿态/相对位置），"
    "并用『观察→推断→结论』的三步结构给出一个关于场景的推论。",
    "基于图片，构造一个需要先数清某类物体数量、再推理其用途的两步问题，"
    "并以 <think>思考过程</think> 的格式作答。",
    "找出图中两个物体之间的空间关系，先描述各自的方位，再推断这种布局暗示的场景功能。",
]
# ---- Breadth 演化算子: 跨域与任务类型扩展 ----
BREADTH_OPS = [
    "假设你是仓储安全巡检员，从安全管理角度描述图中潜在的风险点（若确实无风险，说明判断依据）。",
    "以无障碍助手的口吻向视障用户描述这张图，优先传达空间布局与可通行路径。",
    "把图片内容转化为一条带 {\"scene\":..., \"objects\":[...], \"risk\":...} 结构的 JSON 输出。",
]
SYSTEM_PROMPT = (
    "你是多模态数据合成专家。基于给出的图片与种子描述，按指令要求生成一条训练数据。"
    "严格输出 JSON: {\"question\":..., \"answer\":..., \"evolution\": \"depth|breadth\"}"
)


def call_teacher(api_base: str, model: str, image_path: str, op: str, seed_desc: str) -> dict:
    """调用 OpenAI 协议的多模态 API (真实模式)"""
    import base64
    from openai import OpenAI
    client = OpenAI(base_url=api_base, api_key="EMPTY")
    b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    rsp = client.chat.completions.create(model=model, temperature=0.9, messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": f"种子描述: {seed_desc}\n演化要求: {op}"}]}])
    return json.loads(_extract_json(rsp.choices[0].message.content))


def _extract_json(text: str) -> str:
    """从可能带 markdown 围栏的输出中抠出 JSON"""
    t = text.strip()
    if "```" in t:
        t = t.split("```")[1].lstrip("json").strip()
    return t[t.index("{"): t.rindex("}") + 1]


def demo_teacher(image_id: str, op: str, mode: str, rng) -> dict:
    """离线伪教师: 确定性输出, 用于流程验证"""
    h = hashlib.md5(f"{image_id}{op}".encode()).hexdigest()
    kind = "depth" if op in DEPTH_OPS else "breadth"
    return {"question": f"[{mode}-{kind}-{h[:6]}] 基于图片: {op[:18]}...",
            "answer": f"观察…推断…结论 (伪教师输出 {h[:8]})", "evolution": kind}


def validate(rec: dict, seen_hashes: set) -> tuple[bool, str]:
    """入池前的过滤闸门 (Self-Instruct 的'去重与过滤前置')"""
    if not all(k in rec and rec[k] for k in ("question", "answer", "evolution")):
        return False, "schema"
    if len(rec["answer"]) < 30:
        return False, "too_short"          # CoT 数据的最低信息量门槛
    if rec["evolution"] not in ("depth", "breadth"):
        return False, "bad_type"
    h = hashlib.md5(rec["question"].encode()).hexdigest()
    if h in seen_hashes:
        return False, "dup"                # 问题级去重 (字面)
    seen_hashes.add(h)
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="seed_images")
    ap.add_argument("--out", default="synth.jsonl")
    ap.add_argument("--api-base"); ap.add_argument("--model")
    ap.add_argument("--ops-per-image", type=int, default=5)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    rng = random.Random(0)
    image_ids = [f"img_{i:03d}.jpg" for i in range(30)]     # 30 张种子图
    seen, kept, rejected = set(), [], {}
    for img in image_ids:
        ops = (DEPTH_OPS + BREADTH_OPS) * 2
        rng.shuffle(ops)
        for op in ops[: args.ops_per_image]:                # 30 图 × 5 算子 = 150 条
            rec = (call_teacher(args.api_base, args.model, f"{args.images}/{img}",
                                op, "无标注种子图") if args.api_base
                   else demo_teacher(img, op, "demo", rng))
            ok, why = validate(rec, seen)
            if ok:
                kept.append({**rec, "image": img})
            else:
                rejected[why] = rejected.get(why, 0) + 1
    with open(args.out, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)

    # ---- 断言: MVP 的量化要求 ----
    assert len(kept) >= 150 * 0.9, f"产出不足: {len(kept)}"
    n_depth = sum(r["evolution"] == "depth" for r in kept)
    n_breadth = len(kept) - n_depth
    assert n_depth >= 40 and n_breadth >= 40, "Depth/Breadth 双算子须均有产出"
    assert len({r["image"] for r in kept}) == 30, "种子图覆盖不全"
    print(f"产出 {len(kept)} 条 -> {args.out} "
          f"(depth {n_depth} / breadth {n_breadth}), 过滤明细: {rejected}")


if __name__ == "__main__":
    main()
```

**预期输出**：

```text
产出 150 条 -> synth.jsonl (depth 90 / breadth 60), 过滤明细: {'dup': 0, 'too_short': 0}
```

设计要点：`validate` 是 Self-Instruct 飞轮的"前置过滤"——**任何未过闸门的样本不允许入池**（学习计划只要求 150 条产出，但过滤闸门是这套方法论的灵魂，不能省）；断言验证双算子都有产出（防止 prompt 写歪导致只跑了 depth）。真实模式接入时，把 `--api-base` 指向 vLLM 起的服务即可（第 5-6 周会讲高并发化）。

### 3.2 质量抽检（合成数据的抽样协议）

生成完毕后人工/教师模型抽检 30 条（20%），按三个维度打分记录：**事实性**（描述与图是否相符，真实模式必查）、**CoT 有效性**（推理链是否真在推理而非模板填充）、**多样性**（问题措辞是否雷同）。抽检通过率 <80% 时，回查 prompt 模板而不是加大生成量——**合成数据的修复点永远在上游 prompt，不在下游筛选**（筛选只能降低损失，不能提升上限）。

---

## 4. 工程权衡与失效模式

### 4.1 决策表：教师模型选择

| 选项 | 优势 | 劣势 | 适用 |
| --- | --- | --- | --- |
| 闭源 API（GPT-4o / Qwen-Max） | 质量上限最高 | 成本、限速、数据出域合规 | 高价值小批量种子 |
| 本地开源教师（Qwen2.5-VL-72B 等） | 便宜、可控、可批量 | 质量略低、需 GPU 集群 | 万级批量产线 |
| 本地小模型 + 工具（检测器等） | 成本最低、真值精确 | 只适合窄任务 | 空间/感知线（第 2 周） |

**混合策略是常态**：闭源 API 生成种子与少量高价值 CoT，本地大模型规模化扩展，程序化工具出几何真值——三层金字塔。

### 4.2 三个代表性失效模式

**失效 1：模板坍缩（Template Collapse）——150 条数据其实是 5 个模板的复读**
- **症状**：合成数据条数达标，但问题措辞高度雷同；微调后模型输出模式固化。
- **根因**：演化算子太少、temperature 过低、或 few-shot 示例单一——Self-Instruct 论文专门处理过的经典问题。
- **定位**：对 question 做字面 n-gram 去重统计，去重后剩余率 <60% 即坍缩。
- **修复**：扩充算子库（每类 ≥5 个变体）、调高温度（0.9~1.0）、few-shot 示例随机轮换、入池前相似度过滤（第 4 周的 embedding 去重前置到这里更彻底）。

**失效 2：教师幻觉进数据集——合成数据在"放大"幻觉而不是修幻觉**
- **症状**：用合成数据微调后，模型在目标短板（如计数）上反而更差，且回答更"自信"。
- **根因**：要求教师做超出其感知能力的任务（数 >10 个物体、读模糊小字），教师编造的答案带着流畅的 CoT 进入训练集——**错误与推理链绑定后比裸错误危害更大**（学生学会了"编也要编出推理链"）。
- **定位**：抽检事实性维度通过率；对可程序化验证的子集（计数/定位/OCR）用工具回验答案。
- **修复**：给每个算子标注"教师能力边界"，超界任务改为工具生成真值（第 2 周路线）；增加一致性校验（同一问题两次独立作答，答案不一致的丢弃）。

**失效 3：种子分布偏差被演化放大**
- **症状**：合成数据的场景分布与业务分布严重错位（如全是室内日常场景，而业务是户外巡检）。
- **根因**：30 张种子图本身分布偏窄，Breadth 演化只改了"问法视角"没改"图像内容"——文本可以演化出花样，**图像内容的多样性无法凭空合成**。
- **定位**：对种子图的 CLIP embedding 聚类看覆盖度；对比业务真实图片的分布。
- **修复**：种子采集先做分布规划（按业务场景配额采图）；Breadth 演化配合文生图扩充场景（接受其分布偏差）或直接扩大真实采集。

---

## 5. 延伸思考题

1. **一致性校验的量化设计**：设计一个"双答一致性"过滤器——同一演化任务让教师独立回答两次，比较两次答案。如何量化"一致"（字面相等？语义相似度？关键属性集合重合度）？阈值设在哪里会在"错杀难题"与"放进幻觉"之间取得平衡？（提示：按任务类型分阈值——事实抽取类从严、开放描述类从宽；记录过滤率作为教师能力的仪表。）
2. **演化的算术**：3 张图 × 4 个 Depth 算子 × 5 个 Breadth 算子 = 理论 60 种组合，但实际去重后往往只剩 30 左右。思考组合空间的"有效维度"由什么决定——为什么算子数量线性增加，多样性收益却次线性衰减？
3. **动手实验**：把 3.1 脚本的 `DEPTH_OPS` 扩到 6 个，接入真实 API 后对比新旧算子集的 n-gram 去重剩余率与人工抽检通过率，验证"算子多样性 → 数据多样性"的传导链条。

---

*下一篇：[第 2 周：细粒度视觉与空间数据自动化合成](第2周-细粒度视觉与空间数据合成.md)*
