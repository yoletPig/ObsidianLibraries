# 第 4 周教程：筛选算法验证与 "Less is More" 消融实验

> **本周要回答的三个问题**
> 1. A/B/C 三组消融（全量 / 随机 30% / 筛选 30%）怎么设计才能让结论可信？
> 2. 打分阶段如何用 vLLM 把 10k+ 数据的 NLL 计算压缩到几十分钟？
> 3. "筛选 30% ≥ 全量 98% 性能"这个结论需要哪些证据链才算被证明？

对应学习计划：第 4 周。交付物：① 10,000 条混合数据集上导出 30% 随机子集与 30% 算法筛选子集；② 同参数微调后在 POPE/MMBench 上评测；③ 消融对比报告，证明筛选子集 > 随机子集、且 ≥ 全量性能的 98%。

---

## 1. 第一性原理：消融实验的证明责任

### 1.1 根本矛盾：你想证明的 vs 数据能证明的

你想证明的命题是："**筛选算法选择的 30% 子集，训练效果优于随机 30%，且逼近全量**"。这是一个关于**算法有效性**的因果命题。它的证明责任要求排除所有替代解释：

| 替代解释（混杂变量） | 排除手段 |
| --- | --- |
| "效果差只是因为数据少" | B 组（随机 30%）的存在——它是同预算对照 |
| "效果差异是训练随机性" | 锁种子 + 关键组复跑；报告方差 |
| "效果差异是超参巧合" | 三组完全同超参（同 lr/epoch/batch/模板） |
| "评测波动" | 固定评测协议（Stage 3 纪律）：框架版本/提取器/裁判归档 |
| "筛选过程泄漏了评测信息" | 打分只用参考模型对**训练数据**计算，绝不触碰评测集 |

**B 组（随机 30%）是整个实验的灵魂**。没有它，"筛选 30% 表现好"什么也证明不了——可能是数据少的普遍失败（需要 A 组参照），也可能是任何 30% 子集都行（需要 B 组证伪）。文献中不少"数据选择有效"的论文在加入随机基线后失效（XMAS 论文明确指出"现有方法无一能在不同子集规模上稳定超过随机选择"——arXiv:2510.01454），这就是 B 组的威慑力。

### 1.2 判定阈值的算术

学习计划的验收线："筛选 30% ≥ 全量 × 98%"。在基准分数上操作化：

$$
\text{retention} = \frac{\bar{S}_{\text{filtered}}}{\bar{S}_{\text{full}}} \geq 0.98
$$

其中 $\bar{S}$ 是各基准的均值（或加权均值）。三个必须写进报告的限定：

1. **多基准聚合口径**：单基准 98% 容易过/不过全凭运气，报告应给出逐基准 retention + 聚合 retention；
2. **噪声边界**：评测噪声（±0.5~1 分量级，Stage 3 经验）下，97.5% 与 98.5% 无实质差别——报告应附"随机 30% 的 retention"作对照，真正的证据是 **C 组 − B 组 的差值及其方向一致性**（在多数基准上 C > B），而非卡死 98% 这条线；
3. **成本维度**：筛选算法的打分成本（GPU 时）+ 训练节省的成本 = 净收益，"98% 性能 + 70% 训练成本节省"才是完整命题。

### 1.3 数据 Scaling 视角

三组消融只测了"30% 比例点"。更有信息量的做法是**扫描比例曲线**：筛选子集 vs 随机子集在 {10%, 20%, 30%, 50%, 100%} 各点训练评测，画两条性能-数据量曲线。预期形态：低比例区两曲线差距最大（筛选价值最凸显），高比例区收敛（数据多了随机也够）。**两条曲线的横向间距 = 筛选算法的"等效数据倍率"**（如"筛选 30% ≈ 随机 60%"），这是比 retention 更强的算法价值度量。预算允许时至少补 10% 与 50% 两个点。

---

## 2. 实现与验证

### 2.1 高效批量打分：vLLM 的 prompt_logprobs 模式

第 1 周用 transformers 原生前向算 NLL（口径正确但慢）；批量场景（10k+）迁移到 vLLM：`SamplingParams(prompt_logprobs=0)` 可在一次生成式调用中返回 prompt 每个位置的 logprob，配合预先拼好的"prompt+回答"序列与手工对齐的回答区偏移，即可批量取回答区 NLL。

```python
"""
vLLM 批量 NLL 打分 (回答区): 10k+ 数据的分钟级打分。
运行方式: 先启动 vLLM 服务 (见第4阶段教程), 再:
  python stage5_week4_vllm_score.py --data sft_10k.jsonl --out scores_10k.jsonl
依赖: vllm (或 openai 客户端连服务), transformers (仅 tokenizer)
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="scores_10k.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model)
    # 打分用纯文本镜像: 多模态版把 <图像占位> 保留, offline LLM 直接喂多模态输入
    # (此处演示文本主干口径; 完整多模态打分用 vllm 的 multi_modal_data 输入, 结构相同)
    llm = LLM(model=args.model, max_model_len=4096,
              enable_prefix_caching=True)          # chat 模板前缀缓存 (Stage4 红利)

    rows = [json.loads(l) for l in open(args.data)]
    prompts, spans = [], []                          # spans: 回答区 [start, end) token 下标
    for r in rows:
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": r["question"]}]}]
        p = processor.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True)
        tok_p = processor.tokenizer(p, add_special_tokens=False)["input_ids"]
        tok_a = processor.tokenizer(r["answer"], add_special_tokens=False)["input_ids"]
        prompts.append({"prompt_token_ids": tok_p + tok_a})
        spans.append((len(tok_p), len(tok_p) + len(tok_a)))

    sp = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0)  # 只要 prompt logprobs
    outs = llm.generate(prompts, sp)

    import math
    with open(args.out, "w") as f:
        for r, o, (s, e) in zip(rows, outs, spans):
            lp = o.prompt_logprobs                     # 长度 = prompt 长度, [0]=None
            if lp is None or e - s < 1:
                continue
            nlls = []
            for t in range(s, e):
                d = lp[t]                              # {token_id: Logprob}
                nlls.append(-next(iter(d.values())).logprob)
            avg = sum(nlls) / len(nlls)
            f.write(json.dumps({**r, "nll": round(avg, 4),
                                "ppl": round(math.exp(avg), 2)}, ensure_ascii=False) + "\n")
    print(f"打分完成 -> {args.out}")


if __name__ == "__main__":
    main()
```

**量级参考**（须以实测为准）：7B 模型、10k 条 × ~1.5k Token、单卡 A100 + prefix caching，打分约 10~40 分钟——比 transformers 逐批前向快 3~10 倍（主要来自 continuous batching 与 KV 复用）。**口径一致性警告**：vLLM 与 transformers 的数值可能有微小差异（kernel/精度实现不同），**同一次筛选实验内的所有打分必须用同一引擎**，不要 A 组用 transformers 打分、C 组用 vLLM 打分。

### 2.2 三组消融的执行清单

```bash
# 数据准备
python - <<'PY'
import json, random
rows = [json.loads(l) for l in open("sft_10k.jsonl")]
coreset = {json.dumps(r, sort_keys=True) for r in map(json.loads, open("coreset_30.jsonl"))}
random.Random(0).shuffle(rows)
rand30 = rows[:3000]
# 过滤式抽取保证三组互斥命名清晰 (训练脚本按目录读)
json.dump(rows, open("exp_A_full.json", "w"), ensure_ascii=False)
json.dump(rand30, open("exp_B_random30.json", "w"), ensure_ascii=False)
json.dump([r for r in rows if json.dumps(r, sort_keys=True) in coreset],
          open("exp_C_filtered30.json", "w"), ensure_ascii=False)
PY

# 训练: 完全同配置, 仅数据字段不同 (LLaMA-Factory YAML 三份, diff 只有 dataset/output_dir)
for g in A_full B_random30 C_filtered30; do
  FORCE_TORCHRUN=1 llamafactory-cli train cfg_${g}.yaml
done

# 评测: 同协议跑 POPE / MMBench (Stage 3 第 2 周流程)
for g in A_full B_random30 C_filtered30; do
  python run.py --model saves/${g}/merged --data POPE MMBench \
    --work-dir evals/${g} --nproc 1
done
```

**配置一致性核对表（跑之前逐项打勾）**：

- [ ] 三份 YAML 逐字段 diff，仅 `dataset` 与 `output_dir` 不同；
- [ ] 同一随机种子、同一底座 checkpoint、同一框架 commit；
- [ ] 评测的 work-dir 全新（防 Stage 3 失效 3 的缓存复用）；
- [ ] C 组子集确实来自第 3 周 Pipeline 产物（文件 hash 归档）；
- [ ] B 组随机种子记录在案（换种子复跑时用 2~3 个）。

### 2.3 消融报告模板

```markdown
# Less-is-More 消融报告 v1
参考模型: Qwen2-VL-2B (打分引擎 vLLM commit xxx) | 筛选: NLL两端(5%/20%) + KMeans(k=10)
训练: Qwen2-VL-7B LoRA r=16, lr 1e-4, 2ep, seed 0/1/2 | 评测: VLMEvalKit <commit>, POPE/MMBench

| 组 | 数据量 | POPE F1 | MMBench | 聚合 | retention |
|---|---|---|---|---|---|
| A 全量 | 10000 | 0.868 | 68.4 | 基准 | 100% |
| B 随机30% | 3000 | 0.851 | 66.9 | -2.1% | 97.6% |
| C 筛选30% | 3000 | 0.866 | 68.1 | -0.4% | 99.4% |

结论: C > B 于 2/2 基准 (方向一致); C retention = 99.4% ≥ 98% 达标。
成本: 打分 35 min (单卡) + 训练 0.3x 全量 => 等效节省 ~2.1x 端到端成本。
复现: 脚本清单 + 种子 + YAML diff 附后。
```

三段式结论纪律：**方向性**（C 是否稳定优于 B）、**达标性**（retention 是否 ≥98%）、**经济性**（打分成本 vs 训练节省的净账）。缺任何一段，"Less is More"都只是口号。

---

## 3. 工程权衡与失效模式

### 3.1 决策表：消融规模的选择

| 预算 | 实验设计 | 说明 |
| --- | --- | --- |
| 极小（教学验证） | 三组 × 1 种子 × POPE 单基准 | 1 天内完成，先看方向 |
| 标准（本周 MVP） | 三组 × 2~3 种子 × 2 基准 | 报告方差与方向一致性 |
| 充分（论文级） | 比例扫描 {10,30,50,100}% × 3 种子 × 4 基准 | 画 Scaling 曲线与等效倍率 |

### 3.2 三个代表性失效模式

**失效 1：单种子单基准下的"伪显著"**
- **症状**：C 比 B 高 0.3 分，宣布算法有效；换种子后反转。
- **根因**：评测噪声 ±0.5~1 分与训练随机性叠加，单次运行的差异完全在噪声带内（Stage 3 第 3 周失效 2 的同源问题）。
- **定位**：同配置换 2~3 个种子，看差值的符号稳定性；做配对比较（同种子下 C-B）而非独立比较。
- **修复**：报告均值±方差与方向一致率；结论限定"在本数据集/底座/预算下"。

**失效 2：筛选子集与全量数据的"隐性泄漏"使 C 组虚高**
- **症状**：C 组在某基准上异常突出，甚至超过 A 组。
- **根因**：打分阶段若曾用任何与评测相关的信号（如用同源模型在评测集上的表现校准阈值），筛选就泄漏了评测先验；或 Coreset 恰好富集了与评测集同分布的数据（若打分数据与评测数据同源抓取）。
- **定位**：审计打分管线的输入输出——确认除参考模型 NLL 外无外部信号混入；对比三组子集与评测集的 CLIP 分布距离。
- **修复**：打分输入与评测集物理隔离；C>A 的结果要格外警惕并复跑验证（超全量通常意味着泄漏或噪声，DEITA 式"超过全量"结论需极强的对照支撑）。

**失效 3：训练步数未对齐，"数据少 = 训练少"混淆变量**
- **症状**：B/C 组欠训练（30% 数据 × 同 epoch = 30% 步数），性能差被归因于"数据选择无用"。
- **根因**：同 epoch 配置下小数据组可见数据遍历次数相同但总步数少，收敛不充分；反之配同步数则过拟合。
- **定位**：看三组 WandB 训练曲线——B/C 组的 eval loss 是否已进入平台期。
- **修复**：主实验固定同 epoch（遍历次数公平）；附加"同总步数"的辅助组消除歧义；报告两种口径下结论的一致性。

---

## 4. 延伸思考题

1. **等效数据倍率**：完成 {10%, 30%, 50%} 三点的比例扫描后，插值出"筛选子集达到随机子集 X% 性能时的等效倍率"。这个倍率在 10% 点大还是 50% 点大？为什么？（提示：低比例区筛选的"挑精华"空间最大，倍率最高；随比例上升两种选择的集合趋同，倍率收敛到 1。）
2. **筛选的过拟合**：筛选算法的超参（截断比例、K、ΔL 权重）是拿"下游训练效果"调优的——这本身构成对评测集的选择性压力。设计一个"筛选超参的留出协议"，防止算法在评测集上被隐式调参。（提示：超参调优用一组基准，最终报告换到从未参与调参的 held-out 基准；或按时间切分——用 2024 前基准调参，2024 后基准报告。）
3. **反事实思考**：假如你的筛选算法在 B 组对照下毫无优势（C ≈ B），有哪些可能的解释？至少列四个，并设计各自的最小判别实验。（提示：数据本身冗余度低（随机删也删不到冗余）；筛选代理与真实训练价值脱相关；训练对数据质量不敏感（模型容量/训练长度主导）；或——最有趣的一种——你的数据集已经是"筛过的"（Stage 4 清洗后），筛选的边际收益被前置消耗了。）

---

*下一篇：[阶段五自测验收与复盘](阶段五自测验收与复盘.md)*
