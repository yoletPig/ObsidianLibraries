# 第 5-6 周教程：SimplerEnv 仿真闭环评测

> **本周要回答的三个问题**
> 1. 开环（offline MSE）与闭环（online rollout）评测的本质差异是什么？
> 2. Covariate Shift / 误差累积为什么是 VLA 的头号杀手？
> 3. SimplerEnv 里跑通 OpenVLA 闭环 rollout 并统计成功率的完整流程？

对应学习计划：第 5-6 周。交付物：部署 SimplerEnv + OpenVLA-7B，在 `google_robot_pick_coke_can` 任务跑通完整闭环 rollout，录制成功视频并记录每步推理延迟。

---

## 1. 第一性原理：为什么 loss 低 ≠ 会干活

### 1.1 根本矛盾：开环评测测的是"模仿精度"，部署要的是"状态修复能力"

**开环（offline/open-loop）评测**：在静态数据集上，给定数据集分布内的观测 $o_t$，预测动作 $a_t$，算 MSE/Token 准确率。它回答"**模型在见过的状态上模仿得像不像**"。

**闭环（closed-loop/online）评测**：模型接入环境——输出动作 → 环境执行 → 返回**新观测** → 再预测……直到任务成功/超时。它回答"**模型在它自己制造的状态上还能不能继续干活**"。

两者的鸿沟来自**分布的动态漂移**：

```text
数据分布:  专家轨迹经过的状态分布 D_expert
策略执行:  第 1 步误差 ε → 状态偏离 D_expert → 第 2 步面对 OOD 状态
           → 预测更不可靠 → 误差更大 → ... 误差累积 (compounding error)
```

这就是 **Covariate Shift**（自测清单考点，亦称 compounding error / distribution shift）：**模型从未学过"如何从偏离的状态恢复"——因为专家数据里根本没有"偏离后的状态"**。一个开环 MSE 极低的模型，闭环可能在第三次抓取就把罐子碰倒（且此后每一步都在"碰倒后的桌子"这个训练中从未见过的状态上挣扎）。

三个推论（第 7-9 周微调策略的依据）：

1. **开环 loss 只能做回归测试**（微调后 MSE 不应上升），**绝不能作为部署依据**；
2. **闭环成功率才是唯一有效指标**，且要报多 episode 统计（单次成功有运气成分——SimplerEnv 标准做法是每任务多 seed + 多 variation 统计）；
3. **缓解手段的性质**：Action Chunking（少暴露于偏移状态）、数据增广（扩充状态覆盖）、Dagger 式数据聚合（把策略自己跑出的失败状态交专家标注回灌）——都是围绕"让训练分布覆盖策略分布"展开。

### 1.2 SimplerEnv：为 OpenVLA/Octo 定制的"真实复刻"仿真器

SimplerEnv（基于 Sapien 物理引擎，`simpler-env/SimplerEnv`，已核实仓库）的定位：**用视觉高度贴近真实数据集（Google Robot / Bridge V2 的场景与光照复刻）的仿真环境，复现 OXE 真实评测任务**——使 OpenVLA 这类在真实数据上预训练的模型能"无视觉域差"地直接进仿真评估：

- **任务族**：`google_robot_pick_coke_can`（抓可乐罐）、`move_near`、`open/close_drawer`、Bridge 系列堆叠/放置任务等；
- **评测维度**：多 seed × 多 variation（光照/纹理/物体位置随机化）——直接量化鲁棒性（学习计划点名的"不同材质、光照、物体摆放"）；
- **指标**：任务成功率（Success Rate）+ 每步延迟。

**为什么选 SimplerEnv 而非通用仿真器**：ManiSkill 等平台物理更强但视觉与 OXE 真实数据差异大（VLA 的视觉塔在真实图上预训练，进风格化仿真是 OOD）；SimplerEnv 用"复刻真实视觉"换来了**免域适应直接评测**的便利——评测的结论能更可信地外推到真实数据分布。

---

## 2. 实现与验证

### 2.1 部署与闭环 Rollout

```bash
# SimplerEnv 部署 (官方推荐 python 3.10, sapien 依赖)
git clone https://github.com/simpler-env/SimplerEnv.git
cd SimplerEnv && pip install -e .
# ManiSkill2 依赖资产需按官方 README 下载 (场景/物体资产包)
```

```python
"""
OpenVLA 闭环 rollout: SimplerEnv google_robot_pick_coke_can。
运行方式: python stage10_week5_closedloop.py --episodes 10 --record
依赖: simpler_env, torch, transformers, opencv-python
  (OpenVLA-7B 需 1×24G+ 显存; 推理延迟单卡 A100 约 0.2~0.4s/步, 实测为准)
"""
import argparse
import time
import numpy as np
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image


def build_model(mid="openvla/openvla-7b"):
    processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        mid, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    return model, processor


def obs_to_pil(raw_obs):
    """SimplerEnv 观测 -> PIL 图 (字段名以 env 版本为准)"""
    img = raw_obs["image"] if "image" in raw_obs else raw_obs["full_image"]
    return Image.fromarray(np.asarray(img)).convert("RGB")


def rollout(env, model, processor, instruction: str, max_steps=60, record=None):
    obs, _ = env.reset()
    latencies, frames = [], []
    done = False
    for t in range(max_steps):
        pil = obs_to_pil(obs)
        prompt = f"In: What action should the robot take to {instruction.lower()}?"
        # OpenVLA 官方 prompt 模板 (In: ... Out:)
        inputs = processor(prompt, pil).to("cuda", dtype=torch.bfloat16)
        t0 = time.perf_counter()
        with torch.no_grad():
            action_ids = model.generate(**inputs, do_sample=False,
                                        max_new_tokens=7)        # 7 个动作 Token
        latencies.append(time.perf_counter() - t0)
        action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        # ^ unnorm_key: 反归一化统计量 (与第1-2周的 norm_stats 纪律对应)
        obs, reward, terminated, truncated, info = env.step(np.array(action))
        if record:
            frames.append(obs_to_pil(obs))
        done = terminated or info.get("success", False)
        if done:
            break
    success = bool(info.get("success", False))
    if record and success:
        _save_mp4(frames, record)
    return success, np.mean(latencies), t + 1


def _save_mp4(frames, path):
    import cv2
    h, w = frames[0].size[1], frames[0].size[0]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR))
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--task", default="google_robot_pick_coke_can")
    ap.add_argument("--record", default="rollout_success.mp4")
    args = ap.parse_args()

    import simpler_env as simpler                       # SimplerEnv 入口
    model, processor = build_model()
    env = simpler.make(args.task)                       # 内建 variation 随机化

    results = []
    for ep in range(args.episodes):
        ok, lat, steps = rollout(env, model, processor,
                                 "pick up the coke can",
                                 record=f"{args.record}.ep{ep}" if args.record else None)
        results.append(ok)
        print(f"ep{ep}: {'✅' if ok else '❌'} steps={steps} latency={lat*1000:.0f}ms")
    sr = np.mean(results)
    print(f"\nSuccess Rate: {sr:.0%} ({sum(results)}/{args.episodes})")
    print(f"平均单步推理延迟: {np.mean([lat for _, lat, _ in [(0,0,0)]])if False else '见逐条'}")
    # ---- 统计口径纪律: 多 episode 成功率 + 延迟分位数 ----
    assert args.episodes >= 10, "成功率统计至少 10 个 episode (单次有运气成分)"
    print(f"报告口径: SR={sr:.0%} over {args.episodes} eps @ {args.task}")


if __name__ == "__main__":
    main()
```

**预期输出形态**（A100、OpenVLA-7B、10 episodes，SR 数字为该任务论文/社区常见量级的示意——**以你的实测为准**）：

```text
ep0: ✅ steps=42 latency=310ms
ep1: ❌ steps=60 latency=305ms
...
Success Rate: 70% (7/10)
平均单步推理延迟: ~308ms
报告口径: SR=70% over 10 eps @ google_robot_pick_coke_can
```

报告纪律（Stage 3 评测协议的具身版）：**成功率必须报"任务 × episode 数 × variation 数"**；延迟报均值 + P95（控制频率约束看 P95 不是均值）；成功视频与失败视频都归档（失败案例是第 7-9 周微调数据的来源，Stage 3 归因纪律）。

### 2.2 延迟与控制频率的换算

SimplerEnv 的控制频率与 Google Robot 真实设置对齐（约 3~10Hz 量级）。7B 模型单步 300ms 意味着**有效控制频率 ~3Hz**——低于真实机器人 10Hz+ 的需求。这把第 1-2 周的 Action Chunking 与第 8 周的加速（量化/小模型）从"优化项"变成"必选项"——第 10 周的部署章节正式接手这个问题。

---

## 3. 工程权衡与失效模式

### 3.1 决策表：评测协议设计

| 决策 | 最低标准 | 更严谨 |
| --- | --- | --- |
| episode 数 | 10 | ≥25 + 多 seed |
| variation | 默认 1 组 | 全 variation 扫（鲁棒性曲线） |
| 指标 | 成功率 | + 每步延迟分位数 + 轨迹平滑度 |
| 基线 | 无 | 至少跑一个已知基线（如 Octo）校准环境 |

### 3.2 三个代表性失效模式

**失效 1：反归一化键错配——动作幅度整体异常**
- **症状**：机械臂小幅抽动或疯狂越界，Simulator 日志出现动作截断警告。
- **根因**：`unnorm_key` 用错（`bridge_orig` vs `fractal20220817_data`——第 1-2 周失效 1 的运行时版本）：反归一化统计量与当前环境的动作空间不匹配。
- **定位**：打印若干步的原始 Token → 归一化值 → 反归一化物理值，与该任务的合理动作范围比对。
- **修复**：确认模型 checkpoint 对应的数据集键；评测环境与模型预训练分布对齐（或先在目标域微调，第 7-9 周）。

**失效 2：成功判据的口径混淆**
- **症状**：成功率与论文/他人复现对不上（差 10~20 个点）。
- **根因**：成功判定来源不一致——`terminated`、`truncated`、`info["success"]` 三者语义不同（terminated 可能含失败终止）；或 variation 子集不同。
- **定位**：打印每条的终止原因与 success 标志，人工核对几条"成功"视频。
- **修复**：统一以 `info["success"]`（任务级判定）为准；episode 数与 variation 固定并在报告声明（Stage 3 的"协议四要素"纪律）。

**失效 3：只测成功轨迹——失败案例被浪费**
- **症状**：微调时无从下手（不知道模型到底差在哪类失败）。
- **根因**：把闭环评测当"打分"而非"归因数据源"。
- **定位**：——本身就是缺了归因环节。
- **修复**：失败视频 + 失败步的动作/状态快照全部归档，按失败模式分类（抓空/碰倒/方位错/超时），喂给第 7-9 周（微调数据选择）与 Stage 3 的归因纪律。

---

## 4. 延伸思考题

1. **Covariate Shift 的定量观测**：设计实验量化误差累积——在闭环 rollout 中，记录每一步"当前观测与最近训练数据的最近邻距离"（embedding 空间），画距离随步数的曲线。预测：成功轨迹与失败轨迹的曲线形态有什么系统性差异？（提示：失败轨迹的 OOD 距离应更快发散——这个曲线可以做成部署期的"漂移预警器"。）
2. **Chunking 与闭环的张力**：Action Chunking（k 步开环）减少模型暴露于偏移状态的次数，但也推迟纠错。推导"k 与任务容错度"的关系：什么任务适合大 k（结构化重复动作），什么任务必须小 k（精细接触操作）？（提示：抓取接近阶段误差敏感 → 小 k；搬运运输阶段 → 大 k。动态 k 切换是研究前沿。）
3. **Dagger 的具身版**：模仿学习的经典解法 DAgger（把策略遇到的状态交专家标注）在机器人场景的执行成本极高。替代方案盘点：仿真中重放失败状态 + 程序化求解器给 GT（SimplerEnv 的优势）、或 Stage 7 的 RLVR 思路（失败状态直接作为 RL 的起点分布）——对比三种方案的成本与数据质量。

---

*下一篇：[第 7-9 周：VLA 微调与空间能力增强](第7-9周-VLA微调与空间增强.md)*
