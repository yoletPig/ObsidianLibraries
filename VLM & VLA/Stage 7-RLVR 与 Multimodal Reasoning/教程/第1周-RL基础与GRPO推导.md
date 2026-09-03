# 第 1 周教程：RL 基础、PPO 痛点与 GRPO 算法推导

> **本周要回答的三个问题**
> 1. LLM 语境下的 RL 基本量（State/Action/Reward）怎么定义？PPO 的显存卡点在哪？
> 2. GRPO 为什么不需要 Critic？组内相对优势的推导与含义是什么？
> 3. clip 与 KL 两个机制各自在防什么？_std+ε_ 里那个 ε 是干嘛的？

对应学习计划：第 1 周。交付物：纯 NumPy/PyTorch 实现独立的 `GRPO Advantage & Loss Calculator`，输入 5 个 Rollout 的奖励（如 `[1.0, 0.0, 0.0, 1.0, 0.5]`），推导组内 Advantage 并计算梯度，验证 KL 惩罚与 clip 的截断效果。

---

## 1. 第一性原理：RLHF 的量词映射与 Critic 的本质

### 1.1 LLM 语境下的 RL 基本量

把语言生成翻译成马尔可夫决策过程（MDP）的术语：

| RL 术语 | LLM 对应物 | 说明 |
| --- | --- | --- |
| 状态 State $s_t$ | Prompt + 已生成的前缀 $(q, y_{<t})$ | 部分生成的文本就是当前状态 |
| 动作 Action $a_t$ | 下一个 Token $y_t$ | 词表大小的离散动作空间（数万维） |
| 策略 Policy $\pi_\theta$ | 语言模型本身 | $\pi_\theta(a_t \mid s_t) = \pi_\theta(y_t \mid q, y_{<t})$ |
| 轨迹 Trajectory | 一次完整回答 $y$ | Token 级决策序列 |
| 奖励 Reward $R$ | 序列末的验证分 / RM 打分 | **稀疏**：只在序列结束才结算 |
| 回报 Return | $R$（单步结局） | 生成任务的折扣因子通常取 1 |

两个对后续一切推导至关重要的性质：

1. **奖励稀疏且序列级**：整条回答只有一个总分，无法直接知道"第 37 个 Token 是好是坏"；
2. **动作空间巨大**：每步在数万 Token 中选择，Critic（价值网络）要为每个状态估一个标量，本身就与策略同量级。

### 1.2 PPO 的显存卡点：四个模型的账本

第 6 周已经算过一次 PPO 的账，这里从 RL 系统视角重述（数字：7B bf16）：

| 组件 | 职责 | 驻留成本 |
| --- | --- | --- |
| Actor $\pi_\theta$ | 策略（被训练） | 14GB + 梯度 + 优化器状态 |
| Reference $\pi_{\text{ref}}$ | KL 正则的锚 | 14GB（冻结） |
| Reward $r_\phi$ | 打分（RLHF 路线） | 14GB（冻结） |
| **Critic $V_\psi$** | **估计每个状态的价值基线** | **14GB + 梯度 + 优化器状态（第二个全量可训练模型！）** |

Critic 的职能：策略梯度定理要求用优势（Advantage）$A_t = R_t - V(s_t)$ 替代原始回报，以降低梯度方差——$V(s_t)$ 就是"这个状态平均能拿多少分"的基线，减掉它，"好于平均"的动作才获得正梯度。**但为省方差而引入的 Critic，本身是一个与策略同规模的模型**：训练它需要自己的前向/反向/优化器状态，且它的估值误差会直接污染优势信号（价值模型学不好 = 梯度噪声更大）。

**GRPO 的第一性洞察**：在 LLM 的 RLVR 场景里，同一个 prompt 天然可以采样 $G$ 个回答——**这组回答的平均奖励就是一个现成的、无偏的基线估计**（蒙特卡洛基线）。用组内均值替代学习的价值网络，Critic 整个消失：省 14GB+ 的模型与其优化器状态，且不引入价值估计误差。这是"组"（Group）的全部含义——**Group 不是工程技巧，是把基线从"学习的"换成"采样的"**。

### 1.3 组内相对优势的推导与含义

对 prompt $q$ 采样 $G$ 个回答 $\{o_1, \dots, o_G\}$，奖励 $\{r_1, \dots, r_G\}$（RLVR 下多为 0/1 或分段分）。GRPO 的优势定义：

$$
A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r}) + \epsilon}
$$

逐项解读：

- **$r_i - \text{mean}(\mathbf{r})$**：以组内均值替代 Critic 的 $V(q)$——"这条回答比同组的平均好多少"。减去基线的目的（方差缩减）与 PPO 完全一致，只是基线来源从学习变为采样。
- **除以 $\text{std}(\mathbf{r})$**：把优势归一化到无量纲。奖励尺度变化（换 Verifier、调 reward 配比）不影响梯度尺度——**RL 训练对奖励的绝对尺度敏感、对相对排序不敏感**，归一化让超参（lr）跨任务可迁移。
- **$\epsilon$（如 $10^{-4}$）**：防除零保护项。当组内奖励几乎全同时（如全 0），std→0，若无 ε 则优势爆成巨大噪声；加 ε 后小 std → 大分母 → 优势被压向小值——**与"组内无差异则无学习信号"的语义自洽**（第 6-7 周的轨迹过滤正是这个性质的工程延伸）。

注意一个微妙点：GRPO 是 **outcome-level 优势**——组内每个回答共享一个 $A_i$，施加在该回答的**所有 Token** 上。它不做 Token 级 credit assignment（哪些 Token 是"关键推理步"）。这是简洁性的代价，也是 PRIME、过程奖励模型等后续工作的改进方向。

### 1.4 GRPO 损失：PPO 式 clip + KL 正则

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^{G} \min \left( \rho_i A_i,\ \text{clip}(\rho_i, 1-\epsilon_{\text{clip}}, 1+\epsilon_{\text{clip}}) A_i \right) + \beta\, D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})
$$

其中 $\rho_i = \frac{\pi_\theta(o_i \mid q)}{\pi_{\text{old}}(o_i \mid q)}$ 是重要性采样比（**on-policy 多轮更新**下，$\pi_{\text{old}}$ 是本 batch 采样时的旧策略，同一批数据多次 mini-batch 更新时 $\rho \neq 1$）。

两个稳定机制的分工：

- **clip（$\epsilon_{\text{clip}}$，常取 0.2）防单步过冲**：当 $\rho$ 超出 $[1-\epsilon, 1+\epsilon]$，min 取被截断的分支——对"把概率推得过远的方向"，梯度被冻结。直觉：RL 更新依据的是**旧策略采样出的经验**，若一步把策略改得太狠，新策略下的估计就失真了（off-policy 误差）。clip 是"信任域"的廉价实现。
- **$\beta D_{KL}$（KL 正则）防长期漂移**：clip 只管单次更新的幅度，管不住多步累积的漂移方向。KL 项把 $\pi_\theta$ 锚在 $\pi_{\text{ref}}$（SFT 模型）附近——与 DPO 的 β 完全同源（Stage 6 第 1 周的 KL 约束目标）。DeepSeekMath 的工程细节：其 KL 用无偏估计器 $D_{KL} \approx \frac{\pi_{\text{ref}}(y|q)}{\pi_\theta(y|q)} - \log\frac{\pi_{\text{ref}}(y|q)}{\pi_\theta(y|q)} - 1$（k3 估计量），避免了对数项的有偏估计。

### 1.5 GRPO 与 PPO/DPO 的系统对照

| 维度 | PPO | DPO | GRPO |
| --- | --- | --- | --- |
| Critic | 需要学习的价值网络 | 无 | **无（组均值当基线）** |
| Ref 模型 | 有 | 有 | 有（KL 正则；部分变体如 Dr.GRPO/DAPO 探讨去除） |
| 数据 | 在线采样 | 离线偏好对 | **在线采样 + 组内对比** |
| 优势信号 | $A_t = R_t - V(s_t)$（GAE） | 隐含在配对分差 | $A_i = (r_i - \bar{r})/(\sigma_r + \epsilon)$ |
| 适用 | 通用 RL（有过程奖励） | 有偏好对、无环境 | **可验证结果奖励、可批量采样** |

---

## 2. 实现与验证

### 2.1 本周 MVP：独立的 GRPO Advantage & Loss Calculator

```python
"""
纯 PyTorch GRPO Advantage/Loss 计算器 + 机制单元测试。
运行方式: python stage7_week1_grpo.py
依赖: torch
"""
import torch
import torch.nn.functional as F


def group_advantage(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """A_i = (r_i - mean(r)) / (std(r) + eps); rewards: [G]"""
    assert rewards.dim() == 1 and rewards.numel() >= 2
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)


def grpo_loss(logp_ratios: torch.Tensor, advantages: torch.Tensor,
              clip_eps: float = 0.2) -> torch.Tensor:
    """logp_ratios ρ = π_θ(o_i|q)/π_old(o_i|q), 形状 [G] (或 Token 级展平)"""
    unclipped = logp_ratios * advantages
    clipped = torch.clamp(logp_ratios, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()


def kl_penalty(policy_logps: torch.Tensor, ref_logps: torch.Tensor) -> torch.Tensor:
    """DeepSeekMath 的 k3 无偏 KL 估计: ref/p - log(ref/p) - 1 (对同序列逐 Token 平均)"""
    log_ratio = ref_logps - policy_logps
    return (torch.exp(log_ratio) - log_ratio - 1).mean()


def _tests():
    torch.manual_seed(0)
    eps = 1e-4

    # ---- 测试 1: 学习计划的示例奖励 [1,0,0,1,0.5] 的组内优势 ----
    r = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.5])
    A = group_advantage(r)
    assert abs(A.mean().item()) < 1e-5, "优势均值应为 0 (组内中心化的定义)"
    assert A[0] > 0 and A[1] < 0, "奖励高于均值应得正优势, 低于均值得负"
    # 手算: mean=0.5, std≈0.4472 -> A0 ≈ 1.118
    expected_a0 = (1.0 - 0.5) / (r.std(unbiased=False) + eps)
    assert abs(A[0].item() - expected_a0.item()) < 1e-5

    # ---- 测试 2: 全同分组 -> 优势趋零 (无学习信号, 第6-7周过滤的依据) ----
    A_same = group_advantage(torch.zeros(5))
    assert A_same.abs().max() < 1e-3, "全 0 奖励组的优势应接近 0 (eps 保护生效)"

    # ---- 测试 3: clip 的截断效果 ----
    rho = torch.tensor([0.5, 0.9, 1.1, 1.6])
    adv = torch.tensor([1.0, 1.0, 1.0, 1.0])
    loss_unclipped = grpo_loss(rho, adv, clip_eps=0.2)
    # 逐点核对: ρ=0.5 -> clip 到 0.8 -> min(0.5, 0.8)=0.5 (未截断分支胜出)
    #           ρ=1.6 -> clip 到 1.2 -> min(1.6, 1.2)=1.2 (截断生效, 梯度冻结)
    x = rho.clone().requires_grad_(True)
    l = grpo_loss(x, adv)
    l.backward()
    assert x.grad[0].abs() > 0, "截断未生效处应有梯度"
    assert x.grad[3].abs() < 1e-6, "ρ 超界处梯度应被 clip 冻结 (防单步过冲)"

    # ---- 测试 4: KL 惩罚的形态 ----
    p_logps = torch.tensor([-1.0, -2.0, -1.5])
    r_logps = torch.tensor([-1.0, -1.0, -2.0])
    kl = kl_penalty(p_logps, r_logps)
    assert kl.item() > 0, "策略偏离 ref 时 KL 惩罚应为正"
    assert torch.isclose(kl_penalty(p_logps, p_logps),
                         torch.tensor(0.0), atol=1e-6), "策略=ref 时 KL 应为 0"

    # ---- 测试 5: 端到端 toy 训练 (奖励最大化方向正确) ----
    # 玩家: 两个可学习 logit, softmax 选 Token A/B; 环境奖励: 选 A 得 1 分
    logits = torch.zeros(2, requires_grad=True)
    opt = torch.optim.AdamW([logits], lr=0.1)
    rewards_hist = []
    for step in range(60):
        dist = torch.distributions.Categorical(logits=logits)
        G = 8
        acts = dist.sample((G,))                              # 组采样
        rewards = (acts == 0).float()                         # 选 A(=0) 得分
        adv = group_advantage(rewards).detach()
        logp = dist.log_prob(acts)                            # 当前策略 logp
        old_logp = logp.detach()                              # 本步的 old 策略
        rho = torch.exp(logp - old_logp)
        loss = grpo_loss(rho, adv) + 0.01 * kl_penalty(logp, torch.zeros_like(logp))
        opt.zero_grad(); loss.backward(); opt.step()
        rewards_hist.append(rewards.mean().item())
    assert rewards_hist[-10:] and sum(rewards_hist[-10:]) / 10 > 0.9, \
        f"toy 环境应收敛到选 A (奖励≈1): 实际 {sum(rewards_hist[-10:]) / 10:.2f}"
    print(f"✅ GRPO 五项测试通过; toy 训练末段平均奖励 = {sum(rewards_hist[-10:]) / 10:.2f}")


if __name__ == "__main__":
    _tests()
```

**预期输出**：

```text
✅ GRPO 五项测试通过; toy 训练末段平均奖励 = 1.00
```

五项测试各验证一个机制性结论：① 示例奖励的组内中心化（优势均值恒为 0）；② 全同分组无信号（ε 保护 + 第 6-7 周过滤依据）；③ clip 冻结超界梯度（`x.grad[3]≈0` 是 clip 语义的直接证明）；④ KL 惩罚的零点与正性；⑤ **60 步 toy 训练真实收敛**——整条 GRPO 管线（采样→组优势→clip 损失→更新）在方向上被端到端验证，而非只测公式。

---

## 3. 工程权衡与失效模式

### 3.1 超参的量级与权衡

| 超参 | 常用量级 | 权衡 |
| --- | --- | --- |
| 组大小 $G$ | 4~16（DAPO 动态采样场景更高） | 大 → 基线更稳、更贵（采样成本 ×G）；小 → std 估计噪声大 |
| $\epsilon_{\text{clip}}$ | 0.2 | 小 → 更保守、训练慢；大 → 单步过冲风险 |
| KL 系数 $\beta_{KL}$ | 0 ~ 0.01 量级（RLVR 中常很小甚至 0） | 大 → 锚死探索；0 → 依赖 clip 与奖励设计（DAPO 实证可去 KL，但需要更强的其他稳定机制） |
| lr | ~1e-6 量级 | RL 对 lr 比 SFT 敏感；与 G、clip 联动 |
| 采样 temperature | 0.7~1.2 | 高 → 组内多样性大（有信号组多）；过高 → 乱码轨迹浪费算力 |

### 3.2 三个代表性失效模式

**失效 1：组内全同——优势为零，训练静默停滞**
- **症状**：训练正常跑，loss 在动，但 `reward/mean` 数百步不涨；日志里大量组 reward 全 0（或全 1）。
- **根因**：任务太难（全错，$A_i=0$）或太易（全对）；std≈0 + ε 使优势全部趋零——**组相对优势的天然盲区**。
- **定位**：统计每 batch 的"有效组占比"（std > 阈值的组）。
- **修复**：动态重采样无效组（第 6-7 周）；课程化调整 prompt 难度（Stage 5 Sweet Spot 的 RL 版）；DAPO 的 dynamic sampling 思路（采样直到组内有区分度）。

**失效 2：KL 系数失衡——要么冻结要么跑飞**
- **症状**：β_KL 过大：模型行为与 SFT 几乎无异，reward 不动；β_KL=0 且无其他稳定机制：reward 涨但输出开始出现复读/格式漂移（熵坍塌前兆）。
- **定位**：`actor/kl_divergence` 曲线——健康形态是缓慢小幅上升并稳定；飙升至数个 nat 即跑飞。
- **修复**：β 从小值起调（如 0.001~0.01），配合熵监控（熵骤降 = 过早收敛，第 6-7 周失败模式）。

**失效 3：off-policy 程度过深——重要性比爆掉**
- **症状**：loss 出现巨大尖刺，甚至 NaN；`ρ` 的分布长尾严重。
- **根因**：同一批 rollout 被过多 mini-batch 复用（μ 次更新），策略已远离 π_old，重要性采样比超出 clip 有效范围，估计失真。
- **定位**：记录每个 batch 的 μ 与 ρ 的分位数（P99）。
- **修复**：μ 降到 1~4；或用 GSPO 类序列级比率稳定长序列场景；必要时配合梯度裁剪兜底。

---

## 4. 延伸思考题

1. **基线的方差缩减证明**：证明"减去与动作无关的基线 $b$，策略梯度期望不变"（$\nabla \mathbb{E}[R] = \mathbb{E}[(R-b)\nabla\log\pi]$ 的基线无关性），并推导使梯度方差最小的最优基线恰为 $E[R^2 \nabla\log\pi] / E[\nabla\log\pi]$ 的加权形式（状态值函数是其特例）。这解释了为什么"组均值"是一个合法且合理的基线。
2. **std 归一化的争议**：Dr.GRPO 指出除以 std 会让"简单题的高奖励组"与"难题的低奖励组"被强行拉到同一尺度，造成有偏的优化权重。分析：把 std 归一去掉后（只减均值），训练动力学会如何变化？什么任务形态更适合去 std？（开放题——这是 2025 年 GRPO 修正系工作的核心争论点。）
3. **动手实验**：把 2.1 的 toy 环境改成"奖励只在组内恰好一半选对时给 1 分"（制造稀疏信号），观察 G=4 与 G=16 下的收敛速度差异——直接体感"组大小 = 基线质量"的权衡。

---

*下一篇：[第 2 周：RLVR 可验证奖励与 Verifier Engine 设计](第2周-RLVR与Verifier引擎.md)*
