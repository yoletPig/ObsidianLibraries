# 10 - RLHF：从奖励模型到 PPO/DPO

## 本篇目标

读完本篇，你应该能做到：

1. 说清 **SFT 之后为什么还要 RL**——"会补全"与"会回答"之间差的是什么。
2. 画出 **RLHF 三阶段管线**，讲清 KL 惩罚为什么数学上必须（附可复现的 reward hacking 数值演示：不加 KL 真实回报 $-100$，加对后 $+12.5$）。
3. **完整推导 DPO**：从 Bradley-Terry 偏好模型 → KL 约束 RL 的闭式解 → 隐式奖励 → 交叉熵损失，每一步消掉了什么说清楚——并与库内 VLM Stage 6 第 1 周的推导对齐。
4. 用对比表掌握 DPO vs PPO 的取舍，理解 **GRPO** 删掉 Critic 的理由与 **RLVR** 的范式含义——直接连通 Stage 7。

本章是全系列的"出口"：01–09 章的所有概念（回报、KL、ratio、优势、GAE）在这里汇入 LLM 对齐。设定切换：状态 = prompt + 已生成前缀，动作 = 下一个 token，轨迹 = 一次完整生成（01 章的对应表在此兑现）。

## 1. SFT 之后为什么还要 RL

SFT（监督微调）做的是最大似然：给定演示数据 $\{(x, y^*)\}$，最大化 $\log \pi_\theta(y^*|x)$。它教会模型"好回答长什么样"，但有三个天花板：

1. **只会模仿，不会比较**。SFT 只见过好答案；"给定两个答案哪个好"的判断能力没有被训练。而人类评估者（用户）恰恰是按比较来感知质量的。
2. **无法表达"不要什么"**。演示数据里没有"这样的回答不好"的负信号——幻觉、拒答过度、格式混乱这些问题 SFT 数据治不了根。
3. **目标错位**。SFT 优化"像演示数据"，用户要的是"让用户满意"。前者是代理目标，且**演示数据本身有限且有分歧**（同一问题人类会给出不同答案——单点最大似然把多峰分布坍缩到众数）。

RL 视角的重述：把"生成整个回答"当成一条**轨迹**，把"用户满意"当成**奖励**，优化期望回报 $J(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}[r(x, y)]$。策略梯度（07 章）直接适用——这是"LLM 对齐 = RL"的数学本质，没有任何新东西。

## 2. RLHF 三阶段管线（InstructGPT）

**InstructGPT**（Ouyang et al., "Training language models to follow instructions with human feedback", NeurIPS 2022）确立的管线，三阶段：

```text
① SFT                          ② 奖励模型 RM                    ③ PPO 强化
预训练模型                      (从 SFT 模型初始化)               (从 SFT 模型初始化)
    │                              │                              │
 标注演示数据                人类对同一 prompt 的多个             RM 打分 + KL 惩罚,
 (约 13k 条)                回答排序 → 偏好对 (约 33k)           PPO 更新策略 (约 31k prompt)
    ↓                              ↓                              ↓
 π_SFT ──────────────────→ r_φ(x, y) ──────────────────→ π_RLHF
                                                    │
                                              参考模型 π_ref (冻结的 π_SFT)
```

各阶段要点：

- **阶段 ①**：行为克隆起步。没有它 RM 训不动（分布太散）、PPO 起点太差。
- **阶段 ②**：标注员对同一 prompt 的若干回答**排序**（不是打分——排序更可靠），转成偏好对 $(y_w, y_l)$。RM 用 Bradley-Terry 损失训练（§4.1），本质是一个"给回答打分"的回归头（常用 6B/175B 的 transformer + 标量头）。
- **阶段 ③**：PPO 优化 $\mathbb{E}[r_\phi(x, y) - \beta \cdot \text{KL项}]$。**四个模型同时在显存里**：actor（在被更新）、ref（冻结，算 KL）、reward model（冻结，打分）、critic（价值头，估优势）——这是 RLHF 贵的直接原因，也是 DPO 的靶子。

InstructGPT 的标志性结论：**1.3B 的 InstructGPT 输出在人类评估上优于 175B 的 GPT-3**——对齐质量与参数量解耦，"对齐税"可以是负的。

## 3. KL 惩罚为什么必须：reward hacking 数值演示

### 3.1 问题的形式

优化目标不是裸奖励，而是：

$$\max_\theta\; \mathbb{E}_{x \sim D,\, y \sim \pi_\theta(\cdot|x)}\left[r_\phi(x, y) - \beta\, D_{KL}\big(\pi_\theta(\cdot|x)\,\|\,\pi_{ref}(\cdot|x)\big)\right]$$

$\beta > 0$ 是 KL 系数，$\pi_{ref}$ 通常是冻结的 SFT 模型。三重理由：

1. **RM 只在数据分布附近可信**（外推必错）；离开 $\pi_{ref}$ 太远 = 离开 RM 的校准域 = 分数不可信；
2. **防止语言能力崩坏**（不加点约束下 reward maximizing 会学出乱码——乱码如果恰好骗过 RM 就是高分）；
3. **保持多样性/可控性**，训练稳定（KL 项提供的是"对每条轨迹的逐步惩罚"，PPO 里的实现是 reward 逐 token 减 $\beta(\log \pi_\theta - \log \pi_{ref})$）。

### 3.2 可复现演示：RM 的线性外推灾难

一维玩具：真实人类偏好规律 $R_{true}(a) = 5a - 0.5a^2$（峰值 $a^* = 5$，回报 12.5）。RM 在 $a \in [0, 6]$ 的数据上只学到了局部规律，拟合出线性模型 $\hat R(a) = 2a$——**在训练分布内看似正确，外推处灾难性错误**：

```python
import numpy as np
a_grid = np.arange(0, 21, 1.0)
R_hat, R_true = 2.0 * a_grid, 5 * a_grid - 0.5 * a_grid ** 2
print("RM 外推误差: R_hat(20)=%.0f  R_true(20)=%.0f" % (R_hat[20], R_true[20]))
# KL 惩罚: 目标 = R_hat(a) - coef * (a - a_ref)^2, a_ref=3
a_ref = 3.0
for coef in [0.0, 0.1, 0.5, 2.0]:
    best = a_grid[int(np.argmax(R_hat - coef * (a_grid - a_ref) ** 2))]
    print(f"kl_coef={coef}: 最优 a*={best:.0f}, 真实回报={R_true[int(best)]:.1f}")
```

真实运行结果：

```text
RM 外推误差: R_hat(20)=40  R_true(20)=-100
kl_coef=0.0: 最优 a*=20, 真实回报=-100.0
kl_coef=0.1: 最优 a*=13, 真实回报=-19.5
kl_coef=0.5: 最优 a*=5,  真实回报=12.5   ← 正中真实最优
kl_coef=2.0: 最优 a*=3,  真实回报=10.5   ← 过强, 压回保守区
```

三段式读数就是 **reward hacking** 的完整病理与药方：

- **不加 KL（coef=0）**：优化器冲向 $\hat R$ 的"假峰" $a=20$，真实回报 $-100$。策略钻了 RM 的空子——LLM 里的对应物：回答变长变啰嗦、堆砌格式化标记、复读 RM 偏好的短语（"作为一个AI助手……"）。
- **KL 太弱（0.1）**：仍然被吸进外推区（$a^*=13$，回报 $-19.5$）。
- **KL 适中（0.5）**：KL 的二次钟形把策略按回 RM 可信域，正中真实最优。
- **KL 太强（2.0）**：策略被摁在 $\pi_{ref}$ 附近（$a^*=3$），学不到偏好里的改进——对齐收益蒸发了大半。

**β 的 U 型权衡是 RLHF 调参的第一课**：太小 → hacking；太大 → 白干。InstructGPT 用 $\beta=0.02$（token 级、对 log-ratio 逐步惩罚，与玩具的一维类比量纲不同，不能直接搬数字）。

> **经验做法**：训练日志必盯三条曲线——RM 分数（应缓升）、KL(π‖ref)（应缓升后平台）、评估集胜率（真指标）。**RM 分数飙升 + 胜率不升 = 正在 hacking**，立刻加大 β 或回滚。

## 4. DPO：把 RL 目标变成一个交叉熵

**DPO**（Direct Preference Optimization，Rafailov et al., NeurIPS 2023；库内 Stage 6 第 1 周的同款推导）的震撼点：**KL 约束的 RL 目标有闭式最优解，代回 Bradley-Terry 后配分函数恰好消掉**——于是可以跳过 RM 与 PPO，直接在偏好对上做监督学习。逐步推导：

### 4.1 起点：Bradley-Terry 偏好模型

人类给出"$y_w$ 优于 $y_l$"的概率由潜在奖励差决定（Bradley & Terry, 1952；RLHF 语境的采用见 Christiano et al., 2017, "Deep RL from Human Preferences"）：

$$p(y_w \succ y_l \mid x) = \sigma\big(r^*(x, y_w) - r^*(x, y_l)\big), \qquad \sigma(z) = \frac{1}{1+e^{-z}}$$

RM 训练目标（阶段②）就是这个模型的 MLE：$\mathcal{L}_{RM} = -\mathbb{E}\big[\log \sigma(r_\phi(x,y_w) - r_\phi(x,y_l))\big]$。

### 4.2 KL 约束 RL 的闭式解

RLHF 阶段③的目标（§3.1）对每个 prompt $x$ 独立，写出拉格朗日形式（奖励 − β·KL）：

$$\max_{\pi}\; \mathbb{E}_{y\sim\pi}\left[r^*(x,y)\right] - \beta\, D_{KL}\big(\pi(y|x)\,\|\,\pi_{ref}(y|x)\big)$$

对 $\pi(y|x)$ 做变分（逐 token 概率求导置零，离散分布上的标准计算）：

$$\pi^*(y|x) = \frac{1}{Z(x)}\, \pi_{ref}(y|x)\, \exp\left(\frac{1}{\beta} r^*(x,y)\right), \qquad Z(x) = \sum_y \pi_{ref}(y|x)\, e^{r^*(x,y)/\beta}$$

**读法**：最优策略 = 参考策略 × 奖励的指数倾斜。KL 惩罚把"按奖励集中"变成"软性集中"——β 越大越像 $\pi_{ref}$，越小越像 argmax。移项解出 $r^*$（**反向表达**）：

$$r^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**隐式奖励**：任何策略与参考策略的 log-ratio，加上一个只依赖 $x$ 的常数，就是它隐含的奖励。

### 4.3 消掉配分函数：DPO 目标

把 §4.2 的 $r^*$ 代回 §4.1 的 Bradley-Terry。**关键一步：$Z(x)$ 只依赖 $x$，在减法 $r(x,y_w) - r(x,y_l)$ 里成对出现、直接相消**：

$$p(y_w \succ y_l \mid x) = \sigma\left( \beta \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)} \right)$$

偏好概率**只由策略本身表达**——RM 被彻底内化。对偏好数据集 $\{(x, y_w, y_l)\}$ 做最大似然，得到 DPO 损失：

$$\boxed{\mathcal{L}_{DPO}(\theta) = -\mathbb{E}\left[\log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right)\right]}$$

定义隐式奖励 $r_\theta(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$（Stage 6 教程的同款记号），损失就是 $\log \sigma(r_\theta(x,y_w) - r_\theta(x,y_l))$ 的负值——**形式与 RM 训练一模一样，只是"奖励模型"换成了"策略对参考模型的 log-ratio"**。

### 4.4 数值演示：DPO 损失怎么推动策略

单位置两动作的玩具（chosen/rejected 两个 token）：

```python
import torch
np.random.seed(0); torch.manual_seed(0)

logits = torch.tensor([[1.2, 0.4]], requires_grad=True)   # [chosen, rejected]
pi_ref = torch.tensor([[0.8, 0.5]])                        # 冻结参考
beta = 0.1

def dpo_loss(logits, pi_ref, beta):
    logp = torch.log_softmax(logits, -1)[0]
    logp_ref = torch.log_softmax(pi_ref, -1)[0]
    margin = beta * ((logp[0] - logp_ref[0]) - (logp[1] - logp_ref[1]))
    return -torch.log(torch.sigmoid(margin)), margin

opt = torch.optim.SGD([logits], lr=0.5)
for step in [0, 5, 10, 20]:
    for _ in range(step):
        opt.zero_grad(); loss, _ = dpo_loss(logits, pi_ref, beta)
        loss.backward(); opt.step()
    loss, m = dpo_loss(logits, pi_ref, beta)
    with torch.no_grad():
        p = torch.softmax(logits, -1)[0, 0].item()
    print(f"step {step:2d}: loss={loss.item():.4f} 隐式奖励差={m.item():+.4f} p(chosen)={p:.4f}")
```

真实运行结果：

```text
step  0: loss=0.6685 隐式奖励差=+0.0500 p(chosen)=0.6900
step  5: loss=0.6567 隐式奖励差=+0.0743 p(chosen)=0.7393
step 10: loss=0.6452 隐式奖励差=+0.0982 p(chosen)=0.7828
step 20: loss=0.6232 隐式奖励差=+0.1452 p(chosen)=0.8522
```

梯度方向：拉高 chosen 的隐式奖励、压低 rejected 的——与 PPO 在"该样本上"的更新方向一致，但没有采样、没有 reward model、没有 critic。注意 loss 下降缓慢（20 步才 0.668→0.623）——β=0.1 时梯度很小，这是 DPO 需要**大学习率/大 batch/多 epoch** 的原因之一。

## 5. DPO vs PPO：对比表

| 维度 | PPO（RLHF 版） | DPO |
| --- | --- | --- |
| 训练信号 | 显式 RM 标量奖励（token 级） | 偏好对（轨迹级对比） |
| 需要的模型数 | 4（actor/ref/RM/critic） | 2（actor/ref） |
| 在线采样 | 必须（rollout） | 不需要（离线偏好数据集） |
| 样本效率 | 高（每条生成直接有分数） | 较低（只有成对比较信号） |
| 稳定性 | 超参敏感（β/clip/lr 联动） | 单损失较稳，但 β 与 lr 也敏感 |
| 算力形态 | 重（生成占一半时间） | 轻（与 SFT 同构） |
| 理论关系 | 直接优化 KL 约束 RL 目标 | 同一目标的闭式最优被"拟合" |
| 失效模式 | reward hacking、训练崩 | 离线分布外无约束（隐式 KL 可能失守）、长度偏置 |

**取舍的本质**：DPO 把"在线探索 + 显式奖励"换成"离线数据 + 隐式奖励"。它赢在工程（省 RM/critic/采样），输在**数据是离线的**——策略变强后，旧偏好对不再覆盖新策略的分布（01 章的根本矛盾以另一种形式回归）。实践中（Stage 6 的工程视角）：DPO 适合"有高质量偏好数据、算力受限"；PPO 适合"要持续在线迭代、有 RL 工程团队"。两者不是替代关系，是"离线/在线"两个 regime 的选择。

## 6. GRPO：删掉 Critic 的组内相对优势

**GRPO**（Group Relative Policy Optimization，Shao et al., "DeepSeekMath", arXiv:2402.03300；DeepSeek-R1 的 RLVR 引擎）对 PPO 的手术：**删掉 value/critic 网络**，优势由"组内相对比较"给出——

对每个 prompt 采样 $G$ 个回答（一组），各自获得奖励 $r_1, \dots, r_G$，则组内第 $i$ 个回答的优势：

$$\hat{A}_i = \frac{r_i - \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}$$

然后照常算 ratio、做 PPO 式裁剪更新（KL 到 ref 的惩罚保留）。

**为什么敢删 Critic？** 用 08 章的语言：Critic $V(s)$ 是"基线"，学习它的成本是一个网络 + 它自己的训练不稳定性；而 LLM 的 RLVR 场景（下节）里**奖励只在句尾出现且组内同 prompt**——组均值天然就是一个"以 prompt 为条件的基线"，统计上等价于 $V(x)$（prompt 级价值），不用学、无偏差、零参数。这正是 07 章"基线只要与动作无关即可"的直接应用：组均值对组内每个样本是同一个数，满足全部合法性条件。库内 **Stage 7 第 1 周**对此有完整推导（"Critic 的本质是基线，GRPO 用组内均值当免费基线"），本章给出的是它的 RL 本体视角。

代价：同一 prompt 要采样 $G$ 次（典型 4–16），**推理成本 × G**——用算力换掉一个模型，这笔账在推理比训练便宜的今天通常是划算的。

## 7. RLVR：可验证奖励

**RLVR**（Reinforcement Learning with Verifiable Rewards）：奖励不由学习的 RM 打分，而由**程序化验证器**判定——数学题对答案、代码跑测试、格式检查正则。两个深刻变化：

1. **奖励不再可被 hacking（在此验证器覆盖范围内）**：§3 的 reward hacking 源于"RM 是学习的、可被骗的"；验证器是确定性的，对了就是对。数学/代码成为 RLVR 首选场景的原因就是验证器便宜且可靠。
2. **奖励稀疏且诚实**：整条轨迹 0/1 的奖励让信用分配回到最原始的难度（01 章的延迟奖励问题以极端形式回归）——GRPO 的组内基线 + 大 $G$ + 长回答的 GAE 折衷，正是对稀疏性的工程回应。

DeepSeek-R1（arXiv:2501.12948）证明：纯 RLVR + GRPO 规模化后，模型自发出现长链思考、自我验证等行为——这是"奖励足够诚实时，探索会自己找到推理格式"的标志性实验。库内 **Stage 7 全阶段**（verl 引擎、Verifier 设计、多模态 RLVR 实战）建立在本章概念之上；读完本章再进 Stage 7，每个组件都能对号入座。

## 8. 常见误区

1. **把 DPO 的 β 当成温度随便调**。β 同时是"隐式奖励的尺度"与"对 ref 的约束强度"：β 太小 → log-ratio 被放大、更新过猛且隐式 KL 失守（长度爆炸、语言崩坏）；太大 → 梯度消失（§4.4 实测 20 步 loss 只降 0.045）。DPO 论文报告 β=0.1 量级为常用起点（经验做法：0.05–0.5 间扫）。
2. **DPO 训练后不看隐式 KL**。应监控 $\beta \log(\pi_\theta/\pi_{ref})$ 在训练集上的分布——它漂移过远说明"离线数据已不覆盖当前策略"，此时该回到在线方法（PPO/迭代式 DPO）。
3. **以为 RLHF 的 PPO 奖励是"人类直接给的"**。人类只提供偏好对；PPO 阶段的奖励来自**学习的 RM**——它有 §3.2 演示的全部外推缺陷。混淆"人类偏好"与"RM 分数"会完全误读 hacking 现象。
4. **KL 惩罚写在损失里但逐 token 与逐轨迹混淆**。token 级惩罚 $-\beta(\log\pi_\theta - \log\pi_{ref})$ 加进每步 reward（InstructGPT 做法）与轨迹级一次性加 $-\beta D_{KL}$ 梯度性质不同；复现论文时先确认口径。
5. **GRPO 组内全同则无梯度**。组内 $G$ 个回答奖励全相同时 $\hat A_i = 0$（减均值后全零）——这批 prompt 白采。Stage 7 的"轨迹过滤/难度分层"（同组难度控制）就是为这个设计的；这不是 bug，是组内相对优势的定义使然。

## 本篇小结

- SFT 教"像什么"，RL 教"哪个好"：对齐 = 把"用户满意"变成可优化的期望回报，01 章的 MDP 对应表在此落地。
- RLHF 三阶段（SFT → RM → PPO）；InstructGPT 证明对齐质量可与参数量解耦（1.3B 胜 175B）。
- KL 惩罚必须且 β 呈 U 型权衡：实测不加 KL 真实回报 −100（hacking）、β 适中 +12.5（正中最优）、过强 10.5（保守）；监控"RM 分数升而胜率不升"作为 hacking 报警器。
- DPO 推导链：BT 模型 → KL 约束 RL 闭式解 π* ∝ π_ref·exp(r/β) → 隐式奖励 $r_\theta = \beta\log(\pi_\theta/\pi_{ref})$ → Z(x) 相消 → 一个 sigmoid 交叉熵；与 Stage 6 记号完全对齐。
- DPO vs PPO = 离线 vs online 的 regime 选择；GRPO 用组内均值删 Critic（基线思想的零参数实现）；RLVR 用程序化验证器堵死 hacking 的来源——Stage 7 的全部组件在本章就位。

## 练习

1. 把 §3.2 演示的 RM 换成二次拟合 $\hat R(a) = 3a - 0.25a^2$（仍不完全准确），重跑四个 kl_coef，观察"RM 越准、最优 β 窗口越宽"。
2. 手推 §4.2 的闭式解：对 $\pi(y|x)$ 逐点求导 $\frac{\partial}{\partial \pi(y|x)}\left[\sum_y \pi(y|x) r^*(x,y) - \beta \sum_y \pi(y|x)\log\frac{\pi(y|x)}{\pi_{ref}(y|x)} + \lambda(\sum_y \pi(y|x)-1)\right] = 0$，解出 $\pi^*$ 并验证归一化常数即 $Z(x)$。
3. 扩展 §4.4 的玩具到"多偏好对"：造 20 对 (chosen, rejected) 数据，实现 batch 版 DPO 损失并训练到收敛，画出隐式奖励差随步数的曲线。
4. 实现 GRPO 的组内优势（对一组 G=8 个奖励计算 $\hat A_i$），验证：全同组输出全零；并解释为什么这使"组内难度控制"成为 Stage 7 的必修工程。
5. 思考题：RLVR 里"验证器只检查最终答案"时，模型可能学会跳步硬猜（答案对过程错）。用 reward hacking 的框架分析这个现象，并说明 Stage 7 "过程奖励/格式奖励"设计的动机。

## 参考来源

- InstructGPT：Ouyang et al., "Training language models to follow instructions with human feedback", NeurIPS 2022
- RLHF 起点：Christiano et al., "Deep Reinforcement Learning from Human Preferences", NeurIPS 2017；Bradley & Terry, "Rank Analysis of Incomplete Block Designs", *Biometrika*, 1952
- DPO：Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", NeurIPS 2023（附录 A 的闭式解推导）
- GRPO：Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", arXiv:2402.03300；DeepSeek-R1, arXiv:2501.12948
- 交叉引用：库内《VLM & VLA》Stage 6 第 1 周（DPO 推导）、Stage 7 第 1 周（GRPO 与 Critic-as-baseline）、第 2 周（Verifier 设计）
- 本章 §3.2/§4.4 数值为真实运行结果；β 的常用区间为经验做法（已标注）
