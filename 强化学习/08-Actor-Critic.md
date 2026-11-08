# 08 - Actor-Critic

## 本篇目标

读完本篇，你应该能做到：

1. 把 04 章"MC vs TD"的偏差-方差之争接到策略梯度上，说清 **REINFORCE（纯 MC）差在哪、Critic 补什么**。
2. 推导 **GAE**（广义优势估计）——把 07 章的基线、04 章的 TD 误差、资格迹统一成一个公式；理解 $\lambda$ 的物理含义。
3. 实现共享主干的 A2C（CartPole，实测 ep700 评估满分），并复现一个真实调参事故：**MSE 价值损失通过共享主干摧毁策略，换 Huber 后梯度范数从 18–40 降到 1–4**。
4. 讲清 A3C 的异步思想，以及它为什么被同步版本（A2C/PPO）取代——并行范式之争的工程逻辑。

## 1. 偏差-方差权衡的延续：REINFORCE 缺的那块

07 章结尾的问题：REINFORCE 用实测回报 $G_t$ 当权重——**无偏但方差巨大**（数据用一次、整条轨迹的随机性全算进去）。04 章给过答案的方向：**自举**（用 $r + \gamma V(s')$ 当目标）以引入少量偏差为代价砍掉大量方差。

把这两个拼起来就是 Actor-Critic 的设计图（Sutton & Barto, 2nd ed., §13.5、§13.7）：

| | 权重信号 | 偏差 | 方差 | 数据要求 |
| --- | --- | --- | --- | --- |
| REINFORCE | $G_t$（实测回报） | 无 | 高 | 必须 episodic、数据用一次 |
| Actor-Critic | $A(s_t, a_t) \approx \delta_t$ 或 GAE | 有（来自 $V$ 的误差） | 低 | 可在线增量、可用部分轨迹 |

其中优势函数 $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ 就是 07 章"基线"的函数近似版：**$V(s)$ 是基线，$Q-V$ 是去均值后的信号**。它回答的问题是"这个动作比*按当前策略平均表现*好多少"——比原始回报 $G_t$ 的信噪比高得多（$G_t$ 里混着"状态本身好不好"的成分，优势把这部分减掉了）。

**实现上优势几乎从不用 $Q$ 网络算**（另养一个 $Q$ 网络成本翻倍），而是用 TD 误差：

$$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

可以证明（Sutton & Barto, 2nd ed., §13.5）$\mathbb{E}[\delta_t | s_t, a_t] = A^\pi(s_t,a_t)$——**TD 误差是优势的无偏单样本估计**。一个 $V$ 网络，既当基线又当 critic，还顺带解决了"continuing 任务没有回合终点"的问题。这就是"值函数线的资产被策略梯度线收购"的全过程。

## 2. GAE：把光谱参数化

### 2.1 推导

07 章说过策略梯度信号可以用任意基线；§1 说 TD 误差 $\delta_t$ 是单步优势估计（偏差来自 $V$，方差只有一步）。$\lambda$-优势把 $k$ 步优势估计指数加权混合（Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation", arXiv:1506.02438, ICLR 2016）。

先定义 $k$ 步优势（$k$ 步 TD 的策略梯度版，04 章 $G^{(n)}$ 的亲戚）：

$$\hat{A}_t^{(1)} = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \quad\text{（1 步：偏差大方差小）}$$
$$\hat{A}_t^{(2)} = r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t) = \delta_t + \gamma\delta_{t+1}$$
$$\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}$$
$$\hat{A}_t^{(\infty)} = G_t - V(s_t) \quad\text{（MC：无偏差差）}$$

**GAE = 对这些估计做指数加权平均**，权重 $\propto \lambda^{k-1}$：

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = (1-\lambda)\sum_{k=1}^{\infty} \lambda^{k-1} \hat{A}_t^{(k)}$$

把 $\hat{A}^{(k)} = \sum_l \gamma^l \delta_{t+l}$ 代入交换求和顺序，权重 $(1-\lambda)\lambda^{k-1}$ 对 $k$ 求和恰好归一，得到极简形式：

$$\boxed{\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l} = \delta_t + \gamma\lambda \hat{A}_{t+1}^{\text{GAE}}}$$

实现就是**一行倒序递推**：`gae = delta[t] + gamma * lam * gae`。两个端点：

- $\lambda = 0$：$\hat A_t = \delta_t$，纯一步 TD 优势（方差最小、偏差全靠 $V$ 的准确性）；
- $\lambda = 1$：$\hat A_t = G_t - V(s_t)$，MC 减基线（无偏、方差回到 REINFORCE 水平）。

$\lambda$ 就是 04 章 n-step 光谱的"资格迹参数化"——07 章结尾说"GAE 是 $\lambda$ 的现代继承者"，指的就是这个公式。工程甜点：$\lambda \in [0.9, 0.97]$，09 章 PPO 默认 0.95。

### 2.2 λ 消融实验（真实数据）

在 §4 的实现上只改 $\lambda$，各跑 500 回合（seed=0，训练期回报的 50 回合窗口均值）：

| $\lambda$ | 含义 | 最后 50 回合训练均值 |
| --- | --- | --- |
| 0.0 | 纯 TD(0) 残差 $\delta_t$ | **9.3**（不学习） |
| 0.9 | 偏自举 | 217.7 |
| 0.95 | 默认值 | 175.8 |
| 1.0 | MC − 基线 | 133.9 |

读数：**$\lambda=0$ 完全不学习**——单步 TD 优势的偏差（初期 $V$ 极不准）把策略梯度信号毁掉了，信用分配被压得太局部；$\lambda \ge 0.9$ 才有足够的视野；$\lambda=1$ 反而略慢（方差回归）。这不是普适结论（不同任务最优 $\lambda$ 会移动），但"$\lambda=0$ 崩、中间甜点区宽"的形状是普遍的。**经验做法**：永远不要从 $\lambda=0$ 调起；从 0.95 出发，先调 lr 再动 $\lambda$。

## 3. A2C 架构

### 3.1 网络：共享主干还是分开

A2C（Advantage Actor-Critic，Mnih et al. 2016 的同步版）一个网络两条头：

```text
        输入 s (4 维)
            │
      Linear(4→128) + ReLU        ← 共享主干 trunk
        │           │
   Linear(128→2)  Linear(128→1)
        │           │
    策略头 π(a|s)   价值头 V(s)
   (Actor 梯度)   (Critic 梯度)
```

共享主干省一半算力、让两条头共享特征；**代价是两条头的梯度互相干扰**（§5 的真实事故）。任务更复杂时常用分离网络（或 PPO 实现里的 `share_encoder` 开关）。

### 3.2 损失函数三项

$$L = \underbrace{-\mathbb{E}\big[\hat{A}_t \log \pi_\theta(a_t|s_t)\big]}_{\text{策略损失（07 章的负号进 loss）}} + c_v \underbrace{\big(\hat{R}_t - V_\phi(s_t)\big)^2}_{\text{价值损失：回归 GAE 的目标}} - c_e \underbrace{H[\pi_\theta(\cdot|s_t)]}_{\text{熵奖励（鼓励探索）}}$$

三项各自的道理：

- **策略损失**就是 07 章的 REINFORCE 损失，权重从 $G_t$ 换成 $\hat A_t^{\text{GAE}}$；
- **价值损失**教 $V$ 回归"优势校正后的回报" $\hat R_t = \hat A_t + V(s_t)$（GAE 的配套 return，数学上 $\hat R_t$ 是 $\lambda$-return，Sutton & Barto, 2nd ed., §12.8 的 off-policy 版本同源）；$c_v$ 一般 0.5；
- **熵奖励**防止策略过早塌缩成确定性（探索退场太早 = 02 章 ε 固定过小的策略梯度版）；$c_e$ 一般 0.01。

## 4. 完整实现（CartPole）

### 4.1 超参表

| 超参 | 值 | 说明 |
| --- | --- | --- |
| 网络 | trunk 4→128 ReLU；π 头 128→2；V 头 128→1 | 共享主干 |
| 优化器 | Adam, lr=1e-3 | 单一 lr 管三条头 |
| $\gamma$ / $\lambda$ | 0.99 / 0.95 | GAE 标准 |
| rollout | **整回合** | 教学版；工程版用固定 T（见 §4.3 注） |
| 优势处理 | 回合内归一化（减均值除标准差） | 样本 ≥ 2 才做，防除零 |
| 价值损失 | **Huber（smooth_l1）** | 见 §5 事故复盘 |
| $c_v$ / $c_e$ | 0.5 / 0.01 | 惯例值 |
| 梯度裁剪 | max_norm=0.5 | 三项合力的保险 |
| 熵系数说明 | 整回合 rollout 通常几千步 | 自动探索充足 |

### 4.2 全量代码

```python
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Categorical

np.random.seed(0); torch.manual_seed(0)
env = gym.make("CartPole-v1")

class ActorCritic(nn.Module):
    def __init__(self, nS=4, nA=2, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(nS, hidden), nn.ReLU())
        self.pi_head = nn.Linear(hidden, nA)
        self.v_head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.trunk(x)
        return Categorical(logits=self.pi_head(h)), self.v_head(h)

ac = ActorCritic()
opt = torch.optim.Adam(ac.parameters(), lr=1e-3)
gamma, lam = 0.99, 0.95
vf_coef, ent_coef = 0.5, 0.01

def evaluate():
    """固定评估协议: 贪心 + 独立种子 + 10 回合均值"""
    ev = gym.make("CartPole-v1")
    total = 0.0
    for ep in range(10):
        obs, _ = ev.reset(seed=10000 + ep)
        done = False
        while not done:
            with torch.no_grad():
                dist, _ = ac(torch.tensor(obs, dtype=torch.float32))
                a = int(torch.argmax(dist.logits))
            obs, r, term, trunc, _ = ev.step(a)
            total += r; done = term or trunc
    return total / 10

for ep in range(800):
    obs, _ = env.reset(seed=ep)
    S, A, R, done = [], [], [], False
    while not done:                          # ---- 采样整条回合 ----
        dist, v = ac(torch.tensor(obs, dtype=torch.float32))
        a = dist.sample()
        ns, r, term, trunc, _ = env.step(int(a))
        S.append(obs); A.append(a); R.append(r)
        obs = ns; done = term or trunc
    with torch.no_grad():                    # ---- GAE 优势估计 ----
        vals = np.array([ac(torch.tensor(st, dtype=torch.float32))[1].item()
                         for st in S])
    T = len(R)
    next_v = 0.0                             # 真终止才为 0; 截断时应 bootstrap
    deltas = np.zeros(T)
    for t in reversed(range(T)):
        vn = vals[t + 1] if t + 1 < T else next_v
        deltas[t] = R[t] + gamma * vn - vals[t]     # TD 误差
    adv = np.zeros(T); gae = 0.0
    for t in reversed(range(T)):
        gae = deltas[t] + gamma * lam * gae         # GAE 一行递推
        adv[t] = gae
    ret = adv + vals                         # λ-return, critic 的回归目标
    # ---- 三项损失 ----
    adv_t = torch.tensor(adv, dtype=torch.float32)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)   # 回合内归一化
    ret_t = torch.tensor(ret, dtype=torch.float32)
    dist, v = ac(torch.tensor(np.array(S), dtype=torch.float32))
    logp = dist.log_prob(torch.stack(A))
    pg_loss = -(adv_t * logp).mean()
    v_loss = nn.functional.smooth_l1_loss(v.squeeze(-1), ret_t)  # Huber!
    entropy = dist.entropy().mean()
    loss = pg_loss + vf_coef * v_loss - ent_coef * entropy
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
    opt.step()
    tr.append(sum(R))                           # 缓存回合回报(循环前 tr = [])
    if (ep + 1) % 100 == 0:
        print(f"ep {ep+1:4d} | eval={evaluate():6.1f} | "
              f"train50={np.mean(tr[-50:]):6.1f}")
```

> 注意：eval 是唯一可信进度条——训练期回报含采样噪声（§4.3 会看到两者背离）。

### 4.3 真实运行结果

seed=0、CPU 实测（每 100 回合：eval = 贪心 10 回合均值；|g| = 裁剪后梯度范数）：

```text
ep  100 | eval=  35.3 | |g|=1.32
ep  200 | eval= 227.1 | |g|=1.90
ep  300 | eval= 409.0 | |g|=2.47
ep  400 | eval= 439.6 | |g|=3.33
ep  500 | eval= 301.1 | |g|=3.80     ← 中途回落（A2C 的常态波动）
ep  600 | eval= 297.7 | |g|=4.25
ep  700 | eval= 500.0 | |g|=5.36     ← 满分并保持
ep  800 | eval= 500.0 | |g|=4.33
```

对比 07 章 REINFORCE（300 回合满分）：本实现更慢。原因诚实地说：整回合 rollout 的 A2C 每回合只更新一次，跟 REINFORCE 的更新节奏一样，而 Critic 的引入收益（部分轨迹可用、在线更新）在这个设定下用不上。**A2C 的真正优势在固定 T 的部分轨迹更新**：哪怕 agent 正在一条长回合中间，每攒 16 步就能更新一次参数。工程版（固定 T + 向量化环境）在 12 章实现；本版的价值在于看清三个损失的组合与 GAE 的位置。

## 5. 调参事故复盘：MSE 价值损失毁掉共享主干（真实数据）

本章实现过程中真实发生的事故，比任何"常见误区"都有说服力：

**现象**：初版用 `MSELoss` 算价值损失（其余全部同款超参），训练 600 回合评估分**始终卡在 9–25**（随机水平是 ~22），梯度范数（裁剪前）长期在 6–20。

**诊断**：CartPole 的 return 量级 10–500，初期 $V(s)\approx 0$，MSE 损失 $(0 - G)^2$ 量级几百且梯度极大——**价值头的巨大梯度穿过共享主干，把策略头赖以工作的特征每个更新都拽一次**。策略相当于"在地震中学习"，熵塌缩后锁死在坏策略上。换 **Huber（smooth_l1）** 后：

```text
MSE 版:  |g| = 6.5 ~ 20.1, eval 600 回合 ≤ 25        （学不动）
Huber 版: |g| = 1.3 ~ 5.4, eval ep700 = 500          （正常收敛）
```

**教训**（Sutton & Barto §13.5 未强调、但工程上致命）：共享主干时，**两条头的梯度尺度必须匹配**。价值目标量级大 → 用 Huber 或对 return 做归一化；或者干脆分离网络；或者给价值头单独的低学习率。排查顺序（经验做法）：不收敛 + 梯度范数异常大 → 先看哪一项损失的量级失控，再谈调 lr。

## 6. A3C：异步思想与它的退场

**A3C**（Asynchronous Advantage Actor-Critic，Mnih et al., ICML 2016）的核心不是算法而是**并行架构**：16 个 worker 线程各自持有环境副本与网络副本，独立采样、独立算梯度、**异步地**把梯度推给全局共享参数；每个 worker 周期性拉取最新参数继续采样。

两个贡献：

1. **数据多样性**：16 个 worker 的环境状态各不相同（有的在开局、有的在深处），等效于 16 个并行的探索线，天然打破时间相关性（06 章经验回放在 on-policy 语境下的替代品）；
2. **lock-free 更新**：不等慢 worker，GPU/CPU 永不空转——2016 年算力背景下的务实设计。

**为什么被 A2C/PPO 的同步版本取代**（OpenAI 的对比实验见 OpenAI Baselines 博客 2017, "A2C is A3C without asynchrony"）：

- **陈旧梯度（stale gradients）**：worker 更新期间全局参数已被别人改过，推上去的梯度基于旧参数——等效于不可控的"过去策略"数据混入，on-policy 性质被悄悄破坏；
- **GPU 时代并行成本反转**：异步多线程的意义在 CPU 集群；GPU 上批量同步计算反而更快。同步 A2C 用"8 个环境各采 5 步凑一批（batch=40）"达到同等多样性，实现简单、完全可复现；
- **复现性差**：异步时序不可固定种子，bug 无法重放。

**经验做法**：现代实现里"异步"移到了**环境侧**（向量化环境并行采样，12 章），"同步批量更新"留在**学习侧**。记住分工：采样并行化解决的是吞吐，梯度同步化解决的是正确性与可复现。

## 7. 常见误区

1. **价值损失用 MSE 且量级失控**（§5 全文）。症状优先级最高的排查项：训练完全不学 + 裁剪前梯度范数 > 10。先检查价值目标的量级，再检查共享主干。
2. **$\lambda$ 从 0 开始调**。§2.2 实测 $\lambda=0$ 直接不学习。GAE 的 $\lambda$ 永远从 0.95 起步——它不是"探索率"那种可大幅调节的旋钮，而是"信用分配视野"的微调。
3. **截断（truncation）时把 bootstrap 清零**。整回合 rollout 在 `truncation`（时间上限）结束时 `next_v` 应该用 $V(s_{T+1})$ 而不是 0——与 05/06 章同一个 bug 的 A2C 版。CartPole 500 步上限的场景下症状：策略在 480 步附近开始送死。
4. **优势归一化不分场合。** 回合内归一化（本实现）在整回合几百步时是稳态增益；但固定 T=8 的小 rollout 上，把 8 个数强行归一到 ±1 会把纯噪声放大成满幅信号（§4 教学版之所以用整回合就是这个原因）。**经验做法**：归一化的样本规模至少几十，且记住它只是稳定器不是算法部件。
5. **熵系数设 0 之后困惑"为什么突然不探索了"。** 策略梯度没有外生探索机制（对比 DQN 的 ε-greedy），策略分布本身的熵是唯一探索来源。训练中期评估分骤降，先查熵曲线是否塌缩。

## 本篇小结

- Actor-Critic = 策略梯度 + 学出来的基线：$\delta_t$ 是优势的无偏单样本估计，一个 $V$ 网络同时解决基线、自举、continuing 三个问题。
- GAE：$\hat A_t = \sum (\gamma\lambda)^l \delta_{t+l}$，一行倒序递推；$\lambda$ 参数化偏差-方差光谱，实测 $\lambda=0$ 完全不学（9.3）、0.9–1.0 才可用。
- A2C 三项损失：策略（GAE 加权 log-prob）+ 价值（Huber 回归 λ-return）− 熵奖励；共享主干时两头的梯度尺度必须匹配（真实事故：MSE 价值损失让 eval 卡死 ≤25，Huber 后 ep700 满分）。
- A3C 的异步范式被同步版本取代：陈旧梯度破坏 on-policy 性质、不可复现；现代分工是"环境并行 + 学习同步"。
- 08 章的 A2C 离 SOTA 还差最后一块：策略更新没有"步长控制"，一次大更新就能毁掉策略——09 章 PPO 的 clip 登场。

## 练习

1. 把 §4 的价值损失换回 MSE 复现 §5 的事故：打印每 50 回合的 `v_loss.item()`、梯度范数与 eval，画出"价值梯度淹没策略梯度"的证据曲线。
2. 把 rollout 从整回合改成固定 T=8（回合中断时保留跨 rollout 的中间量：用 `delta` 递推需要 $\hat A_{t+1}$，试着正确实现跨段 GAE），对比整回合版的收敛速度——体会 12 章向量化环境存在的理由。
3. 实现 $\lambda$-return 的显式版本（$\hat R_t = \sum (\gamma\lambda)^k G_{t}^{(k+1)}$ 的混合式）并验证它与 `adv + vals` 数值一致（提示：两者在数学上恒等，这是 GAE 论文的式 (16)）。
4. 给策略头加"熵塌缩监控"：每 100 回合打印 `dist.entropy().mean().item()`，观察它与评估分回落（ep 500–600）的对应关系。
5. 思考题：DQN 的目标网络"冻结目标 $C$ 步"与 PPO 的"同一批数据只更新几个 epoch"都在控制"目标漂移速度"。从 09 章 TRPO 的"信任域"视角，说明这两者是同一思想在不同部件（值函数/策略）上的实例化。

## 参考来源

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), §13.5（半梯度策略梯度的 critic）、§13.7（ Actor-Critic 与 TD 误差作优势估计）、§12.8（λ-return）
- GAE 原始论文：Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation", ICLR 2016（arXiv:1506.02438，式 16）
- A3C：Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning", ICML 2016
- A2C 同步版对比：OpenAI Baselines 博客 "A2C is A3C without asynchrony"，2017
- 本章 eval 曲线、λ 消融、MSE/Huber 事故数据均为 seed=0 真实运行结果，代码可复现
