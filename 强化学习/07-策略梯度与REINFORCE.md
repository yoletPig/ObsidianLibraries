# 07 - 策略梯度与 REINFORCE

## 本篇目标

读完本篇，你应该能做到：

1. 说清"为什么要直接学策略"的两个硬理由：连续动作空间、随机策略——以及 DQN 在这两处的天花板。
2. **分步推导策略梯度定理**（score function 视角），对"轨迹似然的梯度为什么可以不依赖环境梯度"形成肌肉记忆——这是 08/09/10 章一切算法的数学源头。
3. 实现带回报归一化的 REINFORCE（CartPole，实测 300 回合评估满分），并解释它为什么是"最简单也最难调"的算法。
4. 推导基线为什么降方差而不改变期望，并用固定策略实验实测：方差降低 44%（均值基线）与 62%（回归最优基线）。

至此值函数线走完（04–06 章）。本章换赛道：**跳过"先学值再导出策略"，直接对策略本身求梯度**（Sutton & Barto, 2nd ed., 第 13 章；Williams, 1992 原始提出 REINFORCE）。

## 1. 为什么要直接学策略

### 1.1 连续动作：DQN 的硬天花板

DQN 的策略是 $\arg\max_a Q(s,a;\theta)$——需要对所有动作取 max。离散 2 个动作没问题；**连续动作空间（如关节力矩 $\in [-2, 2]$）里 max 是一个内层优化问题**，每选一次动作都要跑一次梯度上升，既慢又不保证全局最优。直接学策略 $\pi_\theta(a|s)$ 则一步前向就输出动作，连续动作下输出高斯分布的均值即可。这个差别在机器人领域是生死级的。

### 1.2 随机策略有时就是最优的

值函数法导出的策略是确定性的（argmax）。但有些任务**最优策略本身就是随机的**：石头剪刀布（确定性策略必被针对）、部分可观测任务里的对称状态（POMDP，Sutton & Barto, 2nd ed., §13.1 的讨论）。策略梯度法直接输出分布参数，随机性是内建能力。

### 1.3 平滑性

$\arg\max$ 处梯度处处为零或不连续，$Q(s,a;\theta)$ 对动作的优化是"尖的"；而 $J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]$ 对 $\theta$ 是平滑的（分布对参数平滑），可以直接梯度上升。**代价也在这里：期望的梯度 = 梯度的期望，方差极大**——本章的基线、08 章 Actor-Critic、09 章 PPO 都在为压这个方差打工。

## 2. 目标函数与策略梯度定理

### 2.1 目标函数

设定：策略 $\pi_\theta(a|s)$ 是可微函数（如 `logits → softmax`），环境固定。要最大化**期望回报**：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \sum_\tau P(\tau;\theta)\, R(\tau)$$

其中轨迹 $\tau = (s_0, a_0, r_1, s_1, \dots)$，$R(\tau) = \sum_t \gamma^t r_{t+1}$，轨迹概率由环境与策略共同生成：

$$P(\tau;\theta) = \rho_0(s_0) \prod_t P(s_{t+1}|s_t, a_t)\, \pi_\theta(a_t|s_t)$$

### 2.2 策略梯度定理推导（四步）

**第 1 步：对 $J$ 求梯度，碰到"环境梯度不可得"的墙。**

$$\nabla_\theta J(\theta) = \nabla_\theta \sum_\tau P(\tau;\theta) R(\tau) = \sum_\tau \nabla_\theta P(\tau;\theta)\, R(\tau)$$

$R(\tau)$ 与 $\theta$ 无关（环境奖励在动作之后给定），所以梯度只打在 $P$ 上。但 $P$ 里含环境的 $P(s_{t+1}|s_t,a_t)$——**真实环境里你不知道转移函数，$\nabla_\theta P$ 拿不到**。

**第 2 步：log-导数技巧（score function trick）把环境因子消掉。**

对任意正函数 $f$，$\nabla f = f \nabla \log f$。应用于 $P(\tau;\theta)$：

$$\nabla_\theta P(\tau;\theta) = P(\tau;\theta) \nabla_\theta \log P(\tau;\theta)$$

而 $\log P(\tau;\theta) = \log \rho_0(s_0) + \sum_t \left[\log P(s_{t+1}|s_t,a_t) + \log \pi_\theta(a_t|s_t)\right]$，对 $\theta$ 求梯度时**环境项与初始分布项全部归零**（它们不含 $\theta$）：

$$\nabla_\theta \log P(\tau;\theta) = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)$$

**环境的贡献在 log 空间里消失了**——这就是"为什么不需要知道环境模型也能求策略梯度"的全部秘密。

**第 3 步：代回并还原成期望。**

$$\nabla_\theta J(\theta) = \sum_\tau P(\tau;\theta) \left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\right] R(\tau) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\, R(\tau)\right]$$

**第 4 步：因果性收紧（reward-to-go）。** $\log \pi_\theta(a_t|s_t)$ 不影响 $t$ 之前的奖励（过去的已经发生，改变 $\theta$ 不能改变历史），所以 $R(\tau)$ 可以换成 $t$ 之后的折扣回报 $G_t$（Sutton & Barto, 2nd ed., 式 13.5 及其关于 causality 的讨论）：

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\, G_t\right]}$$

**读法**（score function 视角）：$\nabla_\theta \log \pi_\theta(a_t|s_t)$ 叫 **score function**——"提高这个动作概率的方向"；$G_t$ 是权重——"这个方向该加权多少"。整个公式读作：**把"得分高的轨迹的动作概率推高、得分低的推低"**。RL 的本质第一次显形：不是拟合标签，是**按结果加权调整生成概率**。

用数值验证一遍推导（这是永远的好习惯，PyTorch autograd 当裁判）：

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

torch.manual_seed(0)
logits_net = nn.Linear(4, 2)
obs = torch.randn(4)
a = torch.tensor([1])

# 自动微分直接算 ∇J (单样本蒙特卡洛估计)
logits = logits_net(obs)
logp = Categorical(logits=logits).log_prob(a)
logp.backward()
g_autodiff = logits_net.weight.grad.clone()

# 手工 score function: ∂logπ(a|s)/∂θ = ∂logits[a]/∂θ - Σ_a' π(a'|s) ∂logits[a']/∂θ
logits_net.zero_grad()
logits = logits_net(obs)
logits[a].backward()
g_direct = logits_net.weight.grad.clone()      # ∂logits[a]/∂θ
logits_net.zero_grad()
(logits.softmax(-1) * logits).sum().backward()  # Σπ·logits 的梯度（∂/∂θ E_z[logits_z]）
g_expect = logits_net.weight.grad.clone()      # 对"采样分布期望"的部分
print("autograd 与 score function 手工式之差:",
      (g_autodiff - (g_direct - g_expect)).abs().max().item())   # ≈ 0
```

运行结果：

```text
autograd 与 score function 手工式之差: 4.76837158203125e-07   （浮点误差级，推导无误）
```

$\nabla_a \log \pi(a|s) = \partial_a \text{logits}[a] - \sum_{a'} \pi(a'|s)\,\partial_a \text{logits}[a']$——"实际动作的 logits 梯度减去采样分布期望的 logits 梯度"。softmax 梯度的这个结构在 09 章 PPO 的实现里会原样复用。

## 3. REINFORCE 完整实现（CartPole）

### 3.1 算法与代码

REINFORCE（Williams, 1992, *Machine Learning*；Sutton & Barto, 2nd ed., §13.3）= 策略梯度定理的蒙特卡洛版本：**整条轨迹跑完，用实测 $G_t$ 当权重更新一次**。

```python
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Categorical

np.random.seed(0); torch.manual_seed(0)
env = gym.make("CartPole-v1")

pi = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2))
opt = torch.optim.Adam(pi.parameters(), lr=1e-3)
gamma = 0.99

def evaluate():
    """固定评估协议: 贪心 + 独立种子 + 10 回合均值"""
    ev = gym.make("CartPole-v1")
    total = 0.0
    for ep in range(10):
        obs, _ = ev.reset(seed=10000 + ep)
        done = False
        while not done:
            with torch.no_grad():
                a = int(torch.argmax(pi(torch.tensor(obs, dtype=torch.float32))))
            obs, r, term, trunc, _ = ev.step(a)
            total += r; done = term or trunc
    return total / 10

for ep in range(600):
    obs, _ = env.reset(seed=ep)
    logps, rewards, done = [], [], False
    while not done:
        dist = Categorical(logits=pi(torch.tensor(obs, dtype=torch.float32)))
        a = dist.sample()                       # 从当前策略采样（on-policy）
        logps.append(dist.log_prob(a))
        obs, r, term, trunc, _ = env.step(int(a))
        rewards.append(r); done = term or trunc
    # 倒序算 reward-to-go
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    adv = np.array(returns, dtype=np.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # 回报警归一化（实用关键，见 §4）
    loss = -(torch.tensor(adv) * torch.stack(logps)).sum()   # 逐步加权的负对数似然
    opt.zero_grad(); loss.backward(); opt.step()
    tr.append(len(rewards))                     # 记录回合长度当训练回报
    if (ep + 1) % 100 == 0:
        print(f"ep {ep+1:4d} | eval={evaluate():6.1f} | "
              f"train50均={np.mean(tr[-50:]):6.1f}")
```

真实运行结果（seed=0，每 100 回合；tr 需在循环前初始化为 `tr = []`）：

```text
ep  100 | eval= 132.4 | train(50回合窗口)=  33.6
ep  200 | eval= 432.3 | train(50回合窗口)=  83.3
ep  300 | eval= 500.0 | train(50回合窗口)= 155.5    ← 评估满分并保持
ep  400 | eval= 490.2 | train(50回合窗口)= 252.0
ep  500 | eval= 500.0 | train(50回合窗口)= 331.2
ep  600 | eval= 500.0 | train(50回合窗口)= 385.5
```

实现要点逐条说：

- **`loss = -(adv * logps).sum()`**：负号是因为优化器做**下降**而我们要求**上升**；`sum` 而不是 `mean` 意味着"轨迹越长梯度整体越大"（更长的信用分配链），二者皆可，但与学习率的匹配不同（经验做法：用 `sum` 时 lr 减半起步更稳）。
- **回报归一化**（`(adv - mean)/std`）：只缩放不改变各步相对大小，等效自适应学习率，是 REINFORCE 能稳定收敛的头号功臣——它就是一种**强基线**（§4）。
- **没有回放、没有目标网络**：REINFORCE 是纯 on-policy + MC（无自举）——06 章 Deadly Triad 三件套一件都不占，**这是它理论干净的原因，也是它样本效率低的原因**（每条数据用一次就扔）。

### 3.2 高方差：从数据看

对比 06 章 DQN：DQN 200 回合 eval 306；REINFORCE 300 回合才满分。慢在哪？梯度估计的方差。单条轨迹的 $G_t$ 量级 1–500（CartPole 回合长度波动巨大），同样均值下波动越大 SGD 步子越乱。这引出本章最后一节。

## 4. 基线：不改变期望，只压方差

### 4.1 推导

把策略梯度里的 $G_t$ 换成 $G_t - b$（$b$ 是任何**与动作 $a_t$ 无关**的量）：

$$\mathbb{E}_{a\sim\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s)(G_t - b)\right] = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) G_t\right] - b\, \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s)\right]$$

第二个期望恒为零：

$$\mathbb{E}_{a\sim\pi}\left[\nabla_\theta \log \pi(a|s)\right] = \sum_a \pi(a|s) \frac{\nabla_\theta \pi(a|s)}{\pi(a|s)} = \nabla_\theta \sum_a \pi(a|s) = \nabla_\theta 1 = 0$$

**基线不改变梯度期望（无偏），但能降低估计方差**（Sutton & Barto, 2nd ed., §13.4）。最优基线有解析解（对 $\text{Var}(\nabla \log \pi \cdot (G-b))$ 求导置零）：

$$b^* = \frac{\mathbb{E}\left[\|\nabla_\theta \log \pi\|^2\, G\right]}{\mathbb{E}\left[\|\nabla_\theta \log \pi\|^2\right]}$$

即"按 score 的平方加权的回报均值"——通常接近但不等于 $\mathbb{E}[G]$。

### 4.2 实测：固定策略下三种基线的方差

实验设计：训练 120 回合得到中等水平策略，**冻结参数**，从它采样 300 条轨迹；对每条轨迹分别用三种权重计算梯度估计向量 $g = \sum_t \nabla \log \pi(a_t|s_t)\, (G_t - b)$，统计 300 个梯度样本的逐维方差：

```python
# 核心计算（完整脚本见练习提示；此处展示关键三行）
b = float(all_G.mean())                       # 全体逐步回报的均值
g = torch.autograd.grad(-((ret_t - b) * torch.stack(logps)).sum(), params)
# b=0 / b=均值 / b=回归最优 三种情形分别统计 torch.stack(g_list).var(dim=0).mean()
```

真实运行结果（N=300 条轨迹，冻结策略平均回合长约 50 步）：

| 基线 | 逐步回报全体均值 | 梯度方差（逐维均值） | 相对 b=0 |
| --- | --- | --- | --- |
| $b=0$ | — | 121.4 | 100% |
| $b=\mathbb{E}[G_t]$ | 26.21 | 68.1 | **56%**（降 44%） |
| $b=b^*$（回归估计） | 17.75 | 46.7 | **38%**（降 62%） |

三个读数：

1. 均值基线降 44% 方差——**免费午餐**（期望不变、纯降方差），这就是"回报归一化"有效性的定量解释；
2. 回归最优基线比朴素均值更低（46.7 vs 68.1）——$b^*$ 与 $\mathbb{E}[G]$ 确实不是一回事（26.2 vs 17.8）；
3. 基线把"平均回报高"的Common成分从梯度里剔掉，只留"这个动作**相对**好多少"——这正是 08 章**优势函数** $A(s,a) = Q(s,a) - V(s)$ 的定义：**V(s) 就是基线，A 就是去均值后的信号**。08/09 章的一切优势估计都是本节结论的函数近似版。

## 5. 与 DQN 的本质差异

| 维度 | DQN（值函数线） | REINFORCE（策略梯度线） |
| --- | --- | --- |
| 学什么 | $Q(s,a;\theta)$，策略 = argmax | $\pi_\theta(a|s)$ 本身 |
| 目标 | 贝尔曼方程的 TD 误差 | $J(\theta)$ 的策略梯度 |
| 数据用法 | off-policy，回放反复用 | on-policy，一批用一次 |
| 自举 | 有（TD 目标） | 无（MC 实测回报） |
| 连续动作 | 难（内层 max 优化） | 自然（输出分布参数） |
| 随机策略 | 不行（argmax 是确定的） | 天然支持 |
| 样本效率 | 高（数据复用+自举） | 低（数据用一次） |
| 收敛性质 | Deadly Triad 不稳定 | 无偏但高方差（基线可缓解） |

一句话总结：**DQN 用"值"换"样本效率"，REINFORCE 用"样本"换"表达力与稳定性"**。Actor-Critic（08 章）就是把两者嫁接：策略网络保留表达力，值网络补回样本效率。

## 6. 常见误区

1. **对 `log_prob` 求 `mean` 而不是按步加权。** 损失写成 `-(adv * logps).mean()` 把"每步梯度"除以了轨迹长度，长轨迹的单步信用被稀释——行为上表现为长任务学习极慢。正确式是逐元素乘后 `sum`（或明确 `mean` over batch 维，二选一并固定）。
2. **忘了 `dist.sample()` 必须在 `no_grad` 之外**（训练时）。采样本身不需要梯度，`log_prob` 才需要——常见错误是把整个分布对象放进 `no_grad` 再取 `log_prob`，梯度全断，loss 反向报错 `does not require grad`。
3. **基线与动作相关。** 基线合法性的唯一条件是"不能依赖当前动作"（§4.1 推导只用到了 $\sum_a \pi \nabla \log \pi = 0$）。把基线取成 $Q(s,a)$ 或 $G_t$ 本身，梯度期望被破坏——学习方向错而不自知。**经验做法**：基线只允许依赖状态（$V(s)$）或轨迹统计量（batch 均值）。
4. **回报不归一化直接上。** CartPole 回合长度从 8 步到 500 步，$G_t$ 动态范围 50 倍以上，不归一化时梯度方向被"长回合轨迹"垄断。症状：训练前期极慢，某次运气好的超长回合后突然巨震。归一化是 REINFORCE 实用化的第一步（§3 实测曲线依赖它）。
5. **把 REINFORCE 曲线的锯齿当 bug。** 无偏 + 高方差意味着损失和回报曲线必然锯齿密布；判断学习是否健康看的是**几十回合窗口均值**的趋势，不是单点。这与 01 章"RL 评估纪律"是同一条原则。

## 本篇小结

- 直接学策略的三个理由：连续动作无内层 max、随机策略可表达、目标平滑可微。
- 策略梯度定理四步推导：梯度打在轨迹概率上 → log-导数技巧消掉环境因子 → 还原为期望 → 因果性收紧到 reward-to-go。score function 视角："按回报加权推拉动作概率"。数值验证 autograd 与手工式一致（误差 5e-7）。
- REINFORCE = 策略梯度的 MC 版：整轨迹 + 实测 $G_t$；回报归一化是实用关键；实测 300 回合评估满分，但样本效率低于 DQN（数据用一次就扔）。
- 基线定理：$\mathbb{E}[\nabla\log\pi \cdot b] = 0$ 对任意与动作无关的 $b$ 成立——无偏降方差；实测方差 121.4 → 68.1（均值基线）→ 46.7（最优基线）。
- 优势函数 $A = Q - V$ 就是"以 $V$ 为基线"的策略梯度信号——08 章的桥已经搭好。

## 练习

1. 把 §3 代码的回报归一化删掉重跑 600 回合，画 eval 曲线与正文对比。量化"动态范围失控"对前期收敛速度的影响。
2. 实现 §4.2 的完整方差实验脚本（训练 120 回合预热 → 冻结 → 采样 300 条 → 三种基线统计方差），验证你得到的数字与正文同量级（绝对值因策略水平不同会变，比例关系应保持）。
3. 把 CartPole 换成 `Pendulum-v1`（连续动作）：策略输出改为高斯分布的均值与 log 标准差（`Normal(mu, sigma)`，动作 tanh 压缩到 $[-2,2]$），实现连续版 REINFORCE。体会"策略梯度对连续动作免费"是什么意思。
4. 推导题：对 $\text{Var}[g(b)] = \mathbb{E}[\|\nabla \log \pi\|^2 (G-b)^2] - \|\mathbb{E}[\nabla \log \pi (G-b)]\|^2$ 关于 $b$ 求导，证明最优基线为 $b^* = \frac{\mathbb{E}[\|\nabla \log \pi\|^2 G]}{\mathbb{E}[\|\nabla \log \pi\|^2]}$。
5. 思考题：GRPO（库内 Stage 7 的主角）对同一条 prompt 采样一组回答，用"组内回报减组均值"当优势——用本章语言说明这个操作属于什么（基线？归一化？），以及它为什么天然适配"单条轨迹 = 整个回答、奖励只在末尾"的 LLM 场景。（10 章正式展开。）

## 参考来源

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), §13.1–13.4（策略梯度定理、REINFORCE、基线）；图 13.2/13.4 的实验设计思想
- REINFORCE 原始论文：Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning", *Machine Learning* 8, 1992
- score function 与 log-导数技巧的统计学背景：Casella & Berger, *Statistical Inference*, §3.4（指数族与得分函数）
- 最优基线：Greensmith, Barto & Wolfe, "Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning", JMLR 5, 2004
- 本章所有数值（eval 曲线、方差表、autograd 验证）为 seed=0 真实运行结果，代码可复现
