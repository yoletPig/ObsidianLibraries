# 06 - DQN：值函数遇上深度网络

## 本篇目标

读完本篇，你应该能做到：

1. 说清 **Deadly Triad**（自举 + 函数近似 + off-policy）为什么让深度 RL 天生不稳定，以及"表格式稳定证明"在神经网络面前失效的具体环节。
2. 解释经验回放的两个作用（打破时间相关性 + 提高样本复用）与目标网络的作用（给自举目标一个"固定的靶子"），并能预判"去掉其中一个会发生什么"。
3. 完整实现 DQN（PyTorch，CartPole，含超参表），跑出**真实的收敛曲线与真实的中途崩塌**——本章两个实验数据都来自实际训练。
4. 用一句话原理掌握 Double DQN / Dueling / Priority Replay 三个变体，理解它们分别修 DQN 的哪个部件。

05 章结尾的结论：CartPole 状态连续，表格放不下，需要 $Q(s,a;\theta)$。本章把这个"换表为网"做对——你会发现 DQN 没有任何新数学，全部难度都在**稳定性工程**上（Mnih et al., 2013 workshop 版 / 2015, *Nature*）。

## 1. Deadly Triad：不稳定是默认现象

04 章预警过的三件套（Sutton & Barto, 2nd ed., §11.3）：

$$\underbrace{\text{自举}}_{\text{TD 的 } r + \gamma V(s')} + \underbrace{\text{函数近似}}_{\text{网络 } Q(s,a;\theta)} + \underbrace{\text{off-policy}}_{\text{数据分布} \neq \text{目标分布}}$$

为什么三者同时出现就危险？逐项拆开：

- **自举**的目标里含有 $Q(s';\theta)$ 本身——梯度更新"追一个自己也在动的靶子"；
- **函数近似**让 $\theta$ 的一次更新**同时改变所有状态**的 $Q$ 值。你修正状态 $s_1$ 的误差，可能把 $s_2, s_3$ 的估计推歪（03 章表格法的"每个状态独立记账"彻底失去）；
- **off-policy** 使数据分布 $\rho_\beta(s)$ 与目标分布不匹配，网络在被采样到的状态上过拟合、在没采到的状态上任意漂移。

三者叠加的正反馈回路：$Q$ 高估某个动作 → 策略倾向它 → 更多相关数据 → 更新高估 → 发散。DQN 的两大机制各自切断这个回路的一环。

## 2. 经验回放：打破时间相关性

**机制**：把交互四元组 $(s, a, r, s')$ 存进容量 $N$ 的循环缓冲区，训练时从缓冲区**均匀随机**抽 batch（Mnih et al., 2015, *Nature*, Experience Replay）。

两个作用：

1. **打破时间相关性**。监督学习默认样本 i.i.d.；RL 数据是一条轨迹上连续决策产生的——相邻样本高度相关（CartPole 连续 100 步都是"杆往左倾"的状态）。SGD 的前提被破坏，梯度估计有系统性偏移。随机抽样把"时间上连续"变成"分布上均匀"，近似恢复 i.i.d.。
2. **提高样本复用**。每条经验参与多次梯度更新。RL 的数据是环境交互换来的（贵），监督学习数据可以无限重读（便宜）——回放是 RL 里少有的"白捡"的样本效率。

```python
from collections import deque
import random

buf = deque(maxlen=50000)                    # 循环缓冲：满了自动挤掉最旧
# 存：buf.append((obs, a, r, ns, float(term)))   term=1 才记终止，trunc 不记
# 取：s, a, r, ns, d = zip(*random.sample(buf, batch))
```

**为什么 `term` 与 `trunc` 必须分开存**（01 章埋的伏笔在此兑现）：`truncated`（到步数上限）时下一状态不是真实终止，未来仍有价值，target 里 $V(s')$ **不能清零**；只有 `terminated`（杆真的倒了）才清零。05 章表格法的教训在深度版里同样致命。

## 3. 目标网络：给自举一个固定靶子

**机制**：维护两份网络——在线网络 $Q(s,a;\theta)$（每步更新）与目标网络 $Q(s,a;\theta^-)$（每 $C$ 步从在线网络**硬拷贝**一次权重，本系列取 $C=500$ 步）。

TD 目标写成：

$$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

**为什么冻结能稳定？** 看"追自己尾巴"问题：没有目标网络时，损失 $\big(Q(s,a;\theta) - [r + \gamma\max Q(s';\theta)]\big)^2$ 里 $\theta$ 同时出现在两端——你把 $Q(s,a;\theta)$ 往下压，反向传播同时把 $Q(s';\theta)$ 也往下拉，目标跟着移动，可能形成振荡不收敛的追逐。目标网络让目标在 $C$ 步内**常数化**，梯度下降退化为"朝固定目标回归"——正是 03 章策略评估的收敛模式。代价是目标滞后于现实 $C$ 步（偏差换稳定，Deep RL 里反复出现的交易）。

**经验做法**：目标网络更新频率 $C$ 与学习率是联动超参——lr 大时目标要更新得更勤（否则滞后偏差累积）；硬拷贝（直接赋值）vs 软更新（$\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$，DDPG 引入）差异不大，软更新实现更平滑，CartPole 级任务两者都能收敛。

## 4. 完整实现（CartPole）

### 4.1 超参表

| 超参 | 值 | 说明 |
| --- | --- | --- |
| 网络 | MLP 2×128, ReLU | 小任务不需要 CNN/更大网络 |
| 优化器 | Adam, lr=1e-3 | 偏大；追求稳可降 5e-4 |
| $\gamma$ | 0.99 | CartPole 惯例 |
| 回放容量 | 50000 | CartPole 轨迹短，够存几千回合 |
| batch size | 128 | CPU 友好 |
| 学习起始 | 2000 步 | 先攒够数据再学，避免早期噪声主导 |
| 训练频率 | 每 4 步一次更新 | 交互/更新比 4:1 |
| 目标网络同步 | 每 500 步硬拷贝 | §3 的 $C$ |
| ε | 1.0 → 0.05, ×0.995/步 | 按步衰减（不是按回合） |
| 损失 | Huber（smooth_l1） | 比 MSE 对离群 TD 误差钝感 |
| 梯度裁剪 | max_norm=10.0 | 死保护 |

### 4.2 全量代码

```python
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)  # 三处种子全固定

env = gym.make("CartPole-v1")
nS, nA = env.observation_space.shape[0], env.action_space.n   # 4, 2

class QNet(nn.Module):
    def __init__(self, nS, nA, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nS, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, nA))
    def forward(self, x):
        return self.net(x)                    # 输出每个动作的 Q 值 (batch, nA)

q, qt = QNet(nS, nA), QNet(nS, nA)
qt.load_state_dict(q.state_dict())            # 目标网络初始同步
opt = torch.optim.Adam(q.parameters(), lr=1e-3)
buf = deque(maxlen=50000)
gamma, batch, eps, eps_min, eps_decay = 0.99, 128, 1.0, 0.05, 0.995
target_sync, learn_start, train_freq = 500, 2000, 4

def act(obs, eps):
    if random.random() < eps:
        return random.randrange(nA)
    with torch.no_grad():                     # 采样时不建图
        return int(torch.argmax(q(torch.tensor(obs, dtype=torch.float32))))

def evaluate(qnet, episodes=10):
    """固定评估协议: 10 个 episode 均值, 贪心策略, 独立种子"""
    ev = gym.make("CartPole-v1")
    total = 0.0
    for ep in range(episodes):
        obs, _ = ev.reset(seed=10000 + ep)
        done = False
        while not done:
            with torch.no_grad():
                a = int(torch.argmax(qnet(torch.tensor(obs, dtype=torch.float32))))
            obs, r, term, trunc, _ = ev.step(a)
            total += r; done = term or trunc
    return total / episodes

step_count, ep_returns = 0, []
for ep in range(800):
    obs, _ = env.reset(seed=ep)
    total, done = 0.0, False
    while not done:
        a = act(obs, eps)
        ns, r, term, trunc, _ = env.step(a)
        buf.append((obs, a, r, ns, float(term)))   # 关键: term 不是 trunc
        obs = ns; total += r; done = term or trunc
        step_count += 1
        eps = max(eps_min, eps * eps_decay)
        if step_count >= learn_start and step_count % train_freq == 0:
            s, a_b, r_b, ns_b, d_b = zip(*random.sample(buf, batch))
            s     = torch.tensor(np.array(s), dtype=torch.float32)      # (128, 4)
            a_b   = torch.tensor(a_b, dtype=torch.int64)                # (128,)
            r_b   = torch.tensor(r_b, dtype=torch.float32)              # (128,)
            ns_b  = torch.tensor(np.array(ns_b), dtype=torch.float32)   # (128, 4)
            d_b   = torch.tensor(d_b, dtype=torch.float32)              # (128,)
            q_sa = q(s).gather(1, a_b.view(-1, 1)).squeeze(1)   # Q(s,a) 只取执行动作列
            with torch.no_grad():                               # 目标不回传梯度
                max_next = qt(ns_b).max(dim=1).values
                target = r_b + gamma * (1 - d_b) * max_next     # (1-d): 终止清零
            loss = nn.functional.smooth_l1_loss(q_sa, target)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
        if step_count % target_sync == 0:
            qt.load_state_dict(q.state_dict())                  # 硬拷贝同步
    ep_returns.append(total)
    if ep % 100 == 99:
        print(f"ep {ep+1:4d} | eval={evaluate(q, 10):6.1f} | "
              f"train100均={np.mean(ep_returns[-100:]):6.1f}")
```

### 4.3 真实运行结果

seed=0、CPU、gymnasium 1.4.0 / PyTorch 2.14 实测（每 100 回合：评估 10 回合均值 + 训练期最近 100 回合均值）：

```text
ep  100 | eval=   9.5 | train100均=  35.0
ep  200 | eval= 306.2 | train100均=  76.7
ep  300 | eval= 312.1 | train100均= 270.6
ep  400 | eval= 500.0 | train100均= 243.0     ← 评估满分
ep  500 | eval= 500.0 | train100均= 331.7     ← 满分持续
ep  600 | eval= 104.0 | train100均= 123.3     ← 崩塌
ep  700 | eval= 117.4 | train100均= 117.5
ep  800 | eval= 235.4 | train100均= 127.7
```

三个读数：

1. **能学**：200 回合内评估分从 9.5 升到 306，400 回合满分——DQN 在 CartPole 上学习效率确实高。
2. **会崩**：600 回合处评估分 500 → 104。这不是 bug，是 §1 正反馈回路的真实发作（Q 值整体被推高 → 策略失衡 → 崩塌）。**多种子下崩塌时点不同但迟早发生**，这是 Deadly Triad 的日常。
3. **eval 与 train 曲线长期背离**（ε 探索拖低训练期回报）——再次强调 01 章的评估纪律：看 eval，不看 train。

修复方向即 §5 的变体。先把 Double DQN 换进 §4.2 的 target 计算（其余不动）：

```python
with torch.no_grad():
    a_star = q(ns_b).argmax(dim=1, keepdim=True)        # 在线网络选动作
    max_next = qt(ns_b).gather(1, a_star).squeeze(1)    # 目标网络估值
    target = r_b + gamma * (1 - d_b) * max_next
```

同超参同种子实测：

```text
ep  100 | eval=  95.1 | train100均=  45.3
ep  200 | eval= 153.2 | train100均= 196.5
ep  400 | eval= 362.6 | train100均= 261.3
ep  500 | eval= 500.0 | train100均= 282.2
ep  600 | eval= 500.0 | train100均= 435.6     ← 多稳了 100 回合
ep  700 | eval=  21.3 | train100均= 340.8     ← 仍会崩（但更晚）
```

Double DQN 把满分窗口延长了 100 回合，但**没有根除崩塌**——诚实的结论：CartPole 级任务上 DQN 家族"能学但不保证一直好"，要稳定要么继续加机制（Rainbow 全家桶），要么换策略梯度路线（07–09 章）。这个判断对选型很重要。

## 5. 改进变体概览

三个最重要的变体，各修一个部件（合并版合称 Rainbow，Hessel et al., AAAI 2018）：

| 变体 | 一句话原理 | 修的是哪个部件 |
| --- | --- | --- |
| **Double DQN**（van Hasselt et al., AAAI 2016） | 把"选动作"和"评估动作"分给两个网络：$\max_a Q_-(s',a)$ 改为 $Q_-(s', \arg\max_a Q(s',a))$ | §2.2 的乐观偏差——两个网络的估计误差不相关，取 max 前先"过一道独立的裁判"，系统性高估被解耦掉 |
| **Dueling DQN**（Wang et al., ICML 2016） | 把 $Q$ 分解为 $V(s) + A(s,a) - \bar{A}(s)$ 两条头分别估计 | 状态价值与动作优势解耦——很多状态"动作不重要"，不必每个动作都学得好 |
| **Priority Replay**（Schaul et al., ICLR 2016） | 回放抽样概率 ∝ TD 误差大小（学得差的经验多学），配重要性采样权重纠偏 | 均匀回放的浪费——高 TD 误差的样本信息量大，优先喂给网络 |

三者相互正交（Double 改 target、Dueling 改网络结构、Priority 改采样），可叠加。**经验做法**：现代实践中 DQN 家族的优先级是——先 Double（一行改动），再 Priority（收益明确但要小心 $\beta$ 调度），Dueling 收益视任务而定；若是连续动作，整个 DQN 家族直接出局（`max_a` 对连续 $a$ 无法枚举，需要每次梯度上升求解内层优化）——这条限制就是 07 章策略梯度的入场理由之一。

## 6. 常见误区

1. **`term` 与 `trunc` 不分**。`done = term or trunc` 存进 buffer 当终止标志：CartPole 500 步上限触发时未来明明有价值却被清零。症状：训练后期 agent 学会"在 490 步附近故意松手"（早死不亏）。01/05 章两次铺垫，这是 RL 实现第一大 bug。
2. **采样时没包 `torch.no_grad()`**。推理选动作时建了计算图，缓存堆积内存暴涨，且 argmax 结果不受影响纯属浪费；更隐蔽的是把图留到了训练步导致梯度错乱。经验做法：一切"用网络算数但不训练"的调用都包 `no_grad`。
3. **目标计算没有 `no_grad`**。`target = r + gamma * max_next` 若在图内，梯度会穿过目标传播——等于"同时优化目标和预测"，TD 误差恒 0，loss 表面收敛、Q 值死水一潭。症状：loss 很小但 eval 一直随机水平。
4. **回放里存了未归一化/未复制的 numpy 视图**。`buf.append((obs, ...))` 若 `obs` 是环境内部数组的引用，`env.step` 会原地改写它——buffer 里全部变成同一个最新状态。gymnasium 的 obs 通常已拷贝，但自定义环境（12 章）常踩。症状：训练数据"看起来都是终点附近的状态"。
5. **只看 train 返回评估进度**。ε 探索下 train 100 均值长期低于 eval 数倍（本章 500 vs 243~332），把 train 曲线当进度条会提前停止训练或误判崩溃。固定评估协议（独立种子 + 贪心 + 10 回合均值）是唯一可信进度条。
6. **把单次崩塌当成"代码有 bug"**。如 §4.3 实测，600 回合崩塌在正确实现下照常发生。诊断流程：先多种子重跑（≥5 个）确认是系统性崩溃还是随机崩塌，再查 §6 前五条，最后才怀疑代码。

## 本篇小结

- Deadly Triad（自举+函数近似+off-policy）的正反馈回路使 DQN 不稳定是默认现象；本章实测：500 分后 100 回合内崩到 104。
- 经验回放：打破时间相关性 + 复用样本；buffer 里 `terminated` 与 `truncated` 必须分开记。
- 目标网络：冻结自举目标 $C$ 步，"追自己尾巴"变"朝固定靶回归"；偏差换稳定的又一例。
- DQN 全量实现要点：`gather` 取执行动作列、目标计算包 `no_grad`、Huber 损失、三处种子固定；实测 200 回合内从 9.5 升到 306。
- Double DQN（选动作用在线网络、估值用目标网络）把满分窗口延长 100 回合但不根除崩塌；Dueling 分解 $V+A$，Priority Replay 按 TD 误差优先采样。
- 连续动作是 DQN 家族的硬天花板——07 章策略梯度从另一条路解决它。

## 练习

1. 去掉目标网络（target 直接用 `q(ns_b).max(...)`，注意把目标也放进 `no_grad`），重跑 §4.2。对比前 300 回合的 eval 曲线——量化"没有固定靶子"对早期稳定性的影响。
2. 把 `float(term)` 改成 `float(term or trunc)` 重跑。观察训练后期是否出现"约 490 步送死"行为，并解释为什么评估分最终上不了 500。
3. 实现 Priority Replay 简化版：抽样概率 ∝ $(|\delta| + 10^{-4})$，重要性权重 $w \propto (N \cdot p)^{-\beta}$（$\beta$ 从 0.4 线性升到 1）。对比均匀回放，报告前 200 回合的 eval 差异。
4. 把网络换成 Dueling 结构（$V$ 头 + $A$ 头，$Q = V + A - A.\text{mean}()$），保持其余不变重跑，比较 300–500 回合区间的表现。
5. 思考题：DQN 经验回放是 off-policy 的根本原因（数据可以来自旧策略），但 PPO（09 章）不用回放。从"重要性采样比率在高维动作空间里方差爆炸"的角度，说明为什么 PPO 宁可扔数据也不用旧数据。（09 章给出正式推导，先建立直觉。）

## 参考来源

- DQN：Mnih et al., "Playing Atari with Deep Reinforcement Learning", NIPS Deep Learning Workshop 2013；"Human-level control through deep reinforcement learning", *Nature* 518, 2015
- Deadly Triad 与发散：Sutton & Barto, 2nd ed., §11.3；Baird, "Residual Algorithms: Reinforcement Learning with Function Approximation", ICML 1995（发散反例原始构造）
- Double DQN：van Hasselt, Guez & Silver, AAAI 2016；Dueling：Wang et al., ICML 2016；Priority Replay：Schaul et al., ICLR 2016；Rainbow：Hessel et al., AAAI 2018
- Huber 损失与梯度裁剪为 DQN 原论文的工程配置（经验做法部分已单独标注）
- 本章训练数值为 seed=0、CPU 实测真实结果，代码全量给出可复现；崩塌现象在多种子下普遍存在，时点随机
