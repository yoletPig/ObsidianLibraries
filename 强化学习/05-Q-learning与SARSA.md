# 05 - Q-learning 与 SARSA

## 本篇目标

读完本篇，你应该能做到：

1. **辨析 on-policy 与 off-policy**——RL 里最容易被说混的一对概念，用"数据是谁采的、更新的是谁的值"两问定位任何算法。
2. 从贝尔曼最优方程推导 Q-learning 更新式，并理解 `max` 算子引入的**乐观偏差**（maximization bias）——06 章 Double DQN 的动机在此预演。
3. 实现 Q-learning 与 SARSA，在 CliffWalking 上复现经典现象：**Q-learning 学出最短路线，SARSA 学出安全路线**，并可视化两条路径。
4. 判断表格法什么时候够用，以及从 Q-table 到 Q-network 的动机链条（通向 06 章）。

04 章解决了"预测"（给定策略学 $V^\pi$）；本章升级为"控制"（同时学值与改策略）。控制需要的是动作值 $Q(s,a)$ 而不是 $V(s)$——因为"选哪个动作"需要比较动作，而 $V(s)$ 已经把动作平均掉了。

## 1. On-policy vs Off-policy：先把概念钉死

两个问题，两问定位（Sutton & Barto, 2nd ed., §5.4–5.5、§6.4–6.5）：

- **问一：评估/改进的目标是谁？**——被学习的策略叫**目标策略（target policy）**。
- **问二：产生数据的是谁？**——与环境交互采数据的策略叫**行为策略（behavior policy）**。

| | 行为策略 = 目标策略 | 行为策略 ≠ 目标策略 |
| --- | --- | --- |
| 名称 | **on-policy** | **off-policy** |
| 典型算法 | SARSA、REINFORCE、A2C、PPO | Q-learning、DQN |
| 能否复用旧数据/其他来源数据 | 不能（分布必须匹配目标策略） | 能（经验回放、人类数据、专家数据） |
| 探索如何"计入"学习 | 探索动作的价值会被学进 Q（学到的是含探索的策略的值） | 探索只是采样手段，学的是贪心目标的值 |
| 安全性直觉 | 保守：学"我实际怎么走" | 激进：学"如果贪心会怎样" |

**为什么 Q-learning 能 off-policy？** 它的更新目标 $\max_{a'} Q(s',a')$ 只依赖目标策略（贪心），不依赖产生数据的行为策略——数学上它收敛到 $q_*$（贝尔曼最优方程的解）与数据来自哪个策略无关（只要覆盖所有 $(s,a)$ 且步长条件满足，Watkins & Dayan, 1992, *Machine Learning*）。**为什么 SARSA 必须 on-policy？** 它的更新目标 $Q(s',a')$ 里的 $a'$ 是行为策略**实际将要执行**的动作——换掉行为策略，学习目标就变了。

> **经验做法**：判断一个新算法，别背标签，就问两个问题——"更新公式里的 $a'$（或下一动作）是实际执行的那个吗？"（是 → on-policy 倾向）"更新公式里有没有 $\max$ / 期望算子把行为策略消掉？"（有 → off-policy 倾向）。09 章 PPO 属于 on-policy：它的 ratio 定义在"当前策略生成数据的分布"上，旧数据用几次就得扔。

## 2. Q-learning：向最优看齐的 off-policy 控制

### 2.1 更新式推导

目标：学 $q_*(s,a) = \max_\pi q^\pi(s,a)$，它满足贝尔曼最优方程的 $Q$ 版本（Sutton & Barto, 2nd ed., 式 6.8 前后）：

$$q_*(s,a) = \mathbb{E}\left[r + \gamma \max_{a'} q_*(s', a') \mid s, a\right]$$

把期望换成**采样一个** $(r, s')$（04 章"期望 → 样本"的又一次使用），右边不动点变成更新目标：

$$\boxed{Q(s,a) \leftarrow Q(s,a) + \alpha\left[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,\right]}$$

对比 SARSA 的唯一差别：目标里的 $\max_{a'} Q(s',a')$ 换成 $Q(s', a'_{\text{实际}})$。Q-learning 里的 $\max$ 就是 03 章"改进线"的极致压缩——**评估与改进在同一个更新里**，连显式策略都不用存。

### 2.2 `max` 的乐观偏差：一个可复现的小实验

$\max$ 作用在**带噪声的估计**上必然产生向上偏差：$E[\max(X_1,\dots,X_n)] \ge \max E[X_i]$（Jensen 型论证；在 RL 语境下即 maximization bias，Sutton & Barto, 2nd ed., §6.7，也是 Double Q-learning 的动机）。

```python
import numpy as np
rng = np.random.default_rng(0)
# 两台老虎机真实价值都是 0；agent 每臂采 n 个样本后取 max 当"最优值"的估计
for n in [1, 5, 20]:
    est = rng.normal(0, 1, size=(200000, 2, n)).mean(axis=2)  # 每臂 n 样本的均值
    print(f"每臂{n:2d}个样本: E[max(Q估)] = {est.max(axis=1).mean():.4f}   (真值 = 0)")
```

运行结果：

```text
每臂 1个样本: E[max(Q估)] = 0.5665   (真值 = 0)
每臂 5个样本: E[max(Q估)] = 0.2529   (真值 = 0)
每臂20个样本: E[max(Q估)] = 0.1256   (真值 = 0)
```

真值是 0，但"在两个噪声估计里挑最大"的期望从 +0.57 一路虚高。映射到 Q-learning：早期样本少、$Q$ 噪声大，$\max_{a'} Q(s',a')$ 系统性高估，agent 会"高估未充分探索的动作"。表格法 + 收敛后偏差会消散（样本多了）；**函数近似 + 持续自举时偏差不消散**——这正是 06 章 DQN 高估问题与 Double DQN 的伏笔。

## 3. SARSA：诚实的 on-policy 控制

SARSA 名字就是更新五元组 $(S, A, R, S', A')$ 的缩写（Sutton & Barto, 2nd ed., §6.4）：

$$Q(s,a) \leftarrow Q(s,a) + \alpha\left[\,r + \gamma Q(s',a') - Q(s,a)\,\right], \quad a' \sim \pi(\cdot|s')$$

注意实现顺序：**必须先算 $a'$（在 $s'$ 状态下按行为策略采样），才能完成这次更新**——数据流是 $s,a,r,s',a'$ 依次产生。这决定了 SARSA 学到的是"含 $\epsilon$ 探索在内的实际策略"的值：如果 $\epsilon$-greedy 会概率性掉悬崖，SARSA 的 $Q$ 里就写着掉悬崖的代价。

## 4. CliffWalking 实验：一条悬崖，两种性格

### 4.1 环境与两个算法

CliffWalking（Sutton & Barto, 2nd ed., 例 6.6；gymnasium `CliffWalking-v1`）：4×12 网格，左下出发、右下终点，底部一排是悬崖。任意步奖励 $-1$，**掉进悬崖奖励 $-100$ 并被送回起点**（注意：gymnasium 实现里掉悬崖不终止回合，是"传送回起点"；只有到达终点才 `terminated=True`）。环境无默认步数上限，训练代码用 `TimeLimit` 包一层防随机游走的超长回合。

```python
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

def eps_greedy(Q, s, eps, rng, nA):
    """ε-greedy: eps 概率均匀探索, 否则贪心（平局时 argmax 取先到的）"""
    if rng.random() < eps:
        return int(rng.integers(nA))
    return int(np.argmax(Q[s]))

def q_learning(env, n_episodes=500, alpha=0.1, gamma=0.99,
               eps_init=1.0, eps_min=0.05, decay=0.995, seed=0):
    rng = np.random.default_rng(seed)
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    eps, returns = eps_init, []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        eps = max(eps_min, eps * decay)          # ε 线性于回合数指数衰减
        total, done = 0.0, False
        while not done:
            a = eps_greedy(Q, obs, eps, rng, nA)
            ns, r, term, trunc, _ = env.step(a)
            target = r + gamma * (0.0 if term else np.max(Q[ns]))  # off-policy: max
            Q[obs, a] += alpha * (target - Q[obs, a])
            obs, total, done = ns, total + r, term or trunc
        returns.append(total)
    return Q, np.array(returns)

def sarsa(env, n_episodes=500, alpha=0.1, gamma=0.99,
          eps_init=1.0, eps_min=0.05, decay=0.995, seed=0):
    rng = np.random.default_rng(seed)
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    eps, returns = eps_init, []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        eps = max(eps_min, eps * decay)
        a = eps_greedy(Q, obs, eps, rng, nA)     # 先采 a
        total, done = 0.0, False
        while not done:
            ns, r, term, trunc, _ = env.step(a)
            na = eps_greedy(Q, ns, eps, rng, nA) # 先采 a' 再更新（SARSA 的顺序）
            target = r + gamma * (0.0 if term else Q[ns, na])      # on-policy: 实际 a'
            Q[obs, a] += alpha * (target - Q[obs, a])
            obs, a, total, done = ns, na, total + r, term or trunc
        returns.append(total)
    return Q, np.array(returns)

env = TimeLimit(gym.make("CliffWalking-v1"), max_episode_steps=500)
Q_ql, r_ql = q_learning(env, seed=0)
Q_sa, r_sa = sarsa(env, seed=0)
```

### 4.2 真实运行结果

训练过程回报（`seed=0`，TimeLimit 500 步）：

| 阶段（回合窗口均值） | Q-learning | SARSA |
| --- | --- | --- |
| 前 10 回合 | $-4283.6$ | $-4085.3$ |
| 100–200 回合 | $-365.5$ | $-80.4$ |
| 最后 100 回合 | $-66.1$ | $-21.0$ |

20 个独立种子平均（最后 100 回合）：Q-learning $-52.1$、SARSA $-22.9$。**训练期间的"在线表现" SARSA 明显更好**——尽管接下来会看到 Q-learning 的贪心策略其实更短。这就是 Sutton & Barto 图 6.4 的经典对比。

提取贪心路径（训练后 $\epsilon=0$ 跑一遍）：

```python
def greedy_path(Q, seed=0):
    env = TimeLimit(gym.make("CliffWalking-v1"), max_episode_steps=100)
    obs, _ = env.reset(seed=seed)
    path = [obs]
    total = 0.0
    for _ in range(100):
        a = int(np.argmax(Q[obs]))           # 纯贪心
        obs, r, term, trunc, _ = env.step(a)
        path.append(obs); total += r
        if term or trunc: break
    return path, total

path_ql, ret_ql = greedy_path(Q_ql)
path_sa, ret_sa = greedy_path(Q_sa)
print("Q-learning 贪心路径回报:", ret_ql, "步数:", len(path_ql) - 1)
print("SARSA       贪心路径回报:", ret_sa, "步数:", len(path_sa) - 1)
```

运行结果：

```text
Q-learning 贪心路径回报: -13.0 步数: 13
SARSA       贪心路径回报: -17.0 步数: 17
```

路径可视化（`*` 为轨迹，`C` 为悬崖，`G` 为终点）：

```text
Q-learning 路径（贴着悬崖走，13 步，最优最短）:    SARSA 路径（远离悬崖，17 步，安全）:
. . . . . . . . . . . .                        * * * * * * * * * * * *
. . . . . . . . . . . .                        * . . . . . . . . . . *
* * * * * * * * * * * *                        * . . . . . . . . . . *
* C C C C C C C C C C *                        * C C C C C C C C C C *
```

**为什么"更优"的 Q-learning 在线表现反而差？** 两层解释：

1. Q-learning 收敛目标是 $q_*$（无探索策略的最优），学出贴悬崖的最短路；SARSA 收敛目标是 $\epsilon$-greedy 策略的值，悬崖边上的状态因为"有 $\epsilon$ 概率掉下去"而值被拉低，于是 $Q$ 自动偏好离悬崖远的格子。
2. 训练期间两者都带 $\epsilon$ 探索：贴悬崖路线 + 探索 = 频繁掉悬崖（$-100$，观察 Q-learning 中期 $-365$）；安全路线 + 探索 = 偶尔多走几步（$-1$）。SARSA 把探索的代价计入了学习，Q-learning 没有。

哪个更好取决于问题：**如果 $\epsilon$ 最终会归零、评估用纯贪心**，Q-learning 更优；**如果部署时永远带探索（或环境有不可逆的昂贵失误）**，SARSA 的保守是美德。机器人、交易、医疗这类"摔一跤就出局"的领域，安全的 on-policy 学法常常是更合理的选择。

## 5. ε 的衰减与探索的退场

本章代码用 $\epsilon_t = \max(\epsilon_{\min},\, \epsilon_0 \cdot d^t)$，$\epsilon_0=1.0$、$d=0.995$、$\epsilon_{\min}=0.05$。三个设计点的理由：

- **初期高 $\epsilon$**（前 100 回合从 1.0 降到 0.61）：$Q$ 全是噪声时贪心没有意义，先广撒网；
- **指数衰减而非线性**：中期每回合衰减的绝对量小，给"刚发现的捷径"留足采样次数；
- **$\epsilon_{\min}=0.05$ 不归零**：非平稳环境（对手在变、用户在变）需要持续收集新鲜数据；CLiffWalking 是平稳任务，可以设为 0。

> **经验做法**：ε 衰减节奏与学习率耦合——ε 降太快而 $Q$ 还没学准，贪心被早期噪声锁死（02 章 ε-greedy 线性 regret 的时序版）；ε 降太慢，前期浪费交互。判断标准看回报曲线：如果中期出现"平台→突然跳升"，通常是 ε 终于降到 $Q$ 噪声水平以下的标志。

## 6. 表格法什么时候够用 & 到神经网络的动机

表格 $Q \in \mathbb{R}^{|S| \times |A|}$ 的容量账（经验做法，按 float32 计）：

| 任务 | $\|S\| \times \|A\|$ | Q 表内存 |
| --- | --- | --- |
| CliffWalking | $48 \times 4$ | 768 B |
| CartPole | 离散化 100²×100²×2≈$10^8$ | 400 MB（还没算好） |
| Atari（84×84×4 帧） | $\approx 10^{6130}$ | 宇宙原子不够 |

三条判据，全满足才用表格法：**(1) 状态可枚举且数量小**（$10^4$–$10^5$ 以内）；**(2) 无需泛化**——每个状态都会被访问足够多次；**(3) 状态表示本身稳定**（不需要从像素学特征）。

CartPole 的状态是 4 维连续量，离散化要么丢精度要么爆炸——**核心矛盾是"泛化"**：相似的状态应该共享经验，表格法做不到。解法是把 $Q(s,a)$ 换成参数化函数 $Q(s, a; \theta)$，用神经网络当近似器（监督学习那套全都能用上）：

$$Q(s,a;\theta) \approx q_*(s,a)$$

但这引入了一个新的不稳定源：$\theta$ 的更新改变所有状态的 $Q$ 估计（一个状态的梯度会牵动别的状态），加上 04 章预警的 Deadly Triad（自举 + 函数近似 + off-policy），训练崩坏成为默认现象——06 章 DQN 的经验回放与目标网络就是为驯服它而生的两个机制。

## 7. 常见误区

1. **SARSA 忘了"先采 $a'$ 再更新"。** 更新目标里的 $a'$ 必须是行为策略在 $s'$ 上**实际会执行**的动作。写成 `np.argmax(Q[ns])` 就变成了期望 SARSA（Expected SARSA，另一个合法算法）；写成 `Q[ns, a]`（旧动作）则完全是 bug。症状：SARSA 表现出"贴悬崖"行为，与理论预期相反。
2. **Q-learning 的 target 忘记终止处理。** 终点状态的 `np.max(Q[ns])` 必须替换为 0（`0.0 if term else ...`）。CliffWalking 的症状：终点附近的 $Q$ 值被自举污染成负数，agent 到达终点后"不敢结束"，在终点旁打转。
3. **把"on-policy 更安全"绝对化。** SARSA 安全的前提是**行为策略就是部署策略**。如果训练用 $\epsilon=0.5$ 部署用 $\epsilon=0$，SARSA 学到的"安全"是基于错误行为分布的——这叫策略失配（policy mismatch），此时正确做法反而是 off-policy 方法。
4. **ε 衰减与学习率同时调，把锅甩给算法。** 两者都控制"新数据 vs 旧估计"的权重，一起乱调时训练曲线完全不可解读。调参纪律（经验做法）：一次只动一个，且固定随机种子对比——本章所有数字都建立在 `seed=0` 可复现之上。
5. **用训练期回报评估 Q-learning 的策略质量。** Q-learning 训练期回报含探索噪声（本章：$-66$），贪心回报才是策略本身的分数（本章：$-13$）。报告哪个数字，取决于你关心的部署场景；混淆两者是论文审稿和面试里的高频错误。

## 本篇小结

- on/off-policy 用两问定位：数据谁采的（行为策略）、更新谁（目标策略）。Q-learning off-policy（max 消掉行为策略，可复用旧数据），SARSA on-policy（目标含实际下一动作，学到含探索的策略的值）。
- Q-learning 从贝尔曼最优方程"期望→样本"而来；`max` 作用在噪声估计上产生乐观偏差（实测：每臂 1 样本时 $E[\max]=0.57$ vs 真值 0），是 06 章 Double DQN 的种子。
- CliffWalking 实验（真实数值）：贪心策略 Q-learning $-13$ 步 13（最短、贴悬崖），SARSA $-17$ 步 17（安全、离悬崖）；在线表现 SARSA 全面更好（$-21$ vs $-66$）——"学得优"与"走得稳"是两个目标。
- ε 衰减三件套：高起步、指数衰减、$\epsilon_{\min}$ 兜底；衰减节奏要匹配 $Q$ 的学习进度。
- 表格法三判据：可枚举、能覆盖、表示稳定；CartPole 起就需要函数近似与泛化，Deadly Triad 随之登场——06 章正题。

## 练习

1. 实现期望 SARSA：目标用 $\sum_{a'} \pi(a'|s') Q(s',a')$（$\pi$ 为当前 ε-greedy 分布）。在 CliffWalking 上与 Q-learning/SARSA 对比在线回报，解释为什么它介于两者之间且对 ε 不那么敏感。
2. 把 ε 固定在 0.1（不衰减）重跑 §4，20 种子平均比较最后 100 回合。观察 Q-learning 在线回报是否显著变差——用 §4.2 的两层解释说明原因。
3. 实现 Double Q-learning（Sutton & Barto §6.7 的表格式，维护 $Q_A, Q_B$ 两张表，随机选一张更新、用另一张评估 $a'$），在同一实验里测量它是否缓解了悬崖边的过度悲观/乐观。
4. 把奖励改为"掉悬崖 $-1$"（悬崖不再可怕），重跑两个算法。预期：两条路径重合。用这个对照实验说明"SARSA 保守是奖励结构的函数，不是算法性格"。
5. 思考题：DQN 用经验回放（off-policy 数据）训练，为什么它的探索仍依赖 ε-greedy 而不能像 PPO 那样靠策略熵？提示：行为策略与目标策略的关系、以及 §1 表格第三行的"能否复用旧数据"。

## 参考来源

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), §6.4（Q-learning）、§6.5（SARSA 及悬崖例）、例 6.6 与图 6.4（CliffWalking 对比）、§6.7（maximization bias 与 Double Q-learning）
- Q-learning 收敛：Watkins & Dayan, "Q-learning", *Machine Learning* 8, 1992
- 期望 SARSA：Van Seijen et al., "True Online TD(λ)" 之前身——原始出处为 Sutton & Barto (1st ed.) §6.6 的讨论及 Van Seijen et al. 2009
- gymnasium CliffWalking-v1 掉悬崖行为（$-100$ 回起点、不终止）：gymnasium 官方源码 `gymnasium/envs/toy_text/cliffwalking.py`
- 本章数值为种子固定（seed=0；多种子处标注）的真实运行结果
