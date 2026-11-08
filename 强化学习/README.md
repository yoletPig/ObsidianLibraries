# 强化学习底层：从 MDP 到 PPO，全部手写

> 面向 **Python + PyTorch 熟练、但没系统学过强化学习本体**的学习者：从"MDP 是什么"一路讲到"亲手写出 PPO、再连通 RLHF/DPO"，补上 RL 算法工程师的地基。
> 全系列为中文原创，代码全部 **CPU 可跑**（需要 GPU 的地方单独标注）。库内《VLM & VLA》的 Stage 6（DPO 偏好对齐）、Stage 7（RLVR 推理强化）是"应用层"——本系列补的就是它们脚下缺的那块 RL 本体地基。

## 学习路径

| 篇 | 主题 | 你将学会 |
| --- | --- | --- |
| [01-强化学习是什么与MDP](01-强化学习是什么与MDP.md) | MDP 建模 | 与监督学习的本质区别（数据分布由策略决定/奖励延迟）、交互回路、MDP 五元组、回报与折扣、episodic vs continuing、智能体分类谱系 |
| [02-多臂老虎机与探索](02-多臂老虎机与探索.md) | 探索与利用 | ε-greedy 与 regret、UCB1 推导、Thompson 采样、10 臂老虎机三方法对比实验、探索问题在推荐/AB 测试的映射 |
| [03-动态规划-已知环境的求解](03-动态规划-已知环境的求解.md) | 策略迭代/值迭代 | 贝尔曼期望方程推导、策略评估与迭代、值迭代、GridWorld 完整实现、广义策略迭代（GPI）思想、DP 为何现实少用 |
| [04-蒙特卡洛与时序差分](04-蒙特卡洛与时序差分.md) | 无模型值学习 | MC 首访/每次访问、增量式均值、TD(0)、偏差 vs 方差对比表、n-step 与资格迹、TD 误差、两种方法收敛实验 |
| [05-Q-learning与SARSA](05-Q-learning与SARSA.md) | 表格法控制 | off-policy vs on-policy 辨析、Q-learning 更新式与乐观偏差、SARSA、ε 衰减、CliffWalking 两算法路径可视化、表格法何时够用 |
| [06-DQN](06-DQN.md) | 值函数 + 深度网络 | Deadly Triad 为何不稳定、经验回放打破时间相关性、目标网络、DQN 完整 PyTorch 实现（CartPole）、Double/Dueling/Priority Replay 原理 |
| [07-策略梯度与REINFORCE](07-策略梯度与REINFORCE.md) | 直接学策略 | 为什么跳过值函数、策略梯度定理分步推导（score function 视角）、REINFORCE 实现（CartPole）、基线降方差推导与实验、与 DQN 的本质差异 |
| [08-Actor-Critic](08-Actor-Critic.md) | 价值 + 策略混合 | 偏差-方差权衡、A2C 架构与共享主干、GAE 推导、完整 A2C 实现（CartPole）、A3C 异步思想与为何被同步版本取代 |
| [09-PPO从零实现](09-PPO从零实现.md) | 信赖域策略优化 | TRPO 动机、代理目标与重要性采样、ratio 裁剪机制、PPO 完整实现（CartPole）、调参经验、为何 PPO 统治 RLHF |
| [10-RLHF-从奖励模型到PPO-DPO](10-RLHF-从奖励模型到PPO-DPO.md) | LLM 对齐 | RLHF 三阶段管线、KL 惩罚与 reward hacking、InstructGPT 要点、DPO 全推导（Bradley-Terry→DPO）、DPO vs PPO 对比、GRPO 与 RLVR |
| [11-现代RL前沿概览](11-现代RL前沿概览.md) | 前沿地图 | model-based（Dyna/MuZero）、离线 RL 与外推误差、奖励塑形（potential-based）、分层 RL、游戏之外的落地事实 |
| [12-实战-完整项目与环境搭建](12-实战-完整项目与环境搭建.md) | 综合工程 | 从零搭建 PPO 训练 Pipeline（分文件全量代码）、gymnasium 自定义环境规范、不收敛 10 大原因排查表、面试高频 25 题 |

### 阶段划分

1. **地基（01–02）**：MDP 数学框架与探索问题。学完你就知道"强化学习在优化什么"、以及为什么它和监督学习根本不同。
2. **表格法全貌（03–05）**：已知模型用 DP，未知模型用 MC/TD，控制用 Q-learning/SARSA。所有 RL 核心概念（回报、贝尔曼、TD 误差、on/off-policy）在这里建立，**后面深度 RL 只是把"表"换成"网络"**。
3. **深度 RL（06–08）**：DQN → REINFORCE → Actor-Critic。从两条路线（值函数法、策略梯度法）分别进入深度时代，在 GAE 处汇合。
4. **SOTA 与对齐（09–10）**：PPO 是策略梯度的工程完成态，也是 RLHF 的引擎；10 章把它接到 LLM 上，并推导 DPO/GRPO——直接连通库内 VLM Stage 6/7。
5. **视野与工程（11–12）**：前沿概览建立"地图"，综合项目把全部知识落成可复现的工程实践。

## 与本仓库其他方向的衔接

- **VLM & VLA Stage 6（Post-training 偏好对齐与 DPO 实战）**：Stage 6 讲 DPO 的训练实战与工程配置；DPO 的数学从哪来？它就是本系列 07 章策略梯度的"闭式化"——RL 目标 + KL 约束 + Bradley-Terry 消掉奖励模型，推出 $\pi_\theta$ 的交叉熵损失。10 章给出与 Stage 6 第 1 周完全对齐的完整推导链。
- **VLM & VLA Stage 7（RLVR 与 Multimodal Reasoning）**：Stage 7 讲 GRPO/verl/多模态 RLVR 实战；GRPO "去掉 Critic、组内相对优势"的设计，站在本系列 08（Critic 与基线）、09（PPO 的优势函数）肩膀上才看得懂。10 章末节给出 GRPO 思想与 RLVR 概念的底层视角。
- **数学基础 09（梯度与优化入门）**：策略梯度定理的推导（07 章）、KL 约束优化（09 章 TRPO）都是 09 章梯度与梯度下降的直接延伸；贝叶斯推断（07 章）则呼应 02 章 Thompson 采样。
- **PyTorch 与深度学习系列**：DQN/PPO 的实现全部建立在该系列 03 章（`nn.Module` 与训练循环）、09 章（训练稳定性）之上——RL 网络比监督网络小得多，但训练稳定性问题严重一个量级，正好是该系列 09 章知识的"高压练兵场"。

## 约定

- 代码环境：Python 3.10+，**gymnasium**（经典控制任务：CartPole、CliffWalking 等）、**NumPy**、**PyTorch 2.x**（CPU 版即可）；写作时验证于 gymnasium 1.4.0 / NumPy 1.26.4 / PyTorch 2.14。
- **所有示例默认 CPU 可跑**。仅 06 章 DQN 的 Atari 版本、09 章 PPO 的大批量场景建议 GPU，均已用注释 `# [GPU]` 标出并给出 CPU 降级参数；CartPole/CliffWalking 级别任务 CPU 数秒到数分钟内收敛。
- 每章关键实验给出**真实运行结果数值**（固定种子），你本地运行若版本不同数值会有微小差异，但量级与结论一致。
- 引用规范：关键论断给出 Sutton & Barto《Reinforcement Learning: An Introduction》(2nd ed.) 章节号或原论文（年份+会议）；无法查证的内容一律标注"经验做法"，不冒充定论。
- 随机种子：所有代码默认 `seed=0`（或显式传入），`env.reset(seed=...)`、`np.random.seed`、`torch.manual_seed` 三处都固定，保证可复现——RL 的方差远大于监督学习，不复现等于没法调试。
