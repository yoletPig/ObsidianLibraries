# 09 - PPO 从零实现

## 本篇目标

读完本篇，你应该能做到：

1. 从"一步更新太大就崩"推出 **TRPO 的动机与约束优化形式**，再看懂 PPO 如何用一行 `clamp` 把 TRPO 的二阶问题变回一阶。
2. 分步推导**代理目标**与**重要性采样视角**，理解 ratio $\frac{\pi_\theta}{\pi_{old}}$ 为什么是"策略改动量的测量仪"。
3. 实现完整 PPO（CartPole，10 项标准超参），跑出真实收敛曲线，并用消融实验实测**去掉 clip 后 KL 飙到 3.2、训练震荡**——clip 的稳定性作用有数据背书。
4. 掌握 PPO 调参的"先看什么再动什么"，以及它统治 RLHF 的工程逻辑（10 章的入口）。

08 章 A2C 的遗留问题：策略梯度是**一阶方法但没有步长控制**。学习率稍大，一次更新把策略推离采样数据的分布太远 → 新策略在旧数据上"过度解读" → 策略崩坏（07 章高方差的另一面）。PPO（Proximal Policy Optimization，Schulman et al., 2017, arXiv:1707.06347）用最朴素的手段解决它，并成为至今 most-widely-used 的 RL 算法。

## 1. TRPO 的动机：一步更新太大就崩

考虑一个具体失败：当前策略 $\pi_{old}$ 采样了 1000 步数据，其中动作 $a$ 平均回报不错。你做了一次"过猛"的策略更新，$\pi(a|s)$ 从 0.3 拉到 0.95——但这次更新的依据只有旧策略访问过的那些状态。在**旧数据没覆盖的状态**上，新策略的行为完全是外推，可能一塌糊涂。RL 里没有"验证集报警"：崩了就崩了，接下来只能从崩坏策略采的崩坏数据里慢慢爬（01 章"数据分布由策略决定"的惩罚）。

**TRPO**（Trust Region Policy Optimization，Schulman et al., ICML 2015）的形式化解法：把策略改进写成**约束优化**——在"新旧策略不要离得太远"的约束下最大化改进量：

$$\max_\theta \; L_{\pi_{old}}(\theta) \quad \text{s.t.} \quad \bar{D}_{KL}(\pi_{old}, \pi_\theta) \le \delta$$

$L_{\pi_{old}}(\theta)$ 是**代理目标**（下一节），$\bar D_{KL}$ 是新旧策略的平均 KL 散度（"信任域"半径）。名字的含义就在这：只信任当前数据所支撑的一小片策略空间（trust region），出了域的改进不予考虑。TRPO 用二阶方法（KL 的 Fisher 信息矩阵近似 + 共轭梯度）求解——**精确但昂贵且难实现**，PPO 的全部动机就是把这个二阶问题拍回一阶。

## 2. 代理目标：从策略梯度定理出发

### 2.1 重要性采样视角

策略梯度定理（07 章）的期望是**按当前策略 $\pi_\theta$ 采样**定义的——每更新一步 $\theta$，旧数据的分布就不匹配了，数学上不能再直接用。**重要性采样**（importance sampling）允许"用旧分布的数据估计新分布的期望"：

$$\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right]$$

把 $p$ 换成新策略、$q$ 换成旧策略，轨迹级期望的修正因子是新旧概率之比的连乘——这就是 **ratio**。TRPO 论文用 Kaplan 近似把它化简到单步（per-state）形式：

$$L_{\pi_{old}}(\theta) = \mathbb{E}_{(s,a) \sim \pi_{old}}\left[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} A^{\pi_{old}}(s,a)\right]$$

两个好消息：ratio 只需要旧数据里存下 $\log \pi_{old}(a|s)$（一个标量/样本），不用重算轨迹概率；$A^{\pi_{old}}$ 用 08 章的 GAE 估计。

### 2.2 读出三个性质

- $\theta = \theta_{old}$ 时 ratio = 1，$L$ 就是当前策略的优势均值——起点处的梯度与普通策略梯度一致（可以验证：对 $\theta$ 求导在 ratio=1 处退化为 $\nabla \log \pi \cdot A$）；
- ratio > 1 表示"新策略更爱这个动作"，$L$ 按优势正负决定该不该继续推高它；
- **ratio 就是"策略改动量的测量仪"**——它偏离 1 多远，更新就偏离采集数据的策略多远。PPO 的一切机制都围绕"看着这个仪表盘开车"。

## 3. Ratio 裁剪：一行的信任域

### 3.1 从约束到 clamp

PPO 的观察：TRPO 想要的只是"别让 ratio 跑太远"，那何必二阶优化？直接把目标**裁剪**：

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t,\; \text{clip}\big(r_t(\theta), 1-\epsilon, 1+\epsilon\big)\hat{A}_t\right)\right], \qquad r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$$

$\epsilon = 0.2$。逐情形拆开（配图，ratio 横轴、目标值纵轴）：

```text
情况一: A > 0 (好动作, 想推高概率)

  目标值
   │            min 取哪支?
   │      ╭───────────  r·A        ← 未裁剪支 (r > 1+ε 后无效)
   │     ╱
   │    ╱ (1+ε)·A ←─ 裁剪支在此封顶
   │   ╱
   └──┴────┴──────────→ ratio
      1  1+ε
  裁剪效果: r 超过 1+ε 后目标不再增长 → 梯度为 0 → 停止推高
  （好动作推到"比旧策略多 20%"就收手）

情况二: A < 0 (坏动作, 想压低概率 → r 变小, 取 min 时选未裁剪支? 注意细节)

   │ r·A      ╲                    (A<0, r 越大目标越负)
   │           ╲
   │  (1-ε)·A──╲─────────         ← 裁剪支在 r < 1-ε 后封底
   │            ╲
   └──┴─────────┴──────→ ratio
     1-ε   1
  r 低于 1-ε 后目标被 (1-ε)·A 封住 → 继续压低也无梯度 → 停止
```

**为什么取 min 还能"单向放行"？** 关键的不对称：裁剪只挡"进一步优化"的方向，不挡"退回"的方向——若 $\hat A > 0$ 且 $r$ 已经超过 $1+\epsilon$，min 取裁剪支（梯度 0，不再推高）；但若此时做一次让 $r$ 回落的更新，min 会重新选到未裁剪支，梯度恢复。**bad case 被锁定，correction 仍被允许**——这是它比"硬约束"聪明的地方，也是论文标题 Proximal（"邻近的"）的含义：优化被限制在数据邻域内。

### 3.2 实测：clip 的稳定性作用（消融实验）

同一实现（§5，lr=7e-4，10 epochs），只把 $\epsilon$ 从 0.2 改成 $10^9$（等效无裁剪），400 回合对比：

| 指标 | clip=0.2 | 无 clip（$\epsilon=10^9$） |
| --- | --- | --- |
| ep 100 训练回报（50 窗口） | 466.1 | 108.8 |
| ep 350 训练回报 | **500.0** | 485.4 |
| ep 400 训练回报 | 500.0 | **423.4（回落）** |
| approx_kl 全程范围 | 0.0002–0.022 | **0.012–3.20**（震荡） |
| 症状 | 稳定爬升 | 曲线反复、KL 数次爆炸（最大 3.2） |

无 clip 时 KL 飙到 3.2（健康值约 0.01 量级）：第 2–10 个 epoch 的更新不断把策略推离采集数据的策略，后面 epoch 的"梯度"在优化一个已经失真的代理目标——**没有信任域的 on-policy 复用数据，等于变相 off-policy 还不做任何修正**。这就是"为什么 clip 稳定"的完整答案：它在**数据的有效性范围内**截断优化。

## 4. 与 A2C 的差别清单（先给结论）

| 部件 | A2C（08 章） | PPO |
| --- | --- | --- |
| 数据用量 | 每步采集后立即更新一次，用完即弃 | 一批数据（rollout）反复用 3–10 epoch |
| 步长控制 | 无（lr 即一切） | clip + early stop（approx_kl 监控） |
| 优势估计 | GAE（同） | GAE（同） |
| 网络 | Actor-Critic 共享主干（同） | 同 |
| 熵 | 熵奖励（同） | 同 |
| off-policy 程度 | 纯 on-policy | "半 on-policy"：数据来自 $\pi_{old}$，重用期间做 ratio 修正 |

PPO = A2C + **重用数据的机制 + 保险丝**。理解了 08 章，PPO 没有新数学。

## 5. 完整实现（CartPole）

### 5.1 超参表

| 超参 | 值 | 说明 |
| --- | --- | --- |
| 网络 | trunk 4→128 ReLU；π 128→2；V 128→1 | 与 08 章相同 |
| 优化器 | Adam, lr=7e-4 | CartPole 上经验值（经验做法） |
| rollout | 10 个回合（约 500–2500 步） | 教学版按回合；工程版固定 2048 步 |
| $\gamma$ / $\lambda$ | 0.99 / 0.95 | GAE 标准配置 |
| clip $\epsilon$ | 0.2 | 论文默认 |
| epochs | 10 | 每批数据过 10 遍 |
| minibatch | 256 | CPU 友好 |
| 价值损失 | Huber, $c_v=0.5$ | 08 章教训直接继承 |
| 熵系数 $c_e$ | 0.01 | |
| 梯度裁剪 | max_norm=0.5 | |
| 终止处理 | `done_flags` 只记 terminated；rollout 末尾截断时 bootstrap $V(s_T)$ | 01/05/06/08 章同一主题第 5 次 |

### 5.2 全量代码

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
opt = torch.optim.Adam(ac.parameters(), lr=7e-4)
gamma, lam, clip, epochs, minibatch = 0.99, 0.95, 0.2, 10, 256
vf_coef, ent_coef = 0.5, 0.01

def evaluate():
    ev = gym.make("CartPole-v1"); total = 0.0
    for ep in range(10):
        obs, _ = ev.reset(seed=10000 + ep); done = False
        while not done:
            with torch.no_grad():
                dist, _ = ac(torch.tensor(obs, dtype=torch.float32))
                a = int(torch.argmax(dist.logits))
            obs, r, term, trunc, _ = ev.step(a)
            total += r; done = term or trunc
    return total / 10

tr, kls_all, cf_all = [], [], []
for ep in range(400):
    # ---- ① 采集: 10 个回合的 rollout, 记录 logp_old ----
    S, A, logp_old, R, done_flags = [], [], [], [], []
    eps_returns, last_obs, last_term = [], None, True
    for ep_i in range(10):
        obs, _ = env.reset(seed=100000 + ep * 100 + ep_i)
        done, ep_ret = False, 0.0
        while not done:
            with torch.no_grad():                       # 采集期不建图
                dist, v = ac(torch.tensor(obs, dtype=torch.float32))
                a = dist.sample()
                la = dist.log_prob(a).item()            # 关键存档: 旧策略对数概率
            ns, r, term, trunc, _ = env.step(int(a))
            S.append(obs); A.append(a); logp_old.append(la)
            R.append(r); done_flags.append(float(term)) # truncation 不算终止
            ep_ret += r
            obs = ns; done = term or trunc
            last_obs, last_term = obs, term
        eps_returns.append(ep_ret)
    # ---- ② GAE(与 08 章完全同款) ----
    with torch.no_grad():
        vals = np.array([ac(torch.tensor(st, dtype=torch.float32))[1].item() for st in S])
        next_v = 0.0 if last_term else ac(torch.tensor(last_obs, dtype=torch.float32))[1].item()
    T = len(R)
    adv = np.zeros(T); gae = 0.0
    for t in reversed(range(T)):
        vn = vals[t + 1] if t + 1 < T else next_v
        gae = (R[t] + gamma * vn * (1 - done_flags[t]) - vals[t]) \
              + gamma * lam * gae * (1 - done_flags[t])   # done 处截断递推链
        adv[t] = gae
    ret = adv + vals                                     # λ-return
    adv_t = torch.tensor(adv, dtype=torch.float32)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t = torch.tensor(ret, dtype=torch.float32)
    S_t   = torch.tensor(np.array(S), dtype=torch.float32)
    A_t   = torch.stack(A)
    logp_old_t = torch.tensor(logp_old, dtype=torch.float32)
    # ---- ③ 多 epoch 小批量更新: ratio + clip ----
    idx = np.arange(T)
    for epoch in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, T, minibatch):
            mb = idx[start:start + minibatch]
            dist, v = ac(S_t[mb])
            logp = dist.log_prob(A_t[mb])
            ratio = (logp - logp_old_t[mb]).exp()        # π_new/π_old, 核心
            surr1 = ratio * adv_t[mb]
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[mb]
            pg_loss = -torch.min(surr1, surr2).mean()    # 裁剪代理目标
            v_loss = nn.functional.smooth_l1_loss(v.squeeze(-1), ret_t[mb])
            loss = pg_loss + vf_coef * v_loss - ent_coef * dist.entropy().mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
            opt.step()
            with torch.no_grad():                        # 监控仪表盘
                kls_all.append((-logp + logp_old_t[mb]).mean().item())
                cf_all.append(((ratio - 1).abs() > clip).float().mean().item())
    tr.append(np.mean(eps_returns))
    if (ep + 1) % 50 == 0:
        print(f"ep {ep+1:4d} | train50={np.mean(tr[-50:]):6.1f} | "
              f"kl={np.mean(kls_all[-100:]):.4f} | clipfrac={np.mean(cf_all[-100:]):.2f}")
```

### 5.3 真实运行结果

seed=0、CPU 实测：

```text
ep   50 | train50= 164.9 | kl=0.0037 | clipfrac=0.03
ep  100 | train50= 466.1 | kl=0.0077 | clipfrac=0.12
ep  150 | train50= 484.1 | kl=0.0221 | clipfrac=0.38   ← KL 尖峰, 依靠 clip 自动收敛
ep  200 | train50= 458.5 | kl=0.0059 | clipfrac=0.05
ep  250 | train50= 474.9 | kl=0.0071 | clipfrac=0.06
ep  300 | train50= 495.5 | kl=0.0024 | clipfrac=0.01
ep  350 | train50= 500.0 | kl=0.0029 | clipfrac=0.03   ← 满分并保持
ep  400 | train50= 500.0 | kl=0.0030 | clipfrac=0.03
```

读数三则：

1. **ep150 的 KL 尖峰是活教材**：clipfrac 冲到 0.38（38% 样本的 ratio 出界被裁剪）——裁剪机制自动"踩刹车"，KL 下一窗口回落到 0.006。训练日志里看到这种尖峰不用慌，**它是机制在工作，不是失控**；持续 >0.05 才需要干预。
2. 对照组（lr=2.5e-4，其余同）：400 回合 train50=488.1——太保守，学得慢。lr 与 KL 监控的联动见 §6。
3. 对比 08 章 A2C（ep700 才满分）：样本步数上 PPO（~150 万步）没有优势，但**全程没有任何一次崩塌**（对照 06 章 DQN 的 ep600 崩塌）——稳定性正是 PPO 的卖点。

## 6. PPO 调参经验（经验做法）

以下为社区长期共识 + 本文实验支撑，逐条标注依据：

1. **lr 与 epochs 的乘积感**：每批数据的有效学习量 ≈ lr × epochs × 过批次数。CartPole 上 lr=7e-4 × 10 epochs 快而稳（§5.3）；lr=2.5e-4 × 10 保守（488@400）。**先固定 epochs=10 动 lr，再看 KL 响应**。
2. **clip $\epsilon$ 几乎不用动**：0.2 是万金油（论文消融在 0.1–0.3 间差异不大）。需要更"信任域"的场景（大模型 RLHF）常见 0.1–0.2；更激进的机器人任务 0.3。
3. **监控 approx_kl 与 clipfrac，而不是盯着 loss**：健康区间（经验做法）kl ∈ [0.003, 0.03]、clipfrac < 0.3。kl 持续 > 0.05 → lr 太大或 epochs 太多；clipfrac 长期 ≈ 0 → 学得太保守（可加 lr/epochs）。PPO 论文和 Spinning Up 都以此为核心诊断量。
4. **minibatch 内的 advantage 归一化**：全局（每批 rollout）归一化一次（本实现）；不要在 minibatch 内各自归一（样本少、噪声被放大——08 章同款教训）。
5. **价值损失系数**：共享主干 → Huber + $c_v=0.5$（08 章 §5 事故的直接继承）；PPO 论文原版用 MSE + $c_v=1.0$，在大任务上配合 return 归一化使用。
6. **每批数据 epochs 上限由 KL early-stop 兜底**：工程实现常加"若 approx_kl > 1.5×目标值则提前终止本批 epoch 循环"（Spinning Up 的实现包含此逻辑）——教学版省略，生产版必备。

## 7. 为什么 PPO 统治了 RLHF（10 章的引子）

四个原因，全部是工程性质而非理论性质：

1. **兼容现有基建**：PPO = 监督训练循环 + 采样 + 两个网络， transformer 生态（PyTorch/HF）上改造成本最低；TRPO 的二阶求解器没有生产级实现。
2. **超参稳定域宽**：clip 的"自动刹车"让 lr/epochs 的错误在一定程度上自我修正（§5.3 的 KL 尖峰自愈就是证据）——这对**奖励信号噪声极大的 LLM**（奖励模型打分天然带噪）是刚需。
3. **天然支持 KL 正则**：RLHF 需要约束策略别离 SFT 参考模型太远（10 章主题），PPO 的 per-token ratio + KL penalty 机制正好承载——InstructGPT（Ouyang et al., 2022）的 RL 阶段就是 PPO。
4. **分布式友好**：采样（生成）与更新解耦，rollout 可以在推理集群上跑——10 章会看到 Stage 7 的 verl 把这个解耦推到极致（HybridEngine）。

代价也要说清：LLM 上 PPO 需要同时驻留 4 个模型（actor、ref、reward、critic），显存与工程复杂度极高——这正是 DPO（10 章）"用推导换工程"的出发点。

## 8. 常见误区

1. **ratio 忘了 `exp()` 或把 log 直接相除**。`ratio = logp / logp_old`（错）与 `ratio = (logp - logp_old).exp()`（对，利用对数恒等式数值更稳）。错误实现的 ratio 量级完全失真，clip 机制形同虚设。
2. **`logp_old` 存档时不是 `no_grad` 或存成了带图的张量**。采集时 `dist.log_prob(a)` 建图后 `.item()` 存 float 是对的；存张量本身会抱着整张图，第二批 epoch 更新时报"backward twice"。另一个常见错误：用**当前网络**重算 logp 当 old——那样 ratio 恒 1，clip 永不触发，训练退化为普通 A2C。
3. **minibatch 梯度不裁剪 / 裁剪阈值忘改**。PPO 单步更新间隔大（rollout 级），梯度尖峰比 A2C 更猛，max_norm=0.5 的保险不能省。
4. **把 clipfrac 当失败信号**。§5.3 显示 clipfrac=0.38 的尖峰后训练照常到满分——它和 KL 一起看才有意义（clipfrac 高 + KL 回落 = 机制在工作；两者都高且持续 = lr 过大）。
5. **rollout 太小 + epochs 太多**。批数据 500 步配 10 epochs，相当于同一小撮数据反复读十遍——过拟合单批数据，KL 飙升。**经验做法**：每批 rollout 至少 ≥ 1000 步（工程默认 2048），epochs 与 batch 大小反比调节。
6. **价值头回归目标是 GAE 优势而不是 return**。critic 应拟合 $\hat R_t = \hat A_t + V(s_t)$（λ-return），直接拿 $\hat A_t$ 当回归目标会把 V 学成优势（尺度完全错），策略损失的基线随之失效。

## 本篇小结

- TRPO：约束优化实现"稳步改进"；PPO 用 `min(r·A, clip(r,1±ε)·A)` 一行实现同等意图——裁剪挡住"继续推"，不挡"退回修正"。
- ratio 是策略改动量的测量仪；重要性采样视角解释了为什么 on-policy 数据能重用 10 个 epoch。
- 消融实测：无 clip 时 KL 飙到 3.2、曲线震荡（ep400 回落 423）；有 clip 时 KL 尖峰自愈、ep350 满分——clip 的稳定性有数据背书。
- 调参核心仪表盘是 approx_kl 与 clipfrac，健康区间 [0.003, 0.03] / < 0.3（经验做法）；loss 本身几乎不提供信息。
- PPO 统治 RLHF 靠的是工程性质：基建兼容、超参稳定域宽、天然承载 KL 正则、采样与更新解耦。10 章把它接上 LLM。

## 练习

1. 复现 §3.2 的无 clip 消融（$\epsilon=10^9$），画出你的 kl 曲线。确认 KL 是否出现过 > 1 的尖峰，并观察它与训练回报回落的时间对应关系。
2. 实现 KL early-stop：在每个 minibatch 更新后检查 `approx_kl > 1.5 * 0.01` 则跳出本批 epoch 循环（Spinning Up 逻辑）。对比有/无 early-stop 在 ep150 尖峰处的行为差异。
3. 把 GAE 换成 MC 优势（$\hat A_t = G_t - V(s_t)$，即 $\lambda=1$），重跑 400 回合。对比收敛速度与稳定性——验证 08 章 λ 消融的结论在 PPO 里是否保持。
4. 把 rollout 从 10 回合减到 1 回合（epochs 保持 10），观察 KL 与 clipfrac 如何爆炸。用 §8 误区 5 解释。
5. 思考题：GRPO 把 PPO 的 value/critic 整个删掉，用"组内回报减均值"替代优势（库内 Stage 7 的核心设计）。用本章的仪表盘语言说明：GRPO 保留了 PPO 的哪些保险丝、删掉了哪些部件？它为什么敢删（提示：LLM 场景奖励只在句尾、且稀疏）？（10 章正式展开。）

## 参考来源

- PPO 原始论文：Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347, 2017
- TRPO：Schulman et al., "Trust Region Policy Optimization", ICML 2015
- 重要性采样与代理目标的严谨处理：Schulman et al. TRPO 论文 §3；Agarwal et al., "On the Theory of Policy Gradient Methods", JMLR 2021
- 实现参照：OpenAI Spinning Up "PPO"（KL early-stop 与评估协议）、CleanRL `ppo.py`（单文件实现的工程细节）
- 本章训练/消融数值均为 seed=0、CPU 实测真实结果；KL 健康区间为经验做法（已标注）
