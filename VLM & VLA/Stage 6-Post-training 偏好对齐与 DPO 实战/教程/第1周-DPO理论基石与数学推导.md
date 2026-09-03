# 第 1 周教程：DPO 理论基石与数学推导

> **本周要回答的三个问题**
> 1. RLHF 三阶段到底重在哪里？PPO 的崩溃模式有哪些？
> 2. 从 KL 约束的 RL 目标到 DPO 损失的完整推导链是什么？
> 3. β 的物理意义是什么？它怎么影响训练动力学？

对应学习计划：第 1 周。交付物：纯 PyTorch 手写 `DPOLoss`（输入四组 logps + β，输出 loss / chosen_rewards / rejected_rewards），单元测试验证梯度传导。

---

## 1. 第一性原理：对齐的根本矛盾与 RLHF 的工程之重

### 1.1 根本矛盾：可验证的目标 vs 不可验证的"好"

SFT 的目标函数（模仿标注答案）可以大规模自动化优化，但它只能教"复述"，教不了"在多个合法回答中偏向更好的那个"——因为"更好"没有标注答案可比对，只有**相对偏好**。偏好对齐的根本矛盾因此是：**我们只有人类给的相对判断（A 比 B 好），却想优化一个绝对的目标（生成更好的回答）**。

RLHF 的经典解法把这个矛盾拆成三步：

```text
① SFT: 预训练模型 -> 会说话的模型
② RM: 人类偏好对 (x, y_w, y_l) -> 训练打分模型 r_φ(x, y)
        损失: L_RM = -log σ(r_φ(x, y_w) - r_φ(x, y_l))     (Bradley-Terry)
③ PPO: 用 r_φ 当环境奖励, 强化学习更新策略 π_θ
        目标: max_θ E[r_φ(x, y)] - β·KL(π_θ ‖ π_ref)
```

**Bradley-Terry（BT）偏好模型**是第②步的统计地基：假设人类选择 $y_w$ 的概率由分差决定：

$$
P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)
$$

（$\sigma$ 是 sigmoid；分差越大，偏好越确定。）它把"相对偏好"转化为对**标量奖励函数**的极大似然估计——注意这个形式，第 1.3 节 DPO 会原样借用它。

### 1.2 PPO 的痛点：四个模型的重量与三种崩溃

第③步 PPO 的工程清单（这正是 DPO 论文标题 "Your Language Model is Secretly a Reward Model" 要推翻的东西）：

| 组件 | 角色 | 显存（7B fp16 计） |
| --- | --- | --- |
| Actor（$\pi_\theta$） | 被训练的策略 | 14 GB + 梯度 + 优化器 |
| Reference（$\pi_{\text{ref}}$） | KL 锚点（冻结的 SFT 模型） | 14 GB |
| Reward（$r_\phi$） | 环境打分 | 14 GB |
| Critic（$V_\psi$） | PPO 的价值基线 | 14 GB + 梯度 + 优化器 |

**合计常超 60~80GB（不含激活）**——7B 模型的 PPO 实际是多卡工程。除显存外，训练流程本身是"生成 → 打分 → 更新"的在线循环（每步都要采样生成，慢），且有三种经典崩溃模式：

1. **KL 爆炸**：策略跑飞远离 π_ref，输出开始"胡言乱语但奖励高"（reward hacking 的载体）；
2. **奖励漂移（Reward Drift/Reward Hacking）**：策略找到 RM 的漏洞（如堆砌 RM 喜欢的词），奖励分数上涨而真实质量下降——RM 是个不完美的代理，强化学习会无情地优化它的漏洞；
3. **训练不稳定**：PPO 的超参敏感（clip ratio、GAE 参数、batch 内样本量），数值振荡常见，调参成本高。

**DPO 的战略洞察**：这三步里最贵的不是显存，而是"**显式训练一个 RM 再用它当环境**"这个中间商。如果能用数学把 RM 从优化目标里消掉，直接从偏好对更新策略，PPO 的全部痛点（在线采样、四模型、reward hacking 的中间层）就一起消失。

### 1.3 核心推导：从 RL 目标到 DPO 损失

**第一步：KL 约束的 RL 目标**（RLHF 第③步的形式化）：

$$
\max_{\pi_\theta} \ \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)} \big[ r^*(x, y) \big] - \beta \, \mathbb{D}_{\mathrm{KL}}\big[ \pi_\theta(y|x) \,\|\, \pi_{\text{ref}}(y|x) \big]
$$

$\beta > 0$ 控制"向奖励看齐"与"不偏离参考模型"之间的张力。

**第二步：求闭式最优解**。这个目标有一个著名的解析解（把 KL 展开成期望差，配方变换）：

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\!\left( \frac{1}{\beta} r^*(x, y) \right)
$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(r^*(x,y)/\beta)$ 是配分函数（只依赖 $x$，与 $y$ 无关——记住这点，它是消元的钥匙）。

**第三步：反解奖励（隐式奖励的诞生）**。对上式取对数并整理：

$$
r^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

**任何**奖励函数（真值 $r^*$ 或其估计）都可以用"最优策略与参考策略的对数概率比"表示，只多出一个 $\beta \log Z(x)$ 项。把 $\pi^*$ 换成参数化策略 $\pi_\theta$，得到**隐式奖励**：

$$
\boxed{\ r_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}\ }
$$

（这里故意省略了 $Z(x)$ 项——它马上会被差分消掉。）这就是论文标题的含义：**策略对数比本身就是奖励函数的重参数化**。

**第四步：代入 BT 模型，配分函数被差分消灭**。把隐式奖励代回 BT 偏好分布：

$$
P(y_w \succ y_l | x) = \sigma\Big( \underbrace{\beta \log \tfrac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} + \beta \log Z(x)}_{r_\theta(x, y_w)} - \underbrace{\beta \log \tfrac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \beta \log Z(x)}_{r_\theta(x, y_l)} \Big)
$$

**同一 $x$ 下 $Z(x)$ 相同，差分中完美抵消**——这就是为什么 DPO 只需要偏好对（同一问题的两个回答），不需要绝对分数。剩下的对 $-\log P(y_w \succ y_l)$ 取负对数似然，即得：

$$
\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

推导链总结（自测考点，要求能默写）：**KL 约束 RL 目标 → 闭式最优解 → 反解出隐式奖励 → 代入 BT 偏好模型 → 差分消 Z(x) → 负对数似然 = DPO 损失**。每一步都是代数操作，没有任何近似——这是 DPO "理论干净"的原因，也是它与后续各种启发式变体（第 2 周）的本质区别。

### 1.4 β 的物理意义与动力学

从两个角度看同一个旋钮：

1. **KL 惩罚强度**（原始目标视角）：$\beta$ 大 → 重罚偏离 $\pi_{\text{ref}}$ → 策略被锚在 SFT 模型附近，学到的偏好差异保守；$\beta$ 小 → 放松锚定 → 策略可以大幅移动，学得猛但崩坏风险高。
2. **奖励尺度**（隐式奖励视角）：$r_\theta = \beta \log(\pi_\theta/\pi_{\text{ref}})$，$\beta$ 同时缩放奖励的量纲。损失对 log 比值的"锐度"由 $\beta$ 控制——β 大，sigmoid 饱和慢，梯度持续存在；β 小，sigmoid 快速饱和（分差稍微拉开就接近 0 损失），**容易过早停止学习或过拟合**。

经验锚点（学习计划口径一致）：**β = 0.1** 是最常见起点；**0.05** 用于希望更强对齐力度的场景；LC（length-controlled）AlpacaEval 类场景有报告用 0.01~0.3 的范围扫参。**β 必须与学习率联动调**（两者共同决定策略移动速度），单独调 β 而不调 lr 是 DPO 调参最常见的误区（第 4 周展开）。

**一语双关的隐式奖励监控**：DPO 训练日志里的 `rewards/chosen`、`rewards/rejected` 就是 $\beta$ 缩放后的 log 比值（在当前 batch 上的均值）。理想曲线：chosen 缓慢上升（或保持）、rejected 下降、margin $= r_w - r_l$ 持续增大、`rewards/accuracies`（分差为正的样本占比）爬向 0.7~0.9。这些曲线是第 4 周的训练体检仪。

---

## 2. 实现与验证

### 2.1 本周 MVP：纯 PyTorch 手写 DPOLoss

```python
"""
手写 DPOLoss: 输入 policy/ref 的 chosen/rejected logps, 输出 loss 与隐式奖励。
运行方式: python stage6_week1_dpo_loss.py   (自带单元测试)
依赖: torch
"""
import torch
import torch.nn.functional as F


def dpo_loss(policy_chosen_logps: torch.Tensor,
             policy_rejected_logps: torch.Tensor,
             ref_chosen_logps: torch.Tensor,
             ref_rejected_logps: torch.Tensor,
             beta: float = 0.1):
    """
    全部 logps 为回答区 per-sequence 对数概率之和 (sum over answer tokens),
    形状 [B]。返回 loss 标量, 以及隐式奖励 r_theta(x,y) = beta * log(pi/pi_ref)。
    """
    # ---- 隐式奖励: beta * log(pi_theta / pi_ref) ----
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    # ---- BT 偏好的负对数似然 ----
    logits = chosen_rewards - rejected_rewards            # 分差, Z(x) 已被差分消去
    loss = -F.logsigmoid(logits).mean()

    acc = (logits > 0).float().mean()                     # 偏好判断准确率
    return loss, chosen_rewards, rejected_rewards, acc


def simpo_loss(policy_chosen_logps: torch.Tensor,
               policy_rejected_logps: torch.Tensor,
               len_chosen: torch.Tensor, len_rejected: torch.Tensor,
               beta: float = 2.0, gamma: float = 0.5):
    """第2周预告: SimPO (长度归一化 + margin, 无 ref)。此处一并给出便于对比。"""
    chosen = beta / len_chosen * policy_chosen_logps
    rejected = beta / len_rejected * policy_rejected_logps
    loss = -F.logsigmoid(chosen - rejected - gamma).mean()
    return loss, chosen, rejected


class _ReqGrad(torch.Tensor):
    pass


def _run_tests():
    torch.manual_seed(0)
    B, beta = 16, 0.1

    def make(logp_w, logp_l, ref_gap=2.0):
        """构造带梯度的 policy logps 与固定 ref logps"""
        pc = (torch.zeros(B) + logp_w + torch.randn(B) * 0.1).requires_grad_(True)
        pl = (torch.zeros(B) + logp_l + torch.randn(B) * 0.1).requires_grad_(True)
        rc = torch.zeros(B) + logp_w + ref_gap            # ref: chosen 比 policy 高 ref_gap
        rl = torch.zeros(B) + logp_l + ref_gap
        return pc, pl, rc, rl

    # ---- 测试 1: 初始状态 (policy == ref) 时 loss = -log σ(0) = log 2 ----
    pc, pl, rc, rl = make(-100.0, -120.0, ref_gap=0.0)
    loss, cw, rw, acc = dpo_loss(pc, pl, rc, rl, beta)
    assert torch.isclose(loss, torch.log(torch.tensor(2.0)), atol=1e-5), \
        "policy==ref 时分差应为 0, loss 应为 log2 (DPO 的'零起点'性质)"
    assert abs(acc.item()) < 1e-6

    # ---- 测试 2: 梯度传导与方向性 ----
    pc, pl, rc, rl = make(-100.0, -120.0, ref_gap=0.0)
    loss, cw, rw, acc = dpo_loss(pc, pl, rc, rl, beta)
    loss.backward()
    assert pc.grad is not None and pl.grad is not None, "梯度未回传"
    # 提高 chosen 概率应降低 loss => d(loss)/d(chosen_logps) < 0
    assert (pc.grad < 0).all(), "chosen logp 的梯度应为负 (推高 chosen)"
    # 降低 rejected 概率应降低 loss => d(loss)/d(rejected_logps) > 0
    assert (pl.grad > 0).all(), "rejected logp 的梯度应为正 (压低 rejected)"

    # ---- 测试 3: margin 增大 => loss 单调下降 ----
    prev = float("inf")
    for gap in [0.0, 0.5, 1.0, 2.0, 4.0]:
        pc, pl, rc, rl = make(-100.0, -120.0 - gap, ref_gap=0.0)
        loss, *_ = dpo_loss(pc, pl, rc, rl, beta)
        assert loss.item() < prev, f"margin={gap} 时 loss 未单调下降"
        prev = loss.item()

    # ---- 测试 4: beta 缩放奖励 ----
    pc, pl, rc, rl = make(-100.0, -120.0, ref_gap=1.0)
    _, cw1, _, _ = dpo_loss(pc, pl.detach(), rc, rl, beta=0.1)
    _, cw2, _, _ = dpo_loss(pc, pl.detach(), rc, rl, beta=0.2)
    assert torch.isclose(cw2, 2 * cw1, atol=1e-4).all(), "奖励应随 beta 线性缩放"

    # ---- 测试 5: ref 项必须参与 (去掉 ref 则退化为普通 BT, 无 KL 锚) ----
    pc, pl, rc, rl = make(-100.0, -120.0, ref_gap=0.0)
    loss_no_ref, *_ = dpo_loss(pc, pl, pc.detach(), pl.detach(), beta)  # ref=policy: 恒零分差
    assert torch.isclose(loss_no_ref, torch.log(torch.tensor(2.0)), atol=1e-5)

    print("✅ DPOLoss 五项测试通过: 零起点性质 / 梯度方向 / margin单调性 / β缩放 / ref必要性")


if __name__ == "__main__":
    _run_tests()
```

**预期输出**：

```text
✅ DPOLoss 五项测试通过: 零起点性质 / 梯度方向 / margin单调性 / β缩放 / ref必要性
```

五个测试各验证一个**关键性质**而非表面格式：① `policy==ref` 时 loss=log 2（DPO 从 SFT 状态零起点，与 Stage 4 LoRA 的 B=0 初始化同一设计哲学）；② 梯度方向与"推高 chosen、压低 rejected"的直觉一致；③ margin 增大 loss 单调下降（优化目标的方向正确性）；④ β 线性缩放奖励（1.4 节论断的数值验证）；⑤ 去掉 ref 项损失失去锚定（说明 ref 不是可有可无的装饰）。

### 2.2 从 per-sequence logps 到训练循环

真实训练中，`*_logps` 的来源是：对 `(x, y)` 拼接序列做一次前向，取回答区（掩码后）各 Token logprob **求和**（per-sequence sum，DPO 标准口径；per-token 平均是 SimPO 的改法，第 2 周讲）。第 4 周会用 TRL 的 `DPOTrainer`，其内部对 batch 做了 chosen/rejected 拼接成一条的前向优化——机制与手写版完全一致，只是工程上更省前向次数。

---

## 3. 工程权衡与失效模式

### 3.1 DPO 与 PPO 的系统对比

| 维度 | PPO (RLHF) | DPO |
| --- | --- | --- |
| 在显存中的模型 | Actor+Ref+RM+Critic（4 个） | Policy+Ref（2 个） |
| 训练范式 | 在线（生成→打分→更新循环） | 离线（固定偏好对，类 SFT） |
| 调参难度 | 高（clip/GAE/batch 采样多旋钮） | 中（β/lr 两旋钮为主） |
| 奖励漏洞利用 | 显式 RM 可被 hack | 无显式 RM（但偏好数据本身可被"利用"） |
| 上限 | 理论上限更高（在线探索） | 受限于偏好数据覆盖 |
| 适用 | 大规模对齐管线（有基建） | 资源有限、快速迭代、离线数据充足 |

DPO 并非全面优于 PPO：它把"RM 的质量"问题转化为"**偏好数据的质量**"问题（第 3 周的主题）；在线探索类目标（如需要模型自己发现新策略的场景）仍是 PPO 系的领地。

### 3.2 三个代表性失效模式

**失效 1：β 与 lr 失配导致"要么没学要么崩"**
- **症状**：loss 迅速降到接近 0、`rewards/margins` 冲高后模型输出退化（重复/胡话）；或 loss 长期停在 log2 附近纹丝不动。
- **根因**：前者是 β 太小 + lr 偏大——sigmoid 快速饱和后梯度消失前的"过冲"，策略已大幅偏离 π_ref；后者是 β 太大 + lr 偏小——奖励尺度被压缩，梯度太弱。
- **定位**：看 `logps/chosen` 与 `logps/rejected` 的绝对值轨迹——前者应缓慢下降（生成概率仍高）、后者下降更快；若 chosen logps 暴跌即策略崩坏信号。
- **修复**：联动调参——β=0.1 配 DPO 常用 lr（SFT 的 1/10 以下，如 5e-7~5e-6，学习计划口径）；崩坏时升 β 或降 lr，不学时反向。

**失效 2：长度偏置——DPO 奖励"更长"而非"更好"**
- **症状**：DPO 后输出显著变长，内容质量未升；评测中长答案在 BLEU 类指标上占便宜、Judge 评分下降。
- **根因**：per-sequence sum 口径下，长回答的 log 概率绝对值天然更低（更多负数相加）；当 rejected 恰好比 chosen 长时，优化会系统性推高长回答的概率——**长度与质量在数据里偶然相关，DPO 学到的是相关而非因果**。
- **定位**：统计训练前后平均回答长度；按长度分桶看奖励。
- **修复**：长度平衡采样（偏好对内 y_w/y_l 长度接近）；或直接换 SimPO（长度归一化 + margin，第 2 周）；或 RLHF-V 式细粒度段级 DPO（只对差异段算偏好）。

**失效 3：偏好数据质量塌方——garbage in, aligned garbage out**
- **症状**：训练曲线完美（margin 涨、acc 到 0.9+），但人工评估与 benchmark 全面变差。
- **根因**：偏好对的 y_l 标注错误（好的被标 rejected）或 y_w/y_l 无实质语义差异（只差标点）——DPO 会忠实地学会错误偏好或无意义偏好。**DPO 没有外部裁判能纠正数据错误**（这正是消掉 RM 的代价）。
- **定位**：抽 50 条偏好对人工盲评——标注一致率 <85% 时数据不可用；检查 `rewards/accuracies` 是否异常逼近 1.0（太快太满 = 数据可分性过强 = 可能是模板痕迹而非语义差异）。
- **修复**：数据侧根治（第 3 周的构造方法与对比度校验）；小数据高质量（RLHF-V 用 1.4k 人工细粒度样本超过 10k 粗标数据的 LLaVA-RLHF，已核实 CVPR 2024 报告数字）优于大数据低质。

---

## 4. 延伸思考题

1. **推导补全**：1.3 节第二步"闭式最优解"的推导被跳过了。自己完成它：把 KL 展开为期望之差，对 $\pi(y|x)$ 做受约束变分（总概率=1 的 Lagrange 乘子法），验证解的正则化常数恰为 $Z(x)$。（这是面试高频题，值得徒手推一遍。）
2. **β→0 与 β→∞ 的极限**：分析两个极限下 DPO 损失的形态。β→0 时会发生什么（提示：奖励尺度消失，log 比值被放大——等价于无 KL 约束的 BT 拟合，过拟合偏好数据）；β→∞ 时呢（策略冻结）？由此解释为什么 β 与 lr 必须联动。
3. **动手实验**：把 2.1 的 `dpo_loss` 接上一个 2 层玩具 LM，构造 100 条合成偏好对（chosen 是某固定句式、rejected 是打乱版），跑 200 步训练，画 margin/acc 曲线；然后把 10% 的偏好标签随机翻转，观察训练曲线与最终 acc 的变化——这是"标签噪声敏感性"的最小实证。

---

*下一篇：[第 2 周：偏好对齐演进算法（SimPO/KTO/IPO/ORPO）](第2周-对齐算法演进与变体.md)*
