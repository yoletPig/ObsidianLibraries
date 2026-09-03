# 第 3 周教程：verl 强化学习框架架构与分布式 Rollout

> **本周要回答的三个问题**
> 1. RL 训练的算力形态与 SFT/DPO 有何本质不同？为什么需要 HybridEngine？
> 2. verl 的 Ray 分布式架构里各 Worker 怎么协作？3D-HybridEngine 消除的是什么开销？
> 3. verl 的配置树（`actor_rollout_ref` / `reward` / `data` / `algorithm`）怎么读？

对应学习计划：第 3 周。交付物：搭建 verl 环境，单卡/多卡跑通官方 GRPO 最小 Demo（Qwen 0.5B 级 + GSM8K），WandB 监控 `actor/kl_divergence`、`reward/mean`、`response_length`。

**框架说明**（已核实 GitHub main 分支，2026-09，v0.9.0）：verl 已迁移至 `verl-project/verl`，是 HybridFlow 论文（arXiv:2409.19256）的开源实现；官方对 3D-HybridEngine 的描述为"efficient actor model resharding—eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases"。框架演进快（vLLM/SGLang 版本要求、配置字段随版本变化），**以你安装版本的官方文档为准**，本文聚焦不随版本漂移的架构原理。

---

## 1. 第一性原理：RL 训练的算力形态

### 1.1 根本矛盾：两种异构负载要在同一批 GPU 上交替发生

对比三种训练的 GPU 时间构成：

| 训练类型 | 前向 | 反向 | 生成（decode） | 负载形态 |
| --- | --- | --- | --- | --- |
| SFT | 1 | 1 | 无 | 纯计算密集，batch 规整 |
| DPO | 2（policy+ref） | 1 | 无 | 同上，略重 |
| **RL（GRPO）** | 2+ | 1 | **G × 每条 prompt 数百 Token** | **生成 + 训练交替的混合体** |

RL 每个迭代步的两个阶段：

1. **Rollout 阶段**：Actor 对每条 prompt **自回归生成 G 条回答**——decode 是访存受限的，训练框架的 forward 不擅长（无 continuous batching 时 GPU 大量空转）；
2. **Training 阶段**：对 G 条回答算 old/ref/policy 三套 logprob + 反向更新——这是训练框架的主场。

**矛盾在于两阶段的最优引擎不同**：Rollout 要 vLLM/SGLang（PagedAttention、continuous batching），Training 要 FSDP/Megatron（分布式参数分片、优化器）。而两者操作的是**同一份 Actor 权重**——每个迭代步都要把权重从"训练布局"（分片在多卡）转成"推理布局"（推理引擎可用的形态），用完再转回来。

**朴素方案的代价**：每步全量权重广播/收集 + 推理引擎重新初始化 KV cache 池——权重传输与引擎初始化的耗时可以超过真正的计算时间（尤其大模型 + 高频迭代）。官方博客口径：verl 相比朴素 RLHF 实现可获得最高约 20 倍的吞吐提升（数字来自字节跳动团队发布公告，条件为其测试环境）——提升的主要来源正是消除这个切换开销。

### 1.2 HybridEngine：resharding 而非重新加载

3D-HybridEngine 的核心：**训练结束后不销毁权重、推理开始前不重新加载，而是在两种并行布局之间做原地的分片重排（resharding）**：

```text
[Training 阶段]                    [切换]                      [Rollout 阶段]
FSDP/Megatron 并行布局   ──>  增量 resharding +   ──>   vLLM/SGLang 推理布局
(参数分片在 TP×DP×PP)         通信原语优化               (推理引擎的分片形态)
                              (官方: 消除显存冗余
                               与切换期通信开销)
        ↑                                                    │
        └────────────── 每个迭代步往返一次 ←──────────────────┘
```

配合 **Ray 的资源编排**：同一组 GPU 在不同阶段被重新"扮演"不同 Worker 角色（Actor-Worker ↔ Rollout-Worker），权重常驻显存、按需转换形态。这是理解 verl 架构的总钥匙：**它管理的不只是模型，而是"GPU 角色与权重形态的生命周期"**。

### 1.3 verl 的混合控制器编程模型

verl 的设计哲学（HybridFlow）：**控制流用单进程 Ray 编排（人类易写），数据流在 worker 组内多机多卡高效执行**。解耦了"计算"与"数据依赖"，使换算法（PPO→GRPO）只改控制流、换引擎（FSDP→Megatron、vLLM→SGLang）只改数据流——这也是它成为 RLVR 事实标准的原因之一（TinyZero、DAPO、Easy-R1 等大量生态项目基于它，已核实其 Awesome 列表）。

---

## 2. 系统架构：Worker 协作与配置树

### 2.1 Ray 集群中的角色分工

```text
RayDriver (单进程控制器: 编排数据流, 按算法写好的 DAG 调度)
   │
   ├─ ActorRolloutRefWorker  (通常同组 GPU 复用三角色)
   │     ├─ Actor 角色:   训练态前向/反向 (FSDP/Megatron)
   │     ├─ Rollout 角色: HybridEngine 切换 -> vLLM/SGLang 采样 G 条
   │     └─ Ref 角色:     冻结策略的 logp 计算 (KL 用)
   ├─ CriticWorker            (PPO 需要; GRPO 关闭)
   ├─ RewardWorker            (model-based RM; RLVR 时退化为调用你的 reward_function)
   └─ (多模态: VLM 的视觉塔随 Actor 一起 reshard)
```

GRPO 下的关键简化：**CriticWorker 不存在**（第 1 周的组均值基线）；RewardWorker 在 RLVR 下不需要驻留任何模型——它就是一个 Python 函数调用（第 2 周 Verifier Engine 的挂载点，verl 文档称 function-based reward）。

### 2.2 一个迭代步的数据流（GRPO）

```text
DataLoader (prompts, batch=B)
   ▼ ① Rollout: HybridEngine -> vLLM, 每 prompt 采样 G 条回答
   ▼ ② 打分: reward_function (Verifier Engine) -> 每条一个标量
   ▼ ③ 组优势: A_i = (r_i - mean)/(std + ε)  (CPU/单卡, 秒级)
   ▼ ④ logp 三连: old_logp(rollout时已存) / ref_logp / policy_logp (回答区)
   ▼ ⑤ GRPO loss (clip + KL) -> 反向 -> Actor 更新
   ▼ ⑥ 权重同步回 Rollout 引擎 (下一迭代)
```

多模态差异集中在 ①（图像作为 prompt 前缀，rollout 时视觉特征随 prompt 一次编码、G 条生成复用——prefix cache 的主场）与 ④（多模态模型前向需带 pixel_values）。

### 2.3 配置树速读

verl 用 Hydra 风格的嵌套配置，主干节点：

```yaml
data:
  train_files: ...             # parquet; 多模态带 image 列
  prompt_key: prompt
  max_prompt_length: 2048      # 含图像 token 预算!
  max_response_length: 1024

algorithm:
  adv_estimator: grpo          # 优势估计器 = 算法选择的总开关
  use_kl_in_reward: true/false # KL 挂在奖励上 or 挂在损失上 (两种实现路径)

actor_rollout_ref:             # 三角色合一节点 (GRPO 的主战场)
  actor:
    strategy: fsdp2            # 训练后端 (fsdp/fsdp2/megatron)
    optim: {lr: 1e-6}
    loss_agg_mode: token-mean
  rollout:
    name: vllm                 # 或 sglang
    n: 8                       # GRPO 的组大小 G
    temperature: 1.0
    gpu_memory_utilization: 0.4 # 给推理引擎留的显存比例 (关键旋钮!)
  ref:
    strategy: fsdp2            # GRPO 也需要 ref (KL 正则)

reward_model:
  enable: false                # RLVR: 关掉 model-based, 用 function-based
custom_reward_function:
  path: my_verifier.py         # 第 2 周 Verifier Engine 挂载点
  name: compute_score

trainer:
  total_epochs / n_gpus_per_node / nnodes / project_name(=wandb)
```

（字段名随版本微调，语义不变：**数据 / 算法 / 三合一的角色组 / 奖励 / 训练器**五大块。）三个最容易踩的配置：`rollout.gpu_memory_utilization`（训练态与推理态共享显存，给 vLLM 留太少会 OOM、留太多训练侧 OOM——性能调优文档的核心议题）；`max_prompt_length`（多模态时必须算上图像 Token）；`rollout.n`（就是 GRPO 的 G）。

---

## 3. 实现与验证

### 3.1 环境搭建与最小 Demo

```bash
# 安装 (版本矩阵敏感: vllm/sglang/torch 需配套, 优先用官方 docker 或 uv)
git clone https://github.com/verl-project/verl.git
cd verl && pip install -e .
# 或 docker: 官方镜像自带版本矩阵 (如 vllm==0.12.0, sglang==0.5.6 的稳定镜像)

# 官方 GRPO 最小示例 (0.5B 数学模型 + GSM8K, 单卡可跑)
# 数据准备 (脚本内置 GSM8K 下载与预处理)
python examples/data_preprocess/gsm8k.py

# 单卡 GRPO (示例脚本, 字段以所装版本为准)
export WANDB_API_KEY=xxx
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=64 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  trainer.total_epochs=1 \
  trainer.project_name=verl_grpo_demo \
  trainer.n_gpus_per_node=1 trainer.nnodes=1
```

（官方文档 Quickstart/GRPO 页有对应脚本；VLM 版本见 `examples/grpo_trainer/run_qwen2_5_vl_*.sh`，已核实其存在。）

### 3.2 WandB 监控的验收清单（本周 MVP 的核心交付）

| 指标 | 健康形态 | 判读 |
| --- | --- | --- |
| `reward/mean` | 缓慢上升（RLVR 下从基座水平起步） | 训练在起效的总信号 |
| `actor/kl_divergence` | 小幅上升后稳定（<1 nat 量级） | 飙升 = 跑飞；恒 0 = KL 没开或 β=0 |
| `response_length` | 初期波动，随训练可能缓增（CoT 变长）或稳定 | **单调暴涨 = 复读 hack**；骤降 = 摆烂/拒答 |
| `critic/…` | GRPO 下不应存在 | 出现了说明 adv_estimator 不是 grpo |
| `timing/…` | rollout 与 training 的耗时占比 | rollout 占比 >70% 时该调 rollout 并发/GPU 分配 |

### 3.3 跑通的最小自检脚本

```python
"""对 WandB 导出或日志做三项 sanity check (伪代码级骨架, 按实际日志格式解析)"""
def check(run_metrics: dict):
    rm = run_metrics["reward/mean"]
    assert rm[-1] > rm[0], "reward 未增长, 检查 adv_estimator/verifier/lr"
    kl = run_metrics["actor/kl_divergence"]
    assert all(k < 10.0 for k in kl), "KL 飙升, 降低 lr 或开/调大 KL 正则"
    rl = run_metrics["response_length"]
    assert rl[-1] < 4 * rl[0], "response_length 暴涨, 疑似复读 hack (查长度惩罚)"
    print("✅ 三项 sanity check 通过")
```

---

## 4. 工程权衡与失效模式

### 4.1 决策表：单卡/小集群的配置起点

| 资源 | 建议配置 | 说明 |
| --- | --- | --- |
| 1×24G（0.5B~1.5B） | FSDP + vLLM，`gpu_memory_utilization=0.35`，G=8 | 官方小模型 Demo 级 |
| 4×40G（3~7B） | FSDP2 + vLLM，LoRA RL 可选（官方支持 Multi-gpu LoRA RL） | LoRA 显著省训练态显存 |
| 8×80G（7~32B） | FSDP2/Megatron + vLLM/SGLang | SGLang 对多轮/VLM RL 支持活跃（官方推荐） |

### 4.2 三个代表性失效模式

**失效 1：训练态与推理态的显存打架（最高频 OOM）**
- **症状**：Rollout 阶段 OOM（vLLM 分配 KV cache 时）；或 Training 阶段 OOM（优化器状态装不下）。
- **根因**：`rollout.gpu_memory_utilization` 是给推理引擎的显存预算上限——设太高，训练态的梯度/优化器状态挤不下；设太低，rollout 的 KV cache 不够、长回答采样失败。
- **定位**：看 OOM 发生的阶段（日志里的 phase 标记）与当时的显存水位。
- **修复**：压 `max_response_length`、降 G、开 actor 的 CPU offload（FSDP2 offload 与梯度累积兼容，官方 PR #1026）、或减小 mini-batch。这是一个**两阶段预算分配问题**，没有万能值，按 `timing` 曲线迭代。

**失效 2：权重同步静默失败——策略与 rollout 引擎脱节**
- **症状**：reward 长期不动，但 loss 正常下降；同一条 prompt 的 rollout 输出与手动用训练后 checkpoint 生成的不一致。
- **根因**：权重 resharding/同步环节出错（版本不匹配的 vLLM 补丁、FSDP 状态字典转换 bug），rollout 引擎一直在用旧权重采样——训练在"旧策略上反复优化"。
- **定位**：训练 N 步后保存 checkpoint，手动用 vLLM 加载生成对照——与训练日志中的 rollout 样本 diff。
- **修复**：用官方稳定版本矩阵（docker）；升级 vLLM/SGLang 前查 verl 的兼容声明（官方明确建议避开 vllm 0.7.x）。

**失效 3：多模态 prompt 长度预算漏算图像 Token**
- **症状**：VLM GRPO 启动即报长度截断/OOM，或训练正常但模型"看不见图"（视觉 Token 被截掉）。
- **根因**：`max_prompt_length` 按文本长度设置，高分辨率图像的视觉 Token（Stage 1 的 Token 公式，可达数千）挤爆预算。
- **定位**：打印一条预处理后样本的总 Token 数（prompt + image + response）。
- **修复**：限制输入图片分辨率（Stage 2 的管线纪律）；`max_prompt_length` 按图像 Token 上限重设；rollout 侧同步调 `max_model_len`。

---

## 5. 延伸思考题

1. **HybridEngine 的替代设计**：除了"resharding 原地切换"，还可以（a）训练与推理用独立 GPU 池（代价：权重定期跨池传输）；（b）完全异步（rollout 池持续采旧权重，训练池持续更新——off-policy 化）。三种设计在吞吐、显存、on-policy 程度上的取舍各是什么？verl 的 experimental 目录里已有 fully_async_policy 等探索（已核实），说明这个维度在活跃演化。
2. **把 toy 训练放大**：把第 1 周 2.1 的 toy GRPO 从 2-Token 环境扩到"生成 20 个 Token 的算式"环境（策略输出 `a+b=c` 形式，Verifier 判 c 对错），跑通一个"真正的生成式 RLVR"最小回路——这是理解 verl 各组件职责的最小完整映射。
3. **成本建模**：估算一次 VLM GRPO 迭代的耗时构成（rollout G 条 × 数百 Token 的 decode 时间 vs 训练态 3 次前向 + 1 次反向），代入你的 GPU 型号与 G=8，算出 rollout/training 的耗时比——这个比值决定你该把调优精力放在采样侧还是训练侧。

---

*下一篇：[第 4-5 周：多模态 Visual CoT 与 Vision-Language RLVR 实战](第4-5周-多模态RLVR与VisualCoT实战.md)*
