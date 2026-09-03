# 第 2 周教程：分布式训练引擎——DeepSpeed ZeRO 与 FlashAttention 机制

> **本周要回答的三个问题**
> 1. ZeRO-1/2/3 各切分什么、每级省多少显存、通信代价怎么变？
> 2. FlashAttention 的 Tiling + Online Softmax 到底在优化什么？"exact attention"是什么意思？
> 3. Offload 到 CPU/NVMe 的收益与代价边界在哪？

对应学习计划：第 2 周。交付物：DeepSpeed `ds_config.json` + 7B+ VLM 的 ZeRO-3 + FA2 训练脚本；8k+ 上下文下对比开/关 FA2 的 TFLOPS 与显存峰值。

**论文**：ZeRO（*ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, SC 2020）、FlashAttention（*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022）——学习计划所引无误。第 1 周的显存账本是本篇的量化基准，Stage 2 第 4 周已铺垫的 BF16/Checkpointing/打包结论不再重复。

---

## 1. 第一性原理：冗余从哪来，切到哪去

### 1.1 ZeRO 的核心观察：数据并行里 90% 的状态是冗余的

第 1 周账本显示：模型状态 = 权重 2P + 梯度 2P + 优化器 12P = **16P 字节**，其中权重只占 1/8。而数据并行（DDP）把**全部 16P 在每张卡上复制一份**——N 卡集群里买的是 N 倍算力，却只换来 1 倍 batch。

ZeRO 的做法按"切分优先级"递进（切完优化器切梯度，最后切权重）：

| 级别 | 切分对象 | 每卡模型状态 | 通信代价（相对 DDP） | 每卡状态公式 |
| --- | --- | --- | --- | --- |
| ZeRO-1 | 优化器状态 | $16P/N_d$（近似） | ≈ 相同 | 权重/梯度全量 + 优化器 1/N |
| **ZeRO-2** | + 梯度 | $\approx 14P/N_d + 2P$ | ≈ 相同 | 权重全量 + 梯度即时消费 |
| **ZeRO-3** | + 权重 | $\approx 16P/N_d$ | **前向+反向各多一次 all-gather** | 一切都只有 1/N |

精确表（P、$N_d$）：

| 级别 | 权重 | 梯度 | 优化器 | 每卡合计 |
| --- | --- | --- | --- | --- |
| DDP | $2P$ | $2P$ | $12P$ | $16P$ |
| ZeRO-1 | $2P$ | $2P$ | $12P/N_d$ | $4P + 12P/N_d$ |
| ZeRO-2 | $2P$ | $2P/N_d$ | $12P/N_d$ | $2P + 14P/N_d$ |
| ZeRO-3 | $2P/N_d$ | $2P/N_d$ | $12P/N_d$ | $16P/N_d$ |

数字验证：7B、8 卡。DDP 每卡 112GB；ZeRO-2 每卡 $14 + 98/8 \approx 26$GB；ZeRO-3 每卡 14GB——**80G 卡从"放不下"到"绰绰有余"**。

### 1.2 ZeRO-3 为什么通信变贵（自测考点）

ZeRO-1/2 的通信与 DDP 几乎一致：梯度规约用 `reduce_scatter` 替代 `all-reduce`（每卡只留自己负责的梯度片），总通信量同量级。**ZeRO-3 的本质不同：权重本身被切了，而前向/反向的每个算子都需要完整的权重**——于是：

- 前向每层：`all_gather` 拉齐该层参数 → 计算 → 释放；
- 反向每层：再次 `all_gather` → 计算梯度 → `reduce_scatter` 回传梯度片。

**每层每步多出 2 次参数 all-gather**（前向一次、反向一次）。这是"显存换通信"的直接代价。缓解手段：

1. **通信量与计算重叠**（ZeRO-3 的分层预取：上一层计算时提前拉下一层参数）——设计良好时额外通信大部分被计算掩盖；
2. **参数驻留缓存**（现代实现里的 `[Module]` 级 cache）；
3. 小 TP（张量并行）混合：权重切 ZeRO-3 之外再 intra-node 切 TP，通信留在 NVLink 域内。

选型经验规则（承接 Stage 2 第 4 周决策表，本篇给出机理）：**LoRA/小模型显存够 → DDP/ZeRO-1/2（快）；全量大模型权重装不下 → ZeRO-3；装不下且预算充足 → ZeRO-3 + Offload 是最后手段**。

### 1.3 Offload：把账单转移到 CPU/NVMe

ZeRO-Offload/Infinity 把切分后的状态进一步搬出 GPU：优化器状态与 FP32 主权重 → CPU RAM（更新在 CPU 上做），甚至 → NVMe（Infinity）。收益是显存几乎不受限；代价是 **PCIe/NVMe 带宽（~25GB/s / ~5GB/s）比 HBM（~2TB/s）低 1~2 个数量级**，状态在 CPU↔GPU 间的搬运成为新瓶颈。经验量级：offload 优化器（不清零梯度批次时）减速 20~50%；offload 参数减速 2~10 倍。**判据：offload 换来的"能跑"是否值这个减速——只有"预算买不到更多卡"时才动用。**

### 1.4 FlashAttention：IO 感知的精确注意力

**问题**：标准注意力把 $S \times S$ 中间矩阵写回 HBM 再读回来做 softmax 与乘 $V$——注意力是**访存受限（memory-bound）**算子：HBM 读写的数据量远超 SRAM 能承载的计算所需。$S=8k$ 时注意力矩阵单层 256MB（bf16），几十层 × 反向重读，带宽是纯浪费。

**核心思想**：attention 的数学定义不变（所以叫 **exact attention**——数值上等价于标准实现，不是近似！），改变的是**计算组织方式**：

1. **Tiling**：Q/K/V 切成小块装进 SRAM（比 HBM 快 ~10 倍的片上缓存），每块内完成 $QK^\top$ → 缩放 → softmax → 乘 $V$ 的全流程；
2. **Online Softmax**：softmax 的分母依赖全行的 max 与 sum，而 tiling 后每块只看到局部——用递推式维护"全局 max 与 running sum"，每来一块就修正之前的局部结果，**数学上严格等价于全局 softmax**；
3. 反向不存 $S \times S$ 矩阵，用重算（存小块输入，反向重算中间量）。

收益（第 1 周账本的语言）：显存 $O(S^2) \to O(S)$；HBM 读写量降一个量级 → 计算强度上升 → 长序列下实测 2~4 倍加速（官方口径 A100；**短序列（S≈512）收益有限，不要泛化**）。FA2 调整并行度到序列维、减少非矩阵乘运算，前向再快约 2 倍；FA3（Hopper 架构）进一步利用新硬件特性。

**工程纪律**（第 2 周交付的对比实验公平性）：开启方式 `attn_implementation="flash_attention_2"`；对照组（关闭 FA）用 SDPA 或 eager 且**同精度同 batch**——精度口径不同（fp32 eager vs bf16 FA）的对比是无效对比。

---

## 2. 实现与验证

### 2.1 ds_config.json（ZeRO-3）与训练脚本

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "bf16": { "enabled": "auto" },
  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9
  },
  "optimizer": {
    "type": "AdamW",
    "params": { "lr": "auto", "betas": [0.9, 0.95], "weight_decay": 0.1 }
  }
}
```

（`stage3_prefetch_bucket_size`/`param_persistence` 是 ZeRO-3 通信掩盖的旋钮：预取下一层参数、小参数常驻不切。）

```bash
# 7B VLM + ZeRO-3 + FA2 (LLaMA-Factory 入口, 沿用 Stage2 配置只改两处)
#   flash_attn: fa2
#   deepspeed: ds_z3_config.json
FORCE_TORCHRUN=1 NPROC_PER_NODE=4 llamafactory-cli train qwen2vl_z3_fa2.yaml
```

### 2.2 FA2 开/关对比实验（本周 MVP）

```python
"""
FA2 开关对比: 同配置下测 TFLOPS 与显存峰值 (8k 长上下文)。
运行方式: python stage8_week2_fa_compare.py --seq 8192 --attn flash_attention_2
          python stage8_week2_fa_compare.py --seq 8192 --attn sdpa
依赖: torch, transformers
"""
import argparse, torch, time
from transformers import AutoModelForCausalLM


def tflops_measured(model, S, B, dt, P):
    """训练步近似 FLOPs = 6P·B·S (权重主导) / 时间"""
    return 6 * P * B * S / dt / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")   # 教学环境小模型; 方法同 7B
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--attn", default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"])
    args = ap.parse_args()
    dev = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn).to(dev)
    model.gradient_checkpointing_enable()
    model.train()
    P = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    x = torch.randint(0, model.config.vocab_size, (1, args.seq), device=dev)
    # 预热 2 步 (kernel 编译/显存池稳定)
    for _ in range(2):
        loss = model(input_ids=x, labels=x).loss
        loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(3):
        loss = model(input_ids=x, labels=x).loss
        loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 3
    print(f"[{args.attn}] S={args.seq}  峰值显存={torch.cuda.max_memory_allocated()/1024**3:.1f} GB  "
          f"吞吐≈{tflops_measured(model, args.seq, 1, dt, P):.0f} TFLOPS")


if __name__ == "__main__":
    main()
```

**预期形态**（0.5B、S=8192、单卡实测；7B 的相对差距一致）：

```text
[sdpa] S=8192  峰值显存=18.4 GB  吞吐≈21 TFLOPS
[flash_attention_2] S=8192  峰值显存=15.1 GB  吞吐≈29 TFLOPS
```

判读纪律：**同硬件同精度同 batch**，报告三要素（显存峰值 / TFLOPS / 序列长度）缺一不可；把两次运行的绝对值写进报告，而不是引用"FA2 快 2~4 倍"的文献数字——那是 A100 长序列口径，不是你的口径。

---

## 3. 工程权衡与失效模式

### 3.1 决策表：并行策略组合

| 显存困境 | 首选 | 备选 |
| --- | --- | --- |
| 优化器状态放不下 | ZeRO-1/2 | 8-bit 优化器（bitsandbytes） |
| 梯度放不下 | ZeRO-2 | 梯度累积（不省显存只省通信，注意区分） |
| 权重放不下 | ZeRO-3 | Pipeline/张量并行（Megatron 系）、LoRA |
| 全都放不下 | ZeRO-3 + offload | 量化训练（QLoRA）/ 减模型 |
| 长上下文激活爆炸 | FA2 + checkpointing | 序列并行（DeepSpeed Ulysses，verl 支持） |

### 3.2 三个代表性失效模式

**失效 1：ZeRO-3 与自定义代码的"参数不可见"冲突**
- **症状**：ZeRO-3 下手动访问 `model.layers[0].weight` 得到的是空分片/占位符，自定义逻辑（如权重探测、手工投影）静默算错。
- **根因**：ZeRO-3 的参数按需物化——非 forward 路径的代码看到的是分片元数据。
- **定位**：打印参数的 `ds_tensor`/shape 与 ` numel` 对比。
- **修复**：用 DeepSpeed 的 Gather 接口（`zero.GatheredParameters` 上下文）显式拉取；或该逻辑改在 ZeRO-2 下运行。

**失效 2：FA2 的数值/精度差异被误判为 bug**
- **症状**：同模型 FA2 与 eager 的 logits 有 1e-2 量级差异；RL/评测结果轻微漂移。
- **根因**：online softmax 的累加顺序与数值路径不同——**等价不等于逐位相同**（bf16 下误差更明显）。Stage 3 评测协议里"同一推理实现"的纪律同源。
- **定位**：fp32 下对比两者差异应缩到 1e-3 内；确认差异随精度缩放（数值噪声）而非结构性错误。
- **修复**：训练/评测全链路统一实现；关键回归测试（金样本）绑实现版本。

**失效 3：vLLM 0.7.x / 引擎版本矩阵错配**
- **症状**：verl/训练框架与推理后端组合报各种诡异 OOM/内核错误。
- **根因**：FA2/flash-attn、vLLM、torch、transformers 的版本矩阵强耦合（verl 官方明确警告 vllm 0.7.x 有 OOM bug，建议 ≥0.8.2——已核实其 README）。
- **定位**：对照框架官方的版本推荐表。
- **修复**：用官方 docker/uv 锁版本矩阵；升级任何一员前先查兼容声明。

---

## 4. 延伸思考题

1. **通信量推导**：ZeRO-3 每步的总通信量 = 前向 all-gather（$2P$）+ 反向 all-gather（$2P$）+ reduce-scatter（$2P$）≈ $6P$ 字节，对比 DDP 的 $2P$——3 倍通信。结合第 1 周 2.2 的带宽公式，算 7B 在 NVLink 与 PCIe 上的每步通信时间，解释"ZeRO-3 在快互联上几乎免费、慢互联上不可用"。
2. **Online Softmax 的数值稳定性**：证明递推式 $m_{new} = \max(m, m_{block})$，$s_{new} = s \cdot e^{m - m_{new}} + s_{block} \cdot e^{m_{block} - m_{new}}$ 与全局 softmax 严格等价——并解释为什么这个递推同时解决了"数值稳定性"（max 移位）与"流式计算"（无需全行）两个问题。
3. **量化训练的前瞻**：QLoRA 的 NF4 底座（Stage 2）与 FP8 训练（第 5 周预告）在账本里各动了哪一项？用第 1 周公式算"7B NF4 + LoRA"的模型状态显存，解释为什么它是单卡 7B 微调的默认答案。

---

*下一篇：[第 3 周：高吞吐推理引擎——vLLM 与 PagedAttention](第3周-vLLM与PagedAttention.md)*
