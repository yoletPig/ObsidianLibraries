# 第 5 周教程：性能诊断 Profiling 与量化/内核加速

> **本周要回答的三个问题**
> 1. 怎么用 PyTorch Profiler / Nsight Systems 区分 Compute-Bound 与 Memory-Bound？
> 2. AWQ、GPTQ、FP8、SmoothQuant 的机理差异与选型依据是什么？
> 3. 一份合格的"系统性能诊断与优化报告"长什么样？

对应学习计划：第 5 周。交付物：PyTorch Profiler 抓取 VLM 推理 Trace（Chrome `chrome://tracing` 打开），定位 Top 3 CUDA Kernels；开启 AWQ/FP8 量化，报告前后吞吐与显存缩减比，产出 2 页纸诊断报告。

**本篇是全阶段的方法论收束**：第 1-4 周的所有技术（ZeRO/FA/PagedAttention/Radix）本质上都是"对某个已知瓶颈的对症药"——本篇教你**在没有先验答案时，如何自己找到瓶颈**。

---

## 1. 第一性原理：瓶颈判定的第一性判据

### 1.1 Roofline：一个算子到底卡在哪

任一 GPU 算子的两个属性：

- **算术强度** $I = \frac{\text{FLOPs}}{\text{显存读写字节数}}$（每字节带宽支撑多少计算）；
- 硬件的两个峰值：计算峰值 $P_{\text{compute}}$（如 A100 bf16 ≈ 312 TFLOPS）与带宽峰值 $P_{bw}$（如 A100 ≈ 2TB/s）。**脊点** $I^* = P_{\text{compute}} / P_{bw}$（A100 bf16 ≈ 156 FLOP/Byte）。

$$
\text{可达性能} = \min(P_{\text{compute}},\ I \times P_{bw})
$$

- $I < I^*$ → **Memory-Bound**（带宽受限）：性能由 HBM 带宽决定，加算力无用；
- $I > I^*$ → **Compute-Bound**（算力受限）：性能由张量核决定，加带宽无用。

**记住几个典型算子的归类**（这是 Profiler 读数时的"地图"）：

| 算子 | 算术强度 | 归类 | 对症药 |
| --- | --- | --- | --- |
| 大 GEMM（大 batch 矩阵乘） | 高（数百） | Compute-Bound | 张量核、精度（FP8）、tile 效率 |
| **Decode 单 Token GEMV** | **≈ 2（每权重只乘一次）** | **强 Memory-Bound** | **batch 摊薄（continuous batching）、量化降权重字节** |
| 传统 softmax/attention 中间量 | 低 | Memory-Bound | FlashAttention（第 2 周） |
| Element-wise/ Layernorm | 极低 | Memory-Bound | 算子融合（fused kernel） |

**为什么 decode 量化收益巨大而 prefill 收益小**：decode 的 GEMV 强度 ≈ 2，远低于脊点——时间 ≈ 权重字节数 ÷ 带宽，权重砍一半（INT4）时间近乎减半；prefill 的大 GEMM 是 compute-bound，权重字节数不是瓶颈，INT4 收益主要在显存而非速度。**这条判据直接决定第 3 节量化收益的解读。**

### 1.2 Profiler 的两个层次

- **PyTorch Profiler**：算子级时间线（CPU op + CUDA kernel），导出 Chrome Trace（`chrome://tracing` 可视化），适合"哪个 kernel 最耗时"的定位——本周 MVP 的工具；
- **NVIDIA Nsight Systems（Nsys）**：更底层——kernel 执行、显存拷贝、NCCL 通信、CPU-GPU 空隙全时间线，适合"GPU 利用率为什么低"（gap 在哪：同步等待？数据搬运？）。

判读流程：先 Profiler 找 Top Kernels → 判断它们该不该耗时这么长（对照 1.1 的地图）→ 再用 Nsys 看 kernel 之间的空隙（空闲 = 调度/同步/数据问题，而非算子问题）。

---

## 2. 量化技术：给权重与激活"减字节"

### 2.1 三条技术路线的机理与选型

| 路线 | 代表 | 量化对象 | 精度损失 | 速度收益来源 | 适用 |
| --- | --- | --- | --- | --- | --- |
| **Weight-Only** | **AWQ**、GPTQ | 权重 → INT4/INT8 | 极小（AWQ 有激活感知保护） | decode（memory-bound GEMV 的字节数减半/四分之一） | **单请求低 batch 推理（主流默认）** |
| **Weight+Activation** | FP8 (W8A8)、SmoothQuant (INT8) | 权重+激活 → FP8/INT8 | 小（FP8）/ 中（INT8 需校准） | prefill 大 GEMM 的张量核翻倍（H100 FP8） | 高 batch 服务、Hopper 硬件 |
| **KV Cache 量化** | FP8 KV Cache | 缓存 | 小 | KV 显存减半 → 并发容量翻倍（第 3 周账本直接受益） | 长上下文高并发 |

机理要点：

- **AWQ**（*AWQ: Activation-aware Weight Quantization*, MLSys 2024）：观察"少数关键权重通道对应激活的大值"，量化时对这些通道做缩放保护——**按激活分布加权保护权重**，4-bit 下精度损失远小于朴素 RTN；
- **GPTQ**（*GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, ICLR 2023）：基于二阶信息（Hessian 近似）逐层最小化量化误差，逐列更新补偿；
- **FP8（W8A8）**：E4M3/E5M2 浮点格式（Stage 2 第 4 周位分配分析的推理版），硬件原生支持（H100/H200），prefill 大 GEMM 直接翻倍吞吐；
- **SmoothQuant**：把激活的离群值"平滑"迁移到权重上（$\text{act}/s \cdot W \cdot s$ 数学等价变换），使两侧都进入 INT8 可表示范围。

选型决策路径：**单请求/低并发、显存是痛 → AWQ-INT4（KV 不量化）；高并发服务、Hopper 卡 → FP8 W8A8 + FP8 KV；长上下文并发 → FP8 KV Cache；消费卡部署大模型 → AWQ/GPTQ-INT4**。

### 2.2 量化收益的正确解读框架

用第 1 周的账本语言预判收益，再用实测验证：

- **显存**：权重字节按比例降（INT4 ≈ 1/4 of bf16）——账本硬收益，实测必然兑现；
- **decode 吞吐**：memory-bound 下近似随权重字节比提升（INT4 理论 2~3 倍，实测打折）；
- **prefill 吞吐**：weight-only 量化几乎无收益（compute-bound）——**没有 prefill 提升却报"整体快 4 倍"的报告可以合上了**；
- **精度**：必须跑基准回归（AWQ 论文口径的通用任务损失 ~1% 内，但你的领域任务要自己测——Stage 3 评测纪律的应用）。

---

## 3. 实现与验证

### 3.1 PyTorch Profiler 抓 Trace + Top Kernels

```python
"""
VLM 推理 Profiling: 抓 Chrome Trace + 打印 Top CUDA Kernels。
运行方式: python stage8_week5_profile.py --out trace.json
依赖: torch, transformers, pillow
"""
import argparse
import torch
from torch.profiler import profile, ProfilerActivity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--out", default="trace.json")
    args = ap.parse_args()
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from PIL import Image

    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).cuda().eval()
    processor = AutoProcessor.from_pretrained(args.model)
    img = Image.new("RGB", (448, 448))
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "详细描述图片。"}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[img], return_tensors="pt").to("cuda")

    # 预热 (排除编译/缓存噪声)
    for _ in range(2):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=64)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=128)

    prof.export_chrome_trace(args.out)
    print(f"Trace 已导出: {args.out}  (Chrome 打开 chrome://tracing 拖入查看)")

    # Top CUDA Kernels (按 CUDA 时间排序)
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=8,
        max_name_column_width=60))


if __name__ == "__main__":
    main()
```

**预期输出形态**（Top kernels 表，Llama/Qwen 系典型构成）：

```text
Trace 已导出: trace.json
---------------------------------  -------  -------
Name                               CUDA %   Calls
---------------------------------  -------  -------
gemm/gemm_variant_kernel (矩阵乘)    61.2%    3,842
flash::flash_fwd_kernel              18.7%      512
vectorized_elementwise (逐元素)       6.3%    9,107
rms_norm_kernel                       3.1%      512
... (采样/softmax/拷贝等)             10.7%
---------------------------------  -------  -------
```

**判读训练**：GEMM（含 decode 的 GEMV）占 60%+ 正常（decode 每步 2 次大 GEMV × 每层，累计最多）；flash kernel 是注意力；elementwise 占比 >10% 时是**算子融合的机会**（fused kernel / torch.compile）；拷贝类 kernel（`Memcpy DtoH` 等）占比高说明有 CPU-GPU 数据搬运问题。Chrome tracing 里还要看**kernel 之间的 gap**——大量空隙说明是调度/同步瓶颈而非算子瓶颈（此时换更快的 kernel 无用，要看 Nsys 找空隙来源）。

### 3.2 AWQ 量化前后对比

```python
"""AWQ 量化前后: 显存与吞吐对比 (vLLM 加载两种精度)"""
from vllm import LLM, SamplingParams
import time, torch

def bench(llm, n=32):
    sp = SamplingParams(max_tokens=256, temperature=0)
    prompts = [f"请详细解释原理 {i}。" for i in range(n)]
    t0 = time.perf_counter()
    llm.generate(prompts, sp)
    dt = time.perf_counter() - t0
    mem = torch.cuda.max_memory_allocated() / 1024**3
    return dt / n, mem

if __name__ == "__main__":
    torch.cuda.reset_peak_memory_stats()
    llm_bf16 = LLM(model="Qwen/Qwen2.5-7B-Instruct",
                   gpu_memory_utilization=0.85, max_model_len=4096)
    t_bf16, m_bf16 = bench(llm_bf16)
    del llm_bf16; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    llm_awq = LLM(model="Qwen/Qwen2.5-7B-Instruct-AWQ",
                  quantization="awq", gpu_memory_utilization=0.85, max_model_len=4096)
    t_awq, m_awq = bench(llm_awq)

    print(f"bf16 : {t_bf16*1000:.0f} ms/req, 峰值 {m_bf16:.1f} GB")
    print(f"AWQ-4: {t_awq*1000:.0f} ms/req, 峰值 {m_awq:.1f} GB")
    print(f"吞吐比 {t_bf16/t_awq:.2f}x, 显存比 {m_bf16/m_awq:.2f}x")
    # 精度回归 (别省): 用同 prompt 集对比两模型的答案一致性/基准分
```

**预期形态**（7B、A100、批量 32，以实测为准）：显存约 **3.5~4×缩减**（5.5GB vs 15GB 量级权重）；吞吐提升 **1.5~2.5×**（decode memory-bound 的收益，prefill 部分摊薄）；精度回归必须做（领域 50 题 + 通用基准抽样，Stage 3 协议）。

### 3.3 两页纸诊断报告模板

```markdown
# VLM 推理性能诊断与优化报告
环境: A100-80G / vLLM <ver> / torch <ver> / 模型 7B bf16 / 负载: 448px 图 + 50~500 Token 出

## 1. 瓶颈定位 (证据)
Profiler Top3: GEMM 61% (decode GEMV, memory-bound) / flash 19% / elementwise 6%。
Nsys: decode 段 GPU util 高但 SM 占用低 -> memory-bound 确认; kernel 间隙 <3% (无调度问题)。
结论: 瓶颈在 decode 权重带宽, 而非调度/注意力。

## 2. 干预与收益 (对照实验, 同负载同指标)
| 措施 | 显存峰值 | QPS | ITL p50 | 备注 |
|---|---|---|---|---|
| 基线 bf16 | 48.2 GB | 3.1 | 28ms | — |
| AWQ-INT4 | 14.9 GB | 5.8 | 17ms | 通用基准掉 0.8pt (可接受) |
| 基线+FP8 KV | 33.5 GB | 4.9 | 25ms | 长上下文并发容量 x1.8 |
| AWQ + FP8 KV | 11.2 GB | 7.4 | 14ms | 组合拳 |

## 3. 残留问题与下一步
- elementwise 6%: 尝试 torch.compile 融合 (预估 -3% 耗时);
- VLM 视觉编码在 TTFT 占 38%: 待测图像分辨率降级的影响;
- 未做: FP8 W8A8 (硬件不支持 FP8, 需 Hopper)。

## 4. 复现
脚本/参数/种子/数据集 hash 附后; 所有对比同环境同日完成。
```

报告纪律：**每个结论有 Trace/实测证据；每个数字有对照组；残留问题诚实列出**——这份模板在第 5 周交付，也是 Stage 9/10 系统工作的通用格式。

---

## 4. 工程权衡与失效模式

### 4.1 诊断流程决策树

```text
性能不达标
├─ GPU util 低? → 是: 调度/数据问题 (Nsys 找 gap: CPU 预处理? 同步? 通信?)
└─ util 高但慢 → 看 kernel 构成
    ├─ GEMM 占主导 → decode? → memory-bound → 量化/batch 摊薄
    │                → prefill? → compute-bound → FP8/更优 tile/并行
    ├─ attention 占主导 → FA 是否开启/版本; 序列是否过长
    ├─ elementwise/拷贝高 → 算子融合/减少搬运
    └─ 大量小 kernel + gap → CPU 瓶颈 → 编译/融合/异步化
```

### 4.2 三个代表性失效模式

**失效 1：量化在"错误的负载"上测收益**
- **症状**：AWQ 标称 2~3×提速，实测只有 1.1×，得出"量化无用"。
- **根因**：测试负载是长 prompt 短输出——prefill 主导且 compute-bound，量化（memory-bound 药方）打错了靶子。
- **定位**：把负载拆成 TTFT（prefill）与 decode 时间分别对比——收益应在 decode 段。
- **修复**：按负载形态选方案（长输入场景 FP8 W8A8/张量并行的 prefill 优化更对口）。

**失效 2：Profiler 数据被预热与采样噪声污染**
- **症状**：两次 Profile 的 Top kernel 排序不一致。
- **根因**：未预热（首步含 CUDA context/kernel JIT）；CUDA 采样率（`with_stack`/activities 配置）引入开销改变了时间分布。
- **定位**：对比预热前后；调 `schedule`（wait/warmup/active 三段）只取稳态窗口。
- **修复**：预热 ≥2 步；多轮取中位数；Profiler 开销本身计入报告声明。

**失效 3：量化后不做精度回归——省了显存赔了业务**
- **症状**：INT4 部署后，长尾查询（生僻实体/复杂指令）错误率上升，通用基准几乎无差。
- **根因**：通用 benchmark 平均掉的精度损失，在长尾/领域分布上放大（Stage 3 "画像而非总分"的量化版）。
- **定位**：领域 held-out 集 + 量化前后逐题 diff，找出恶化集中的题型。
- **修复**：领域数据校准（AWQ 的校准集换成领域样本）；恶化集中的请求路由到高精度模型（混合部署）。

---

## 5. 延伸思考题

1. **Roofline 实战**：用 1.1 的公式算"7B decode GEMV（batch=1）"与"prefill GEMM（batch=32 prefill 2k Token）"的算术强度，标到 A100 的 Roofline 图上；再算 batch=64 的 decode——解释为什么 continuous batching 把 decode 从"强 memory-bound"拉向"接近 compute-bound"，以及这条曲线的尽头在哪（batch 大到算力饱和后，继续加并发的边际收益趋零，转而被 KV 显存限制——接第 3 周 PagedAttention 的价值）。
2. **量化与 RL 的交叉**：Stage 7 的 GRPO 训练里 rollout 占一半以上算力——rollout 引擎用 AWQ 量化模型采样，对 RL 训练的正确性有什么风险？（提示：采样分布与训练分布的引擎级 mismatch——第 2 周失效 2 的"数值等价不等于逐位相同"在 RL 语境被放大；verl 社区的 vexact（零失配 rollout）项目正是针对此，已核实存在。）
3. **写一份你自己的报告**：取本周任意一个干预（FA2/量化/chunked prefill/前缀缓存），按 3.3 模板在你环境里完成完整闭环——本阶段所有教程的最终检验，就是你能不能生产一份"有对照组、有证据链、有残留问题"的诚实报告。

---

*下一篇：[阶段八自测验收与复盘](阶段八自测验收与复盘.md)*
