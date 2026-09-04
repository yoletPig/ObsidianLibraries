# 第 1 周教程：昇腾生态全景与 CANN 架构

> **本周要回答的三个问题**
> 1. 达芬奇架构的 Cube / Vector / Scalar 三单元各适合什么计算？
> 2. CANN 软件栈怎么分层？它与你熟悉的 CUDA 生态怎么逐项对照？
> 3. 310P 与 910B 的定位差异是什么？你的两块阵地各干什么？

对应学习计划：第 1 周。交付物：① 310P 环境体检（驱动、`npu-smi`、CANN 版本、样例跑通）；② 画一张「CUDA 生态 ↔ 昇腾生态」对照大图。

---

## 1. 第一性原理：为什么需要一块"国产 AI 芯片生态"

### 1.1 生态 = 硬件 + 软件栈 + 工具链 + 社区

一块芯片能否用，不只看算力峰值，更看**软件栈成熟度**：编译器、算子库、框架适配、调试工具、社区资料。NVIDIA 的护城河本质是 CUDA 生态十几年积累。昇腾（Ascend）的目标是构建一套可对标的完整生态——这就是 CANN。

### 1.2 你的学习价值

会写 Ascend C 算子、能迁移模型到昇腾，是当前**最稀缺**的技能之一：国产化替代是确定性趋势，而生态还在建设期，先入者有巨大优势。这也是本 Stage 值得投入八周的原因。

---

## 2. 达芬奇架构：三单元分工

### 2.1 三种计算单元

昇腾的 AI Core 基于**达芬奇架构**，内含三类计算单元：

| 单元 | 擅长 | 对应算子 |
| --- | --- | --- |
| **Cube** | 矩阵乘（高算力密度） | GEMM / 卷积 / Attention 的矩阵部分 |
| **Vector** | 向量逐元素/归约 | 激活函数、Norm、Softmax、Add |
| **Scalar** | 标量控制/分支 | 循环控制、地址计算、条件逻辑 |

**关键直觉**：一个算子常需三者配合——比如 Attention，Cube 做 QK^T 与乘 V，Vector 做 Softmax，Scalar 做循环与地址控制。这与 NVIDIA 的 Tensor Core + CUDA Core 分工类似，但昇腾把"向量单元"提升为一等公民。

### 2.2 与 NVIDIA 的架构对照

| | NVIDIA | 昇腾 |
| --- | --- | --- |
| 矩阵算力 | Tensor Core | **Cube** |
| 标量/通用算力 | CUDA Core | Scalar + Vector |
| 编程接口 | CUDA C++ | **Ascend C** |
| 内存模型 | 显存 + 共享内存 + 寄存器 | HBM + L1/UB（Unified Buffer）+ 寄存器 |

**Unified Buffer（UB）** 类比 CUDA 的共享内存但更大，是 Ascend C 里数据搬运与计算的核心舞台（第 5 周会用到）。

### 2.3 910B 与 310P 的定位

- **910B**：训练卡，大算力、大显存，用于训练/微调与算子开发验证；
- **310P**：推理卡，低功耗、性价比，用于部署与端侧/边缘推理。

你的分工：**本地 310P 练部署**（第 2-3 周），**云上 910B 练训练与算子开发**（第 4-8 周）。

---

## 3. CANN 软件栈分层

### 3.1 自底向上

```
框架层        PyTorch(torch_npu) / MindSpore / TensorFlow
                 │
图引擎层      GE（Graph Engine，图编译与优化）
                 │
算子库层      CANN Kernel（预置算子）+ 自定义算子（Ascend C）
                 │
运行时层      Runtime（内存管理、流、任务调度）
                 │
驱动层        Driver（硬件抽象）
                 │
硬件          Ascend NPU
```

### 3.2 与 CUDA 生态逐项对照（本周交付图的内容）

| 概念/工具 | CUDA 生态 | 昇腾生态 |
| --- | --- | --- |
| 编程模型 | CUDA C++ | Ascend C |
| 运行时 API | CUDA Runtime | **AscendCL**（ACL） |
| 算子库 | cuBLAS / cuDNN | **CANN Kernel / CANN 算子库** |
| 图编译 | TensorRT（推理优化） | **ATC**（模型转 OM）+ GE |
| 推理部署 | TensorRT / Triton | **ACL 推理 / MindIE** |
| Profiling | Nsight（nsys/ncu） | **msprof** |
| 集合通信 | NCCL | **msccl / HCCL** |
| 框架适配 | 原生支持 | **torch_npu**（PyTorch） |
| IDE | — | MindStudio |
| 设备查询 | `nvidia-smi` | **`npu-smi`** |

这张对照表就是你本周要画的大图，也是面试时"讲清两个生态差异"的底稿。

### 3.3 工具链地图

- **ATC**：模型转换器（第 2 周主角），ONNX → OM；
- **msprof**：性能分析（第 4、7 周用）；
- **HCCL/msccl**：多卡集合通信（多机训练时用）；
- **MindStudio**：图形化 IDE（可选）。

---

## 4. 环境体检（交付核心）

### 4.1 310P 体检命令

```bash
# 1. 驱动与固件版本
npu-smi info
# 关注：NPU 型号（310P）、驱动版本、固件版本、健康状态

# 2. CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null \
  || ls /usr/local/Ascend/

# 3. Python 环境
python3 -c "import acl; print('ACL python 可用')" 2>/dev/null || echo "需配置 ACL python"

# 4. 环境变量（CANN 需要）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo $ASCEND_HOME_PATH
```

**预期**：`npu-smi info` 能列出 310P 设备且状态正常；CANN 路径存在。任何一步失败，先对照官方安装文档核对驱动与 CANN 版本**是否匹配**（版本不匹配是昇腾环境最常见的坑）。

### 4.2 跑通官方样例

```bash
# 克隆官方 samples，跑最简单的推理示例
git clone https://gitee.com/ascend/samples.git
cd samples/python/level2_simple_inference/...
# 按各 sample 的 README 准备模型（通常需先用 ATC 转一个 .om）
python3 main.py
```

**目的**：确认"驱动 → CANN → ACL → 推理"整条链路通。这是后续所有实验的前提。

### 4.3 云上 910B 申请

- 华为云 ModelArts 或第三方算力平台租用带 910B 的实例；
- 确认实例已装 CANN + torch_npu；
- 记录申请流程（第 4 周要用）。

---

## 5. 工程权衡与失效模式

### 5.1 权衡

- **版本匹配**：驱动、固件、CANN、torch_npu 版本有严格配套关系，不能各自最新版混搭；
- **本地推理 + 云端训练**：310P 内存小，别强行跑训练；910B 贵，用完即释放。

### 5.2 失效模式

1. **驱动与 CANN 版本不匹配**：`npu-smi` 正常但 CANN 报错。修复：查官方版本配套表，统一版本。
2. **环境变量未加载**：找不到 ACL/ATC。修复：`source set_env.sh` 并写进 shell 配置。
3. **样例跑不通**：模型缺失或算子不支持。修复：按 README 先用 ATC 转模型；查算子支持列表。
4. **310P 内存不足**：加载大模型失败。修复：换更小模型或切云端。

---

## 6. 延伸思考题（含解析）

**Q1**：达芬奇 Cube/Vector/Scalar 各适合什么计算？举具体算子。
**A**：Cube 做矩阵乘（GEMM、卷积、Attention 矩阵部分）；Vector 做逐元素与归约（激活、Norm、Softmax）；Scalar 做控制流与地址计算。一个 Attention 算子就需要三者配合。

**Q2**：昇腾和 NVIDIA 架构的本质差异？
**A**：两者都是"矩阵单元 + 通用单元"思路，但昇腾把向量单元（Vector）提升为一等公民、配大 Unified Buffer，编程模型 Ascend C 更显式地暴露数据搬运与流水；CUDA 更通用、生态更成熟。差异本质在生态成熟度与编程抽象层次。

**Q3**：CANN 里对应 CUDA Runtime 的是什么？对应 TensorRT 的呢？
**A**：AscendCL（ACL）对应 CUDA Runtime（运行时 API）；ATC + GE 对应 TensorRT 的模型优化/编译角色（但 ATC 是转换器，把模型转成 OM）。

**Q4**：为什么 910B 和 310P 要分开用？
**A**：910B 算力/显存大，适合训练与算子开发验证；310P 低功耗、内存小，适合推理部署。在 310P 上强行训练会内存不足且慢，在 910B 上做端侧部署验证又不真实。

**Q5**：什么业务值得投入昇腾适配？
**A**：三维判断：① 国产化/信创要求（政策刚需）；② 成本与供货稳定性（昇腾供应更可控）；③ 生态成熟度（目标模型/算子是否已适配）。有国产化要求且模型在昇腾已适配的业务最值得。

---

## 本周交付清单

- [ ] 310P 体检：`npu-smi info`、CANN 版本、环境变量、样例跑通。
- [ ] 画「CUDA ↔ 昇腾」生态对照大图（覆盖编程模型/运行时/算子库/编译/推理/Profiling/通信/框架）。
- [ ] 记录云上 910B 的申请途径。
- [ ] 能闭卷讲清三单元分工与 910B/310P 定位。
