# 第 4 周教程：RKNN 自定义算子与 MTK-NeuroPilot——补上 NPU 的算子缺口

> **本周要回答的问题**
> 1. NPU 不支持的算子怎么办？「CPU 自定义算子 + NPU 主体」混合执行的调用链是什么？
> 2. `rknn_custom_op` 的注册流程是什么？NCHW/NHWC 与量化参数透传要注意什么？
> 3. 自定义算子落 CPU 后，数据搬运成本怎么算？什么时候必须写？
> 4. MTK NeuroPilot 工具链（.tflite → APU）与 RKNN 流程的异同？

对应学习计划：第 5 周。交付物：① 在 RKNN 上实现一个自定义算子（如 RMSNorm），混合部署并验证正确性；② MTK 板完成同一模型的 NeuroPilot 转换与推理；③ 输出「RK3588 vs MTK」同模型对比表。

---

## 1. 第一性原理：NPU 是"算子白名单"硬件，自定义算子是逃生门

### 1.1 为什么总会撞上算子缺口

NPU 编译器只认固定的算子集（Conv/MatMul/常见激活等）。你的语音模型随时可能带上"新东西"：CIF 积分发射、flow matching 的 ODE 步、自定义归一化。转换报错只是表象，本质是**硬件的确定性换来了算子集的封闭性**。

逃生门：**混合执行**——模型图中不支持的算子"落"CPU（用你自己的实现），其余部分仍在 NPU 上跑。运行时在两种执行域之间切换，数据经共享内存传递。

### 1.2 rknn_custom_op 的注册与执行链

```
转换期：标记目标算子为 custom（op_type 与自定义名一致）
         ─► .rknn 图中该节点是"黑洞"，只记录输入/输出张量描述
运行期：推理到该节点 ─► 回调你的 compute 函数（CPU）
         ─► 你把输出写回，NPU 继续下游节点
```

注册三要素：① **op 描述**（名字、输入输出个数与形状推导）；② **compute 回调**（拿到输入指针、算、写输出）；③ **内存与格式约定**——两个高频坑：

- **布局**：NPU 内部常用 NHWC，框架是 NCHW——自定义算子拿到的可能是 NHWC，按错布局读数据结果全错；
- **量化参数透传**：若自定义算子的输入是量化张量（INT8 + scale/zp），你的 CPU 实现要么先反量化成浮点算、要么在整数域实现并保持 scale 语义——**忘了透传 = 输出数值错一个 scale 倍数**。

### 1.3 搬运成本模型（面试高频）

设自定义算子输入/输出合计 $B$ 字节，每秒调用 $f$ 次：

$$
t_{\text{copy}} \approx \frac{2B}{BW_{\text{NPU↔CPU}}}, \qquad \text{占比} = t_{\text{copy}} \cdot f \ /\ t_{\text{frame}}
$$

判定：搬运占比 < 5% 可接受；> 20% 要考虑替代方案（改写模型结构绕开该算子、或把相邻算子一起拉下 CPU 减少往返次数）。**"自定义算子的代价不在计算，在搬运"**——这是本节的核心结论。

### 1.4 MTK NeuroPilot：平行宇宙的同构流程

MTK（Genio 系列）的链路：

$$
\text{模型} \xrightarrow{\text{TFLite}} \xrightarrow[\text{量化}]{\text{NeuroPilot Converter}} \text{.tflite(APU 优化)} \xrightarrow{\text{APU Runtime}} \text{APU}
$$

与 RKNN 的对应关系（记忆锚点）：

| 环节 | RKNN | NeuroPilot |
| --- | --- | --- |
| 中间格式 | ONNX | TFLite（也支持 ONNX 转换） |
| 转换工具 | rknn-toolkit2 | NeuroPilot Converter / ai-devtool |
| 量化 | 非对称 + 校准集 | INT8/INT16 + 校准；量化选项类似 |
| 运行时 | rknn_runtime（C API） | NeuroPilot SDK（C/C++ API） |
| 异构调度 | core_mask 核绑定 | APU/GPU/CPU 调度策略（按算力分配） |

差异点：MTK 走 TFLite 生态（算子集以 TFLite 为准），自定义算子路径不同（TFLite custom op 注册机制）；Genio 的 APU 算力按 TOPS 分档，转换时要指定目标 APU 能力等级。

---

## 2. 实现与验证（本周交付核心）

### 2.1 自定义算子实战：RMSNorm（RKNN）

**步骤 1：转换侧标记自定义算子**（Python，x86 主机）

```python
# build_custom_rmsnorm.py —— 把模型中的 RMSNorm 替换为自定义节点
from rknn.api import RKNN
rknn = RKNN()
rknn.config(target_platform="RK3588")
rknn.load_onnx(model="model_with_rmsnorm.onnx",
               # 关键：声明该算子走自定义实现（按工具链版本语法填写节点名）
               )
rknn.build(do_quantization=True, dataset="calib_list.txt")
rknn.export_rknn("model_custom.rknn")
```

**步骤 2：C 侧注册与实现**（板端，骨架代码）

```c
// custom_rmsnorm.c —— 逐行注释版（真实签名以 rknn_custom_op.h 为准）
#include "rknn_api.h"
#include <math.h>

// ① 计算回调：拿到输入张量，算完写输出
static int rmsnorm_compute(RKNNCustomOpContext* ctx,
                           RKNNCustomOpTensor* inputs,
                           RKNNCustomOpTensor* outputs) {
    // ② 读描述：形状、数据类型、量化参数（别忘！）
    int n = inputs[0].attr.n_elems;          // 元素总数
    float* x = (float*)inputs[0].buf;        // 若为 INT8 需先按 scale 反量化
    float* y = (float*)outputs[0].buf;
    float eps = 1e-6f;

    // ③ RMSNorm 本体：平方和 → 均值 → rsqrt → 缩放
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * inv;   // 增益 g 可作为第 2 个输入
    return 0;
}

// ④ 注册表：名字必须与转换期标记的一致
RKNNCustomOp rmsnorm_op = {
    .op_type = "CustomRMSNorm",       // 与转换侧对齐
    .compute = rmsnorm_compute,
    // ... 形状推导/初始化回调按头文件补全
};
// 运行时：rknn_register_custom_ops(&rmsnorm_op, 1) 后再 init_runtime
```

**步骤 3：正确性验证**（交付的断言）

```python
# verify_custom.py —— 同输入对比：浮点参考 vs 板端自定义执行
import numpy as np
from rknnlite.api import RKNNLite          # 板端
def ref_rmsnorm(x, eps=1e-6):
    return x / np.sqrt((x**2).mean(-1, keepdims=True) + eps)

x = np.load("test_input.npy")
rknn = RKNNLite(); rknn.load_rknn("model_custom.rknn")
rknn.init_runtime()
y_board = rknn.inference(inputs=[x])[0]
y_ref = ref_rmsnorm(x)                      # 端到端参考（略去其余层时用单层图验证）
cos = (y_board.flatten() @ y_ref.flatten()) / (
      np.linalg.norm(y_board) * np.linalg.norm(y_ref))
assert cos > 0.999, f"自定义算子输出与参考不一致，cos={cos:.4f}"
print(f"自定义 RMSNorm 验证通过 ✓（cos={cos:.5f}）")
```

### 2.2 MTK 侧同模型转换（对照实验）

```bash
# NeuroPilot Converter（x86 主机，按官方 SDK 版本）
# 1) 模型 → TFLite（量化后）
# 2) converter 指定目标 Genio 平台与 APU 能力，产出 APU 优化模型
# 3) 板端用 NeuroPilot C API 推理，记录延迟
# 交付：把与 §2.1 相同的模型与相同输入跑一遍，作为对比表的右半边
```

### 2.3 「RK3588 vs MTK」对比表（交付）

| 维度 | RK3588（NPU） | MTK（APU） |
| --- | --- | --- |
| 同一模型延迟 | _实测_ | _实测_ |
| 转换掉点（端到端指标） | _实测（第 3 周数据）_ | _实测_ |
| 自定义算子支持 | C 回调，本周已验证 | TFLite custom op（记录流程差异） |
| 工具链坑点记录 | _你的实录_ | _你的实录_ |

---

## 3. 工程权衡与失效模式

### 3.1 权衡

- **自定义算子的粒度**：算子越小、调用越频繁，搬运占比越高——有时把"一串小算子"打包成一个自定义算子反而更快（一次往返干多件事，融合思想在异构场景的复用）。
- **CPU 实现的优化责任**：落 CPU 的算子没人帮你做 SIMD 特化——热点自定义算子要自己上 NEON/AVX（第 2 周技能复用）。
- **平台锁定成本**：RKNN 与 NeuroPilot 的转换脚本、自定义算子互不通用——若产品要双平台，把"模型本体（ONNX）"作为唯一事实源，平台产物全部脚本化再生成。

### 3.2 失效模式

1. **布局错乱**：症状——自定义算子输出是"有规律的乱码"（通道错位）；根因——NPU 给的是 NHWC、按 NCHW 读了；修复——打印输入 attr 的实际布局，转换期强制统一。
2. **量化参数没透传**：症状——输出整体偏大/偏小固定倍数；根因——把 INT8 输入当数值直接用，没乘 scale；修复——读张量量化参数，先反量化或整数域实现。
3. **搬运吃掉收益**：症状——加了自定义算子后比纯 CPU 还慢；根因——每帧多次小块往返；修复——合并算子、或用 §1.3 的成本模型先算账再动手。
4. **转换工具与板端运行时版本错配**：症状——加载即崩或结果错；根因——厂商工具链版本矩阵严格；修复——锁定版本组合并写进环境文档（RKNN 与 MTK 都有此坑）。

---

## 4. 延伸思考题（含解析）

**Q1**：什么时候"必须"写自定义算子，什么时候应该改写模型绕开它？
**A**：算子处于热路径且搬运成本可接受 → 自定义算子；算子只在离线/低频路径 → 可忍；算子引发高频小块往返、或实现复杂度过高 → 优先改写模型结构（用支持的算子等价替换）。先算搬运账，再谈实现——顺序不能反。

**Q2**：自定义算子的 CPU 实现为什么常常要自己写 SIMD？
**A**：NPU 工具链只对图内算子做编译优化；你的回调函数就是普通 C 代码，编译器不会替你调 NEON 内联。它在热路径上，性能就是你的责任——这是"混合执行"的隐性成本。

**Q3**：为什么两个平台都坚持从 ONNX 出发？
**A**：ONNX 是训练框架与所有部署工具链的最大公约数。以它为唯一事实源，任何平台的产物都可脚本化再生成——平台锁定只锁住转换脚本，锁不住模型资产。这是部署工程的可移植性原则。

**Q4**：RMSNorm 落 CPU 真的划算吗？什么情况下它应该留在 NPU？
**A**：RMSNorm 计算轻、但每层每帧都调用——若张量大（长序列），搬运可能超过计算。若 NPU 支持等价结构（多数支持：Mul+ReduceMean+Rsqrt+Mul 可编译融合），就该留 NPU；只有当它带特殊变体（如你的模型自定义的归一化）时才落 CPU。**能用原生算子拼出来，就不要自定义**。

**Q5**：对比表的"转换掉点"一列，如果 MTK 明显优于 RK3588，可能的原因有哪些？
**A**：① 量化策略差异（校准算法、敏感度处理）；② 算子融合粒度不同（融合越多中间量化误差越少）；③ 目标平台配置是否指定正确（错误的算力档位会换内核）。排查顺序：先确认配置正确，再看逐层对比，最后才是算法差异——别把配置错误当成平台差距。

---

## 5. 双板实战记录模板（两板都要实操的证据链）

### 5.1 环境登记表（每次实验前填）

| 项 | RK3588 板 | MTK 板 |
| --- | --- | --- |
| 系统/固件版本 | _填_ | _填_ |
| 工具链版本 | rknn-toolkit2 _x.y.z_ | NeuroPilot SDK _x.y_ |
| runtime 版本 | _填（必须与工具链匹配）_ | _填_ |
| 内存/存储 | _填_ | _填_ |
| 散热条件 | 被动/风扇 | 被动/风扇 |

### 5.2 同模型双板对比的最小实验设计

```
同一模型（语音方向成果，如 DeepFilterNet3 或小型 VAD）：
① 各自工具链转换（INT8，同校准集）
② 同输入推理 100 次：记录 平均延迟 / P99 延迟 / 吞吐
③ 端到端指标：降噪用 PESQ，VAD 用 F1（与 FP32 参考比）
④ 功耗：稳态推理时的板级功率（若有测量条件）
⑤ 记录"最痛的一个坑"：各自写 3 行（现象/根因/解法）
```

**纪律**：⑤ 的踩坑记录是最有面试价值的部分——"我在 RK3588 上遇到 X，定位到 Y，用 Z 解决"这种一手经验，远比背文档有说服力。

### 5.3 自定义算子的性能验收清单

写完自定义算子只验证"正确"还不够，还要验收"不拖后腿"：

| 验收项 | 方法 | 通过标准 |
| --- | --- | --- |
| 正确性 | §2.1 的余弦断言 | cos > 0.999 |
| 搬运占比 | §1.3 成本模型实测 | < 5%（< 20% 可接受） |
| CPU 占用 | `perf top` 看回调函数占时 | 不在 Top3 之外失控 |
| 与纯 NPU 全图对比 | 同模型改写成原生算子版 | 混合版延迟增幅 < 15% |

---

## 本周交付清单

- [ ] 自定义算子三件套完成：转换标记 + C 实现 + 正确性断言（cos > 0.999）。
- [ ] 验证布局与量化参数两个坑（故意写错一版，记录症状——面试素材）。
- [ ] 用 §1.3 成本模型为你的算子算一次搬运占比，写进实验记录。
- [ ] MTK 板完成同模型转换与推理，填完「RK3588 vs MTK」对比表。
- [ ] 把两平台的工具链版本组合与踩坑点写成一页笔记（双板实操的证据链）。
