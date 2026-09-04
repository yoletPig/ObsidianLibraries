# 第 2 周教程：ATC 模型转换与 ACL 离线推理

> **本周要回答的三个问题**
> 1. PyTorch → ONNX → ATC → OM 的转换链每一环在做什么？
> 2. ACL 推理的完整流程（资源初始化 → 加载 → 执行 → 释放）怎么走？
> 3. ATC 转换后精度掉了，怎么一步步定位到具体哪一层？

对应学习计划：第 2 周。交付物：在 310P 上走通「语音模型 → ONNX → OM → ACL 推理」全链路，输出转换前后精度对比与单帧延迟，测 AIPP 开关影响。

---

## 1. 第一性原理：为什么要"转换"而不是直接跑

### 1.1 训练格式 ≠ 推理最优格式

PyTorch 的 `.pt` 是**训练导向**的（动态图、算子通用、保留梯度信息）。推理需要**静态图 + 目标硬件优化**（算子融合、量化、内存规划）。两者目标不同，所以要转换。

### 1.2 转换链全景

$$
\text{PyTorch} \xrightarrow{\text{torch.onnx.export}} \text{ONNX} \xrightarrow{\text{ATC}} \text{OM（.om）} \xrightarrow{\text{ACL}} \text{NPU 推理}
$$

- **ONNX**：中间表示，跨框架通用；
- **OM**：昇腾的离线模型格式，已针对 NPU 编译优化（含图优化、算子选择、量化）；
- **ACL**：运行时，加载 OM 执行推理。

---

## 2. PyTorch → ONNX

### 2.1 导出

```python
import torch

model = load_your_model()          # 如 DeepFilterNet / 小型 ASR encoder
model.eval()

dummy = torch.randn(1, 80, 100)    # 按你的输入形状（例：1 秒 80 维特征）
torch.onnx.export(
    model, dummy, "model.onnx",
    opset_version=17,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch", 2: "time"},
                  "output": {0: "batch"}},
)
```

### 2.2 导出常见坑

1. **动态维度**：语音时长可变，必须声明 `dynamic_axes`，否则导出固定长度；
2. **不支持的算子**：某些控制流/特殊算子 ONNX 不支持 → 简化模型或用等价实现；
3. **精度差异**：导出后用 ONNX Runtime 与 PyTorch 对比，误差应 < 1e-4。

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("model.onnx")
onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
torch_out = model(dummy).detach().numpy()
err = np.max(np.abs(onnx_out - torch_out))
print(f"PyTorch vs ONNX 误差: {err:.2e}")
assert err < 1e-4, "ONNX 导出精度异常"
```

---

## 3. ONNX → OM（ATC）

### 3.1 ATC 命令

```bash
atc --model=model.onnx \
    --framework=5 \                    # 5 = ONNX
    --output=model \                   # 输出 model.om
    --soc_version=Ascend310P3 \        # 按你的 310P 型号
    --input_shape="input:1,80,100" \
    --log=error
```

**关键参数**：
- `--soc_version`：目标芯片（310P3 / 310B / 910B，决定指令集）；
- `--input_shape`：静态化输入形状；
- 量化相关：`--input_format`、`--precision_mode` 等。

### 3.2 动态 shape

若输入时长可变，用动态 shape 配置：

```bash
atc ... \
    --input_shape_range="input:1,80,-1" \   # -1 表示动态维
```

动态 shape 灵活但性能略低于固定形状（无法完全预编译优化）。

### 3.3 量化联动（对接 Stage 1-2 知识）

ATC 支持转换时做 PTQ。对比两种量化路径：

| 路径 | 做法 | 精度 |
| --- | --- | --- |
| ONNX 侧量化 | 先在 ONNX 做量化，ATC 直接转 | 取决于 ONNX 量化 |
| ATC 侧量化 | 转 OM 时由 ATC 用校准数据量化 | 取决于 ATC 校准 |

用你的校准集对比两者精度——这是"量化在哪一层做"的实践认知。

---

## 4. ACL 推理编程（八步流程）

### 4.1 流程骨架

ACL（AscendCL）推理的标准步骤：

```
1. acl.init()                        初始化
2. acl.rt.set_device(device_id)      选择设备
3. acl.rt.create_context()           创建上下文
4. 加载模型: acl.mdl.load_from_file("model.om")
5. 创建输入/输出数据集 (acl.mdl.create_dataset)
6. 拷贝输入数据到 device (acl.rt.memcpy)
7. 执行: acl.mdl.execute(model_id, input, output)
8. 取回结果 + 释放资源 (unload / destroy / reset_device / acl.finalize)
```

### 4.2 Python 接口示例

```python
import acl
import numpy as np

# 1-3. 初始化
acl.init()
acl.rt.set_device(0)
ctx, _ = acl.rt.create_context(0)

# 4. 加载 OM
model_id, _ = acl.mdl.load_from_file("model.om")

# 准备输入
input_np = np.random.randn(1, 80, 100).astype(np.float32)

# 5. 创建数据集与 buffer（简化：实际需按模型 IO 描述申请 device 内存）
# input_size = input_np.nbytes
# input_dev, _ = acl.rt.malloc(input_size, acl.ACL_MEM_MALLOC_HUGE_FIRST)
# acl.rt.memcpy(input_dev, input_size, input_np.ctypes.data, input_size, acl.ACL_HOST_TO_DEVICE)
# ...（完整代码参考 ascend/samples 的 ACL Python 示例）

# 7. 执行
# acl.mdl.execute(model_id, input_dataset, output_dataset)

print("ACL 推理八步流程：见注释骨架，完整实现以官方 sample 为准")
```

**务必对照** `ascend/samples/python/level2_simple_inference` 的完整示例补全内存申请与数据集构造——ACL 的内存管理（host/device 拷贝、malloc 策略）是新手最容易出错的地方。

### 4.3 性能调优要点

- **AIPP**：图像预处理（缩放、归一化、色彩转换）卸载到硬件，省 CPU/减少搬运。语音侧若有特征前处理可类比优化；
- **零拷贝**：减少 host↔device 往返，输入直接驻留 device；
- **多模型多流**：用 stream 并发组织多模型/多请求，隐藏传输延迟。

---

## 5. 精度排查：转换后掉了怎么办（交付核心）

### 5.1 排查流程

```
1. 定位：是量化问题还是算子问题？
   → 先转不量化版本，看精度是否恢复
2. 若量化问题：逐层 dump 对比
   → 找出掉点最严重的层
3. 若算子问题：检查该算子在昇腾的实现/精度模式
```

### 5.2 逐层 dump 对比

```bash
# ATC 支持在转换时插入 dump 点，或开启精度调试
atc ... --precision_mode=must_keep_origin_dtype   # 保精度模式（降速换精度）
```

对掉点层：
- 试 `--precision_mode` 不同档位；
- 该层回退高精度；
- 检查校准数据是否代表该层分布。

### 5.3 交付实验

```
转换前后精度对比表：
| 版本 | 精度指标 | 单帧延迟 |
| PyTorch 原版 | | |
| ONNX（FP32） | | |
| OM（FP32） | | |
| OM（量化） | | |

AIPP 开关延迟对比：
| AIPP | 单帧延迟 |
| 关 | |
| 开 | |
```

---

## 6. 工程权衡与失效模式

### 6.1 权衡

- **静态 vs 动态 shape**：静态快、动态灵活；
- **量化位置**：ONNX 量化与 ATC 量化各有优势，实测对比；
- **精度模式**：保精度模式（must_keep）牺牲速度换精度。

### 6.2 失效模式

1. **算子不支持**：ATC 报某算子无昇腾实现。修复：换等价算子、或用自定义算子（第 5 周技能）补齐。
2. **量化掉点**：某些层量化敏感。修复：逐层定位 + 该层保高精度。
3. **动态 shape 性能差**：编译优化受限。修复：若场景固定，改静态。
4. **ACL 内存泄漏**：未正确释放。修复：严格按八步流程释放资源。

---

## 7. 延伸思考题（含解析）

**Q1**：为什么不能直接在 NPU 上跑 PyTorch 的 .pt？
**A**：.pt 是训练导向的动态图格式，含梯度信息、算子通用未优化。NPU 推理需要静态图 + 硬件特化（算子融合、量化、内存规划）。必须经 ONNX→ATC 转成 OM 才能高效推理。

**Q2**：ATC 转换掉精度，第一步排查什么？
**A**：先判断是量化还是算子问题——转一个不量化版本看精度是否恢复。若是量化，逐层 dump 定位掉点层；若是算子，查该算子昇腾实现或换精度模式。

**Q3**：AIPP 解决什么问题？
**A**：把预处理（缩放/归一化/色彩转换）卸载到专用硬件，减少 CPU 负担与 host↔device 数据搬运，降低端到端延迟。

**Q4**：动态 shape 的代价是什么？
**A**：无法在编译期完全确定形状、做极致优化（内存规划、算子选择受限），性能低于固定形状。输入维度可变时必用，但能固定就固定。

**Q5**：你的语音模型转 OM 后，哪个环节最可能掉精度？
**A**：量化环节（尤其激活含离群值的层，呼应 Stage 1-2）。其次是某些昇腾支持较弱的算子。先关量化复测即可区分。

---

## 本周交付清单

- [ ] 走通 语音模型 → ONNX → OM 转换，精度对比误差 < 1e-4。
- [ ] ACL 推理八步流程跑通，输出推理结果。
- [ ] 输出转换前后精度对比表 + 单帧延迟 + AIPP 开关对比。
- [ ] 掌握精度排查流程（量化/算子二分到逐层定位）。
