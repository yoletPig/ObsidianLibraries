# 第 4 周教程：torch_npu 训练迁移与精度排查

> **本周要回答的三个问题**
> 1. 一段在 CUDA 上跑的训练脚本，迁到 910B 需要改哪几行？
> 2. 迁移后精度对不齐，系统化的排查路径是什么？
> 3. 哪些算子会"悄悄" fallback 到 CPU？为什么这是性能杀手？

对应学习计划：第 4 周。交付物：云上 910B 完成一次「CUDA 脚本 → torch_npu」迁移（用你的 LoRA 微调脚本），输出训练速度/loss 对比，记录 ≥3 个踩坑。

---

## 1. 第一性原理：torch_npu 在 PyTorch 里动了什么

### 1.1 device 抽象的可扩展性

PyTorch 把"设备"做成可扩展的抽象。`torch_npu` 通过注册机制把 `npu` 变成一种新设备类型，于是：

```python
import torch
import torch_npu                     # 导入即注册 'npu' 设备

x = torch.randn(4, 4).to("npu")      # 张量上 NPU
y = x @ x.T                          # 算子在 NPU 上执行（走 CANN）
```

**你只需导入 `torch_npu` 并把 `.cuda()` 换成 `.npu()`**——这是迁移的第一板斧。

### 1.2 算子的三条去路

调用一个算子时，torch_npu 有三种可能：

1. **昇腾有实现** → 在 NPU 上跑（理想）；
2. **昇腾无实现** → **fallback 到 CPU** 计算，再搬回（性能杀手）；
3. **算子映射到等价实现** → 行为可能微调。

**fallback 是隐蔽的性能杀手**——代码照常跑，但某些算子在 CPU 上慢几十倍，还夹杂大量 NPU↔CPU 数据搬运。

---

## 2. 迁移三板斧

### 2.1 第一斧：device 改造

```python
# 迁移前（CUDA）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 迁移后（NPU）—— 推荐写法：环境无关
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_npu
        if torch.npu.is_available():
            return torch.device("npu")
    except ImportError:
        pass
    return torch.device("cpu")

device = get_device()
model = model.to(device)
```

其余改动：`.cuda()` → `.to(device)`、`torch.cuda.amp` → 对应 NPU 混合精度 API、CUDA 特有的同步/计时调用替换。

### 2.2 第二斧：精度对齐

迁移后第一件事不是跑全量训练，而是**对齐验证**：

```python
# 同一输入、同一初始化，分别在 CUDA 与 NPU 前向，对比输出
torch.manual_seed(42)
x = torch.randn(2, 8, 64)

out_cuda = model(x.to("cuda")).detach().cpu()
out_npu  = model(x.to("npu")).detach().cpu()

err = (out_cuda - out_npu).abs().max().item()
print(f"前向误差: {err:.2e}")
# FP32 下应 < 1e-4；混合精度下放宽到 1e-2 量级并关注趋势
```

**逐层对齐**（误差大时）：在关键层后插钩子，逐层比较，定位第一个发散的层。

### 2.3 第三斧：性能调优

1. **开混合精度**（BF16/FP16）：昇腾对 BF16 支持好；
2. **打开融合算子**：torch_npu 提供的融合接口；
3. **检查 fallback**：确认没有算子掉到 CPU；
4. **用 msprof 看**：NPU 利用率、算子耗时分布。

---

## 3. 精度问题排查（交付核心技能）

### 3.1 排查决策树

```
精度不对齐
├─ 是数值差异还是功能错误？
│   ├─ 功能错误（结果完全错）→ 算子不支持/行为差异 → 查算子映射表
│   └─ 数值微差 → 正常浮点差异，放宽阈值
├─ 差异随训练放大？
│   ├─ 是 → 某算子精度不足 → 逐层定位 + 该算子开高精度模式
│   └─ 否 → 可接受
└─ 只在混合精度下出现？
    └─ 检查溢出/下溢 → 换 BF16（范围更大）或开 loss scaling
```

### 3.2 BF16 vs FP16 在昇腾上

- **BF16**：指数位多、范围大、不易溢出，昇腾推荐，训练更稳；
- **FP16**：尾数位多、精度高但范围小，易上下溢，需 loss scaling。

**经验**：昇腾训练优先 BF16。

### 3.3 检测 fallback

```python
# torch_npu 提供的分析手段（以实际版本接口为准）：
# 1. 环境变量开启算子日志，观察哪些算子未在 NPU 执行
# 2. msprof 抓 trace，看是否有 CPU 算子夹杂
# 3. torch_npu 的算子支持查询接口（如有）
```

发现 fallback 算子 → 换等价实现 / 用支持的算子替代 / （必要时）写自定义算子。

---

## 4. 实战：迁移一次 LoRA 微调（交付）

### 4.1 流程

```bash
# 云上 910B 实例，确认环境
python -c "import torch, torch_npu; print(torch.npu.is_available())"   # True
npu-smi info
```

### 4.2 迁移你的微调脚本

拿你用过的微调脚本（如 Qwen2.5 LoRA），按三板斧改造：

```python
import torch_npu
# device 改为 npu；混合精度用 BF16
# 其余训练循环基本不动

# 训练时用 msprof 观察
```

### 4.3 对比记录

```
| 指标 | CUDA (GPU) | NPU (910B) |
| 单步耗时 | | |
| 样本吞吐 | | |
| 最终 loss | | |
| loss 曲线趋势 | 一致? | |
```

**关键验证**：两边 loss 曲线趋势一致、最终精度相当——证明迁移正确（绝对值因浮点差异可微差）。

### 4.4 踩坑清单（交付 ≥3 条）

按四类记录：

```
环境类：如版本不匹配 → 解法
算子类：如某算子 fallback → 解法
精度类：如混合精度溢出 → 换 BF16
性能类：如数据加载成瓶颈 → 调 num_workers/预取
```

---

## 5. 工程权衡与失效模式

### 5.1 权衡

- **BF16 vs FP16**：BF16 稳、FP16 精，昇腾优先 BF16；
- **兼容写法**：写环境无关的 device 选择，代码两边都能跑；
- **融合算子**：提速但绑定昇腾，权衡可移植性。

### 5.2 失效模式

1. **隐蔽 fallback**：某算子掉 CPU，训练极慢。诊断：算子日志/trace；修复：换支持算子。
2. **混合精度溢出**：FP16 下 loss 变 NaN。修复：换 BF16、开 loss scaling。
3. **数据加载瓶颈**：NPU 算得快但数据喂不上。修复：加 `num_workers`、预取、提前处理数据。
4. **版本不匹配**：torch 与 torch_npu 版本错。修复：查配套版本表。

---

## 6. 延伸思考题（含解析）

**Q1**：CUDA 代码搬到昇腾，核心改哪几处？
**A**：导入 `torch_npu` 注册设备；`.cuda()`/device 改 `npu`（最好写环境无关）；CUDA 特有的混合精度、同步、计时 API 换昇腾对应物。训练循环逻辑基本不动。

**Q2**：什么是算子 fallback？为什么危险？
**A**：昇腾无某算子实现时，torch_npu 把它放到 CPU 算再搬回。危险在隐蔽——代码照常跑，但 CPU 慢几十倍且夹杂大量数据搬运，整体性能骤降而不易察觉。

**Q3**：昇腾训练为什么优先 BF16 而不是 FP16？
**A**：BF16 指数位多、动态范围大、不易上下溢，训练更稳；FP16 范围小，混合精度下易溢出需 loss scaling。昇腾对 BF16 支持好，故优先。

**Q4**：迁移后怎么验证"对了"？
**A**：① 同输入前向输出误差在阈值内；② 训练 loss 曲线趋势与 GPU 一致、最终精度相当。数值可有浮点微差，但趋势必须一致。

**Q5**：数据加载成了瓶颈，怎么判断和解决？
**A**：判断：msprof/利用率显示 NPU 空闲等待。解决：加 `num_workers`、开预取、把预处理离线化、用更快的存储。别让昂贵的 NPU 等数据。

---

## 本周交付清单

- [ ] 云上 910B 确认环境（`torch.npu.is_available()`、`npu-smi`）。
- [ ] 迁移一个 LoRA 微调脚本，用环境无关的 device 写法。
- [ ] 前向误差对齐验证 + 训练 loss 趋势对比。
- [ ] 检查无隐蔽 fallback（算子日志/trace）。
- [ ] 记录踩坑清单 ≥3 条（环境/算子/精度/性能分类）。
