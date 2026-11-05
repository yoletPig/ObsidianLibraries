# 第 4 周教程：稀疏 BEV 与流式时序

> **本周要回答的四个问题**
> 1. 为什么"不建 BEV 网格"反而可能是更好的选择？稀疏路线的算力账怎么算？
> 2. DETR3D 怎么用 3D query 投影采样 2D 特征？免 NMS 的集合预测训练时怎么对齐预测和真值？
> 3. Sparse4D 从 v1 到 v3 改了什么？StreamPETR 的"流式 query 传播"和 motion-aware layer normalization 解决什么？
> 4. 给一颗 30 TOPS 的端侧芯片，稠密还是稀疏？怎么选型？

对应学习计划：第 4 周。交付物：NumPy 稀疏 query 投影采样 + scipy 匈牙利匹配 + StreamPETR 式运动补偿 query 传播三件套，全部 CPU 可跑。

---

## 1. 稀疏动机：算力该跟着目标走，而不是跟着格子走

前两周的稠密路线有一个共同动作：**为整个感知范围建满 BEV 网格**。上周 demo 里的账（重列）：0.5 m 分辨率、±51.2 m 范围 → 200×200 = **40000 格**；0.25 m → 400×400 ≈ **160000 格**。而一帧 nuScenes 场景里的真实目标（车/行人/骑行者）通常只有**几十到几百个**。第 2 周例 5 也实测过：72 个视锥特征只点亮 100 根 pillar 中的 6 根。

**空区域是纯开销**：40000 格里占绝大多数的"路面/天空/远方空格"，在 BEV encoder 里消耗和目标格完全一样的算力与显存，最后的检测头再从这片"空海"里把目标挑出来。而且这个开销**随分辨率平方增长**——想要远距离细分辨率，代价爆炸。

稀疏路线（sparse query-based）的破法一句话：**取消 BEV 网格这个中间表示，直接维护一小撮"目标候选"（3D query），让每个候选自己去多相机图像上取证**。算力从 O(网格数) 变成 O(query 数 = 潜在目标数)，与网格分辨率、感知范围解耦。代价是失去一张统一的 BEV 图——**检测/跟踪任务收益最大，而 BEV 分割、占据栅格（occupancy）这类全空间任务不适用**，这是选型的第一分界线。

---

## 2. DETR3D：稀疏路线的开山

DETR3D（Wang et al., CoRL 2021, [arXiv:2110.06922](https://arxiv.org/abs/2110.06922)）把 DETR 的"query 集合预测"从 2D 图像搬进 3D 驾驶场景，三步：

### 2.1 第一步：3D query + 投影采样

初始化 N 个可学习的 3D object query（N 取数百量级），每个 query 预测一个 3D 参考点 $p \in \mathbb{R}^3$（ego 系 xyz），然后用标定矩阵**正投影**回每个相机：

$$
(u_n, v_n) \propto K_n \big( R_n \cdot p + t_n \big), \quad n = 1..6
$$

**逐符号**：与第 3 周 SCA 的投影公式完全同形——$K_n$ 内参、$R_n, t_n$ ego→相机外参。区别在于：BEVFormer 有 40000 根柱都要投，DETR3D 只有 N 个 query 投。投影落在图像范围内的才做**双线性采样**（bilinear sampling）取特征，视野外直接跳过——上周 demo 例 1 实测：12 个 query 只算 12 次采样，命中与否不影响其余 query 的计算量。

采样到的特征经 layer norm 后与 query 相加，再过 self-attention/FFN 精化框预测——迭代若干层（论文 layer 数 6），每层重新预测参考点、重新采样（query 的位置越猜越准，采样越聚焦）。

### 2.2 第二步：集合预测与匈牙利匹配（训练核心）

DETR3D 每个 query 输出恰好一个 3D 框，**没有 NMS、没有 anchor**——"集合预测（set prediction）"的核心承诺是预测集合与真值集合 1-1 对应。训练时怎么知道哪个预测对应哪个真值？**匈牙利匹配（Hungarian matching，二分图最优分配）**：

$$
\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_M} \sum_{i=1}^{N} \mathcal{C}\big(y_i,\; \hat{y}_{\sigma(i)}\big)
$$

**逐符号**：$\mathfrak{S}_M$ 是预测与真值的全部 1-1 分配方案集合；$\mathcal{C}(\cdot,\cdot)$ 是配对代价，**代价越小越该配对**——DETR3D 用的是"类别代价 + 框回归代价（L1 距离加权组合）"，本周 demo 例 3 为演示只用了 3D 中心距离；$\hat{\sigma}$ 是代价总和最小的唯一分配（匈牙利算法多项式时间求解，scipy 一行）。

匹配定了之后总损失：

$$
\mathcal{L} = \sum_{\text{匹配对}(i,\sigma(i))} \big[ \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{box}} \big] + \sum_{\text{未匹配 } j} \mathcal{L}_{\text{cls}}(\varnothing, \hat{y}_j)
$$

**逐符号**：匹配对算"分类交叉熵 + 框 L1 回归"；**没匹配上的 query 只算"预测'无目标'类"的分类损失**——这是对"多余的预测"的全部惩罚。第 5 节 demo 例 3 实测：5 预测 × 4 真值，最优匹配总代价 13.84，预测 #1 没配对 → 只学"这里没东西"。

**免 NMS 的原理**：NMS 的存在是因为传统检测器对同一目标出多框；集合预测下每个目标至多被一个 query 负责，重复框在训练中就被抑制。代价是收敛慢、对匹配质量敏感——这是 DETR 家族的共同课。

**成绩**（官方仓库 README 自报，查证）：nuScenes val，ResNet-101 + DCN：34.7% mAP / 42.2% NDS（+CBGS：34.9/43.4）；test 集 VoVNet 骨干：41.2% mAP / 47.9% NDS。对比 BEVFormer（test 56.9 NDS）差 9 个点——**开创性方法被后续超越，但 query→投影采样→集合预测的三件套被所有稀疏路线继承**。

---

## 3. Sparse4D 系列：把稀疏检测推向端到端

Sparse4D（Lin Xuewu 等，旷视 Megvii 团队，官方仓库曾挂在 megvii-research 组织下）三代演进（均查证）：

### 3.1 v1：4D 关键点（arXiv:2211.10581，2022-11）

标题 *Sparse4D: Multi-view 3D Object Detection with Sparse Spatial-Temporal Fusion*。核心改进：每个 3D anchor 不只投影一个中心点，而是分配**多个 4D 关键点（4D key points）**——3D 空间关键点 + 1 维时间（当前帧与前两帧），投影到多相机、多尺度、多时间戳的特征图上采样，再做层级融合。另加**实例级深度重加权（instance-level depth reweight）**：上周说过单目深度病态，v1 用实例级的深度重标定缓解"3D→2D 投影的病态性"。论文明确宣称该方法**对边缘设备部署友好**（无稠密视角转换、无全局 attention）。

### 3.2 v2：循环时序融合（arXiv:2305.14018，2023-05）

标题 *Sparse4D v2: Recurrent Temporal Fusion with Sparse Model*。把多帧特征采样改成**循环（recurrent）形式**：当前帧的 anchor/实例带着历史特征递归传播——时序信息"白拿"（近乎零额外算力），长时序融合不再随帧数堆显存。这是稀疏路线时序化的关键一步，与下节 StreamPETR 同期同思路（2023-05 与 2023-03 相隔两个月）。

### 3.3 v3：端到端检测 + 跟踪（arXiv:2311.11722，2023-11）

标题 *Sparse4D v3: Advancing End-to-End 3D Detection and Tracking*。两个辅助训练任务 + 一个结构改动：

1. **时序实例去噪（temporal instance denoising）**：训练时把真值框加噪、按运动外推后混进 query，让模型学会"从带噪历史实例恢复目标"——等于把跟踪的身份延续直接教进网络；
2. **质量估计（quality estimation）**：额外预测框的置信质量（类似 TIDE/GIoU 级别的辅助头），为排序与跟踪稳定性提供信号；
3. **解耦注意力（decoupled attention）**：特征与几何用不同的注意力头处理。

**成绩**（论文摘要自报，查证）：ResNet-50 骨干下比基线 +3.0% mAP / +2.2% NDS / +7.6% AMOTA，达 **46.9% mAP / 56.1% NDS / 49.0% AMOTA**；最佳模型 nuScenes test **71.9% NDS / 67.7% AMOTA**。跟踪不需要额外 tracker——推理时直接给实例赋 ID，**端到端**。

**仓库坑（实测查证）**：官方仓库原在 `megvii-research/Sparse4D`，**现已迁移到 [HorizonRobotics/Sparse4D](https://github.com/HorizonRobotics/Sparse4D)**（2025-11 实测旧地址 404）；v1/v2 亦有作者仓库 [linxuewu/Sparse4D](https://github.com/linxuewu/Sparse4D)。搜教程引用旧链接会撞 404，收藏 HorizonRobotics 版。

---

## 4. StreamPETR：流式 object query 传播

StreamPETR（Wang et al., 2023-03, [arXiv:2303.11926](https://arxiv.org/abs/2303.11926)，标题 *Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection*，官方代码 [exiawsh/StreamPETR](https://github.com/exiawsh/StreamPETR)，ICCV 2023）在 PETR 系（PETR, arXiv:2203.05625，3D position embedding 编码进 2D 特征的稀疏检测器）基础上，把时序做成**流（stream）**：

### 4.1 object query 流式传播

第 3 周 BEVFormer 的时序是"网格级"：warp 整张 BEV 特征。StreamPETR 的时序是"目标级"：上一帧的 **object query 本身**（含全部历史特征）直接传给当前帧当输入——

$$
q^{t} = \text{MotionComp}\big(q^{t-1}\big) \;\oplus\; \text{新初始化的 query}
$$

**逐符号**：$q^{t-1}$ 是上一帧的 top-k 实例 query（带位置与特征）；MotionComp 用 ego 运动 + 目标速度做**运动补偿**（demo 例 4 实测：上帧位置 (10, -2) + 目标速度 (10, 0)×0.1s − ego 位移 (3.0, 0.1) → 本帧预期 (8.0, -2.1)——目标自己往前开、车也往前开，两个运动都要补偿，剩下的是"预测残差"交给网络细化）；$\oplus$ 表示与新增 query 拼接成当前帧 query 集合。**传播的是特征而非重建**——历史任意长，显存恒定（只存一份 query），这是"流式（online/streaming）"的含义。

与 BEVFormer TSA 的分工差异：**网格级对齐对静止/慢目标好（warp 即对齐），目标级传播对动目标好（速度直接建模）**——运动补偿后的 query 初始位置几乎就落在真值附近，decoder 只需小幅细化，还天然带实例 ID（跟踪免费）。

### 4.2 Motion-aware Layer Normalization（MLN）

传播的 query 需要"知道自己动了多少"。MLN 的做法：把运动信息（ego 运动与两帧时间差）经过小网络变成**仿射调制参数** $(\gamma, \beta)$，注入 LayerNorm：

$$
\text{MLN}(x \mid m) = \gamma(m) \odot \frac{x - \mu}{\sigma} + \beta(m)
$$

**逐符号**：$x$ 是 query 特征；$\mu, \sigma$ 是常规 LayerNorm 的统计量；$m$ 是运动条件（位移/时间差向量）；$\gamma(m), \beta(m)$ 是由 $m$ 生成的逐通道缩放与偏移（$\odot$ 为逐元素乘）。**直觉**：LayerNorm 本来是"无量纲化"，现在让它**按运动量有偏地归一化**——同一辆车位移 3 m 和静止两种情形，query 特征被规范化到不同的坐标系，网络不必再用特征容量去"记住"运动差。这是把运动先验塞进归一化层的极廉价设计（几个小 FC），论文强调时序收益"以可忽略的额外计算获得"。

**成绩**（查证：论文摘要 + 官方 README，均为自报）：论文最佳 **67.6% NDS / 65.3% AMOTA**（nuScenes val，与激光方法可比的首个纯视觉流式方案）；轻量版 **45.0% mAP @ 31.7 FPS**（RTX 3090，bs=1，6 视图 FP32，README 口径），超 SOLOFusion 2.3 mAP 且快 1.8 倍；README 模型表：R50（428 query，NuImg 预训练，60ep）val **54.6 NDS / 44.9 mAP / 31.7 FPS**。

---

## 5. 可运行代码：采样、匹配、传播三件套

```python
# w4_sparse.py —— 第4周demo：稀疏 query 的投影采样、集合预测匹配、流式时序传播
# 机制演示（numpy/scipy），非真实模型。所有随机量固定种子。
import numpy as np
from scipy.optimize import linear_sum_assignment

rng = np.random.default_rng(42)

# ---------- 例1：DETR3D 式稀疏采样——N 个 3D query 投影到 2 相机采样特征 ----------
N_q, d_model = 12, 8                       # 12 个 3D 目标 query，特征 8 维
queries = rng.normal(0, 1, (N_q, d_model))            # 可学习 query（模拟）
pred_xyz = np.column_stack([rng.uniform(2, 40, N_q),   # 预测的 3D 参考点（x 前向）
                            rng.uniform(-10, 10, N_q), # y 左向
                            rng.uniform(0.2, 2.5, N_q)])  # z 高度
fx = fy = 400.0; cx, cy = 400.0, 225.0; W_img, H_img = 800, 450
T = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)   # 对齐系->相机系
feat_map = rng.normal(0, 1, (H_img//50, W_img//50, d_model))  # 虚拟前向相机特征图

sampled = np.zeros((N_q, d_model)); n_hit = 0
for i, (x, y, z) in enumerate(pred_xyz):
    u = fx * (-y) / x + cx                # 简化投影：cam_x = -y(右)，cam_z = x(前)
    v = fy * (z - 1.6) / x + cy           # 相机高 1.6 m
    if 0 <= u < W_img and 0 <= v < H_img: # 视野内才采样（DETR3D 同款 mask 逻辑）
        r, c = int(v // 50), int(u // 50)
        sampled[i] = feat_map[r, c]
        n_hit += 1
print(f"例1 {N_q} 个 3D query 投影采样，命中 {n_hit} 个 -> 只算 {n_hit} 次 bilinear 采样")
print(f"   采样特征 shape = {sampled.shape}（每 query 恰好 1 份特征，与视野外 query 数无关）")

# ---------- 例2：稀疏 vs 稠密的算力量级对比（每帧一次前向的采样次数） ----------
for res in [0.5, 0.25]:
    side = int(102.4 / res)
    print(f"例2 稠密路线：{side}x{side} 网格 = {side*side} 个 BEV 格子都要填（分辨率 {res} m）")
print(f"   稀疏路线：~{N_q}~900 个 query，算力随目标数（几十~几百）增长，与网格分辨率无关")

# ---------- 例3：集合预测的匈牙利匹配（scipy 线性和分配，DETR3D 训练核心一步） ----------
n_pred, n_gt = 5, 4                        # 5 个预测框 vs 4 个真值框（多出的预测匹配"无目标"）
# 代价矩阵：这里用简化的"3D 中心距离"（真实 DETR3D 用类别+框回归的加权代价）
pred_c = np.array([[5, 2], [12, 6], [20, 8], [27, 4], [33, 9.]])   # (x, y) 米
gt_c   = np.array([[5.5, 2.5], [19, 9], [26.5, 3.5], [40, 0.5]])
cost = np.linalg.norm(pred_c[:, None, :] - gt_c[None, :, :], axis=-1)  # (5, 4)
row, col = linear_sum_assignment(cost)     # 匈牙利算法：总代价最小的 1-1 匹配
print("\n例3 代价矩阵（预测 x 真值，单位 m）：")
print(np.round(cost, 2))
print(f"最优匹配: {list(zip(row.tolist(), col.tolist()))}")
print(f"对应代价 = {np.round(cost[row, col], 2).tolist()}, 总代价 = {cost[row, col].sum():.2f}")
print("未匹配的预测:", [r for r in range(n_pred) if r not in row], "-> 训练时只算分类'无目标'损失")

# ---------- 例4：StreamPETR 式流式传播——上一帧 query 经 ego 运动补偿后当成本帧输入 ----------
n_obj = 3
prev_pos = np.array([[10.0, -2.0], [25.0, 1.5], [35.0, 4.0]])  # 上一帧目标位置 (x, y)
vel = np.array([[10.0, 0.0], [-8.0, 1.0], [0.0, 0.0]])         # 目标自身速度（ego系, m/s）
dt = 0.1                                   # 100 ms 一帧
ego_dx, ego_dy = 3.0, 0.1                  # 本帧 ego 相对上一帧的位移
# 运动补偿：目标上一帧位置 + 自身运动 - ego 运动 = 本帧预期位置
curr_pos = prev_pos + vel * dt - np.array([ego_dx, ego_dy])
print("\n例4 流式传播（上一帧 3 个目标 query -> 本帧初始位置）：")
for p, v, c in zip(prev_pos, vel, curr_pos):
    print(f"  上帧 ({p[0]:5.1f},{p[1]:5.1f}) m + 目标速度{v.tolist()}-ego位移 -> 本帧 "
          f"({c[0]:5.2f},{c[1]:5.2f}) m")
print("  传播后的 query 带着历史特征直接进 Decoder 细化 -> 免 NMS、时序免费白拿")
print("  StreamPETR 的 motion-aware layer normalization：把运动量注入 LN 的仿射参数，"
      "让归一化本身知道'目标动了多少'")

# ---------- 例5：参数量参考（解析计算 torchvision 标准 ResNet，验证量级感） ----------
def resnet_params(blocks=(3, 4, 6, 3)):
    """按结构逐层解析累加参数量（与 torchvision 实现同构：conv 无 bias，BN 有 2 个参数）"""
    total = 9408 + 128                       # stem: 7x7x3x64 conv + BN
    mid_w, out_w, in_c = 64, 64, 64
    for stage, n_blk in enumerate(blocks):
        out_c = 256 * 2**stage               # 256/512/1024/2048
        for b in range(n_blk):
            i_c = in_c if b == 0 else out_c  # 首块的输入是上一 stage 输出
            m = mid_w * 2**stage             # 瓶颈中间宽度 64/128/256/512
            total += (1*1*i_c*m + 2*m + 3*3*m*m + 2*m + 1*1*m*out_c + 2*out_c)
            if b == 0 and (stage > 0 or i_c != out_c):   # downsample 1x1 + BN
                total += 1*1*i_c*out_c + 2*out_c
        in_c = out_c
    fc = 2048 * 1000                         # 分类头 ImageNet 1000 类
    return total + fc

for name, blocks in [("ResNet-50", (3, 4, 6, 3)), ("ResNet-101", (3, 4, 23, 3))]:
    n = resnet_params(blocks)
    print(f"例5 {name}: {n/1e6:.2f} M 参数（骨干网在端侧模型里通常占大头）")
print("   对照 torchvision 文档常引用值：ResNet-50 25.56 M / ResNet-101 44.55 M（含 fc 头）")
```

**预期输出**（实测）：

```
例1 12 个 3D query 投影采样，命中 12 个 -> 只算 12 次 bilinear 采样
   采样特征 shape = (12, 8)（每 query 恰好 1 份特征，与视野外 query 数无关）
例2 稠密路线：204x204 网格 = 41616 个 BEV 格子都要填（分辨率 0.5 m）
例2 稠密路线：409x409 网格 = 167281 个 BEV 格子都要填（分辨率 0.25 m）
   稀疏路线：~12~900 个 query，算力随目标数（几十~几百）增长，与网格分辨率无关

例3 代价矩阵（预测 x 真值，单位 m）：
[[ 0.71 15.65 21.55 35.03]
 [ 7.38  7.62 14.71 28.54]
 [15.51  1.41  7.91 21.36]
 [21.55  9.43  0.71 13.46]
 [28.26 14.    8.51 11.01]]
最优匹配: [(0, 0), (2, 1), (3, 2), (4, 3)]
对应代价 = [0.71, 1.41, 0.71, 11.01], 总代价 = 13.84
未匹配的预测: [1] -> 训练时只算分类'无目标'损失

例4 流式传播（上一帧 3 个目标 query -> 本帧初始位置）：
  上帧 ( 10.0, -2.0) m + 目标速度[10.0, 0.0]-ego位移 -> 本帧 ( 8.00,-2.10) m
  上帧 ( 25.0,  1.5) m + 目标速度[-8.0, 1.0]-ego位移 -> 本帧 (21.20, 1.50) m
  上帧 ( 35.0,  4.0) m + 目标速度[0.0, 0.0]-ego位移 -> 本帧 (32.00, 3.90) m
  传播后的 query 带着历史特征直接进 Decoder 细化 -> 免 NMS、时序免费白拿
  StreamPETR 的 motion-aware layer normalization：把运动量注入 LN 的仿射参数，让归一化本身知道'目标动了多少'
例5 ResNet-50: 25.56 M 参数（骨干网在端侧模型里通常占大头）
例5 ResNet-101: 44.55 M 参数（骨干网在端侧模型里通常占大头）
   对照 torchvision 文档常引用值：ResNet-50 25.56 M / ResNet-101 44.55 M（含 fc 头）
```

**看懂这个输出**：

1. 例 3 的匹配验证了匈牙利算法的"全局最优"本性：**逐个取最小代价会错**——预测 #4 距 gt#3 只有 0.71，但预测 #3 距 gt#2 也是 0.71；贪心会撞车，全局最优解 (0,0),(2,1),(3,2),(4,3) 总代价 13.84 是所有 1-1 方案里最小的。预测 #1（12, 6）离任何真值都远，被留给"无目标"分支。
2. 例 4 的三个目标传播方向各不相同：#1 前进（自身快）、#2 后退（相对 ego 在减速/远离）、#3 基本跟车（相对静止）——**运动补偿后"相对静止"的目标 query 位置不变**，这正是速度建模的语义。
3. 例 5 的参数账：骨干 25~45 M 在整个模型里占大头，而稀疏路线的 decoder/匹配头参数量远小于稠密路线的 BEV encoder——端侧压缩的第一刀从来都在骨干网。

> **动手验证**：
> 1. 把例 1 的某个 query 改成 `pred_xyz[0] = [1.0, 30.0, 1.0]`（y=30 m 侧前方）：投影 u = 400×(−30)/1 + 400 大幅出画 → 命中数减 1。感受"视野 mask 对 query 数量不敏感"。
> 2. 把例 3 的 pred_c[1] 改成 `[19.5, 8.8]`（贴近 gt#1）：重跑后预测 #1 应被匹配到 gt#1——代价矩阵变了，全局最优解跟着变，体会"匹配是分配问题不是最近邻问题"。
> 3. 把例 4 的 ego_dx 改成 0（静止）：三个目标的补偿量只来自自身速度——**ego 运动补偿与目标运动补偿是两个独立项**，这也是为什么两者都要进模型。

---

## 6. 稀疏 vs 稀疏对比表

（所有数字为论文/官方仓库**自报**的 nuScenes 成绩；val/test 集不同，不能横向直接比，看演进趋势。）

| 方法 | 时间 | 时序方式 | nuScenes 成绩（自报） | 延迟/FPS | 论文/仓库 |
| --- | --- | --- | --- | --- | --- |
| DETR3D | CoRL 2021 | 无（单帧） | test：41.2 mAP / 47.9 NDS（VoV）；val：34.7/42.2（R101） | 未官方公布 | [2110.06922](https://arxiv.org/abs/2110.06922) / WangYueFt/detr3d |
| Sparse4D v1 | 2022-11 | 多帧特征采样 | 超 sparse 系与多数 BEV 系（论文定性） | 未公布 | [2211.10581](https://arxiv.org/abs/2211.10581) / linxuewu 或 HorizonRobotics/Sparse4D |
| Sparse4D v2 | 2023-05 | 循环时序（recurrent） | nuScenes SOTA（论文定性，无摘要数字） | 未公布 | [2305.14018](https://arxiv.org/abs/2305.14018) |
| Sparse4D v3 | 2023-11 | 循环 + 实例去噪/质量估计 | R50：46.9 mAP / 56.1 NDS / 49.0 AMOTA；test 最佳 71.9 NDS / 67.7 AMOTA | 未公布 | [2311.11722](https://arxiv.org/abs/2311.11722) / HorizonRobotics/Sparse4D |
| StreamPETR | 2023-03 | 流式 query 传播 + MLN | 论文最佳 67.6 NDS / 65.3 AMOTA；R50 轻量版 44.9 mAP / 54.6 NDS @ 31.7 FPS（RTX 3090 实测口径） | 31.7 FPS（R50 轻量版） | [2303.11926](https://arxiv.org/abs/2303.11926) / exiawsh/StreamPETR |

**趋势读法**：2021→2023 的两年里，稀疏路线靠"时序"把 NDS 从 47.9（DETR3D test）推到 71.9（Sparse4D v3 test），同时把轻量版做到 30+ FPS——**精度与速度同时逼近激光方案**，这正是 Sparse4D v3 论文标题"End-to-End"和 StreamPETR"rival lidar-based approaches"两个宣称的分量。

---

## 7. 端侧部署视角：稀疏路线为什么对芯片友好

1. **算子友好**：稀疏路线的主力算子 = 投影（矩阵乘/查表）+ 双线性采样（grid_sample）+ 小规模 attention（N~900 个 query 的 self-attention，序列长度极短）——全是 NPU 编译器熟透的标准算子。稠密路线的大张量 scatter（LSS 池化）与 40000-query 大注意力（BEVFormer）在 NPU 上要么算子不支持要么带宽吃亏。
2. **算力-分辨率解耦**：想要 150 m 感知？稀疏路线只是把 query 的参考点撒得更远，**计算量不变**；稠密路线网格平方爆炸。感知范围需求（如 AEB 远距）对稀疏路线是"免费午餐"。
3. **算力-目标数解耦**：空旷高速上 5 个目标与闹市 200 个目标，稠密路线算力完全相同（都建满网格）——浪费在空旷场景；稀疏路线算力随 query 数增长，**忙时多花、闲时少花**，天然负载均衡（query 数上限决定最坏延迟，可做硬实时）。
4. **免 NMS 后处理**：集合预测天然 1-1，后处理只剩阈值过滤——后处理是 CPU 瓶颈的常见来源，砍掉即是延迟。
5. **端到端跟踪**：Sparse4D v3 / StreamPETR 的实例 ID 传播让 tracker 退化成一个滤波器（甚至省掉），跟踪-预测-规划链路的延迟进一步缩短。

**稀疏的代价也要记**：没有统一 BEV 图 → BEV 分割/占据栅格/在线建图类任务不适配（Occupy 系任务稠密 BEV 仍是主流）；query 数上限硬编码了"最坏同时目标数"，极端场景（超密集人群）可能漏检；训练收敛比稠密慢、对匹配与去噪策略敏感。

---

## 8. 选型建议与本阶段收束

**选型决策树**（按任务 × 平台）：

- **做检测+跟踪、芯片 ≤ 100 TOPS** → 稀疏路线（StreamPETR/Sparse4D 系），精度够、延迟稳、算子友好；
- **做占据栅格/在线建图/BEV 分割** → 稠密 BEV 表示不可绕开（LSS 系 encoder 或 Transformer 系 BEV 特征），考虑用稀疏检测 + 轻量稠密占据的混合架构；
- **相机+激光多传感器** → BEVFusion（Liu et al., ICRA 2023, [arXiv:2205.13542](https://arxiv.org/abs/2205.13542)，[mit-han-lab/bevfusion](https://github.com/mit-han-lab/bevfusion)）把相机 BEV 特征与激光 BEV 特征统一进一个 BEV 空间再做融合——**BEV 表示的价值在多传感器融合场景最大化**，这是 Stage 4 后融合知识的"特征级升级版"；
- **数据闭环/大模型前处理** → 稠密 BEV 提供的统一空间表征便于跨任务复用（检测/地图/占据共享一张图），工程生态（mmdet3d 系）成熟。

**学习路线收束**：从第 1 周的"一条射线一个自由度"出发，走过三条找回深度的路——撒满概率（LSS）、主动询问（BEVFormer）、只在目标处问（DETR3D/Sparse4D/StreamPETR）。三条路的共同地基是**标定矩阵 + ego 运动补偿**：前者决定"空间对不对"，后者决定"时间对不对"。这也为 Stage 6（V2X）铺垫了核心问题——当"ego"从单车扩展成"车-路-车多体"，运动补偿的对象从自车位姿扩展成多节点时空对齐，BEV 的统一空间思想正是 V2X 协同感知的公共语言。

---

## 9. 延伸思考题（含解析）

**Q1**：DETR3D 为什么能免 NMS？什么情况下它仍会出"重复框"？
**A**：集合预测 + 匈牙利匹配保证训练时每个真值至多被一个 query 负责，模型学会相互谦让；推理时无 anchor 无多框堆叠。但若两个 query 收敛到同一目标（训练不充分或目标极大），推理仍可能出近重复框——所以部分工程实现仍保留一个极低阈值的后处理兜底。

**Q2**：StreamPETR 的 query 传播和 BEVFormer 的 TSA，对"静止目标"和"匀速横穿行人"分别谁更稳？
**A**：静止目标：TSA 更稳——warp 后格子索引严格对齐，注意力只判断"同一格还是同一物体"，几何先验最强。匀速横穿行人：目标级传播更稳——速度直接进运动补偿，query 初始位置就贴住真值；TSA 的格子对齐假设"目标随世界静止"，横穿目标跨帧偏离对齐位置，时序证据被稀释。**这解释了为什么 StreamPETR 的 NDS 提升集中在 mAVE（速度误差）上。**

**Q3**：motion-aware layer normalization 为什么不用"把运动量拼进特征"这种更直白的方式？
**A**：拼特征可行但扰动大：运动量纲（米）与特征分布（深层抽象）差异大，拼接后网络需要额外容量学习融合；MLN 把运动变成归一化的仿射条件，等于给特征一个"按运动重标定的坐标系"，不增加序列长度、不扰动主干、参数开销是几个 FC——与 FiLM 类条件归一化同族思想。

**Q4**：端侧芯片 30 TOPS、内存 8 GB，任务只有 3D 检测+跟踪，你选哪条路线？给三条理由。
**A**：选稀疏（StreamPETR/Sparse4D 系轻量版）。理由：① 算子全是 NPU 标准件，无大张量 scatter；② 延迟随目标数可预算，硬实时可控（query 上限封顶）；③ 免 NMS + 免 tracker，后处理 CPU 开销小。参考点：StreamPETR R50 轻量版在 RTX 3090 上 31.7 FPS（FP32），30 TOPS 量级 NPU 上同骨架做 INT8 有量产先例。

---

## 本周交付清单

- [ ] 跑通 `w4_sparse.py`，确认输出与文中一致（匹配总代价 13.84、传播三目标位置逐个核对）。
- [ ] 闭卷写出匈牙利匹配公式与"未匹配预测只算无目标分类损失"的训练结构。
- [ ] 手算一遍例 4 的运动补偿（上帧位置 + 目标速度×dt − ego 位移）。
- [ ] 能回答：稀疏路线对芯片友好的四个理由？它的两个不适用场景？
- [ ] 用学习计划的"阶段终极自测清单"验收，然后进入 Stage 6（V2X 协同感知）。
