# 第 2 周教程：稠密 BEV 路线 I——LSS 与 BEVDet

> **本周要回答的四个问题**
> 1. Lift-Splat-Shoot（LSS）的 Lift 到底在做什么？为什么"每个像素预测一个深度分布"能解决深度未知？
> 2. 外积（outer product）那一步，2 像素 × 3 深度档的完整手算长什么样？
> 3. Splat 怎么把几百万个视锥特征塞进 200×200 的 BEV 网格？和 PointPillars 什么关系？
> 4. 稠密 BEV 的算力/显存账单是多少？BEVDet 在 LSS 上补了哪些工程课？

对应学习计划：第 2 周。交付物：NumPy 缩小版 Lift（4×6 像素、3 深度档）+ 简化 Splat，验证"深度分布和为 1""外积后沿深度求和还原原特征"，打印完整 shape 链。

---

## 1. 第一性原理：把"深度未知"变成"深度概率"

上周的结论：像素 → 3D 必须指定深度 $d$，而 $d$ 无观测。LSS（Lift, Splat, Shoot, Philion & Fidler, NVIDIA, ECCV 2020, [arXiv:2008.05711](https://arxiv.org/abs/2008.05711)，官方代码 [nv-tlabs/lift-splat-shoot](https://github.com/nv-tlabs/lift-splat-shoot)）的答案是**不要赌一个深度，把所有可能的深度都撒一遍，按概率加权**：

$$
\text{不预测 } d \text{，而是预测 } \alpha = [\alpha_1, \dots, \alpha_D], \quad \sum_i \alpha_i = 1
$$

**这个式子在说什么**：每个像素输出一个**离散深度分布**（categorical distribution）——D 个深度档各是多少概率。与其说"这个像素是 23.7 m"，不如说"这个像素 20 m 档占 30%、25 m 档占 50%、30 m 档占 20%"。深度估计从回归问题变成分类问题——分类问题好训练（softmax + 交叉熵）、好表达多峰不确定性（玻璃幕墙后面到底是玻璃还是后面的车？两个峰都保留）。

LSS 名字里的三个动词就是整条流水线：

```
Lift  ：每张图像 → 视锥特征（frustum features）      [2D + 深度分布 → 伪 3D]
Splat ：所有相机的视锥特征 → BEV pillar 特征         [伪 3D → BEV]
Shoot ：BEV 特征 → 分割/规划轨迹                     [BEV → 下游任务]
```

---

## 2. Lift 三步全拆解

### 2.1 第一步：深度分布 α（depthnet + softmax）

图像 backbone（原论文用 EfficientNet-B0）输出每像素特征后，接一个 `depthnet`（1×1 卷积）输出 D 个深度 logit，再 softmax：

$$
\alpha_{u,v,d} = \frac{\exp(z_{u,v,d})}{\sum_{d'=1}^{D} \exp(z_{u,v,d'})}
$$

**逐符号**：$z_{u,v,d}$ 是像素 $(u,v)$ 在深度档 $d$ 上的原始 logit（网络输出，无界）；$\alpha_{u,v,d}$ 是归一化后的概率，对每个像素沿深度维求和恒为 1。深度档 $d$ 不是抽象索引，而是**真实距离**：LSS 默认 `dbound=[4.0, 45.0, 1.0]`——从 4 m 到 45 m、步长 1 m、共 **41 档**（查证自官方 `src/explore.py`）。所以 $\alpha_{u,v,17}$ 直译为"像素 (u,v) 处的物体在约 21 m 深度档上的概率"。

### 2.2 第二步：上下文特征 c

同一特征图再过一层得到**上下文特征（context features）** $c_{u,v} \in \mathbb{R}^C$（原论文 C=64）。它是"这个像素看起来的样子"——颜色、语义、纹理的抽象编码，**不含深度信息**（深度已被拆到 α 里）。原文就是把 backbone 特征用 1×1 卷积分成 depth（D 通道）和 context（C 通道）两个头。

### 2.3 第三步：外积得视锥特征（核心公式）

$$
F_{u,v,d} = \alpha_{u,v,d} \cdot c_{u,v} \quad \in \mathbb{R}^C
$$

**逐符号讲清这个公式在干什么**：

- $c_{u,v}$：像素的 C 维上下文特征；
- $\alpha_{u,v,d}$：该像素在深度档 $d$ 上的概率（标量）；
- $F_{u,v,d}$：把上下文特征"复制"到第 $d$ 档深度上、再按概率缩放后得到的 C 维向量。对所有 $d$ 拼起来，每个像素的 C 维特征膨胀成 $D \times C$ 的块——整个图像物化成 $(H, W, D, C)$ 的四维张量，这就是**视锥特征（frustum features）**；
- 几何含义：特征 $c_{u,v}$ 现在沿整条像素射线"摊开"分布——每个深度档上放一份被 $\alpha$ 加权的副本。**信息没有被创造或销毁，只是从"深度未知的一维"摊成了"深度概率加权的 D 份"**；
- 原文称之为 outer product（外积）：把 $(\alpha \in \mathbb{R}^D)$ 与 $(c \in \mathbb{R}^C)$ 做外积得到 $D \times C$ 矩阵，形式上与向量外积 $uv^T$ 同构。

**一个立刻能验证的性质**（第 5 节代码实测）：沿深度档求和

$$
\sum_{d=1}^{D} F_{u,v,d} = \left(\sum_d \alpha_{u,v,d}\right) c_{u,v} = 1 \cdot c_{u,v} = c_{u,v}
$$

分布和为 1 ⇒ **视锥特征沿深度求和精确还原原上下文特征**。信息守恒的直接证据。

### 2.4 手算例：2 像素 × 3 深度档完整算一遍

设深度 3 档（10 m / 25 m / 40 m）。两个像素：

- **像素 A**：深度分布 $\alpha_A = [0.7, 0.2, 0.1]$，上下文特征 $c_A = [1, 0]$（二维简化）；
- **像素 B**：深度分布 $\alpha_B = [0.1, 0.8, 0.1]$，上下文特征 $c_B = [0, 2]$。

外积（每行一个深度档，每列一个特征维）：

$$
F_A = \begin{pmatrix} 0.7 & 0 \\ 0.2 & 0 \\ 0.1 & 0 \end{pmatrix}, \qquad
F_B = \begin{pmatrix} 0 & 0.2 \\ 0 & 1.6 \\ 0 & 0.2 \end{pmatrix}
$$

沿深度求和：$F_A \to (1, 0) = c_A$，$F_B \to (0, 2) = c_B$。信息守恒成立。直觉解读：像素 A 最可能在 10 m 处且"看起来像特征 [1,0]"，所以 10 m 档放了 0.7 份；它有 10% 的可能性其实在 40 m，于是 40 m 档也留了 0.1 份——**不确定性被完整带进了 3D**。Splat 池化时，如果 10 m 和 40 m 两个位置都有别的证据支持，这 0.1 份会作为弱证据参与投票；反之被稀释。

### 2.5 视锥 → ego 系：像素坐标 × 标定 × 深度档

视锥特征的下标是 $(u, v, d)$，但 Splat 要的是 ego 系位置。做法就是上周的反投影公式**对整个视锥批量执行**：

$$
P_{ego}(u, v, d) = R \cdot K^{-1} \cdot \big(d \cdot [u, v, 1]^T\big) + t \quad \text{(合并内参外参的紧凑写法)}
$$

每个 $(u,v,d)$ 格点得到一个 ego 3D 点。原论文实现里（`src/models.py` 的 `voxel_pooling`）先用标定把视锥点变换到 ego 坐标，再把坐标减去网格原点、除以分辨率、floor 取整，得到 BEV 网格索引。真实规模感受一下：6 相机 × 128×352 像素 × 41 档 ≈ **1100 万个视锥特征**要计算索引并写入 pillar。

---

## 3. Splat：pillar pool + PointPillars 式 encoder

### 3.1 Pillar pooling

LSS 的 BEV 网格 x/y ∈ [-50, 50] m、0.5 m 分辨率 → **200×200 根 pillar（柱）**，z 维压扁（`zbound=[-10,10,20.0]` 一档）。落入同一根 pillar 的视锥特征用 **sum pooling** 累加：

$$
B_{i,j} = \sum_{(u,v,d)\,:\,P_{ego}(u,v,d) \in \text{pillar}(i,j)} F_{u,v,d}
$$

**逐符号**：$B_{i,j} \in \mathbb{R}^C$ 是第 $(i,j)$ 根 pillar 的 C 维特征；求和条件是"该视锥特征点的 ego 位置落在 pillar $(i,j)$ 的 x/y 范围内"。这一步就是第 1 周"反投影 + 网格索引"的批量版，也是 Stage 2 PointPillars 里"点云散落到 pillar"的同款操作——**只不过散落的从激光点换成了图像视锥特征**。

### 3.2 BEV encoder

pillar 特征图 (200×200×C) 过一个 ResNet-18 骨干的 encoder（原论文 `BevEncode`），把稀疏、带洞的特征图补成平滑的 BEV 语义图，输出 BEV 分割。结构上与 Stage 2 PointPillars 的 "pillar feature network + CNN backbone" 完全同型：**LSS = 点云入口换成图像视锥的 PointPillars**。这个观察很重要：Stage 2 学的 BEV encoder 知识全部平移复用。

### 3.3 Shoot

原论文的下游任务是 BEV 语义分割 + 规划：在 BEV 特征上卷积出每个格子的占据代价，构成可微代价图，一条候选轨迹的总代价 = 图上采样求和，训练时直接用驾驶数据回传梯度——"shoot"就是让规划器在代价图上"射击"轨迹。**注意**：LSS 论文本身不做 3D 框检测；把它变成检测框架的是 BEVDet（第 6 节）。

---

## 4. 可运行代码：缩小版 Lift-Splat

```python
# w2_lss.py —— 第2周demo：缩小版 Lift-Splat（机制演示，非真实模型）
# 规模：1 张图像 4x6 像素，特征 C=4 维，深度 D=3 档。所有随机量固定种子。
import numpy as np

rng = np.random.default_rng(42)
H, W, C, D = 4, 6, 4, 3

# ---------- 例1：Lift 第一步——每个像素预测深度分布 α（softmax） ----------
# 真实 LSS 里这是 depthnet 输出 D 个 logit 再 softmax；这里用固定随机 logit 模拟
depth_logits = rng.normal(0, 1, (H, W, D))          # (H, W, D) 每像素 D 个深度 logit
alpha = np.exp(depth_logits - depth_logits.max(-1, keepdims=True))
alpha /= alpha.sum(-1, keepdims=True)               # softmax：沿深度维归一化
print("例1 深度分布 α：")
print("shape =", alpha.shape)
print("每个像素沿深度档的和（应全为 1）:",
      np.allclose(alpha.sum(-1), 1.0))
print("像素(0,0) 的 α =", np.round(alpha[0, 0], 4))

# ---------- 例2：Lift 第二步——上下文特征 c（backbone 输出，模拟） ----------
context = rng.normal(0, 1, (H, W, C))               # (H, W, C) 每像素 C 维特征
print("\n例2 上下文特征 c shape =", context.shape)

# ---------- 例3：Lift 第三步——外积得视锥特征 ----------
# frustum[u,v,d,:] = α[u,v,d] * c[u,v,:]，shape (H, W, D, C)
frustum = alpha[..., None] * context[:, :, None, :]  # (H,W,D,1) * (H,W,1,C)
print("\n例3 视锥特征 frustum shape =", frustum.shape)
# 关键验证：沿深度档求和应还原上下文特征（分布和为1 => 加权和就是原特征）
recovered = frustum.sum(axis=2)                      # (H, W, C)
print("沿 D 求和还原 c：", np.allclose(recovered, context))
# 这就是 LSS 论文里的 outer product：C 维特征被 α 加权复制到 D 个深度档上

# ---------- 例4：手算例核对（2 像素 x 3 深度档，正文表格的数值） ----------
a_px = np.array([0.7, 0.2, 0.1]); c_px = np.array([1.0, 0.0])   # 像素A
b_px = np.array([0.1, 0.8, 0.1]); d_px = np.array([0.0, 2.0])   # 像素B
FA = a_px[:, None] * c_px[None, :]    # (3,2)
FB = b_px[:, None] * d_px[None, :]
print("\n例4 手算例（像素A α=[0.7,0.2,0.1], c=[1,0]）：")
print("视锥特征 F_A =\n", FA)
print("像素B α=[0.1,0.8,0.1], c=[0,2]，F_B =\n", FB)
print("A 沿深度求和 =", FA.sum(0), "  B 沿深度求和 =", FB.sum(0))

# ---------- 例5：Splat——反投影到 ego 并按 pillar 池化 ----------
# 相机几何（同第1周）：fx=fy=200, 图 4x6 像素，光心 (2.5, 3)（行,列中心）
fx_ = fy_ = 200.0
v_c, u_c = (H - 1) / 2, (W - 1) / 2
ds_m = np.array([10.0, 25.0, 40.0])   # 3 档深度对应的真实距离（米）
# BEV 网格：x/y in [-50, 50]，分辨率 10 m -> 10x10 格（真实 LSS 是 200x200）
xb = (-50.0, 50.0, 10.0)
nb = int((xb[1] - xb[0]) / xb[2])
bev = np.zeros((nb, nb, C))           # BEV 特征图 (4, 4, C)
n_out = 0
for i in range(H):
    for j in range(W):
        for k, d in enumerate(ds_m):
            # 像素 -> 相机系归一化 -> 乘深度 -> ego（简化：ego_x = 深度方向，ego_y = 横向）
            xc = (j - u_c) / fx_      # 列偏移 -> 横向正切
            yc = (i - v_c) / fx_      # 行偏移 -> 垂直正切
            x_e = d                   # 前向距离
            y_e = xc * d              # 横向位置
            if xb[0] <= x_e < xb[1] and xb[0] <= y_e < xb[1]:
                gi = int((x_e - xb[0]) / xb[2])   # 前向 -> 行
                gj = int((y_e - xb[0]) / xb[2])   # 横向 -> 列
                bev[gi, gj] += frustum[i, j, k]   # sum pooling 入柱
            else:
                n_out += 1
print(f"\n例5 Splat：BEV 特征图 shape = {bev.shape}，"
      f"落入网格的视锥特征数 = {H*W*D - n_out}/{H*W*D}（越界 {n_out} 个）")
print("非零 pillar 数 =", int((np.abs(bev).sum(-1) > 1e-9).sum()), "/", nb*nb)

# ---------- 例6：稠密 BEV 的代价——显存随网格平方增长 ----------
C_real = 80                            # 真实模型的 BEV 通道数（几十~几百）
for res, side in [(0.5, 200), (0.25, 400), (0.125, 800)]:
    mem = side * side * C_real * 4 / 1024**2   # float32，单帧单层
    print(f"分辨率 {res} m -> {side}x{side} 格 x {C_real} 通道 float32 ≈ {mem:.1f} MiB")
print("（真实 LSS 视锥是 H*W*D*C：128x352x41x64 ≈",
      round(128*352*41*64*4/1024**2, 1), "MiB，Lift 阶段就要先物化这个中间量）")
```

**预期输出**（实测）：

```
例1 深度分布 α：
shape = (4, 6, 3)
每个像素沿深度档的和（应全为 1）: True
像素(0,0) 的 α = [0.3543 0.0923 0.5533]

例2 上下文特征 c shape = (4, 6, 4)

例3 视锥特征 frustum shape = (4, 6, 3, 4)
沿 D 求和还原 c： True

例4 手算例（像素A α=[0.7,0.2,0.1], c=[1,0]）：
视锥特征 F_A =
 [[0.7 0. ]
 [0.2 0. ]
 [0.1 0. ]]
像素B α=[0.1,0.8,0.1], c=[0,2]，F_B =
 [[0.  0.2]
 [0.  1.6]
 [0.  0.2]]
A 沿深度求和 = [1. 0.]   B 沿深度求和 = [0. 2.]

例5 Splat：BEV 特征图 shape = (10, 10, 4)，落入网格的视锥特征数 = 72/72（越界 0 个）
非零 pillar 数 = 6 / 100
分辨率 0.5 m -> 200x200 格 x 80 通道 float32 ≈ 12.2 MiB
分辨率 0.25 m -> 400x400 格 x 80 通道 float32 ≈ 48.8 MiB
分辨率 0.125 m -> 800x800 格 x 80 通道 float32 ≈ 195.3 MiB
（真实 LSS 视锥是 H*W*D*C：128x352x41x64 ≈ 451.0 MiB，Lift 阶段就要先物化这个中间量）
```

**Shape 链全记录**（缩小版）：

$$
\underbrace{(4, 6, 3)}_{\text{深度分布 }\alpha} \cdot \underbrace{(4, 6, 4)}_{\text{上下文 }c} \to \underbrace{(4, 6, 3, 4)}_{\text{视锥 }F} \xrightarrow{\text{反投影+池化}} \underbrace{(10, 10, 4)}_{\text{BEV }B}
$$

真实版对应：$(H, W, D, C) = (128, 352, 41, 64)$ × 6 相机 → $(200, 200, C_{bev})$。

**看懂这个输出**：

1. 例 3 的 `True` 是 LSS 最核心的一条性质：**视锥特征是上下文特征的深度维无损展开**，后面所有"多深度档竞争"都发生在 Splat 池化和 BEV encoder 里，Lift 本身不丢信息。
2. 例 5 的"非零 pillar = 6/100"是重点：72 个视锥特征只点亮了 6 根 pillar——**真实 BEV 也是如此，绝大多数 pillar 是空的**，但稠密路线的 BEV encoder 仍要对 200×200 全格计算。这个"稀疏的真相、稠密的算"就是第 4 周稀疏路线的靶子。
3. 例 6 的账单：单层 BEV 特征 12.2 MiB（0.5 m）看似不大，但 BEV encoder 有几十层中间量、训练还有梯度与 6 相机视锥的物化，显存轻松到几十 GB——BEVFormer 论文里 base 模型训练显存标注 28.5 GB（官方 README 模型表）。

> **动手验证**：
> 1. 把例 1 的 `depth_logits` 改成全部相同值（如 `np.ones((H,W,D))`）：α 应全部变为 [1/3, 1/3, 1/3]，例 3 的"还原 c"仍为 True——**均匀分布时特征在 3 档间均摊，深度信息彻底丢失**，感受"分布尖不尖"对深度定位的决定作用。
> 2. 把例 4 像素 B 的 α 改成 [0.1, 0.1, 0.8]（40 m 档为主），重算 F_B 与 Splat 落点：同一像素的 80% 特征质量从 25 m pillar 搬到 40 m pillar——深度分布就是"特征质量的搬运指令"。
> 3. （进阶）下载 `nv-tlabs/lift-splat-shoot`，读 `models.py` 的 `get_geometry` 与 `voxel_pooling`，核对：`geom` 张量 shape 是 (B, N, D, H, W, 3)（N=相机数），与本周 demo 的循环逻辑一一对应。

---

## 5. BEVDet：把 LSS 变成检测框架的工程课

BEVDet（Huang et al., 2021-12, [arXiv:2112.11790](https://arxiv.org/abs/2112.11790)）保留 LSS 式 view transformer，框架变成四段：**图像视图 encoder → view transformer → BEV encoder → 检测头**，检测头直接沿用 Stage 2 的 CenterPoint-style 头。相对原始 LSS 的关键工程改进（论文自述）一句话版：

- **双空间数据增广**：透视图像空间增广（防过拟合）+ BEV 空间增广（弥补视锥导致的 BEV 过拟合），这是 BEVDet 涨点的核心一招；
- **更精细的 Lift/Splat 实现与 BEV encoder**（配合 mmdet3d 工程栈的成熟算子）；
- **为该范式定制的 NMS**：稠密 BEV 头输出大量候选框，NMS 策略按新范式调整。

成绩（论文自报，nuScenes val / test）：BEVDet-Tiny 31.2% mAP / 39.2% NDS（val，计算量约为 FCOS3D 的 1/11）；BEVDet-Base 38.4% mAP / 46.1% NDS（val）；在 nuScenes test 榜（v3）上 42.2% mAP / 48.2% NDS，是当时纯视觉方法第一。后续 BEVDet4D（2022）加上了时序版本——把 LSS 的 Splat 扩到 4D 空间，用前帧 BEV 特征按 ego 运动 warp 后拼接，验证了"稠密 BEV + 时序 warp"的可行版型（第 3 周 BEVFormer 的 TSA 是同一思想的注意力版）。

---

## 6. 稠密 BEV 的代价：账单与本质

第 5 节例 6 已给出数字，这里把账单的本质说透：

1. **算力花在空区域**：200×200 pillar 中真正有目标的不过几十格（例 5 实测 6/100），但 BEV encoder 的卷积/后续头对全网格等价计算——**感知信息密度与算力开销不成比例**，这是稠密路线的原罪。
2. **视锥中间量巨大**：Lift 阶段物化 (H×W×D×C) ≈ 451 MiB 的中间张量（每帧、每 batch），且 6 相机并行——端侧 NPU 对这类大张量的搬运成本高于计算成本。
3. **分辨率不可局部调节**：想要 150 m 远处 0.1 m 分辨率？稠密路线只能全网升到 0.1 m（200→1000 格，25 倍代价），无法"只在前方 30° 精细"。
4. **深度误差的传播**：α 分布尖峰预测错 1 档（1 m），特征整份撒错 pillar——池化不可撤销。这催生了 BEVDepth 类工作用激光监督深度、以及"干脆别撒满"的稀疏路线。

**对比表**（稠密两系，第 3 周展开 Transformer 系）：

| 维度 | LSS 系（显式深度） | Transformer 系（隐式查询） |
| --- | --- | --- |
| 深度处理 | 每像素显式深度分布，可被激光监督 | 藏在注意力权重里，不可直读 |
| 算子形态 | 大量 for-loop 级 scatter/索引（工程上靠 CUDA 优化） | 标准 attention 算子，芯片生态友好 |
| 典型代表 | LSS、BEVDet、BEVDepth | BEVFormer、PETR 系 |

---

## 7. 延伸思考题（含解析）

**Q1**：为什么 LSS 选"分类 D 档深度"而不是"回归 1 个深度"？
**A**：回归给不了不确定性表达，且深度病态区域（平面、反光）回归会出不可控的离群值；分类分布天然多峰、可稀释、可被激光深度图逐像素监督（BEVDepth 的做法）。代价是 D 档的显存与池化开销。

**Q2**：视锥特征沿深度求和精确还原 c——那"深度分布还有什么用"？为什么不直接用 c？
**A**：Splat 之后各 pillar 拿到的是**位置加权**的证据：同一个 c 若 α 尖，全部质量落在一根 pillar（定位明确）；若 α 平，质量摊在多根 pillar（定位模糊）。池化后的 BEV 图分布完全不同——深度分布的价值体现在**空间分配**上，而非特征数值上。

**Q3**：LSS 的 Splat 用 sum pooling，PointPillars 用 max/mean 类池化——选择依据？
**A**：sum 对"多像素证据累积"友好（远处目标像素少，多档弱证据累加成强证据），且可微、梯度均匀；max 保最强响应但对噪声敏感、多峰信息丢失。工程上两者都有用，BEV 检测框架多随头设计而定。

**Q4**：为什么 BEVDet 要在 BEV 空间再做数据增广？
**A**：视锥投影后 BEV 特征的分布与透视域不同（近密远疏 + 池化洞），仅图像域增广无法覆盖 BEV 域的过拟合模式（如对特定 pillar 位置的记忆）。BEV 域旋转/翻转/缩放增广直接对称化 BEV 空间，论文中显著涨点。

---

## 本周交付清单

- [ ] 跑通 `w2_lss.py`，确认输出与文中一致（含例 4 手算数字逐个核对）。
- [ ] 闭卷手算 2 像素 × 3 深度档的外积视锥 + 沿深度求和还原。
- [ ] 写出 LSS 完整 shape 链（真实版 6 相机规模），并说出每步的物理含义。
- [ ] 能回答：为什么"非零 pillar 6/100"是稀疏路线的靶子？BEVDet 的双空间增广在补什么？
