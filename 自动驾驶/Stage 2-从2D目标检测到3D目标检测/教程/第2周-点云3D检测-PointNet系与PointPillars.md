# 第 2 周教程：点云 3D 检测——PointNet 系与 PointPillars

> **本周要回答的四个问题**
> 1. 3D 检测的输出长什么样——7 自由度框每个量是什么？和 2D 框换算关系如何？
> 2. 点云是**无序**的，CNN 吃的图像是**有序网格**——无序数据怎么进网络？
> 3. 体素化后做 3D 卷积为什么慢得离谱？稀疏卷积（sparse convolution）怎么救？
> 4. PointPillars（量产上车最多的 3D 检测结构）从点云到 3D 框的每一步 shape 是多少？anchor-free 的 heatmap 头又是怎么来的？

对应学习计划：第 2 周。交付物：NumPy 验证 PointNet 置换不变性；NumPy 模拟 pillar 化 + PFN + scatter 全流程 shape 链。

---

## 1. 3D 检测输出定义：7 自由度框

2D 检测输出 4 个数 $(x_1, y_1, x_2, y_2)$；3D 检测输出**7 个数**：

$$
(x,\ y,\ z,\ w,\ l,\ h,\ \text{yaw})
$$

**这行式子在干什么**：位置 3 个 + 尺寸 3 个 + 朝向 1 个，共 7 个自由度（Degrees of Freedom, DoF）：

- $(x, y, z)$：**框中心**在传感器/车体坐标系的位置（米）。注意各数据集约定不同：KITTI 的标注以**框底部中心**为锚点，nuScenes 以**几何中心**为锚点——跨数据集移植模型时忘掉这个差异，框会凭空"浮起或下沉"半米；
- $(w, l, h)$：宽、长、高（米）。约定 $l$ 沿目标朝向（车头方向）、$w$ 垂直朝向、$h$ 沿重力方向；
- $\text{yaw}$：朝向角（弧度），目标绕重力轴（z 轴）的旋转。只需要这一个角——车不会飞，俯仰/横滚由地面近似约束掉了，所以是 7 而不是 9 自由度。

**结果拿来做什么**：第 1 周的 2D 度量衡全部升级——IoU 变成 BEV/3D IoU（第 4 周手写），NMS 在 BEV 平面做，mAP 协议由数据集定。模型要回归的量从 4 维变成 7 维，且 $\text{yaw}$ 是角度（有周期性 $-\pi \sim \pi$，回归时要用 $\sin/\cos$ 编码避免 $2\pi$ 跳变，第 3 周 CenterPoint 处理）。

---

## 2. 点云直接检测的挑战：三座大山

Stage 1 已经知道点云是 $N×4$ 的无序集合 $(x, y, z, \text{intensity})$。它和图像有本质差异，给检测带来三座大山：

1. **无序性（permutation）**：$N$ 个点没有固定顺序，交换两个点的存储位置，物理上还是同一片点云。图像里交换两行像素就是另一幅图；点云里任何依赖"点的输入顺序"的网络都失效——网络必须对点的排列**置换不变（permutation invariant）**。
2. **稀疏且不均匀**：点云在 3D 空间里极度稀疏（一个 51.2m×51.2m×6.4m 的检测范围约有 10⁷ 个 0.1m 体素，非空的往往不到 5%），且近密远疏。密集 3D 卷积在空体素上浪费绝大部分计算。
3. **分辨率不均**：同一个目标 5m 处有几千个点、50m 处可能只有几十个点，特征质量随距离剧烈变化。

针对这三座山，演化出三条路线：**直接处理点**（PointNet 系）、**体素化后稀疏卷积**（VoxelNet → SECOND）、**柱体化后 2D 卷积**（PointPillars）。本周按这个顺序讲。

---

## 3. PointNet：对称函数搞定无序性

### 3.1 结构：共享 MLP + max pooling

PointNet（CVPR 2017，Qi et al.）是第一个直接吃原始点云的网络。结构只有两句话：

1. 每个点独立过**同一个**多层感知机（共享 MLP，shared MLP）——权重对所有点相同，天然与点的数量和顺序无关；
2. 对所有点的特征**逐通道取 max**，把 $N$ 个点压缩成一个全局特征向量。

$$
f(P) = \underbrace{\max_{i=1,\dots,N}}_{\text{对点维取 max}}\ \ \gamma\big(\ \{\, h(x_i) \mid x_i \in P \,\}\big)
$$

**这个公式在干什么**（逐符号拆解）：

- $P$：点集 $\{x_1, \dots, x_N\}$，$x_i \in \mathbb{R}^3$（或加 intensity 成 $\mathbb{R}^4$），**集合**而非序列——公式里没有下标顺序的任何假设；
- $h(\cdot)$：共享 MLP，把每个点映射成一个特征向量（论文中 $h$ 输出 1024 维）；
- $\{\cdot\}$：把所有点的特征装进一个集合；
- $\max_{i}$：**对称函数（symmetric function）**——对 $N$ 个特征向量逐通道取最大值。max 满足 $\max(a, b) = \max(b, a)$，所以无论点的顺序怎么排，输出逐元素相同；
- $\gamma$：对称函数之后的普通 MLP，对全局特征做进一步变换。

**为什么 max 能兼顾"不变性"与"信息量"**：max 是"每个特征通道上的最大激活"——直觉上网络学到的是"检测器"：某通道对"车顶平面"响应，只要 1024 个点里有一个点激活它，全局特征就记下了"这辆车存在"。max 同时起了**特征选择**作用（每个通道只留最强响应），这也是论文实验里 max 明显优于 sum/mean 的原因——mean 会被无关点稀释，max 不会。

（PointNet 还有一个 T-Net 小模块预测仿射变换矩阵对输入做姿态对齐，一句话带过；PointNet++ 在此基础上补"局部性"：最远点采样（Farthest Point Sampling, FPS）挑中心点 + 球查询（ball query）取邻域点，再在每个邻域内递归跑 PointNet——层级化提取局部特征，解决了原版只有全局特征、对局部细节不敏感的问题。）

### 3.2 NumPy 验证置换不变性

```python
import numpy as np

rng = np.random.default_rng(0)

def mlp_and_maxpool(points, W1, b1, W2, b2):
    """对 (N,3) 点集做共享 MLP（两层），再对点维做 max，输出 (64,) 全局特征。"""
    h = np.maximum(points @ W1 + b1, 0)   # (N, 32) ReLU，每个点独立过同一组权重
    f = np.maximum(h @ W2 + b2, 0)        # (N, 64)
    return f.max(axis=0)                  # (64,) 逐通道取 N 个点中的最大值

# 随机固定一组"共享权重"（相当于训练好的 PointNet）
W1 = rng.normal(0, 0.5, size=(3, 32))
b1 = rng.normal(0, 0.1, size=32)
W2 = rng.normal(0, 0.5, size=(32, 64))
b2 = rng.normal(0, 0.1, size=64)

# 一片"车顶"点云：N=1024 个点
pts = rng.normal(0, 0.5, size=(1024, 3))
pts[:, 2] += 1.6                    # 抬高模拟车顶高度

feat = mlp_and_maxpool(pts, W1, b1, W2, b2)

# 实验1：随机打乱点的顺序（置换）
perm = rng.permutation(len(pts))
feat_perm = mlp_and_maxpool(pts[perm], W1, b1, W2, b2)
d1 = np.abs(feat - feat_perm).max()
print(f"实验1 随机置换 1024 个点后，全局特征最大差异 = {d1:.2e}")
assert d1 < 1e-12

# 实验2：完全逆序
feat_rev = mlp_and_maxpool(pts[::-1], W1, b1, W2, b2)
d2 = np.abs(feat - feat_rev).max()
print(f"实验2 完全逆序后，全局特征最大差异          = {d2:.2e}")
assert d2 < 1e-12

# 对照：max/mean 是对称函数，但"取第一个点"不是——它的结果依赖输入顺序
feat_first = np.maximum(pts[0] @ W1 + b1, 0) @ W2 + b2   # 只取第 1 个点
feat_first_rev = np.maximum(pts[::-1][0] @ W1 + b1, 0) @ W2 + b2
d3 = np.abs(feat_first - feat_first_rev).max()
print(f"对照  '取第一个点'在逆序后差异（非对称）      = {d3:.2e}（≠0，会因顺序变化）")
assert d3 > 1e-6
print("验证通过：max pooling 是对称函数，PointNet 输出与点的输入顺序无关")
```

**真实运行输出**：

```
实验1 随机置换 1024 个点后，全局特征最大差异 = 0.00e+00
实验2 完全逆序后，全局特征最大差异          = 8.88e-16
对照  '取第一个点'在逆序后差异（非对称）      = 2.44e+00（≠0，会因顺序变化）
```

实验 1 差异精确为 0（max 不看顺序）；实验 2 的 8.88e-16 是浮点累加顺序带来的机器精度噪声，物理上为 0；对照组证明**不变性来自最后那步对称函数**，而不是共享 MLP 本身——如果最后一步换成依赖顺序的操作，一切归零。

---

## 4. VoxelNet 到 SECOND：为什么要稀疏卷积

### 4.1 VoxelNet：体素化 + 3D 卷积

VoxelNet（CVPR 2018，Zhou & Tuzel, Apple）把点云切成小立方体（voxel，体素），每个体素内用一个小 PointNet（Voxel Feature Encoding，体素特征编码）聚成一个特征向量，然后在体素网格上做 **3D 卷积**。思路自然，但代价致命：检测范围 0.1m 体素化后是几百 × 几百 × 几十 的 3D 网格，**3D 卷积核是立体的**（如 $3×3×3$），单层计算量比同尺寸 2D 卷积高一个"深度"维度，且绝大多数计算发生在空体素上——点云非空体素占比常不到 5%。

### 4.2 SECOND：稀疏卷积

SECOND（Sensors 2018, Yan et al.）的洞见：**既然非空体素占比 <5%，就只在非空体素上存数据、算卷积**——这就是稀疏卷积（sparse convolution）。数据结构从稠密张量变成"坐标 + 特征"列表（hash 索引），卷积输出只计算"输入感受野内有非空体素"的输出位置。数学上与稠密卷积在非空位置完全等价，计算量却只与非空体素数成正比。SECOND 还实现了 GPU 友好的哈希查表，把 3D 卷积从"跑不动"变成"能上车"，后续所有 voxel-based 方法（Voxel R-CNN、CenterPoint 的 voxel 版本等）的 3D backbone 都建立在它之上。

但稀疏卷积的 GPU 实现（不规则内存访问、动态索引）工程门槛高、在某些车载芯片上优化困难——这就给了"干脆不用 3D 卷积"的 PointPillars 上位的机会。

---

## 5. PointPillars 全流程拆解（本阶段最重要）

PointPillars（CVPR 2019，Lang et al., nuTonomy/Aptiv）的口号：**把整个 3D 检测网络变成纯 2D 卷积**。论文报告 KITTI 上 62 Hz 推理，比当时方法快 2-4 倍，且精度不降——这个速度让它至今仍是量产方案与部署基准的主力结构。

### 5.1 核心想法：pillar = 不切 z 的 voxel

体素化慢的根源是 z 维（高度）带来的 3D 卷积。PointPillars 干脆**不沿 z 切分**：只在 BEV 平面按 0.2~0.5m 的网格划分，每一竖条点云（从地面到天空的一根"柱子"）叫一个 pillar（柱体）。柱内用一个小 PointNet（PFN）聚特征，然后 scatter 回 2D 网格——得到一张"伪图像（pseudo-image）"，后面接标准 2D 卷积。三段式：

$$
\underbrace{\text{点云}}_{(N,4)} \xrightarrow{\text{pillar 化 + PFN}} \underbrace{\text{伪图像}}_{(C, H, W)} \xrightarrow{\text{2D backbone}} \xrightarrow{\text{SSD 头}} \underbrace{\text{3D 框}}_{7\text{DoF}}
$$

### 5.2 逐步追踪 shape 链（NumPy 模拟）

```python
import numpy as np

rng = np.random.default_rng(1)

# ---------- 0. 模拟一片点云：1000 个点，x∈[0,51.2], y∈[-25.6,25.6]（KITTI 常用范围） ----------
N = 1000
pts = np.empty((N, 4))
pts[:, 0] = rng.uniform(0, 51.2, N)        # x 前
pts[:, 1] = rng.uniform(-25.6, 25.6, N)    # y 左
pts[:, 2] = rng.uniform(-1.7, 2.5, N)      # z 上
pts[:, 3] = rng.uniform(0, 1, N)           # intensity

# ---------- 1. pillar 化：0.5m x 0.5m 网格 ----------
pill = 0.5
# 每个点落入哪个 pillar：ix = floor(x / 0.5) ∈ [0, 102)，iy 同理
ix = np.floor(pts[:, 0] / pill).astype(int)
iy = np.floor((pts[:, 1] + 25.6) / pill).astype(int)
ix = np.clip(ix, 0, 101)     # x=51.2 恰好落在边界上时 floor 会得到 102，夹回网格内
iy = np.clip(iy, 0, 101)
print(f"点云 shape: {pts.shape} -> pillar 索引 shape: {ix.shape}")
print(f"x 方向 pillar 数 = 51.2/0.5 = {51.2 // pill:.0f}, y 方向 = 51.2/0.5 = {51.2 // pill:.0f}")

pillar_id = ix * 102 + iy                       # 二维网格拉平成一维 ID
uniq, inverse = np.unique(pillar_id, return_inverse=True)
M = len(uniq)
print(f"非空 pillar 数 M = {M}（网格总数 102x102={102*102}，稀疏！绝大多数是空的）")

# ---------- 2. 每个非空 pillar：取最多 max_p 个点，补齐/填充 ----------
max_p = 16
D = 9                                           # 特征维度 x,y,z,r + 4 个 xc,yc,zc + x_off,y_off
feats = np.zeros((M, max_p, D))
feats[:] = np.nan                               # 用 NaN 标记"填充点"，实际框架用 0 并靠 mask 屏蔽
for k in range(M):
    idx = np.where(inverse == k)[0][:max_p]     # 超过 16 个点随机截断（论文用随机采样）
    n = len(idx)
    center = pts[idx, :3].mean(axis=0)          # pillar 内点集的算术中心 (xc,yc,zc)
    feats[k, :n, 0:4] = pts[idx]                # x,y,z,r
    feats[k, :n, 4:7] = center                  # xc,yc,zc（pillar 内广播，所有点相同）
    feats[k, :n, 7:9] = pts[idx, :2] - center[:2]  # 点到中心的偏移 x-xc, y-yc
print(f"聚合后张量 shape: (M, max_p, D) = {feats.shape}   ← 论文中的 (D, P, N) 转置版")

# ---------- 3. 简化版 PFN：线性层 + ReLU + 逐 pillar max ----------
# 固定随机权重模拟"学到的"线性层，把 D=9 维映到 C=8 维
C = 8
valid = ~np.isnan(feats[..., 0])                # (M, max_p) 有效点 mask
f = np.nan_to_num(feats)                        # (M, max_p, 9)
W = rng.normal(0, 0.5, size=(D, C)); b = rng.normal(0, 0.1, size=C)
h = np.maximum(f @ W + b, 0)                    # (M, max_p, 8) 共享线性层（逐点）
h[~valid] = 0                                   # 填充点置 0，max 时被忽略
pillar_feat = h.max(axis=1)                     # (M, 8) 对点维 max —— 又是 PointNet 的对称函数！
print(f"PFN 输出 shape: (M, C) = {pillar_feat.shape}")

# ---------- 4. scatter 回网格：生成伪图像 ----------
H, Wg = 102, 102
pseudo = np.zeros((C, H, Wg))
gx, gy = uniq // 102, uniq % 102                # 拉平 ID 还原成网格坐标
pseudo[:, gx, gy] = pillar_feat.T               # 只有 M 个位置非零
print(f"伪图像 shape: (C, H, W) = {pseudo.shape}  ← 之后直接接 2D 卷积 backbone")
print(f"伪图像非零通道占比: {(pseudo != 0).mean() * 100:.1f}%（稀疏性延续到 2D 特征图）")

# ---------- 断言 ----------
assert feats.shape[1:] == (max_p, D)
assert pillar_feat.shape == (M, C)
assert pseudo.shape == (C, H, Wg)
print("验证通过：点云 (N,4) -> pillar 张量 (M,16,9) -> PFN (M,8) -> 伪图像 (8,102,102)")
```

**真实运行输出**：

```
点云 shape: (1000, 4) -> pillar 索引 shape: (1000,)
x 方向 pillar 数 = 51.2/0.5 = 102, y 方向 = 51.2/0.5 = 102
非空 pillar 数 M = 961（网格总数 102x102=10404，稀疏！绝大多数是空的）
聚合后张量 shape: (M, max_p, D) = (961, 16, 9)   ← 论文中的 (D, P, N) 转置版
PFN 输出 shape: (M, C) = (961, 8)
伪图像 shape: (C, H, W) = (8, 102, 102)  ← 之后直接接 2D 卷积 backbone
伪图像非零通道占比: 3.5%（稀疏性延续到 2D 特征图）
验证通过：点云 (N,4) -> pillar 张量 (M,16,9) -> PFN (M,8) -> 伪图像 (8,102,102)
```

### 5.3 逐模块解读

**（1）pillar 化**：每个点用两次 `floor` 找到自己的柱子。整个检测范围 51.2m × 51.2m ÷ 0.5m = 102 × 102 = 10404 个格子，只有 M 个非空（真实点云里 M ≈ 1~2 万，占比 5~10%）。**注意 pillar 数 M 是逐帧变化的**——这就是 pillar 化与图像不同的地方：输入不是固定 shape，需要scatter/gather 这类"不规则"操作。

**（2）增强特征 D=9**：原始 4 维 $(x, y, z, r)$ 之外，补 5 维——柱内点集的算术中心 $(x_c, y_c, z_c)$（同柱内广播，让网络知道"这根柱子在哪"）与点到中心的偏移 $(x - x_c, y - y_c)$（柱内相对几何）。这 9 维是论文明确给出的特征；缺了"柱子在哪"的信息，PFN 学到的特征将无法区分"前排的车"与"后排的车"。

**（3）PFN（Pillar Feature Network）**：每个 pillar 内部就是一个 mini PointNet——逐点过共享线性层（+BN+ReLU），再对点维做 max。每个 pillar 截断/填充到固定 `max_p`（论文 100，演示用 16）个点，保证张量形状规整可以拼 batch。**第 3.2 节的置换不变性在此复用：柱内点序无关紧要。**

**（4）scatter 回网格（生成伪图像）**：把 M 个 pillar 特征按各自 (ix, iy) 放回 2D 网格，空格子填 0。从此"无序点云"变成了"有空间的 2D 图像"——CNN 的世界欢迎你。

**（5）2D backbone + SSD 头**：backbone 是自顶向下+自底向上的两路结构（top-down 分支逐级下采样提语义、second 分支上采样回原分辨率后逐级拼接，思想上就是第 1 周的 FPN），输出 3 个分辨率的 2D 特征图；检测头用单次多框检测器（Single Shot MultiBox Detector, SSD，ECCV 2016）风格：在每个网格位置放 anchor（BEV 里对每个类别放 0° 和 90° 两个朝向的框），直接回归修正量 + 类别。损失 = 分类（focal loss）+ 回归（Smooth L1，对 (x,y,z,w,l,h,yaw) 与方向分类）。

**完整 shape 链**（KITTI 典型配置，C=64）：

$$
(N, 4) \to (M, 100, 9) \to (M, 64) \to (64, 512, 512) \to \cdots \to (A \times (7+2+1),\ 512,\ 512)
$$

最后一项是 SSD 头输出：每个 anchor 7 个框回归量 + 2 个朝向 bin 分类 + 1 个类别得分（A = 每位置 anchor 数）。

### 5.4 为什么量产选它

- **全是 2D 卷积**：不需要 SECOND 那套稀疏卷积 CUDA 算子，任何芯片的 2D conv 加速器都吃得很饱；
- **延迟低且稳定**：62 Hz（论文数字）意味着 16ms 内出结果，余量给下游跟踪规划；
- **精度够用**：在 BEV 感知的精度-速度曲线上，PointPillars 至今仍是最优性价比点之一。

---

## 6. anchor-free heatmap 头：CenterPoint 的由来

SSD 式 anchor 头继承自 2D，用起来有三大痛点：anchor 尺寸/朝向要按类别手工设计（卡车和行人差一个数量级）；正负样本分配逻辑繁琐；输出通道里 anchor 维度把计算翻倍。CenterPoint（CVPR 2021）把 2D 里的 CenterNet 思路搬进 BEV：**不要 anchor，直接在 BEV 特征图上预测"物体中心热图"**——每个类别一张热图，物体中心处激活值接近 1，背景接近 0；再在峰值位置回归出尺寸/朝向/高度/速度。柱内点序无关 → BEV 峰值处逐点解码，第 3 周完整拆解这条路线。

> **动手验证**：把 5.2 代码里的 `max_p` 改成 4 再跑——非空 pillar 里超过 4 个点的柱会被截断，伪图像非零占比略降但 shape 链不变；把 `pill` 改成 0.2——网格变成 256×256，体会"pillar 尺寸 = 空间分辨率 vs 计算量"的权衡。
>
> **一句话总结**：PointPillars = 把 PointNet 的对称函数用在"每根竖柱"上聚特征 → scatter 成伪图像 → 之后的一切都是成熟 2D 检测（FPN 思想 backbone + anchor 头）。

---

## 7. 工程权衡与常见坑

1. **z 范围与 pillar 高度信息丢失**：pillar 不分高度，上下两辆车（如立交桥）在 BEV 上不可分；z 信息只存在于每个点的 z 值和 z_c 里。悬空/多层场景要回到 voxel-based。
2. **max_p 截断偏差**：地面点柱（一根柱里几百个地面点）被截到 max_p 后全是地面点，柱特征被地面主导——所以前处理普遍先做地面去除（Stage 1 工具），否则远处小目标特征被地面淹没。
3. **坐标系转错**：pillar 索引 `floor((y + y_min)/pill)` 里的 `y_min` 偏移项漏加，网格整体错位半个场景，loss 不收敛但报错却没有——第一排查项。
4. **intensity 未归一化**：不同雷达 intensity 量纲不同（0-1 / 0-255），训练推理不一致时 BEV 特征整体漂移。
5. **验证 shape 链时 M 逐帧变化**：拼 batch 时 pillar 数不同，需要 pad 或使用 scatter 算子——自研部署代码最常见崩溃点。

---

## 8. 延伸思考题（含解析）

**Q1**：为什么 PointPillars 不需要稀疏卷积也不慢？
**A**：因为它把 3D 卷积整个绕开了：pillar 化 + PFN 是逐点小 MLP（计算量 O(N·D·C)），scatter 后的伪图像用稠密 2D 卷积（有成熟加速器）。代价是放弃了 z 维结构，用"每点带 z 的特征"弥补。

**Q2**：PFN 里如果用 mean 替换 max，会发生什么？
**A**：仍然是置换不变的（mean 也是对称函数），但柱特征变成"柱内平均"——一根柱里 100 个地面点 + 1 个车顶点，mean 几乎全是地面；max 则保住车顶的强响应。max 的特征选择能力是它胜出的原因。

**Q3**：pillar 大小从 0.5m 改成 0.2m，计算量和精度怎么变？
**A**：网格 102×102 → 256×256（BEV 特征图面积 6.25 倍，2D 卷积计算量同步涨）；空间分辨率提高，远处小目标（行人 0.5m 宽）不再与地面点共柱，精度通常提升。量产常在 0.2~0.4m 之间按算力预算选。

**Q4**：7 自由度里为什么没有 pitch/roll？
**A**：路面目标近似直立，pitch/roll 由地面法向确定（约 0），标注与回归它们的信息增益极低；个别场景（坡道）通过把坐标旋转到局部地面系隐式处理。9 自由度只在空中目标检测中需要。

---

## 本周交付清单

- [ ] 跑通置换不变性实验：max pooling 在随机置换/逆序下差异 < 1e-12，对照"取第一个点"差异显著。
- [ ] 跑通 pillar 化模拟：`(1000,4) → (961,16,9) → (961,8) → (8,102,102)`，并理解每一步为什么是这个 shape。
- [ ] 把 `pill` 改 0.2、`max_p` 改 4 各跑一次，记录 shape 与非零占比变化。
- [ ] 能不看资料复述：对称函数是什么、稀疏卷积省在哪、PointPillars 三段式各模块的输入输出 shape。
