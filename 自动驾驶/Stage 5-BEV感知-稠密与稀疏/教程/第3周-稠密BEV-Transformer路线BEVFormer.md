# 第 3 周教程：稠密 BEV 路线 II——Transformer 与 BEVFormer

> **本周要回答的四个问题**
> 1. BEVFormer 的 BEV query 是什么？"格子主动问图像"和 LSS 的"像素撒向格子"方向差在哪？
> 2. Spatial Cross-Attention 的公式逐符号是什么？多相机重叠、视野外的格子怎么处理？
> 3. 时序融合怎么做？ego 在动，上一帧的 BEV 特征怎么"搬"到当前帧对齐？
> 4. BEVFormer 和 DETR 是什么关系？LSS 系 vs Transformer 系到底怎么选？

对应学习计划：第 3 周。交付物：NumPy 模拟 1 个 BEV query 柱（4 个高度参考点）投影到 3 个虚拟相机做带 mask 的 cross-attention；验证注意力权重和为 1；实现 ego 运动 warp 的格子对齐计算。

---

## 1. 方向性反转：从"像素撒"到"格子问"

LSS 的信息流：**图像是主体**，每个像素带着深度分布撒向 BEV 网格——网格被动接收。BEVFormer（Li et al., ECCV 2022, [arXiv:2203.17270](https://arxiv.org/abs/2203.17270)，官方代码 [fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer)）把方向反过来：**BEV 网格是主体**。

### 1.1 BEV query：可学习的"格栅问句"

定义一组可学习参数 $Q \in \mathbb{R}^{H_B \times W_B \times C}$（nuScenes 惯例 $H_B = W_B = 200$，C=256），每个元素对应一根 BEV 柱，称为 **BEV query**。把它读成一句"问话"：

> "我是 ego 系里 (x=20.5 m, y=-15.4 m) 这一格。谁经过这里？长什么样？过去几帧有没有动？"

回答问话的方式不是让像素来投（Lift-Splat），而是**每个 query 主动去 6 个相机的特征图上找证据**——找哪里、信多少，由注意力机制学出来。这个"主动问"的设计直接继承自 DETR（Carion et al., ECCV 2020）：DETR 用 ~100 个 object query 去图像里"问目标"，BEVFormer 把 query 数量放大到 200×200 = 40000 个、语义从"问目标"换成"问空间格"。所以面试里"BEVFormer 和 DETR 什么关系"的标准答案：**同一种 query-based 集合表征，query 从目标级换成空间网格级，注意力从单相机换成多相机 + 多时刻**。

### 1.2 整体结构

$$
\text{BEV queries} \;\Big[\; \text{Temporal Self-Attention} \to \text{Spatial Cross-Attention} \to \text{FFN} \;\Big] \times 3 \; \to \; \text{BEV 特征}
$$

nuScenes 配置下（官方 `bevformer_base.py`）：BEV 200×200（x/y ∈ [-51.2, 51.2] m @ 0.512 m）、输入 6 相机 704×1600、编码器 **3 层**（每层先时序后空间）。BEV 特征出来后接 CenterPoint-style 检测头（同 Stage 2），外加地图分割头。

---

## 2. Spatial Cross-Attention（SCA）公式逐符号

这是 BEVFormer 的视角转换核心。问题：BEV 柱 (i, j) 应该看 6 张图的哪些像素？

### 2.1 第一步：柱上参考点投影（把"看哪里"定下来）

BEVFormer 沿每根柱取 **4 个不同高度的参考点**（`num_points_in_pillar=4`，官方配置值；柱高范围 -5 m 到 3 m，取 4 个采样高度）。参考点 $p_k$（k=1..4）是 ego 系 3D 点：

$$
p_k = \big(x_i,\; y_j,\; z_k\big) \quad \text{其中 } (x_i, y_j) \text{ 是柱的平面中心，} z_k \text{ 是第 k 个高度}
$$

**为什么要在柱上取多个高度而不是柱心一个点**：上周反投影公式告诉我们，一个像素对应一条射线；反过来，一个 3D 点只对应图像上一个像素。柱是"竖直的一串空间"，不同高度的物体（车顶、车身、车轮）落在图像的不同行上——**只取一个高度会漏掉柱内其他高度的物体**。4 个高度就是把柱"竖着采样 4 次"。

每个参考点用标定矩阵正投影到 6 个相机（上周公式反向用）：

$$
(u_{k,n}, v_{k,n}) \;\propto\; K_n \big( R_n \cdot p_k + t_n \big), \quad n = 1, \dots, 6
$$

**逐符号**：$K_n, R_n, t_n$ 是第 n 个相机的内参/ego→相机外参（Stage 3 的产物，冻结常量）；$(u_{k,n}, v_{k,n})$ 是参考点在第 n 张图上的像素坐标。投影后做两道筛选（**masked attention**）：

1. **视野 mask**：像素落在图像边界外 → 该相机对该参考点不可见，剔除；
2. **相机 mask**：官方实现只保留参考点命中的相机（每个 3D 点通常只被 1~2 个相机看到，环视相机视场互斥大部分区域）。

**结果**：4 个高度 × 命中相机数 = 每根柱最终只有**少量有效采样位置**。论文原文描述：一根柱通常只被 1~2 个相机看到，SCA 只在这些相机的投影位置上计算——被所有相机都看不到的柱不更新。

### 2.2 第二步：在命中位置做 deformable attention

$$
\text{SCA}(Q_{i,j}) = \frac{1}{|\mathcal{V}_{i,j}|} \sum_{n \in \mathcal{V}_{i,j}} \sum_{k=1}^{4} \text{DeformAttn}\big(Q_{i,j},\; \mathcal{P}(p_k, n),\; F_n\big)
$$

**逐符号讲清这个公式在干什么**：

- $Q_{i,j} \in \mathbb{R}^{C}$：第 (i, j) 根柱的 query 向量；
- $\mathcal{V}_{i,j}$：该柱**有效相机集合**（投影命中的相机 id 集合）——这就是"只对命中的相机算注意力"的形式化；
- $n \in \mathcal{V}_{i,j}$ 外层求和：对每个命中相机分别收集证据；
- $k = 1..4$ 内层求和：柱上 4 个高度参考点各自贡献一份证据；
- $\mathcal{P}(p_k, n)$：参考点 $p_k$ 在第 n 张图上的投影像素位置（2.1 的产物）；
- $F_n \in \mathbb{R}^{H' \times W' \times C}$：第 n 相机的图像特征图（backbone 输出，704×1600 下采样若干倍）；
- $\text{DeformAttn}$：**可变形注意力**（deformable attention，出自 Deformable DETR, Zhu et al., ICLR 2021）——不做全图 attention，只在参考位置 $\mathcal{P}$ 附近采少数点、并让网络预测每个采样点的注意力权重。直觉：query 拿着自己的"问句"到指定像素周围小邻域内"凑近看"，权重由"问句 × 附近特征"算出；
- 外层 $1/|\mathcal{V}_{i,j}|$：对命中相机数取平均，使梯度尺度不因可见相机多少而漂移。

**全部拼起来的直觉**：一根柱 = "我这里（x, y）这个竖直空间里有什么？"→ 取 4 个高度、投影到能看到它的相机 → 每个投影点附近采样图像特征 → 按相关性加权 → 平均回 query。**LSS 是"把特征撒过去"，SCA 是"把证据收回来"**——同一次坐标变换的两种用法，方向相反。

### 2.3 与 LSS 的三个关键差异

| 维度 | LSS | BEVFormer SCA |
| --- | --- | --- |
| 深度 | 显式 41 档分布，撒满整条射线 | 隐式：采样点固定在柱的 4 个高度上，"深度对不对"由注意力权重与训练信号隐式修正 |
| 算子 | 大规模 scatter/索引（custom CUDA 友好） | deformable attention（生态标准算子，A100/Orin 都有优化） |
| 相机间交互 | 各相机独立撒，重叠区在池化时叠加 | 一根柱显式跨相机收集证据 |

---

## 3. Temporal Self-Attention：时序对齐

### 3.1 问题：ego 在动，世界也没动

BEV 感知要回答"横穿行人"这类问题必须有速度信息——单帧 BEV 特征里"位置"有、"速度"没有。BEVFormer 的方案：**把上一帧算好的 BEV 特征搬进当前帧一起算**（recurrent，论文称 temporal fusion 跨 8 帧）。但两帧的 BEV 网格是"贴在车上"的坐标系——车前进 3.4 m 后，同一辆车在网格里的格子索引变了。直接把两帧特征按索引做 self-attention 是**错位**的。

### 3.2 解法：ego-motion warp（运动补偿）

$$
B^{t-1}_{\text{aligned}}(i, j) = B^{t-1}\big(g^{-1}(i, j)\big)
$$

**逐符号**：$B^{t-1}$ 是上一帧 BEV 特征；$g$ 是本次 ego 运动对应的网格坐标变换（由定位/里程计给出位移与 yaw）；$g^{-1}(i,j)$ 是"当前帧格子 (i,j) 的内容在上一帧网格里的位置"——通常非整数，实现上用双线性插值（grid_sample）。第 5 节代码实测：上一帧格子 (100,100)，ego 前进 (3.4, -0.2) m，该格内容对应当前帧 (93.36, 100.39)——**车向前开，旧内容在当前网格索引上向后移**。这个操作就是 Stage 1 学的"多帧叠加需要位姿变换"在特征层面的版本，也和 Stage 2 CenterPoint 的时序做法（把前帧点云 warp 后拼接）同一个思想：**车动了、世界没动，先把坐标系统一**。

### 3.3 warp 之后的 Temporal Self-Attention

对齐后，当前帧每根柱的 query 与**同一格子索引**上对齐后的上一帧特征做 self-attention：

$$
\text{TSA}(Q_{i,j}^{t}) = \text{softmax}\left(\frac{Q_{i,j}^{t} K^T}{\sqrt{C'}}\right) \begin{bmatrix} B^{t-1}_{\text{aligned}}(:, i, j) \\ Q_{i,j}^{t}\end{bmatrix} V
$$

**逐符号**：key/value 只取两处——当前柱自身与对齐后的同位置历史特征（**空间局部**：每根柱只看自己那一格的历史，不是全图注意力，这是省算力的关键设计）；$C'$ 是缩放维度；输出把历史信息"揉"进当前 query。直觉：柱在问"我这个格子上一刻有什么、现在有什么——它们是同一个东西在动吗？"跨帧的对应关系让网络能学出**速度**。

**失效模式**（官方实现注释同款提醒）：ego 高速/大转向时 warp 插值误差大，超出网格范围的格子直接置零；静止目标跨 warp 反而"动"了（对齐的是世界，目标自己也在动——动目标的时序建模是第 4 周稀疏路线用"目标级传播"解决得更好的点）。

---

## 4. 可运行代码：1 根 BEV 柱的 SCA + ego warp

```python
# w3_bevformer.py —— 第3周demo：Spatial Cross-Attention 缩小版模拟（机制演示，非真实模型）
# 1 个 BEV query 柱（4 个参考点）投影到 3 个虚拟相机，命中的做注意力加权，未命中被 mask。
# 所有随机量固定种子。
import numpy as np

rng = np.random.default_rng(42)

# ---------- 例1：BEV 柱与参考点（num_points_in_pillar=4，官方配置同款参数） ----------
pillar_xy = (20.0, 5.0)                    # 该 BEV 格子的 ego 平面坐标（米）
zs = np.array([-1.0, 0.5, 2.0, 3.5])       # 柱上 4 个不同高度的参考点
print("例1 BEV 柱 @ (x,y) =", pillar_xy, " 参考点高度 z =", zs)

# ---------- 例2：3 个虚拟相机，统一构造 ego->cam 变换 ----------
fx = fy = 800.0; cx, cy = 800.0, 450.0; W_img, H_img = 1600, 900
# 相机系：x 右、y 下、z 前；ego 系：x 前、y 左、z 上。
# cam = T @ M^T @ (p - cam_pos)：
#   M   = 把 ego 轴转到"相机前向对齐"系（列 = 前向/左/上 方向在 ego 下的表示）
#   T   = 对齐系 -> 相机系（cam_z=前向, cam_x=右=-左, cam_y=下=-上）
T = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)

def make_cam(pos, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    M = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)   # 列 = (前, 左, 上)
    return pos, T @ M.T

cams = [("CAM_FRONT",       (1.5, 0.0, 1.6), 0.0),        # 朝前
        ("CAM_FRONT_LEFT",  (0.5, 0.5, 1.6), np.pi/4),     # 朝左前 45°
        ("CAM_BACK",        (-2.0, 0.0, 1.6), np.pi)]      # 朝后

def ego_to_pixel(P, pos, R):
    Pc = R @ (P - np.array(pos))            # ego 3D 点 -> 相机系
    if Pc[2] <= 0:                          # 光学轴背后，投影无意义 -> mask
        return None
    return fx * Pc[0] / Pc[2] + cx, fy * Pc[1] / Pc[2] + cy  # 针孔投影

# ---------- 例3：Spatial Cross-Attention 投影与 mask ----------
pts = np.array([[pillar_xy[0], pillar_xy[1], z] for z in zs])   # 4 个 3D 参考点
hits = []                                  # (相机id, 参考点id, 像素坐标)
for cam_id, (name, pos, yaw) in enumerate(cams):
    for i, P in enumerate(pts):
        uv = ego_to_pixel(P, pos, make_cam(pos, yaw)[1])
        ok = uv is not None and 0 <= uv[0] < W_img and 0 <= uv[1] < H_img
        tag = "命中" if ok else ("背后" if uv is None else "出画")
        loc = f"({uv[0]:6.0f},{uv[1]:6.0f})" if uv else "   无投影   "
        print(f"  参考点 z={zs[i]:+.1f} -> {name}: {loc} [{tag}]")
        if ok:
            hits.append((cam_id, i, uv))
print(f"例3 共 {len(pts)}x{len(cams)} 次投影尝试，命中 {len(hits)} 个（其余被 mask，不参与注意力）")

# ---------- 例4：对命中的采样特征做注意力加权 ----------
d_model = 8
q = rng.normal(0, 1, d_model)              # 该 BEV 柱的 query 向量（可学习参数，模拟）
img_feat = [rng.normal(0, 1, (H_img//100, W_img//100, d_model))  # 缩小的虚拟特征图
            for _ in range(3)]
K_list, V_list = [], []
for cam_id, i, (u, v) in hits:
    r, c = int(v // 100), int(u // 100)    # 简化：最近邻"采样"该位置特征
    K_list.append(img_feat[cam_id][r, c])  # key = 采样位置特征
    V_list.append(img_feat[cam_id][r, c])  # value = 同一特征（真实实现有线性投影）
K = np.array(K_list); V = np.array(V_list)
attn = np.exp(K @ q / np.sqrt(d_model))    # 打分 = q·k/√d，先 exp
attn /= attn.sum()                         # softmax：只在命中的采样上归一化
out = (attn[:, None] * V).sum(0)           # 加权和 -> 该 BEV 柱的更新特征
print(f"\n例4 命中采样数 = {len(hits)}, 注意力权重 = {np.round(attn, 3)}, 和 = {attn.sum():.4f}")
print("BEV 柱更新后特征（前3维） =", np.round(out[:3], 3))

# ---------- 例5：Temporal Self-Alignment——上一帧 BEV 按 ego 运动对齐到当前帧 ----------
dx, dy = 3.4, -0.2                         # 两帧间 ego 位移（10Hz 下的典型量级）
bev_res, bev_size = 0.512, 200             # nuScenes 惯例 200x200 @ 0.512 m
i0, j0 = 100, 100                          # 上一帧某格
x_prev = (i0 - bev_size/2 + 0.5) * bev_res
y_prev = (j0 - bev_size/2 + 0.5) * bev_res
x_cur, y_cur = x_prev - dx, y_prev - dy    # 车动了、世界没动：格内容对应新坐标
i1 = x_cur / bev_res + bev_size/2 - 0.5
j1 = y_cur / bev_res + bev_size/2 - 0.5
print(f"\n例5 上一帧格子({i0},{j0}) 中心 ego=({x_prev:.2f},{y_prev:.2f}) m")
print(f"  ego 前进 (dx={dx}, dy={dy}) 后，该格内容对应当前帧格子 ({i1:.2f},{j1:.2f})")
print("  非整数 -> 真实实现用双线性插值取值（官方代码中即 by ego-motion warp）")

# ---------- 例6：Shape 链一览（BEVFormer 官方配置量级） ----------
n_cams, layers = 6, 3
print(f"\n例6 官方配置量级：{n_cams} 相机 x 704x1600 输入，BEV 200x200 = "
      f"{200*200} 个 query，编码器 {layers} 层，每层 SCA 每柱 {len(zs)} 个参考点投影到 6 相机")
```

**预期输出**（实测）：

```
例1 BEV 柱 @ (x,y) = (20.0, 5.0)  参考点高度 z = [-1.   0.5  2.   3.5]
  参考点 z=-1.0 -> CAM_FRONT: (   584,   562) [命中]
  参考点 z=+0.5 -> CAM_FRONT: (   584,   498) [命中]
  参考点 z=+2.0 -> CAM_FRONT: (   584,   433) [命中]
  参考点 z=+3.5 -> CAM_FRONT: (   584,   368) [命中]
  参考点 z=-1.0 -> CAM_FRONT_LEFT: (  1300,   573) [命中]
  参考点 z=+0.5 -> CAM_FRONT_LEFT: (  1300,   502) [命中]
  参考点 z=+2.0 -> CAM_FRONT_LEFT: (  1300,   431) [命中]
  参考点 z=+3.5 -> CAM_FRONT_LEFT: (  1300,   360) [命中]
  参考点 z=-1.0 -> CAM_BACK:    无投影    [背后]
  参考点 z=+0.5 -> CAM_BACK:    无投影    [背后]
  参考点 z=+2.0 -> CAM_BACK:    无投影    [背后]
  参考点 z=+3.5 -> CAM_BACK:    无投影    [背后]
例3 共 4x3 次投影尝试，命中 8 个（其余被 mask，不参与注意力）

例4 命中采样数 = 8, 注意力权重 = [0.138 0.023 0.023 0.355 0.144 0.144 0.042 0.132], 和 = 1.0000
BEV 柱更新后特征（前3维） = [-0.016 -1.11   0.27 ]

例5 上一帧格子(100,100) 中心 ego=(0.26,0.26) m
  ego 前进 (dx=3.4, dy=-0.2) 后，该格内容对应当前帧格子 (93.36,100.39)
  非整数 -> 真实实现用双线性插值取值（官方代码中即 by ego-motion warp）

例6 官方配置量级：6 相机 x 704x1600 输入，BEV 200x200 = 40000 个 query，编码器 3 层，每层 SCA 每柱 4 个参考点投影到 6 相机
```

**看懂这个输出**：

1. 例 3：同一根柱的 4 个高度，在 CAM_FRONT 上 u 恒为 584（同一水平位置），v 从 562 变到 368——**柱上的高度差就是图像上的行差**，这正是"柱要取多个高度"的几何证据；CAM_BACK 全部"背后"被 mask，对应真实场景中 3D 点只会被少数相机看到。
2. 例 4：8 个命中采样的注意力权重和精确为 1（masked softmax 的性质——**mask 只是把无效项从归一化集合里去掉，不破坏概率语义**）；权重差异大（0.023 到 0.355）说明"问句 q 与不同位置特征的匹配度"确实在起作用。
3. 例 5：格子索引移动 6.64 格 = 位移 3.4 m ÷ 0.512 m ≈ 6.64——**warp 的量纲自检**：索引位移 × 分辨率应等于 ego 位移，以后调 bug 时先查这个等式。

> **动手验证**：
> 1. 把 `pillar_xy` 改成 `(20.0, -8.0)`（右前方）：CAM_FRONT_LEFT 的投影应变到 u < 800 或出画，CAM_FRONT 仍在——用环视相机朝向核对。
> 2. 把例 5 的 dx 换成 -3.4（倒车）：格子索引应向 +x 方向移（93.36 → 106.6 左右）——方向感自查。
> 3. （进阶）读官方 `spatial_cross_attention.py`：核对 `reference_points` 的生成（pillar 采样 z 从 -5 到 3 分 4 点）与 `point_sampling` 中 `shift_reference_points` 的 ego→lidar→相机链路，和本 demo 的简化投影一一对应。

---

## 5. Shape 链全记录（官方配置量级）

$$
\underbrace{(6 \times 3 \times 256 \times 200 \times 200)}_{\text{BEV queries + 残差流}}
\xleftarrow{\text{SCA/TSA}}
\underbrace{(6, 256, H', W')}_{\text{6 相机特征图}}
\quad \to \quad
\underbrace{(200 \times 200 \times 256)}_{\text{BEV 特征}}
$$

- BEV query 数：200 × 200 = **40000**（可学习参数 40000×256 ≈ 10.2 M 个 float，这是"稠密"的参数代价之一）；
- 每层每柱 SCA 投影计算量：4 参考点 × 6 相机 = 24 次投影 + ~数次采样（实际命中后），3 层编码器 × 40000 柱 = 数百万次投影/帧——好在投影是查表级运算，deformable attention 在 GPU/NPU 上有标准算子；
- 官方 README 模型表（查证）：BEVFormer-base（R101-DCN）val 集 41.6% mAP / 51.7% NDS，**test 集 37.5% mAP / 56.9% NDS**（NDS 相比前 SOTA DETR3D 的 47.9% 提升 9.0 个点，论文摘要原文数据）；BEVFormer-large（VoV-99）test 集 41.6% mAP / 58.5% NDS。训练显存 base 28.5 GB、small 10.5 GB（官方模型表标注）——**Transformer 系训练比 LSS 系更吃显存**。

### 稠密路线对比表（LSS 系 vs Transformer 系）

| 维度 | LSS 系（BEVDet 为代表） | Transformer 系（BEVFormer 为代表） |
| --- | --- | --- |
| 深度建模 | 显式深度分布（可激光监督） | 隐式（藏在注意力里） |
| 训练显存 | 中（Tiny 可小卡） | 高（base 28.5 GB） |
| 算子友好度 | scatter/池化需定制优化，NPU 移植成本高 | deformable attention 生态成熟，移植性好 |
| nuScenes 成绩（自报） | BEVDet-Base val 38.4 mAP / 46.1 NDS | BEVFormer-base test 37.5 mAP / 56.9 NDS |
| 时序 | 后续版本拼接（BEVDet4D） | 原生 TSA/SCA recurrent 融合 |
| 部署生态 | 语义分割/占据栅格任务天然适配 | 检测头复用 CenterPoint 系成熟栈 |

（注：mAP/NDS 为各自论文/官方仓库自报，且 test/val 集不同，不能直接横向比大小；看趋势即可。）

---

## 6. 工程权衡与失效模式

### 6.1 权衡速查

- **BEV 分辨率**：0.512 m 的格子对 0.6 m 宽的行人只占 1~2 格，BEVFormer 的检测上限受网格分辨率约束——后续工作（如 BEVFormer v2、SOLOFusion）在输入分辨率与网格上继续加码；
- **3 层编码器**：每层都是全量 40000 query 的 TSA+SCA，层数与延迟线性相关；端侧复刻常砍到 1~2 层；
- **warp 依赖定位精度**：TSA 的对齐质量受里程计/定位误差直接影响——定位漂移时序融合反而引入拖影（与 Stage 3 第 4 周时间标定呼应：同步误差在 warp 里同样放大）。

### 6.2 典型失效模式

1. **视野交界柱信息缺失**：两相机边界处参考点恰好在两者边缘出画，柱没有证据 → BEV 图"竖缝"。**缓解**：相机覆盖重叠区设计 + mask 策略放宽（投影到多相机）。
2. **大转向帧 warp 错位**：yaw 变化大时双线性插值取到错误邻域。**症状**：转急弯时远处时序拖影。**缓解**：只用可靠位姿帧、限制跨帧数。
3. **注意力权重不可解释**：没有显式深度分布可监督，深度学错难定位。**缓解**：加深度辅助监督（如 BEVDepth 思路迁移）或用带显式深度的混合架构。

---

## 7. 延伸思考题（含解析）

**Q1**：SCA 为什么用"参考点附近的 deformable attention"而不是"全图 attention"？
**A**：全图 attention 的 key 数 = 6×(H'×W') 数万级，40000 柱 × 3 层根本算不动；且几何上柱的答案只可能在它投影位置附近——参考点投影就是强先验。deformable attention 把计算量压到"每柱 × 每命中相机 × 每高度 × 少数采样点"，同时保留可学习的微调能力。

**Q2**：TSA 只对"同格子"做注意力，为什么不需要更多历史格子？
**A**：warp 已把"世界同一位置"对齐到同一格子索引——空间对应关系由几何完成了，注意力只需判断"同一格的内容跨帧是不是同一物体"。运动目标会偏离 warp 预测，这是该设计的已知局限（第 4 周目标级传播的动机之一）。

**Q3**：BEVFormer 的 40000 个 BEV query 和 DETR 的 100 个 object query，训练上的最大不同？
**A**：DETR 用匈牙利匹配让 query 与目标 1-1 对应（稀疏集合预测）；BEVFormer 是稠密网格表征——每个 query 固定绑死一个空间格子，不存在匹配问题，检测头再从 BEV 特征出框。所以 BEVFormer 的 query 是"空间问句"，DETR 的 query 是"目标候选"——第 4 周 DETR3D 会把后者搬进 3D。

**Q4**：为什么 BEVFormer-base 的 test 集 mAP（37.5）比 val（41.6）还低，但 NDS 反而更高（56.9 vs 51.7）？
**A**：val 和 test 是不同标注子集，难度分布不同；NDS 是 mAP、mATE、mASE、mAOE、mAVE、mAAE 的复合指标，时序融合显著降低速度误差 mAVE，NDS 对"定位准、速度准"的加权让 BEVFormer 的时序优势在 NDS 上放大。**教训**：比较方法必须同集合同版本，单看 mAP 或单看 NDS 都会被误导。

---

## 本周交付清单

- [ ] 跑通 `w3_bevformer.py`，确认输出与文中一致（8 命中 / 4 mask、权重和 = 1、warp 量纲自检）。
- [ ] 闭卷写出 SCA 两步公式（投影 + masked deformable attention）并逐符号解释。
- [ ] 手算：ego 位移 (3.4, -0.2) m 在 0.512 m 分辨率下对应多少格子索引位移。
- [ ] 能回答：BEV query 与 DETR object query 的异同？TSA 为什么必须先 warp？
