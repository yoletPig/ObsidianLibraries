# 第 2 周教程：相机-LiDAR 外参标定

> **本周要回答的三个问题**
> 1. "相机-LiDAR 外参"这个 $4\times4$ 矩阵到底怎么定义？方向写反了会发生什么？
> 2. PnP 为什么能用几个 3D 点 + 它们的像素位置解出 6 自由度位姿？怎么用 SVD 求解？
> 3. LiDAR-Camera 联合标定的"标定板路线"和"无目标路线"各靠什么约束？标完怎么验证、验收标准是什么？

对应学习计划：第 2 周。交付物：① 多平面法向对齐外参标定（合成数据）；② 点云投影叠加 demo 与外参敏感度体检。

---

## 1. 外参 T：一个 4×4 矩阵，两个致命细节

### 1.1 齐次矩阵与复合

刚体变换 = 旋转 R（3×3，正交、行列式 +1）+ 平移 t（3×1），打包成齐次矩阵：

$$
T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}, \qquad
\tilde{P}_B = T_{B \leftarrow A}\, \tilde{P}_A, \qquad
\tilde{P} = \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
$$

**这个公式在干什么**：把"旋转 + 平移"两步合成一次矩阵乘法。$T_{B\leftarrow A}$ 读作"A 系坐标变到 B 系坐标"。

复合规则：从 A 到 C 经过 B 中转：

$$
T_{C \leftarrow A} = T_{C \leftarrow B}\; T_{B \leftarrow A}
$$

（右边的先作用。矩阵乘法从右往左读。）逆变换：

$$
T_{A \leftarrow B} = T_{B \leftarrow A}^{-1} = \begin{bmatrix} R^\top & -R^\top t \\ 0 & 1 \end{bmatrix}
$$

**注意逆矩阵不是把 t 取负**——旋转也要转置。这是手算最常见的错误。

### 1.2 两个致命细节

**细节一：方向约定。** "相机-LiDAR 外参"在文献里既可能指 $T_{cam \leftarrow lidar}$ 也可能指 $T_{lidar \leftarrow cam}$。本教程统一用**左箭头记法**：$T_{dst \leftarrow src}$ 把 src 系的点变到 dst 系。工程防呆三招：

1. 配置文件字段写全名（`T_camera_lidar` 必须注释清楚是 "points in lidar frame -> camera frame" 还是反过来）；
2. 上线前用**可观测的物理事实**自检：把地面点投到图像，地面必须出现在画面下半部分。方向反了，地面会飞到天上——一眼识破；
3. 单元测试里放一个"往返不变量"：$T^{-1} T = I$，变换两次应回到原点。

**细节二：外参是"传感器对"，不是"传感器"。** 同一个相机和不同 LiDAR 组合就有不同外参。换装任何一个传感器，整个标定作废重做。

---

## 2. PnP：从 3D-2D 对应解位姿

### 2.1 问题定义与几何直觉

已知：$n$ 个 3D 点 $X_i$（在某个物体系中）和它们在图像里的像素 $\tilde{u}_i$，以及内参 K。求：物体系到相机系的位姿 $[R|t]$（6 自由度）。

几何直觉：每个像素 $\tilde{u}_i$ 反投影出一条从光心出发的**射线**；未知位姿必须让所有 3D 点恰好落到各自射线上。3 个点对应 3 条射线——这就是名字里的 "Perspective-3-Point"（P3P）；实际用 $n \ge 6$ 个点做最小二乘更稳。二乘的目标是**重投影误差**：

$$
\min_{R, t} \sum_i \big\| \tilde{u}_i - \pi\big(K (R X_i + t)\big) \big\|^2, \qquad
\pi\left(\begin{bmatrix} a \\ b \\ c \end{bmatrix}\right) = \begin{bmatrix} a/c \\ b/c \end{bmatrix}
$$

### 2.2 DLT：把非线性投影变成 $Ax=0$

直接优化上式是非线性的（除法 + 三角函数）。DLT（Direct Linear Transform）用齐次坐标把它变成线性问题。记 $P = [R | t]$（3×4），投影 $s\tilde{u} = P \tilde{X}$。展开每个点给两行方程：

$$
\begin{bmatrix}
\tilde{X}^\top & 0 & -u\,\tilde{X}^\top \\
0 & \tilde{X}^\top & -v\,\tilde{X}^\top
\end{bmatrix}
\mathrm{vec}(P) = 0
$$

**这个公式在干什么**：投影 $u = \frac{p_1 \cdot \tilde{X}}{p_3 \cdot \tilde{X}}$ 交叉相乘消掉分母 $s$，得到 $p_1 \cdot \tilde{X} - u\, p_3 \cdot \tilde{X} = 0$——对 P 的 12 个元素**线性**。$n$ 个点堆成 $2n \times 12$ 矩阵，SVD 取最小奇异值对应的右奇异向量，reshape 回 3×4。

两个实操要点：

1. **Hartley 归一化**：直接用原始像素坐标（量级 ~640）与米制坐标混在一起，矩阵列间量级悬殊、病态。先把 2D/3D 点各自平移到质心、缩放到平均距离 $\sqrt{2}$，解完再反变换回去。不做归一化误差能大一个数量级；
2. **从 P 拆回 R, t**：DLT 解出的是 11 自由度的完整 $P = K'[R|t]$（K' 未必等于你的 K）。用 RQ 分解拆出 $K'$ 和 $R$，$t = K'^{-1} p_4$。

### 2.3 数值实验：DLT vs PnP，同噪声下差几十倍

立体靶标：边长 0.4 m 立方体的 8 个角点 + 6 个面心（**真正非共面**——这一点很重要，共面点会让 11 自由度 DLT 病态，第 6 节坑里细说）。

```python
# exp2_pnp.py（核心部分；完整脚本见 实验脚本/exp2_pnp.py）
import numpy as np, cv2
rng = np.random.default_rng(1)
K = np.array([[800.0, 0, 640.0], [0, 800.0, 360.0], [0, 0, 1.0]])

S = 0.4
corners = np.array([[x, y, z] for x in (0, S) for y in (0, S) for z in (0, S)], float)
faces = np.array([[S/2, S/2, 0], [S/2, S/2, S], [S/2, 0, S/2],
                  [S/2, S, S/2], [0, S/2, S/2], [S, S/2, S/2]])
X_obj = np.vstack([corners, faces])                      # 14 个非共面 3D 点

R_true, _ = cv2.Rodrigues(np.array([0.15, -0.3, 0.1]))
t_true = np.array([0.5, -0.2, 2.5])

def dlt_pnp(X_obj, uv):
    """手写 DLT：Hartley 归一化 -> 齐次线性方程组 -> SVD 最小奇异向量 -> 反归一化。"""
    cu, su = uv.mean(0), np.sqrt(2) / uv.std(0).mean()
    T2 = np.array([[su, 0, -su*cu[0]], [0, su, -su*cu[1]], [0, 0, 1]])
    cx, sx = X_obj.mean(0), np.sqrt(2) / X_obj.std(0).mean()
    T3 = np.eye(4); T3[:3, :3] = sx * np.eye(3); T3[:3, 3] = -sx * cx
    uv_n = (T2 @ np.column_stack([uv, np.ones(len(uv))]).T).T[:, :2]
    X_n = (T3 @ np.column_stack([X_obj, np.ones(len(X_obj))]).T).T[:, :3]
    A = np.zeros((2 * len(X_n), 12))
    for i, (Xw, (u, v)) in enumerate(zip(X_n, uv_n)):
        Xh = np.append(Xw, 1.0)
        A[2*i, 0:4] = Xh;     A[2*i, 8:12] = -u * Xh
        A[2*i+1, 4:8] = Xh;   A[2*i+1, 8:12] = -v * Xh
    _, _, Vt = np.linalg.svd(A)
    return np.linalg.inv(T2) @ Vt[-1].reshape(3, 4) @ T3

def decompose(P):
    """P = K'[R|t]：RQ 分解拆出 K、R，t = K^-1 p4。"""
    from scipy.linalg import rq
    P = P / np.linalg.norm(P[2, :3])
    if P[2, 3] < 0: P = -P
    K_r, R = rq(P[:, :3])
    D = np.diag(np.sign(np.diag(K_r)))
    K = K_r @ D; R = D @ R; K /= K[2, 2]
    return R, np.linalg.inv(K) @ P[:, 3]

def solve(noise_px, n_pts):
    idx = rng.choice(len(X_obj), n_pts, replace=False)
    Xs = X_obj[idx]
    Xc = (R_true @ Xs.T).T + t_true
    uv = (K @ Xc.T).T
    uv = uv[:, :2] / uv[:, 2:3] + rng.normal(0, noise_px, (n_pts, 2))
    R_dlt, t_dlt = decompose(dlt_pnp(Xs, uv))            # 方法 A：11 自由度线性解
    ok, rvec, t_pnp = cv2.solvePnP(Xs.astype(np.float32), uv.astype(np.float32),
                                   K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    R_pnp = cv2.Rodrigues(rvec)[0]                       # 方法 B：K 已知，6 自由度 + LM
    rerr = lambda R: np.degrees(np.arccos(np.clip((np.trace(R.T @ R_true) - 1) / 2, -1, 1)))
    return rerr(R_dlt), np.linalg.norm(t_dlt - t_true), rerr(R_pnp), np.linalg.norm(t_pnp.ravel() - t_true)
```

真实输出（完整脚本含噪声/点数扫描，`exp2_pnp.py`）：

```
14 个 3D 点、噪声 0.5 px：
  手写 DLT + SVD      : 旋转误差  0.619 deg, 平移误差  18.79 cm
  cv2.solvePnP (LM)   : 旋转误差  0.170 deg, 平移误差   0.10 cm

噪声扫描（14 点，5 个种子平均）：
    噪声(px) | DLT 旋转误差      DLT 平移误差 | PnP 旋转误差     PnP 平移误差
       0.0 |      0.000°       0.00cm  |      0.000°      0.00cm
       0.5 |      1.862°      10.21cm  |      0.168°      0.60cm
       1.0 |      4.976°      31.02cm  |      0.490°      1.55cm
       2.0 |      8.040°      48.68cm  |      0.670°      1.11cm

点数扫描（噪声 0.5 px，5 个种子平均；靶标共 14 点，6 点为理论最小值）：
   6 点 | DLT 旋转 110.734°  平移 219.02 cm | PnP 旋转  97.688°  平移 282.84 cm
  10 点 | DLT 旋转   4.604°  平移  26.74 cm | PnP 旋转   0.235°  平移   0.41 cm
  14 点 | DLT 旋转   2.218°  平移  14.42 cm | PnP 旋转   0.252°  平移   0.51 cm
```

**读数**：无噪声两者都精确（算法正确）；有噪声后 DLT 明显更差，因为 (1) 它解 11 自由度，K 的误差被吸收进 R/t；(2) 它最小化代数误差而非几何误差。cv2.solvePnP 知道 K、只解 6 自由度、并用 LM 最小化像素级重投影误差（等价于高斯噪声下的最大似然）——所以**实践中永远用 solvePnP，DLT 只用来理解原理和给初值**。另注意 6 点（理论最小值）时两种方法都会在部分随机样本上崩溃——"最少点数"只是数学可行，不是工程可用。

---

## 3. LiDAR-Camera 联合标定：两条路线

目标：求 $T_{cam \leftarrow lidar}$（把点云变到相机系的刚体变换）。两条路线的本质区别是**约束从哪来**。

### 3.1 路线一：基于标定板（target-based）

**核心思想**：摆一块标定板，让两个传感器同时观测它，同一块板在两个坐标系里各给出一个"平面描述"，两个平面是同一块物理板 → 提供约束。

**相机侧**（内参已知）：检测棋盘角点 → PnP → 板在相机系的位姿 → 平面方程 $(n_c, d_c)$：$n_c \cdot x + d_c = 0$。

**LiDAR 侧**：分割出板上的点云（强度/深度阈值 + 聚类）→ SVD 平面拟合 → $(n_l, d_l)$。

> **SVD 平面拟合在干什么**：把点减去质心得到去均值矩阵，它的最小奇异值对应的右奇异向量就是"点云最扁的方向" = 法向。d 用 $d = -n \cdot \text{centroid}$ 补上。

**求 R（Wahba 问题）**：$n_c = R\, n_l$ 对每块板成立。多块**不平行**的板堆叠成：

$$
\max_{R \in SO(3)} \sum_i n_{c,i}^\top R\, n_{l,i}
$$

**这个公式在干什么**：找旋转，让所有"LiDAR 法向"转到与"相机法向"最一致。这是正交 Procrustes / Wahba 问题，闭式解：

$$
M = \sum_i n_{c,i}\, n_{l,i}^\top = U \Sigma V^\top,\qquad
R = U\,\mathrm{diag}\big(1, 1, \det(U V^\top)\big)\,V^\top
$$

det 修正项保证解是"纯旋转"而不是"旋转+镜像"（SVD 只保证正交，不保证行列式 +1）。

**求 t（平面偏移方程）**：对每块板，同一物理平面在两系中的方程必须描述同一个点集。把 $P_{cam} = R\,P_{lidar} + t$ 代入相机系平面方程，与 LiDAR 系方程对比系数，得一条线性方程：

$$
n_{c,i} \cdot t = d_{l,i} - d_{c,i}
$$

≥3 块法向不共线的板 → 3×3 方程组直接解出 t。

**最少几块板？** 一块板只能约束 2 个旋转自由度（法向对齐）+ 1 个平移（沿法向的偏移），板内滑动方向完全自由。所以工程上至少 **3 块法向彼此张开 > 30° 的板位**，实践中摆 5~10 个姿态。

完整数值实验（5 块不同姿态虚拟板、LiDAR 5 mm 测距噪声）：

```python
# exp3_lidar_cam_planes.py（核心求解部分；完整场景生成见 实验脚本/exp3_lidar_cam_planes.py）
normals_cam, normals_lidar = [...]   # 每块板两系的单位法向（SVD 拟合得到）
d_cam, d_lidar = [...]               # 每块板两系的平面偏距

# 第一步：求 R（Wahba，SVD 正交 Procrustes）
M = normals_cam.T @ normals_lidar                     # 3x3
U, _, Vt = np.linalg.svd(M)
S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
R_est = U @ S @ Vt

# 第二步：求 t（每块板一条线性方程 n_c·t = d_l - d_c）
A = normals_cam.copy()
b = d_lidar - d_cam
t_est, *_ = np.linalg.lstsq(A, b, rcond=None)
```

真实输出（完整脚本含噪声扫描）：

```
5 块不同姿态的标定板，LiDAR 测距噪声 5 mm：
  旋转误差: 0.1083 deg
  平移误差: 5.34 mm
  R 真值第一行: [ 0.98103 -0.17107  0.09123]
  R 估计第一行: [ 0.98128 -0.16993  0.09062]
  t 真值: [ 0.05 -0.02  0.1 ] m,  t 估计: [ 0.0517 -0.025   0.0993] m
  各板法向夹角（应接近 0）: [7.929 9.24  2.68  6.662 7.752]

LiDAR 测距噪声扫描（3 个重复，平均）：
    0 mm  ->  旋转误差  0.000 deg, 平移误差    0.00 mm
    5 mm  ->  旋转误差  0.151 deg, 平移误差    3.17 mm
   20 mm  ->  旋转误差  0.236 deg, 平移误差   14.50 mm
   50 mm  ->  旋转误差  1.370 deg, 平移误差   18.71 mm
```

**读数**：5 mm 噪声（典型机械式 LiDAR 的单点测距精度）下旋转 0.15°、平移 3~5 mm——这就是靶标法能企及的精度。噪声 50 mm（远距离边缘点混入）时退化到 1.4°，所以**板面点云的分割质量决定标定上限**，分割要用强度+深度双重阈值并剔除边缘混合点。

> 保守估计各板块数/姿态的建议值来自实验设计而非某篇论文的硬性结论；不同传感器组合的可达精度，以你自己复现的合成实验和真实数据验收为准。

### 3.2 路线二：无目标（Targetless）

标定板要摆、要收、要在标定间做——量产车队不可能每次都进标定间。无目标方法用**环境本身**当标定物，核心是让"点云的某种属性"与"图像的某种属性"对齐：

- **强度投影（intensity projection）**：LiDAR 回波强度与图像灰度都反映表面材质反射率。把点云按强度染色投到图像上，对齐时边缘应该重合。Pandey et al. 用互信息（Mutual Information, MI）作为对齐目标函数（AAAI 2012 会议版 / JFR 2015 期刊版），是无目标标定的经典起点；
- **边缘对齐**：图像边缘（深度不连续处）与点云深度不连续边缘对齐。`hku-mars/livox_camera_calib` 用图像边缘像素转成射线、最小化射线到点云边缘点的距离，Ceres 优化 4 自由度（其余自由度由地面约束固定）；
- **直接法**：`koide3/direct_visual_lidar_calibration`（ICRA 2023）把强度投影图与图像做直接光度对齐，支持任意投影模型、单次采样自动完成。

三条共性：都需要**好的初值**（这些方法本质都是局部优化）；都需要**纹理/结构丰富的场景**（对着一面白墙永远标不出来）；精度通常逊于靶标法（边缘定位精度受点云分辨率限制），适合做**在线校验与漂移补偿**，首次标定仍建议进标定间。

### 3.3 开源方案速查（均经查证存在且活跃）

| 项目 | 类型 | 要点 |
| --- | --- | --- |
| `ankitdhall/lidar_camera_calibration` | 靶标 | ArUco+棋盘，3D-3D 点对应（配套论文 arXiv:1705.09785） |
| `UMich-BipedLab/extrinsic_lidar_camera_calibration` | 靶标 | 对 Dhall 方法的系统改进，含不确定度分析 |
| `hku-mars/livox_camera_calib` | 无目标 | 边缘对齐，Ceres，支持 Livox 固态雷达 |
| `koide3/direct_visual_lidar_calibration` | 无目标 | 直接法，通用投影模型，ICRA 2023 |
| `Livox-SDK/livox_camera_lidar_calibration` | 靶标（半自动） | 官方工具，PnP 角点 + 手动点云选点 |

---

## 4. 评估标准：标完怎么知道它是对的

### 4.1 重投影误差（定量）

相机内部：重投影 RMS < 0.5 px 算合格（第 1 周）。外参层面：用留出的**独立**标定板数据，把 LiDAR 板面点投到图像，与检测到的板角点/边缘比距离——好的靶标标定能到**个位数像素**。注意重投影误差小只代表"对训练数据自洽"，验收必须用独立数据。

### 4.2 目视检查（qualitative 但不可替代）

把点云投到图像上叠起来看：建筑边缘是否压着建筑、车道线是否压着车道线、树冠是否在树冠上。经典错误一眼可见：方向反了（地面在天上）、差半个车身（translation 错）、远端发散（rotation 有小角度误差被距离放大）。

目视的主观性可以量化。下面的 demo 用"板上点落入板内比例"当体检指标：

```python
# exp4_projection.py（核心部分；场景：3 m 外一块 0.8x0.6 m 板，完整脚本见 实验脚本/exp4_projection.py）
def project(P_lidar, R, t):
    """LiDAR 点 -> 相机系 -> 像素。P_cam = R @ P_lidar + t"""
    Xc = (R @ P_lidar.T).T + t
    z = Xc[:, 2]
    uv = (K @ Xc.T).T
    return uv[:, :2] / z[:, None], z

# 板四角（板坐标系，位姿已知）投影成像素多边形
uv_pix, _ = cv2.projectPoints(corners_b, cv2.Rodrigues(R_b)[0], t_b, K, None)
poly = uv_pix.reshape(-1, 2).astype(np.float32)

def inside_ratio(P_lidar, R, t):
    """板面点云投影后落入板多边形内的比例——外参质量体检指标。"""
    uv, z = project(P_lidar, R, t)
    ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    hit = sum(cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
              for (x, y) in uv[ok])
    return hit, len(P_lidar)          # 与教程输出一致：分母是全部板点数
```

真实输出：

```
把 LiDAR 点云投影到虚拟图像，用'板点是否落入板内'做外参质量体检：
  真值外参                         板上点落入板内:  589/600 ( 98.2%)   背景墙点误入板内:  11/1000 ( 1.1%)
  真值 + yaw 偏 0.5 deg           板上点落入板内:  582/600 ( 97.0%)   背景墙点误入板内:  12/1000 ( 1.2%)
  真值 + yaw 偏 1.0 deg           板上点落入板内:  556/600 ( 92.7%)   背景墙点误入板内:  12/1000 ( 1.2%)
  真值 + yaw 偏 2.0 deg           板上点落入板内:  522/600 ( 87.0%)   背景墙点误入板内:  10/1000 ( 1.0%)
  真值 + y 平移偏 10 cm             板上点落入板内:  488/600 ( 81.3%)   背景墙点误入板内:  14/1000 ( 1.4%)
```

**读数**：3 m 距离上 yaw 偏 2°（约 10 cm 横向错位）使落入率从 98% 掉到 87%——**投影检查对平移误差比对小角度旋转更敏感**（小角度在 3 m 处的位移太小）。反过来在 100 m 远处，0.5° 的旋转就是 0.87 m 的错位，目视立刻露馅。所以目视检查要**近中远三个距离段**都看。

> **动手验证**：运行完整脚本会保存 `projection_overlay.png`（绿=真值外参投影、红=yaw 偏 2°），打开亲眼看红色点云怎么"滑出"白框——对微小外参误差建立视觉直觉。

---

## 5. 常见坑

1. **方向约定写反**：$T_{cam\leftarrow lidar}$ 与 $T_{lidar\leftarrow cam}$ 混用。症状：融合结果整体镜像/翻转，点云"穿地"。防呆见 1.2 节。
2. **内参没标好就标外参**：PnP 用错 K，解出的板位姿带系统偏差，外参跟着错。顺序永远是先内参后外参。
3. **标定板分割不干净**：LiDAR 在板边缘的点是"半打在板上半打在背景"的混合回波，强度/深度都不可信。不剔除会把平面拟歪。对策：向内收缩 2~3 个点距再用。
4. **板位姿态太少太平行**：5 块板法向都朝前，旋转的自由度约束不出来（尤其 roll）。至少 3 块法向彼此 > 30°。
5. **共面点做 11 自由度 DLT**：所有 3D 点躺在一个平面上时 $2n\times12$ 方程组退化（实验 2 里我们把靶标做成立体才敢用 DLT）。共面场景请走单应/PnP。
6. **只看重投影误差不看独立验证**：第 1 周的教训在外参上同样成立。
7. **无目标方法当首次标定用**：无目标依赖好初值和丰富纹理，冷启动请用靶标法，无目标做日常校验与漂移监测。

---

## 6. 小结

- 外参 $T_{dst\leftarrow src}$ 是 4×4 齐次矩阵；逆不是简单取负 t；方向约定必须工程防呆。
- PnP：像素反投影射线 + 3D 点落线约束；DLT 用齐次坐标把投影线性化为 $Ax=0$（SVD 求解），但 11 自由度代数误差版本精度远逊于 K 已知的 6 自由度 LM 版本——工程用 `cv2.solvePnP`。
- LiDAR-Camera 标定两条路线：**靶标法**（板平面在两系各拟合一次 → Wahba SVD 解 R → 平面偏移线性方程解 t，5 mm 噪声下可达 0.15°/5 mm）与**无目标法**（强度/边缘/MI 对齐，适合在线校验）。
- 验收 = 定量（独立数据重投影）+ 定性（近中远三段目视投影检查）。

## 7. 自测题（带答案）

**Q1：为什么一块标定板标不出完整外参？至少要几块、怎么摆？**
A：一块板提供一个法向（2 个旋转自由度）+ 一个平面偏距（1 个平移自由度），板内两个滑动方向无约束。至少 3 块法向彼此 >30° 的板位才把 6 自由度全部约束住；工程上摆 5~10 个姿态做冗余最小二乘。

**Q2：Wahba 问题的 SVD 解里 $\mathrm{diag}(1,1,\det(UV^\top))$ 是干什么的？**
A：SVD 只保证 $R$ 正交（$R^\top R=I$），正交矩阵还包括"镜像"（行列式 -1）。物理旋转必须 $\det=+1$，当 $\det(UV^\top)=-1$ 时把第三个奇异方向翻号。没有这一步，法向近似共面时可能解出反射矩阵。

**Q3：无目标标定对着一面白墙为什么永远不收敛？**
A：白墙没有纹理和深度不连续，强度/边缘/MI 等所有对齐特征都是常数——目标函数对位姿参数的梯度为零。无目标方法的隐含前提是环境提供可分辨的结构（边缘、角点、反射率变化）。

**Q4：实验中 DLT 平移误差（0.5 px 噪声下 10 cm）比 PnP（0.6 cm）大一个数量级，两个原因？**
A：(1) DLT 解 11 自由度完整投影矩阵，K 的微小估计误差被吸收进 R/t（自由度越多，噪声分到的方向越多）；(2) DLT 最小化代数误差（齐次方程残差），与像素几何误差不对齐；PnP 用 LM 直接最小化重投影误差，在高斯像素噪声下等价于最大似然估计。

**Q5：LiDAR 测距噪声从 5 mm 涨到 50 mm，旋转误差从 0.15° 涨到 1.37°（实验数据）。给出两条工程对策。**
A：(1) 源头治理——分割时剔除边缘混合回波点、只用板面中心区域的点、用强度门限过滤低反射噪声点；(2) 冗余平摊——增加板位姿态数（lstsq 的超定方程把独立噪声平均掉，误差随 $\sqrt{N}$ 下降），并对每块板的法向拟合做残差卡方检验、剔除离群板位。
