# 04 - PCA 与降维

## 本篇目标

读完本篇，你应该能做到：

1. 说清协方差矩阵每个元素的几何含义，以及它与相关系数的换算关系。
2. 独立完成 PCA 四步流程：中心化 → 协方差 → 特征分解 → 投影，并能解释每一步在做什么。
3. 说明 PCA 与 SVD 的关系，会读方差解释率（explained variance ratio）。
4. 完成 3D 数据降到 2D 的完整实例，并知道"该不该先标准化"这个经典陷阱的判断标准。

PCA 是 03 篇特征分解的第一个大应用，也是量化投资（协方差矩阵）、embedding 可视化（t-SNE/UMAP 的前置直觉）的共同基础。

## 1. 协方差矩阵：数据形状的完整描述

### 1.1 从方差到协方差

方差衡量单个特征自身的散布：$\mathrm{Var}(X) = \mathbb{E}[(X-\mu)^2]$（严格的期望定义在 06 篇给出）。两个特征"是否一起动"由协方差刻画：

$$
\mathrm{Cov}(X, Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)]
$$

把 $d$ 个特征两两之间的协方差排成矩阵：

$$
\Sigma = \begin{bmatrix}
\mathrm{Var}(X_1) & \mathrm{Cov}(X_1,X_2) & \cdots \\
\mathrm{Cov}(X_2,X_1) & \mathrm{Var}(X_2) & \cdots \\
\vdots & & \ddots
\end{bmatrix}
$$

三个要点（Strang《Linear Algebra》§6.6；完整的概率论定义见 Casella & Berger §4.3）：

1. **对称**（$\mathrm{Cov}(X,Y)=\mathrm{Cov}(Y,X)$），且**半正定**（任何方向上的投影方差非负——03 篇的谱定理因此适用）；
2. 对角线是各特征的方差，**非对角线是"联动强度"**：正值 = 同涨同跌，负值 = 此消彼长，0 = 线性不相关；
3. 它完整描述了椭圆状点云的**朝向和胖瘦**——PCA 就是在给这个椭圆找主轴。

### 1.2 与相关系数的关系

协方差的数值受量纲影响（米→厘米直接放大 100 倍），归一化后就是相关系数：

$$
\rho_{XY} = \frac{\mathrm{Cov}(X,Y)}{\sigma_X\,\sigma_Y}, \qquad \rho \in [-1, 1]
$$

例：$\sigma_X = 2,\ \sigma_Y = 1,\ \mathrm{Cov}(X,Y) = 1.2$，则 $\rho = \frac{1.2}{2\times1} = 0.6$。量化投资里"两只股票相关性 0.9"说的就是这个 $\rho$，而组合优化真正计算时用的是协方差矩阵——两者由上式互通。

```python
import numpy as np
rng = np.random.default_rng(0)
X = rng.multivariate_normal([0, 0], [[4.0, 1.2], [1.2, 1.0]], size=200000)
Sigma = np.cov(X.T)
print(Sigma)          # ≈ [[4.0, 1.2], [1.2, 1.0]] 大样本下逼近设定值
rho = Sigma[0,1] / np.sqrt(Sigma[0,0] * Sigma[1,1])
print(rho)            # ≈ 0.6
```

## 2. PCA 四步流程

### 2.1 步骤

给定数据矩阵 $X \in \mathbb{R}^{n\times d}$（$n$ 个样本、$d$ 个特征）：

1. **中心化**：减去每列均值，$\tilde{X} = X - \bar{\mathbf{x}}$。原因：PCA 找的是"围绕重心的主轴"，不中心化的话第一主成分会被均值方向劫持。
2. **算协方差**：$S = \frac{1}{n-1}\tilde{X}^\top\tilde{X}$（$d\times d$ 对称矩阵）。
3. **特征分解**：求 $S = Q\Lambda Q^\top$，特征值从大到小排：$\lambda_1 \ge \lambda_2 \ge \cdots$。特征向量就是**主成分**（主轴方向），特征值就是**沿该主轴的方差**。
4. **投影**：取前 $k$ 个特征向量拼成 $W \in \mathbb{R}^{d\times k}$，降维结果 $Z = \tilde{X} W$。

### 2.2 手算一个最小例子

四个点：$(0,0), (1,1), (2,2), (3,3)$。均值 $(1.5, 1.5)$，中心化后偏差为 $(-1.5,-1.5), (-0.5,-0.5), (0.5,0.5), (1.5,1.5)$：

$$
S = \begin{bmatrix} 1.25 & 1.25 \\ 1.25 & 1.25 \end{bmatrix},\quad
\lambda_1 = 2.5,\ \mathbf{q}_1 = \tfrac{1}{\sqrt2}[1,1]^\top;\quad \lambda_2 = 0,\ \mathbf{q}_2 = \tfrac{1}{\sqrt2}[1,-1]^\top
$$

第一主成分沿对角线方向——数据本来就躺在这条线上；$\lambda_2 = 0$ 说明垂直方向完全没有散布。**这正是"降维无损"的条件**：$\lambda_i = 0$ 的方向丢掉不损失任何信息。

### 2.3 与 SVD 的关系

对中心化矩阵 $\tilde{X}$ 做 SVD：$\tilde{X} = U\Sigma V^\top$，代入 $S$ 的定义：

$$
S = \frac{1}{n-1}V\Sigma^2 V^\top
$$

所以：**$V$ 的列 = 协方差矩阵的特征向量 = 主成分；$\sigma_i^2/(n-1)$ = 特征值 = 主轴方差**。工程实现（`sklearn` 内部）走的就是 SVD 路线——数值上比直接特征分解更稳（**经验做法**：$n \gg d$ 时 `np.linalg.svd(X_centered, full_matrices=False)` 比 `np.linalg.eigh(cov)` 更稳定省内存）。

```python
Xc = np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.]]) - np.array([1.5,1.5])
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
print(s**2 / 3)      # [2.5 0.]  特征值,与手算一致
print(Vt)            # 第一行 ≈ [0.707, 0.707] 即 [1,1]/sqrt(2)
Z = Xc @ Vt[0]       # 投影到第一主成分
print(Z)             # [-2.121 -0.707 0.707 2.121] 一维坐标,信息零损失
```

## 3. 方差解释率怎么读

### 3.1 定义

第 $i$ 个主成分的**方差解释率**：

$$
\mathrm{EVR}_i = \frac{\lambda_i}{\sum_j \lambda_j}
$$

前 $k$ 个的累计和告诉你"保留 $k$ 维留住了百分之多少的总方差"。

### 3.2 怎么用

- **碎石图**（scree plot）：把 $\lambda_i$ 从大到小画出来，找"肘部"——之后都是平缓的碎石；
- 常见经验阈值：累计 90%–95%（**经验做法**，非定理：可视化用 2–3 维看 EVR；预处理用累计 95%；如果第一个主成分就吃掉 99%，先怀疑是不是量纲没统一，见第 5 节）；
- 注意读法：解释率衡量的是**方差**留存，不等于**信息/预测力**留存——与标签相关的小方差方向可能恰恰重要，监督任务里 PCA 盲砍小方差方向偶尔会砍掉关键特征。

```python
lam = np.array([5.0, 3.0, 1.0, 0.5, 0.3, 0.2])
evr = lam / lam.sum()
print(evr)                    # [0.5 0.3 0.1 0.05 0.03 0.02]
print(evr.cumsum())           # 累计:[0.5 0.8 0.9 0.95 0.98 1.0]
print("保留 3 维留住 90%,4 维 95%")
```

## 4. 完整实例：3D 数据降到 2D

造一个"本质上是 2 维、但被嵌在 3 维里"的数据集——一个倾斜的椭圆盘：

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n = 500
# 真实结构是 2D 高斯(方差 4 和 1,不相关),再经过一个 3D 旋转矩阵
z2 = rng.multivariate_normal([0, 0], [[4.0, 0.0], [0.0, 1.0]], size=n)
th, ph = 0.6, 0.9                       # 两个旋转角
R1 = np.array([[np.cos(th), -np.sin(th), 0],
               [np.sin(th),  np.cos(th), 0],
               [0, 0, 1]])
R2 = np.array([[1, 0, 0],
               [0, np.cos(ph), -np.sin(ph)],
               [0, np.sin(ph),  np.cos(ph)]])
X3 = z2 @ (R1 @ R2).T                   # (500, 3) 倾斜的椭圆盘

# ---- PCA 四步 ----
Xc = X3 - X3.mean(axis=0)               # 1. 中心化
S = Xc.T @ Xc / (n - 1)                 # 2. 协方差矩阵
lam, Q = np.linalg.eigh(S)              # 3. 特征分解(eigh 要求升序,反转)
order = np.argsort(lam)[::-1]
lam, Q = lam[order], Q[:, order]
print("特征值:", lam)                    # ≈ [4.0, 1.0, 0.0]——真相还原!
print("方差解释率:", lam / lam.sum())    # ≈ [0.80, 0.20, 0.00]
X2 = Xc @ Q[:, :2]                      # 4. 投影到前两个主成分

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(*X3.T, s=4, alpha=0.4); ax1.set_title('3D view (tilted disk)')
ax2 = fig.add_subplot(122)
ax2.scatter(*X2.T, s=4, alpha=0.4); ax2.set_title('after PCA (2D)')
ax2.set_aspect('equal')
plt.savefig('pca3d.png', dpi=120)
# 图注:3D 视角里椭圆盘斜着放;PCA 旋转回正后,数据又变回水平的椭圆,
# 且方差恰好回到设定的 4 和 1——PCA 本质是"把坐标轴转到数据的主轴上"。
```

关键观察：特征值 $\approx [4.0, 1.0, 0.0]$——第三维特征值接近 0，因为数据确实躺在 2 维平面上；旋转矩阵 $R_1R_2$ 的效果被 PCA **精确撤销**了。这就是"降维不是压缩坐标，是旋转坐标后丢掉空的方向"。

### 4.1 在 embedding 可视化中的应用

把 768 维的 BERT embedding 降到 2 维画散点，工程管线通常是：**先 PCA 到 20–50 维 → 再用 UMAP/t-SNE 降到 2 维**（**经验做法**：sklearn 官方 t-SNE 文档与 UMAP 作者均如此建议；原因是高维距离在 $10^2$ 维以上趋于"挤成一团"（维度灾难），先用 PCA 去掉纯噪声方向，后续非线性降维的簇结构明显更干净、速度更快）。本篇的 PCA 部分正是这条管线的第一站。

## 5. PCA 陷阱：该先标准化吗？

### 5.1 问题的根源

PCA 的特征值是**方差**，而方差带量纲。如果特征 1 是"收入"（单位：元，方差 $10^8$ 量级），特征 2 是"年龄"（方差 $10^2$ 量级），协方差矩阵会被收入完全统治：第一主成分≈收入轴，"年龄"的信息在方差竞赛里连参赛资格都没有。

### 5.2 演示

```python
rng = np.random.default_rng(1)
# 两特征真实相关,但尺度差 100 倍
base = rng.standard_normal(300)
X_raw = np.stack([100 * base + 1e-3 * rng.standard_normal(300),  # 收入:方差~1e4
                  base], axis=1)                                  # 年龄:方差~1
Xc = X_raw - X_raw.mean(axis=0)
lam, Q = np.linalg.eigh(Xc.T @ Xc / 299)
print("未标准化的第一主成分:", Q[:, -1])   # ≈ [1, 0.01]——几乎纯收入方向
# 标准化(每列除以标准差)后:
Xs = Xc / Xc.std(axis=0)
lam2, Q2 = np.linalg.eigh(Xs.T @ Xs / 299)
print("标准化后第一主成分:", Q2[:, -1])    # ≈ [0.707, 0.707]——两个特征均衡参与
```

### 5.3 判断标准（经验做法）

- 特征**量纲不同**（元/千克/岁）或尺度差 >1 个数量级：**先标准化**（等价于对相关系数矩阵做 PCA）；
- 特征同量纲且尺度本身有意义（图像像素、同源传感器）：**只中心化**，不要标准化——尺度是信号的一部分；
- 拿不准时问一句：把单位从"米"换成"厘米"，我希望的降维结果变吗？**变 → 该标准化；不变 → 不该**。

## 本篇小结

- 协方差矩阵是对角线放方差、非对角线放联动的对称半正定矩阵，是点云椭圆的完整说明书；除以两个标准差就得到相关系数。
- PCA 四步：中心化 → $\tilde{X}^\top\tilde{X}/(n-1)$ → 特征分解 → 投影。特征向量是主轴，特征值是主轴方差；与 SVD 的关系是 $\lambda_i = \sigma_i^2/(n-1)$。
- 方差解释率 $\lambda_i/\sum\lambda_j$ 读作"留住了多少散布"；累计 95% 是常用经验阈值，但小方差方向未必不重要。
- 3D→2D 实例证明 PCA 能精确还原"被旋转的 2 维盘"；embedding 可视化管线 = PCA 粗降 + UMAP/t-SNE 细降。
- 量纲不一致时先标准化，否则大方差特征会垄断主成分。

## 练习

1. **（手算）** 数据点 $(-1,0),(1,0),(0,-1),(0,1)$（已中心化）。求协方差矩阵、全部特征值与主成分方向，并说明方差解释率。（答案：$S=\frac{2}{3}I$，特征值均为 $\frac{2}{3}$，两个方向的解释率都是 50%——特征值相等时主成分方向不唯一，任何正交方向都一样，这是"各向同性数据"的特征。）
2. **（推导）** 证明协方差矩阵半正定：对任意向量 $\mathbf{a}$，$\mathbf{a}^\top S\mathbf{a} = \frac{1}{n-1}\sum_i (\mathbf{a}^\top(\mathbf{x}_i-\bar{\mathbf{x}}))^2 \ge 0$。并解释这个量的统计含义（沿 $\mathbf{a}$ 方向投影的方差）。
3. **（编程）** 用 sklearn 的 `PCA(n_components=2)` 重跑 4 节的 3D 数据，核对 `explained_variance_` 与手写实现的 `lam` 一致；再用 `pca.inverse_transform` 重构 3D 数据，计算重构误差 $\|X_3-\hat{X}\|_F/\|X_3\|_F$，确认约等于 $\sqrt{\lambda_3/(\lambda_1+\lambda_2+\lambda_3)}$ 的量级。
4. **（编程）** 生成 300 个样本、5 个特征的数据，其中只有前 2 个特征有真实相关结构、后 3 个是独立噪声。画出碎石图与累计方差解释率，验证"肘部"出现在真实维数附近。
5. **（思考+编程）** 对 5.2 节的尺度失衡数据，把"收入"的单位换成"万元"（除以 10000）再跑 PCA，比较三种情况（原始、万元、标准化）的第一主成分方向，写两句话解释为什么答案依赖单位——这正是"该不该标准化"问题的本质。
