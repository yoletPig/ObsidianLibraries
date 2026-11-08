# 03 - 特征值与 SVD

## 本篇目标

读完本篇，你应该能做到：

1. 用一句话说出特征向量的本质：**变换下方向不变的那些特殊方向**，并能手算 $2\times2$ 矩阵的特征对。
2. 知道对称矩阵的特征值必为实数（谱定理），并用代码验证。
3. 把 SVD 读成"**任意线性变换 = 旋转·缩放·旋转**"，说清奇异值与特征值的关系。
4. 亲手做一次图像的秩-k 近似压缩，算出压缩比；知道伪逆是"逆"在奇异矩阵上的合法替身。

本篇是 04 篇 PCA 的直接前置：PCA 的全部输出（主成分、方差、投影）都来自对协方差矩阵做本篇的操作。

## 1. 特征值与特征向量：变换的不变方向

### 1.1 定义与直觉

若非零向量 $\mathbf{v}$ 满足

$$
A\mathbf{v} = \lambda \mathbf{v}
$$

则 $\mathbf{v}$ 是 $A$ 的**特征向量**（eigenvector），$\lambda$ 是对应的**特征值**（eigenvalue）。几何含义：$A$ 把大多数向量又转又拉，但对特征向量**只拉不转**——它落在变换的"主轴"上，$\lambda$ 就是沿这个主轴的拉伸倍数。

典型例子：剪切矩阵 $\begin{bmatrix}1&1\\0&1\end{bmatrix}$ 把 $(0,1)$ 推到 $(1,1)$（方向变了），但水平轴上的向量方向纹丝不动——它的特征向量就是 $(1,0)$，$\lambda = 1$。

### 1.2 怎么求：特征方程

把定义改写为 $(A-\lambda I)\mathbf{v} = \mathbf{0}$。要有非零解，矩阵 $A-\lambda I$ 必须奇异（压扁），即行列式为零：

$$
\det(A - \lambda I) = 0
$$

这个关于 $\lambda$ 的多项式叫**特征多项式**（Strang《Linear Algebra》§6.1）。

### 1.3 2x2 手算示例

取 $A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$：

$$
\det\begin{bmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{bmatrix}
= (4-\lambda)(3-\lambda) - 2 = \lambda^2 - 7\lambda + 10 = (\lambda-2)(\lambda-5)
$$

特征值 $\lambda_1 = 5,\ \lambda_2 = 2$。求特征向量：

- $\lambda_1 = 5$：$\begin{bmatrix}-1&1\\2&-2\end{bmatrix}\mathbf{v}=\mathbf{0} \Rightarrow \mathbf{v}_1 = [1, 1]^\top$；
- $\lambda_2 = 2$：$\begin{bmatrix}2&1\\2&1\end{bmatrix}\mathbf{v}=\mathbf{0} \Rightarrow \mathbf{v}_2 = [1, -2]^\top$。

验证：$A[1,1]^\top = [5,5]^\top = 5\,[1,1]^\top$ ✓。

### 1.4 对角化：用特征基看变换

若 $A$ 有 $n$ 个线性无关特征向量，拼成 $V$，则 $A = V\Lambda V^{-1}$。读法：**进入特征坐标系（$V^{-1}$）→ 沿各主轴独立拉伸（$\Lambda$）→ 回到原坐标系（$V$）**。复杂的"又转又拉"在特征基里只是简单的坐标轴缩放——这就是特征分解的全部意义。

```python
import numpy as np

A = np.array([[4.0, 1.0], [2.0, 3.0]])
lam, V = np.linalg.eig(A)
print(lam)        # [5. 2.]
print(V)          # 列是特征向量，方向即 [1,1] 与 [1,-2]
# 验证 A v = lam v
for i in range(2):
    print(np.allclose(A @ V[:, i], lam[i] * V[:, i]))   # True True
```

## 2. 对称矩阵的特殊性：谱定理

### 2.1 定理内容（不证明，演示）

**谱定理**：实对称矩阵 $A = A^\top$ 的特征值全为实数，且总能找到一组**互相正交**的特征向量，使 $A = Q\Lambda Q^\top$（$Q$ 正交，$Q^{-1}=Q^\top$）。（出处：Strang《Linear Algebra》§6.4；完整证明见 Horn & Johnson《Matrix Analysis》。）

### 2.2 为什么工程上偏爱对称矩阵

- 不会出现复数特征值（省心）；
- 特征向量天然正交（直接当新坐标轴用）；
- 协方差矩阵、核矩阵、图拉普拉斯**都是对称的**——04 篇 PCA 就是给对称矩阵做特征分解。

```python
S = np.array([[2.0, 1.0], [1.0, 2.0]])   # 对称
lam, V = np.linalg.eigh(S)               # eigh: 专用于对称矩阵
print(lam)                                # [1. 3.]
print(V @ V.T)                            # 单位阵——特征向量正交
# 对比:普通矩阵可能出现复特征值
B = np.array([[0.0, -1.0], [1.0, 0.0]])  # 旋转90度:没有实的不变方向
print(np.linalg.eigvals(B))               # [0.+1.j 0.-1.j] 旋转没有实特征值
```

直觉：旋转 90° 把每个方向都转走了，所以没有实的不变方向——这就是"非对称矩阵才有复特征值"的几何根源。

## 3. SVD：任意矩阵的"旋转·缩放·旋转"

### 3.1 定义

任何 $m\times n$ 实矩阵都可分解为

$$
A = U \Sigma V^\top
$$

其中 $U$（$m\times m$）、$V$（$n\times n$）是正交矩阵，$\Sigma$ 是 $m\times n$ 对角阵，对角元 $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ 称**奇异值**（Strang《Linear Algebra》§6.7）。

### 3.2 几何图像

把 $A\mathbf{x}$ 分三步读：**$V^\top$ 旋转/对齐到一组标准正交方向 → $\Sigma$ 沿各方向拉伸 $\sigma_i$ 倍 → $U$ 再旋转到输出方位**。特征分解只对可对角化的方阵成立，SVD 对**任意矩形矩阵**都成立——所以它是线性代数的"万能分解"。两者关系：$A$ 的奇异值 = $A A^\top$ 特征值的平方根。

### 3.3 手算一个完整 SVD

取对称矩阵 $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$。它是对称正定的，$U = V$：

- $A^\top A = A^2$ 的特征值为 $9$ 和 $1$（对应特征向量 $[1,1]^\top/\sqrt2$ 与 $[1,-1]^\top/\sqrt2$）；
- 所以 $\sigma_1 = 3,\ \sigma_2 = 1$，$V = \frac{1}{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}$，$U = V$。

验证：$U\Sigma V^\top = \frac12\begin{bmatrix}1&1\\1&-1\end{bmatrix}\begin{bmatrix}3&0\\0&1\end{bmatrix}\begin{bmatrix}1&1\\1&-1\end{bmatrix} = \frac12\begin{bmatrix}4&2\\2&4\end{bmatrix} = \begin{bmatrix}2&1\\1&2\end{bmatrix}$ ✓。

```python
A = np.array([[2.0, 1.0], [1.0, 2.0]])
U, s, Vt = np.linalg.svd(A)
print(s)                    # [3. 1.]
print(U, Vt, sep='\n')      # 正负号顺序可能与手算不同，但张成相同
# 用外积重构：A = s1*u1*v1^T + s2*u2*v2^T
print(s[0] * np.outer(U[:, 0], Vt[0, :]) + s[1] * np.outer(U[:, 1], Vt[1, :]))
# [[2. 1.], [1. 2.]]  与原矩阵一致
```

注意最后一行：**SVD 把 $A$ 写成秩 1 矩阵的加权和**，权重是奇异值——这是下一节压缩的原理。

## 4. SVD 应用：图像压缩实测

### 4.1 原理

图像就是一个矩阵（灰度图）$M$（$m\times n$）。做 SVD 后**只保留前 $k$ 个奇异值**：

$$
M \approx M_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top
$$

由 Eckart–Young 定理，$M_k$ 是所有秩 $\le k$ 矩阵中在 Frobenius 范数意义下对 $M$ 的**最优逼近**（出处：Eckart & Young, 1936；现代教材见 Strang §6.7）。

### 4.2 压缩比怎么算

原矩阵存 $mn$ 个数；秩-k 近似只需 $k$ 个 $\sigma_i$ + $k$ 个 $\mathbf{u}_i$（$mk$ 个数）+ $k$ 个 $\mathbf{v}_i$（$kn$ 个数）：

$$
\text{压缩比} = \frac{mk + k + kn}{mn} = \frac{k(m+n+1)}{mn}
$$

例：$m = n = 200,\ k = 10$：$\frac{10 \times 401}{40000} = 0.1$，即约 10% 的存储量。

### 4.3 完整实验

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
# 造一张"有结构"的 200x200 图:低秩成分(平滑渐变) + 少量噪声
x, y = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
M = np.sin(3 * np.pi * x) * np.cos(2 * np.pi * y) + 0.05 * rng.standard_normal((200, 200))

U, s, Vt = np.linalg.svd(M, full_matrices=False)
energy = s**2 / np.sum(s**2)                 # 每个奇异值的"能量占比"
print("前10个奇异值的能量占比:", energy[:10].sum())   # 低秩信号为主时接近 1

for k in [2, 10, 50]:
    Mk = (U[:, :k] * s[:k]) @ Vt[:k, :]
    ratio = k * (M.shape[0] + M.shape[1] + 1) / (M.size)
    err = np.linalg.norm(M - Mk) / np.linalg.norm(M)
    print(f"k={k:3d}  压缩比={ratio:.3f}  相对误差={err:.3f}")

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
axes[0].imshow(M, cmap='gray');  axes[0].set_title('original (rank 200)')
for ax, k in zip(axes[1:], [2, 10, 50]):
    ax.imshow((U[:, :k] * s[:k]) @ Vt[:k, :], cmap='gray')
    ax.set_title(f'rank-{k}')
for ax in axes: ax.axis('off')
plt.savefig('svd_compress.png', dpi=120)
# 图注:k=2 只剩大致条纹;k=10 主结构清晰、噪声被抹掉;k=50 几乎无差别。
# 有趣的副作用:秩-10 图像比原图更"干净"——截断 SVD 顺手做了去噪(低秩+稀疏模型的直觉)。
```

典型输出（本例结构下）：`k=10 压缩比≈0.100 相对误差≈0.05` 量级；自然照片能量集中没这么极端，**经验做法**：一般照片前 10% 的奇异值通常携带 90% 以上的能量，视内容而定。

## 5. 矩阵伪逆

### 5.1 定义

当 $A$ 奇异或非方阵时，逆不存在，但**伪逆**（Moore–Penrose）总存在：

$$
A^{+} = V \Sigma^{+} U^\top, \qquad \Sigma^{+}\ \text{把非零}\ \sigma_i\ \text{取倒数、零保持零}
$$

它给出的是"最像逆的东西"：最小二乘问题的解正是 $\hat{\mathbf{x}} = A^{+}\mathbf{b}$（在所有最小化 $\|A\mathbf{x}-\mathbf{b}\|$ 的解中取范数最小的那个）。

### 5.2 用场

- 欠定/超定方程组的统一解法；
- $A$ 秩亏时给出**最小范数解**——正则化的味道；
- NumPy 一行：`np.linalg.pinv(A)`，内部走 SVD（截掉过小的奇异值，`rcond` 参数控制阈值——实践中几乎总该设一个合理值，**经验做法**：取 `rcond=1e-10 * 最大奇异值` 量级，避免数值噪声被放大）。

```python
A = np.array([[1.0, 2.0], [2.0, 4.0]])   # 秩 1,逆不存在
b = np.array([3.0, 6.0])                  # b 在列空间内
x = np.linalg.pinv(A) @ b
print(x)             # [0.6 1.2]  最小范数解
print(A @ x)         # [3. 6.]   精确还原 b
```

## 本篇小结

- 特征向量是变换的**不变方向**，特征值是该方向的拉伸倍数；求法是解 $\det(A-\lambda I)=0$；对角化 $A=V\Lambda V^{-1}$ 表示"换到特征坐标系里变换只是缩放"。
- 对称矩阵（协方差、核矩阵都是）特征值全实、特征向量正交——谱定理保证了 PCA 的合法性。
- SVD = 旋转·缩放·旋转，对任意矩阵成立；奇异值是 $AA^\top$ 特征值的平方根；**截断 SVD 是秩-k 最优逼近**（Eckart–Young），图像压缩实测压缩比 $\frac{k(m+n+1)}{mn}$。
- 伪逆 $A^+ = V\Sigma^+U^\top$ 是奇异/矩形矩阵的合法替身，给出最小二乘/最小范数解。

## 练习

1. **（手算）** 求 $A=\begin{bmatrix}0&-2\\1&3\end{bmatrix}$ 的特征值与特征向量。（答案：$\lambda=1,2$；$\mathbf{v}_1=[-2,1]^\top$，$\mathbf{v}_2=[-1,1]^\top$。）
2. **（推导）** 证明对称矩阵的不同特征值对应的特征向量正交。（提示：$\lambda_1 \mathbf{v}_1^\top\mathbf{v}_2 = (A\mathbf{v}_1)^\top \mathbf{v}_2 = \mathbf{v}_1^\top A \mathbf{v}_2 = \lambda_2 \mathbf{v}_1^\top\mathbf{v}_2$，故 $(\lambda_1-\lambda_2)\mathbf{v}_1^\top\mathbf{v}_2=0$。）
3. **（手算+编程）** 对 $A=\begin{bmatrix}3&0\\4&5\end{bmatrix}$ 求 $A^\top A$、其特征值、以及 $A$ 的奇异值。（答案：$A^\top A=\begin{bmatrix}25&20\\20&25\end{bmatrix}$，特征值 $45, 5$，奇异值 $\sqrt{45}=3\sqrt5\approx6.708$ 与 $\sqrt5\approx2.236$。用 `np.linalg.svd` 核对。）
4. **（编程）** 把 4.3 节实验里的噪声从 0.05 加大到 0.3，观察需要保留多少奇异值才能恢复主结构；画出"重构误差 vs k"的曲线，说明误差曲线的"拐点"大致对应信号的真实秩。
5. **（编程）** 用 `np.linalg.pinv` 解一个 $m=6, n=2$ 的超定系统（自己造带噪声数据），与 `np.linalg.lstsq` 结果对比；再故意把 $A$ 的第二列设成第一列的副本，观察伪逆如何给出最小范数解。
