# exp3_pillars_shape_chain.py —— 第 2 周：NumPy 模拟 PointPillars 的 pillar 化 + PFN + scatter 全流程
# 复现：python3 exp3_pillars_shape_chain.py（无需参数，输出与教程"真实运行输出"一致）
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
