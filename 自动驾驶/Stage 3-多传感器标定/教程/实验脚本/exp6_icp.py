import numpy as np
from scipy.spatial import cKDTree
import cv2
rng = np.random.default_rng(5)

# ---- 场景：两面墙 + 地面（ LiDAR A 视角）。大平面 + 稀疏采样，正是 ICP 的危险场景 ----
n_pts = 800
wall1 = np.column_stack([rng.uniform(-4, 4, n_pts), rng.uniform(-2, 2, n_pts), np.zeros(n_pts)])
ground = np.column_stack([rng.uniform(-4, 4, n_pts), np.full(n_pts, -2.0), rng.uniform(0, 3, n_pts)])
wall2 = np.column_stack([np.full(n_pts, 4.0), rng.uniform(-2, 2, n_pts), rng.uniform(0, 3, n_pts)])
cloudA = np.vstack([wall1, ground, wall2])               # 2400 点，单位：米

# 目标云的法向：KNN 邻域 PCA 取最小特征值方向（真实 LiDAR 也这么估计）
def estimate_normals(pts, k=15):
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k)
    nbrs = pts[idx]                                       # (N, k, 3)
    centered = nbrs - nbrs.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k
    w, v = np.linalg.eigh(cov)
    return v[:, :, 0]                                     # 最小特征值对应的方向

normalsA = estimate_normals(cloudA)

# ---- LiDAR B：真实外参 + 遮挡 + 噪声 ----
# 约定：cloudB 中的点在 A 系中对应点为 cloudA[B_idx]，即 cloudA[B] = R_true @ cloudB + t_true，
# 因此 icp(cloudB -> cloudA) 解出的 (R, t) 就应当等于 (R_true, t_true)。
R_true, _ = cv2.Rodrigues(np.array([np.radians(2), np.radians(3), np.radians(5)]))
t_true = np.array([0.30, -0.20, 0.10])
B_idx = rng.choice(len(cloudA), 1800, replace=False)
cloudB = (R_true.T @ (cloudA[B_idx] - t_true).T).T + rng.normal(0, 0.01, (1800, 3))

def report(R_est, t_est, label, rms):
    ang = np.degrees(np.arccos(np.clip((np.trace(R_est.T @ R_true) - 1) / 2, -1, 1)))
    print(f"  {label:<26s} 旋转误差 {ang:7.3f} deg   平移误差 {np.linalg.norm(t_est - t_true)*100:6.2f} cm   RMS {rms*100:5.2f} cm")
    return ang

def icp_point2point(src, dst, max_iter=60, tol=1e-9):
    R = np.eye(3); t = np.zeros(3)
    tree = cKDTree(dst); prev = np.inf
    for it in range(max_iter):
        cur = (R @ src.T).T + t
        dist, idx = tree.query(cur)
        mu_s, mu_d = cur.mean(0), dst[idx].mean(0)
        H = (cur - mu_s).T @ (dst[idx] - mu_d)
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        Rs = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        ts = mu_d - Rs @ mu_s
        R, t = Rs @ R, Rs @ t + ts
        rms = np.sqrt((dist ** 2).mean())
        if abs(prev - rms) < tol: break
        prev = rms
    return R, t, rms

def icp_point2plane(src, dst, dst_normals, max_iter=60, tol=1e-9):
    """点-面 ICP：每步把小扭转 delta=(v, w) 线性化后最小二乘。
    残差用点到切平面的距离 n·(p_dst - p_cur)，比点到点的距离有判别力得多。"""
    R = np.eye(3); t = np.zeros(3)
    tree = cKDTree(dst); prev = np.inf
    for it in range(max_iter):
        cur = (R @ src.T).T + t
        dist, idx = tree.query(cur)
        n = dst_normals[idx]                              # (N,3)
        b = np.einsum("ij,ij->i", n, dst[idx] - cur)      # 残差 n·(dst-cur)
        A = np.hstack([n, np.cross(cur, n)])              # 每行 [n, cur×n]，未知量 (v,w)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        v, w = sol[:3], sol[3:]
        Rs, _ = cv2.Rodrigues(w)                          # 小扭转左乘到当前变换
        R, t = Rs @ R, Rs @ t + v
        rms = np.sqrt((dist ** 2).mean())
        if abs(prev - rms) < tol: break
        prev = rms
    return R, t, rms

print("同一份数据、同一初值（恒等变换），两种 ICP 的结局：")
R1, t1, rms1 = icp_point2point(cloudB, cloudA)
report(R1, t1, "点-点 ICP", rms1)
R2, t2, rms2 = icp_point2plane(cloudB, cloudA, normalsA)
report(R2, t2, "点-面 ICP", rms2)

# ---- 大初值误差：掉坑演示（收敛域有限）----
R0, _ = cv2.Rodrigues(np.array([0, 0, np.radians(90)]))
src0 = (R0 @ cloudB.T).T
R3, t3, rms3 = icp_point2plane(src0, cloudA, normalsA)
R_tot = R3 @ R0
ang3 = np.degrees(np.arccos(np.clip((np.trace(R_tot.T @ R_true) - 1) / 2, -1, 1)))
print(f"\n初值差 90°: 点-面 ICP 旋转误差 {ang3:.2f} deg, RMS {rms3*100:.2f} cm  <- 收敛到了错误位姿")
print("-> ICP 只在初值附近收敛；大初值误差需要 NDT/特征配准/粗对准先兜底")
