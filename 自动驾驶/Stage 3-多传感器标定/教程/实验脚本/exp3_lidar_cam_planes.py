import numpy as np, cv2
rng = np.random.default_rng(2)

# ---- 真值外参：LiDAR 系 -> 相机系（待求量）----
def rodrigues(rx, ry, rz):
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
    return R

R_true = rodrigues(np.radians(3), np.radians(5), np.radians(10))   # roll, pitch, yaw
t_true = np.array([0.05, -0.02, 0.10])                             # 相机在 LiDAR 系中偏移的反向组合

K = np.array([[800.0, 0, 640.0], [0, 800.0, 360.0], [0, 0, 1.0]])

# ---- 标定板：0.8 x 0.6 m 平面，棋盘角点供相机做 PnP ----
BW, BH = 0.8, 0.6
gx, gy = np.meshgrid(np.linspace(-BW/2, BW/2, 9), np.linspace(-BH/2, BH/2, 7))
board_pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])  # 板坐标系

def plane_from_pts(P):
    """SVD 拟合平面：n·x + d = 0，n 单位法向，d 为原点到平面的有符号距离。"""
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c)
    n = Vt[-1]
    d = -n @ c
    return n, d

def make_scene(i):
    """第 i 块标定板：相机观测得到板在相机系的位姿；LiDAR 观测得到板上点云。"""
    r = np.random.default_rng(100 + i)
    # 板在相机系的位姿：正对相机，带不同俯仰/偏航 -> 法向彼此不平行
    R_c = rodrigues(np.radians(r.uniform(-25, 25)), np.radians(r.uniform(-30, 30)), 0)
    t_c = np.array([r.uniform(-0.5, 0.5), r.uniform(-0.3, 0.3), r.uniform(2.0, 3.5)])
    # 板在 LiDAR 系 = T_lidar_cam @ 板在相机系
    R_l = R_true.T @ R_c
    t_l = R_true.T @ (t_c - t_true)
    # LiDAR 在板上采样点（真实 LiDAR 还会采到背景，这里只取板面 + 测距噪声）
    r2 = np.random.default_rng(200 + i)
    u = r2.uniform(-BW/2, BW/2, 400); v = r2.uniform(-BH/2, BH/2, 400)
    pts_cam = (R_c @ np.column_stack([u, v, np.zeros(400)]).T).T + t_c
    pts_lidar = (R_true.T @ (pts_cam - t_true).T).T
    pts_lidar += rng.normal(0, 0.005, pts_lidar.shape)      # LiDAR 测距噪声 5 mm
    return board_pts @ R_c.T + t_c, plane_from_pts(pts_lidar)

# ---- 收集每个板位的两个平面描述 ----
normals_cam, normals_lidar, d_cam, d_lidar = [], [], [], []
for i in range(5):
    pts_c, (n_l, d_l) = make_scene(i)
    n_c, d_c = plane_from_pts(pts_c)
    if n_c @ n_l < 0:                       # SVD 法向方向不定，统一朝向相机
        n_l, d_l = -n_l, -d_l
    normals_cam.append(n_c); normals_lidar.append(n_l)
    d_cam.append(d_c); d_lidar.append(d_l)
normals_cam = np.array(normals_cam); normals_lidar = np.array(normals_lidar)
d_cam = np.array(d_cam); d_lidar = np.array(d_lidar)

# ---- 第一步：求 R（Wahba 问题，SVD / 正交 Procrustes）----
M = normals_cam.T @ normals_lidar            # 3x3 协方差矩阵
U, _, Vt = np.linalg.svd(M)
S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
R_est = U @ S @ Vt

# ---- 第二步：求 t（每块板给一条线性方程 n_c·t = d_l - d_c）----
# 推导：P_cam = R P_lidar + t，同一物理平面在两系中方程相等 -> n_c·t = d_l - d_c
A = normals_cam.copy()
b = d_lidar - d_cam
t_est, *_ = np.linalg.lstsq(A, b, rcond=None)

def r_err(R):
    return np.degrees(np.arccos(np.clip((np.trace(R.T @ R_true) - 1) / 2, -1, 1)))

print("5 块不同姿态的标定板，LiDAR 测距噪声 5 mm：")
print(f"  旋转误差: {r_err(R_est):.4f} deg")
print(f"  平移误差: {np.linalg.norm(t_est - t_true)*1000:.2f} mm")
print(f"  R 真值第一行: {np.round(R_true[0], 5)}")
print(f"  R 估计第一行: {np.round(R_est[0], 5)}")
print(f"  t 真值: {np.round(t_true, 4)} m,  t 估计: {np.round(t_est, 4)} m")

# ---- 法向夹角质量检查：标定前的健康诊断 ----
angs = [np.degrees(np.arccos(np.clip(a @ b, -1, 1)))
        for a, b in zip(normals_cam, normals_lidar)]
print("  各板法向夹角（应接近 0）:", np.round(angs, 3))

# ---- LiDAR 噪声扫描 ----
def run(noise_mm, n_boards=5):
    NC, NL, dc, dl = [], [], [], []
    for i in range(n_boards):
        pts_c, (n_l, d_l) = make_scene_noisy(i, noise_mm)
        n_c, d_c = plane_from_pts(pts_c)
        if n_c @ n_l < 0: n_l, d_l = -n_l, -d_l
        NC.append(n_c); NL.append(n_l); dc.append(d_c); dl.append(d_l)
    NC, NL = np.array(NC), np.array(NL)
    U, _, Vt = np.linalg.svd(NC.T @ NL)
    R_e = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    t_e, *_ = np.linalg.lstsq(NC, np.array(dl) - np.array(dc), rcond=None)
    return r_err(R_e), np.linalg.norm(t_e - t_true) * 1000

def make_scene_noisy(i, noise_mm):
    r = np.random.default_rng(100 + i)
    R_c = rodrigues(np.radians(r.uniform(-25, 25)), np.radians(r.uniform(-30, 30)), 0)
    t_c = np.array([r.uniform(-0.5, 0.5), r.uniform(-0.3, 0.3), r.uniform(2.0, 3.5)])
    r2 = np.random.default_rng(200 + i)
    u = r2.uniform(-BW/2, BW/2, 400); v = r2.uniform(-BH/2, BH/2, 400)
    pts_cam = (R_c @ np.column_stack([u, v, np.zeros(400)]).T).T + t_c
    pts_lidar = (R_true.T @ (pts_cam - t_true).T).T
    pts_lidar += rng.normal(0, noise_mm / 1000, pts_lidar.shape)
    return pts_cam, plane_from_pts(pts_lidar)

print("\nLiDAR 测距噪声扫描（3 个重复，平均）：")
for nz in [0, 5, 20, 50]:
    res = np.array([run(nz) for _ in range(3)])
    print(f"  {nz:3d} mm  ->  旋转误差 {res[:,0].mean():6.3f} deg, 平移误差 {res[:,1].mean():7.2f} mm")
