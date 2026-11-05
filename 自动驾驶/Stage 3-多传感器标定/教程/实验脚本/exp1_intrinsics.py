import numpy as np, cv2
rng = np.random.default_rng(0)

W, H = 1280, 720
K_true = np.array([[800.0, 0, 640.0], [0, 802.0, 360.0], [0, 0, 1.0]])
dist_true = np.array([-0.15, 0.05, 0.001, -0.001, 0.0])  # OpenCV 顺序 (k1, k2, p1, p2, k3)

# 9x6 内角点棋盘，方格边长 0.06 m（约 54x36 cm 的大棋盘）
NX, NY, SQ = 9, 6, 0.06
objp = np.zeros((NX * NY, 3), np.float32)
objp[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2) * SQ

def make_pose(seed, near, wide, tilt):
    """随机棋盘位姿（相机系）。near/wide 控制远近与偏移范围，tilt 控制倾斜幅度。"""
    r = np.random.default_rng(seed)
    R, _ = cv2.Rodrigues(np.array([r.uniform(-tilt, tilt), r.uniform(-tilt, tilt),
                                   r.uniform(-0.5, 0.5)]))
    tx = 0.55 if wide else 0.12
    ty = 0.35 if wide else 0.08
    z0, z1 = (0.55, 1.6) if near else (0.9, 2.0)
    t = np.array([r.uniform(-tx, tx), r.uniform(-ty, ty), r.uniform(z0, z1)])
    pts, _ = cv2.projectPoints(objp, cv2.Rodrigues(R)[0], t, K_true, dist_true)
    pts = pts.reshape(-1, 2)
    if pts[:, 0].min() > 5 and pts[:, 0].max() < W - 5 and pts[:, 1].min() > 5 and pts[:, 1].max() < H - 5:
        return R, t
    return None

def calibrate(noise_px, n_views, near=True, wide=True, tilt=0.4, seed0=0):
    objpts, imgpts = [], []
    got, s = 0, seed0
    while got < n_views:
        p = make_pose(s, near, wide, tilt); s += 1
        if p is None:
            continue
        R, t = p
        pts, _ = cv2.projectPoints(objp, cv2.Rodrigues(R)[0], t, K_true, dist_true)
        pts = pts.reshape(-1, 2) + rng.normal(0, noise_px, (NX * NY, 2))
        objpts.append(objp.copy()); imgpts.append(pts.astype(np.float32)); got += 1
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, (W, H), None, None)
    return rms, K, np.asarray(dist).ravel()

# ---- 主实验：满幅覆盖，噪声 0.3 px，20 张图 ----
rms, K, dist = calibrate(0.3, 20)
print(f"重投影 RMS = {rms:.3f} px")
print("            fx      fy      cx      cy")
print(f"真值      {K_true[0,0]:7.1f}  {K_true[1,1]:7.1f}  {K_true[0,2]:7.1f}  {K_true[1,2]:7.1f}")
print(f"估计      {K[0,0]:7.1f}  {K[1,1]:7.1f}  {K[0,2]:7.1f}  {K[1,2]:7.1f}")
print(f"误差       {K[0,0]-K_true[0,0]:+7.1f}  {K[1,1]-K_true[1,1]:+7.1f}  {K[0,2]-K_true[0,2]:+7.1f}  {K[1,2]-K_true[1,2]:+7.1f}")
names = ["k1", "k2", "p1", "p2", "k3"]
print("畸变:", "  ".join(f"{n}: est={d:+.4f} true={t:+.4f}" for n, d, t in zip(names, dist, dist_true)))

# ---- 对照：棋盘只摆画面中心（远、偏移小）----
print("\n坑演示：同样的 20 张图，但棋盘只出现在画面中心区域")
rms2, K2, dist2 = calibrate(0.3, 20, near=False, wide=False, tilt=0.2)
print(f"  RMS = {rms2:.3f} px   fx = {K2[0,0]:.1f} (err {K2[0,0]-800:+.1f})   "
      f"k1 = {dist2[0]:+.3f} (true -0.150)   k2 = {dist2[1]:+.3f} (true +0.050)   k3 = {dist2[4]:+.3f} (true 0)")
print("  -> RMS 更低不代表更准：畸变参数没被约束住，误差被'吃'进了 fx/fy")

# ---- 噪声扫描 ----
print("\n角点噪声对 fx 估计的影响（20 张图，3 个种子平均）：")
for sigma in [0.0, 0.1, 0.3, 0.5, 1.0]:
    errs = [abs(calibrate(sigma, 20, seed0=s * 100)[1][0, 0] - 800.0) for s in range(3)]
    print(f"  sigma = {sigma:.1f} px  ->  |fx - 800| 平均 {np.mean(errs):5.2f} px (最差 {np.max(errs):5.2f})")

# ---- 图像数量扫描 ----
print("\n图像数量对 fx 估计的影响（噪声 0.3 px，3 个种子平均）：")
for n in [3, 5, 10, 20]:
    errs = [abs(calibrate(0.3, n, seed0=s * 100)[1][0, 0] - 800.0) for s in range(3)]
    print(f"  {n:2d} 张图  ->  |fx - 800| 平均 {np.mean(errs):5.2f} px (最差 {np.max(errs):5.2f})")
