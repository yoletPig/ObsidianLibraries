import numpy as np, cv2
rng = np.random.default_rng(1)

W, H = 1280, 720
K = np.array([[800.0, 0, 640.0], [0, 800.0, 360.0], [0, 0, 1.0]])
Kinv = np.linalg.inv(K)

def rodrigues(rx, ry, rz):
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
    return R

def project(P, X):
    Xc = (P[:, :3] @ X.T).T + P[:, 3]
    uv = (K @ Xc.T).T
    return uv[:, :2] / uv[:, 2:3]

# ---- 立体靶标：边长 0.4 m 的立方体的 8 个角点 + 每个面中心，真正非共面 ----
S = 0.4
corners = np.array([[x, y, z] for x in (0, S) for y in (0, S) for z in (0, S)], float)
faces = np.array([[S/2, S/2, 0], [S/2, S/2, S], [S/2, 0, S/2], [S/2, S, S/2],
                  [0, S/2, S/2], [S, S/2, S/2]])
X_obj = np.vstack([corners, faces])

# 真值位姿：物体系 -> 相机系（把立方体摆到相机前方稍偏右）
R_true = rodrigues(0.15, -0.3, 0.1)
t_true = np.array([0.5, -0.2, 2.5])
P_true = np.hstack([R_true, t_true.reshape(3, 1)])

def dlt_pnp(X_obj, uv):
    """手写 DLT：投影方程 -> 12 未知数齐次线性方程组，SVD 取最小奇异值的右奇异向量。
    先做 Hartley 归一化（点集平移到质心、缩放到平均距离 ~sqrt(2)），否则病态。"""
    cu, su = uv.mean(0), np.sqrt(2) / uv.std(0).mean()
    T2 = np.array([[su, 0, -su * cu[0]], [0, su, -su * cu[1]], [0, 0, 1]])
    cx, sx = X_obj.mean(0), np.sqrt(2) / X_obj.std(0).mean()
    T3 = np.eye(4); T3[:3, :3] = sx * np.eye(3); T3[:3, 3] = -sx * cx
    uv_n = (T2 @ np.column_stack([uv, np.ones(len(uv))]).T).T[:, :2]
    X_n = (T3 @ np.column_stack([X_obj, np.ones(len(X_obj))]).T).T[:, :3]

    A = np.zeros((2 * len(X_n), 12))
    for i, (Xw, (u, v)) in enumerate(zip(X_n, uv_n)):
        Xh = np.append(Xw, 1.0)
        A[2 * i, 0:4] = Xh;           A[2 * i, 8:12] = -u * Xh
        A[2 * i + 1, 4:8] = Xh;       A[2 * i + 1, 8:12] = -v * Xh
    _, _, Vt = np.linalg.svd(A)
    P_n = Vt[-1].reshape(3, 4)
    return np.linalg.inv(T2) @ P_n @ T3      # 反归一化回原坐标系

def decompose(P):
    """P = K'[R|t]：RQ 分解拆出 K、R，t = K^-1 p4。返回 R, t（欧氏坐标）。"""
    from scipy.linalg import rq
    P = P / np.linalg.norm(P[2, :3])
    if P[2, 3] < 0:
        P = -P
    K_r, R = rq(P[:, :3])
    D = np.diag(np.sign(np.diag(K_r)))
    K = K_r @ D; R = D @ R; K /= K[2, 2]
    return R, np.linalg.inv(K) @ P[:, 3]

def r_err(R_est):
    return np.degrees(np.arccos(np.clip((np.trace(R_est.T @ R_true) - 1) / 2, -1, 1)))

def solve(noise_px, n_pts):
    idx = rng.choice(len(X_obj), n_pts, replace=False)
    Xs = X_obj[idx]
    uv = project(P_true, Xs) + rng.normal(0, noise_px, (n_pts, 2))

    # 方法 A：手写 DLT（线性，SVD，解的是 11 自由度的完整 P）
    R_dlt, t_dlt = decompose(dlt_pnp(Xs, uv))

    # 方法 B：cv2.solvePnP（K 已知，只解 6 自由度，非线性 LM 精化）
    ok, rvec, t_pnp = cv2.solvePnP(Xs.astype(np.float32), uv.astype(np.float32), K, None,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    R_pnp = rodrigues(*rvec.ravel())
    return (r_err(R_dlt), np.linalg.norm(t_dlt - t_true),
            r_err(R_pnp), np.linalg.norm(t_pnp.ravel() - t_true))

# ---- 主实验：14 个点，噪声 0.5 px ----
re, te, re2, te2 = solve(0.5, 14)
print("14 个 3D 点、噪声 0.5 px：")
print(f"  手写 DLT + SVD      : 旋转误差 {re:6.3f} deg, 平移误差 {te*100:6.2f} cm")
print(f"  cv2.solvePnP (LM)   : 旋转误差 {re2:6.3f} deg, 平移误差 {te2*100:6.2f} cm")

# ---- 噪声扫描（5 个种子平均）----
print("\n噪声扫描（14 点，5 个种子平均）：")
print(f"  {'噪声(px)':>8} | {'DLT 旋转误差':>12} {'DLT 平移误差':>13} | {'PnP 旋转误差':>12} {'PnP 平移误差':>12}")
for sigma in [0.0, 0.5, 1.0, 2.0]:
    res = np.array([solve(sigma, 14) for _ in range(5)])
    print(f"  {sigma:8.1f} | {res[:,0].mean():10.3f}°  {res[:,1].mean()*100:9.2f}cm  | "
          f"{res[:,2].mean():10.3f}°  {res[:,3].mean()*100:8.2f}cm")

# ---- 点数扫描 ----
print("\n点数扫描（噪声 0.5 px，5 个种子平均；靶标共 14 点，6 点为理论最小值）：")
for n in [6, 10, 14]:
    res = np.array([solve(0.5, n) for _ in range(5)])
    print(f"  {n:2d} 点 | DLT 旋转 {res[:,0].mean():7.3f}°  平移 {res[:,1].mean()*100:7.2f} cm | "
          f"PnP 旋转 {res[:,2].mean():7.3f}°  平移 {res[:,3].mean()*100:7.2f} cm")
