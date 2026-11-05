import numpy as np, cv2
from itertools import combinations
from scipy.optimize import least_squares
rng = np.random.default_rng(4)

def make_T(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T

def rand_R(max_deg=50, min_deg=20):
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    R, _ = cv2.Rodrigues(axis * np.radians(rng.uniform(min_deg, max_deg)))
    return R

def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def rot2quat2(R):
    """Tsai 的 P = 2 sin(theta/2) * 轴（= 2 倍四元数向量部）"""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-12:
        return np.zeros(3)
    n = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return 2 * np.sin(th / 2) * n

def r_err(R, R_true):
    return np.degrees(np.arccos(np.clip((np.trace(R.T @ R_true) - 1) / 2, -1, 1)))

def tsai_lenz(G, C):
    """两步法（Tsai & Lenz 1989，OpenCV 同源实现）：
    1) 旋转：skew(Pg+Pc) @ x = Pc-Pg（x 是 Gibbs 向量 tan(th/2)*axis），SVD 最小二乘，
       再按 eq14 缩放回单位四元数向量部；
    2) 平移：(Rg - I) @ t = R_x @ tc - tg，逐运动对堆叠成超定方程。
    只保留旋转角在约 17°~120° 之间的运动对（太小/接近 pi 的对信息量差）。"""
    M, y, keep = [], [], []
    for i, j in combinations(range(len(G)), 2):
        Hg = np.linalg.inv(G[j]) @ G[i]        # 末端运动 A
        Hc = C[j] @ np.linalg.inv(C[i])        # 相机运动 B（AX=XB）
        Pg, Pc = rot2quat2(Hg[:3, :3]), rot2quat2(Hc[:3, :3])
        if not (0.3 <= np.linalg.norm(Pg) <= 1.7 and 0.3 <= np.linalg.norm(Pc) <= 1.7):
            continue
        M.append(skew(Pg + Pc)); y.append(Pc - Pg); keep.append((Hg, Hc))
    assert len(keep) >= 2, "有效运动对不足：需要更大、更多样的旋转"
    x, *_ = np.linalg.lstsq(np.vstack(M), np.hstack(y), rcond=None)
    q = x / np.sqrt(1 + x @ x)                 # eq14：Gibbs -> 单位四元数向量部
    w = np.sqrt(max(0.0, 1 - q @ q))
    Rx = 2 * np.outer(q, q) + 2 * w * skew(q) + np.eye(3) - 2 * (q @ q) * np.eye(3)
    Mt, yt = [], []
    for Hg, Hc in keep:
        Mt.append(Hg[:3, :3] - np.eye(3))
        yt.append(Rx @ Hc[:3, 3] - Hg[:3, 3])
    t, *_ = np.linalg.lstsq(np.vstack(Mt), np.hstack(yt), rcond=None)
    return make_T(Rx, t)

def refine(X0, G, C):
    """非线性精化：直接最小化所有运动对的 ||AX - XB||_F（几何误差，比线性法更准）。"""
    def unpack(p):
        R, _ = cv2.Rodrigues(p[:3])
        return make_T(R, p[3:6])
    def resid(p):
        X = unpack(p)
        r = []
        for i, j in combinations(range(len(G)), 2):
            A = np.linalg.inv(G[j]) @ G[i]
            B = C[j] @ np.linalg.inv(C[i])
            r.append((A @ X - X @ B).ravel())
        return np.hstack(r)
    p0 = np.hstack([cv2.Rodrigues(X0[:3, :3])[0].ravel(), X0[:3, 3]])
    sol = least_squares(resid, p0, method="lm")
    return unpack(sol.x)

# ---- 场景：eye-in-hand，基座上有固定标定板 ----
X_true = make_T(rand_R(), np.array([0.05, -0.03, 0.12]))       # 待求：相机系 -> 末端法兰系
T_base_target = make_T(rand_R(), np.array([1.2, 0.4, 0.3]))    # 标定板在基座系（固定）

def make_poses(n, noise_rot_deg, noise_trans_mm):
    G, C = [], []
    for i in range(n):
        g = make_T(rand_R(50, 25), np.array([rng.uniform(-0.3, 0.3),
                                             rng.uniform(-0.3, 0.3), rng.uniform(0.4, 0.9)]))
        G.append(g)
        Cb = np.linalg.inv(g @ X_true) @ T_base_target         # 相机看到的板（真值）
        w = rng.normal(0, np.radians(noise_rot_deg), 3)
        Rn, _ = cv2.Rodrigues(w)
        C.append(make_T(Cb[:3, :3] @ Rn, Cb[:3, 3] + rng.normal(0, noise_trans_mm / 1000, 3)))
    return G, C

def run(n, nr, nt):
    G, C = make_poses(n, nr, nt)
    X_tl = tsai_lenz(G, C)
    X_lm = refine(X_tl, G, C)
    e = lambda X: (r_err(X[:3, :3], X_true[:3, :3]),
                   np.linalg.norm(X[:3, 3] - X_true[:3, 3]) * 1000)
    return e(X_tl), e(X_lm)

# ---- 结构自检：无噪声必须精确恢复 ----
G0, C0 = make_poses(8, 0.0, 0.0)
X0 = tsai_lenz(G0, C0)
print("无噪声自检: Tsai 旋转误差 %.2e deg, 平移误差 %.2e mm"
      % (r_err(X0[:3, :3], X_true[:3, :3]), np.linalg.norm(X0[:3, 3] - X_true[:3, 3]) * 1000))

# ---- 主实验 ----
(e1, e2), (e3, e4) = run(10, 0.1, 1.0)
print("\n10 组末端位姿、观测噪声 0.1 deg / 1 mm：")
print(f"  Tsai-Lenz 线性两步法   : 旋转误差 {e1:6.3f} deg, 平移误差 {e2:6.2f} mm")
print(f"  + 非线性精化 (LM)      : 旋转误差 {e3:6.3f} deg, 平移误差 {e4:6.2f} mm")

# ---- 位姿组数扫描 ----
print("\n位姿组数扫描（噪声 0.1 deg / 1 mm，5 个重复平均）：")
for n in [3, 4, 6, 10, 20]:
    res = np.array([run(n, 0.1, 1.0) for _ in range(5)])
    print(f"  {n:2d} 组 -> 线性旋转 {res[:,0,0].mean():7.3f} deg 平移 {res[:,0,1].mean():7.2f} mm | "
          f"精化旋转 {res[:,1,0].mean():7.3f} deg 平移 {res[:,1,1].mean():7.2f} mm")

# ---- 噪声扫描 ----
print("\n观测噪声扫描（10 组位姿，5 个重复平均）：")
for nr, nt in [(0.0, 0.0), (0.1, 1.0), (0.5, 5.0), (1.0, 10.0)]:
    res = np.array([run(10, nr, nt) for _ in range(5)])
    print(f"  {nr:.1f} deg/{nt:4.1f} mm -> 线性旋转 {res[:,0,0].mean():7.3f} deg 平移 {res[:,0,1].mean():7.2f} mm | "
          f"精化旋转 {res[:,1,0].mean():7.3f} deg 平移 {res[:,1,1].mean():7.2f} mm")
