# exp3_ekf_ukf.py —— 第2周：极坐标雷达观测下的 线性KF(冻结线性化) vs EKF vs UKF
# 外加协方差不确定度椭圆收缩图（matplotlib，CPU 可跑）
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

np.random.seed(11)
dt = 0.1
N = 80                                     # 8 s
SENSOR = np.array([0.0, 0.0])              # 雷达位于原点
SIG_R, SIG_TH = 1.0, np.radians(0.5)       # 测距 1 m，测角 0.5 deg

# 真值：从 (120, 60) 出发，v = (-12, -5) m/s，一路开向雷达附近
x_true = np.zeros((N, 4))
x_true[0] = [120.0, 60.0, -12.0, -5.0]
F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
              [0, 0, 1, 0], [0, 0, 0, 1]])
Q = np.diag([0.01, 0.01, 0.04, 0.04]) * dt
for k in range(1, N):
    x_true[k] = F @ x_true[k - 1]

# 观测：极坐标 (r, theta)
def h(x):
    d = x[:2] - SENSOR
    r = np.hypot(d[0], d[1])
    th = np.arctan2(d[1], d[0])
    return np.array([r, th])

def jacobian_H(x):
    d = x[:2] - SENSOR
    r = np.hypot(d[0], d[1])
    return np.array([[d[0] / r, d[1] / r, 0, 0],
                     [-d[1] / r**2, d[0] / r**2, 0, 0]])

Z = np.array([h(x) + np.array([SIG_R * np.random.randn(),
                               SIG_TH * np.random.randn()]) for x in x_true])

# ---------- 三种滤波器共用：预测 ----------
def predict(x, P):
    return F @ x, F @ P @ F.T + Q

# ---------- 滤波器 1：线性 KF，把观测线性化一次后冻结 ----------
# 在第 0 帧真值附近做泰勒展开：z ≈ h(x0) + H0 (x - x0)，之后永远用这个线性模型
H0 = jacobian_H(x_true[0])
z0 = h(x_true[0])
b0 = z0 - H0 @ x_true[0]                   # 冻结的偏置项
R_mat = np.diag([SIG_R**2, SIG_TH**2])

def kf_frozen(x, P, z):
    x, P = predict(x, P)
    y = z - (H0 @ x + b0)                  # 线性观测模型
    S = H0 @ P @ H0.T + R_mat
    K = P @ H0.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye(4) - K @ H0) @ P
    return x, P

# ---------- 滤波器 2：EKF，每步在当前预测点重新线性化 ----------
def ekf(x, P, z):
    x, P = predict(x, P)
    H = jacobian_H(x)                      # 关键区别：每次都用最新的 x
    y = z - h(x)
    S = H @ P @ H.T + R_mat
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye(4) - K @ H) @ P
    return x, P

# ---------- 滤波器 3：UKF，sigma 点过非线性函数，不求雅可比 ----------
def ukf(x, P, z):
    n = 4
    alpha, beta, kappa = 1e-3, 2.0, 0.0
    lam = alpha**2 * (n + kappa) - n
    c = n + lam
    Wm = np.full(2 * n + 1, 1.0 / (2 * c)); Wm[0] = lam / c
    Wc = Wm.copy(); Wc[0] += 1 - alpha**2 + beta
    x, P = predict(x, P)
    sqrtP = np.linalg.cholesky(c * P)
    Xs = np.vstack([x, x + sqrtP.T, x - sqrtP.T])   # 2n+1 个 sigma 点
    Zs = np.array([h(s) for s in Xs])               # 整体过非线性观测函数
    z_hat = Wm @ Zs
    dz = Zs - z_hat
    S = R_mat + sum(Wc[i] * np.outer(dz[i], dz[i]) for i in range(2 * n + 1))
    dx = Xs - x
    C = sum(Wc[i] * np.outer(dx[i], dz[i]) for i in range(2 * n + 1))
    K = C @ np.linalg.inv(S)
    x = x + K @ (z - z_hat)
    P = P - K @ S @ K.T
    return x, P

def run(step_fn):
    x = np.array([120.0, 60.0, 0.0, 0.0])            # 速度未知，初值 0
    P = np.diag([25.0, 25.0, 100.0, 100.0])
    est = np.zeros((N, 4)); Ps = np.zeros((N, 2, 2))
    for i in range(N):
        x, P = step_fn(x, P, Z[i])
        est[i] = x; Ps[i] = P[:2, :2]
    return est, Ps

print("==== 极坐标雷达跟踪：三种滤波器位置误差对比 ====")
results = {}
for fn, name in [(kf_frozen, "KF(冻结线性化)"), (ekf, "EKF(每步重线性化)"), (ukf, "UKF(sigma点)")]:
    est, Ps = run(fn)
    rmse = np.sqrt(np.mean(np.sum((est[:, :2] - x_true[:, :2]) ** 2, axis=1)))
    print(f"{name:20s} 位置RMSE = {rmse:7.3f} m")
    results[name] = (est, Ps)

# 不确定度椭圆收缩：EKF 位置协方差
est_ekf, Ps_ekf = results["EKF(每步重线性化)"]
print("\n==== EKF 位置不确定度椭圆（2sigma 长轴）====")
for i in [0, 5, 10, 20, 40, 79]:
    a = 2 * np.sqrt(np.linalg.eigvalsh(Ps_ekf[i])[1])
    print(f"帧 {i:3d} (距离 {h(x_true[i])[0]:6.1f} m): 2sigma 长轴 = {a:6.3f} m")

# 画图
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_true[:, 0], x_true[:, 1], "k-", label="ground truth")
ax.plot(Z[:, 0] * np.cos(Z[:, 1]), Z[:, 0] * np.sin(Z[:, 1]), "r.", ms=3,
        alpha=0.5, label="meas (polar->xy)")
ax.plot(est_ekf[:, 0], est_ekf[:, 1], "b--", label="EKF")
for i in [0, 5, 10, 20, 40, 79]:
    evals, evecs = np.linalg.eigh(Ps_ekf[i])
    ang = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
    ell = Ellipse(est_ekf[i, :2], 4 * np.sqrt(evals[1]), 4 * np.sqrt(evals[0]),
                  angle=ang, fill=False, color="b")
    ax.add_patch(ell)
ax.plot(0, 0, "k^", ms=10, label="radar")
ax.set_aspect("equal"); ax.legend(); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("EKF tracking with covariance ellipses (2 sigma)")
fig.savefig("/tmp/ad_s4/cov_ellipse.png", dpi=120)
print("\n图已保存: /tmp/ad_s4/cov_ellipse.png")
