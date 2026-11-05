# exp2_kalman_qr.py —— 第2周：CV 模型 KF + 调 Q/R 数值实验（固定种子）
import numpy as np

np.random.seed(7)
dt = 0.1
N = 100                                   # 10 s @ 10 Hz

# 真值：0~5s 匀速 10 m/s；5~10s 匀加速 +4 m/s^2（机动，速度 10 -> 30 m/s）
t = np.arange(N) * dt
acc = np.where(t < 5.0, 0.0, 4.0)
v_true = 10.0 + np.cumsum(acc) * dt
p_true = 20.0 + np.cumsum(v_true) * dt

F = np.array([[1.0, dt], [0.0, 1.0]])     # CV 状态转移：x' = x + v*dt
Q_OFF = np.diag([1e-4, 1e-3])             # 过程噪声基准（模型不确定度）

def run_kf(sigma_z, q_scale):
    """x = [位置 p, 速度 v]，观测只有 p。初值速度已知（已跟踪一段）。"""
    R = sigma_z ** 2
    Q = q_scale * Q_OFF
    H = np.array([[1.0, 0.0]])
    x = np.array([p_true[0], 10.0])       # 初始状态给准，排除启动瞬态干扰
    P = np.diag([1.0, 1.0])
    est = np.zeros((N, 2))
    for k in range(N):
        # ---- 预测 ----
        x = F @ x
        P = F @ P @ F.T + Q
        # ---- 更新 ----
        z = p_true[k] + sigma_z * np.random.randn()
        y = z - (H @ x)[0]                    # 残差（新息）
        S = (H @ P @ H.T)[0, 0] + R           # 残差方差
        K = (P @ H.T / S).ravel()             # 卡尔曼增益
        x = x + K * y
        P = (np.eye(2) - np.outer(K, H)) @ P
        est[k] = x
    return est

def report(name, est):
    rmse_p = np.sqrt(np.mean((est[:, 0] - p_true) ** 2))
    # 机动后 (t in [6,10] s) 的平均滞后：估计位置落后真值多少（正=落后）
    w = t >= 6.0
    lag = np.mean(p_true[w] - est[w, 0])
    print(f"{name:26s} 位置RMSE = {rmse_p:6.3f} m   机动后平均滞后 = {lag:+7.3f} m")
    return rmse_p, lag

print("==== 固定 Q，改变 R（R 大 -> 更信模型、响应变慢）====")
est_R1 = run_kf(sigma_z=1.0, q_scale=1.0)
report("R=1m^2 (信测量, 响应快)", est_R1)
est_R25 = run_kf(sigma_z=5.0, q_scale=1.0)
report("R=25m^2 (半信半疑)", est_R25)
est_R400 = run_kf(sigma_z=20.0, q_scale=1.0)
report("R=400m^2 (几乎不信测量)", est_R400)

print("\n==== 固定 R=1，改变 Q（Q 大 -> 更信'模型会变'、跟机动快但更抖）====")
report("Q x 0.01 (死信CV模型)", run_kf(1.0, 0.01))
report("Q x 1   (基准)", est_R1)
report("Q x 100 (很信机动)", run_kf(1.0, 100.0))
