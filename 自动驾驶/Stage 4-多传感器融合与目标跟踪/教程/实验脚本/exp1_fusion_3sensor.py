# exp1_fusion_3sensor.py —— 第1周：三传感器 2D 位置观测的加权平均融合
# 固定种子，输出与教程正文一致
import numpy as np

np.random.seed(42)

dt = 0.1                      # 帧 10 Hz
n_frames = 50                 # 5 秒
t = np.arange(n_frames) * dt

# 真值：一辆车沿 x 方向匀速 10 m/s，从 (20, 30) 出发
true_pos = np.stack([20 + 10 * t, 30 + 0 * t], axis=1)   # (n, 2)

# 三个传感器，同一时刻对同一目标各给一个 2D 位置观测（已配准到同一坐标系）
# 误差模型：高斯 + 固定系统偏差（真实传感器都有 bias）
SIGMA = {"lidar": 0.15, "camera": 0.60, "radar": 1.50}
BIAS  = {"lidar": (0.0, 0.0), "camera": (0.0, 0.0), "radar": (0.0, 0.6)}
obs = {}
for name in SIGMA:
    b = np.array(BIAS[name])
    obs[name] = true_pos + b + np.random.randn(n_frames, 2) * SIGMA[name]

def rmse(est, gt):
    return np.sqrt(np.mean(np.sum((est - gt) ** 2, axis=1)))

print("==== 单传感器误差 ====")
for name in SIGMA:
    print(f"{name:8s} RMSE = {rmse(obs[name], true_pos):.3f} m")

# ---- 方法A：朴素等权平均 ----
fuse_avg = (obs["lidar"] + obs["camera"] + obs["radar"]) / 3.0

# ---- 方法B：逆方差加权（inverse-variance weighting）----
# 权重 w_i = 1 / sigma_i^2；融合值 = sum(w_i x_i) / sum(w_i)
sig = np.array([SIGMA["lidar"], SIGMA["camera"], SIGMA["radar"]])
w = 1.0 / sig ** 2                       # 三个标量权重
fuse_ivw = (w[0] * obs["lidar"] + w[1] * obs["camera"] + w[2] * obs["radar"]) / w.sum()
print(f"\n逆方差权重 [lidar, camera, radar] = {w / w.sum()}")

print("\n==== 融合误差对比 ====")
print(f"等权平均   RMSE = {rmse(fuse_avg, true_pos):.3f} m")
print(f"逆方差加权 RMSE = {rmse(fuse_ivw, true_pos):.3f} m")

# 理论验证：若观测是 bias=0、方差已知的高斯，逆方差加权的理论方差
# var_fused = 1 / sum(1/sigma_i^2)
theo = np.sqrt(1.0 / (1.0 / 0.15**2 + 1.0 / 0.60**2 + 1.0 / 1.50**2))
print(f"逆方差加权理论标准差 = {theo:.3f} m")

# ---- 异步时延外推补偿 demo：雷达观测晚到 100 ms ----
print("\n==== 异步时延外推补偿 ====")
speed = 10.0                              # 目标沿 x 速度 m/s
latency = 0.1                             # 雷达数据链路延迟 100 ms
# 雷达在 t 时刻上报的是 t-latency 时刻测到的位置（目标已经又往前开了）
radar_meas_at_t_latency = true_pos.copy()
radar_meas_at_t_latency[:, 0] -= speed * latency   # 测量值"慢了一拍"
v_est = 10.0                              # 雷达自带的径向速度观测
compensated = radar_meas_at_t_latency.copy()
compensated[:, 0] += v_est * latency               # 用速度外推补回
err_raw = rmse(radar_meas_at_t_latency, true_pos)
err_cmp = rmse(compensated, true_pos)
print(f"目标速度 {speed} m/s，时延 {latency*1000:.0f} ms -> 空间错位 {speed*latency:.2f} m")
print(f"补偿前 RMSE = {err_raw:.3f} m")
print(f"外推补偿后 RMSE = {err_cmp:.3f} m")
