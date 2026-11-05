import numpy as np
rng = np.random.default_rng(6)

# ---- 场景：车辆绕弯，yaw 角速度是一个平滑信号 ----
# 真实角速度 w(t)（rad/s），由几个正弦叠加模拟转弯过程
def yaw_rate(t):
    return 0.08 * np.sin(2 * np.pi * 0.5 * t) + 0.05 * np.sin(2 * np.pi * 0.23 * t + 1.0) \
           + 0.02 * np.sin(2 * np.pi * 1.1 * t + 2.0)

T_END = 20.0
# IMU：100 Hz，几乎无延迟
t_imu = np.arange(0, T_END, 0.01)
w_imu = yaw_rate(t_imu) + rng.normal(0, 0.002, len(t_imu))
# 相机侧信号（如特征点横向视差推算的角速度）：20 Hz，且时钟比 IMU 慢 37 ms
DT_TRUE = 0.037
t_cam = np.arange(0, T_END, 0.05)
w_cam = yaw_rate(t_cam + DT_TRUE) + rng.normal(0, 0.006, len(t_cam))
# 含义：t_cam 标签下的 w_cam 实际是 (t_cam + DT_TRUE) 时刻的物理量

# ---- 估计：在候选偏移 dt 上把相机信号对齐到 IMU 时间轴，选残差最小的 ----
grid = np.arange(-0.10, 0.1001, 0.001)
rmse = np.array([np.sqrt(np.mean((np.interp(t_imu, t_cam + dt, w_cam) - w_imu) ** 2))
                 for dt in grid])
i = int(np.argmin(rmse))
dt_est, rmse_best = grid[i], rmse[i]
print(f"真实时间偏移  : {DT_TRUE*1000:.0f} ms")
print(f"估计时间偏移  : {dt_est*1000:.1f} ms  (对齐后 RMSE = {rmse_best:.5f} rad/s)")
print(f"120 km/h 下 37 ms 错位 = {120/3.6*0.037:.2f} m 的空间错位")
print(f"残余错位 {abs(dt_est-DT_TRUE)*1000:.1f} ms = {120/3.6*abs(dt_est-DT_TRUE)*100:.1f} cm")

# ---- 对比：不校时间直接用，误差多大 ----
pred0 = np.interp(t_imu, t_cam, w_cam)
rmse0 = np.sqrt(np.mean((pred0 - w_imu) ** 2))
print(f"\n不校时间 RMSE = {rmse0:.5f} rad/s, 校准后 = {rmse_best:.5f} rad/s "
      f"(降到 {rmse_best/rmse0*100:.0f}%)")

# ---- 精化：抛物线插值细化谷底（亚网格精度），并跨噪声种子看统计 ----
y0, y1, y2 = rmse[i-1], rmse[i], rmse[i+1]
den = y0 - 2*y1 + y2
dt_ref = grid[i] + (0.5 * (y0 - y2) / den if den != 0 else 0.0) * 0.001
print(f"\n抛物线细化: dt = {dt_ref*1000:.2f} ms (真值 37.00 ms, 残差 {abs(dt_ref-DT_TRUE)*1000:.2f} ms)")

est = []
for s in range(5):
    r = np.random.default_rng(100 + s)
    w_i = yaw_rate(t_imu) + r.normal(0, 0.002, len(t_imu))
    w_c = yaw_rate(t_cam + DT_TRUE) + r.normal(0, 0.006, len(t_cam))
    rr = np.array([np.sqrt(np.mean((np.interp(t_imu, t_cam + dt, w_c) - w_i) ** 2))
                   for dt in grid])
    j = int(np.argmin(rr))
    a, b, c = rr[j-1], rr[j], rr[j+1]
    den = a - 2*b + c
    est.append((grid[j] + (0.5 * (a - c) / den if den != 0 else 0.0) * 0.001) * 1000)
print("5 个噪声种子的细化估计 (ms):", np.round(est, 2))
print(f"均值 {np.mean(est):.2f} ms, 标准差 {np.std(est):.2f} ms")

# ---- 120 km/h 的账：不同时间偏移对应的空间错位 ----
print("\n120 km/h (33.3 m/s) 下时间偏移的空间代价：")
for ms in [1, 5, 10, 20, 33, 50]:
    print(f"  {ms:2d} ms  ->  {33.33*ms/1000:5.2f} m")
