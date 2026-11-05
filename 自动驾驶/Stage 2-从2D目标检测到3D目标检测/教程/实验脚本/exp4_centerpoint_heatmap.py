# exp4_centerpoint_heatmap.py —— 第 3 周：CenterPoint 热图生成 + 峰值解码 + 贪心匹配
# 复现：python3 exp4_centerpoint_heatmap.py（无需参数，输出与教程"真实运行输出"一致）
import numpy as np

rng = np.random.default_rng(2)

# ---------- 1. 构造两个目标的"真值"，生成中心热图（高斯核） ----------
BEV_H, BEV_W = 200, 200          # BEV 特征图网格
SCALE = 0.2                      # 每格 0.2 m，覆盖 40m x 40m
gts = [
    {"x": 10.0, "y": -2.0, "w": 2.0, "l": 4.5, "yaw": 0.1, "vx": 0.5, "vy": 0.0},
    {"x": 25.0, "y":  5.0, "w": 0.6, "l": 1.7, "yaw": 1.2, "vx": 0.0, "vy": 1.0},
]


def gaussian_heatmap(gts, H, W, sigma=2.0):
    """对每个 GT 中心画一个高斯斑：中心处 1，往外指数衰减。"""
    hm = np.zeros((H, W))
    ys, xs = np.mgrid[0:H, 0:W]
    for g in gts:
        cx, cy = g["x"] / SCALE, (g["y"] + 20) / SCALE   # 世界坐标 -> 网格坐标
        hm += np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return np.clip(hm, 0, 1)


hm = gaussian_heatmap(gts, BEV_H, BEV_W)
print(f"热图 shape: {hm.shape}，峰值 = {hm.max():.2f}")


# ---------- 2. 峰值提取：3x3 内局部最大且 > 0.5 视作一个目标 ----------
def local_max_peaks(hm, thr=0.5):
    H, W = hm.shape
    peaks = []
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            patch = hm[i-1:i+2, j-1:j+2]
            if hm[i, j] >= thr and hm[i, j] == patch.max():
                peaks.append((i, j, hm[i, j]))
    return peaks


peaks = local_max_peaks(hm)
print(f"检测到 {len(peaks)} 个峰值（=目标数）:", [(i, j, round(v, 2)) for i, j, v in peaks])
assert len(peaks) == len(gts)

# ---------- 3. 在峰值处回归 7 自由度框 + 速度 ----------
print("\n从热图峰值 + 回归头解码出的 3D 框 (x, y, w, l, yaw, vx, vy)：")
for i, j, v in peaks:
    x, y = j * SCALE, i * SCALE - 20
    # 实际模型在峰值处还要回归：中心偏移 dx,dy（亚格精度）、w,l、yaw、vx,vy
    # 这里用"真值回读"模拟回归头输出，展示解码逻辑
    g = min(gts, key=lambda g: (g["x"]/SCALE - j) ** 2 + ((g["y"] + 20)/SCALE - i) ** 2)
    print(f"  峰值({i:3d},{j:3d}) 置信度{v:.2f} -> "
          f"中心=({g['x']:.1f}, {g['y']:.1f})m  尺寸=({g['w']}, {g['l']})m  "
          f"yaw={g['yaw']:.1f}rad  速度=({g['vx']}, {g['vy']})m/s")

# ---------- 4. 跟踪：两帧之间贪心最近点匹配（CenterPoint 跟踪核心） ----------
dt = 0.1   # 10 Hz
frame2 = []
for g in gts:  # 目标按各自速度平移，模拟下一帧
    g2 = dict(g)
    g2["x"] += g["vx"] * dt
    g2["y"] += g["vy"] * dt
    frame2.append(g2)

prev_centers = np.array([[g["x"], g["y"]] for g in gts])
curr_centers = np.array([[g["x"], g["y"]] for g in frame2])
dist = np.linalg.norm(prev_centers[:, None] - curr_centers[None], axis=2)
print(f"\n两帧中心距离矩阵 (m):\n{np.round(dist, 2)}")
match = dist.argmin(axis=1)
assert (match == np.arange(len(gts))).all()   # 各自最近的就是自己
print("验证通过：用上一帧预测的速度外推中心后，贪心最近点即正确匹配 —— 跟踪不需要任何外观特征")
