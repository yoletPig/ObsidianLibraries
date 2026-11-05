import numpy as np, cv2
rng = np.random.default_rng(3)

W, H = 1280, 720
K = np.array([[800.0, 0, 640.0], [0, 800.0, 360.0], [0, 0, 1.0]])

def rodrigues(rx, ry, rz):
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
    return R

def project(P_lidar, R, t):
    """LiDAR 点 -> 相机系 -> 像素。P_cam = R @ P_lidar + t"""
    Xc = (R @ P_lidar.T).T + t
    z = Xc[:, 2]
    uv = (K @ Xc.T).T
    return uv[:, :2] / z[:, None], z

# ---- 构造虚拟场景（都在 LiDAR 坐标系中）----
# 真值外参：LiDAR -> 相机
R_true = rodrigues(np.radians(2), np.radians(-1), np.radians(8))
t_true = np.array([0.05, -0.02, 0.10])

# 标定板在相机系中正对相机（这是"检测"得到的已知量）
R_b = rodrigues(0.0, 0.0, 0.0)
t_b = np.array([0.0, 0.0, 3.0])
BW, BH = 0.8, 0.6
# 板四角（板坐标系） -> LiDAR 系（模拟"点云里看到的板"）
corners_b = np.array([[[-BW/2, -BH/2, 0], [BW/2, -BH/2, 0],
                       [BW/2, BH/2, 0], [-BW/2, BH/2, 0]]])
R_l = R_true.T @ R_b
t_l = R_true.T @ (t_b - t_true)
corners_lidar = (R_l @ corners_b[0].T).T + t_l

# LiDAR 在板面上采样 600 个点（加 5mm 噪声）
u = rng.uniform(-BW/2, BW/2, 600); v = rng.uniform(-BH/2, BH/2, 600)
board_pts_lidar = (R_l @ np.column_stack([u, v, np.zeros(600)]).T).T + t_l
board_pts_lidar += rng.normal(0, 0.005, board_pts_lidar.shape)

# 背景点云：更远的一面墙（不会落在板内）
uw = rng.uniform(-4, 4, 1000); vw = rng.uniform(-2, 2, 1000)
wall_lidar = (R_l @ np.column_stack([uw, vw, np.full(1000, -3.0)]).T).T \
             + t_l + np.array([0, 0, 3.0])

def board_mask_in_image(uv):
    """判断像素是否落在板四角构成的四边形内（cv2.pointPolygonTest）。"""
    uv_pix, _ = cv2.projectPoints(corners_b, cv2.Rodrigues(R_b)[0], t_b, K, None)
    poly = uv_pix.reshape(-1, 2).astype(np.float32)
    inside = np.zeros(len(uv), bool)
    for i, (x, y) in enumerate(uv):
        inside[i] = cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
    return inside

def evaluate(R, t, label):
    uv, z = project(board_pts_lidar, R, t)
    ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    inside = np.zeros(len(uv), bool)
    uv_in = uv[ok]
    uv_pix, _ = cv2.projectPoints(corners_b, cv2.Rodrigues(R_b)[0], t_b, K, None)
    poly = uv_pix.reshape(-1, 2).astype(np.float32)
    for i, (x, y) in enumerate(uv_in):
        inside[np.where(ok)[0][i]] = cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
    # 背景墙误入板内的比例（假阳性）
    uvw, zw = project(wall_lidar, R, t)
    okw = (zw > 0.1) & (uvw[:, 0] >= 0) & (uvw[:, 0] < W) & (uvw[:, 1] >= 0) & (uvw[:, 1] < H)
    inside_w = np.zeros(len(uvw), bool)
    for i, (x, y) in enumerate(uvw[okw]):
        inside_w[np.where(okw)[0][i]] = cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
    print(f"  {label:<28s} 板上点落入板内: {inside.sum():4d}/600 ({inside.mean()*100:5.1f}%)   "
          f"背景墙点误入板内: {inside_w.sum():3d}/1000 ({inside_w.mean()*100:4.1f}%)")

print("把 LiDAR 点云投影到虚拟图像，用'板点是否落在板内'做外参质量体检：")
evaluate(R_true, t_true, "真值外参")
for dyaw in [0.5, 1.0, 2.0]:
    evaluate(R_true @ rodrigues(0, np.radians(dyaw), 0), t_true,
             f"真值 + yaw 偏 {dyaw:.1f} deg")
evaluate(R_true, t_true + np.array([0.0, 0.10, 0.0]), "真值 + y 平移偏 10 cm")

# ---- 渲染一张真正的叠加图（保存 PNG，肉眼验收用）----
img = np.zeros((H, W, 3), np.uint8)
img[:] = (40, 40, 40)
uv_pix, _ = cv2.projectPoints(corners_b, cv2.Rodrigues(R_b)[0], t_b, K, None)
poly = uv_pix.reshape(-1, 2).astype(np.int32)
cv2.polylines(img, [poly], True, (255, 255, 255), 2)          # 画板框
uv, z = project(board_pts_lidar, R_true, t_true)
ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
for x, y in uv[ok].astype(int):
    cv2.circle(img, (x, y), 2, (0, 255, 0), -1)               # 点云涂绿
uv_bad, z_bad = project(board_pts_lidar, R_true @ rodrigues(0, np.radians(2), 0), t_true)
okb = (z_bad > 0.1) & (uv_bad[:, 0] >= 0) & (uv_bad[:, 0] < W) & (uv_bad[:, 1] >= 0) & (uv_bad[:, 1] < H)
for x, y in uv_bad[okb].astype(int):
    cv2.circle(img, (x, y), 2, (0, 0, 255), -1)               # 错外参涂红
import os
cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), "projection_overlay.png"), img)
print("\n叠加图已保存到脚本目录 projection_overlay.png（绿=真值外参投影，红=yaw 偏 2° 的投影）")
