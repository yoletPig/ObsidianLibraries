# exp5_bev_rotated_iou.py —— 第 4 周：BEV 旋转框 IoU（角点展开 + Sutherland–Hodgman 裁剪 + 鞋带公式）
# 复现：python3 exp5_bev_rotated_iou.py（无需参数，输出与教程"真实运行输出"一致）
import numpy as np


def box2corners(x, y, w, l, yaw):
    """BEV 旋转框 (x,y,w,l,yaw) -> 4 个角点，逆时针顺序，shape (4,2)。
    约定：l 沿朝向 yaw 方向，w 垂直朝向。"""
    # 本体坐标系（车头朝 +x_local）：沿 l 的半长、沿 w 的半宽，按逆时针排列
    local = np.array([
        [ l/2, -w/2],
        [ l/2,  w/2],
        [-l/2,  w/2],
        [-l/2, -w/2],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([x, y])


def poly_area(poly):
    """鞋带公式（shoelace）计算多边形面积，输入 (K,2)。"""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip_convex(subject, clip):
    """Sutherland–Hodgman：用凸多边形 clip 裁剪 subject，返回裁剪后顶点。
    两个多边形都须是逆时针。"""
    output = subject
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        # 边 a->b 的"内侧" = 叉积 >= 0（左侧）
        edge = b - a
        inputs = output
        output = []
        if len(inputs) == 0:
            break

        def inside(p):
            return np.cross(edge, p - a) >= -1e-12

        p_prev = inputs[-1]
        prev_in = inside(p_prev)
        for p in inputs:
            cur_in = inside(p)
            if cur_in != prev_in:
                # 线段 p_prev->p 与直线 a->b 求交
                t = np.cross(a - p_prev, edge) / np.cross(p - p_prev, edge)
                output.append(p_prev + t * (p - p_prev))
            if cur_in:
                output.append(p)
            p_prev, prev_in = p, cur_in
    return np.array(output)


def bev_rotated_iou(box_a, box_b):
    """两个 BEV 旋转框的 IoU。输入 (x, y, w, l, yaw)。"""
    ca = box2corners(*box_a)
    cb = box2corners(*box_b)
    area_a = poly_area(ca)
    area_b = poly_area(cb)
    inter_poly = clip_convex(ca, cb) if area_a > 0 and area_b > 0 else []
    inter = poly_area(inter_poly) if len(inter_poly) >= 3 else 0.0
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def point_in_box(p, box):
    """蒙特卡洛用：判断点是否在旋转框内（转到本体坐标判 |lx|,|ly|）。"""
    x, y, w, l, yaw = box
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = p[:, 0] - x, p[:, 1] - y
    lx = c * dx + s * dy          # R^T @ d 的第一行
    ly = -s * dx + c * dy
    return (np.abs(lx) <= l/2) & (np.abs(ly) <= w/2)


def iou_monte_carlo(box_a, box_b, n=200_000, rng=None):
    """蒙特卡洛撒点估计 IoU（用于交叉验证解析值）。"""
    rng = rng or np.random.default_rng(7)
    corners = np.vstack([box2corners(*box_a), box2corners(*box_b)])
    lo, hi = corners.min(0) - 0.5, corners.max(0) + 0.5
    pts = rng.uniform(lo, hi, size=(n, 2))
    ina = point_in_box(pts, box_a)
    inb = point_in_box(pts, box_b)
    area_rect = np.prod(hi - lo)
    inter = (ina & inb).mean() * area_rect
    union = (ina | inb).mean() * area_rect
    return inter / union


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    # ---- 测试1：完全重合 -> 1.0 ----
    b = (10.0, 5.0, 2.0, 4.5, 0.3)
    print(f"测试1 完全重合      IoU = {bev_rotated_iou(b, b):.6f}")
    assert abs(bev_rotated_iou(b, b) - 1.0) < 1e-9

    # ---- 测试2：轴对齐偏移，手算 2/14 ----
    # A=[0,0,4,2] 面积8, B=[3,0,4,2] 偏移后交集 1x2=2, 并集 8+8-2=14
    a2, b2 = (0, 0, 2, 4, 0.0), (3, 0, 2, 4, 0.0)
    i2 = bev_rotated_iou(a2, b2)
    print(f"测试2 轴对齐偏移    IoU = {i2:.6f}（手算 2/14 = {2/14:.6f}）")
    assert abs(i2 - 2/14) < 1e-9

    # ---- 测试3：同一中心、旋转 90°，手算 4/12 ----
    # A: w=2,l=4 面积8；B 旋转90°后 w=4,l=2 面积8；交集 = 中央 2x2 = 4；并集 12
    a3 = (0, 0, 2, 4, 0.0)
    b3 = (0, 0, 2, 4, np.pi / 2)
    i3 = bev_rotated_iou(a3, b3)
    print(f"测试3 旋转90°      IoU = {i3:.6f}（手算 4/12 = {4/12:.6f}）")
    assert abs(i3 - 4/12) < 1e-9

    # ---- 测试4：任意旋转 vs 20万点蒙特卡洛 ----
    a4 = (2.0, -1.0, 2.0, 4.5, 0.7)
    b4 = (2.5, -0.5, 1.8, 4.2, -0.4)
    i4 = bev_rotated_iou(a4, b4)
    i4_mc = iou_monte_carlo(a4, b4, n=200_000)
    print(f"测试4 任意旋转     IoU 解析 = {i4:.6f}, 蒙特卡洛 = {i4_mc:.6f}, 差异 = {abs(i4-i4_mc):.4f}")
    assert abs(i4 - i4_mc) < 0.01

    # ---- 测试5：不相交 -> 0 ----
    i5 = bev_rotated_iou((0, 0, 2, 4, 0.3), (50, 50, 2, 4, 1.0))
    print(f"测试5 完全分离      IoU = {i5:.6f}")
    assert i5 == 0.0
    print("验证通过：BEV 旋转框 IoU 与手算/蒙特卡洛全部一致")
