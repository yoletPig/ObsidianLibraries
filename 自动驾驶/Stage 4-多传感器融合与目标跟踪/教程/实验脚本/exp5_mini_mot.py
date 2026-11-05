# exp5_mini_mot.py —— 第4周：KF + 匈牙利 + 轨迹状态机的 mini-MOT（固定种子）
# 合成场景：3 个目标交叉运动 + 10% 漏检 + 均匀杂波，跑通 tracking-by-detection 全链路
# 用法：python3 exp5_mini_mot.py [young_gate]
#   young_gate = on/off，对照"年轻轨迹欧氏门控"这一条工程修正的影响
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

np.random.seed(2024)
USE_YOUNG_GATE = (len(sys.argv) < 2) or (sys.argv[1] != "off")

# ================= 场景与检测生成 =================
DT, N = 0.1, 100                       # 10 Hz，10 秒
SIG_Z = 0.5                            # 检测位置噪声 0.5 m
P_MISS = 0.10                          # 漏检率
CLUTTER_LAM = 2.0                      # 每帧平均 2 个杂波点
AREA = 50.0                            # 场景 [-50, 50]^2
GATE = 9.21                            # chi2(2, 0.99) 门控
MAX_AGE = 10                           # 连续 miss 10 帧删除
MIN_HITS = 3                           # tentative -> confirmed 需连续/累计 3 次命中

# 3 个目标：T0 向右、T1 向下（与 T0 空间交叉）、T2 斜线（更晚穿过同一区域）
gt_state = {                           # [x, y, vx, vy]
    0: np.array([-30.0, 0.0, 10.0, 0.0]),
    1: np.array([10.0, 30.0, 0.0, -10.0]),
    2: np.array([-40.0, -40.0, 5.0, 5.0]),
}

def gt_pos(k, i):
    s = gt_state[i]
    return s[:2] + s[2:] * DT * k

# 生成检测：列表 [(frame, pos)]
dets_by_frame = [[] for _ in range(N)]
for k in range(N):
    for i in gt_state:
        if np.random.rand() > P_MISS:                       # 90% 检出
            z = gt_pos(k, i) + np.random.randn(2) * SIG_Z
            dets_by_frame[k].append(z)
    n_clutter = np.random.poisson(CLUTTER_LAM)
    for _ in range(n_clutter):
        dets_by_frame[k].append(np.random.uniform(-AREA, AREA, 2))

# ================= 卡尔曼滤波（CV 模型）=================
F = np.array([[1, 0, DT, 0], [0, 1, 0, DT],
              [0, 0, 1, 0], [0, 0, 0, 1]])
Q = np.diag([0.01, 0.01, 0.25, 0.25]) * DT
H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
R = np.eye(2) * SIG_Z ** 2

class Track:
    _next = 1
    def __init__(self, z):
        self.id = Track._next; Track._next += 1
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([SIG_Z**2, SIG_Z**2, 25.0, 25.0])
        self.hits, self.misses = 1, 0
        self.confirmed = False
    def predict(self):
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
    def update(self, z):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.hits += 1; self.misses = 0
        if self.hits >= MIN_HITS:
            self.confirmed = True
    def mahalanobis2(self, z):
        S = H @ self.P @ H.T + R
        d = z - H @ self.x
        return d @ np.linalg.inv(S) @ d

tracks, output = [], []               # output: (frame, track_id, pos)
for k in range(N):
    for tr in tracks:
        tr.predict()
    # ---- 关联：马氏门控 + 匈牙利 ----
    BIG = 1e6
    cost = np.full((len(tracks), len(dets_by_frame[k])), BIG)
    YOUNG_GATE_M = 3.0        # 年轻轨迹（未确认）额外欧氏门控：防不确定度爆炸
    for i, tr in enumerate(tracks):
        for j, z in enumerate(dets_by_frame[k]):
            d2 = tr.mahalanobis2(z)
            if d2 <= GATE:
                if (USE_YOUNG_GATE and tr.hits < MIN_HITS
                        and np.linalg.norm(z - tr.x[:2]) > YOUNG_GATE_M):
                    continue          # 年轻轨迹只信 3m 内的观测，防止"速度乱飞、门控罩全场"
                cost[i, j] = d2
    if cost.size:
        ri, ci = linear_sum_assignment(cost)
        matched_t, matched_d = set(), set()
        for i, j in zip(ri, ci):
            if cost[i, j] < BIG:
                tracks[i].update(dets_by_frame[k][j])
                matched_t.add(i); matched_d.add(j)
    else:
        matched_t, matched_d = set(), set()
    # ---- 未匹配轨迹：miss 计数 ----
    alive = []
    for i, tr in enumerate(tracks):
        if i not in matched_t:
            tr.misses += 1
        if tr.misses <= MAX_AGE:
            alive.append(tr)
        # ---- 未匹配检测：新 tentative ----
    for j, z in enumerate(dets_by_frame[k]):
        if j not in matched_d:
            alive.append(Track(z))
    tracks = alive
    # ---- 输出 confirmed 轨迹 ----
    for tr in tracks:
        if tr.confirmed:
            output.append((k, tr.id, tr.x[:2].copy()))

# ================= 评估：MOTA / MOTP / IDSW =================
EVAL_GATE = 2.0
gt2track = {i: None for i in gt_state}    # GT 身份 -> 上一次匹配的 track id
FN = FP = IDSW = 0; dists = []
for k in range(N):
    hyp = [(tid, pos) for (f, tid, pos) in output if f == k]
    gts = [(i, gt_pos(k, i)) for i in gt_state]
    n = len(gts); m = len(hyp)
    cost = np.full((n, m), 1e6)
    for a, (gi, gp) in enumerate(gts):
        for b, (tid, hp) in enumerate(hyp):
            d = np.linalg.norm(gp - hp)
            if d <= EVAL_GATE:
                cost[a, b] = d
    ri, ci = linear_sum_assignment(cost)
    matched_g, matched_h = set(), set()
    for a, b in zip(ri, ci):
        if cost[a, b] < 1e6:
            gi = gts[a][0]; tid = hyp[b][0]
            dists.append(cost[a, b])
            matched_g.add(a); matched_h.add(b)
            if gt2track[gi] is not None and gt2track[gi] != tid:
                IDSW += 1
            gt2track[gi] = tid
    FN += n - len(matched_g)              # GT 没被任何 confirmed 轨迹覆盖
    FP += m - len(matched_h)              # confirmed 轨迹没对应任何 GT
total_gt = N * len(gt_state)
MOTA = 1 - (FN + FP + IDSW) / total_gt
MOTP = np.mean(dists)

print("==== mini-MOT 运行结果 ====")
print(f"轨迹池累计创建 Track 数 = {Track._next - 1}（含杂波产生的短命 tentative）")
n_conf = len({tid for _, tid, _ in output})
print(f"成为 confirmed 并有输出的 Track 数 = {n_conf}")
# 判定哪些 confirmed 轨迹是真目标、哪些是杂波幸存者
for tid in sorted({tid for _, tid, _ in output}):
    pts = np.array([pos for f, t, pos in output if t == tid])
    # 找该轨迹全程与哪个 GT 最近
    d_best, gi_best = 1e9, None
    for f, t, pos in output:
        if t != tid: continue
        for i in gt_state:
            d = np.linalg.norm(pos - gt_pos(f, i))
            if d < d_best: d_best, gi_best = d, i
    kind = f"真目标 GT{gi_best}（全程最近距离 {d_best:.2f} m）" if d_best <= 2.0 else "杂波/虚假轨迹"
    print(f"  Track {tid}: 输出 {len(pts):4d} 帧,  判定 = {kind}")
print(f"\nFN = {FN}, FP = {FP}, IDSW = {IDSW}, GT 总帧数 = {total_gt}")
print(f"MOTA = {MOTA:.4f}")
print(f"MOTP = {MOTP:.3f} m（匹配对平均位置误差）")
