# exp4_hungarian.py —— 第3周：匈牙利算法 vs 贪心、马氏距离门控（固定种子）
import numpy as np
from scipy.optimize import linear_sum_assignment

np.random.seed(3)

# ---------- 实验1：3x3 交叉场景的全局最优指派 ----------
# 3 条轨迹的预测位置 vs 本帧 3 个检测，代价 = 欧氏距离（米）
tracks = np.array([[50.0, 100.0],      # track A 预测在 (50,100)
                   [60.0, 102.0],      # track B
                   [52.0, 105.0]])     # track C
dets = np.array([[51.5, 101.0],        # det 0
                 [61.0, 103.0],        # det 1
                 [49.0, 99.0]])        # det 2
cost = np.linalg.norm(tracks[:, None, :] - dets[None, :, :], axis=2)

print("==== 代价矩阵（行=轨迹，列=检测，单位 m）====")
print(np.array2string(cost, precision=2, suppress_small=True))
r, c = linear_sum_assignment(cost)
total = cost[r, c].sum()
print("\n匈牙利算法全局最优指派：")
for i, j in zip(r, c):
    print(f"  track {i}  <->  det {j}   (代价 {cost[i, j]:.2f} m)")
print(f"总代价 = {total:.2f} m")

# ---------- 实验2：贪心反例（经典代价矩阵）----------
print("\n==== 贪心 vs 全局最优 反例 ====")
cost2 = np.array([[1.0, 2.0, 5.0],
                  [2.0, 4.0, 5.0],
                  [5.0, 5.0, 5.0]])
print("代价矩阵 =")
print(cost2)
# 贪心：每一步取当前全局最小且行列未占用的格子
greedy = []
rows, cols = set(), set()
c2 = cost2.copy()
while len(greedy) < 3:
    i, j = np.unravel_index(np.argmin(c2), c2.shape)
    greedy.append((int(i), int(j), c2[i, j]))
    rows.add(i); cols.add(j)
    c2[sorted(rows), :] = np.inf
    c2[:, sorted(cols)] = np.inf
gt = sum(v for _, _, v in greedy)
r2, cidx = linear_sum_assignment(cost2)
opt = cost2[r2, cidx].sum()
print(f"贪心指派 {[(i, j) for i, j, _ in greedy]}  总代价 = {gt:.0f}")
print(f"匈牙利指派 {list(zip(r2, cidx))}  总代价 = {opt:.0f}")
print(f"贪心比最优差 {gt - opt:.0f}")

# ---------- 实验3：马氏距离门控 ----------
print("\n==== 马氏距离门控（chi2, 2自由度, 95% = 5.991）====")
# 某轨迹预测位置 (50, 100)，位置协方差（预测后）P
P = np.diag([4.0, 1.0])               # x 方向 2σ=4m，y 方向 2σ=1m（沿路方向不确定度大）
x_pred = np.array([50.0, 100.0])
S_inv = np.linalg.inv(P)              # 残差协方差 S = P + R，教学版先忽略 R
dets3 = np.array([[52.0, 100.5],      # 合理检测
                  [58.0, 101.0],      # 偏远，但 x 方向门控可能放行？
                  [51.0, 110.0]])     # 横向偏 10 m，明显是杂波
GATE = 5.991
for d in dets3:
    diff = d - x_pred
    d2 = diff @ S_inv @ diff
    eu = np.linalg.norm(diff)
    verdict = "通过" if d2 <= GATE else "拒绝"
    print(f"det {d}: 欧氏距离 = {eu:5.2f} m, 马氏距离^2 = {d2:7.2f} -> {verdict}")

# ---------- 实验4：全帧门控+匈牙利完整关联 ----------
print("\n==== 门控 + 匈牙利 组合流程 ====")
tracks4 = np.array([[50.0, 100.0], [60.0, 102.0]])
P4 = [np.diag([4.0, 1.0]), np.diag([4.0, 1.0])]
dets4 = np.array([[51.5, 101.0], [61.0, 103.0], [90.0, 90.0]])   # 第3个是远处杂波
BIG = 1e6
cost4 = np.zeros((2, 3))
for i in range(2):
    Si = np.linalg.inv(P4[i])
    for j in range(3):
        diff = dets4[j] - tracks4[i]
        d2 = diff @ Si @ diff
        cost4[i, j] = d2 if d2 <= GATE else BIG        # 超门控 -> 代价置无穷
print("门控后代价矩阵（1e6 = 被门控剪掉）:")
print(cost4)
r4, c4_ = linear_sum_assignment(cost4)
for i, j in zip(r4, c4_):
    if cost4[i, j] >= BIG:
        print(f"track {i}: 无通过门控的检测 -> 本帧未匹配（进入 miss 计数）")
    else:
        print(f"track {i} <-> det {j}  (马氏^2 = {cost4[i, j]:.2f})")
matched_cols = set(c4_[cost4[r4, c4_] < BIG])
for j in range(3):
    if j not in matched_cols:
        print(f"det {j}: 未被任何轨迹认领 -> 候选杂波或新目标")
