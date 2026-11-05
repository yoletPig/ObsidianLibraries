# exp6_nds_handcalc.py —— 第 4 周：手算一个 nuScenes NDS 例子，与公式逐项核对
# 复现：python3 exp6_nds_handcalc.py（无需参数，输出与教程"真实运行输出"一致）
import numpy as np

# ---- 假设只有一个类别"car"，评测器给出四个匹配档下的 AP ----
aps = {"0.5m": 0.20, "1m": 0.40, "2m": 0.60, "4m": 0.80}
mAP = np.mean(list(aps.values()))
print(f"各中心距离阈值下的 AP: {aps}")
print(f"mAP = mean(0.2, 0.4, 0.6, 0.8) = {mAP:.4f}")

# ---- 5 类 TP 错误（各类平均后的均值），先换成得分 max(1 - err, 0) ----
errs = {"mATE": 0.30, "mASE": 0.20, "mAOE": 0.10, "mAVE": 0.20, "mAAE": 0.10}
scores = {k: max(1.0 - v, 0.0) for k, v in errs.items()}
for k in errs:
    print(f"{k}: 误差 {errs[k]:.2f} -> 得分 1-err = {scores[k]:.2f}")

# ---- NDS = (5*mAP + sum(TP scores)) / 10 ----
nds = (5 * mAP + sum(scores.values())) / 10
print(f"NDS = (5 x {mAP:.1f} + {sum(scores.values()):.2f}) / 10 = {nds:.4f}")
assert abs(nds - 0.66) < 1e-9
print("验证通过：NDS = (5x0.5 + 0.7+0.8+0.9+0.8+0.9)/10 = (2.5+4.1)/10 = 0.66")
