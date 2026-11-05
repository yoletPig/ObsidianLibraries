# exp1_iou_nms_ap.py —— 第 1 周：IoU + NMS + AP 三件套手写验证
# 复现：python3 exp1_iou_nms_ap.py（无需参数，输出与教程"真实运行输出"一致）
import numpy as np


def iou_xyxy(box_a, box_b):
    """计算两个 xyxy 框的 IoU（交并比）。"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union


def nms_xyxy(boxes, scores, iou_thr=0.5):
    """非极大值抑制：按分数从高到低保留框，删掉与已保留框 IoU 超阈值的框。"""
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in rest])
        order = rest[ious <= iou_thr]
    return keep


def demo_nms():
    # 5 个假框：0/1/3 是同一辆车的三重检测，2/4 是另外两个目标
    boxes = np.array([
        [100, 100, 210, 210],   # 0: 车 A（最高分）
        [102, 103, 214, 208],   # 1: 车 A 的重复框
        [400, 120, 500, 300],   # 2: 行人 B
        [ 98,  97, 205, 212],   # 3: 车 A 的又一个重复框
        [700, 300, 780, 380],   # 4: 骑行者 C
    ], dtype=np.float64)
    scores = np.array([0.92, 0.81, 0.77, 0.60, 0.55])

    print("两两 IoU 矩阵：")
    for i in range(5):
        row = " ".join(f"{iou_xyxy(boxes[i], boxes[j]):.2f}" for j in range(5))
        print(f"  框{i}: {row}")

    keep = nms_xyxy(boxes, scores, iou_thr=0.5)
    print("NMS 保留的框索引:", keep)
    print("保留框分数    :", scores[keep])

    assert keep == [0, 2, 4], keep
    print("验证通过：3 个目标，5 个预测框去重后剩 3 个（去掉 2 个重复框）")


def demo_ap():
    # 每行 = [置信度, 是否命中]。"是否命中"是提前用 IoU>=0.5 匹配 GT 判定好的
    # 8 个预测框：3 个真命中(TP)、5 个误报(FP)；图中共 3 个 GT
    dets = np.array([
        [0.95, 1],   # TP
        [0.90, 0],   # FP
        [0.85, 1],   # TP
        [0.80, 1],   # TP
        [0.70, 0],   # FP
        [0.60, 0],   # FP
        [0.50, 0],   # FP
        [0.30, 0],   # FP
    ])
    conf = dets[:, 0]
    tp = dets[:, 1].astype(bool)
    n_gt = 3

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(~tp)
    recall    = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum)

    print(" 置信度  TP累计 FP累计  P      R    类型")
    for c, t, f, p, r in zip(conf, tp_cum, fp_cum, precision, recall):
        print(f"  {c:.2f}    {t}      {f}     {p:.3f}  {r:.3f}   {'TP' if tp[list(conf).index(c)] else 'FP'}")

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    print(f"\n代码积分 AP = {ap:.4f}")

    ap_hand = (1/3 - 0) * 1.0 + (2/3 - 1/3) * 0.75 + (1 - 2/3) * 0.75
    print(f"手工分段积分 AP = {ap_hand:.4f}")
    assert abs(ap - ap_hand) < 1e-9
    print("验证通过：AP = 1/3×1.0 + 1/3×0.75 + 1/3×0.75 = 0.8333")


if __name__ == "__main__":
    demo_nms()
    print()
    demo_ap()
