# Stage 2 教程导航：从 2D 目标检测到 3D 目标检测

本目录是「自动驾驶感知」系列的第二阶段（Stage 2）教程。目标读者：以 Python 为主、有 DL 基础（懂 CNN）但没做过检测任务的工程师。学习路线偏**路侧/车端多传感器感知**的工程方向。

对应总学习计划：[[../学习计划]]

---

## 本阶段解决什么问题

Stage 1 你已经知道数据从哪来、长什么样；Stage 2 回答的是"怎么从这些数据里把车、行人、骑行者一个个框出来"。路线刻意设计成"先 2D 后 3D"：检测的一切度量衡（IoU、NMS、mAP）和一切经典结构（anchor、FPN、heatmap）都在 2D 检测里先建立，然后逐条映射到点云 3D 检测。

| 周次 | 教程 | 核心问题 | 一句话收获 |
| --- | --- | --- | --- |
| 第 1 周 | [[第1周-2D目标检测基础与YOLO演进]] | 检测和分类差在哪？两个框怎么算"重合"？模型多报了怎么办？ | IoU + NMS + mAP 三件套手写；YOLO v1→v8 演进逻辑 |
| 第 2 周 | [[第2周-点云3D检测-PointNet系与PointPillars]] | 无序的几万个点怎么变成有序的 3D 框？ | PointNet 对称函数 + PointPillars 全流程 shape 链 |
| 第 3 周 | [[第3周-CenterPoint与3D检测算法谱系]] | anchor-based 之后为什么大家都转 heatmap？四条技术路线怎么选？ | CenterPoint 拆解 + Point/Voxel/Pillar/Transformer 对比表 |
| 第 4 周 | [[第4周-3D检测评测与数据集实战]] | 论文表格里的数字怎么算出来的？KITTI 和 nuScenes 差在哪？ | KITTI/nuScenes 双评测体系 + 手写 BEV 旋转框 IoU |

**阅读顺序**：严格按周次。第 1 周的 IoU/mAP 在第 4 周评测体系里会以 3D 形态回归；第 2 周的 PointPillars shape 链是第 3 周 CenterPoint 的载体，也是 Stage 3 训练实战的直接地基。

---

## 依赖说明

本阶段所有代码只依赖 Python 科学计算栈，无 GPU 要求（所有示例在 CPU、无 torch 依赖下可跑）：

```bash
# 核心依赖（大概率已有）
pip install numpy matplotlib

# 可选：第 3 周对照演示用（教程给 numpy 手写版，装不上不影响主线）
pip install --user torch --index-url https://download.pytorch.org/whl/cpu

# 第 5 周机动用：跑预训练 PointPillars 推理（较重，可选）
# git clone https://github.com/open-mmlab/OpenPCDet
```

各教程代码均为完整可运行脚本，复制到本地即可执行；文中的"预期输出"均经过真实运行验证（Python 3.12 + numpy 1.26，seed 固定）。验证环境差异可能导致浮点显示末位略有不同，不影响结论。

## 实验清单（每篇教程中的可运行实验）

完整脚本在 [`实验脚本/`](实验脚本/) 目录，全部在 CPU 上真实运行验证过（numpy 1.26，随机种子固定），教程中的"预期输出"即真实输出：

| 脚本 | 位置 | 验证内容 |
| --- | --- | --- |
| `exp1_iou_nms_ap.py` | 第 1 周 | 手写 IoU 矩阵 + NMS 对 5 个假框去重（保留 {0,2,4}）；8 预测/3 GT 手算 AP = 0.8333 |
| `exp2_pointnet_invariance.py` | 第 2 周 | PointNet max pooling 置换不变性（随机置换/逆序差异 < 1e-12）；"取第一个点"对照失效 |
| `exp3_pillars_shape_chain.py` | 第 2 周 | PointPillars shape 链：`(1000,4) → (961,16,9) → (961,8) → (8,102,102)` |
| `exp4_centerpoint_heatmap.py` | 第 3 周 | 高斯热图生成 + 局部峰值解码 2 个 7-DoF 框 + 速度外推贪心匹配 |
| `exp5_bev_rotated_iou.py` | 第 4 周 | BEV 旋转框 IoU：重合=1、轴对齐 2/14、旋转 90° 4/12、蒙特卡洛差异 < 0.01 |
| `exp6_nds_handcalc.py` | 第 4 周 | NDS 手算例子：mAP=0.5 + 五项 TP 得分 4.1 → NDS = 0.66 |

复现方式：`python3 实验脚本/expN_*.py`（无需参数，随机种子固定，输出与教程一致）。

## 数据集准备（第 4 周起）

教程示例默认用**合成数据**（NumPy 模拟框与点云），零下载即可跑通。想看真实评测流程时：

1. **KITTI**：从 [cvlibs.net](https://www.cvlibs.net/datasets/kitti/) 下载 object detection 子集（约 12GB：7481 训练 / 7518 测试帧）。第 4 周用它理解评测协议。
2. **nuScenes**：下载 [mini split](https://www.nuscenes.org/nuscenes)（约 4GB），用官方 devkit 跑一次 mini 集评测，对照第 4 周的 NDS 公式逐项核对。
3. **Waymo**：只需浏览 [waymo.com/open](https://waymo.com/open/about) 的数据统计页建立量级概念，不必下载（v1.0 起约 2000 个 20 s 片段，TFRecord 格式较重）。

## 约定

- 术语首现给英文原名，如：交并比（Intersection over Union, IoU）。
- 每个公式配"这个公式在干什么"的大白话解读；代码示例附真实预期输出。
- 坐标系沿用 Stage 1：激光雷达系 x 前 / y 左 / z 上；2D 框用 xyxy（左上、右下）格式；3D 框用 7 自由度 (x, y, z, w, l, h, yaw)。
- "检测器输出几维"这类问题一律先问坐标系再谈数字——KITTI 与 nuScenes 的朝向角约定在第 4 周单独说明。
