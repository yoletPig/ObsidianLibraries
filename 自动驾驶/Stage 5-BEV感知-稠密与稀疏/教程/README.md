# Stage 5 教程：BEV 感知——稠密与稀疏

本目录是 **Stage 5-BEV感知（稠密与稀疏）** 的配套教程，共 4 周。读者应已学完 Stage 1-4（传感器、2D/3D 检测、标定、融合），本教程会直接引用其中的概念（如 PointPillars 的 pillar、内参 K 与外参 T）而不重复展开。

## 学习地图

| 周次 | 文件 | 主线问题 | 核心公式/机制 | 可运行 Demo |
| --- | --- | --- | --- | --- |
| 第 1 周 | `第1周-为什么需要BEV与坐标变换基础.md` | 透视图病在哪？BEV 好在哪？2D→BEV 卡在哪一步？ | 反投影公式（深度 = 自由度）、视锥（frustum） | `w1_bev.py` |
| 第 2 周 | `第2周-稠密BEV-LSS与BEVDet.md` | 图像特征怎么"撒"进 BEV 网格？ | LSS：深度 softmax + 外积 + pillar pool | `w2_lss.py` |
| 第 3 周 | `第3周-稠密BEV-Transformer路线BEVFormer.md` | 不撒满、改成"格子主动问"，怎么问？ | Spatial Cross-Attention、时序 ego-motion warp | `w3_bevformer.py` |
| 第 4 周 | `第4周-稀疏BEV与流式时序.md` | 干脆不建 BEV 网格，只看目标，行不行？ | DETR3D 投影采样、匈牙利匹配、流式 query 传播 | `w4_sparse.py` |

两条路线一张图：

```
                    ┌─ 稠密：先建满 BEV 网格，再接检测/分割头
                    │   ├─ LSS 系：像素→深度分布→外积→pillar pool（第2周）
                    │   └─ Transformer 系：BEV query 反向查图像（第3周）
多相机图像 ─→ 视角转换 ┤
                    └─ 稀疏：跳过 BEV 网格，少量 3D query 投影采样（第4周）
                        ├─ DETR3D（单帧）
                        ├─ Sparse4D v1→v3（4D 关键点→循环时序→端到端跟踪）
                        └─ StreamPETR（流式 query 传播）
```

## 环境依赖

- Python 3.10+，`numpy`、`scipy`、`matplotlib`（本教程所有 demo 在 CPU 上可完整复现，无需 GPU）。
- 所有 demo 为**机制演示**（缩小版，固定种子）：用 NumPy 手工复现核心公式，**不是**真实模型权重的前向推理。真实模型训练/推理需要 GPU + mmdet3d 系环境，超出本教程范围。
- Demo 脚本统一放在 `/tmp/ad_s5/` 下运行（临时目录，代码内嵌在教程中，可自行复制重建）。

## 事实查证说明

教程中所有论文信息（发表时间、venue、arXiv 编号、nuScenes 成绩、开源仓库）均经查证，出处随文标注。nuScenes 成绩一律为**论文/官方仓库自报**，注意 test set 与 val set 的区别。一处版本变更需要注意：Sparse4D 官方仓库原在 `megvii-research/Sparse4D`，现已迁移至 `HorizonRobotics/Sparse4D`（旧地址 404）。

## 建议学法

每周先读"本周要回答的问题"，跑通 demo 再读公式；手算例（第 2 周 2 像素×3 深度档）务必自己用计算器核一遍。学完后用学习计划里的"阶段终极自测清单"验收，然后进入 Stage 6（V2X）。
