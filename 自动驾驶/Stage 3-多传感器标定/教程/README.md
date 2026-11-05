# Stage 3-多传感器标定 · 教程导航

> 本教程集回答一个问题：**一辆自动驾驶车上的相机、LiDAR、IMU、轮速计，它们的数据怎么才能"对得上"？**
> 对不上分两种：空间上对不上（外参），时间上对不上（时钟偏移）。四周分别解决：相机自身（内参）、相机-LiDAR（外参）、多雷达与手眼（外参进阶）、时间（时间标定与在线自标定）。

---

## 阅读顺序与依赖

| 文件 | 主题 | 前置依赖 | 核心公式 |
| --- | --- | --- | --- |
| [[第1周-相机内参标定-张正友法与畸变]] | 相机内参 K 与畸变系数 | Stage 1 相机模型 | 单应约束 $Av_1\times Av_2 = v_1\times v_2$、$A^{-\top}B A^{-1}=\lambda K^{-\top}K^{-1}$ |
| [[第2周-相机-LiDAR外参标定]] | 相机 ↔ LiDAR 外参 | 第 1 周内参 | PnP 齐次方程、Wahba 问题 SVD 解、平面偏移方程 |
| [[第3周-手眼标定与雷达间外参]] | AX=XB、LiDAR-LiDAR | 第 2 周外参概念 | $AX=XB$、$\mathrm{skew}(P_a{+}P_b)x=P_b{-}P_a$、ICP/NDT |
| [[第4周-时间标定与在线自标定]] | 时间偏移、在线自标定、验收流程 | 第 2/3 周外参 | 互相关对齐、连续时间批量优化（概念） |

先读仓库根目录的 `学习计划.md` 了解整体节奏与每周 MVP。

## 环境准备

本教程集**所有代码无 GPU 依赖**，CPU 上秒级到分钟级跑完。

```bash
# 基础三件套（大概率已有）
pip install --user numpy scipy matplotlib

# OpenCV（headless 版无 GUI 依赖，WSL/服务器友好）
pip install --user opencv-python-headless

# 环境自检
python3 -c "import numpy, scipy, matplotlib, cv2; print('cv2', cv2.__version__)"
```

教程编写时的验证环境：Python 3 + numpy/scipy/matplotlib + **OpenCV 5.0.0**。

> ⚠️ **OpenCV 版本坑**：OpenCV 4.x 提供 `cv2.calibrateHandEye`（手眼标定一键接口）；**OpenCV 5.0 已移除该 API**（枚举常量还在，函数没了）。第 3 周教程因此给出与 OpenCV 官方 C++ 实现同源算法的纯 numpy 版本，任何版本都能跑。若你的 OpenCV 是 4.x，可对照 `cv2.calibrateHandEye` 验证自己的实现。

## 教程的方法论

每篇教程遵循同一个骨架：

1. **本周要回答的问题**——先给问题再给答案；
2. **先直觉后公式**——每个公式配"这个公式在干什么"+ 逐符号物理含义 + 算出来拿来做什么；
3. **可运行代码 + 真实预期输出**——所有"预期输出"都是编写时真实运行的结果，不是编的；
4. **合成数据数值实验**——已知真值生成虚拟对应点 + 噪声，验证误差随噪声的变化。这是本教程集的灵魂：真实图片没有真值，合成数据什么都能验证；
5. **常见坑 + 带答案自测题**。

## 实验清单（每篇教程中的可运行实验）

完整脚本在 [`实验脚本/`](实验脚本/) 目录，全部在 CPU 上真实运行验证过，教程中的"预期输出"即真实输出：

| 脚本 | 位置 | 验证内容 |
| --- | --- | --- |
| `exp1_intrinsics.py` | 第 1 周 | cv2.calibrateCamera 恢复真值 K 与畸变；覆盖范围对照；噪声/张数扫描 |
| `exp2_pnp.py` | 第 2 周 | 手写 DLT+RQ vs cv2.solvePnP；线性代数解 vs 非线性 ML 解的精度差距 |
| `exp3_lidar_cam_planes.py` | 第 2 周 | SVD 解 Wahba + 平面方程解 t；LiDAR 测距噪声扫描 |
| `exp4_projection.py` | 第 2 周 | 点云投影叠加体检；yaw/平移偏差敏感度 |
| `exp5_handeye.py` | 第 3 周 | Tsai-Lenz 无噪声精确恢复；位姿组数/噪声扫描 |
| `exp6_icp.py` | 第 3 周 | 点-点 vs 点-面 ICP；初值差 90° 掉坑 |
| `exp7_timeoffset.py` | 第 4 周 | 互相关恢复 37 ms 时间偏移 + 抛物线精化的诚实局限 |

复现方式：`python3 实验脚本/expN_*.py`（无需参数，随机种子固定，输出与教程一致）。

## 与真实工程的关系

教程里的算法是**骨架**，真实工程还要长肉：亚像素角点提取、异常点剔除（RANSAC）、权重化最小二乘、多帧融合。开源工具对照表：

| 工具 | 用途 | 一句话 |
| --- | --- | --- |
| [ethz-asl/kalibr](https://github.com/ethz-asl/kalibr) | 相机-IMU 时空联合标定 | 连续时间 B 样条框架（Furgale IROS 2013）的事实标准实现 |
| [Livox-SDK/livox_camera_lidar_calibration](https://github.com/Livox-SDK/livox_camera_lidar_calibration) | Livox 官方相机-LiDAR 标定 | 基于棋盘角点 PnP + 手动选点的半自动流程 |
| [ankitdhall/lidar_camera_calibration](https://github.com/ankitdhall/lidar_camera_calibration) | 相机-LiDAR 标定（ROS） | 3D-3D 点对应（arXiv:1705.09785），社区经典 |
| [UMich-BipedLab/extrinsic_lidar_camera_calibration](https://github.com/UMich-BipedLab/extrinsic_lidar_camera_calibration) | 高精度靶标法相机-LiDAR | 对 Dhall 方法的系统性改进 |
| [hku-mars/livox_camera_calib](https://github.com/hku-mars/livox_camera_calib) | **无目标**相机-LiDAR | 图像边缘 + 点云深度/强度边缘对齐，Ceres 优化 |
| [koide3/direct_visual_lidar_calibration](https://github.com/koide3/direct_visual_lidar_calibration) | **无目标**、通用投影模型 | 单次采样自动标定（ICRA 2023） |
| PCL / Open3D / Autoware `ndt_scan_matcher` | 点云配准 | ICP 与 NDT 的工业实现 |

## 术语速查（首现均在正文给出英文）

- **内参（intrinsics）** K：焦距 $f_x, f_y$ 与主点 $c_x, c_y$，相机的"出厂属性"
- **畸变（distortion）**：径向 $k_1,k_2,k_3$ + 切向 $p_1,p_2$
- **外参（extrinsics）** T：两个坐标系之间的刚体变换 $4\times4$ 齐次矩阵
- **重投影误差（reprojection error）**：把 3D 点按估计参数投回像素，与观测像素的距离
- **PnP**（Perspective-n-Point）：3D-2D 对应求位姿
- **手眼标定（hand-eye calibration）**：$AX=XB$，求相机与机械臂末端的固定变换
- **ego 车体系（ego / vehicle frame）**：全车统一的参考系，下游融合都在这里
- **时间偏移（time offset）**：两个传感器时钟之差；**时间戳（timestamp）**：每个数据包的采样时刻标签
