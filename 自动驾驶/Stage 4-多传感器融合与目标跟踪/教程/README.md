# Stage 4-多传感器融合与目标跟踪 · 教程导航

> 本教程集回答一个问题：**相机、激光雷达、毫米波雷达各自吐出的"目标"，怎么合成一个可信的目标？同一个真实目标跨帧出现几百次，怎么始终认得出"还是它"？**
> 四周分别解决：合成的层级选择（前融合/后融合）、单个目标怎么"跟得住"（卡尔曼滤波家族）、多个目标怎么"分得清"（数据关联与匈牙利算法）、整套系统怎么拼起来并量化好坏（tracking-by-detection 与 MOT 指标）。

---

## 阅读顺序与依赖

| 文件 | 主题 | 前置依赖 | 核心公式 |
| --- | --- | --- | --- |
| [[第1周-融合层级与前后融合]] | 融合三层级、五大难点、路侧 vs 车端 | Stage 1 传感器特性、Stage 3 时间标定 | 逆方差加权 $x_f = \sum w_i x_i / \sum w_i,\ w_i = 1/\sigma_i^2$ |
| [[第2周-卡尔曼滤波与EKF-UKF]] | KF/EKF/UKF、Q/R 调参、不确定度椭圆 | 第 1 周的时延/误差概念 | 预测 $\hat x^-=F\hat x,\ P^-=FPF^\top+Q$；更新（K、y、P） |
| [[第3周-数据关联与匈牙利算法]] | 门控、匈牙利、贪心反例、新生/消亡 | 第 2 周协方差（马氏距离要用） | $d_M^2 = y^\top S^{-1} y$，指派总代价最小 |
| [[第4周-多目标跟踪与轨迹管理]] | tracking-by-detection 全流程、MOTA/MOTP/IDF1 | 第 2/3 周（直接拼装） | $MOTA = 1 - (FN+FP+IDSW)/GT$ |

先读仓库根目录的 `学习计划.md` 了解整体节奏与每周 MVP。

## 环境准备

本教程集**所有代码无 GPU 依赖**，CPU 上秒级跑完，无需任何深度学习框架——融合与跟踪的核心是几何、概率与优化，不是网络。

```bash
# 基础三件套（大概率已有）
pip install --user numpy scipy matplotlib

# 环境自检
python3 -c "import numpy, scipy, matplotlib; from scipy.optimize import linear_sum_assignment; print('ok')"
```

教程编写时的验证环境：Python 3 + numpy 1.26.4 + scipy 1.15.3 + matplotlib 3.10.9。所有实验固定随机种子，输出与正文完全一致。

## 教程的方法论

每篇教程遵循同一个骨架（与本库 Stage 1/2/3 一致）：

1. **本周要回答的问题**——先给问题再给答案；
2. **先直觉后公式**——每个公式配"这个公式在干什么"+ 逐符号物理含义 + 算出来拿来做什么；
3. **可运行代码 + 真实预期输出**——所有"预期输出"都是编写时真实运行的结果，不是编的；
4. **合成数据数值实验**——已知真值生成虚拟观测 + 噪声，误差和指标都能精确量化。这是本教程集的灵魂：真实路口数据没有真值，合成数据什么都能验证；
5. **常见坑 + 带答案自测题**。

## 实验清单（每篇教程中的可运行实验）

完整脚本在 [`实验脚本/`](实验脚本/) 目录，全部在 CPU 上真实运行验证过，教程中的"预期输出"即真实输出：

| 脚本 | 位置 | 验证内容 |
| --- | --- | --- |
| `exp1_fusion_3sensor.py` | 第 1 周 | 三传感器 2D 位置观测：单传感器 vs 等权平均 vs 逆方差加权；雷达 100 ms 时延外推补偿 |
| `exp2_kalman_qr.py` | 第 2 周 | CV 模型 KF；固定 Q 调 R、固定 R 调 Q 两张对照表（响应速度与滞后） |
| `exp3_ekf_ukf.py` | 第 2 周 | 极坐标雷达观测：冻结线性化 KF vs EKF vs UKF；协方差椭圆收缩图（保存 `cov_ellipse.png`） |
| `exp4_hungarian.py` | 第 3 周 | 马氏门控 + `linear_sum_assignment` 完整关联；贪心 vs 全局最优反例 |
| `exp5_mini_mot.py` | 第 4 周 | mini-MOT：3 目标交叉 + 漏检 + 杂波，KF+匈牙利+状态机全链路，输出轨迹与 MOTA（`on`/`off` 两模式对照） |

复现方式：`python3 实验脚本/expN_*.py`（无需参数，随机种子固定，输出与教程一致；`exp5` 可加 `off` 参数复现失效模式）。

## 与真实工程的关系

教程里的算法是**骨架**，真实量产系统还要长肉：类别概率融合、外观特征（ReID）、轨迹级生命周期管理、在线噪声估计。开源对照：

| 项目 | 用途 | 一句话 |
| --- | --- | --- |
| [abewley/sort](https://github.com/abewley/sort) | SORT 官方实现 | KF + 匈牙利 + IoU 匹配，2D MOT，260+ Hz（arXiv:1602.00763） |
| [nwojke/deep_sort](https://github.com/nwojke/deep_sort) | DeepSORT 官方实现 | SORT + 外观特征关联（arXiv:1703.07402），ID switch 大幅下降 |
| [cheind/py-motmetrics](https://github.com/cheind/py-motmetrics) | MOT 指标库 | CLEAR MOT / Identity / HOTA 指标，结果与 MOTChallenge 对齐 |
| [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack) | 高分检测器时代的 SOTA | 低分检测框也参与二轮关联，MOTA 上台阶（ECCV 2022） |

## 与本库其他 Stage 的衔接

- **向下游承接 Stage 1/2/3**：第 1 周默认你已知道三类传感器各自的误差特性（Stage 1）与"检测框/3D 框怎么来"（Stage 2）、以及"不同传感器数据为何对不齐、怎么对齐"（Stage 3 时间/空间标定）。
- **向前衔接 Stage 5**：前融合（BEV 特征级融合）是当前量产主线，本阶段只讲"它在哪一层、为什么强"，网络结构细节留给 Stage 5。
- **路侧读者**：第 1 周路侧 vs 车端一节与第 4 周 V2X TrackID 一节专门为路侧感知场景写，与本库 `后融合.md` 笔记一一对应。

## 术语速查（首现均在正文给出英文）

- **前融合（early fusion）/ 数据级融合（data-level fusion）**：原始数据层面的融合
- **特征级融合（feature-level fusion）**：BEV 特征层面的融合，量产主流
- **后融合（late fusion）/ 目标级融合（object-level fusion）**：各传感器独立出目标再融合
- **卡尔曼滤波（Kalman Filter, KF）**：高斯噪声下线性系统的最优递归估计器
- **扩展/无迹卡尔曼滤波（EKF / UKF）**：非线性系统的两种近似
- **过程噪声 Q / 测量噪声 R**：分别代表"信模型"与"信测量"的反面权重
- **数据关联（data association）**：把检测结果对应到已有轨迹
- **马氏距离（Mahalanobis distance）**：按不确定度加权的距离
- **匈牙利算法（Hungarian algorithm）**：指派问题的全局最优解法
- **tracking-by-detection**：先检测后关联的主流跟踪范式
- **MOTA / MOTP / IDF1**：多目标跟踪三大指标（CLEAR MOT / Identity）
