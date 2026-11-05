# exp2_pointnet_invariance.py —— 第 2 周：PointNet 对称函数（max pooling）置换不变性验证
# 复现：python3 exp2_pointnet_invariance.py（无需参数，输出与教程"真实运行输出"一致）
import numpy as np

rng = np.random.default_rng(0)


def mlp_and_maxpool(points, W1, b1, W2, b2):
    """对 (N,3) 点集做共享 MLP（两层），再对点维做 max，输出 (64,) 全局特征。"""
    h = np.maximum(points @ W1 + b1, 0)   # (N, 32) ReLU，每个点独立过同一组权重
    f = np.maximum(h @ W2 + b2, 0)        # (N, 64)
    return f.max(axis=0)                  # (64,) 逐通道取 N 个点中的最大值


# 随机固定一组"共享权重"（相当于训练好的 PointNet）
W1 = rng.normal(0, 0.5, size=(3, 32))
b1 = rng.normal(0, 0.1, size=32)
W2 = rng.normal(0, 0.5, size=(32, 64))
b2 = rng.normal(0, 0.1, size=64)

# 一片"车顶"点云：N=1024 个点
pts = rng.normal(0, 0.5, size=(1024, 3))
pts[:, 2] += 1.6                    # 抬高模拟车顶高度

feat = mlp_and_maxpool(pts, W1, b1, W2, b2)

# 实验1：随机打乱点的顺序（置换）
perm = rng.permutation(len(pts))
feat_perm = mlp_and_maxpool(pts[perm], W1, b1, W2, b2)
d1 = np.abs(feat - feat_perm).max()
print(f"实验1 随机置换 1024 个点后，全局特征最大差异 = {d1:.2e}")
assert d1 < 1e-12

# 实验2：完全逆序
feat_rev = mlp_and_maxpool(pts[::-1], W1, b1, W2, b2)
d2 = np.abs(feat - feat_rev).max()
print(f"实验2 完全逆序后，全局特征最大差异          = {d2:.2e}")
assert d2 < 1e-12

# 对照：max/mean 是对称函数，但"取第一个点"不是——它的结果依赖输入顺序
feat_first = np.maximum(pts[0] @ W1 + b1, 0) @ W2 + b2   # 只取第 1 个点
feat_first_rev = np.maximum(pts[::-1][0] @ W1 + b1, 0) @ W2 + b2
d3 = np.abs(feat_first - feat_first_rev).max()
print(f"对照  '取第一个点'在逆序后差异（非对称）      = {d3:.2e}（≠0，会因顺序变化）")
assert d3 > 1e-6
print("验证通过：max pooling 是对称函数，PointNet 输出与点的输入顺序无关")
