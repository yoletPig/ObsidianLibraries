# 13 - 实战项目：手写数字分类到迷你 GPT

## 本篇目标

读完本篇，你应该能做到：

1. 按工程规范组织一个训练项目：`model.py / data.py / train.py / evaluate.py` 分层，配置与代码分离。
2. 写出带 **TensorBoard 记录、梯度范数监控、checkpoint、早停**的完整训练脚本（代码全量给出）。
3. 用**消融实验记录表**管理实验，用**复现检查清单**保证结果可复现。
4. 把同样的骨架套到 8 章的 miniGPT 上——从此"接手任何训练仓库"都有章法。

本篇是全系列的毕业设计：01–12 的每个知识点都会在代码里找到位置（文中用 `# ←N章` 注释标出处）。所有代码 CPU 可跑。

## 1. 项目结构

```text
mnist_project/
├── configs.py          # 所有超参数集中一处(配置与代码分离)
├── data.py             # 数据管道(04 章)
├── model.py            # 模型定义(03/06 章)
├── train.py            # 训练入口: argparse + 训练循环 + 记录 + checkpoint
├── evaluate.py         # 独立评估脚本: 加载 checkpoint 出测试指标
└── runs/               # 每次实验一个子目录: 日志 + checkpoint
    └── baseline/
        ├── tb/         # TensorBoard 事件文件
        ├── ckpt_best.pt
        └── ckpt_last.pt
```

分层原则：**改模型不动训练循环，改数据不动模型**。`runs/<实验名>/` 一实验一目录，13 章实验管理的最小单位。

## 2. 全量代码

### 2.1 configs.py —— 配置集中

```python
"""configs.py: 所有超参数集中一处, 消融实验只改这里(或用 argparse 覆盖)"""
SEED = 42
DATA_DIR = "./data"
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3                 # AdamW 峰值(09 章)
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0       # 梯度裁剪(05/09 章)
PATIENCE = 3              # 早停耐心(10 章)
VAL_RATIO = 0.1
NUM_WORKERS = 2
RUN_NAME = "baseline"
```

### 2.2 data.py —— 数据管道（04 章直接复用）

```python
"""data.py: 数据加载与划分。职责单一: 只管数据, 不碰模型。"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_transforms():
    train_tf = transforms.Compose([
        transforms.ToTensor(),                       # HWC->CHW, 0-255->0-1 (04 章)
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 训练集统计
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return train_tf, eval_tf

def build_loaders(cfg):
    train_tf, eval_tf = get_transforms()
    full = datasets.MNIST(cfg.DATA_DIR, train=True, download=True, transform=train_tf)
    n_val = int(len(full) * cfg.VAL_RATIO)
    n_train = len(full) - n_val
    gen = torch.Generator().manual_seed(cfg.SEED)            # 划分可复现(04 章)
    train_set, val_set = random_split(
        full, [n_train, n_val], generator=gen)
    # 注意: val_set 共享了 train_tf(含随机增强的话应换成 eval_tf, 见 04 章原则)
    loader_kwargs = dict(num_workers=cfg.NUM_WORKERS,
                         persistent_workers=cfg.NUM_WORKERS > 0)
    g = torch.Generator(); g.manual_seed(cfg.SEED)
    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE,
                              shuffle=True, generator=g, drop_last=True,  # BN 不吃尾批(06 章)
                              **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False,
                            drop_last=False, **loader_kwargs)   # 评估不丢样本(04 章)
    return train_loader, val_loader
```

### 2.3 model.py —— 模型（06 章的 MiniCNN）

```python
"""model.py: 模型定义。只定义前向, 不含训练逻辑。"""
import torch.nn as nn

class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(                # 逐层 shape 注释(06 章习惯)
            nn.Conv2d(1, 16, 3, padding=1),           # (B, 1, 28, 28) -> (B, 16, 28, 28)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # -> (B, 16, 14, 14)
            nn.Conv2d(16, 32, 3, padding=1),          # -> (B, 32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # -> (B, 32, 7, 7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                             # -> (B, 1568)
            nn.Linear(32 * 7 * 7, 10),                # logits (B, 10), 无 Softmax(03 章)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def build_model():
    model = MiniCNN()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")   # 20,586
    return model
```

### 2.4 train.py —— 训练入口（全系列骨架的工程化）

```python
"""train.py: python train.py --lr 1e-3 --run-name baseline"""
import argparse, random, time, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter    # pip install tensorboard

import configs as cfg
from data import build_loaders
from model import build_model

def seed_everything(seed):                            # 04 章三件套
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@torch.no_grad()
def evaluate(model, loader, device):                  # 03 章评估模板
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += loss_fn(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total

def main():
    # ---- 命令行覆盖配置: 消融实验只加参数, 不改代码 ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=cfg.LR)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--weight-decay", type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument("--run-name", type=str, default=cfg.RUN_NAME)
    args = parser.parse_args()

    seed_everything(cfg.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = f"runs/{args.run_name}"
    writer = SummaryWriter(f"{run_dir}/tb")           # TensorBoard 记录器

    train_loader, val_loader = build_loaders(args)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_val = float("inf")
    bad_epochs = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()                     # 03 章四步: 清-前-反-更
            loss = loss_fn(model(x), y)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(   # 09 章裁剪, 返回范数
                model.parameters(), cfg.MAX_GRAD_NORM)
            optimizer.step()

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

        val_loss, val_acc = evaluate(model, val_loader, device)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)
        writer.flush()
        print(f"epoch {epoch}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"({time.time()-t0:.0f}s)")

        # ---- checkpoint 五件套(12 章) + 早停(10 章) ----
        ckpt = {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch, "val_acc": val_acc,
                "rng": torch.get_rng_state()}
        torch.save(ckpt, f"{run_dir}/ckpt_last.pt")
        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(ckpt, f"{run_dir}/ckpt_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.PATIENCE:
                print(f"early stop @ epoch {epoch}")
                break
    writer.close()

if __name__ == "__main__":
    main()
```

典型输出（CPU 每 epoch 约 40–60 秒；不同机器/种子数值有出入）：

```text
模型参数量: 20,586
epoch 1  val_loss=0.0655  val_acc=0.9800  (48s)
epoch 2  val_loss=0.0488  val_acc=0.9848  (47s)
epoch 3  val_loss=0.0461  val_acc=0.9859  (48s)
```

`train/grad_norm` 曲线的读法（09 章）：前期大、随训练收敛、偶发尖峰属于正常；**长期贴着 1.0** 说明每步都在裁剪，lr 该降。

### 2.5 evaluate.py —— 独立评估

```python
"""evaluate.py: python evaluate.py runs/baseline/ckpt_best.pt"""
import sys, torch
from data import build_loaders
from model import build_model
import configs as cfg

@torch.no_grad()
def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "runs/baseline/ckpt_best.pt"
    device = "cpu"
    ckpt = torch.load(ckpt_path, map_location="cpu")       # 12 章: map_location 习惯
    model = build_model().to(device)
    model.load_state_dict(ckpt["model"])                   # strict=True 默认, 键名错了立刻报
    model.eval()

    _, val_loader = build_loaders(cfg)
    correct = total = 0
    confusion = torch.zeros(10, 10, dtype=torch.long)      # 混淆矩阵: 定位"哪两类互相认错"
    for x, y in val_loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        for t, p in zip(y, pred):
            confusion[t, p] += 1
        total += len(y)
    print(f"checkpoint 来自 epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")
    print(f"复算 val_acc = {correct/total:.4f}")
    print("混淆矩阵(行=真实, 列=预测):")
    print(confusion)

if __name__ == "__main__":
    main()
```

混淆矩阵是比单一 acc 强得多的诊断：4↔9、3↔5 的混淆热区直接指出下一步（更多这类样本的增强？更大的模型？）。

### 2.6 TensorBoard / W&B 记录

```bash
tensorboard --logdir runs/          # 浏览器打开 http://localhost:6006
```

最小记录集（本篇 train.py 已含）：`train/loss`、`train/grad_norm`、`train/lr`、`val/loss`、`val/acc`。W&B 版本只需三行替换（`import wandb; wandb.init(project="mnist", name=args.run_name, config=vars(args)); wandb.log({...})`），好处是云端自动汇总每次实验的超参-指标对，消融表自动生成。经验做法：本地开发用 TensorBoard（零依赖），团队协作/长期实验用 W&B。

## 3. 消融实验记录表模板

**一次只改一个变量**（04 章的可复现管道是前提）。模板直接抄进你的实验笔记：

| 实验 ID | 改动 | 关键超参 | val_acc | val_loss | 参数量 | 训练时长 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 无 | lr=1e-3, wd=1e-4, bs=128 | 0.9859 | 0.0461 | 20,586 | 4min | 基准 |
| A1 | 去 BN | 其余同上 | （待填） | | 20,306 | | 验证 BN 贡献 |
| A2 | lr×3 | lr=3e-3 | （待填） | | 20,586 | | 学习率敏感性 |
| A3 | 去梯度裁剪 | max_norm=∞ | （待填） | | 20,586 | | 观察稳定性 |
| A4 | MLP 换 CNN | 03 章模型 | （待填） | | 203,530 | | 结构消融 |

记录纪律（经验做法）：val 指标截断到 4 位小数没有意义——MNIST 上 0.9859 与 0.9862 的差异多在随机噪声范围内；**结论只写"明确超过噪声水平"的差异**，否则标"无显著差异"。这也回应 10 章的"间接过拟合"警告。

## 4. 复现检查清单

发布/存档一个实验前过一遍（逐项对应正文出处）：

```text
[ ] 随机种子: seed_everything 全家桶 + DataLoader generator + worker_init_fn (04 章)
[ ] 配置快照: 该次实验的完整超参(含命令行覆盖)存入 runs/<name>/config.json
[ ] 依赖版本: python/pytorch/torchvision 版本记录(torch.__version__ 一行脚本)
[ ] 数据版本: 数据来源、预处理统计量(如 0.1307/0.3081)与划分方式写入 README (04 章)
[ ] checkpoint: 含 model+optimizer+epoch+rng 五件套 (12 章)
[ ] 指标复算: evaluate.py 对 ckpt_best 复算的 acc 与训练日志一致
[ ] 曲线留存: TensorBoard 事件文件随代码一起归档
[ ] 消融表: 每个实验一行, 变量唯一 (第 3 节)
```

`config.json` 快照的三行实现（放进 train.py 开头）：

```python
import json
with open(f"{run_dir}/config.json", "w") as f:
    json.dump(vars(args) | {"seed": cfg.SEED, "torch": torch.__version__}, f, indent=2)
```

## 5. 迁移到 miniGPT：同一副骨架

8 章的 miniGPT 训练循环与 2.4 的 `train.py` **结构完全一致**，只换三个零件：

```python
# data.py 换成: 字符级语料 + get_batch(08 章 3.1)
#   def get_batch(cfg): ix = torch.randint(len(data)-T-1, (cfg.BATCH_SIZE,)); ...
# model.py 换成: MiniGPT(vocab_size=V, d_model=64, n_layer=2, ...)  (08 章 2.4)
# train.py 换两行:
#   loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))   # 逐位置交叉熵(08 章)
#   writer.add_scalar("val/ppl", math.exp(val_loss), epoch)    # 语言模型看困惑度
```

看困惑度（perplexity $= e^{\text{loss}}$）而非裸 loss 汇报——它与"模型在几个候选里犹豫"直接对应，是语言模型的通用汇报口径。消融表同样适用（换 n_head、换位置编码、去 warmup……08 章练习 1–3 就是现成的消融项）。

## 6. 常见陷阱

1. **把所有代码塞进一个 train.py**：消融时改一处动全身，diff 不可读——分层是实验效率问题，不是洁癖。
2. **run 目录复用**：两次实验写进同一个 `runs/baseline/`，TensorBoard 曲线交叉污染。每次实验新目录（或用时间戳后缀）。
3. **只记 acc 不记 grad_norm/lr**：训练出问题时没有诊断信息，只能重跑——09 章"健康度指标"必须从第一天就记。
4. **验证集参与了超参选择却当测试集汇报**：10 章间接过拟合；正式汇报用从未参与决策的 test split。
5. **checkpoint 只在结束时存**：中断=全部重来；每个 epoch 存 `ckpt_last`，best 单独存（2.4 的双文件策略）。
6. **混淆矩阵忘除以行归一化就目测**：类不平衡时原始计数误导；观察时看"每行内占比"。

## 本篇小结

- 项目四件套 `configs/data/model/train(+evaluate)` 各司其职；消融实验"一次一个变量"，命令行覆盖超参而非改代码。
- 训练脚本标配：seed 全家桶、四步循环、梯度裁剪+范数记录、五件套 checkpoint、早停、TensorBoard 最小记录集。
- 实验管理 = 消融表（变量唯一 + 结论区分噪声） + 复现清单（种子/配置快照/依赖版本/数据版本）。
- 同一副骨架换三个零件就是语言模型训练；指标口径换成困惑度。
- 至此，全系列 01–13 闭环：张量 → 手写训练 → Module/DataLoader → 原理 → CNN/注意力/Transformer → 训练工程 → 微调 → 部署 → 本篇的工程化整合。下一步：去读《VLM & VLA》Stage 1——那里的一切默认你已在这里毕业。

## 练习

1. 跑通 baseline，再跑消融 A1（去 BN：把 `model.py` 里的 `nn.BatchNorm2d` 删掉）与 A2（lr=3e-3），填完整消融表，写出各消融的结论。
2. 给 train.py 加 confusion matrix 的 TensorBoard 记录（`writer.add_figure` 需 matplotlib；或每 epoch 打印文本版）。
3. 把 PATIENCE 改成 1 并故意加一个 `lr=10` 的运行，观察 grad_norm 曲线与早停行为，体会"裁剪/早停作为诊断信号"。
4. 按 5 节把 miniGPT 接入本项目结构，用 8 章语料训练 1000 步，汇报困惑度曲线。
5. （综合）写一个 `sweep.py`：自动循环调用不同超参组合跑训练、解析日志填入消融表 CSV——这就是一次微型实验管理系统的诞生。

## 参考来源

- TensorBoard 与 `torch.utils.tensorboard.SummaryWriter` 官方文档: https://pytorch.org/docs/stable/tensorboard.html
- PyTorch 官方教程《Optimizing Model Parameters》（训练脚本的标准形态）: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- 混淆矩阵与分类诊断的标准做法: scikit-learn 文档《Confusion matrix》: https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
- W&B 快速上手: https://docs.wandb.ai/quickstart
- 实验管理纪律（一次一变量、配置快照）为 MLOps 工程共识，标注：经验做法。
- 8 章 miniGPT 代码出处同 08 篇参考来源（Karpathy, nanoGPT）。
