# 第 3-4 周教程：Open X-Embodiment 数据集与 RLDS 数据流解析

> **本周要回答的三个问题**
> 1. OXE 数据集的构成（多少本体/多少轨迹）与"具身 ImageNet"的意义？
> 2. RLDS 格式的 Episode/Step 数据结构怎么读？
> 3. 如何加载一个 shard、取出 5 条 Episode 并渲染成带指令与动作叠加的 MP4？

对应学习计划：第 3-4 周。交付物：RLDS 数据提取与可视化脚本——下载轻量级数据集（如 `fractal20220817_data` 的一个 shard），读 5 条 Episode，渲染 MP4（画面角落实时打印语言指令与 7-DoF 动作数值）。

---

## 1. 第一性原理：具身领域的"ImageNet"意味着什么

### 1.1 根本矛盾：机器人数据的"孤岛经济学"

Stage 1 的 VLM 之所以可行，是因为互联网图文数据足够大且格式统一。机器人数据的现状曾经相反：**每个实验室一套机器人、一套坐标系、一套数据格式**——孤岛数据无法汇成规模。Open X-Embodiment（OXE，Google DeepMind 联合 30+ 机构，已核实仓库 `google-deepmind/open_x_embodiment`）的贡献是把这条路径走通：

- **多本体（Embodiments）**：数十种机器人（Google Robot、Franka、WidowX、ALOHA、移动机械臂等）；
- **百万级轨迹**：OpenVLA 预训练用了其中 **970k episodes**（已核实）；
- **统一格式**：RLDS（TFDS 生态）+ 统一动作空间约定（各数据集变换到可比较的表示）。

"ImageNet 时刻"的类比要精确：ImageNet 对视觉的意义是"**统一的监督信号来源催生了预训练-微调范式**"——OXE 对具身的意义相同：**OpenVLA/Octo 预训练于 OXE，你的任务是微调**（第 7-9 周）。Stage 4 的"数据分布决定上限"在具身语境的具体化：OXE 覆盖的任务/本体分布，决定了通用 VLA 的能力边界。

### 1.2 RLDS 数据结构：Episode → Step 的两级嵌套

RLDS（Robotic Learning Datasets，基于 `tensorflow_datasets`）的标准结构：

```text
Dataset (按 shard 分文件)
└── Episode (一条完整轨迹)
    ├── steps (tf.data.Dataset: 惰性序列!)
    │    └── Step:
    │         ├── observation: dict
    │         │    ├── image / wrist_image      (主视角 / 手腕相机, H×W×3 uint8)
    │         │    ├── state / joint_position   (本体感受: 关节角/末端位姿)
    │         │    └── (各数据集字段略有差异)
    │         ├── action: [7] float32           (下一帧动作, 已归一化 [-1,1])
    │         ├── language_instruction: str     (如 "pick up the coke can")
    │         └── is_first / is_last / is_terminal: bool (轨迹边界标记)
    └── episode_metadata: {file_path, ...}
```

四个必须建立的工程直觉：

1. **`steps` 是惰性 Dataset 不是 list**——直接 `len()` 会失败；遍历即触发读取（TTFD 数据流）；
2. **`action[t]` 是"从观察 $o_t$ 出发的下一步动作"**——对齐关系是 $o_t \to a_t \to o_{t+1}$（off-by-one 是 RLDS 处理的头号 bug 源）；
3. **图像是原始 uint8**，模型输入的 resize/归一化在训练管线做（Stage 2 的 Processor 职责）；
4. **字段名跨数据集有差异**（`image` vs `image_primary`、`state` vs `robot_state`）——OXE 联合训练靠的是每数据集一个 transform 适配层（OpenVLA 代码里可见），自研管线同理需要 per-dataset 适配。

---

## 2. 实现与验证

### 2.1 本周 MVP：加载 + 可视化脚本

```python
"""
RLDS 加载 + 5 条 Episode 渲染 MP4 (角标打印语言指令与 7-DoF 动作)。
运行方式:
  python stage10_week3_rlds_viz.py --dataset fractal20220817_data --episodes 5
依赖: tensorflow, tensorflow_datasets, matplotlib, opencv-python, numpy
  (首次运行自动下载对应 shard; 需 TF 2.x)
"""
import argparse
import cv2
import numpy as np
import tensorflow as tf


def load_rlds(name: str):
    import tensorflow_datasets as tfds
    # 纯文本指令在 tf.string; TF2 的 str 对象处理注意 decode
    ds, info = tfds.load(name, split="train", with_info=True)
    return ds, info


def to_np(x):
    """TF 张量/字符串 -> numpy (统一处理 str bytes)"""
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return x


def episode_frames(ep, max_frames=200):
    """把一条 Episode 拉平成 [(img, action, instr), ...] (steps 是惰性流)"""
    frames = []
    for i, step in enumerate(ep["steps"]):
        if i >= max_frames:
            break
        img = to_np(step["observation"]["image"])
        act = to_np(step["action"]).astype(np.float32)
        instr = to_np(step["language_instruction"])
        frames.append((img, act, instr))
    return frames


def render_mp4(frames, path="episode.mp4", fps=10):
    h, w = frames[0][0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for img, act, instr in frames:
        canvas = img.copy()
        # 角标 1: 语言指令 (自动换行)
        y = 24
        for line in _wrap(f"Instr: {instr}", 46):
            cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 20
        # 角标 2: 7-DoF 动作数值
        act_str = " ".join(f"{v:+.2f}" for v in act[:7])
        cv2.putText(canvas, f"a[:7]: {act_str}", (8, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        vw.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    vw.release()


def _wrap(text, width):
    return [text[i:i + width] for i in range(0, len(text), width)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fractal20220817_data")
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()

    ds, info = load_rlds(args.dataset)
    # 打印数据集结构 (RLDS 自省: 字段名跨数据集有差异, 先自省再取字段)
    print("observation 字段:", list(info.features["steps"]["observation"].keys()))
    print("action shape:", info.features["steps"]["action"].shape)

    n = 0
    for ep in ds.take(args.episodes):                 # take(5) 只读 5 条 (惰性优势)
        frames = episode_frames(ep)
        assert len(frames) >= 5, f"Episode 过短: {len(frames)} 帧"
        # ---- 数据完整性断言 ----
        img, act, instr = frames[0]
        assert img.dtype == np.uint8 and img.ndim == 3, "图像应为 uint8 HWC"
        assert act.shape == (7,), f"动作应为 7-DoF: {act.shape}"
        assert np.abs(act).max() <= 1.0 + 1e-5, "动作应归一化到 [-1,1]"
        assert isinstance(instr, str) and len(instr) > 0, "语言指令缺失"
        render_mp4(frames, f"episode_{n}.mp4")
        print(f"Episode {n}: {len(frames)} 帧 | 指令: {instr[:60]} | a[0]: {np.round(act[:3], 3)}")
        n += 1
    assert n == args.episodes
    print(f"✅ 已渲染 {n} 条 Episode -> episode_0..{n-1}.mp4")


if __name__ == "__main__":
    main()
```

**预期输出形态**：

```text
observation 字段: ['image', 'wrist_image', 'state', ...]
action shape: (7,)
Episode 0: 87 帧 | 指令: pick up the coke can | a[0]: [-0.012 0.031 0.988]
...
✅ 已渲染 5 条 Episode -> episode_0..4.mp4
```

断言验证数据管线的四个正确性基点：图像 dtype/布局、动作 7-DoF 与归一化区间（第 1-2 周 tokenizer 的输入前提）、语言指令存在、Episode 帧数合理。**`take(5)` 的惰性读取**保证了"只下载/解析 5 条"的效率——RLDS 流式设计的直接收益。

### 2.2 自省优先的字段适配

跨数据集字段差异是 RLDS 工程的主矛盾，标准做法是**先 `info.features` 自省、再写 per-dataset 适配器**：

```python
FIELD_ALIASES = {
    "image": ["image", "image_primary", "observation/image"],
    "wrist": ["wrist_image", "image_wrist"],
    "state": ["state", "robot_state", "joint_position"],
}

def get_field(obs: dict, canonical: str):
    for k in FIELD_ALIASES[canonical]:
        if k in obs:
            return obs[k]
    raise KeyError(f"数据集缺少 {canonical} 字段, 实际字段: {list(obs.keys())}")
```

（OpenVLA 官方代码为每个 OXE 子数据集写了专门的 transform 函数，机制相同——**适配层是具身数据工程的常驻成本**，不是一次性工作。）

---

## 3. 工程权衡与失效模式

### 3.1 数据选取的决策表（微调视角）

| 目标 | 选什么数据 |
| --- | --- |
| 通用抓取微调 | 与目标本体同款的数据（fractal=Google Robot / bridge=WidthX / DROID=Franka） |
| 语言接地强化 | 多任务多物体的数据集（语言指令多样性高） |
| 空间精度强化 | 高质量坐标类数据（+ Stage 4 合成的空间 GT，第 7-9 周） |
| 防遗忘回放 | OXE 原始混合子集（第 7-9 周 Co-finetuning） |

### 3.2 三个代表性失效模式

**失效 1：off-by-one 的观测-动作错位**
- **症状**：训练 loss 正常，仿真闭环里动作"慢半拍"（抓取时机总是晚一步）。
- **根因**：`action[t]` 对应的是"从 $o_t$ 出发的动作"，若错位成 $o_t \to a_{t+1}$（或反之），模型学到的时序关系整体平移——数据量越大拟合越"自信地错"。
- **定位**：可视化连续 3 帧：手在移动的方向 vs 对应 action 的位移方向应一致。
- **修复**：对齐规则写进适配层并加断言（用位移方向 vs action 方向的余弦相似度做自动检查，Stage 4 的验证器思维）。

**失效 2：跨数据集字段/尺度静默错配**
- **症状**：混合训练时某个数据集的动作"永远打不满"或图像异常。
- **根因**：字段别名取错（取到手腕图当主视角）；或某些数据集动作未归一化/归一化基准不同——混合后各数据集的"有效动作范围"不一致。
- **定位**：分数据集统计 action 的 min/max 与图像均值——异常者即错配。
- **修复**：适配层 + per-dataset 断言（2.1 的检查逐数据集执行）；统计量进配置并版本化。

**失效 3：TF 管线的性能陷阱**
- **症状**：GPU 利用率低（Stage 2 第 4 周的决策树：数据饥饿）。
- **根因**：`steps` 惰性流无并行预取；图像解码在主线程；episode 内串行读取。
- **定位**：`tf.data` 加 `prefetch`/`parallel_interleave` 前后的吞吐对比。
- **修复**：`ds.shuffle().prefetch(tf.data.AUTOTUNE)` + episode 级并行；图像 decode 放 `map(num_parallel_calls=AUTOTUNE)`。

---

## 4. 延伸思考题

1. **本体差异的数据价值**：Franka（高精度固定臂）与 ALOHA（双手低成本）的动作分布差异巨大，混合训练时是"噪声"还是"多样性红利"？结合 Stage 5 的数据配比思想，给出多本体混合的配比设计原则（提示：同任务跨本体的数据教"任务不变性"，同本体跨任务的数据教"技能"——两者是正交的多样性轴）。
2. **语言指令的质量轴**：OXE 各数据集的指令质量参差（从模板句到人工标注）。设计一个"指令-行为一致性"的自动检验器（类似 Stage 9 轨迹的逻辑断层检测）：判定"指令要求的物体/动作是否在图像与轨迹中真实发生"。（提示：Grounding + 轨迹终态验证；这本质上是 Stage 4 执行验证器的数据版。）
3. **动手实验**：把 2.1 脚本扩展成"轨迹统计报告"——对 100 条 Episode 统计：轨迹长度分布、动作各维的直方图、指令去重后的任务类型数。这份报告就是你微调前（第 7-9 周）判断"目标数据与 OXE 预训练分布的距离"的依据。

---

*下一篇：[第 5-6 周：SimplerEnv 仿真闭环评测](第5-6周-SimplerEnv闭环评测.md)*
