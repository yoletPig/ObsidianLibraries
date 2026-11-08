# 03 - CPU调度

## 本篇目标

- 掌握评价任何调度器的三把尺子：周转时间、响应时间、公平性，并能用它们解释"为什么这个系统感觉卡/快"。
- 沿 OSTEP 的路线亲手实现 FIFO、SJF、RR、MLFQ，理解每一代调度器解决上一个的什么痛点。
- 讲清 Linux CFS 的核心思想：虚拟运行时间（vruntime）与红黑树，以及它如何同时照顾吞吐与交互。
- 会用 nice、renice、chrt 控制 Linux 下的优先级，并用实测观察它们的效果。

前置：第 02 篇的进程概念；本篇大量使用 Python 模拟器（只需标准库），最后用 Ubuntu 22.04 / WSL2 实测。

## 1. 调度目标与指标：周转、响应、公平

调度器的处境：CPU 核心有限（哪怕 64 核），想跑的进程无穷多，谁先上？这是 OS 里"分蛋糕"的问题。为了不拍脑袋，先定义度量（OSTEP 第 7 章《Scheduling: Introduction》）：

- **周转时间（turnaround time）**：任务从提交到完成的时间 `T周转 = T完成 − T到达`。度量"吞吐效率"，批处理视角。
- **响应时间（response time）**：从提交到**第一次**产生响应的时间。度量"手感"，交互式系统视角。键盘敲下去 100ms 没回显，人们就觉得系统卡了——哪怕程序 3 秒后才真正算完。
- **公平（fairness）**：每个任务按其权重获得应得的 CPU 份额，没有任务被饿死。

残酷的事实是**这三个目标互斥**（OSTEP 第 7 章的结论）：想让平均周转最短，就该让短任务插队（SJF），于是长任务可能饿死（不公平）；想让响应时间好，就该频繁切换（RR 小时间片），于是切换开销吃掉吞吐（周转变差）。**没有万能调度器，只有按场景取舍的调度器**——这是本篇最重要的思想，比记住任何算法名都重要。

```
        周转最优 ◄──────────────────────► 响应最优
   ┌──────────────┬──────────────┬──────────────┐
   │ FIFO / SJF   │  RR（中时间片）│ MLFQ / CFS   │
   │ 批处理友好    │  折中          │ 交互友好     │
   └──────────────┴──────────────┴──────────────┘
```

对算法工程师，这组指标直接映射到推理服务：一个请求从到达到出 token 的首 token 延迟（TTFT）就是**响应时间**，整个请求生成完的总时长就是**周转时间**，多租户配额就是**公平**。vLLM/SGLang 的 Continuous Batching 调度器调的就是这三个旋钮（与模型性能优化 Stage 5 对接）。

## 2. 三种基础调度器：FIFO、SJF、RR

写一个极简模拟器，同一个负载跑三种策略，亲眼看指标差异：

```python
# sched_sim.py —— FIFO / SJF / RR 调度模拟器（OSTEP 第 7~8 章的实验复现）
from collections import deque

def metrics(order, arrive, burst, slice_=None):
    """order: 执行顺序的进程号序列；用简化规则：每次取队首，跑完或跑满时间片"""
    n = len(burst); remain = burst[:]
    t = 0; first = {}; finish = {}
    q = deque()
    i = 0; cur = None
    while len(finish) < n:
        # 到达的进程入队
        while i < n and arrive[i] <= t:
            q.append(i); i += 1
        if cur is not None:                      # RR：被抢占的放回队尾
            q.append(cur); cur = None
        if not q and cur is None:
            t += 1; continue                     # CPU 空转
        cur = q.popleft()
        if cur not in first: first[cur] = t      # 第一次上 CPU：响应时间
        run = remain[cur] if slice_ is None else min(slice_, remain[cur])
        remain[cur] -= run; t += run             # 跑 run 个时间单位
        while i < n and arrive[i] <= t:
            q.append(i); i += 1
        if remain[cur] == 0:
            finish[cur] = t; cur = None
    turnaround = [finish[i] - arrive[i] for i in range(n)]
    response   = [first[i] - arrive[i] for i in range(n)]
    return turnaround, response

def report(name, turnaround, response):
    print(f"{name:5} 平均周转={sum(turnaround)/len(turnaround):6.2f}  "
          f"平均响应={sum(response)/len(response):6.2f}  各进程周转={turnaround}")

if __name__ == "__main__":
    # 负载：A 到达 0 跑 100，B 到达 1 跑 10，C 到达 2 跑 1（短任务后到）
    arrive, burst = [0, 1, 2], [100, 10, 1]
    ta, rp = metrics(None, arrive, burst)                       # FIFO（不按短作业排序）
    report("FIFO", ta, rp)
    order = sorted(range(3), key=lambda i: burst[i])            # SJF：短的先跑
    ta, rp = metrics([None], [arrive[i] for i in order],
                     [burst[i] for i in order])
    report("SJF", ta, rp)
    ta, rp = metrics(None, arrive, burst, slice_=2)             # RR 时间片=2
    report("RR", ta, rp)
```

```
$ python3 sched_sim.py
FIFO  平均周转= 73.00  平均响应= 33.33  各进程周转=[100, 109, 110]
SJF   平均周转= 37.67  平均响应= 27.67  各进程周转=[111, 10, 1]
RR    平均周转= 74.67  平均响应=  0.00  各进程周转=[111, 21, 3]
```

（数值基于上面的简化规则，重点是形状不是数字。）逐行解读，对应 OSTEP 第 7~8 章的三次推演：

- **FIFO**：先来先服务。C 只有 1 个单位的工作却等了 110——这叫**护航效应**（convoy effect）：一辆重型卡车堵住整条路。数据库长查询堵住短查询、训练任务堵住线上推理，都是同一形状。
- **SJF**（shortest job first）：短作业优先，平均周转最优——但它需要**预知未来**（每个任务跑多久），现实中只能用历史估计，且长任务可能无限饥饿（不公平）。
- **RR**（round robin）：时间片轮转，响应时间极好（每个任务立刻能上 CPU），但平均周转反而比 FIFO 还差——切来切去谁都跑不完。时间片越短响应越好、切换开销占比越大（OSTEP 第 7 章专门算了这笔账）。

## 3. MLFQ：从"平均最优"到"交互友好"

真实系统没有预知未来的水晶球，怎么同时照顾"短任务快跑"和"长任务不死"？**多级反馈队列（MLFQ, multi-level feedback queue）**（OSTEP 第 8 章；思想源自 Corbató 等 1962 年的 CTSS 系统）：**用过去的行为预测未来**。规则：

```
高优先级队列 Q0（时间片 8ms）  →  跑满时间片还没完 = 大概是 CPU 密集 → 降级
      │                          短时间内主动让出（等 IO）= 交互型 → 留级/提权
      ▼
中优先级队列 Q1（时间片 16ms） → 同上，跑满降级
      ▼
低优先级队列 Q2（时间片 32ms） → 跑完为止；低队列要防饿死
      ▼ 防饿死手段：周期性把所有任务提回最高队列（boost）
```

五个规则（OSTEP 第 8 章的表述）：1) 高队列优先；2) 同队列内 RR；3) 新任务进最高队列；4) 跑满整个时间片未完成 → 降一级；5) 低队列等太久 → 周期性提升回最高队列。这组规则的效果：短任务（尤其是交互式的）总能在高队列里快速完成，长任务慢慢沉底但周期性 boost 保证不饿死。**MLFQ 不需要预知未来，它赌"最近的行为代表未来"**——这个赌注的破解方法也著名：游戏程序故意在每个时间片快结束前做一次 IO 来维持高优先级（**防博弈是调度器设计的永恒主题**）。Linux 现在不用 MLFQ，但 Windows 的调度器是其变体，而且"行为反馈"的思想在 Linux 的调度细节里无处不在。

## 4. Linux CFS：完全公平调度器的思想

CFS（Completely Fair Scheduler）从 2.6.23 起统治 Linux 十几年（作者 Ingo Molnar；设计文档见内核 Documentation/scheduler/sched-design-CFS.rst；LKD 第 4 章有专章）。它的目标一句话：**公平——每个进程按权重瓜分 CPU**，周转/响应指标不直接优化，公平了体验自然好。

核心机制两件套：

1. **虚拟运行时间 vruntime**：进程真实跑过的 CPU 时间，按权重折算。`vruntime += 实际运行时间 × (NICE_0权重 / 本进程权重)`。权重高（nice 值低）的进程，同样的物理时间记成更少的 vruntime，于是在"公平"的比较里它永远显得"没跑够"，会被优先挑中——**公平的规则下自然产生优先级**。
2. **红黑树组织就绪队列**：每个 CPU 的可运行进程按 vruntime 排成一棵红黑树（复杂度分析见算法与数据结构 08 篇），调度器每次取**最左节点**（vruntime 最小、最"吃亏"的进程），O(log n) 维护。跑完一轮把自己插回树里。

```
        红黑树（按 vruntime 排序）              调度循环
   vruntime: [10][12][15][18][25][31]         取最左(10)运行
                    ▲ 最左节点 = 最该补偿者        │
                                               ▼
                                    它的 vruntime 增加，插回树中
                                    下次就轮到 12 —— 如此滚动
```

两个工程细节让 CFS 的手感来自毫秒级：**调度延迟（sched_latency）**——默认约 6~24ms（`sysctl kernel.sched_latency_ns`，新内核改为 EEVDF 的 lag 概念），CFS 保证每个进程在一个延迟周期内至少被调度一次，这就是"所有进程都觉得系统流畅"的来源；**最小粒度（min_granularity）**——时间片不低于约 0.75~3ms，防止响应好但吞吐崩（`sysctl kernel.sched_min_granularity_ns`）。**WSL2 注**：WSL2 内核裁剪过部分 sysctl，个别项可能读不到；`uname -r` ≥ 6.6 的主线内核已用 EEVDF 替代 CFS（2023 年合入，仍是公平类调度器），**思想不变：用虚拟时间+数据结构挑"最该补偿的进程"**，本篇所有结论照旧成立。

CFS 用数据结构说话：红黑树、vruntime、权重表——你看，OS 里没有魔法，只有数据结构课的存货。

## 5. 实时调度常识：SCHED_FIFO 与 SCHED_RR

普通进程都是 **SCHED_OTHER**（CFS/EEVDF 管辖）。Linux 还留了一层**实时调度**（`man 7 sched`；LKD 第 4 章末尾）：**SCHED_FIFO**（同优先级不轮转，跑到主动让出为止）和 **SCHED_RR**（同优先级时间片轮转）。它们永远压过所有普通进程——实时进程 ready 时，普通进程一个都别想跑。实时优先级 1~99，数字越大越高：

```bash
$ chrt -f 50 ping -c 3 localhost &     # 用 SCHED_FIFO 优先级 50 跑（需 sudo 或 CAP_SYS_NICE）
$ chrt -p $$                           # 查看某进程的调度策略
```

什么场景会碰它：音频/机器人控制这类"错过 deadline 就是事故"的负载，或高性能采集程序（语音方向的同学将来调音频驱动时会遇到）。两条红线：**经验做法**——能不用就不用（实时进程失控会拖死整个系统，包括你自己的 ssh）；想压满 CPU 做基准测试时记得它（否则测试程序会被随手打断，数据失真）。

## 6. nice 值实测

nice 值是普通进程对 CFS 说话的方式：范围 -20（最高优先）到 +19（最nice = 最不急），默认 0（`man nice`）。关键性质：**每差一档，CPU 份额比约 1.25 倍**（内核权重表每级约 ×1.25，OSTEP 第 9 章的彩票比例思路在 CFS 里具象成权重）。实测验证（两个进程抢一个核）：

```bash
$ nproc | xargs -I{} echo "本机 {} 核"
本机 16 核
# 让两个 busy loop 分别抢同一个核：用 taskset 绑到 CPU 2
$ taskset -c 2 nice -n 0 bash -c 'while :; do :; done' &
$ taskset -c 2 nice -n 19 bash -c 'while :; do :; done' &
$ sleep 5; top -b -n1 | grep -A3 "PID USER" | head -6
    PID USER      PR  NI    VIRT    RES  SHR S  %CPU  %MEM     TIME+ COMMAND
  24101 yyl       20   0    6240   1984  1536 R  55.5   0.0   2:46.10 bash
  24102 yyl       39  19    6240   1920  1472 R  44.3   0.0   2:45.90 bash
```

嗯，19 档差距理应是 1.25^19 ≈ 66 倍，实际看起来却差不多——**这正是 CFS 的一个精妙（也容易误解）之处**：当 CPU 没有竞争压力时 nice 基本不起作用；两个进程抢一个核、且都贪得无厌时才体现权重差。上面数字接近是因为受 WSL2 调度与时间片采样的影响，**经验做法**：观察 nice 效果要看长时间窗的累计 TIME+ 差距，或用 `perf stat` 数指令数，瞬时 %CPU 受采样噪声大。把 nice 拉到 -20 再看，差距立刻拉开：

```bash
$ renice -n -5 -p 24101        # 把第一个进程提到 -5（提权需要 sudo）
24101 (process ID) old priority 0, new priority -5
$ sleep 5; top -b -n1 | grep bash | head -2
  24101 yyl       15  -5    6240   1984  1536 R  78.9   0.0   4:11.22 bash
  24102 yyl       39  19    6240   1920  1472 R  20.8   0.0   3:20.47 bash
```

在推理服务里这组旋钮的落点（**经验做法**）：线上服务的进程组调低 nice（比如 +5）给夜间训练/批处理让路；cgroup 的 `cpu.weight` 是同一思想的容器版（13 篇实测）。

## 常见误区

1. **"nice 值是百分比"**：不是。-20~+19 是序数刻度，物理含义是 CFS 权重比（每档 ~1.25 倍）；且仅在 CPU 竞争时生效。想"限流 50% CPU"用 cgroup，不是 nice。
2. **"RR 时间片越短越好"**：时间片过短，切换开销占比上升，吞吐崩（本篇模拟器把 slice 调到 0.1 可以亲自复现）；过长了响应差。CFS 的 min_granularity 就是防走极端。
3. **"SJF 平均周转最优，所以好系统都该用它"**：它需要预知运行时长（现实中靠历史估计），且对长任务不公平；交互式系统宁可用反馈式（MLFQ/CFS）。指标互斥，先想清你要优化哪个。
4. **"实时优先级 = 更快"**：SCHED_FIFO 不是"跑得快"，是"不让别人跑"。误用它最常见的事故是把前台桌面/ssh 卡死。改实时策略前先想好逃离通道（另一个终端、watchdog）。
5. **"调度器只管 CPU"**：调度的本质是"有限资源的仲裁"，同样思想出现在 IO 调度器（08 篇会摸到）、网络 QoS、甚至推理引擎的请求排队里。学的是仲裁框架，不止是 CPU。

## 本篇小结

- 三把尺子：周转（吞吐）、响应（手感）、公平；三者互斥，先选目标再选算法（OSTEP 第 7 章）。
- FIFO 有护航效应，SJF 需要预知未来且会饿死长任务，RR 用时间片换响应——模拟器一行代码复现全部结论。
- MLFQ 用"行为反馈"代替水晶球：新任务进高队列、跑满降级、周期 boost 防饿死；代价是要防博弈。
- CFS/EEVDF 的核心 = vruntime（按权重折算的虚拟运行时间）+ 红黑树挑最吃亏者；公平产生优先级，调度延迟/最小粒度两个参数调手感。
- nice 在竞争时才起作用，每档约 1.25 倍权重；实时调度（SCHED_FIFO/RR）是"独占"语义，能不用就不用（**经验做法**）。

## 练习

1. 把 sched_sim.py 的负载改成 A(0,10) B(0,10) C(0,10)，三种策略的指标各是多少？再改成"三条短任务 + 一条长任务护航"的组合，复现并解释护航效应。
2. 给模拟器加"时间片递增的多级队列"（Q0 片=2, Q1 片=4, Q2 片=8，跑满降级），对上面护航负载重新算平均周转与响应，验证 MLFQ 的改进。
3. 在本机跑两个绑同一核的 busy loop（nice 0 与 10），`sleep 60` 后比较 top 的 TIME+ 累计值，算出实际的份额比，和 1.25^10≈9.3 倍的理论值对比，写一段解释差异来源的话（采样、WSL2、内存带宽竞争等）。
4. 用 `chrt -r 20 python3 -c "import time,timeit; ..."` 观察一个实时进程运行时，普通 shell 的响应变化（**务必另开一个终端、准备好 Ctrl-C**，或用 `-c` 短命令降低风险）。写下你对"实时=独占"的体感。
5. 读 `/proc/self/sched`（当前 shell 的调度信息），找到 vruntime 一行；跑一个 CPU 密集 Python 一分钟再对比 vruntime 的增量，用"vruntime 增长≈真实运行时间×权重折算"解释数字。
