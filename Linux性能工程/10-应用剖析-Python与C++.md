# 10 - 应用剖析：Python 与 C++

## 本篇目标

- 用 cProfile/py-spy 剖析 Python：分清"插桩式"与"采样式"的取舍，理解 py-spy 为什么能无侵入。
- 用 py-spy dump 排查死锁/卡死，用 top 观测 GIL 与多进程的真实形状。
- 剖析 C++：perf + demangle 的完整姿势，解释 gprof 过时的原因，知道 callgrind（计数式剖析）何时仍然不可替代。
- 建立编译器优化常识：-O2 对剖析与断点的影响、PGO/BOLT 概览。
- 完整走一遍"优化一个文本处理程序"：原始 → 剖析 → 重写 → 数字对比。

前置：03 篇（perf，C 层剖析）；操作系统系列 02/03（线程与 GIL 的调度含义）。本篇把观测从系统层拉回到"你写的代码"这一层。

## 1. Python 剖析两件套：cProfile 与 py-spy 的分工

两种剖析范式（Brendan Gregg 的分类：instrumenting vs sampling profiling）：

| 维度 | cProfile（插桩式） | py-spy（采样式） |
| --- | --- | --- |
| 原理 | 每次 Python 函数进出都记账（CPython 的 C 层钩子） | 定时暂停进程（默认 100Hz），抓当前调用栈 |
| 开销 | 2~3 倍（函数多则更糟） | <5%（01 篇的观察者效应分级里接近 perf） |
| 是否改运行方式 | 改（要重新启动进程） | 不改（`--pid` 附着到运行中进程） |
| 精度 | 精确到每次调用次数与耗时 | 统计估计（样本量决定分辨率） |
| 适用 | 离线定位热点函数、看调用次数 | 生产环境在线采样、死锁排查 |

cProfile 最小用法（标准库，无需安装）：

```
$ python3 -m cProfile -s cumulative myprog.py
         120045 function calls (119832 primitive calls) in 1.842 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.001    0.001    1.842    1.842 myprog.py:1(<module>)
     2000    0.882    0.000    1.502    0.001 myprog.py:8(tokenize_line)
   200000    0.512    0.000    0.620    0.000 {built-in method re.match}
     ...
```

列读法（man cProfile；文档 docs.python.org 的 profiler 章节）：`ncalls` 调用次数（**次数异常大本身就是线索**，比如在循环里重复编译正则）、`tottime` 函数自身耗时、`cumtime` 含子调用总耗时、`-s cumulative` 按 cumtime 排序最适合"时间花在哪条路径上"。注意 cProfile 只看 Python 层：C 扩展（numpy/torch）只显示为一行 built-in，下钻要用 03 篇的 perf。

**经验做法**：离线优化用 cProfile（要函数调用次数），线上诊断用 py-spy（不改进程）；两者输出都能转火焰图（py-spy record 直接产 SVG，cProfile 用 flameprof 工具）。

## 2. py-spy：为什么能"无侵入"采样

py-spy（`pip install py-spy`，本文 0.3.x）对运行中进程采样却不影响它，原理值得拆开看：

1. **附着**：ptrace 挂到目标进程，短暂暂停它（几十微秒一次）。
2. **读内存**：用 `process_vm_readv` 读出 CPython 解释器内部的线程状态与帧链表（CPython 每个线程栈是一串 frame 对象，链表头在解释器全局状态里）。
3. **回溯**：沿 frame 链表还原 Python 层调用栈（这是它比 perf 强的地方：perf 看到的是 C 帧 `_PyEval_EvalFrameDefault`，py-spy 直接看到你的 `tokenize_line` 函数名）。
4. **恢复运行**：立即继续，整个过程目标进程无感（采样间隔 10ms 级）。

代价与前提：需要 ptrace 权限（同 UID 或 root；容器里要 SYS_PTRACE 能力，11 篇的注）；对每个样本付一次进程停走成本，高频采样下有可测开销——但比 cProfile 低一个量级。三个核心命令：

```
$ py-spy top --pid 48112          # 实时滚动视图（像 top，按栈聚合）
$ py-spy record -o profile.svg --pid 48112 --duration 30   # 30 秒采样出火焰图
$ py-spy dump --pid 48112         # 瞬时全线程栈快照（死锁排查神器，下一节）
```

## 3. 死锁/卡死排查：py-spy dump 一锤定音

制造一个经典死锁（两把锁的循环等待，OS 系列 11 篇的死锁四条件在 Python 里的形态）：

```python
# deadlock.py —— 线程 A：先锁 lock1 再 lock2；线程 B：先 lock2 再 lock1
import threading, time

lock1, lock2 = threading.Lock(), threading.Lock()

def worker_a():
    with lock1:
        time.sleep(1)             # 制造交错窗口
        with lock2:               # ← B 已持有 lock2，A 在这里永远等
            pass

def worker_b():
    with lock2:
        time.sleep(1)
        with lock1:               # ← A 已持有 lock1，B 在这里永远等
            pass

t1 = threading.Thread(target=worker_a); t2 = threading.Thread(target=worker_b)
t1.start(); t2.start()
t1.join(); t2.join()              # 主线程等两个僵尸：进程“卡死”
```

服务"卡死不响应但进程还在"时的排查：

```
$ python3 deadlock.py &
[1] 52301
$ py-spy dump --pid 52301
Thread 52301 (idle): "MainThread"
    join (threading.py:1116)
    <module> (deadlock.py:14)
Thread 52302 (active+deadlocked): "Thread-1 (worker_a)"
    acquire (threading.py:79)
    worker_a (deadlock.py:8)         ← 等 lock2（第 8 行的 with lock2）
    ...
Thread 52303 (active+deadlocked): "Thread-2 (worker_b)"
    acquire (threading.py:79)
    worker_b (deadlock.py:13)        ← 等 lock1（第 13 行的 with lock1）
```

两行输出互指对方持有的锁，死锁的"交叉引用"一眼可见——**不需要重新加日志、不需要重启进程**。这就是 dump 在生产的价值：服务 hang 住的第一反应是 `py-spy dump`（或 C++ 的 `gdb -p PID` + `thread apply all bt`），而不是 kill -9 销毁现场。**经验做法**：排查"卡死"先 dump 全线程栈——区分"都在等锁"、"都在等 IO"、"主线程活着但 worker 全躺了"三种形态，每种对应完全不同的下一站（锁设计/04 篇 off-CPU/线程池配置）。

## 4. GIL 与多进程的观测形状

GIL（全局解释器锁）决定了同一时刻只有一个线程执行 Python 字节码。它在观测上的指纹（mpstat/pidstat 视角）：

```
$ mpstat -P ALL 1 2 | grep -E 'Average.*all|Average.*0$'
Average:   all   6.25    0.00    0.35    0.05    0.00    0.05    0.00    0.00   93.30
Average:   0    93.80    0.00    1.10    0.20    0.00    0.20    0.00    0.00    4.70
Average:   1     0.10    0.00    0.10    0.00    0.00    0.00    0.00    0.00   99.80
```

**一个线程的 Python 死循环 = 一个核 93%、其余 15 个核闲着**（总 %usr 6%）——"总 CPU 很闲但服务扛不住"的经典形状。验证 GIL 归因的对照实验：4 线程 vs 4 进程跑同一 CPU 任务：

```python
# gil_bench.py —— 线程版 vs 进程版：同一个 CPU 函数
import time, threading, multiprocessing

def crunch(n=2_000_000):
    x = 0
    for i in range(n): x += i * i
    return x

def run_threads(): ts = [threading.Thread(target=crunch) for _ in range(4)]
def run_procs():   ps = [multiprocessing.Process(target=crunch) for _ in range(4)]

if __name__ == "__main__":
    for name, maker in (("threads", run_threads), ("procs", run_procs)):
        t0 = time.perf_counter(); maker()
        print(f"{name}: 启动+运行 {time.perf_counter()-t0:.2f}s（worker 内各自计时更准，此处看形状）")
```

结论先行（数据随机器变化，形状不变）：**线程版 4 个任务串行化（GIL），进程版真正并行但付 IPC/启动成本**。经验决策表：CPU 密集 → 多进程/进程池（或把热点挪进 numpy/C 扩展释放 GIL）；IO 密集 → 线程/asyncio 都可（等 IO 时 GIL 会释放）。3.13+ 的 free-threaded 构建是未来变量，本系列以 3.10 的现状为准。

## 5. C++ 剖析：perf + demangle

C++ 程序在 perf 里的完整姿势：

```
$ g++ -O2 -g -fno-omit-frame-pointer -o app app.cpp
#     -O2 优化 + -g 符号（03 篇：两者可共存）+ 帧指针（fp 栈回溯更完整）
$ sudo perf record -F 99 -g -- ./app
$ sudo perf report
    42.10%  app  libstdc++.so.6  [.] std::__detail::_Hash_node<...>::_M_...   ← STL 混淆名自动翻译
    18.30%  app  app              [.] tokenize(std::__cxx11::basic_string...)
```

三个 C++ 特有的坑：

1. **内联吞噬函数**：-O2 后小函数被内联进调用者，perf 里"消失"了。这不是 bug（03 篇符号节）；想看清楚可临时 `-O1` 或 `-fno-inline` 对比剖析，**但优化结论必须回到 -O2 下验证**。
2. **混淆名（mangled name）**：perf 默认 demangle；拿原始名用 `--no-demangle`，手工翻译用 `echo '_ZN6detail9tokenizeB5cxx11Ev' | c++filt`。
3. **STL 一锅粥**：热点常落在 `std::unordered_map` 的实现里——含义是"你的用法让它慢"（如哈希冲突/rehash），要结合调用栈（Children）看是谁在用这个容器。

**gprof 为什么过时**（《Systems Performance》对 instrumentation 剖析器的评价，gprof 原论文 Graham et al. 1982）：它靠 `-pg` 在每次函数进出插入 mcount 记账——插桩开销大（函数密集时 2~5 倍）、改变时序（启发式假设被打破）、看不到动态库与内联后的函数、调用图在优化代码里失真。perf（PMU 采样，零重编译）+ callgrind 覆盖了它全部能力，如今 gprof 只活在教材里。

**callgrind 何时仍不可替代**：它是**计数式剖析**——用模拟 CPU 执行指令（不测真实时间），输出精确到"每条指令/每次缓存访问"的计数。适用场景：短小算法的**精确指令数对比**（A 算法 vs B 算法，排除测量噪声）、单线程确定性剖析（可重复到个位数）、cache 模拟（cg_annotate 看 L1/LLC 模拟命中率）。代价：慢 20~100 倍（01 篇），只适合离线小负载。

```
$ valgrind --tool=callgrind ./app
$ callgrind_annotate callgrind.out.52301
     Ir
 4,102,338  PROGRAM TOTALS
 1,203,441  app:tokenize(std::string)     ← 精确指令数，无任何采样误差
```

## 6. 编译器优化常识：剖析时你看到的是“优化后的程序”

- **-O2 改变一切**：内联（函数消失）、循环展开（单步断点跳行）、指令重排（变量生命周期重叠）——**调试/剖析要在与生产相同的优化等级下做**，`-O0` 的剖析结果会误导（热点分布完全不同）。
- **-g 与 -O2 无冲突**：-g 只生成调试信息，不影响优化；生产二进制带 -g（或保留带符号的副本）是剖析的前提（03 篇符号节）。
- **-fno-omit-frame-pointer**：保留帧指针，perf 的 fp 模式栈回溯完整；代价约 1~2% 性能——**经验做法**：需要常态剖析的服务镜像默认加上。
- **PGO（Profile-Guided Optimization）**：两阶段编译——先用 `-fprofile-generate` 跑真实负载收集 profile，再 `-fprofile-use` 让编译器按真实热点布局分支/内联。官方数据常见 5~15% 提升，适合热点集中的长跑服务。
- **BOLT 概览**：链接后二进制优化器（Meta 开源，github.com/facebookincubator/BOLT），按 perf 采样重排函数与基本块在二进制里的布局（改善 i-cache/分支预测）。PGO 是编译期、BOLT 是链接后，两者可叠加；适用大单二进制（数据库/解释器类），常规服务先 PGO。

## 7. 动手实验：优化一个文本处理程序

任务：统计 2000 万行日志里每类事件的次数（top-K）。先写一个"老实但错误百出"的版本：

```python
# textproc_v1.py —— 原始版：四个可优化点埋伏在里面
import re, sys, time

def top_events(path):
    counts = {}
    for line in open(path):                     # 逐行读（文件 200MB）
        line = line.strip()
        m = re.match(r"(\w+) (\w+) (.*)", line)  # 点1：每次循环重新编译正则
        if m:
            key = m.group(1) + ":" + m.group(2)  # 点2：字符串拼接产生大量临时对象
            if key not in counts:                # 点3：两次哈希查找（先 in 再 []）
                counts[key] = 0
            counts[key] += 1
    return sorted(counts.items(), key=lambda kv: -kv[1])[:10]  # 点4：全量排序只取前10

if __name__ == "__main__":
    t0 = time.perf_counter()
    print(top_events(sys.argv[1]))
    print(f"耗时 {time.perf_counter()-t0:.2f}s")
```

**第一步：症状与剖析。** `python3 textproc_v1.py app.log` 跑出 12.8s（示例机器）。cProfile 定位：

```
$ python3 -m cProfile -s tottime textproc_v1.py app.log
   ncalls  tottime  percall  cumtime  filename:lineno(function)
 20000000   6.102    0.000    9.881    re.match                     ← 热点1：2000 万次 re.match
 20000000   2.341    0.000    2.341    {method 'strip' ...}         ← 热点2（次要）
 20000000   1.203    0.000    1.203    {method 'append' of 'str'}   ← 拼接的临时串
```

**第二步：归因。** 火焰图形状（py-spy record 同样可出）：re.match 占 ~48%。根因不是"正则太慢"，而是**每行重新走 re.compile 的缓存查找与匹配**——点1 是主要矛盾。点3 的双查找、点4 的全量排序是次级问题。

**第三步：重写。**

```python
# textproc_v2.py —— 修复：预编译 + 局部变量绑定 + getdict + heapq
import re, heapq, time, sys
from collections import Counter

_PAT = re.compile(r"(\w+) (\w+) (.*)")            # 修复1：只编译一次

def top_events(path):
    counts = Counter()                            # 修复3：Counter 单次查找
    match = _PAT.match                            # 修复2：绑定到局部变量（避开属性查找）
    with open(path) as f:                         # 顺手：明确关闭
        for line in f:
            m = match(line)                       # strip 并入正则语义，省一次扫描（修复4）
            if m:
                counts[m[1] + ":" + m[2]] += 1
    return heapq.nlargest(10, counts.items(), key=lambda kv: kv[1])  # 修复4：O(n log k)
```

**第四步：同口径对比。**

| 版本 | 耗时（2000 万行） | 提速 | 主要改动 |
| --- | --- | --- | --- |
| v1 原始 | 12.8s | — | — |
| v2 修复 | 4.1s | 3.1x | 预编译正则、Counter、nlargest |
| v3 v2 + 按块读+map(_PAT.match, f) | 3.6s | 3.6x | 减少解释器循环开销 |

复盘要点：**cProfile 显示的第一热点（re.match）的修复方式不是换正则库，而是消灭 2000 万次重复编译路径**——"热点函数"要问"为什么会被调用这么多次"，这是插桩剖析里 ncalls 列的真正价值。修复后重新剖析确认无新热点转移（03 篇的"热点会转移"在此重演：strip 消失后 sorted 才浮出水面成为次热点）。

## 常见误区

1. **"cProfile 的耗时就是生产耗时"**：插桩开销 2~3 倍且改变时序，只用于找热点相对占比；绝对数字以不加剖析的 wall time 为准（01 篇）。
2. **"多开线程就能用满多核（Python）"**：GIL 下 CPU 密集线程串行化（第 4 节的 mpstat 指纹）；要看的是 numpy/C 扩展释放 GIL 或多进程。
3. **"-g 会拖慢程序所以生产不带"**：-g 只增加二进制体积，不影响运行时性能；剖析与调试都依赖它。
4. **"callgrind 报的耗时很准"**：它根本不测时间，报的是指令数；把 Ir 当耗时解读是常见错误，两者仅在"同代码同输入对比"时单调相关。
5. **"剖析一次就完事"**：每轮修复后必须重新剖析（热点转移），并回到同口径 wall time 验证——剖析是循环，不是单发。

## 本篇小结

- cProfile（插桩、精确、离线、2~3 倍开销）与 py-spy（采样、低开销、无侵入）分工明确；py-spy 的原理是 ptrace + process_vm_readv 读 CPython 帧链。
- 卡死排查第一动作是 py-spy dump 全线程栈；死锁两线程互指对方持有的锁，一眼定案。
- GIL 的观测指纹是"单核满、多核闲"；CPU 密集用多进程/释放 GIL 的扩展，IO 密集线程/asyncio 皆可。
- C++ 剖析：-O2 -g -fno-omit-frame-pointer 编译，注意内联吞噬与 STL 一锅粥；gprof 因插桩+失真被 perf/callgrind 取代；callgrind 的价值在精确指令数与缓存模拟。
- PGO/BOLT 是"用真实 profile 重排代码"的两条路；编译器优化常识的核心是：剖析看到的永远是优化后的程序。
- 文本处理实战 3.1 倍：热点修复的常见形态是消灭调用次数，而不是压榨单次调用。

## 练习

1. 用 cProfile 与 py-spy 分别剖析第 7 节的 textproc_v1.py，对比两者给出的热点排序差异；写三句话解释"插桩看到 ncalls、采样看到驻留时间"各自的盲区。
2. 运行 gil_bench.py，并用 mpstat -P ALL 1 记录两种模式下的核利用率形状；再把 crunch 换成 `time.sleep` 占位的 IO 版本，验证"IO 密集时线程不再被 GIL 串行化"。
3. 给 deadlock.py 加第三个线程（只获取 lock1 后睡 10s），用 py-spy dump 观察三线程形态并解释为什么它没死锁但贡献了"卡死假象"。
4. 对一个你自己的 C++ 小程序分别用 perf 与 callgrind 剖析，把两者的"最大热点"排序列出；若排序不一致，检查是否是内联/采样噪声导致，记录你的结论。
5. 完整执行第 7 节实验：生成 2000 万行测试日志（循环生成、注意 06 篇的小文件教训——写成一个文件），跑 v1/v2/v3 三版并填写对比表；给 v3 提一个你自己发现的第四个优化点并验证。
