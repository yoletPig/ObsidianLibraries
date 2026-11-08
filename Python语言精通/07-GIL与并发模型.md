# 07 - GIL 与并发模型

## 本篇目标

1. 说清 GIL 是什么、为什么存在：它锁的是"字节码执行权"，不是"你的代码"；
2. 用可复现代码实测 **CPU 密集 vs IO 密集** 在串行/多线程/多进程下的差异，得出结论而不是背结论；
3. 理解"为什么 GIL 对多线程 IO 程序影响小"——IO 等待时线程会**主动释放** GIL；
4. 如实了解 3.13 free-threading 的现状，并建立**并发选型决策树**（threading/multiprocessing/asyncio）。

## 1. GIL 是什么：一次只一个线程执行字节码

**GIL（Global Interpreter Lock，全局解释器锁）**：CPython 解释器中的一把互斥锁，**同一时刻只允许一个线程执行 Python 字节码**（官方文档 Glossary → GIL："the mechanism used by the CPython interpreter to ensure that only one thread executes Python bytecode at a time"）。三个常见误解先纠正：

1. **GIL 不是 Python 语言的特性，是 CPython 的实现细节**——Jython/IronPython 没有，PyPy 有自己的 GIL，3.13+ 的 free-threading 构建可以没有；
2. **GIL 锁的是"字节码执行"，不是所有操作**——C 扩展（numpy 的大块矩阵运算）、文件读写、`hashlib` 计算等在进入 C 层时可释放 GIL；内存分配等由解释器内部更细的锁保护；
3. **GIL 不保证线程安全**——它保证的是"单条字节码原子"，而 `counter += 1` 是**多条字节码**（读、加、写），中间可被切换（08 篇用银行账户实测这个竞态）。

为什么 GIL 存在：CPython 的内存管理（引用计数，01 篇）不是线程安全的，给每个对象都加细粒度锁会带来巨大开销且极易死锁，于是 1990s 选择了"一把大锁"这个简单方案。代价是：**多线程无法利用多核并行执行 Python 代码**。

线程切换的时机（理解后两个小节的关键）：当前线程在以下情况会让出 GIL——① 执行了一定数量指令后（CPython 3.2 起每 5ms 检查一次切换点，`sys.getswitchinterval()` 可见）；② **遇到阻塞式 IO 调用时立即释放**（socket 等待数据、文件读写、`time.sleep`）；③ C 扩展显式释放（如 numpy、zlib、hashlib 的部分路径）。

## 2. 实测一：CPU 密集——多线程不仅没快，反而更慢

```python
"""cpu_bound.py：4 个线程各做 2 次纯计算。"""
import time
from threading import Thread

def cpu_task(n: int) -> int:
    """纯 Python 计算：GIL 全程被持有。"""
    total = 0
    for i in range(n):
        total += i * i
    return total

N = 20_000_000

# ---------- 串行 ----------
t0 = time.perf_counter()
for _ in range(4):
    cpu_task(N)
serial = time.perf_counter() - t0

# ---------- 多线程 ----------
t0 = time.perf_counter()
threads = [Thread(target=cpu_task, args=(N,)) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
mt = time.perf_counter() - t0

print(f"串行:   {serial:.2f}s")
print(f"多线程: {mt:.2f}s   ← 不是 4 倍加速，反而更慢！")
```

8 核机器上的典型输出：

```text
串行:   4.10s
多线程: 4.35s   ← 持平甚至更慢
```

解释：4 个线程**轮流**持有 GIL 执行字节码，总计算量不变，还叠加了线程切换与 GIL 争抢的开销（《Fluent Python》第 19 章给过同样的实验结论）。CPU 密集想用满多核，正解是**多进程**——每个进程有独立的解释器与 GIL：

```python
"""cpu_bound_mp.py：多进程版。"""
import time
from multiprocessing import Pool

def cpu_task(n: int) -> int:
    total = 0
    for i in range(n):
        total += i * i
    return total

if __name__ == "__main__":          # 多进程必须有的保护（08 篇解释 spawn 机制）
    N = 20_000_000
    t0 = time.perf_counter()
    for _ in range(4):
        cpu_task(N)
    print(f"串行:   {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    with Pool(processes=4) as pool:  # 4 个进程 ≈ 4 个独立 GIL
        pool.map(cpu_task, [N] * 4)
    print(f"多进程: {time.perf_counter()-t0:.2f}s")   # 典型 ≈ 1.2~1.4s，接近线性加速
```

**补充**：numpy/pandas 这类 C 扩展在大数组运算时**在 C 层释放 GIL**，所以"Python 写的 numpy 程序"多线程可能真有加速——判据是"热点在 Python 字节码还是 C 代码"（13 篇的 cProfile 能告诉你）。

## 3. 实测二：IO 密集——多线程真的有效

关键机制：线程发起阻塞式 IO（网络、磁盘、sleep）时**立即释放 GIL**，让其他线程运行；IO 完成后重新排队抢锁。等待时间占大头的负载，多线程能把"串行等待"变成"并行等待"：

```python
"""io_bound.py：模拟 5 个 100ms 的网络请求。"""
import time
from threading import Thread

def io_task(delay: float = 0.1) -> str:
    time.sleep(delay)                # sleep 会释放 GIL，等价于真实网络等待
    return "ok"

# ---------- 串行 ----------
t0 = time.perf_counter()
for _ in range(5):
    io_task()
print(f"串行:   {time.perf_counter()-t0:.2f}s")    # ≈ 0.50s（5 × 100ms）

# ---------- 多线程 ----------
t0 = time.perf_counter()
threads = [Thread(target=io_task) for _ in range(5)]
for t in threads: t.start()
for t in threads: t.join()
print(f"多线程: {time.perf_counter()-t0:.2f}s")    # ≈ 0.10s（等待重叠了！）
```

典型输出：

```text
串行:   0.50s
多线程: 0.10s   ← 5 倍加速，因为瓶颈是"等待"不是"计算"
```

**为什么 GIL 对多线程 IO 程序影响小**，现在可以完整回答：IO 线程的时间主要花在**不持有 GIL 的等待**上；GIL 只约束"执行字节码"那极小的时间片。100 个线程同时等网络，GIL 空闲着谁爱用谁用。所以传统 Python 服务（DB 查询、HTTP 调用为主）用线程池一直够用。真正的短板只在：**大量线程 + 每个 IO 之间仍有可观的 Python 代码**——线程切换与 GIL 争抢开始吃 CPU，且每线程内存栈开销（MB 级）限制了并发规模。这正是 asyncio 出场的地方（09/10 篇）：单线程事件循环，协程只占 KB 级内存。

## 4. 3.13 free-threading：如实介绍现状

**PEP 703（Free-threading CPython）**在 Python 3.13 中作为**实验性**特性落地：官方提供了"禁用 GIL"的自由线程构建（free-threaded build，`python3.13t`），核心工作是把引用计数改为**偏置引用计数（biased reference counting）**并配合内部细粒度锁与延迟引用计数（PEP 703 原文），让多线程真正并行执行字节码。3.14 中该特性进入**官方支持（Phase II）**阶段但仍非默认。截至本系列编写时（2026 年），需要如实记住的几点：

1. **默认构建仍有 GIL**——free-threading 是独立构建/安装选项（uv 可直接 `uv python install 3.13t` 安装试验，uv 系列 04 篇）；
2. **生态在追赶**：很多 C 扩展（含部分推理引擎）尚未声明兼容 free-threading；纯 Python 库大多天然可用；
3. **单线程性能有代价**：free-threaded 构建为线程安全引入开销，官方口径早期版本单线程有百分之几到十几的回归（3.14 已显著缩小，以 [py-free-threading 官方指南](https://py-free-threading.github.io/) 与 What's New 的最新数字为准）；
4. **它不解决 IO 并发编程模型问题**——线程的心智负担（锁、竞态）原样存在；asyncio 的结构化并发仍是高并发 IO 的首选写法。

**经验做法**：2026 年的今天，生产选型**默认仍按"GIL 存在"设计**（进程池/C 扩展/asyncio），free-threading 值得在 CI 里开实验矩阵跟踪兼容性，但不押注关键路径。

## 5. 并发选型决策树

三种工具的本质（官方文档：concurrency 概念散布在 threading/multiprocessing/asyncio 各文档页；《Fluent Python》第 19~21 章三连）：

| 工具 | 调度 | 适用 | 不适用 |
| --- | --- | --- | --- |
| `threading` | OS 抢占式 | 阻塞式库（requests/DB 驱动）的 IO 并发、需要共享内存 | CPU 密集（GIL）；上万并发（线程栈开销） |
| `multiprocessing` | 多进程 OS 调度 | CPU 密集、要绕开 GIL、崩溃隔离 | 高频小任务通信（pickle/IPC 开销大） |
| `asyncio` | 单线程协作式 | 万级并发 IO（Agent 工具并发、爬虫、网关） | CPU 密集（会卡住整个循环）；阻塞式库（必须换异步库或 to_thread） |

决策树（从上往下问，命中即停）：

```text
任务是什么？
├─ CPU 密集（纯 Python 计算）
│    ├─ 热点能换成 numpy/numba 等 C 层？→ 直接换库（顺便解决 GIL）
│    └─ 不能 → multiprocessing / ProcessPoolExecutor
├─ IO 密集
│    ├─ 并发量 < 几百 & 用的都是阻塞库 → threading / ThreadPoolExecutor
│    ├─ 高并发（千级+）& 可用异步库 → asyncio（09/10 篇）
│    └─ 混合 → asyncio + to_thread/run_in_executor（10 篇）
└─ 需要崩溃隔离/多核扩展服务 → 多进程（或干脆多实例 + 负载均衡）
```

**经验做法**：Agent 开发方向的默认答案是 **asyncio**（工具调用并发、流式响应、SSE 全是 IO 等待），CPU 段（ tokenize、重计算）外包给进程池或推理服务；**第一问永远是"热点在等还是在算"**——cProfile 跑一遍再选（13 篇），不要凭感觉。

## 常见陷阱

**陷阱一：给 CPU 密集任务上多线程还以为会加速**——实测第 2 节：不亏已是万幸。修复：进程池或换 C 扩展库。

**陷阱二：以为 GIL 让你不需要锁**——`counter += 1` 多字节码非原子，GIL 会在中途切换线程，数据照烂（08 篇银行账户实验复现）。

**陷阱三：把 `time.sleep` 放进 asyncio 协程**——阻塞整个事件循环，所有协程一起停摆；协程里要用 `await asyncio.sleep`（09 篇详解）。

**陷阱四：`multiprocessing` 脚本没有 `if __name__ == "__main__":` 保护**——Windows/macOS 默认 spawn 启动方式会重新 import 主模块，无保护时无限递归开进程，直接炸（08 篇详解 spawn/fork）。

**陷阱五：听说 free-threading 就重构全项目**——实验性构建 + 生态兼容窗口期，收益不确定；按第 5 节决策树选型才是当下正解。

## 本篇小结

- GIL 是 CPython 实现细节：**同一时刻一个线程执行字节码**；它不保证线程安全，也不阻塞 C 层释放（numpy/sleep/IO）；
- 实测结论：CPU 密集多线程无加速（甚至更慢），**多进程**才有；IO 密集多线程有效，因为**IO 等待时线程释放 GIL**；
- free-threading（PEP 703）：3.13 实验性、3.14 官方支持仍非默认；生产当下按 GIL 存在来设计，CI 跟踪兼容性即可；
- 选型一问定方向：**热点在等（IO）还是在算（CPU）**？等 → 线程/协程（看并发量），算 → 进程/C 扩展；
- Agent/推理服务方向：asyncio 是默认并发模型，CPU 段外包——08/09/10 篇把三条路逐一走通。

## 练习

1. 运行第 2 节 cpu_bound.py（N 可调小到 5_000_000），记录你机器上串行/多线程的耗时，并用 `sys.getswitchinterval()` 查看当前切换间隔；把 N 减小 10 倍后再跑，观察"多线程相对开销"占比变化并解释。
2. 把第 3 节 io_task 的 `time.sleep(0.1)` 换成纯计算 `sum(range(5_000_000))`，重跑对比：多线程还有加速吗？用 GIL 机制解释差异。
3. 写实验证明"`counter += 1` 在 GIL 下不原子"：开 4 线程各加 100_000 次，观察最终值 < 400_000（08 篇会给出加锁修复版，这里先复现问题）。
4. 检查你环境里是否有 free-threading 构建：`uv python list`（找 `3.13t`/`3.14t` 字样）；若有，用 `uv run --python 3.13t` 跑第 2 节的多线程脚本，对比普通构建的结果（差异应显著），并检查常用第三方库是否声明兼容（`pip index`/项目页）。
5. 对你手头的一个真实任务按第 5 节决策树走一遍：先回答"热点在等还是在算"（可用 13 篇的 cProfile 验证），写出选型结论与理由（线程/进程/协程/组合）。
