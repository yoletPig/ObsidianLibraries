# 09 - asyncio 基础

## 本篇目标

1. 理解**事件循环**的原理：单线程协作式调度，"谁让出控制权谁倒霉（也爽）"的运行模型；
2. 建立 **async/await 心智模型**：协程对象是什么、`await` 时发生什么、什么时候让出控制权；
3. 掌握组合器：`gather`/`wait`/`wait_for`，以及 3.11+ 的 `TaskGroup`；
4. 学会**超时与取消**的规范处理（`CancelledException` 的礼仪），识别死锁/饿死反例。

## 1. 事件循环：单线程协作式调度

asyncio 的模型一句话：**一个线程 + 一个事件循环（event loop）+ 一堆可暂停的协程**。事件循环就是一个 `while True`：不断问"哪个协程的 IO 就绪了/定时器到点了"，把控制权交给它，它跑到下一个 `await` 又还回来（官方文档：asyncio → Event Loop；与 04 篇生成器的暂停-恢复机制同源）。三个关键词：

- **单线程**：同一时刻只有一个协程在执行字节码——**没有竞态**（不需要 08 篇的锁来防"读改写"被切断，协程不会在任意两条指令间被切走）；单线程也意味着**一处阻塞，全局卡死**；
- **协作式**：切换只发生在协程**主动 `await`** 的时刻（OS 线程是抢占式，随时可能被切）。你不去 await，别人永远跑不上——这就是"协作"：让出权在你手里；
- **不适合 CPU 密集**：一个协程算 10 秒数学，整个循环卡 10 秒，所有连接全部超时（07 篇决策树的结论在这里落地）。

事件循环轮子示意（帮助理解，不必自己实现）：

```text
事件循环
  ┌────────────────────────────────────────────┐
  │ while True:                                │
  │   tasks = 挑出"就绪"的回调(定时器到期/IO完成)│
  │   for cb in tasks: cb()      ← 协程恢复执行│
  │   等 epoll/select 说"哪些 fd 就绪了"        │
  │   把"等待这些 fd"的协程标记为就绪           │
  └────────────────────────────────────────────┘
```

## 2. async/await 心智模型：什么时候让出控制权

`async def` 定义**协程函数**，调用它得到**协程对象**（类比 04 篇：调用生成器函数得到生成器——**不执行函数体**！）。协程对象要被"驱动"才运行：`await coro` 或包成 Task 交给事件循环：

```python
import asyncio

async def fetch(name: str, delay: float) -> str:
    print(f"  {name}: 开始")
    await asyncio.sleep(delay)      # ★ 让出控制权：事件循环去跑别的协程
    print(f"  {name}: 结束")
    return f"{name}-data"

async def main() -> None:
    coro = fetch("A", 0.5)
    print(type(coro))               # <class 'coroutine'>：此刻函数体一行都没执行！
    result = await coro             # await 才驱动它：跑直到 return
    print(result)

asyncio.run(main())                 # 程序入口：创建事件循环并运行 main 直到完成
```

输出：

```text
<class 'coroutine'>
  A: 开始
  A: 结束
A-data
```

**心智模型：`await` = "我要等一件事，等的时候把 CPU 让给别的协程"**。哪些操作真的让出控制权？

- ✅ **让出**：`await asyncio.sleep()`、异步库的 IO（`httpx.AsyncClient`、`asyncpg`、`aiofiles`）、`await` 其他协程/Task/`asyncio.Queue` 操作；
- ❌ **不让出**：普通函数调用、CPU 计算、**同步阻塞调用（`time.sleep`、`requests.get`、`open().read()`）**——最后这类是 asyncio 第一大事故源：循环被你一个人独占，所有协程停摆。

"sleep vs 真实 IO"的对照实验——把 3 个 1 秒等待并发起来：

```python
import asyncio
import time

async def io_like(seconds: float) -> None:
    await asyncio.sleep(seconds)      # 正确的"等待"

async def main() -> None:
    t0 = time.perf_counter()
    await asyncio.gather(io_like(1), io_like(1), io_like(1))   # 三个并发
    print(f"耗时 {time.perf_counter()-t0:.2f}s")

asyncio.run(main())
# 耗时 1.00s   ← 三个等待重叠（对比串行 await 三个 = 3.0s）
```

同理真实网络：`await client.get(url)` 在等响应包期间让出循环，其他请求的网络等待同期进行——**并发省的是"等待"不是"计算"**（与 07 篇 IO/CPU 的二分完全一致）。

## 3. Task 与组合器：gather / wait / wait_for / TaskGroup

**Task = 被事件循环接管的协程**。`asyncio.create_task(coro)` 立刻把它排入循环"尽快运行"，返回 Task 对象（可 await 取结果）。组合器负责"多协程编排"：

```python
import asyncio

async def job(name: str, delay: float, fail: bool = False) -> str:
    await asyncio.sleep(delay)
    if fail:
        raise RuntimeError(f"{name} 失败")
    return f"{name} 完成"

async def main() -> None:
    # ---------- gather：全部完成，按"提交顺序"返回结果列表 ----------
    results = await asyncio.gather(
        job("A", 0.2),
        job("B", 0.1),
    )
    print(results)                          # ['A 完成', 'B 完成']（顺序=A,B 与耗时无关）

    # gather 的异常语义：默认一个失败 → 立即取消其余并抛出（return_exceptions=True 收集为值）
    results = await asyncio.gather(
        job("A", 0.2, fail=True),
        job("B", 0.1),
        return_exceptions=True,             # 失败变成返回值里的异常对象
    )
    print(results)                          # [RuntimeError('A 失败'), 'B 完成']

    # ---------- wait_for：给单个协程设超时 ----------
    try:
        await asyncio.wait_for(job("C", 5.0), timeout=0.3)
    except TimeoutError:                    # 3.11+ 超时抛 TimeoutError（内层协程被取消）
        print("C 超时")

    # ---------- TaskGroup（3.11+，推荐）：结构化并发 ----------
    async with asyncio.TaskGroup() as tg:   # 任一子任务失败：取消其余，退出时抛 ExceptionGroup
        t1 = tg.create_task(job("D", 0.2))
        t2 = tg.create_task(job("E", 0.1))
    print(t1.result(), t2.result())

asyncio.run(main())
```

`asyncio.wait` 更底层（等"满足条件的集合"，`FIRST_COMPLETED`/`FIRST_EXCEPTION`/`ALL_COMPLETED`），适合"先到先得"类编排；日常优先 `gather`/`TaskGroup`。**经验做法**：3.11+ 新代码用 `TaskGroup`——"结构化并发"保证子任务不会脱离作用域泄漏（失败自动取消兄弟、退出 with 时全部收干净），这是 3.11 官方 What's New 强调的核心改进。

**late binding 在 create_task 的变体**（02/04 篇知识的并发版）：

```python
for url in urls:
    asyncio.create_task(fetch(url))    # ⚠️ 危险：Task 引用没保存 → 可能被 GC（01 篇）！
# 正解：
tasks = [asyncio.create_task(fetch(u)) for u in urls]
await asyncio.gather(*tasks)
```

## 4. 超时与取消：CancelledException 处理规范

取消的机制：`task.cancel()` → 事件循环在该协程**下一个 await 点**抛入 `asyncio.CancelledError`。规范（官方文档 asyncio → Cancellation）：

1. **`CancelledError` 继承自 `BaseException`**（3.8 起）——`except Exception` **接不住它**，这是刻意的：取消不是普通错误，必须向上传播；
2. 清理代码放 `try/finally`（05 篇纪律原样适用）；若必须捕获，**处理完要重新 raise**；
3. 3.11+ 优雅整体超时用 `asyncio.timeout()` 上下文管理器（比 wait_for 覆盖范围更完整）：

```python
import asyncio

async def worker() -> None:
    try:
        print("工作 10s ...")
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        print("被取消：保存进度/释放资源")
        raise                       # ★ 礼仪：接住也要 re-raise，否则"假复活"破坏取消语义
    finally:
        print("清理（无论取消与否都执行）")

async def main() -> None:
    t = asyncio.create_task(worker())
    await asyncio.sleep(0.1)
    t.cancel()                      # 请求取消
    try:
        await t                     # 在这里收到 CancelledError 传播
    except asyncio.CancelledError:
        print("任务确认已取消")

    # 3.11+：timeout 上下文 = 对整块代码生效
    try:
        async with asyncio.timeout(0.5):
            await asyncio.sleep(2)
    except TimeoutError:
        print("0.5s 超时")

asyncio.run(main())
```

输出：

```text
工作 10s ...
被取消：保存进度/释放资源
清理（无论取消与否都执行）
任务确认已取消
0.5s 超时
```

**反例（吞取消）**——`except BaseException: pass` 或 `except asyncio.CancelledError: return` 后不 re-raise：Task 对外表现为"正常完成"，调用方以为成功实际半途而废，且 `TaskGroup` 的级联取消被破坏。**经验做法**：只在"确知如何优雅收尾"的层（连接关闭、事务回滚）捕获 CancelledError，并且**必 re-raise**；想让取消"稍等一下"，用 `asyncio.shield(protected())` 保护关键段（但 shield 外仍会被取消）。

## 5. 死锁与饿死：三个反例

**反例一：同步阻塞卡死全局**（asyncio 第一事故）：

```python
async def handler() -> None:
    data = requests.get("https://api.example.com", timeout=5)   # ❌ 阻塞库！
    # 整个事件循环在这 0.5~5s 内冻结：其他所有协程、心跳、超时全部停摆
# 正解：httpx.AsyncClient / aiohttp；或 await asyncio.to_thread(requests.get, ...)（10 篇）
```

**反例二：CPU 重活饿死别人**：

```python
async def crunch() -> None:
    total = sum(i * i for i in range(50_000_000))   # ❌ 3~5 秒不让出：饿死全部并发
# 正解：await asyncio.to_thread(...) 或进程池（10 篇）；或算一段 await asyncio.sleep(0) 让一手
```

**反例三：协程里用同步锁/同步队列**——`threading.Lock.acquire()` 阻塞的是整个线程（即整个循环），不是当前协程；跨协程同步必须用 `asyncio.Lock`/`asyncio.Queue`（10 篇详解）。另一个隐蔽变体：在协程里 `time.sleep` 当"限速器"——限住的是所有协程。

**经验做法**：asyncio 代码的评审第一问："**这段代码里有没有同步阻塞调用？**"（requests、time.sleep、pandas 大计算、同步 DB 驱动）；第二问："**每个 await 的等待期，循环在干什么？**"——答案应该总是"在跑别的协程"。可用 `loop.set_debug(True)` 或 `PYTHONASYNCIODEBUG=1` 让解释器对"超过 100ms 未让出的回调"发警告（官方调试模式）。

## 常见陷阱

**陷阱一：忘记 await**——`fetch("A", 1)` 裸调用只创建协程对象，得到 `RuntimeWarning: coroutine was never awaited`，函数体从未执行；`create_task` 后的 Task 也要么 await、要么保存引用。

**陷阱二：在协程里调阻塞函数**（第 5 节反例一/二）——症状是"偶发全员卡顿、超时连锁雪崩"；用调试模式定位。

**陷阱三：吞掉 CancelledError**（第 4 节）——`except Exception` 接不住取消（BaseException），但 `try/finally` 之外的"清理后 return"会破坏取消；re-raise 是礼仪。

**陷阱四：gather 异常语义想当然**——默认第一个异常抛出时**其余任务被取消**（不是等全部完成再报）；想要"收集所有结果（含异常）"必须显式 `return_exceptions=True`，否则局部失败连累全局。

**陷阱五：裸跑协程的入口**——`asyncio.run(main())` 才是标准入口（它负责建循环、清理异步生成器等收尾）；自己 `loop = asyncio.get_event_loop()` 手工管理在脚本场景已不推荐（官方文档明确 asyncio.run 是推荐入口）。

## 本篇小结

- 事件循环 = 单线程 while True 分发就绪协程：**协作式**切换只发生在 await 点，一处阻塞全局卡死；
- `async def` 调用得协程对象（**不执行**），`await`/`create_task` 才驱动；await 让出的是"等待期"，省的是等待不是计算；
- 编排：`gather`（顺序齐整/异常语义要记牢）、`wait_for`+`asyncio.timeout`（超时）、`TaskGroup`（3.11+ 结构化并发，推荐）；
- 取消礼仪：`CancelledError` 是 BaseException，**接住必须 re-raise**，清理放 finally；`shield` 护关键段；
- 三大反例：阻塞库、CPU 重活、同步锁——都会冻结整个循环；评审口诀"有没有同步阻塞？await 期间循环在干嘛？"

## 练习

1. 写协程 `fetch`（sleep 模拟延迟并返回数据），分别用（a）顺序 await 三次（b）gather 三个（c）TaskGroup 三个跑延迟为 0.5/0.3/0.2 的任务，记录三种总耗时并解释差异。
2. 构造"漏 await"事故：调用 `asyncio.run(main())` 而 main 里裸调协程不 await，观察 RuntimeWarning；然后修复为 create_task + gather，并故意不保存 Task 引用，用 `gc.collect()` 前后对比验证 Task 可能被回收（01 篇引用计数）。
3. 实现带超时的"先到先得"：两个数据源（一个 0.2s 返回旧数据、一个 1s 返回新数据），用 `asyncio.wait` 的 `FIRST_COMPLETED` 拿到先到的，取消另一个；打印总耗时。
4. 写一个"吞取消"的反例 Task（except CancelledError 后 return），在 TaskGroup 中触发取消，观察与"规范 re-raise 版"的行为差异（提示：ExceptionGroup 是否抛出、兄弟任务是否被取消）。
5. 用 `loop.set_debug(True)`（或 `PYTHONASYNCIODEBUG=1`）复现"协程里 time.sleep(0.2)"的警告日志，然后把 sleep 换成 `await asyncio.sleep(0)` 分段让出，验证警告消失。
