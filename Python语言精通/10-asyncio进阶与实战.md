# 10 - asyncio 进阶与实战

## 本篇目标

1. 掌握 asyncio 同步原语：`Lock`/`Semaphore`/`Queue` 与 08 篇线程版的对应关系；
2. 完成**并发爬虫实战**：限流 + 重试 + 超时，一个完整可运行、可扩展的程序；
3. 理解**异步生成器与 async with**（PEP 492/525），以及协程与线程/进程的混用边界（`to_thread`/`run_in_executor`）；
4. 用实测对比同步 vs 异步的吞吐差异，沉淀 asyncio 写法规范（经验做法）。

## 1. asyncio 同步原语：与线程版一一对应

事件循环是单线程，但协程间**仍需同步**——不是防"字节码被切"，而是防"await 点交错"：协程 A 在 `await queue.get()` 挂起期间，协程 B 完全可能改掉同一个 dict。规则（09 篇反例三的展开）：**跨 await 的共享状态 = 需要保护；线程原语一概不能用**（`threading.Lock` 会阻塞整个循环），要用 asyncio 版本：

| 线程版（08 篇） | asyncio 版 | 用途 |
| --- | --- | --- |
| `threading.Lock` | `asyncio.Lock` | 互斥：`async with lock:` |
| `threading.Semaphore(n)` | `asyncio.Semaphore(n)` | 限并发 n |
| `queue.Queue` | `asyncio.Queue` | 协程间传数据（`await put/get`） |
| `threading.Event` | `asyncio.Event` | 一次性通知（`await event.wait()`） |

```python
import asyncio

shared: dict[str, int] = {}
lock = asyncio.Lock()

async def incr(key: str) -> None:
    # 读改写之间有 await 才需要锁；纯同步段其实不会被打断
    async with lock:                       # 注意：async with（05 篇协议的异步版）
        current = shared.get(key, 0)
        await asyncio.sleep(0.001)         # 模拟一次 IO（比如先查库）——交错窗口出现
        shared[key] = current + 1

async def main() -> None:
    await asyncio.gather(*(incr("hits") for _ in range(100)))
    print(shared)                          # {'hits': 100} —— 没有锁会丢更新

asyncio.run(main())
```

**Semaphore 限流**是实战高频件——"同时最多 N 个在飞"，比"sleep 间隔"更贴近真实限流语义：

```python
async def bounded(job_id: int, sem: asyncio.Semaphore) -> None:
    async with sem:                        # 第 N+1 个协程在此挂起排队
        print(f"job {job_id} 开始")
        await asyncio.sleep(0.1)

asyncio.run(bounded_many())   # 20 个任务、sem=5 → 每 0.1s 完成一批，总耗时约 0.4s
```

`asyncio.Queue` 的生产者消费者把 08 篇的线程版平移到协程版，还多了一层优雅：`put`/`get` 挂起**零成本**（不占线程、不占 OS 资源），万级队列消费者也轻松。

## 2. 并发爬虫实战：限流 + 重试 + 超时

完整可运行（标准库 `urllib` 不异步，这里用 sleep 模拟网络层，结构可直接换成 `httpx.AsyncClient`）。集齐生产四件套：**并发上限（Semaphore）+ 每请求超时（timeout）+ 指数退避重试 + 结果收集**：

```python
"""async_crawler.py：并发限流爬虫骨架（标准库可跑）。"""
import asyncio
import random
import time

MAX_CONCURRENCY = 5          # 同时在飞的请求数
REQUEST_TIMEOUT = 1.0        # 单请求超时
RETRIES = 3                  # 最大重试次数

async def http_get(url: str) -> str:
    """模拟一次网络请求：随机延迟，30% 概率失败。换 httpx 时整个函数替换。"""
    await asyncio.sleep(random.uniform(0.05, 0.2))
    if random.random() < 0.3:
        raise ConnectionError(f"{url} 连接失败")
    return f"<html>{url}</html>"

async def fetch_with_retry(url: str, sem: asyncio.Semaphore) -> str:
    """单个 URL：限流 + 超时 + 指数退避重试。"""
    async with sem:                              # ① 并发上限
        for attempt in range(1, RETRIES + 1):
            try:
                # ② 超时：3.11+ 推荐 asyncio.timeout 上下文（09 篇）
                async with asyncio.timeout(REQUEST_TIMEOUT):
                    return await http_get(url)
            except (TimeoutError, ConnectionError) as exc:   # ③ 只接可重试异常
                if attempt == RETRIES:
                    return f"FAILED {url}: {exc!r}"          # 最终失败：记录不炸全局
                backoff = 0.1 * (2 ** (attempt - 1))         # ④ 指数退避 0.1/0.2/0.4
                print(f"  [retry {attempt}] {url}: {exc!r}, {backoff:.1f}s 后重试")
                await asyncio.sleep(backoff)                 # 让出（绝不能 time.sleep！）
        raise AssertionError("unreachable")

async def crawl(urls: list[str]) -> None:
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    t0 = time.perf_counter()
    # TaskGroup（09 篇）：结构化并发，异常不会丢
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fetch_with_retry(u, sem)) for u in urls]
    for t in tasks:
        print(t.result())
    print(f"\n{len(urls)} 个请求总耗时 {time.perf_counter()-t0:.2f}s")

if __name__ == "__main__":
    asyncio.run(crawl([f"https://example.com/page/{i}" for i in range(20)]))
```

典型输出（随机性每次不同）：

```text
  [retry 1] https://example.com/page/3: ConnectionError('... 连接失败'), 0.1s 后重试
  [retry 2] https://example.com/page/3: ConnectionError('... 连接失败'), 0.2s 后重试
<https://example.com/page/0>
...（20 个结果全部产出，失败者带 FAILED 前缀）
20 个请求总耗时 0.85s
```

**经验做法**：限流参数与重试策略放进配置而不是写死；对 API 还要加"全局速率限制"（令牌桶：`aiolimiter` 库的 `AsyncLimiter(10, 1)` 即"每秒 10 个"）；重试**只接可重试异常**（超时/连接错/5xx），4xx 逻辑错误重试没有意义。

## 3. 异步生成器与 async with：PEP 525/492

**异步迭代协议**（PEP 492）：`__aiter__`/`__anext__`（后者返回 awaitable，结束抛 `StopAsyncIteration`），配 `async for`。**异步生成器**（PEP 525，3.6+）：`async def` + `yield`，在 yield 点之间可以 await——典型场景是**流式数据**：

```python
import asyncio

async def tick(interval: float, n: int):
    """异步生成器：每 interval 秒产出一个序号（流式 SSE/心跳的原型）。"""
    for i in range(n):
        await asyncio.sleep(interval)      # 等待期间循环可以跑别的协程
        yield i                            # 异步产出

async def main() -> None:
    async for i in tick(0.1, 3):           # async for 驱动异步生成器
        print(f"tick {i}")

asyncio.run(main())
# tick 0
# tick 1
# tick 2
```

**异步上下文管理器**（PEP 492）：`__aenter__`/`__aexit__` 返回 awaitable，配 `async with`——异步资源的进入/退出本身可能要 await（建立连接、flush 缓冲）。异步生成器的资源清理要 `aclose()`（09 篇生成器清理的异步版），规范写法是 `async with contextlib.aclosing(gen):`（3.10+ contextlib 补齐了 aclosing）：

```python
import contextlib

async def main() -> None:
    async with contextlib.aclosing(tick(0.05, 100)) as stream:
        async for i, v in zip(range(3), stream):   # 提前退出：aclosing 保证 aclose 被调
            print(v)
    # 提前 break 时，未走完的清理（finally 里的 DB 连接关闭等）仍会执行
```

**经验做法**：异步生成器里的资源（连接、文件）同样用 try/finally 或 async with 包住（05 篇纪律的异步平移）；流式接口（LLM 的流式响应、SSE）几乎都是"异步生成器 + async for"，读框架源码时按这个形状识别。

## 4. 与线程/进程混用：to_thread 与 run_in_executor

asyncio 不是孤岛——现实系统里总有绕不开的同步库（requests、pandas、同步 DB 驱动）与 CPU 重活（09 篇反例）。解法是**把阻塞活外包给线程/进程池**，事件循环不等它：

```python
import asyncio
import concurrent.futures
import time

def blocking_io(url: str) -> str:
    """同步阻塞库（模拟 requests.get）。运行在工作线程里。"""
    time.sleep(0.2)
    return f"{url} ok"

def cpu_heavy(n: int) -> int:
    """CPU 密集：运行在子进程里（07 篇 GIL 结论）。"""
    return sum(i * i for i in range(n))

async def main() -> None:
    # ① 3.9+ 首选：to_thread（默认在线程池跑一个函数）
    r1 = await asyncio.to_thread(blocking_io, "https://a.com")

    # ② 底层接口：run_in_executor（可选进程池）
    loop = asyncio.get_running_loop()
    with concurrent.futures.ProcessPoolExecutor() as pool:
        r2 = await loop.run_in_executor(pool, cpu_heavy, 20_000_000)

    # ③ 阻塞活也能参与并发编排（和真协程一起 gather）
    r3, r4 = await asyncio.gather(
        asyncio.to_thread(blocking_io, "https://b.com"),
        asyncio.sleep(0.1, result="协程任务"),
    )
    print(r1, r2, r3, r4)

asyncio.run(main())
# https://a.com ok 2666666833333366666670000000 https://b.com ok 协程任务
```

三条边界规则（经验做法）：① **to_thread 池默认大小有限**（`min(32, cpu+4)`，同 08 篇），并发阻塞调用过多会排队——池大小当作容量规划的一部分；② **run_in_executor 的返回值/参数要可 pickle**（进程池时）；③ 外包出去的函数里**不要再 await**（它在线程里跑，不是协程）——线程函数与协程函数泾渭分明，混写必炸。

## 5. 性能对比实测：同步 vs 异步

同一批"任务"（80 个请求，每个等待 0.1s），三种写法横向对比（量级稳定，绝对值因机器而异）：

```python
"""bench_sync_async.py"""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

DELAY = 0.1
N = 80

def sync_one(i: int) -> int:
    time.sleep(DELAY)                     # 模拟阻塞请求
    return i

async def async_one(i: int) -> int:
    await asyncio.sleep(DELAY)
    return i

async def main() -> None:
    t0 = time.perf_counter()
    for i in range(N):
        sync_one(i)
    print(f"纯同步:        {time.perf_counter()-t0:.2f}s")   # 80 × 0.1 = 8.0s

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=32) as ex:
        list(ex.map(sync_one, range(N)))
    print(f"线程池(32):    {time.perf_counter()-t0:.2f}s")   # ≈ 0.3s（受限于 32 并发）

    t0 = time.perf_counter()
    await asyncio.gather(*(async_one(i) for i in range(N)))
    print(f"asyncio(80):   {time.perf_counter()-t0:.2f}s")   # ≈ 0.1s（80 个全并发）

asyncio.run(main())
```

典型输出：

```text
纯同步:        8.04s
线程池(32):    0.31s
asyncio(80):   0.10s
```

解读：并发度相当时两者接近（瓶颈都是"等待"），**差异在规模伸缩**——线程开 80 个就有可感知的栈内存与调度开销（万级不可行），协程 80 个、8000 个都是 KB 级挂起成本。**这正是 Agent 服务选 asyncio 的量化理由**：一个会话多轮工具调用、流式 token 转发，连接数大、每连接等得多。

## 常见陷阱

**陷阱一：限流用 `await asyncio.sleep(rate)` 想当然**——那是"每个请求后全队 sleep"，不是速率控制；用 Semaphore（并发上限）+ 令牌桶（速率上限）双管齐下。

**陷阱二：retry 里没有真正让出**——`except` 后 `time.sleep(backoff)`（09 篇反例），或重试循环里全同步；休眠必须 `await asyncio.sleep`。

**陷阱三：在 async 函数里调用 async 函数没 await**（09 篇陷阱一的进阶变体）——装饰器包装 async 函数时尤其常见（03 篇陷阱四），wrapper 必须 `async def` 并 `await func(*args, **kwargs)`。

**陷阱四：共享 aiohttp/ClientSession 未复用**——每个请求新建 Session（TCP 握手、TLS 全重跑），性能反不如同步；**一个事件循环一个 Client**，放进 `async with` 全程复用（httpx 同理）。

**陷阱五：把 `to_thread` 当万能药**——线程池默认 32 上限 + GIL 仍在（07 篇），CPU 重活要显式进程池；且注意 to_thread 的异常会正常传播回 await 点，别忘了接。

## 本篇小结

- 协程间同步用 asyncio 版原语（`async with lock`/`Semaphore`/`Queue`）；判据"**共享状态跨 await 吗**"；
- 实战骨架：Semaphore 限并发 + `asyncio.timeout` 超时 + 指数退避重试（只接可重试异常）+ TaskGroup 收口；
- `async for`/异步生成器（PEP 525）管流式，`async with`/`async with aclosing` 管异步资源清理；
- 混用边界：阻塞库 → `asyncio.to_thread`；CPU 重活 → `run_in_executor` + 进程池；外包函数里不得 await；
- 实测：并发 IO 场景 asyncio 以最小资源开销获得最大并发；Agent 服务（大量等待、大量连接）是它的主场。

## 练习

1. 把第 2 节爬虫的模拟 `http_get` 换成真实实现（标准库不异步，可用 `asyncio.to_thread` 包 `urllib.request.urlopen`），保持限流/重试/超时结构不变，对 10 个 URL 实测并记录成功率与总耗时。
2. 给爬虫加"全局速率限制"：实现一个简易 `AsyncLimiter(rate, per)`（令牌桶：内部 `asyncio.Lock` + 时间窗计数），并验证 1 秒内实际放行的请求数 ≤ rate。
3. 写异步生成器 `tail(path)`：模拟 `tail -f`——每 0.2s 检查文件新增行并 async yield（写入测试用多一个协程往文件追加）；用 `aclosing` 保证退出时打印"监听结束"。
4. 用 `asyncio.Queue` 重写第 2 节爬虫为"生产者-消费者"形态：1 个生产者投递 URL、5 个消费者协程处理；对比 Semaphore 版的限流语义差异（并发上限 vs 消费速度）。
5. 做 09 篇练习 5 的对答案题：用本篇 `to_thread` 写出"10 个工具并发调用（其中 3 个是阻塞库）"的完整可运行骨架，并给每个调用独立超时与失败隔离（一个失败不影响其他）。
