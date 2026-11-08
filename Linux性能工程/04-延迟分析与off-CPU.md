# 04 - 延迟分析：off-CPU 与"CPU 不高但很慢"

## 本篇目标

- 分清 on-CPU 分析与 off-CPU 分析的适用场景：什么时候看"CPU 上在跑什么"，什么时候看"不在 CPU 上时在等谁"。
- 理解 off-CPU 分析的实现原理（sched_switch tracepoint），会用 perf sched 和 BCC offcputime（或等价 bpftrace 思路）量出等待分布。
- 会用 ftrace 的最小操作（events 使能、trace_on、读 trace 管道）看单个事件的现场。
- 完成一个"CPU 不高但很慢"的完整案例：同步 IO 等待的服务，从症状到修复的全过程。

前置：03 篇（perf 与火焰图）；操作系统系列 03（CPU 调度，理解 runqueue）。本篇与 03 篇合起来覆盖 CPU 时间的全部：在 CPU 上的（03）+ 等 CPU 的（本篇调度延迟）+ 等非 CPU 资源的（本篇 off-CPU）。

## 1. 时间都去哪了：请求耗时的三段论

一个请求的墙钟时间（wall time），只有一小部分花在 CPU 上：

```
一个请求的生命周期（墙钟 100ms 示意）
┌──────────────┬───────────────────────┬─────────────┐
│  on-CPU 15ms  │      off-CPU 80ms      │  其他 5ms    │
│  真的在算：    │  等资源被唤醒前：       │  进程被换出   │
│  解释器/计算  │  磁盘 IO / 网络 RTT     │  等自己被调度 │
│  序列化/加解密│  锁/信号量 / sleep      │  (调度延迟)   │
└──────────────┴───────────────────────┴─────────────┘
      03 篇的领地    ←──── 本篇的领地 ────→   runqlat
```

于是排查有了分诊规则（Brendan Gregg, "Off-CPU Analysis", brendangregg.com；亦见《Systems Performance》第 5 章）：

- **CPU 利用率高 + 慢** → on-CPU 分析（03 篇：火焰图找热点）。
- **CPU 利用率低 + 慢** → off-CPU 分析（本篇）：时间花在等待上，perf 的 CPU 火焰图会显示一切正常——因为你只拍了"在 CPU 上"的照片。
- **忽快忽慢、抖动大** → 两边都要看，还要看调度延迟（runqlat）和 01 篇的分布思维。

**经验做法**：拿到"服务慢"的工单，第一条命令永远是 `mpstat 1 5` 或 `top`——先确定 CPU 是不是当事者，再决定去 03 篇还是本篇。

## 2. off-CPU 分析的原理：在调度器门口打卡

off-CPU 分析的核心动作只有一个：**在 sched_switch tracepoint 上，记录线程被换下 CPU 的时刻与原因，等它被换回时结算等待时长**。线程离开 CPU 的路径有三种（内核 `sched_switch` 的 `prev_state`）：

- `prev_state == 0`（TASK_RUNNING）：只是被抢占，还在运行队列里等 CPU → 这是"等 CPU"，属于调度延迟，runqlat 的统计对象。
- `prev_state == 1`（TASK_INTERRUPTIBLE）/ `2`（TASK_UNINTERRUPTIBLE）：主动睡（等锁、等 IO、等网络包）→ off-CPU 分析的主体。D 状态（2）多半在等不可中断的资源如磁盘。
- 被信号杀死/终止等其他状态，排查时一般忽略。

```
线程时间线与 sched_switch 打卡点
        运行        等 IO（off-CPU）        运行
   ────┤A├─────────╱                    ╲──────┤B├──────►
       ▲ sched_switch(prev_state=2)          ▲ sched_switch(next_pid=我)
       │ 记录 t_out=now, 栈=我此刻在哪等待      │ 结算 Δt=now-t_out
                    └──────── Δt = off-CPU 时长 ────────┘
```

把每次 Δt 按调用栈/线程聚合，就得到 off-CPU 火焰图：宽塔 = 等得最久的路径。BCC 的 `offcputime` 工具（`/usr/share/bcc/tools/offcputime`）实现了这个逻辑；原理上它就是 08 篇 eBPF 技术的一个应用。等价 bpftrace 脚本（教学简化版，忽略唤醒栈与精度细节）：

```
# offcpu.bt —— bpftrace 版 off-CPU 直方图（按线程聚合）
tracepoint:sched:sched_switch
/args->prev_state != 0/                     # 只统计阻塞型换出（含 D 状态），不算被抢占
{
    @out_ts[args->prev_pid] = nsecs;        # 打卡：记录换出时刻
    @out_comm[args->prev_pid] = args->prev_comm;
}

tracepoint:sched:sched_switch
/@out_ts[args->next_pid]/
{
    @wait_us[@out_comm[args->next_pid]] =
        hist((nsecs - @out_ts[args->next_pid]) / 1000);
    delete(@out_ts[args->next_pid]);
}
```

字段名 `args->prev_state/prev_pid/next_pid/prev_comm` 来自 tracepoint 格式（`cat /sys/kernel/tracing/events/sched/sched_switch/format` 可验证），不同 bpftrace 版本写法略有差异，`bpftrace -l 'tracepoint:sched:sched_switch' -v` 查看。

off-CPU 分析的两个盲区（《Systems Performance》5.3 节有述，**经验做法**补充）：

1. **只看被阻塞线程自己的栈，看不到"谁该负责"**：等锁的栈只显示 pthread_cond_wait，想找到持锁者需要 waker 分析（记录唤醒线程的栈，BCC offcputime 的 `-w` 选项）。
2. **采样间隔与短等待**：极短等待（<10μs）海量发生时，记录本身开销不可忽略，通常过滤掉短等待（工具默认阈值）。

## 3. perf sched：官方的调度视角

不装任何额外工具，perf 就能做调度分析：

```
$ sudo perf sched record -- sleep 10
$ sudo perf sched latency --sort max

 -----------------------------------------------------------------------------------------------------------------
  Task                  |   Runtime ms  | Switches | Average delay ms | Maximum delay ms | Maximum delay at       |
 -----------------------------------------------------------------------------------------------------------------
  python3:(10)          |    812.033    |      324 | avg   0.512 ms | max   8.412 ms | max at 10:31:22.410532 s
  kswapd0:0             |      4.102    |        4 | avg   0.101 ms | max   1.203 ms | max at ...
 -----------------------------------------------------------------------------------------------------------------
```

`perf sched latency` 的 Average/Maximum delay 就是**调度延迟**：线程从"可运行"到"真的上 CPU"等了多久（对应 03 篇说的 runqueue 等待，OSTEP 第 7 章讲过的"CPU 就绪队列"）。`perf sched timehist`（较新版本）则逐次切换列出 wait time / run time，是最细的时间线。

调度延迟什么时候重要？**CPU 饱和的机器上**：8 个贪吃线程挤 4 个核，每个线程的调度延迟会显著抬升 P99（对延迟敏感型服务尤其致命——即使单请求计算量很小）。诊断信号：`vmstat 1` 的 `r` 列长期 ≥ 核数、`perf sched latency` 的 avg delay 从 0.1ms 级涨到毫秒级。**经验做法**：把"调度延迟 P99"当作和"CPU 利用率"平级的容量指标，前者饱和比后者到 100% 更早报警。

更轻量的实时观察（BCC runqlat / bpftrace 一行版，08 篇剖析原理）：

```
$ sudo /usr/share/bcc/tools/runqlat 1
     usecs               : count     distribution
         0 -> 1          : 1420     |****************                        |
         2 -> 3          : 3210     |****************************************|
         4 -> 7          : 1802     |**********************                  |
      ...
      4096 -> 8191       : 12       |                                        |
```

## 4. ftrace 最小用法：看单个事件的现场

ftrace 是内核自带的追踪框架（man ftrace；内核文档 `Documentation/trace/ftrace.rst`），通过 tracefs 文件操作。四步看"read 系统调用在内核里跑了多久"：

```
$ cd /sys/kernel/tracing                    # 老内核在 /sys/kernel/debug/tracing
$ echo 0 > tracing_on                       # 先关，配置好再开
$ echo 1 > events/syscalls/sys_exit_read/enable
$ echo 1 > tracing_on                       # 开始记录
$ cat /etc/hostname > /dev/null             # 制造一点流量
$ echo 0 > tracing_on                       # 停止
$ cat trace | head -5
# tracer: nop
#
# entries-in-buffer/entries-written: 37/37   #P:16
#
#                                _-----=> irqs-off/BH-disabled
#                               / _----=> need-resched
#                              | / _---=> hardirq/softirq
#                              || / _--=> preempt-depth
#                              ||| /     delay
#           TASK-PID     CPU#  ||||    TIMESTAMP   FUNCTION
#              | |         |   ||||       |             |
              cat-49152   [003] .....  1234.567890: sys_read -> 0x0
```

更常用的还有函数级追踪（`echo function > current_tracer` 看内核函数调用流）和 `echo function_graph > current_tracer`（带耗时的调用图）。**经验做法**：ftrace 适合"临时、单事件、要看原始现场"的场景；要做统计聚合（直方图/频率表）直接上 bpftrace（08 篇），别用 ftrace 的原始输出硬数。**WSL2 注**：tracefs 存在，但个别事件目录为空属正常（内核配置裁剪）；找不到 `/sys/kernel/tracing` 时先 `sudo mount -t tracefs none /sys/kernel/tracing`。

## 5. 动手实验：一个"CPU 不高但很慢"的服务

症状设定：一个"日志归档"服务，每处理一条记录要确认落盘（模拟同步 fsync 的反模式）。CPU 几乎闲着，但吞吐惨不忍睹。先写这个"有问题的程序"：

```python
# slow_writer.py —— 每条记录 open+write+fsync：CPU 不忙，但极慢
import os, time, sys

DATA = b"x" * 1024                        # 1KB 的"一条日志"

def process_records(path, n):
    for i in range(n):
        t0 = time.perf_counter()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        os.write(fd, DATA)
        os.fsync(fd)                      # ← 反模式：每条都强制落盘
        os.close(fd)
        if i % 100 == 0:
            dt = (time.perf_counter() - t0) * 1000
            print(f"rec {i:5d}: {dt:6.2f} ms", file=sys.stderr)

if __name__ == "__main__":
    process_records("/tmp/slow.log", int(sys.argv[1]) if len(sys.argv) > 1 else 200)
```

**第一步：症状确认。** 先看 CPU 是不是当事者：

```
$ python3 slow_writer.py 200 2>&1 | tail -3
rec   198:  11.32 ms
rec   199:  10.98 ms
$ mpstat -P ALL 1 3 | tail -2             # CPU 几乎空闲 → on-CPU 火焰图注定一无所获
all    1.20    0.00    0.60    0.30    0.00    0.10    0.00    0.00   97.80
```

CPU 97.8% idle，但每条记录 11ms——典型 off-CPU 问题，进入本篇流程。

**第二步：观测等待。** 用 `pidstat` 确认 IO 等待，用 iostat 看设备层面：

```
$ pidstat -d 1 2 -p $(pgrep -f slow_writer)
02:10:01 PM   UID       PID   kB_rd/s   kB_wr/s kB_ccwr/s iodelay  Command
02:10:02 PM  1000     50112      0.00   1024.00      0.00     810  python3

$ iostat -x 1 2 | grep -E 'Device|nvme'
Device  r/s  w/s   rkB/s  wkB/s ... w_await aqu-sz ... %util
nvme0n1 0.0 38.0    0.0  1024.0 ...   10.80   0.41 ...  3.60
```

`iodelay`（pidstat 的 IO 等待毫秒计数）与 `w_await≈10.8ms` 指向：每次写都在等存储。再看系统级证据链：

```
$ vmstat 1 3                              # b 列（阻塞在 IO 的任务）间歇出现
 r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st
 1  1      0 ...                     0    0     0  1200 4300 9000  1  1 92  6  0
```

**第三步：抓现场（谁在等什么）。** 用 ftrace 看 fsync 路径（或 08 篇的 BCC fileslower / offcputime）：

```
$ cd /sys/kernel/tracing
$ echo 0 > tracing_on
$ echo 1 > events/syscalls/sys_exit_fsync/enable
$ echo 1 > tracing_on
$ python3 slow_writer.py 20 2>/dev/null
$ echo 0 > tracing_on
$ grep -v '^#' trace | head -3
     python3-50231 [005] ..... 1301.223451: sys_fsync -> 0x0
```

要量化"每次 fsync 多久"，用 strace 的统计模式（小负载下可用，01 篇的开销警告在此不适用）：

```
$ strace -c -e trace=fsync python3 slow_writer.py 50 2>&1 | tail -8
% time     seconds  usecs/call    calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 99.87    0.612345       12246        50           fsync
  0.13    0.000821          16        50           write
------ ----------- ----------- --------- --------- ----------------
```

证据闭环：**50 次 fsync 占了 99.87% 的时间，平均每次 12ms**——每条记录 11ms 的账全在这。这就是 off-CPU 火焰图会显示的塔：`process_records → os.fsync → ext4_sync_file → jbd2 ...`（宽度 99%）。

**第四步：修复。** 批量写 + 单次 fsync（吞吐换延迟的取舍显式化）：

```python
def process_records_fixed(path, n, batch=64):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    buf = []
    for i in range(n):
        buf.append(DATA)
        if len(buf) >= batch or i == n - 1:
            os.write(fd, b"".join(buf))
            os.fsync(fd)                  # 每批确认一次，而不是每条
            buf.clear()
    os.close(fd)
```

**第五步：同口径对比。**

| 版本 | 200 条耗时 | 平均每条 | fsync 次数 | 每次耗时 |
| --- | --- | --- | --- | --- |
| 逐条 fsync（原） | 2.24s | 11.2ms | 200 | ~12ms |
| 批量 64 条（修复） | 0.041s | 0.21ms | 4 | ~12ms |

fsync 单次耗时没变（存储的物理特性），**变的是次数**——53 倍吞吐提升来自把"等"的频率降为 1/53。这个案例的普适结论：**off-CPU 问题优化掉的往往不是单次等待时长，而是等待次数**（批量、合并、连接复用、池化都是这一句话的变体）。

## 常见误区

1. **"CPU 空闲就没瓶颈"**：CPU 只是六类资源之一（01 篇 USE 表）。本案例里瓶颈是盘，CPU 空闲恰恰是"等待型瓶颈"的指纹。
2. **"iowait 高 = 盘慢"**：wa 只说明"有 CPU 在等 IO"，可能是盘慢，也可能是"本来不该同步等盘的代码在等盘"（本案例）。修复代码比换盘便宜。
3. **"off-CPU 火焰图 = 等待火焰图直接改"**：等 IO 的塔底是内核路径，根因常在应用层的调用频率（本案例）或锁顺序。看到内核塔，往上找应用塔的调用者。
4. **"调度延迟不用看"**：CPU 满载时它直接影响 P99；runqlat 是容量预警指标（第 3 节），比"load 高"精确得多。
5. **"ftrace 输出逐行看就够"**：原始 trace 是时间线不是统计，人眼数不出分布；聚合用 bpftrace/perf，ftrace 留给"看单次现场"。

## 本篇小结

- 墙钟时间 = on-CPU + off-CPU（等资源）+ 调度延迟（等 CPU）；CPU 空闲还慢，直接进 off-CPU 流程，别在 03 篇的火焰图里白费劲。
- off-CPU 分析的机制：sched_switch 上对阻塞型换出（prev_state != 0）打卡，换回时结算；BCC offcputime/bpftrace 均可实现；盲区是看不到持锁方，需要 waker 分析。
- perf sched latency 与 runqlat 量化调度延迟；它是 CPU 饱和的早期预警，r 列持续 ≥ 核数时必查。
- ftrace 四步法（enable 事件 → tracing_on → 制造流量 → 读 trace）看单事件现场；统计聚合交给 perf/bpftrace。
- 案例：逐条 fsync 的"CPU 不高但很慢"，strace -c 一锤定音；批量合并把 200 次 12ms 等待变 4 次，吞吐 53 倍——off-CPU 优化的本质是减少等待次数。

## 练习

1. 跑 slow_writer.py，同时开三个终端分别跑 `mpstat 1`、`iostat -x 1`、`vmstat 1`，截取同一时刻三行输出，写一段话把 wa、w_await、b 三个数字串成一条因果链。
2. 把 batch 参数调成 8/64/512 各跑一次，记录"200 条总耗时 + 每批 fsync 单次耗时"，画出"批量大小 vs 吞吐"曲线，解释为什么收益递减、过大批次的代价（内存、单次 fsync 变长）是什么。
3. 若你的环境可用 BCC：跑 `sudo /usr/share/bcc/tools/offcputime -K -p $(pgrep -f slow_writer)`，找到最宽的内核栈；不可用则用第 4 节的 ftrace 步骤抓 `sys_exit_fsync` 时间戳序列，手算相邻差值的平均值。
4. 用 `perf sched record -- sleep 5` 加上"同时跑 2 个、8 个 burn_cpu 死循环"（03 篇 sick_machine.py 可复用）两种负载，对比 `perf sched latency` 的 Average delay，验证"调度延迟随 CPU 饱和上升"。
5. 你的服务每请求要查一次 Redis（RTT 0.5ms）、一次 MySQL（RTT 2ms）。在 N 个请求的批处理里逐个查 vs 合并成 pipeline/multi-get，用 Little's Law（01 篇）估算批量后的等待次数下降比例，并在本地用 redis-py 或伪实现验证。
