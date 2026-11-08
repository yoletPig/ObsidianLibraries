# 08 - eBPF 入门

## 本篇目标

- 说清 eBPF 为什么是观测范式的革命：内核态安全可编程、无需改内核/重启，以及 verifier/JIT/辅助函数的分工。
- 分清 BCC、bpftrace、libbpf/CO-RE 三条技术路线的适用场景。
- 十分钟上手 bpftrace：kprobe、tracepoint、uprobe 各跑通一个实例，掌握 map/聚合/直方图三件套。
- 剖析经典 one-liner 的构造，并亲手写一个"统计系统调用延迟"的完整脚本。

前置：02 篇（观测层次——本篇就是 L4/L5 层的可编程化）；04 篇（off-CPU 的 bpftrace 实现已经在等你）。本篇在 WSL2 (kernel 6.6) 上完全可跑，是受限环境下补齐观测能力的主力武器。

## 1. 为什么 eBPF 是革命：把小程序安全地放进内核

02 篇的观测分层里，L4（tracepoint）以下都需要"在内核里执行代码"。eBPF（extended Berkeley Packet Filter）的突破在于：**允许你在内核态运行自己写的程序，且安全（verifier 验证）、高效（JIT 编译）、无需修改内核源码或加载模块**。

eBPF 程序的执行模型（《Systems Performance》第 15 章 tracing；bpftrace/BCC 官方文档同述）：

```
你的程序(受限 C / bpftrace DSL)
   │ 编译成 eBPF 字节码
   ▼
┌─ 内核 ──────────────────────────────────────────────┐
│ ① verifier：静态检查（无越界、无死循环、边界可达性）    │  ← 安全性的来源
│ ② JIT：字节码 → 本机指令（x86/ARM），性能≈原生代码      │  ← 高效的来源
│ ③ 挂载点(kprobe/tracepoint/uprobe/...) 触发执行        │
│ ④ 与用户态通过 map（哈希表/数组/直方图）交换数据        │
└─────────────────────────────────────────────────────┘
```

对比三条老路的缺陷就明白它革了什么命：

- **内核模块**：一个空指针就 panic 整机；eBPF 被 verifier 拦下，最坏是"加载失败"。
- **改内核源码**：等下一个内核版本发布/重启窗口；eBPF 即时加载即时生效。
- **strace/ptrace**：每次事件陷入用户态，开销 10 倍+（01 篇）；eBPF 在内核内聚合，只把统计结果（几 KB 的 map）传出来，开销 <5%。

代价与限制（诚实清单）：verifier 对循环/指针的保守限制让复杂逻辑难写；版本兼容问题（BTF/CO-RE 解决了大半，见第 2 节）；调试困难（只能打日志）。**经验做法**：观测类任务"先找现成工具（BCC/bpftrace 生态），没有再自己写"，脚本保存进 git，像对待代码一样对待观测脚本。

## 2. 三条技术路线：BCC、bpftrace、libbpf/CO-RE

| 维度 | BCC | bpftrace | libbpf + CO-RE |
| --- | --- | --- | --- |
| 形态 | Python/Lua 前端 + C 内核段 | 高层 DSL，一行命令起步 | C 库 + 编译一次到处运行 |
| 上手难度 | 中 | 低 | 高 |
| 适合 | 完整工具（offcputime/memleak） | 临时探查、one-liner | 产品级集成（Cilium 等） |
| 依赖 | LLVM 运行时编译、内核头文件 | bpftrace 二进制 | 编译期 BTF（`/sys/kernel/btf/vmlinux`） |
| 部署痛点 | 目标机要有编译环境 | 几乎无 | 预编译，无运行时依赖 |

Ubuntu 22.04 安装（版本号见 02 篇约定）：

```
$ sudo apt install -y bpftrace bcc-tools libbcc-examples linux-headers-$(uname -r)
$ bpftrace --version
bpftrace v0.14.0
$ ls /sys/kernel/btf/vmlinux && echo BTF 可用     # CO-RE 的前提，WSL2 kernel 6.6 已具备
/sys/kernel/btf/vmlinux
```

**WSL2 注**：WSL2 内核 6.6 支持 kprobe/tracepoint/uprobe 与 BTF，本篇全部实验可跑；需要 sudo（CAP_BPF/CAP_PERFMON 能力）；个别 tracepoint 缺失时（02 篇矩阵）换等价 kprobe 即可。

## 3. bpftrace 十分钟上手：三种挂载点各一发

bpftrace 的脚本 = 一组 `探针 /过滤/ { 动作 }`。三种最常用探针各来一发。

**(1) tracepoint——内核预埋的稳定探针（02 篇 L4）：谁在开文件？**

```
$ sudo bpftrace -e 'tracepoint:syscalls:sys_enter_openat { @[comm] = count(); }' 
Attaching 1 probe... ^C

@[bpftrace]: 2
@[systemd-journal]: 5
@[python3]: 17        ← map：按进程名聚合 open 次数
```

语法拆解：`@[comm] = count()` 是"以进程名为 key 的哈希 map，每次 +1"；Ctrl-C 时自动打印 map 并退出。tracepoint 的参数用 `args->字段名`（`bpftrace -l 'tracepoint:syscalls:sys_enter_openat' -v` 看字段）。

**(2) kprobe——挂到任意内核函数（02 篇 L5）：每个进程读了多少字节？**

```
$ sudo bpftrace -e 'kprobe:vfs_read /comm == "python3"/ { @bytes[comm] = sum(arg2); }'
Attaching 1 probe... ^C

@[python3]: 2048000   ← arg2 是 vfs_read 的第三个参数(count)，按进程聚合读字节量
```

kprobe 的参数 `arg0..argN` 按函数原型取（`bpftrace -l 'kprobe:vfs_read' -v` 显示原型）。kprobe 灵活但函数名随内核版本变；tracepoint 稳定但位置有限——**经验做法**：优先 tracepoint，实在需要再 kprobe。

**(3) uprobe——用户态函数探针：盯着 libc 的 malloc**

```
$ sudo bpftrace -e 'uprobe:/lib/x86_64-linux-gnu/libc.so.6:malloc /comm=="python3"/ { @sz = hist(arg0); }'
Attaching 1 probe... ^C

@sz:
[256, 512)          12 |@@@@@@               |
[512, 1K)           48 |@@@@@@@@@@@@@@@@@@@@@@@|     ← hist() 内置对数直方图
[1K, 2K)             8 |@@@@                  |
```

uprobe 是"给用户态二进制下断点"的机制，能观测**任何带符号的用户程序**（C/C++ 的 malloc/free/自研函数）。第 2 节的 BCC memleak 就是 malloc/free 双 uprobe + map 的组合。

再补两个高频动作：`printf()` 直接输出（事件流模式）、`profile:hz:99` 采样探针（不依赖事件的 CPU 采样）：

```
$ sudo bpftrace -e 'profile:hz:99 /pid == 52301/ { @[ustack] = count(); }'  # 火焰图数据源
$ sudo bpftrace -e 'tracepoint:sched:sched_switch { @[args->prev_comm] = count(); }'  # 上下文切换
```

## 4. 经典 one-liner 剖析：九发子弹

bpftrace 仓库的 one-liner tutorial（github.com/iovisor/bpftrace/blob/master/docs/tutorial_one_liners.md）是最高效的弹药库，这里挑 9 条逐句拆（文件系统/CPU/内存各 3 条）。

**文件系统：**

```
# F1: 谁在打开什么文件（按 进程+文件名 聚合）
bpftrace -e 'tracepoint:syscalls:sys_enter_openat { @[comm, str(args->filename)] = count(); }'
#     拆解：str() 把内核指针里的字符串拷出来；两层 key 的 map → 输出按组合分组
# F2: read 返回的块大小分布（应用是顺序大块还是随机小块？06 篇的负载画像）
bpftrace -e 'tracepoint:syscalls:sys_exit_read /args->ret > 0/ { @sz = hist(args->ret); }'
#     拆解：exit 探针的 ret 是读到的字节数；过滤 >0 排除 EOF/错误
# F3: 哪些进程在删除文件（排查“磁盘莫名变满/变空”）
bpftrace -e 'tracepoint:syscalls:sys_enter_unlink { printf("%s deleted %s\n", comm, str(args->pathname)); }'
#     拆解：printf 事件流模式，不做聚合，实时刷
```

**CPU：**

```
# C1: 每个进程的系统调用频次（syscall 风暴诊断，06 篇小文件问题的第一现场）
bpftrace -e 'tracepoint:raw_syscalls:sys_enter { @[comm] = count(); }'
# C2: 调度延迟直方图（04 篇 runqlat 的等价实现）
bpftrace -e 'tracepoint:sched:sched_wakeup { @qtime[args->pid] = nsecs; }
             tracepoint:sched:sched_switch /@qtime[args->next_pid]/ {
               @usecs[comm] = hist((nsecs - @qtime[args->next_pid]) / 1000);
               delete(@qtime[args->next_pid]); }'
#     拆解：wakeup 打卡（进运行队列）→ switch 结算（真上 CPU），Δt 就是等调度的时间
# C3: 软中断分布（网络收包压力，07 篇 NET_RX 的量化）
bpftrace -e 'tracepoint:irq:softirq_entry { @[name] = count(); }'
```

**内存：**

```
# M1: 缺页按进程分布（05 篇 minor/major 的实时版）
bpftrace -e 'tracepoint:exceptions:page_fault_user { @[comm] = count(); }'
# M2: 每个进程分配的页面数（谁在快速吃内存）
bpftrace -e 'tracepoint:kmem:mm_page_alloc { @[comm] = sum(args->order == 0 ? 1 : (1 << args->order)); }'
#     拆解：order 是 2^order 页；三元表达式在 bpftrace 里合法
# M3: mmap 调用分布（大文件映射行为，06 篇 mmap 路线的观测）
bpftrace -e 'tracepoint:syscalls:sys_enter_mmap { @[comm] = hist(args->len / 1024 / 1024); }'
```

共性规律（写自己的脚本时照抄这个骨架）：**确定问题 → 找探针（`bpftrace -l` 搜索）→ 定聚合方式（count/sum/avg/hist）→ 决定输出（map 自动打印 or printf 事件流）**。这四步能把 90% 的"我想看 X"变成 5 行内的脚本。

## 5. 动手实验：写一个"系统调用延迟"统计脚本

需求（真实排查场景）：某个 Python 服务响应抖动，怀疑是某些系统调用偶发慢（DNS、盘 IO），需要"每个系统调用的延迟直方图"。用 bpftrace 实现（就是 02 篇 L5 层的"可编程化 strace -T"，但开销低两个数量级）：

```bash
#!/usr/bin/env bpftrace
/* syscall_lat.bt —— 统计每个系统调用的延迟直方图（按 syscall 号）
 * 用法: sudo bpftrace syscall_lat.bt <可选:只看该PID>
 * syscall 号 → 名字的对照: ausyscall --dump 或 /usr/include/x86_64-linux-gnu/asm/unistd_64.h
 */

begin
{
    printf("Tracing syscall latency... Hit Ctrl-C to end.\n");
}

tracepoint:raw_syscalls:sys_enter
{
    @start[tid, args->id] = nsecs;          // 打卡：线程 + syscall 号
}

tracepoint:raw_syscalls:sys_exit
/@start[tid, args->id]/
{
    @usecs[args->id] = hist((nsecs - @start[tid, args->id]) / 1000);
    delete(@start[tid, args->id]);
}

end
{
    printf("\n%-8s %s\n", "syscall#", "latency histogram (usec)");
    print(@usecs);
    clear(@start);
}
```

跑起来，给它喂一个"故意有问题"的目标程序（每个请求先做一次同步 DNS + 一次小文件写）：

```python
# slow_syscalls.py —— 三个可疑动作：getaddrinfo / 小文件写 / 纯计算
import socket, time, os
for i in range(30):
    socket.getaddrinfo("example.com", 80)         # 可疑点1：每请求同步 DNS
    with open("/tmp/tick.txt", "w") as f:         # 可疑点2：open+write+close
        f.write(str(time.time()))
    sum(j * j for j in range(100000))             # 对照组：纯 CPU
    time.sleep(0.05)
```

```
$ sudo bpftrace syscall_lat.bt &        # 后台挂上
$ python3 slow_syscalls.py
^C（结束 bpftrace）
Attaching 4 probes...

syscall#  latency histogram (usec)
42:                                          ← 42 = connect（DNS 建连）
[512, 1K)             3 |@@@                 |
[1K, 2K)              8 |@@@@@@@@             |
[2K, 4K)             19 |@@@@@@@@@@@@@@@@@@@|   ← 19 次 connect 花 2~4ms：同步 DNS 是长尾源
12:                                          ← 12 = brk（内存分配）
[0, 1)                30 |@@@@@@@@@@@@@@@@@@@|
257:                                         ← 257 = openat
[2, 4)                30 |@@@@@@@@@@@@@@@@@@@|
1:                                            ← 1 = write
[1, 2)                30 |@@@@@@@@@@@@@@@@@@@|
```

（syscall 号对照 x86_64 的 unistd_64.h：42=connect、12=brk、257=openat、1=write；ARM64 上数值不同，用 `ausyscall --dump` 对照——**ARM 注**。）

读图结论：write/openat/brk 都在微秒级，唯独 connect（同步 DNS）稳定 2~4ms——P99 抖动的元凶找到。修复方向：进程内 DNS 缓存/连接池复用（把每请求一次 connect 变成长连接），对照 07 篇"每次新建连接付握手税"的结论。用 `time` 对比修复前后吞吐，记录数字，就完成了 01 篇方法论的又一次闭环。

**注意**（观察者效应自查，01 篇）：`@start[tid, args->id]` 对每个 syscall 事件写两次 map，高频 syscall（如 futex）下有可测开销；生产长跑请加过滤（如只统计 `args->id` 的白名单或指定 PID），短时诊断场景忽略不计。

## 6. eBPF 在可观测产品中的位置

今天的主流观测产品大量以 eBPF 为地基（知道即可，便于选型时看穿宣传）：

- **Cilium**（网络）：用 eBPF 做 K8s 的网络策略/负载均衡，数据面绕过 iptables——你在 11 篇看容器网络时遇到它，就明白它快在哪。
- **Parca / Pyroscope**（持续剖析, continuous profiling）：全时低开销采样火焰图存档，理念来自 Google 的 Google-Wide Profiling；采样内核正是本篇的 `profile:hz:99`。
- **Pixie / kubeSphere 类**（K8s 一键观测）：应用零埋点，靠 uprobe/协议解析在内核侧还原请求级指标。
- **Netflix 的经验**：Brendan Gregg 团队把 bpftrace/BCC 做成生产标配（其博客的多篇案例），是"eBPF 能上生产"最著名的背书。

**经验做法**：产品选型时用三个问题检验其 eBPF 宣称——跑在哪些内核版本（BTF 要求）？观测数据在内核里聚合还是全量上传（开销差 10 倍）？探针失败时的降级路径是什么？

## 常见误区

1. **"eBPF 随便挂，零开销"**：单次探针触发是纳秒~微秒级，但"每秒百万事件 × 复杂动作"就能吃掉可观 CPU。任何脚本上线前用 01 篇的对照法量观察者效应。
2. **"verifier 报错 = 我写错了"**：多数报错是"verifier 太保守"（如循环边界不确定、指针可能为空），改写法（先判空、用固定次数展开）即可，不是逻辑错误。
3. **"bpftrace 能看到 Python 函数"**：uprobe 挂的是 C 符号；CPython 的函数在栈上是解释器帧，bpftrace 看不到。Python 层观测回 10 篇（py-spy），两层配合才是完整答案。
4. **"kprobe 脚本永远有效"**：内核函数会改名/删参数（尤其内联后消失），升级内核后脚本要复查；长期部署用 tracepoint 或 CO-RE 的 kfunc。
5. **"BCC 工具在 WSL2 都能跑"**：核心工具（offcputime/runqlat/cachestat/memleak）可用；依赖缺失 tracepoint 或硬件特性的个别工具（如某些 PMU 相关）不可用，报错先对照 02 篇矩阵。

## 本篇小结

- eBPF = verifier（安全）+ JIT（高效）+ map（内核/用户态数据交换），三者合起来支撑"不改内核、可编程、低开销"的观测革命。
- 路线选择：临时探查 bpftrace one-liner、完整工具 BCC、产品集成 libbpf/CO-RE；写自定义脚本的四步骨架：问题→探针→聚合→输出。
- kprobe/tracepoint/uprobe 三种挂载点覆盖内核函数、内核预埋点、用户态函数；`@map = count/sum/hist` 是聚合三件套。
- syscall 延迟脚本实战：sys_enter 打卡 / sys_exit 结算，hist 出直方图；发现"同步 DNS 每请求 2~4ms"这类被均值掩盖的元凶。
- 可观测产品（Cilium/Parca/Pixie）的地基都是本篇技术；选型时检验内核版本要求、聚合位置、降级路径三件事。

## 练习

1. 跑第 4 节的 F1（openat 聚合），找出你系统里 1 分钟内 open 次数最多的进程；再用 F2 看它的 read 块大小分布，判断它是流式读还是小块随机读。
2. 修改 syscall_lat.bt，加上 PID 参数过滤（`/pid == $1/`），只统计 slow_syscalls.py 一个进程；对比过滤前后的直方图差异，体会"过滤越早开销越小"。
3. 用 C2 的 runqlat 脚本，分别在"系统空闲"与"跑满 8 个死循环"时各采 30 秒，把两份直方图并排抄下，写一句话总结调度延迟的量级变化（对照 04 篇 perf sched 的结果）。
4. 给一个你自己的 C 程序（哪怕是 hello world 改成循环 malloc/free）挂 uprobe 统计 malloc 大小直方图；再换成 Python 进程，观察为什么 uprobe 挂 libc:malloc 依然有效（提示：CPython 最终也走 libc/glibc 分配器，但 pymalloc 有自己的路径，对比两图的形状）。
5. 把第 5 节的 slow_syscalls.py 修复成"DNS 结果缓存 + 长连接"，重跑 bpftrace 脚本，记录 connect 直方图的变化；用 time 算吞吐提升，写出这次实验的"三句话报告"（症状/证据/修复）。
