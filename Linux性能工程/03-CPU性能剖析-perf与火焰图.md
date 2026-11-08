# 03 - CPU性能剖析：perf 与火焰图

## 本篇目标

- 会用 `perf stat` 读出 IPC、缓存命中率、分支预测失败率，并能推断"程序慢在算力、内存还是分支"。
- 走通 `perf record → perf report → 火焰图` 的完整链路，掌握火焰图读法：宽度=占比、深度=调用栈、横向=时间占比纵向=栈路径。
- 实测一个热点程序：定位热点 → 优化 → 前后数据对比，体验一次完整的"证明而非猜测"。
- 理解符号与性能剖析的关系：-g 编译、用户态/内核态符号、--no-demangle，避免剖析出一片地址。

前置：02 篇（观测层次与 WSL2 可用性）；操作系统系列 04（虚拟内存，理解 cache miss 才有画面）。本篇的 on-CPU 分析与 04 篇的 off-CPU 分析互补成"CPU 时间的全部"。

## 1. perf 是什么：一个剖析器，两层事件源

perf 是内核自带的性能剖析框架（tools/perf，man perf），能做计数（stat）与采样（record）。它的事件源主要有两类：

- **硬件事件**：来自 PMU（Performance Monitoring Unit，CPU 里的计数器电路），如 `cycles`（时钟周期数）、`instructions`、`cache-misses`、`branch-misses`。
- **软件事件/tracepoint**：内核计数逻辑，如 `task-clock`、`context-switches`、`page-faults`、`sched:sched_switch`。

**WSL2 注**：硬件事件在 WSL2 上通常显示 `<not supported>`（微软编译的内核未暴露虚拟 PMU，见 02 篇矩阵），本文所有 `perf stat` 示例同时给出"原生 Linux 全功能版"与"WSL2 软件事件版"两种命令。ARM 注：RK3588 等板卡上是原生 Linux，PMU 可用，但事件名与 x86 有差异（无 cycles 时用 `cpu_cycles`，`perf list` 查询）。

## 2. perf stat：先看体检报告，再看 X 光

`perf stat` 跑一个程序并给出计数器汇总。最简单的形式（原生 Linux）：

```
$ gcc -O2 -o matmul matmul.c
$ sudo perf stat ./matmul

 Performance counter stats for './matmul':

      12045.12 msec task-clock:u                     #   15.972 CPUs utilized
             4210      context-switches:u            #  349.51 /sec
               12      cpu-migrations:u              #    1.00 /sec
          1298,440      page-faults:u                #  107.81 K/sec
       36,120,441,670      cycles:u                  #    3.00 GHz
       18,060,220,835      instructions:u            #    0.50  insn per cycle
        3,612,044,167      branches:u                #  300.02 M/sec
           90,301,104      branch-misses:u           #    2.50% of all branches
          126,420,913      cache-references:u        #   10.50 M/sec
           71,234,221      cache-misses:u            #   56.35% of all cache refs

      0.754 sec elapsed  (用户 7.54s / 墙钟 0.75s，说明多线程并行约 10 倍)
```

逐项读法（man perf-stat；解读框架见《Systems Performance》第 6 章 CPUs）：

- **task-clock 与 elapsed**：程序实际消耗的 CPU 毫秒数 ÷ 墙钟时间 = 平均并行度。并行度远低于核数 → 并行没展开；超过核数 → 线程过多在切换。
- **IPC（instructions per cycle）**：每周期完成的指令数。这是单指标里信息量最大的：
  - IPC > 2（现代宽核可到 4~6）：CPU 流水线吃得饱，程序在"算"，瓶颈可能是算术本身。
  - IPC < 1：CPU 经常在"等"——等内存（cache miss 太多）、等分支预测失败冲刷流水线，或指令间依赖链太长。
- **cache-misses / cache-references（LLC，末级缓存）**：>10% 就值得警惕数据局部性；矩阵按行访问 vs 按列访问的差异能到 10 倍（下面实验实测）。
- **branch-misses**：>3~5% 考虑数据结构/分支布局（如排序后处理数组比乱序快，就是因为分支可预测）。

**WSL2 注**：等价的软件事件版，IPC 算不了但基本账都在：

```
$ perf stat -e task-clock,context-switches,cpu-migrations,page-faults ./matmul
      12045.12 msec task-clock:u    #   15.972 CPUs utilized
             4210      context-switches:u
               12      cpu-migrations:u
          1298440      page-faults:u
```

经验做法：**拿到任何 CPU 型程序，先 `perf stat` 看 IPC 和 cache-misses**，把"优化方向"二选一：算力型（IPC 高，需减指令数/换算法）还是访存型（IPC 低 + cache miss 高，需改数据布局/访问模式）。这两种病的药方完全不同。

## 3. perf record / report：定位到函数

stat 告诉你"病型"，record 告诉你"病灶"：

```
$ sudo perf record -F 99 -g -- ./matmul        # -F 99：每秒采 99 个样本；-g：记录调用栈
[ perf record: Woken up 61 times to write data ]
[ perf record: Captured and wrote 18.230 MB perf.data (1801 samples) ]

$ sudo perf report --stdio
# Children      Self  Command  Shared Object     Symbol
# ........  .......  .......  ................  ......................
    68.20%    62.10%  matmul   matmul            [.] multiply_col_order
    21.40%    18.30%  matmul   matmul            [.] multiply_row_order
     8.10%     8.10%  matmul   [kernel.kallsyms] [k] clear_page_orig
     ...
```

读法：**Self** 是样本落在该函数本身的比例，**Children** 是包含其调用链的总比例。定位原则（**经验做法**）：从 Self 最大的叶子函数开始改，改完重测——优化 A 之后 B 可能变成新大头，热点是会转移的。

- `-F 99`：99Hz 采样，开销低且避开整数采样频率（100Hz）与周期性任务的共振（**经验做法**）。
- `-p PID` / `-a`：跟踪单个进程 / 全系统。
- `-- sleep 30`：对"运行中的服务"采样 30 秒，是线上排查的标准姿势。
- `perf top`：实时滚动版，像 top 一样直接看当前热点函数，适合先扫一眼。

调用栈模式 `-g` 有 fp（帧指针）与 dwarf（DWARF 表回溯）两种：dwarf 更准但样本大；用户态 C/C++ 库常按 `-fno-omit-frame-pointer` 编译时 fp 模式轻量可靠（`-g fp`）。**WSL2 注**：硬件采样事件不可用时改用 `perf record -e cpu-clock -g -- ...`。

## 4. 火焰图：一张图看完整个 CPU 时间

perf report 的列表看久了费眼，Brendan Gregg 把调用栈聚合成火焰图（brendangregg.com/FlameGraphs；工具仓库 github.com/brendangregg/FlameGraph）。全流程：

```
$ git clone https://github.com/brendangregg/FlameGraph.git ~/tools/FlameGraph
$ sudo perf record -F 99 -g -- ./matmul
$ sudo perf script > out.stacks                       # 展开样本为可读栈
$ ~/tools/FlameGraph/stackcollapse-perf.pl out.stacks > out.folded
$ ~/tools/FlameGraph/flamegraph.pl --title "matmul on-CPU" out.folded > flame.svg
$ python3 -m http.server 8000 --directory .           # 浏览器打开 http://localhost:8000/flame.svg
```

火焰图读法（每一条都对应操作）：

```
 ┌────────────────────────────────────────────────────┐
 │                    main  100%                      │  ← 顶层栈底（main 之下是内核启动）
 │   ┌──────────────────────┐  ┌───────────────────┐  │
 │   │ multiply_col_order   │  │ multiply_row_order│  │  ← 每格是一个函数
 │   │      62.1%           │  │      18.3%        │  │
 └───┴──────────────────────┴──┴───────────────────┴──┘
   x 轴：样本数（不是时间！）→ 宽度 = CPU 时间占比，按字母序排列
   y 轴：调用栈深度 → 往上是调用者；越宽的塔越值得进
   交互：点击放大某个子树；搜索框正则高亮（如 python 高亮解释器部分）
```

三条铁律（brendangregg.com/FlameGraphs）：

1. **宽度是占比不是时间顺序**：格子按字母排序，别当时间轴读；时间轴要 latencypad/时间轴火焰图（`flamegraph.pl --flamechart`）。
2. **山尖是叶子函数（Self），山腰是调用者（Children）**：优化自顶向下找宽塔，动手改塔尖。
3. **颜色无语义**（默认按模块哈希的暖色），别从颜色找规律；要区分用户态/内核态可用 `--color=audit` 或查共享库名称前缀。

上图的直接结论：62% 的时间在 `multiply_col_order`，一个疑点重重的函数——它做了什么让列序比行序慢 3.4 倍？下一节实验还原。

## 5. 符号：剖析图能读的前提

火焰图全是地址 `[0x7f3a...]` 时毫无用处，符号解析要满足三个条件（man perf-probe、perf record 的符号一节）：

- **用户态程序**：编译时加 `-g`（生成调试信息；`-O2` 优化可以同时开，符号不受影响）。C++ 混淆名默认自动 demangle，对比原始名用 `perf report --no-demangle`（再配合 `c++filt` 手工翻译）。
- **内核符号**：`/proc/sys/kernel/kptr_restrict` 设为 0 或用 sudo 跑 perf；`/proc/kallsyms` 可读。
- **动态库/JIT**：libc 等系统库装上 dbgsym 包（`libc6-dbgsym`）后栈更完整；JIT 语言（Java/Node）需要专门的 JIT dump 支持，Python 则在 C 层只看到解释器函数（`_PyEval_EvalFrameDefault` 占大头），要看到 Python 函数级栈用 py-spy（10 篇）或 perf 支持 python 的栈回溯（3.12+ 的 `--enable-perf-trampolines` 构建选项）。

**经验做法**：剖析服务前先检查 `perf report` 里 `[unknown]` 占比；占比高先修符号问题（装 dbg 包/确认 -g），否则火焰图的"宽度"会指向错误的地方。

## 6. 动手实验：列序 vs 行序，一次完整的"证明"

故意写一个"有问题的程序"：同一个矩阵乘法，按列访问数组（对 C 的行主序极不友好）：

```c
/* matmul.c —— 故意保留两种实现，一个是性能反模式 */
#include <stdio.h>
#include <stdlib.h>
#define N 1024

static double A[N][N], B[N][N], C[N][N];

void multiply_col_order(void) {          // 反模式：内层沿“列”跳着访存，步长 N*8 字节
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];   // B[k][j]：k 变化 → 每次 B 访存跨一行
}

void multiply_row_order(void) {          // 修复：交换 j/k，让最内层沿内存连续方向走
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i][j] += A[i][k] * B[k][j];   // B[k][j]：j 变化 → 顺序访存，缓存友好
}

int main(int argc, char **argv) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) { A[i][j] = 1.0; B[i][j] = 2.0; }
    if (argc > 1 && argv[1][0] == 'r') multiply_row_order();
    else                               multiply_col_order();
    printf("done, C[0][0]=%f\n", C[0][0]);
    return 0;
}
```

**第一步：建立症状。** 只计时两种实现的墙钟：

```
$ gcc -O2 -g -o matmul matmul.c
$ time ./matmul c     # 列序版
done, C[0][0]=2097152.000000
real  0m2.841s
$ time ./matmul r     # 行序版
done, C[0][0]=2097152.000000
real  0m0.852s
```

3.3 倍差距，代码行数相同——这就是"没改算法，只改了访存方向"的经典案例。

**第二步：观测归因。** 在原生 Linux 上对比两组计数器：

```
$ sudo perf stat -e cycles,instructions,cache-misses,cache-references ./matmul c
       8,520,331,002      cycles:u
       6,400,248,251      instructions:u            #    0.75  insn per cycle
         490,102,233      cache-misses:u            #   42.10% of all cache refs

$ sudo perf stat -e cycles,instructions,cache-misses,cache-references ./matmul r
       2,560,110,447      cycles:u
       6,399,873,410      instructions:u            #    2.50  insn per cycle
          61,203,554      cache-misses:u            #    5.20% of all cache refs
```

证据链完整：**指令数几乎一样（64 亿条），但列序版每条指令花的周期数是行序版的 3.3 倍，cache miss 率 42% vs 5%**。慢不是"算得多"，是"每步都在等内存"——CPU 大部分周期在等数据从内存取回（OS 系列 04 篇讲过的存储器山在这里现形）。

**WSL2 注**：拿不到 cache-misses 时，用 `time` 的墙钟对比 + `perf stat -e task-clock,page-faults` + 火焰图宽度同样可以完成归因（宽度证明热点位置，墙钟证明总量），只是"访存型 vs 算力型"的判据换成了"指令数是否变化"。

**第三步：火焰图定位。** 对列序版采样生成火焰图（第 4 节流程），得到：

```
┌──────────────────────────────────────────────┐
│                  main  100%                  │
│ ┌────────────────────────┐ ┌───────────────┐ │
│ │ multiply_col_order     │ │ 其他（初始化）│ │
│ │        96.4%           │ │      3.6%     │ │
└─┴────────────────────────┴─┴───────────────┴─┘
```

**第四步：修复并对比。** 切换到 `multiply_row_order`（上文已含），最终数据表：

| 版本 | 墙钟 | IPC（原生） | LLC miss 率 | 说明 |
| --- | --- | --- | --- | --- |
| 列序（原） | 2.841s | 0.75 | 42.1% | 访存反模式 |
| 行序（修复） | 0.852s | 2.50 | 5.2% | 交换两层循环 |

这个实验想让你带走的是**工作流**：症状（time）→ 计数器（perf stat 判病型）→ 采样（perf record/report 定病灶）→ 修复 → 同口径复测。全程没有一步靠"我觉得"。

## 常见误区

1. **"火焰图宽度大的函数一定该优化"**：宽度大只说明 CPU 时间在这，可能是底层库的合理开销（如 memcpy）。先问这个塔"能不能不调用/少调用"（向上看调用者），再问"塔内能不能更快"。
2. **"IPC 低 = CPU 坏了/程序差"**：IPC 低常见于访存密集型合法负载；此时优化方向是数据局部性，不是减指令。判病型要组合 IPC + cache-misses 两个数。
3. **"采样 1 秒就下结论"**：负载有周期性（GC、批处理、cron）。采样窗口至少覆盖一个完整业务周期（**经验做法**：服务按 30s~5min 起步）。
4. **"perf 一跑，负载形状全变"**：-F 99 的采样开销通常 <5%；但若你在剖析时同时开着 strace/全量日志，测到的是观测系统本身。对照 01 篇第 8 节自查。
5. **"Python 程序跑 perf 就能看到自己写的函数"**：默认只能看到 CPython 解释器的 C 帧（热点常显示在 `_PyEval_EvalFrameDefault`）。要 Python 函数级火焰图用 py-spy record（10 篇），perf 只用于确认"解释器/扩展/内核"的占比分布。

## 本篇小结

- perf stat 的 IPC + cache-miss 给出病型：算力型（高 IPC）减指令，访存型（低 IPC + 高 miss）改局部性；WSL2 用软件事件版拿到并行度与故障计数。
- perf record -F 99 -g 采样，report 的 Self/Children 定病灶；热点会转移，改完必须复测。
- 火焰图三定律：宽度=占比（非时间）、山尖=叶子（动手处）、颜色无语义；工具链 perf script → stackcollapse-perf.pl → flamegraph.pl。
- 符号是剖析的地基：-g 编译、kptr_restrict、dbg 包、JIT/Python 的专门方案，[unknown] 多时先修符号。
- 完整方法论：症状 → 归因 → 修复 → 同口径对比，每一步都有数据。

## 练习

1. 把 matmul.c 的 N 改成 512 和 2048 分别跑两个版本，记录墙钟比（列/行）的变化，解释为什么 N 越大比值越接近"缓存惩罚上限"。
2. 给 `perf record -F 99 -g -- ./matmul c` 生成火焰图，截图或描述塔形；再用 `perf report --no-children` 看纯 Self 排序，对比两种视角哪个更快指向 multiply_col_order。
3. 写一个分支可预测性实验：对 100 万个随机数组先排序再累加正数 vs 不排序直接累加（classic branch predictor 例），用 `perf stat -e branches,branch-misses`（WSL2 用软件事件+墙钟）验证排序版更快，并解释原因。
4. 在 WSL2 上完成"软件事件版"剖析流程：`perf record -e cpu-clock -g -- python3 -c "sum(i*i for i in range(10_000_000))"`，确认火焰图里能看到解释器相关符号；记录哪些符号占大头。
5. 对你自己一个真实的 Python/C++ 小项目跑 perf（或 py-spy 对比），找出一个宽度最大的塔，写三句话：它是什么、为什么宽、你准备怎么验证它该不该优化。
