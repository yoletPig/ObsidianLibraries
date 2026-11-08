# Linux 性能工程系列教程：系统地观测、定位、证明性能问题

> 面向以 Python 为主、能读懂 C/C++、在 WSL2 (kernel 6.6) / Ubuntu 22.04 上动手、手边还有 RK3588/RK3576、华为 310P 等 ARM 端侧板子的学习者。本系列的立场：**性能优化不是玄学，而是一门实验科学**——系统地观测、定位、证明，而不是猜。"我觉得这里慢"是猜；"火焰图显示 62% 时间在 tokenize、P99 从 890ms 降到 120ms"是证明。库里的操作系统系列（进程/内存/IO 地基）是本系列的前置，模型性能优化全系列（vLLM/CUDA/端侧）是本系列最大的应用场景。
> 全系列为中文原创，共 12 章。所有工具命令在 Linux/WSL2 上真实可跑，跑不了的（如 WSL2 受限的硬件 PMU、需要真机的 GPU 命令）会明确标注差异或用文档+图说明。关键论断标注出处（Brendan Gregg《Systems Performance》、brendangregg.com、man 手册节号、内核文档路径如 `Documentation/admin-guide/`）；无法查证的经验做法一律标注"经验做法"。

## 学习路径

| 篇 | 主题 | 你将学会 |
| --- | --- | --- |
| [01-性能工程方法论](01-性能工程方法论.md) | 延迟/吞吐/利用率、Little's Law、USE/RED 方法、P99 长尾、性能预算、观察者效应 | 用 Little's Law 推算并发与排队；把"慢"定义成分布而不是均值；用 USE 方法系统地过一遍机器 |
| [02-工具全景与观测层次](02-工具全景与观测层次.md) | 观测分层（库→syscall→计数器→tracepoint→kprobe→PMU）、静态工具一览、动态追踪谱系 | 拿到一台陌生机器知道先跑哪 8 条命令；按开销和可见性选观测工具；知道 WSL2 上什么能用 |
| [03-CPU性能剖析-perf与火焰图](03-CPU性能剖析-perf与火焰图.md) | perf stat/record/report、IPC 与缓存命中率解读、on-CPU 火焰图全流程 | 读出 IPC 背后的瓶颈类型；生成并读懂火焰图（宽度=占比、深度=调用栈）；实测优化一个热点程序 |
| [04-延迟分析与off-CPU](04-延迟分析与off-CPU.md) | on-CPU vs off-CPU、offcputime、ftrace 最小用法、调度延迟、CPU 不高但很慢案例 | 判断该看 CPU 还是看等待；用 tracepoint 量出"时间去哪了"；独立排查一个 IO 等待型慢服务 |
| [05-内存性能分析](05-内存性能分析.md) | free 逐列、available vs free、缺页与工作集、kswapd/swap、泄漏定位三法、OOM 日志逐字段 | 说出内存"够不够"的真正指标；用 tracemalloc/ASan/massif 定位泄漏；逐字段读懂 OOM killer 现场报告 |
| [06-IO与文件系统性能](06-IO与文件系统性能.md) | 页缓存/脏页回写/IO 调度、iostat 逐列与 %util 陷阱、fio 基准、mmap vs read | 用 fio 量出磁盘真实能力并写成报告；避开 %util/await 的经典误读；给"小文件多"的场景开对药方 |
| [07-网络性能分析](07-网络性能分析.md) | ss/tcpdump/iperf3、RTT 分解、丢包排查路径、softirq 收包概览、socket 缓冲区调优 | 抓一个 HTTP 请求的完整时序并算出每段延迟；顺着计数器查丢包；判断吞吐卡在缓冲区还是带宽 |
| [08-eBPF入门](08-eBPF入门.md) | eBPF 原理与革命意义、BCC/bpftrace 对比、kprobe/tracepoint/uprobe 上手、syscall 延迟脚本 | 十分钟写出自定义观测脚本；剖析经典 one-liner；知道 Cilium/Parca 这类产品底下是什么 |
| [09-GPU与端侧观测](09-GPU与端侧观测.md) | nvidia-smi/dmon/pmon/nvtop、Nsight Systems 思想、RK3588 NPU 与热节流、cpufreq governor | 读出 GPU 利用率字段的陷阱；看 kernel 时间与空隙；在端侧板子上盯温度/频率/NPU，识别热节流 |
| [10-应用剖析-Python与C++](10-应用剖析-Python与C++.md) | cProfile、py-spy 无侵入采样、GIL 观测、callgrind、编译优化常识、PGO/BOLT 概览 | 给 Python 进程无侵入采样并 dump 线程栈查死锁；解释 gprof 为何过时；完整走一遍"剖析→重写→数字对比" |
| [11-容器与云环境观测](11-容器与云环境观测.md) | namespace 边界、cgroup throttling 实测、容器 OOM、K8s 观测、邻居噪声、Prometheus 指标设计 | 解释容器内 top 为什么骗人；实测 CPU throttled 时间；把 USE/RED 翻译成合理的监控指标 |
| [12-实战-推理服务性能调优案例](12-实战-推理服务性能调优案例.md) | FastAPI 推理服务逐层调优：基线→USE→P99 长尾→定位→修复→调优报告；面试 25 题 | 把前 11 章串成完整排查流程并写出可交付的调优报告；带走 25 道高频面试题要点 |

### 阶段划分

1. **方法论与工具地基篇（01–02）**：先立规矩再学工具。01 讲清"什么算慢、怎么系统地找"，02 给出全景工具箱和观测分层地图。这两章是全系列的骨架，建议先读。
2. **单资源深潜篇（03–07）**：CPU（perf/火焰图）→ 延迟与等待（off-CPU）→ 内存 → IO → 网络，一章一个案发现场。每章都有"故意写错的程序 → 观测 → 定位 → 修复 → 前后对比"的完整实验。
3. **高级观测与异构篇（08–10）**：eBPF 让你不再受限于现成工具；GPU 与端侧把视野扩展到异构硬件；应用剖析回到你天天写的 Python/C++ 代码。
4. **云原生与综合实战篇（11–12）**：容器/K8s/云环境改变了很多前提（观测边界、cgroup 限制、邻居噪声）；最后一章用一个推理服务把全部方法串成一次完整调优，并产出报告模板。

## 与本仓库其他方向的衔接

- **操作系统系列 = 本系列的前置**：OS 系列负责"概念"（进程/内存/IO/调度），本系列负责"定量归因"。03 篇火焰图里的调度函数、05 篇的缺页、06 篇的页缓存，都假定你已读过 OS 系列 02/04/05/08；遇到陌生概念请回查。
- **模型性能优化 Stage 3（CUDA）**：09 篇的 Nsight Systems "kernel 时间 vs 空隙"就是 CUDA 优化的入口；OS 系列/本系列讲的 PCIe、显存带宽、热节流，是 CUDA kernel 优化躲不开的系统侧约束。
- **模型性能优化 Stage 5（vLLM/推理引擎）**：vLLM 的 P99 延迟抖动、 Continuous Batching 的排队，用 01 篇的 Little's Law 和 04 篇的调度延迟分析；KV cache 的内存压力用 05 篇的方法观测。
- **模型性能优化 Stage 4（端侧部署）**：RK3588 上的 NPU 占用、热节流、内存带宽瓶颈是 09 篇的主题——端侧"跑得慢"多半不是模型问题，而是降频和带宽。
- **量化投资 09（回测系统）**：回测慢通常是 Python 热点循环 + 大 DataFrame 内存拷贝，10 篇的 cProfile/py-spy 方法直接适用；回测加载数据慢是 06 篇的 IO 问题。
- **操作系统系列 13（容器与虚拟化）**：那篇讲 namespace/cgroup 的"原理"，本系列 11 篇讲"限制之下怎么观测"（throttling、容器 OOM、指标设计），两篇配合读。

## 约定

- 平台约定：以 Ubuntu 22.04 / WSL2 (kernel 6.6) 为验证环境；**WSL2 受限的功能会标注"WSL2 注"**（典型：硬件 PMU 计数器通常不可用，perf 的 cycles 事件显示 `<not supported>`，需改用软件事件）；ARM 端侧（RK3588/RK3576、华为 310P）差异单独标注"ARM 注"。
- 工具版本约定：文中注明版本号——sysstat 12.5.2（iostat/mpstat/pidstat）、perf 6.x/5.15+、fio 3.28、tcpdump 4.99、iperf3 3.9、valgrind 3.18、bpftrace 0.14、BCC 0.24、Python 3.10、gcc 11.4。你的版本不同输出细节可能略异。
- 命令约定：需要 root 的命令标 `sudo`；涉及压测/写盘的实验提醒别在生产机执行。输出示例来自作者机器（WSL2, kernel 6.6），地址/数字必然与你不同，**看形状不看数值**。
- 引用约定：《Systems Performance》指 Brendan Gregg, *Systems Performance* 2nd ed (2020)（中译《性能之巅》第 2 版）；brendangregg.com 的 USE 方法、火焰图、off-CPU 分析页面按需引用；man 手册标注节号；内核文档标注路径（如 `Documentation/admin-guide/sysctl/vm.rst`）；经验做法不冒充定论。
- 术语约定：优先中文术语、首次出现附英文，如"缺页"（page fault）、"上下文切换"（context switch）、"尾延迟"（tail latency）。
- 本系列不使用 Obsidian 双链以外的特殊语法，任何 markdown 阅读器都能正常渲染。
