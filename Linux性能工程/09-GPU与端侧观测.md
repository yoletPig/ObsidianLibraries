# 09 - GPU与端侧观测

## 本篇目标

- 读懂 nvidia-smi 的字段与"GPU 利用率"的陷阱，会 dmon/pmon/nvtop 持续观测。
- 建立 Nsight Systems 的时间轴思维：kernel 时间 vs 空隙，把"GPU 慢"分解成"算得慢"还是"喂不饱"。
- 在 ARM 端侧（RK3588 为例）观测 NPU 占用、CPU/GPU 频率与温度，认识热节流与内存带宽瓶颈。
- 形成端侧性能清单：拿到一块新板子，按清单 30 分钟内摸清性能底细。

前置：模型性能优化 Stage 3（CUDA）与 Stage 4（端侧部署）是本篇的应用场景；本篇方法与 01 篇 USE 方法同源——GPU/NPU 只是"新的资源类型"，检查表换成对应工具。

**环境标注**：GPU 命令需要真实 NVIDIA GPU（WSL2 有 CUDA 直通时 nvidia-smi/dmon/nvtop 可用，Nsight 需 Windows 侧配合）；ARM 端侧命令需在 RK3588/RK3576 等板子上执行，无硬件的环境按文档+图理解，命令形态以实际 SDK 为准（**经验做法**：端侧厂商 SDK 版本差异大，本文路径标注"以 `find /sys -name <关键字>` 实际探测为准"）。

## 1. nvidia-smi：第一个陷阱就藏在"利用率"里

```
$ nvidia-smi
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15             Driver Version: 550.54.15     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap         |   Memory-Usage         | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  | 00000000:01:00.0   Off |                  N/A |
| 30%   62C    P2              240W / 450W |  22000MiB / 24564MiB   |    98%      Default |
+-----------------------------------------+------------------------+----------------------+
| Processes:                                                                             |
|  GPU   GI   CI        PID   Type   Process name                        GPU Memory      |
|  ...   ...  ...      48112    C    python3 infer_server.py                  21000MiB   |
+---------------------------------------------------------------------------------------+
```

逐项读法（man nvidia-smi；官方用户文档"GPU Utilization"节）：

- **GPU-Util**："观测窗口内有 kernel 在执行的时间占比"。**陷阱：它不衡量效率**。一个每秒只跑 1ms 小 kernel 的空转循环，GPU-Util 也能显示 98%——kernel 之间的巨大空隙被"有活就算忙"抹掉了。真正的效率要看第 2 节的时间轴。
- **Memory-Usage**：显存占用。注意"分配了"和"常用着"是两回事（PyTorch 的 caching allocator 只进不出，05 篇的 RSS 平台期在显存上重演）；`22GB/24.5GB` 接近满时新请求会 OOM（CUDA out of memory）。
- **Temp / Pwr / Perf 状态**：`P2` 是性能状态（P0 最高，P8 空闲），功耗贴着 Cap 且温度 >83C 时大概率已触发热节流——**端侧与消费卡的隐形杀手，第 6 节专述**。
- **Persistence-M**：持久模式。关闭时驱动随最后一个进程退出而卸载，下次加载要数百 ms 的延迟抖动——**经验做法**：常驻推理服务开启（`sudo nvidia-smi -pm 1`），消除首请求毛刺。

持续观测用 dmon（每秒一行）与 pmon（按进程）：

```
$ nvidia-smi dmon -s pucvmet -d 1
# gpu  pwr gtemp mtemp   sm   mem   enc   dec  mclk  pclk
    0   240    62    54   98     8     0     0  10501   2520
    0   242    63    54   96     6     0     0  10501   2520
# -s 字段：p=功耗 u=利用率 c=时钟 v=显存/PCIe吞吐 e=ECC m=显存带宽/温度 t=PCIe吞吐
$ nvidia-smi pmon -d 1                # 每进程的 sm/mem/enc/dec 占比
# gpu  pid  type    sm   mem   enc   dec   command
    0  48112   C     96     4     0     0   python3
```

dmon 的 `sm`/`mem` 两列就是"算得忙不忙 / 显存通道忙不忙"的粗粒度二分：**sm 高 mem 低 = 计算瓶颈（该优化 kernel）；sm 低 mem 高 = 带宽瓶颈（该减访存/换算子）；两者都低但 GPU-Util 高 = 空隙太大（该做流水线/batching）**——这三分法是 GPU 侧 USE 检查表的雏形。图形化用 nvtop（`sudo apt install nvtop`），交互式看多卡与进程。

## 2. Nsight Systems 思想：时间轴上找空隙

nvidia-smi 回答"忙不忙"，Nsight Systems（nsys）回答"时间花在哪"：把 kernel 执行、memcpy、NCCL 通信、CPU 侧调用铺到同一条时间轴上。安装后 `nsys profile python3 infer.py`，导出的时间轴长这样（ASCII 复刻）：

```
CPU  │██████┐   ┌─██████┐   ┌─██████┐   ┌─██████
     │预处理│   │预处理  │   │预处理  │   │预处理     ← CPU 一直忙
GPU  │      └███┘       └███┘       └███┘          ← kernel 执行
     │      ↑空隙↑     ↑空隙↑     ↑空隙↑
     │      = CPU 在准备数据，GPU 在干等（利用率低但“贵”）
```

读轴三问（对应模型性能优化 Stage 3 的优化清单）：

1. **kernel 时间占比多少？** GPU 真正算的时间 ÷ 总时间。占比 40% = 六成时间在空隙里，先做流水线（异步预取/CUDA stream）再谈 kernel 优化。
2. **空隙是谁造成的？** 空隙对应 CPU 侧在干嘛（数据预处理、同步 memcpy、Python 循环）——用 NVTX 给代码段打标（`torch.cuda.nvtx.range("preprocess")`），时间轴上就能对号入座。
3. **kernel 之间是串行还是重叠？** H2D 拷贝（H2D = host to device）与 kernel 并行了吗？多 stream 是否真的并发？空隙从 60% 压到 5% 往往不写一行 CUDA，只改调度顺序。

每 kernel 的"算得好不好"属于 Nsight Compute（ncu）的领地：SOL（speed of light）指标给出"该 kernel 距离硬件上限还有多远"（compute/mem 两个百分比），就是 03 篇 IPC 思想在 GPU 的翻版。**经验做法**：GPU 排查顺序 = nvidia-smi 定型（第 1 节三分法）→ nsys 找空隙 → ncu 进 kernel；跳过 nsys 直接抠 kernel 是最常见的弯路。

## 3. 显存带宽：为什么"算力没用满"

推理负载多数是 memory-bound（访存受限）：Transformer 解码阶段每生成一个 token 要把整个模型权重读一遍，算术强度（FLOPs/字节）低，**速度≈显存带宽 ÷ 模型大小**。粗算：7B 模型 fp16 约 14GB，带宽 1000GB/s 的卡理论上限 ≈ 70 token/s——实测能到 60+ 就说明工程做得不错，这比"GPU-Util 100%"有价值得多。

带宽上限速查（以官方规格为准）：RTX 4090 ≈ 1.0TB/s（GDDR6X），A100 SXM ≈ 2.0TB/s，H100 SXM ≈ 3.35TB/s（HBM3）。**端侧的残酷现实（ARM 注）**：RK3588 LPDDR4x 约 25~30GB/s（约为 4090 的 1/30），所以同量化等级下 7B 模型在 RK3588 上只有个位数 token/s 是物理定律，不是调优失败——端侧优化方向是"减字节"（更狠的量化/更小的模型/算子融合），05 篇的工作集思维在带宽预算上重现。

## 4. ARM 端侧：/sys 里的一手数据

RK3588（4×A76 + 4×A55 + Mali-G610 + 6TOPS NPU）上，温度/频率/占用都是标准 sysfs 节点（路径以实际内核为准，探测命令附在注释里）：

```bash
# 温度：thermal zone，单位毫摄氏度（/sys/class/thermal/ 下枚举）
$ cat /sys/class/thermal/thermal_zone0/temp
62000
$ for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type): $(($(cat $z/temp)/1000))C"; done
soc_thermal: 62C        # SoC 主温区；GPU/NPU 常有独立温区（sdk 命名不同）

# CPU 频率：cpufreq，每个 cluster 一个 policy（RK3588：policy0=A55×4，policy4/6=A76）
$ cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq
1800000                 # kHz；A76 大核当前 1.8GHz
$ cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_available_governors
schedutil performance powersave conservative ondemand

# GPU 频率与占用：devfreq 框架（Mali-G610）
$ cat /sys/class/devfreq/fb000000.gpu/cur_freq 2>/dev/null || find /sys/class/devfreq -name cur_freq | head -1
1000000000
$ cat /sys/class/devfreq/fb000000.gpu/load 2>/dev/null
38 0                    # NPU/GPU 负载与总时间（格式随驱动版本）

# NPU 占用（RKNN 驱动）：debugfs 节点，路径以厂商 SDK 为准
$ sudo cat /sys/kernel/debug/rknpu/load
NPU load:  62%          # RKNPU 驱动常暴露此节点；rknn 工具链的 rknn_server/性能分析工具走同一数据源
```

一个通用的"端侧体检"监控脚本（**经验做法**：把五路数据合并成一行，方便 `tee` 成日志回放）：

```bash
#!/bin/bash
# edge_watch.sh —— 端侧 1Hz 体检：频率/温度/NPU/内存一次看全
while true; do
  F1=$(awk '{printf "%d", $1/1000}' /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)
  F4=$(awk '{printf "%d", $1/1000}' /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq)
  T=$(awk '{printf "%d", $1/1000}' /sys/class/thermal/thermal_zone0/temp)
  NPU=$(sudo cat /sys/kernel/debug/rknpu/load 2>/dev/null | grep -o '[0-9]*%')
  M=$(free -m | awk '/Mem:/{print $3"/"$2"MB"}')
  echo "$(date +%T) A55=${F1}MHz A76=${F4}MHz temp=${T}C NPU=${NPU:-n/a} mem=${M}"
  sleep 1
done
```

## 5. cpufreq governor：一档开关，数倍差距

governor（调频策略）直接决定 CPU 跑多快（内核文档 `Documentation/admin-guide/pm/cpufreq.rst`）：

- **schedutil**：现代默认，跟随负载动态调频，兼顾省电与性能，但有**升频延迟**（负载出现→频率拉满需几十 ms）。
- **performance**：锁最高频率。延迟最稳，功耗/发热最高——**延迟敏感的推理服务在市电供电时推荐**。
- **powersave/conservative**：保守升频，跑分能差出数倍。

实测思路（RK3588 上执行）：

```bash
# 实验设计：同一个 CPU 密集任务（如 Python 预处理 10 万张图 resize），两种 governor 各跑一次
$ echo schedutil | sudo tee /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
$ time python3 preprocess_bench.py          # 记录墙钟 + edge_watch.sh 里 A76 的平均频率
$ echo performance | sudo tee /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
$ time python3 preprocess_bench.py          # 对比：差距通常 20%~200%，取决于任务是否短促突发
```

判读要点：**短促突发的负载（请求驱动型服务）受升频延迟伤害最大**，schedutil 下每次"睡醒接活"都要先跑几百毫秒低频——这正是端侧推理服务"首 token 慢"的常见系统侧原因（配合 04 篇的调度延迟一起定位）。**ARM 注**：改 governor 需要 root 且重启后失效，量产做法是 init 脚本或 `cpupower frequency-set -g performance` 写进服务启动链。

## 6. 端侧性能清单：热节流与带宽瓶颈

**热节流（thermal throttling）**是端侧性能的头号杀手，机理与识别：

```
负载上升 → 温度爬升 → 触发温控阈值 → 内核/固件降频 → 性能下降
        ↑______________ 温度回落后恢复 ______________|
识别特征（edge_watch.sh 的日志里一眼可见）：
  前半段: A76=2256MHz temp=71C NPU=85%   ← 满血
  后半段: A76=1416MHz temp=89C NPU=48%   ← 温度顶格、频率被压：节流发生
```

**经验做法**清单（拿到板子 30 分钟内做完）：

1. **跑 10 分钟压力录曲线**（`stress-ng --cpu 8` + 业务负载），确认"频率是否随时间下降"——下降即节流，先解散热（风扇/导热垫/外壳）再谈优化，否则任何优化都会被温控抹平。
2. **查供电**：USB 供电的板子在峰值时可能掉压强制降频，`dmesg | grep -iE 'voltage|thermal'` 与厂商温控节点（如 `dmc`/`gpu` 频率联动）必查。
3. **量内存带宽**：跑 `mbw` 或 stream 类工具拿实测带宽，对照 LPDDR 规格打折扣（实际通常 60%~80%）；推理 token/s 与带宽的比值是端侧最有效的健康指标（第 3 节的算法）。
4. **检查 eMMC/UFS 顺序读**：`fio --rw=read --bs=1M`（06 篇同款），模型加载慢先看这里；再确认模型是否被页缓存热住（vmtouch）。
5. **频率/温度/NPU 采集进日志**：任何"跑得慢"的复盘都要有这份数据，口说无凭。

## 7. 动手实验：诊断一个“越跑越慢”的端侧推理

症状设定（RK3588 + NPU 目标检测服务）：冷启动 30FPS，10 分钟后掉到 19FPS。用本篇工具把"越跑越慢"四字拆成数据。

**第一步：症状量化。** 挂上 edge_watch.sh 跑 10 分钟（第 4 节脚本），关键片段：

```
15:02:10 A55=1800MHz A76=2256MHz temp=64C NPU=82% mem=3.2/7.6GB   ← 冷启动
15:05:10 A55=1800MHz A76=2256MHz temp=78C NPU=80% mem=3.4/7.6GB
15:08:10 A55=1200MHz A76=1416MHz temp=89C NPU=47% mem=3.5/7.6GB   ← 温度顶格，频率/负载齐掉
15:10:10 A55=1200MHz A76=1416MHz temp=89C NPU=46% mem=3.6/7.6GB
```

**第二步：归因。** 内存稳定（排除 05 篇泄漏）、NPU 负载从 82% 掉到 47% 而输入队列积压（说明 NPU 吞吐掉了）——下降点与温度顶格时刻重合，**热节流实锤**。交叉验证：临时用散热风扇直吹 SoC 重跑，若 FPS 恢复且频率不再下降，证据闭环。

**第三步：修复与验证。** 分两层：

- 硬件层（治本）：加导热垫/风扇，重跑确认 10 分钟内 temp ≤ 78C 且 A76 维持 2256MHz。
- 软件层（治标兜底）：把 governor 固定 performance + 温控节点里放宽软限制（以 SDK 文档为准）；改后日志：

| 指标 | 冷启动 | 修复前 10min | 修复后 10min |
| --- | --- | --- | --- |
| FPS | 30 | 19 | 28 |
| A76 频率 | 2256MHz | 1416MHz | 2256MHz |
| 温度峰值 | 64C | 89C | 81C |
| NPU 负载 | 82% | 47% | 79% |

**结论句式**：性能衰减由热节流导致（温度-频率-负载三线证据），散热改造恢复 96% 的持续性能；残余差距来自带宽预算内 NPU 输入预处理的 CPU 竞争（下一轮优化对象）。**经验做法**：端侧所有性能结论必须附"温度-频率"曲线，否则指标可能只是节流的副产品，复现性无从谈起。

## 常见误区

1. **"GPU-Util 98% = GPU 高效"**：util 只说"有 kernel 在跑"，空隙与效率要 nsys 时间轴；"sm/mem/空隙"三分法才是有效诊断。
2. **"显存占满 = 显存不足"**：PyTorch caching allocator 的保留显存不释放是常态；判断真不足看"新分配是否 OOM"而非水位。
3. **"端侧跑分低就是板子差"**：先查热节流与供电，再查 governor，最后才怀疑硬件能力；三者抹掉的分数常超过模型优化能赚的。
4. **"Nsight 全家桶随便跑生产"**：ncu 的重放机制（kernel 反复执行）对延迟是灾难性干扰，只用于离线剖析；nsys 采样模式才可低开销长跑。
5. **"sysfs 路径照抄就行"**：端侧内核版本/厂商补丁差异大，本文路径是"常见形状"；用 find/grep 探测实际节点是端侧基本功（**ARM 注**）。

## 本篇小结

- nvidia-smi 的 GPU-Util 是"有活就算忙"，效率诊断用 dmon 的 sm/mem 二分 + nsys 时间轴的空隙分析；每 kernel 深度归因交给 ncu 的 SOL。
- 推理多为 memory-bound：token/s 上限 ≈ 显存带宽 ÷ 模型字节数；端侧带宽低 30 倍，优化第一优先级是减字节。
- 端侧观测一手数据在 sysfs：thermal（温度）、cpufreq（频率/governor）、devfreq（GPU/NPU）、rknpu/load（NPU 占用）；edge_watch.sh 一行式采集是通用起手式。
- governor 决定升频延迟，突发负载型服务建议 performance；热节流的识别特征是"温度顶格 + 频率下降 + 负载下降"三线同步。
- 端侧清单：压力录曲线查节流 → 查供电 → 量带宽 → 测存储 → 数据进日志；一切结论以温度-频率曲线背书。

## 练习

1. 在你的 WSL2 + NVIDIA GPU（如有）上跑 `nvidia-smi dmon -s pucvmet -d 1`，同时跑一个 `python3 -c "import torch; ..."` 小矩阵乘循环，观察 sm/mem 列的形状，判断它属于第 1 节三分法中的哪一类。
2. 无 GPU 环境练习：手绘第 2 节的 ASCII 时间轴变体——"H2D 拷贝与 kernel 重叠"的理想形状与"串行"的反例形状，标注每个空隙对应的 CPU 侧动作。
3. 在任意 Linux（含 WSL2）上演练 sysfs 读取：`cat /sys/class/thermal/thermal_zone*/temp` 与 `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`（WSL2 可能无 cpufreq 节点，记录你环境的缺失项，作为"环境清单"的一部分）。
4. 写出第 5 节的 governor 对比实验完整脚本（含两种策略各跑 3 次取中位数、用 edge_watch.sh 记录频率），在有 cpufreq 的机器上执行并填表；无条件则用 `stress-ng --cpu 8` + `vmstat 1` 观察不同 nice 下 us 列的变化替代。
5. 给你手头（或假想）的 RK3588 项目写"性能底细一页纸"：板卡型号、内存实测带宽、eMMC 顺序读、满载温度/频率、governor 策略、NPU 峰值利用率——以后所有调优都以此为基线。
