# 12 - 分布式存储案例与LLM时代的分布式

## 本篇目标

- 用前 11 章的框架拆解四个真实系统：Redis、etcd、ZooKeeper、Cassandra，看懂它们每条设计选择对应哪章理论。
- 论证"大模型训练为什么天生是分布式问题"，掌握数据并行的 AllReduce（含 Ring AllReduce 手写模拟）。
- 建立张量并行与流水线并行的直觉，分清参数服务器与 AllReduce 两条梯度同步路线。
- 概览推理侧的分布式：KV cache 分片、PD 分离，衔接库内 VLM Stage 8 的多机推理。

## 1. 案例拆解：四个系统各是哪几章的组合

理论只有一套，组合各显神通。先给四个系统各画一张"配方卡"：

| 系统 | 复制 | 一致性/共识 | 分区 | 定位一句话 |
| --- | --- | --- | --- | --- |
| Redis Cluster | 主从 + 哨兵故障切换 | 异步复制（无共识）→ 弱一致 | 16384 固定 slot（第 06 章） | 低延迟数据结构服务器 |
| etcd | Raft 多数派日志（第 05 章） | 线性一致读（ReadIndex/Lease） | 不分区（单 Raft 组，靠内存装） | 一致性元数据小存储 |
| ZooKeeper | ZAB（类 Raft 的原子广播） | 写线性一致、读可读旧 | 不分区 | 协调服务（选主/锁/配置） |
| Cassandra | 无主 Dynamo 派生（第 04 章） | 可调一致性（Quorum） | 一致性哈希 + vnode | 高可用写入型宽表 |

**Redis**：单线程命令执行（OS 09 章 Reactor）+ 内存数据结构；持久化 RDB/AOF；Cluster 用 slot 分区 + gossip 传播路由表（第 11 章）。要点：主从是**异步复制**，哨兵 failover 时"已确认写"可能丢（第 04 章的结论在这里直接适用）——Redis 强在延迟与功能，弱在绝对不丢。

**etcd**：标准 Raft 生产实现（WAL + 快照 + 单节点成员变更），所有写走 Raft 日志多数派，读可选线性一致（ReadIndex）。K8s 用它存全部集群状态。要点：**etcd 是"小而强一致"**——存储量建议 GB 级以内，别拿它当数据库（它是元数据的脑子，不是数据的仓库）。

**ZooKeeper**：ZAB（ZooKeeper Atomic Broadcast，与 Raft 同类的崩溃容错共识，论文 *ZooKeeper: Wait-free coordination for Internet-scale Systems*, USENIX ATC 2010）。API 形态是文件树 + watch；临时节点（ephemeral）+ 会话超时实现活锁检测，是"选主/服务注册"的经典原料。要点：读默认走 follower（可能旧），要线性一致读需 sync() 前置——第 03 章"贴 CP 标签 ≠ 所有读强一致"的活例。

**Cassandra**：Dynamo 论文的工程化（一致性哈希 + vnode + 可调 W/R + hinted handoff + 读修复），存储模型是宽列族。要点：**写路径极简**（追加 commit log + memtable）→ 写吞吐碾压同类，代价是读放大与 tombstone（删除标记）清理的复杂性——AP 侧的极致取舍。

## 2. 为什么大模型训练是分布式问题

三条硬约束（第 01 章天花板的 LLM 版）：

1. **放不下**：175B 参数 fp16 权重 350GB + 梯度 350GB + 优化器状态（Adam 两个动量，fp32）1400GB ≈ 2.1TB，单卡 80GB 完全无望（Shoeybi et al., *Megatron-LM*, arXiv 2019 首次系统拆解此问题）。
2. **算不动**：GPT-3 训练量 3.14e23 FLOPs，单 A100（312 TFLOPS bf16）不吃不睡要 3 万年；数千卡并行数月是常态（计算依据：训练 FLOPs ≈ 6 × 参数量 × token 数，Kaplan et al., *Scaling Laws for Neural Language Models*, arXiv 2020 附录）。
3. **等不起**：即便放得下算得动，单卡迭代一个 step 的时间（内存带宽限制）也远超业务窗口。

于是训练系统 = 把一个巨型 SGD 循环切到 N 卡 × M 机上：切参数（张量并行）、切层（流水线并行）、切数据（数据并行），三者可组合（3D 并行，Megatron-DeepSpeed 体系）。

## 3. 数据并行与 AllReduce（手写 Ring AllReduce）

**数据并行（DP）**：每张卡放完整模型，各吃一份不同数据（micro-batch），前向反向各算各的；**唯一需要通信的是反向结束后的梯度汇总**——把 N 份梯度求平均， everybody 用同一个平均值更新。梯度求和+广播就是 **AllReduce**。

朴素实现：参数服务器（第 3 节）或"主卡收集所有梯度求平均再广播"——主卡带宽是瓶颈（N-1 份上传 + N-1 份下载）。

**Ring AllReduce**（NCCL/Baidu 的标准实现，Baidu 2017 博客《Bringing HPC Techniques to Deep Learning》把 HPC 的 ring 算法带给 DL）：把 N 卡连成环，梯度切成 N 块，两阶段各 N-1 步：

```
阶段一 Reduce-Scatter（每步：收前一块，加到自己对应块，传下一块）
  step1: 卡i 把自己 chunk(i-1) 发给卡 i+1
  ...N-1 步后：每张卡"拥有"一个完整规约好的 chunk（各不相同）
阶段二 AllGather（同方向再传 N-1 步，把规约好的块广播回所有人）
```

**通信量证明**（面试常考）：每卡每阶段发 (N-1)×(总梯度/N) 大小，两阶段合计 `2×(N-1)/N × 梯度总量`——**与卡数 N 几乎无关**（N 很大时趋近 2 倍梯度量），而参数服务器的量是 O(N) 倍。这就是 ring 结构赢的原因：带宽压力被摊到环上每条边。

```python
# ring_allreduce.py —— 4 卡 Ring AllReduce 手写模拟：验证"两阶段、通信量 2(N-1)/N"
import asyncio

N = 4
grads = [[k * 1.0 for k in range(N)] for _ in range(N)]   # 每卡一份"梯度"（各不相同）
# 目标：每卡得到全体的逐元素平均

async def ring_allreduce(node_i: int, buf: list, mailbox: list):
    """节点 i 在环上做 ReduceScatter + AllGather。buf 长度 == N（每块一个元素）"""
    left, right = (node_i - 1) % N, (node_i + 1) % N
    # ---- 阶段一: Reduce-Scatter（N-1 步）----
    for step in range(N - 1):
        chunk = (node_i - step) % N            # 本步我要积累的块号（环上错开）
        await mailbox[node_i].put(("chunk", chunk, buf[chunk], node_i))
        src, c, val, _ = await mailbox[node_i].get()
        buf[c] += val                          # 收到左邻的块 → 累加（规约）
    # ---- 阶段二: AllGather（N-1 步）----
    for step in range(N - 1):
        chunk = (node_i + 1 - step) % N        # 本步我要广播出去的块
        await mailbox[node_i].put(("chunk", chunk, buf[chunk], node_i))
        src, c, val, _ = await mailbox[node_i].get()
        buf[c] = val                           # 覆盖为规约好的最终值
    return buf

async def main():
    mailbox = [asyncio.Queue() for _ in range(N)]
    bufs = [list(g) for g in grads]
    expected = [sum(g[k] for g in grads) / N for k in range(N)]

    # 让所有节点并发跑（消息用共享队列组模拟网络；收发按环配对简化处理）
    for step_round in range(N - 1):            # 手动展开步进以避免死锁，等价于两阶段循环
        sends = []
        for i in range(N):
            chunk = (i - step_round) % N
            sends.append(mailbox[(i + 1) % N].put(("chunk", chunk, bufs[i][chunk], i)))
        await asyncio.gather(*sends)
        gets = []
        for i in range(N):
            msg = await mailbox[i].get()
            _, c, val, _ = msg
            bufs[i][c] += val                  # Reduce-Scatter 累加
    for step_round in range(N - 1):            # AllGather
        sends = []
        for i in range(N):
            chunk = (i + 1 - step_round) % N
            sends.append(mailbox[(i + 1) % N].put(("chunk", chunk, bufs[i][chunk], i)))
        await asyncio.gather(*sends)
        for i in range(N):
            _, c, val, _ = await mailbox[i].get()
            bufs[i][c] = val                   # AllGather 覆盖

    for i in range(N):
        ok = all(abs(bufs[i][k] - expected[k]) < 1e-9 for k in range(N))
        print(f"卡{i}: {bufs[i]} 平均正确: {ok}")
    print(f"理论通信量: 每卡 2*(N-1)/N 倍梯度 = {2*(N-1)/N:.2f}x（参数服务器为 ~2x 恒定但在主卡集中）")

asyncio.run(main())
```

**经验做法**：理解 NCCL 不必读 CUDA，把上面 4 卡模拟跑通即可建立"环上错位流水"的心智模型；实战中 `torch.distributed.all_reduce(grad, op=SUM)` 一行搞定，NCCL 自动选环/树拓扑。通信量数字（2(N-1)/N）直接回答"为什么 8 卡机内 NVLink 全互联也要按环走"。

## 4. 张量并行与流水线并行：直觉

**张量并行（TP）**：把**一层内部**的大矩阵切开。以线性层 Y = XW 为例（Megatron-LM 论文的核心）：按列切 W（每卡拿到 W 的列块），X 全员相同，各卡算出 Y 的列块 → AllGather 拼接；反向时对输入梯度做 AllReduce。前向反向**每个线性层都要通信 1~2 次**，因此 TP 的通信在关键路径上——**经验做法：TP 只放机内 NVLink（带宽 ~900GB/s），不跨机**（跨机网 ~25-50GB/s，通信会淹没计算）。

**流水线并行（PP）**：把**层间**切开（前 1/4 层在卡1、次 1/4 在卡2……），micro-batch 依次流过。难点是"气泡"（bubble）：卡2 必须等卡1 完成 micro-batch0 才有活干。气泡率公式（GPipe 论文，Huang et al., arXiv 2018）：`bubble = (p-1)/(m+p-1)`，p=流水线级数、m=micro-batch 数——**用更多 micro-batch 稀释气泡**（m 越大气泡越小）。1F1B 调度（PipeDream-Megatron 体系）让前向反向交错，把激活值内存从 O(m) 压到 O(p)。

```
时间→  卡1: F0 F1 F2 F3 B0 B1 B2 B3        (F=前向 B=反向, 4 micro-batch)
       卡2:    F0 F1 F2 F3 B0 B1 B2 B3
       卡3:       F0 F1 F2 F3 B0 B1 B2 B3
       卡4:          F0 F1 F2 F3 B0 B1 B2 B3   ← 开头与结尾的空档 = 气泡
```

用纯 Python 把张量并行的"切列"算一遍（不用 numpy，看清切分结构即可）：

```python
# tensor_parallel_demo.py —— 列并行线性层 Y = X·W 的 2 卡切分手工演算
# W 切成 [W0|W1]，每卡算 Y 的对应列块，AllGather 拼接
X = [[1.0, 2.0],                       # 输入 2x2（全员相同）
     [3.0, 4.0]]
W0 = [[1.0], [2.0]]                    # W 的第 0 列（卡0 持有，2x1）
W1 = [[3.0], [4.0]]                    # W 的第 1 列（卡1 持有，2x1）

def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]

y0 = matmul(X, W0)                     # 卡0 本地算出 Y[:, 0]
y1 = matmul(X, W1)                     # 卡1 本地算出 Y[:, 1]
y_allgather = [r0 + r1 for r0, r1 in zip(y0, y1)]   # AllGather：按行拼接两卡结果
y_true = matmul(X, [[1.0, 3.0], [2.0, 4.0]])        # 不切分的完整结果
print("两卡切列结果:", y_allgather)
print("完整矩阵结果:", y_true)
print("一致:", y_allgather == y_true, "—— 这就是 Megatron 列并行的全部数学")
```

```
$ python3 tensor_parallel_demo.py
两卡切列结果: [[5.0, 11.0], [11.0, 25.0]]
完整矩阵结果: [[5.0, 11.0], [11.0, 25.0]]
一致: True —— 这就是 Megatron 列并行的全部数学
```

再看清"行并行"的另一半：W 按行切（每卡拿 W 的行块），输入 X 必须先按列切好（每卡拿 X 的对应列块），各卡算出**完整 Y 的部分和**，最后 AllReduce 求和。Transformer 的 MLP 层正是"列切第一个线性层 + 行切第二个线性层"，两次切分之间**一次通信都不需要**（列并行的输出恰好是行并行需要的形状，Megatron-LM 论文 §3.2 的经典设计）——这就是为什么 Megatron 的 MLP 一层只通信 2 次（前向一次、反向一次），也是 TP 效率设计的精髓。

三种并行的正交组合（3D 并行的分工直觉，经验做法）：**DP 管吞吐**（梯度通信每步一次，通信占比低）、**TP 管单层放不下**（机内、通信频繁）、**PP 管深层放不下**（跨机、气泡靠 micro-batch 稀释）。典型 70B 训练：机内 TP=8，跨机 PP=4，数据并行再铺几十组。

## 5. 参数服务器 vs AllReduce

| 维度 | 参数服务器（PS） | AllReduce（集合通信） |
| --- | --- | --- |
| 架构 | worker ↔ 中央 server 组分片存参数 | 对等节点环形/树形互联 |
| 通信量 | 集中于 server（按参数量 × worker 数增长） | 均摊到环（2(N-1)/N） |
| 容错 | server 分片可冗余（PS-Lite） | 天然对等，节点挂需重启动或备份环 |
| 适用 | **稀疏模型**（Embedding/推荐，worker 各取所需参数）、异构集群 | **稠密梯度**（Transformer 全连接层）、同构 GPU 集群 |
| 代表 | Parameter Server 论文（Li et al., OSDI 2014）、PS-Lite、TensorFlow 1.x 分布式 | NCCL、Horovod（Sergeev & Del Balso, arXiv 2018）、DDP |

Transformer 训练之所以全走 AllReduce：梯度是稠密大矩阵，环上均摊最优；推荐系统embedding 表动辄 TB 级、每 batch 只碰其中一小撮，PS 的"按需拉取"才有意义。`torch.nn.parallel.DistributedDataParallel` 的默认实现就是"反向钩子里对每个参数桶做 all_reduce"，没有 server 节点。

## 6. 推理侧的分布式（衔接 VLM Stage 8）

推理与训练的分布式诉求不同——**单请求延迟优先，吞吐次之**：

- **KV cache 分片**：多卡推理时注意 KV cache 的并行归属——TP 内每卡只存自己切分的注意力头对应 KV；长上下文下 KV cache 显存爆炸（第 01 章 PagedAttention 的动机在 Stage 5 已讲），跨机用 TP2/TP4 是把"放得下"买回来的标准手段。
- **PD 分离（Prefill/Decode disaggregation）**：prefill（吃满算力，计算密集）与 decode（吃显存带宽，访存密集）负载特性相反，分开到不同节点各自扩容，中间用 KV cache 传输衔接（vLLM 的 disaggregated prefill、Mooncake 架构论文 *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*, FAST 2025 前身 arXiv 2024）。这是"按功能拆分（第 01 章 y 轴）"在 LLM 时代的重现。
- **多副本路由**：无状态推理副本 + 负载均衡 = 第 01 章 x 轴；副本感知模型加载时长，扩容要预热（模型加载分钟级，健康检查要区分"端口活了"与"能推理了"——第 11 章服务发现的健康检查分层）。

## 常见误区

1. **"Redis Cluster 是强一致的"**：它的复制是异步、故障切换靠哨兵/ gossip 投票，已确认写可能在切换时丢失（第 04 章结论）。要强一致存储请选 etcd/Raft 系，Redis 的价值在延迟与数据结构。
2. **"etcd 可以当业务数据库"**：Raft 全量写日志 + 内存索引的架构适合万级 QPS 的**元数据**；放业务数据会把 Raft 组压垮且没有查询能力。"小而强一致"是定位不是缺陷。
3. **"卡越多训练越快（线性加速）"**：通信占比随规模上升——DP 的梯度同步、PP 的气泡、TP 的层内通信都在吃并行度。加速比要用"计算/通信比"（如每卡 batch 大小、梯度桶大小、TP 是否机内）逐项核算，Scaling Law 告诉你需要多少卡，通信工程决定这些卡实际有效多少。
4. **"AllReduce 只能求和"**：reduce 操作是参数（SUM/AVG/MIN/MAX），ring 结构不变；且 AllReduce = ReduceScatter + AllGather 的组合可以拆开单独用（ZeRO-1 的梯度分片就用 ReduceScatter 代替全量 AllReduce，Rajbhandari et al., *ZeRO*, SC 2020）。
5. **"PD 分离就是把两个服务部两个机器"**：分离的实质是**KV cache 的传输与复用**（prefill 节点算好的 KV 流式搬运给 decode 节点），只分服务不解决 KV 搬运（带宽、时延、cache 命中路由）等于只做了 y 轴拆分的形式。

## 本篇小结

- 四个存储案例 = 前十一章理论的不同配方：Redis(低延迟+异步复制+slot 分区)、etcd(Raft 线性一致元数据)、ZooKeeper(ZAB 协调树)、Cassandra(Dynamo 无主宽表)；选型先问"一致性 vs 可用性 vs 延迟"的排序。
- LLM 训练三条硬约束（放不下/算不动/等不起）催生 3D 并行：DP 管吞吐、TP 管单层、PP 管深层。
- Ring AllReduce 通信量 2(N-1)/N 与规模无关，是稠密梯度同步的正解；手写 4 卡模拟足以建立心智模型。
- PS 与 AllReduce 的分界线在**梯度稠密性**：稠密走环，稀疏（Embedding）走 server。
- 推理侧分布式三件事：KV cache 归属与显存治理、PD 分离（y 轴拆分的 LLM 版）、多副本路由与健康分层——与库内 Stage 5/8 的实战直接对上。

## 练习

1. 跑通 ring_allreduce.py，把 N 改成 8 并在收发处计数消息总字节数，验证每卡通信量 = 2(N-1)/N × 数据量；对比"主卡收集广播"实现的通信量曲线。
2. 把 ring_allreduce.py 改成 ReduceScatter only（去掉 AllGather 阶段），模拟 ZeRO 的梯度分片：每卡只规约出自己负责的 1/N 参数梯度，写出更新自己那片的伪代码。
3. 用 GPipe 气泡公式 `(p-1)/(m+p-1)` 计算 p=4、m=1/4/16/64 的气泡率并画出曲线；结合 1F1B 的激活内存 O(p) 结论，给出"micro-batch 怎么选"的决策规则。
4. 若本机有 Docker：用 docker compose 起 3 节点 etcd 集群，`etcdctl put/get` 验证线性一致读（对比 `--consistency=s` 与 `l`），再 kill 掉 leader 观察 5 秒内重新选主——把第 05 章选举模拟与生产实现对上。
5. 设计题：一个 70B 模型的推理服务要支撑 500 并发长上下文（32K token）请求。给出你的并行策略（TP/PP/副本数）、KV cache 显存估算（fp16 KV 每token每层 2×hidden×2 字节，层数×hidden 从模型卡查）、以及是否 PD 分离的判断依据。
