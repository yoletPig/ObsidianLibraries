# 03 - CAP与一致性谱系

## 本篇目标

- 给出 CAP 定理的严格表述，拆掉"三选二"这句流行口号的三大误读。
- 补上 CAP 缺失的另一半：PACELC（没有分区时，延迟与一致性也要取舍）。
- 建立从强到弱的一致性谱系：线性一致 → 顺序一致 → 因果一致 → 最终一致，并配读写时序判定。
- 分清"读己之写 / 单调读 / 一致前缀读"这些会话保证，知道业务到底需要哪一档。

## 1. CAP 的严格含义：分区时才谈取舍

CAP 定理（Eric Brewer 在 2000 年 PODC 上的猜想，Gilbert & Lynch 在 2002 年正式证明：*Brewer's Conjecture and the Feasibility of Consistent, Available, Partition-Tolerant Web Services*, ACM SIGACT News）：

- **C（Consistency）**：线性一致性（linearizability）——所有读写表现得像操作一个副本（具体定义见第 4 节）。
- **A（Availability）**：每个**非故障节点**收到的请求都必须得到（非错误的）响应。
- **P（Partition Tolerance）**：网络允许任意分区（消息丢失/延迟）。

严格的定理表述只有一句话：**当网络分区发生时，系统不可能同时保证 C 与 A，必须二选一**。

三大误读逐一纠正：

1. **"分布式系统必须在 C/A/P 三个里选两个"**——错。P 不是可选项：网络分区是物理现实（02 章已论证），你不能"选择不容忍分区"，只能选择分区发生时放弃 C 还是放弃 A。平时没有分区时，C 和 A 可以同时成立。
2. **"C/A/P 是系统级属性，选了 C 就是 CP 系统"**——错。取舍发生在**每次操作**层面：同一个系统可以一部分操作要强一致（如账户扣款）、一部分操作要可用（如商品浏览）。给系统贴"CP/AP"标签只是粗略画像。
3. **"分区结束后就可以高枕无忧"**——错。分区期间积累的不一致要在分区恢复后**收敛**（反熵、读修复，第 04 章），收敛期间读到旧值仍是常态。

一个必须会算的推导：分区发生时，集群分成少数派与多数派（比如 3 副本断成 1+2）。若少数派也接受写入（可用性优先），不同分区各自演进，数据分叉 → 违反 C；若少数派拒绝写入（一致性优先），少数派上节点"可用"却无法服务 → 违反 A。**只能选一边，数学上没有第三条路**（Gilbert & Lynch 证明的核心：任何同时满足 C+A 的算法都构造不出能容忍一条消息丢失的场景）。

```python
# cap_demo.py —— 分区时 C 与 A 二选一的现场演示
import asyncio

class Replica:
    def __init__(self, name):
        self.name = name
        self.store = {}
        self.online = True
        self.peers = []               # 网络视图：分区时对端列表会变化

    async def write(self, key, value, require_quorum=True):
        """写：一致性优先时要求多数派确认（这里 3 副本即至少 2 个）"""
        if not self.online:
            raise ConnectionError(f"{self.name} 不可达")
        acks = 1
        others = [p for p in self.peers if p.online]
        if require_quorum:
            for p in others:                    # 试图复制给其他副本
                try:
                    await asyncio.sleep(0.01)   # 网络延迟
                    p.store[key] = value
                    acks += 1
                except ConnectionError:
                    pass                        # 分区期间发不到的副本就发不到
        self.store[key] = value
        if require_quorum and acks < 2:
            del self.store[key]                 # 拿不到多数派：拒绝服务（选 C 弃 A）
            raise PermissionError(f"{self.name}: 多数派不可达，拒绝写入（CP）")
        return acks

    async def read(self, key):
        if not self.online:
            raise ConnectionError(f"{self.name} 不可达")
        return self.store.get(key, None)

async def main():
    r1, r2, r3 = Replica("R1"), Replica("R2"), Replica("R3")
    for r in (r1, r2, r3):
        r.peers = [x for x in (r1, r2, r3) if x is not r]
    await r1.write("user:1", "A")               # 正常时 C 与 A 同时成立
    print("正常写:", await r1.read("user:1"))

    # ===== 网络分区：r1 与其余副本失联 =====
    r2.online = r3.online = False               # 用"其余副本失联"模拟 r1 被分到少数派
    try:
        await r1.write("user:1", "B", require_quorum=True)
    except PermissionError as e:
        print("CP 行为:", e)                     # 选 C：拒绝写 → 少数派不可用
    r2.online = r3.online = True

    # ===== 换成 AP 模式：允许单副本写 =====
    await r1.write("user:1", "C", require_quorum=False)
    print("AP 写入成功:", await r1.read("user:1"), "← 但此刻 r2/r3 上还是旧值（违反 C）")

asyncio.run(main())
```

## 2. PACELC：没有分区时，延迟与一致性也在取舍

CAP 没有回答"没有分区时怎么办"——答案是 PACELC（Daniel Abadi, *Consistency Tradeoffs in Modern Distributed Database System Design*, IEEE Computer 2012）：**如果分区（P），在可用性（A）与一致性（C）间取舍；否则（Else），在延迟（L）与一致性（C）间取舍**。

解读：即使没有分区，强一致也需要等副本确认（同步复制跨一次网络），延迟必然更高；最终一致可以本地写入立即返回，代价是读到旧值。常见的真实分类：

| 系统 | PACELC 画像 | 含义 |
| --- | --- | --- |
| ZooKeeper / etcd | PC/EC | 任何情况下都付一致性的延迟代价 |
| 传统 MySQL 单主 + 半同步 | PA/EL（可调） | 参数决定选边 |
| Cassandra / Dynamo | PA/EL | 分区时可用优先，平时低延迟优先 |
| Spanner | PC/EC（用 TrueTime 压低 EC 代价） | 一致性优先，且诚实支付延迟 |

经验做法：给每个存储写下它的 PACELC 画像，架构评审时对号入座——"我们的库存扣减能不能容忍 Cassandra 式 PA/EL"这类问题会变得非常具体。

## 3. 一致性谱系：从线性一致到最终一致

"一致性"不是二值属性，是一条谱带。按约束强度从强到弱（分类框架参考 DDIA 第 9 章与 Adya 博士论文 *Weak Consistency: A Generalized Theory...*, MIT 1999）：

**线性一致性（linearizability / 强一致）**：整个系统表现得像只有一个副本，且每个操作在"发起"与"完成"之间存在一个原子时刻。保证：任何读都能读到"刚刚完成"的写；所有客户端看到同一个操作顺序，且该顺序与实时时间一致。代价：每次读写都要与多数派/主副本通信（跨网络）。

**顺序一致性（sequential consistency）**：所有客户端看到同一个操作顺序，但这个顺序**不必与真实时间一致**（Lamport, *How to Make a Multiprocessor Computer That Correctly Executes Multiprocess Programs*, IEEE TC 1979）。可以理解为"时间被抽掉的线性一致"。

**因果一致性（causal consistency）**：只有**有因果关系**的操作需要保序（用 02 章的向量时钟判定），并发操作在不同副本可以不同序。弱于顺序一致，但通常足够。

**最终一致性（eventual consistency）**：停止写入后，经过不确定时间，所有副本收敛到相同值。对并发写只承诺"最终相同"，不承诺收敛到哪个值（需要冲突解决：LWW、CRDT、应用层合并）。

用三个读写场景判定级别（假设 x 初始为 0）：

```
场景一（线性一致必须满足）：
  客户端A: write(x=1) ──完成──►
  客户端B:                        read(x) → 必须读到 1
场景二（顺序一致允许、线性一致不允许）：
  时刻 t1: A write(x=1) 完成（此刻在真实时间上早于 t2）
  时刻 t2: B read(x) 读到 0        ← 顺序一致可以：只要全局序里 B 的读排在写前
场景三（因果一致允许、顺序一致不允许）：
  A: write(x=1) 并发于 C: write(x=2)，不同副本读到 1、2 的顺序可以不同
  但 A 写后 B 读到 1，则 B 之后读永远不能再"回退"到 0
```

**经验做法**：问自己一个问题——"读到一秒前的旧值，业务会出什么事？"答案是钱、库存、唯一性约束 → 线性一致；答案是 feed 流、点赞数、推荐结果 → 最终一致；答案是评论与回复的先后 → 至少因果一致。

```python
# consistency_spectrum.py —— 主从异步复制下，三种会话保证的违反现场
import asyncio, random

class AsyncStore:
    """主从异步复制：主副本写完立即返回，从副本慢慢追"""
    def __init__(self):
        self.leader = {}                 # 主副本数据
        self.follower = {}               # 从副本数据
        self.replication_delay = 0.05

    async def write(self, k, v):
        self.leader[k] = v               # 只写主，立刻返回（低延迟）
        async def replicate():
            await asyncio.sleep(self.replication_delay + random.random() * 0.03)
            self.follower[k] = v         # 异步复制到达
        asyncio.create_task(replicate()) # 模拟异步复制

    async def read(self, k, from_leader=False):
        return self.leader.get(k) if from_leader else self.follower.get(k)

async def main():
    s = AsyncStore()
    await s.write("price", 100)
    await asyncio.sleep(0.1)             # 等复制追上，双方一致

    # ===== 违反"读己之写"：写主读从 =====
    await s.write("price", 200)
    print("写后立刻从从副本读:", await s.read("price"), "← 自己刚写的值没看见")
    await asyncio.sleep(0.1)

    # ===== 违反"单调读"：两次读打到不同副本 =====
    follower_stale = AsyncStore()
    follower_stale.leader = follower_stale.follower = {"k": 1}
    await follower_stale.write("k", 2)
    await asyncio.sleep(0.02)            # 从副本还没追上
    r1 = await follower_stale.read("k", from_leader=True)   # 第一次读打到主：读到 2
    r2 = await follower_stale.read("k")                     # 第二次读到从：读到 1
    print(f"单调读违反: 第一次读 {r1} → 第二次读 {r2}（时间倒流）")

asyncio.run(main())
```

## 4. 会话保证：读己之写、单调读、一致前缀读

客户端视角的三个常见保证（DDIA 第 5 章《复制滞后的问题》系统论述）：

- **读己之写（read-your-writes）**：用户自己写入的数据，自己马上能读到。实现：写后短时间内把该用户的读路由到主副本，或记录"用户最近写入时间戳"、读到比它旧的值就换副本重读。
- **单调读（monotonic reads）**：一个用户连续两次读，第二次不能比第一次更旧。违反现场：第一次读到主（新值），第二次负载均衡到滞后的从（旧值）——"时间倒流"。实现：同一用户粘住同一副本，或副本记录已服务的时间戳、落后太多就拒绝。
- **一致前缀读（consistent prefix reads）**：先看到答案后看到问题是不允许的（因果序的读侧版本）。违反现场：分片存储中"回复"所在分片复制快于"原帖"所在分片。实现：把因果相关的数据放同一分片，或用版本向量校验。

这三个保证都不要求全系统线性一致——它们是"按会话"的弱一致约束，能用低延迟实现，且覆盖了 90% 的实际业务诉求（经验做法：优先实现会话保证，再谈全局强一致）。

## 5. BASE 与一致性的工程选择

BASE（Basically Available, Soft state, Eventually consistent）是 eBay 工程师在 2008 年提出的与 ACID 对立的实践口号（Dan Pritchett, *BASE: An ACID Alternative*, ACM Queue 2008）。它不是理论，是给"AP 侧系统"的行为画像：基本可用（分区时降级服务）、软状态（允许中间态）、最终一致。

给业务选一致性等级的决策表（经验做法）：

| 业务 | 需要什么 | 典型选择 |
| --- | --- | --- |
| 账户余额、库存扣减、唯一 ID | 线性一致（或串行化事务） | 共识系统上的存储（etcd/ZooKeeper/带 Raft 的 KV） |
| 订单创建后的查询 | 读己之写 | 主副本读 + 写后粘滞 |
| 用户资料修改 | 读己之写 + 单调读 | 会话粘滞路由 |
| feed 流、评论树 | 因果一致 | 带版本/时间戳的合并 |
| 浏览计数、点赞数 | 最终一致 | 计数器 + 定期合并 |
| 全局排行榜 | 因果 + 单调读 | Redis 单点/分片 + 快照 |

## 常见误区

1. **"我们选了 CP 系统（如 ZooKeeper），所以数据一定强一致、无后顾之忧"**：CP 只描述分区时的行为；ZooKeeper 默认读走本地 follower（不保证线性一致，除非 sync() 后读）。"贴了 CP 标签"不等于"所有读都线性一致"，要看具体读路径的文档。
2. **"最终一致 = 数据会错很久"**：正常网络下收敛通常是毫秒到秒级；真正的问题是**收敛期间的读写交错**与**并发写冲突**，前者用会话保证缓解，后者需要冲突解决策略（第 04 章 LWW/CRDT）。把"最终"理解成"分钟级乱套"或"永远无所谓"都不对。
3. **"CAP 里 A 是'响应快'"**：CAP 的 A 只要求"非故障节点给出响应"，**与延迟无关**。延迟的取舍在 PACELC 的 L 上——这是把 CAP 用错最多的一处。
4. **"强一致就是事务"**：线性一致约束的是**单个对象**的读写序；事务约束的是**多个对象**的原子性与隔离级别（第 07 章）。一个线性一致的 KV 完全可能让两次 get 看到别人穿插的写——一致性等级与隔离级别是两个正交坐标系。
5. **"读己之写必须用强一致存储"**：不需要。会话粘滞（写后 30 秒内读主）或时间戳门槛就能在异步复制系统上实现，成本低得多（经验做法：把会话保证做在路由层）。

## 本篇小结

- CAP 严格含义：**分区发生时** C 与 A 二选一，P 不可选；取舍是操作级而非系统级标签。
- PACELC 补全了无分区时的取舍：L 与 C 的对赌（同步复制的延迟代价 vs 异步复制的旧值风险）。
- 谱系：线性一致 ⊃ 顺序一致 ⊃ 因果一致 ⊃ 最终一致；判定靠读写时序图，选择靠"读到旧值业务会怎样"。
- 会话保证（读己之写/单调读/一致前缀读）是低成本高覆盖的工程主力，优先于全局强一致。
- BASE 是 AP 侧的实践画像；给存储写 PACELC 画像是架构评审的实用动作。

## 练习

1. 跑通 cap_demo.py，把 `require_quorum=True` 时"拿不到多数派就拒绝"改成"接受写入并记录到待同步队列"，运行后观察 r2/r3 恢复时的数据分叉——写下这个方案还需要什么机制才能收敛（预告：第 04 章反熵）。
2. 对比实验：把 consistency_spectrum.py 的 `AsyncStore.write` 改成"写主 + 等从副本追上再返回"（同步复制），重跑两个违反场景，确认它们不再发生；测量写入延迟变化（`time.monotonic()`），对应 PACELC 的 L↔C 取舍。
3. 画出顺序一致但非线性一致的读写时序图（两个客户端、两个副本、一次异步复制），并用一句话向同事解释"为什么这个场景没毛病但线性一致不行"。
4. 分析题：给你常用的两个系统（比如 Redis 主从、MySQL 主从，或库里 Stage 7 用的向量库）写出 PACELC 画像，并设计它们的"读己之写"实现方案（说明在路由层需要记录什么状态）。
5. 证明题（选做）：仿照 Gilbert & Lynch 的思路，用两副本系统构造反证——假设存在算法在"一条消息丢失"下同时满足 C 与 A，写出矛盾发生的请求序列。
