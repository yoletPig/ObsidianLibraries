# 05 - 共识-Raft详解

## 本篇目标

- 定义共识问题，理解 Paxos 为什么"正确但难懂"，以及 Raft 的设计立场（可理解性优先）。
- 掌握 Raft 三角色、任期、领导者选举的完整规则，并用 Python 模拟选举与脑裂避免。
- 讲透日志复制与提交规则，论证"为什么必须多数派提交"与两条安全性子问题（选举限制、日志匹配）。
- 了解成员变更与日志压缩的概念，能说出生产 Raft 系统（etcd）对这些点的处理。

## 1. 共识问题：分布式系统的皇冠

共识（consensus）：**N 个节点对"某个值"达成一致，且该值必须由某个节点提议过**。更实用的形式是**状态机复制（replicated state machine）**：让 N 个节点以**相同的顺序**执行**相同的命令序列**，则它们的状态永远一致——共识算法的任务就是对这个命令序列达成一致。

为什么它难？因为 02 章的两个事实叠加：节点会宕机（可能永远回不来，也可能 5 秒后复活且不知中间发生了什么）+ 网络会分区（消息任意延迟/丢失）。在这个世界里，你要防的是：

- 两个节点同时认为自己是"决定者"（脑裂）。
- 一个决定在部分节点生效、部分节点不知道。
- 节点宕机重启后与系统状态脱节。

FLP 不可能性（Fischer, Lynch, Paterson, 1985）：在**异步**系统（消息延迟无上界）+ 哪怕只有一个节点可能崩溃的前提下，不存在保证 termination 的确定性共识算法。工程共识算法的出路：引入**超时/随机化**（把"异步"弱化为"部分同步"），放弃"最坏情况下必然终止"而保证"概率上快速终止"。Raft 的随机选举超时、Paxos 的 proposer 冲突退避，都是对 FLP 的工程回应。

## 2. Paxos 为什么难懂（直觉版）

Paxos（Leslie Lamport, *The Part-Time Parliament*, 1998；由于行文用虚构的希腊议会写成、晦涩难懂，业界哀嚎一片，Lamport 又写了 *Paxos Made Simple*, 2001，全文 13 页没有一条图）。它由两阶段组成：

- **Prepare(n)**：提议者选一个编号 n，问大家"你们见过比 n 更大的提议吗？"多数派承诺不再接受 < n 的提议，并回报自己已接受的最大提议。
- **Accept(n, v)**：若 prepare 过半通过，提议者用 prepare 中见到的最高提议的值 v（没有则自选），发出 accept，过半通过则被选定。

它正确，但直接工程化时问题成堆：单值 Paxos 如何扩展成日志（Multi-Paxos）需要自己补工程细节（leader 选举、日志空洞、leader 变更期间的日志对齐），论文几乎没写；状态机多、正确性论证绕；`Paxos 的活人实现少得可怜`（Ongaro 的博士论文原话之一：*There are significant gaps between the description of the Paxos algorithm and the needs of a real-world system*）。

Raft 的回答（Ongaro & Ousterhout, *In Search of an Understandable Consensus Algorithm*, USENIX ATC 2014；Ongaro 博士论文 *Consensus: Bridging Theory and Practice*, 2014）：**问题分解 + 状态空间缩减**。把共识拆成三个相对独立的子问题（领导者选举、日志复制、安全性），每个都有清晰的直觉；再用"强 leader"模型把一切都归到单一决策点，减少需要同时思考的状态组合。

## 3. Raft 三角色与任期

节点任一时刻处于三种角色之一：

```
        ┌────────────── 超时未收到心跳 ──────────────┐
        ▼                                            │
   [Follower] ──── 任期内收到多数派投票 ────► [Leader]
        ▲                                            │
        │  发现更高任期      发现更高任期               │ 心跳(空AppendEntries)持续发给所有人
        └────── [Candidate] ◄────────────────────────┘
                  选举超时后 term+1，投自己，
                  并行拉票(RequestVote)
```

- **Follower**：被动接受一切；只响应来自 leader/candidate 的请求。
- **Candidate**：选举超时后自增任期 term，投自己一票，向所有节点拉票；收到多数派票 → Leader；收到别人更高任期的消息 → 退回 Follower。
- **Leader**：处理所有客户端请求，通过心跳维持权威。

**任期（term）**是 Raft 的逻辑时钟（02 章逻辑时钟的应用）：每次选举 term+1。任期是全局的"第几届政府"编号，**任何消息都携带任期，接收方发现消息任期比自己小就直接拒绝**——这一条规则挡住了绝大多数脑裂场景（旧 leader 复活后它还带着旧 term，所有新请求都会被拒）。网络分区中被隔离的旧 leader 心跳发不进多数派，新区很快选出新 leader（term 更高），旧 leader 即使继续给 follower 发心跳也会因任期过小被拒。

Raft 的核心不变量（可以背下来作为检验理解的标尺）：

1. **选举安全**：一个任期内每个节点至多投一票（先到先得 + 日志检查），多数派 + 一票限制 ⇒ 每任期至多一个 leader。
2. **日志匹配**：若两日志同 index 处 term 相同，则该位置及之前的所有条目全部相同。
3. **Leader 完备性**：term T 当选的 leader 拥有此前所有已提交的日志（靠选举限制保证）。

## 4. 领导者选举：规则 + Python 模拟

选举规则清单（Raft 论文 Figure 2 的 RequestVote 部分）：

1. Follower 在 `election_timeout`（**随机化**在 150~300ms，论文建议值）内没听到 leader 心跳 → 自增 term、转 Candidate、投自己。
2. Candidate 并行向所有节点发 RequestVote(term, candidateId, lastLogIndex, lastLogTerm)。
3. 投票人决策（**选举限制**，防"日志不全者当选"）：
   - 我的 term 更大 → 拒绝（你过时了）；
   - 本任期我已投过别人 → 拒绝（一票限制）；
   - 候选人日志**不比我的新** → 拒绝（比较规则：先比最后一条的 term，再比 index）；
   - 否则投给候选人。
4. Candidate 收到多数派票 → 成为 Leader，立即广播心跳宣示权威。

**为什么必须随机化超时**：如果所有节点超时相同，同时超时、同时拉票、互相投自己 → 活锁（每个任期都选不出）。随机错开让"先醒的那个"大概率在别人超时前拉到多数派。**经验做法**：随机化是分布式避免同步活锁的通用手段，凡"超时触发竞争"的场景都该随机抖动（第 10 章重试抖动同源）。

**为什么必须多数派**：两个 leader（不同分区各一个）必然各持有少于半数的票，而一票限制保证同一任期内票不重复 → 两个多数派集合必相交 → 交点不可能投两票 → 矛盾。多数派相交同时保证了"新 leader 的票箱里有旧 leader 提交过的所有日志"（Leader 完备性），这是数据不丢的根。

```python
# raft_election.py —— 3 节点 Raft 选举模拟：随机超时、一票限制、日志新旧检查、脑裂避免
import asyncio, random

class RaftNode:
    def __init__(self, name, peers, bus, election_range=(0.15, 0.30)):
        self.name, self.peers, self.bus = name, peers, bus   # bus: 共享消息总线(模拟网络)
        self.role = "follower"
        self.term = 0
        self.voted_for = None          # 本任期投给谁（一票限制）
        self.log = []                  # [(term, cmd)] —— 仅用于"日志新旧"比较
        self.election_range = election_range
        self.election_deadline = 0.0
        self._task = asyncio.create_task(self._loop())

    def log_is_newer_than(self, last_idx, last_term):
        """候选日志是否不比我的旧：先比最后条目的term，再比index"""
        my_last_term = self.log[-1][0] if self.log else 0
        if last_term != my_last_term:
            return last_term > my_last_term
        return last_idx >= len(self.log)

    async def _loop(self):
        while True:
            await asyncio.sleep(0.01)
            now = asyncio.get_event_loop().time()
            if self.role != "leader" and now >= self.election_deadline:
                await self.start_election()

    async def start_election(self):
        self.term += 1
        self.role, self.voted_for = "candidate", self.name
        self.election_deadline = asyncio.get_event_loop().time() + random.uniform(*self.election_range)
        votes = {self.name}
        last_term = self.log[-1][0] if self.log else 0
        print(f"  {self.name}: 超时 → 竞选 term={self.term}")
        for p in self.peers:
            await self.bus.send(p.name, ("req_vote", self.term, len(self.log), last_term, self.name))

    async def on_request_vote(self, term, last_idx, last_term, cand):
        if term > self.term:                       # 看到更大任期：更新并退回 follower
            self.term, self.role = term, "follower"
            self.voted_for = None
        if term == self.term and self.voted_for is None and self.log_is_newer_than(last_idx, last_term):
            self.voted_for = cand                  # 一票限制 + 日志检查，先到先得
            await self.bus.send(cand, ("vote", self.term, self.name))
        elif term < self.term:
            await self.bus.send(cand, ("reject", self.term, self.name))  # 旧任期拉票：拒绝

    async def on_vote(self, term, voter):
        if self.role != "candidate" or term != self.term:
            return
        votes = getattr(self, "_votes", {self.name}); votes.add(voter); self._votes = votes
        if len(votes) > (len(self.peers) + 1) // 2:   # 多数派：n=3 → 2票
            self.role = "leader"
            print(f"  {self.name}: 当选 term={self.term} 的 leader（票:{sorted(votes)}）")
            for p in self.peers:                     # 立即广播心跳宣示权威
                await self.bus.send(p.name, ("heartbeat", self.term, self.name))

    async def on_heartbeat(self, term, leader):
        if term >= self.term:                        # 认可权威：更新任期、重置选举计时
            self.term, self.role = max(term, self.term), "follower"
            self.election_deadline = asyncio.get_event_loop().time() + random.uniform(*self.election_range)

class Bus:
    """极简消息总线：按节点名投递，模拟网络（丢消息可扩展）"""
    def __init__(self, nodes): self.nodes = nodes
    async def send(self, to, msg):
        node = self.nodes[to]
        kind = msg[0]
        await node.on_request_vote(*msg[1:]) if kind == "req_vote" else (
            node.on_vote(*msg[1:]) if kind == "vote" else (
            node.on_heartbeat(*msg[1:]) if kind == "heartbeat" else None))

async def main():
    names = ["A", "B", "C"]
    nodes = {n: RaftNode(n, [], None) for n in names}
    for n in nodes.values(): n.peers = [x for x in nodes.values() if x is not n]
    for n in nodes.values(): n.bus = Bus(nodes)   # 补上总线
    await asyncio.sleep(1.0)
    leader = [n for n in nodes.values() if n.role == "leader"]
    print("1秒后 leader:", [(n.name, n.term) for n in leader])

asyncio.run(main())
```

多跑几次可以观察到：几乎每次都在一个任期选出 leader；偶尔出现"瓜分选票"（两个 candidate 各得 1 票）但随机超时让下一轮几乎必然选出——这就是随机化对 FLP 活锁的工程破解。

## 5. 日志复制与提交规则

Leader 收到客户端命令后的流程：

1. 追加到本地日志（未提交状态）。
2. 并行发 AppendEntries 给所有 follower（携带前一条日志的 index/term 做一致性检查）。
3. follower 一致性检查通过 → 追加本地 → 回 ack；检查失败 → leader 递减 nextIndex 重试（日志回溯对齐，修复 follower 缺日志/冲突日志）。
4. **多数派 ack → leader 提交**（应用到状态机）→ 后续心跳携带 commitIndex → follower 也提交。
5. 回复客户端成功。

**为什么必须多数派才能提交**（本节核心论证，面试高频）：

- 若允许少数派提交：分区期间，少数派侧 leader 把命令"提交"了（只有 1/3 节点有）；多数派侧同时选出新 leader 也提交了不同命令 → 同一 index 出现两个不同命令 → 日志匹配性质被破坏 → 状态机分叉，永不可修复。多数派 + 单调 term 保证：**任何已提交的条目必然存在于所有未来 leader 的日志里**（提交过半 ⇒ 任何未来多数派当选票箱与之相交 ⇒ 交点节点持有该条目 ⇒ 选举限制使日志更旧者无法当选）。
- 推论（Raft 论文第 5.4.2 节的提交规则）：**leader 不直接提交前任任期的条目**——只通过提交"自己任期内的新条目"间接把旧条目带过去（因为旧条目即使过半复制，也无法证明"没有更新的 leader 会覆盖它"，新条目的提交才隐含了这一点）。这是 Raft 与直觉差异最大的一条规则。

故障场景走查（思考题格式，先自己推再往下看）：

```
场景：3 节点 [A(旧leader, term=1), B, C]，A 与 B/C 分区
1. A 还活着，继续当"它以为的 leader"，但它发不出多数派
2. B/C 在随机超时后选出新 leader（term=2，比如 B）
3. 客户端连 A 写入 x=1 → A 记日志但永远凑不齐多数派 ack → 无法提交 → 不回成功
4. 客户端连 B 写入 y=2 → 多数派(B,C) → 提交成功
5. 分区恢复：A 发心跳(term=1) → B/C 发现 term 更小 → 拒绝并回 term=2
6. A 退回 follower，回滚未提交的 x=1，从 B 同步 y=2
结论：x=1 从未"成功"过（未提交），系统仍然一致；A 的幽灵权威被任期挡住
```

```python
# raft_append.py —— 日志复制与提交规则的最小模拟（单 leader 正常路径 + 未过半不提交）
import asyncio, random

class AppendNode:
    def __init__(self, name):
        self.name, self.log, self.commit = name, [], -1   # log: [(term,cmd)], commit: 已提交index

    def majority_of(self, total): return total // 2 + 1

async def replicate_command(leader: AppendNode, followers: list, cmd, term):
    """leader 追加 → 并行复制 → 多数派才提交（Raft 提交规则）"""
    leader.log.append((term, cmd))
    my_idx = len(leader.log) - 1

    async def send_to(f):
        if f.log and my_idx > 0 and len(f.log) <= my_idx:
            # 一致性检查：前一条不匹配则拒绝（真实 Raft 会回溯对齐，这里简化为拒绝）
            return False
        f.log = f.log[:my_idx] + [leader.log[my_idx]]      # 截断冲突 + 追加
        return True

    acks = sum(await asyncio.gather(*[send_to(f) for f in followers])) + 1
    if acks >= leader.majority_of(len(followers) + 1):
        leader.commit = my_idx                              # 多数派：提交
        for f in followers:
            f.commit = min(f.commit, my_idx) if len(f.log) > my_idx else f.commit
        return f"已提交 index={my_idx}（{acks} acks）"
    return f"未提交 index={my_idx}（仅 {acks} acks < 多数派）——客户端不得视为成功"

async def main():
    a, b, c = AppendNode("A"), AppendNode("B"), AppendNode("C")
    print("正常:", await replicate_command(a, [b, c], "set x=1", term=1))
    print("A/B 日志:", a.log, b.log, "commit:", a.commit, b.commit)
    # 分区模拟：C 掉线，2/3 仍多数 → 提交成功；再让 B 也掉线 → 1/3 不提交
    c.log, c.commit = [], -1
    print("C 掉线后写:", await replicate_command(a, [b], "set y=2", term=1))
    b.log, b.commit = [], -1
    print("B也掉线后写:", await replicate_command(a, [], "set z=3", term=1))
    print("A 的 commit:", a.commit, "← 未过半的 z=3 不会被应用")

asyncio.run(main())
```

## 6. 安全性论证与成员变更

Raft 正确性由五条性质支撑（论文第 5 节），前两条靠选举，后两条靠复制规则：

1. **选举安全**（一票限制 + 多数派）。
2. **Leader 只追加（Leader Append-Only）**：leader 不删不改自己日志，只追加。
3. **日志匹配**（AppendEntries 的一致性检查 + 回溯对齐）。
4. **Leader 完备性**（选举限制：日志旧者选不上）。
5. **状态机安全**：同 index 同命令应用到所有状态机（4 + 提交规则的推论）。

**成员变更**（集群增删节点）概念：朴素做法"全体同时换配置"有致命缺陷——旧配置 3 节点（AB C 多数派 A,B）与新配置 4 节点（A B C D 多数派 A,B,C... 实际 C,D）可能选出两个不相交多数派的 leader。Raft 论文用**联合共识（joint consensus）**两阶段切换；工程上更常用**单节点变更（single-server change）**（etcd/raft 实现采用，见 *Consensus: Bridging Theory and Practice* 第 4 章）：一次只加/删一个节点，数学上保证新旧配置多数派必相交（3→4：任何旧多数派 {2} 与新多数派 {3} 在 4 节点里必交）。**经验做法**：生产变更一律单节点逐步做，不要一次换一批。

**日志压缩/快照**：日志无限增长会撑爆磁盘并拖慢重启恢复；做法是状态机定期打快照，快照之前的日志整段丢弃， follower 落后太多时 leader 直接发快照（InstallSnapshot RPC）而不是逐条补日志。

## 常见误区

1. **"Raft 是强一致的，所以读写都走 leader 就万事大吉"**：leader 可能已换人（客户端缓存了旧 leader 地址）；读到已复制未提交的数据也可能发生。要线性一致读需要额外动作：ReadIndex 或 Lease Read（etcd 的实现，阅读 etcd 文档《Raft 协议》章节；Lease Read 依赖时钟假设，ReadIndex 不依赖）。直接读 leader 本地状态机是常见的一致性漏洞。
2. **"选举超时随机区间越大越好"**：区间大了故障检测慢（可用性下降），小了容易连续瓜分选票。经验做法：取心跳间隔的 5~10 倍，并保证"最快者完成拉票"的概率压倒性（Raft 论文给出数学分析，结论 150~300ms 对 1ms 心跳够用）。
3. **"分区时少数派也会响应写请求，所以有脑裂"**：少数派 leader 写不进去（凑不齐多数派 ack），客户端得不到成功确认；它唯一能"破坏"的是承诺语义——只要存储端只认"已提交"状态，分区自愈后它会回滚。真正的脑裂防护=任期/纪元 + 多数派，二者缺一不可。
4. **"收到多数派 ack 就可以回复客户端了，无论是谁的 ack"**：必须是**当前任期**的多数派。旧任期里复制出去的多数派 ack 不能直接算提交（第 5 节提交规则）。这条错了会引入已提交条目被覆盖的窗口。
5. **"日志匹配性质天然成立，不用检查"**：它靠 AppendEntries 的一致性检查（带前一条 index/term）+ 失败时回溯来**维护**；跳过检查，follower 上的冲突旧日志（前任 leader 留下的未提交条目）不会被截断，分叉就发生了。

## 本篇小结

- 共识 = 状态机复制的排序问题；FLP 证明纯异步下不可能，工程解=超时+随机化。
- Paxos 正确但工程空洞；Raft 用问题分解 + 强 leader 换可理解性（ATC 2014 论文标题即立场）。
- 任期是逻辑时钟；"消息任期小即拒绝"一条规则挡掉绝大多数脑裂；随机选举超时破解选票瓜分活锁。
- 选举限制（日志新旧检查）+ 一票限制 + 多数派 ⇒ Leader 完备性 ⇒ 已提交条目永不被覆盖。
- 提交必须当前任期多数派；leader 不直接提交旧任期条目；日志匹配靠一致性检查维护。
- 成员变更用单节点变更（etcd 实践）；日志用快照压缩——这两个工程点 13 章实战会用。

## 练习

1. 跑通 raft_election.py：把随机超时区间改成固定值（比如全部 0.2s），统计 100 次实验中出现"任期递增多次才选出 leader"（活锁迹象）的次数；再恢复随机化对比——用数据说明随机化的必要性。
2. 在 raft_election.py 里给节点加日志：给 A 两条 term=1 的日志、B/C 空。模拟"日志最旧的 C 先超时拉票"，验证 on_request_vote 的日志检查会拒绝 C；然后让 B（日志与 A 同新）当选。
3. 用 raft_append.py 构造"前任任期条目"场景：term=1 时 A 复制 x=1 给 B（C 掉线，未过半不提交）；C 回来后 B 当选 term=2，写下"为什么 B 必须先提交一条 term=2 的命令才能安全提交 x=1"。
4. 把 raft_election.py 与 raft_append.py 合并：当选 leader 后持续接收 3 条客户端命令并复制提交，打印三个节点最终的 log 与 commit，验证状态机一致。
5. 阅读题：读 Raft 论文第 5.4 节（安全性）与 etcd 文档的学习页（raft.sourceforge.io 若失效则读 etcd raft README），用自己的话写出"Lease Read 与 ReadIndex 的区别"及各自依赖的假设（提示：时钟假设、quorum 读确认）。
