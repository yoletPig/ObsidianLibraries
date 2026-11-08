# 08 - Redis与缓存

## 本篇目标

- 掌握五种数据结构的真实用法（每个结构配一个场景），不再把 Redis 当"大号字典"。
- 理解 zset 为什么用跳表：与 B+树/平衡树的对比，跳表实现要点。
- 讲清持久化（RDB vs AOF）、过期与淘汰策略（8 种）的权衡。
- 把缓存三问（穿透/击穿/雪崩）逐个讲透：布隆过滤器手写、singleflight 思想、错峰 TTL；缓存一致性与分布式锁的完整方案与争议。

## 1. 安装与五种数据结构的真实用法

```bash
sudo apt install -y redis-server && redis-cli ping    # PONG 即就绪
pip install redis                                     # Python 客户端 redis-py
```

五个结构，五个场景（命令行 + Python 对照，key 设计约定 `业务:实体:id`，如 `sess:u42`、`px:AAPL:20250601`）：

```
┌────────────┬────────────────────────────┬──────────────────────────────┐
│ 结构       │ 场景                        │ 关键命令                     │
├────────────┼────────────────────────────┼──────────────────────────────┤
│ String     │ 缓存对象/计数器/分布式锁     │ SET/GET/INCR/SETNX/SETEX     │
│ List       │ 时间线/任务队列(轻量)        │ LPUSH/RPOP/LRANGE/BRPOP      │
│ Hash       │ 对象字段级存取(部分更新)      │ HSET/HGET/HGETALL/HINCRBY    │
│ Set        │ 标签/去重/共同关注(交并差)    │ SADD/SISMEMBER/SINTER        │
│ ZSet       │ 排行榜/延时队列/滑动窗口限流  │ ZADD/ZINCRBY/ZRANGE/ZREMRANG │
└────────────┴────────────────────────────┴──────────────────────────────┘
```

```python
import redis
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

r.setex("px:AAPL:close", 60, "190.2")            # String: 60s TTL 的行情缓存
r.incr("qps:api")                                 # String 计数器: 原子自增
r.hset("user:42", mapping={"name": "yyl", "score": 88})   # Hash: 字段级更新
r.sadd("agent:skills:42", "sql", "python")        # Set: 技能标签
r.zincrby("rank:score", 10, "user:42")            # ZSet: 加 10 分
r.zrevrange("rank:score", 0, 9, withscores=True)  # ZSet: TOP10 排行榜
r.zadd("delay:jobs", {"job:1001": 1730000000})    # ZSet 延时队列: score=执行时间戳
# 取到期任务: zrangebyscore("delay:jobs", 0, now, start=0, num=5)
```

选型口径（经验做法）：**对象整体缓存用 String（序列化后一次读写）；字段频繁单独更新用 Hash**（HSET 不用重写整个对象）；需要排序/范围 → ZSet；需要去重/集合运算 → Set。还有一个隐藏结构要认识：**Stream**（XADD/XREADGROUP，带消费组的日志型结构，分布式系统 09 章 MQ 主题的 Redis 版），轻量事件流可用，重 MQ 还是上专业队列。

## 2. zset 与跳表：为什么不用 B 树

zset 底层是**跳表（Skip List）+ 哈希表**：哈希表负责 member→score 的 O(1) 点查；跳表负责按 score 有序，支撑范围查询（ZRANGEBYSCORE）与排名。跳表 = 在有序链表上建多层"快速车道"（Redis 源码 t_zset.c，实现见 Pugh 1990 *Skip Lists: A Probabilistic Alternative to Balanced Trees*）：

```
L3:  head ─────────────────────────────── 50 ──────────────→ nil
L2:  head ──────── 12 ─────────────────── 50 ────── 72 ───→ nil
L1:  head ── 5 ── 12 ── 20 ── 30 ──────── 50 ── 61 ─ 72 ──→ nil
L0:  head ── 5 ── 12 ── 20 ── 30 ── 41 ── 50 ── 61 ─ 72 ──→ nil

查找 41: L3 走到 50 太大退回 L2 → 12 → 50 太大退回 L1 → 30 → 41 ✓ (从 11 次比较降到 4 次)
每个节点晋升到上一层的概率 p=1/4(Redis 参数), 期望层数 O(log n), 查找/插入/删除 O(log n)
```

**为什么 Redis 不用 B+树/红黑树**（作者 antirez 在博客 *Random notes on improving Redis* 中给过答案，要点如实转述）：①实现简单（红黑树的旋转/删除修复代码量数倍于跳表，B+树还要管理页与分裂）；②范围查询同样顺底链表走，不输 B+树；③**并发友好性**——插入只需改相邻指针，无全局再平衡；④调试与边界行为直观。跳表与平衡树同为 O(log n)，工程差异在"常数、实现复杂度与并发形态"——这和 06 章"B+树对齐磁盘页"是同一个道理：**数据结构的胜负手在它服务的介质与访问模式**。内存数据库没有"页"的概念，B+树最大的卖点（页级 IO 最少化）就不成立了。

实现要点（若手写，10 章的 memtable 就是简化跳表的好场景）：插入时随机层数（抛硬币 p=1/4，封顶 32）；删除需同层摘链；span 字段可 O(log n) 查排名（ZRANK 的实现）。

## 3. 持久化：RDB vs AOF

Redis 数据在内存，持久化决定"重启丢多少"：

| 维度 | RDB（快照） | AOF（追加日志） |
| --- | --- | --- |
| 原理 | fork 子进程，把某一时刻全量数据写进 `.rdb` | 把每条写命令追加进 `.aof`（先写 OS 缓冲） |
| 落盘时机 | save 配置点（如 900s 内 1 变）/手动 SAVE | appendfsync: always / everysec(默认) / no |
| 丢失窗口 | 上次快照以来的全部（分钟级） | everysec 最多 1s；always 接近 0 但吞吐暴跌 |
| 文件/恢复 | 小、恢复快（直接加载） | 大、恢复要重放命令（有 AOF 重写压缩） |
| 性能影响 | fork 瞬间卡顿（写时复制，内存翻倍风险） | 每秒一次 fsync，稳态代价低 |

```bash
redis-cli config get save appendfsync     # 查看当前配置
redis-cli set k v && redis-cli bgsave     # 手动触发快照
# 经验做法: 生产双开 — RDB 做冷备与快速重启, AOF(everysec) 做数据安全底线
```

对照 06 章：RDB 是"checkpoint"，AOF 是"WAL"——**同一组概念在内存数据库里换了名字**。fsync 代价与 group commit 的物理根源见分布式系统 07 章。

## 4. 过期策略与淘汰策略

**过期（对单个 key，设置了 TTL 的）**：Redis 的组合拳——**惰性删除**（访问时发现过期才删，省 CPU 但留死键）+ **定期抽样删除**（每秒抽样若干带 TTL 的 key，过期比例高就继续抽）。于是死键可能占内存，需要：

**淘汰（内存满了，maxmemory 触发时）8 种策略**：

```
noeviction      默认: 不淘汰, 写命令直接报错 OOM (重要场景选它, 逼你做容量)
allkeys-lru     全部 key 里淘汰最久未使用        ← 纯缓存场景默认选它
volatile-lru    只在设了 TTL 的 key 里 LRU       ← 缓存+持久 key 混居时
allkeys-lfu / volatile-lfu     最不经常使用(4.0+, 适合热点更尖锐的场景)
allkeys-random / volatile-random  随机淘汰(几乎不用)
volatile-ttl    优先淘汰剩余 TTL 最短的
```

注意两个坑（经验做法）：①LRU/LFU 都是**近似实现**（抽样 N 个 key 淘汰其中最该死的，`maxmemory-samples` 可调），不是全局精确；②"设了 TTL 但永不访问"的 key 在 allkeys-lru 下反而比冷数据活得久——TTL 管生死、LRU 管冷热，是两套正交机制。

## 5. 缓存三问：穿透 / 击穿 / 雪崩

缓存层挡在应用与数据库之间，三种典型崩塌路径（配合第 4 节的策略记忆）：

**① 穿透（查不存在的数据，缓存永远 miss，请求全打到库）**——恶意扫描随机 id 时必现。解法双管：

- 缓存空值：miss 的 key 也缓存 `""` + 短 TTL（如 60s），把"查无此数据"也变成可命中。
- **布隆过滤器**：在缓存前加一道"一定不存在就别往下走"的闸门。原理与手写简化版：

```python
# bloom.py —— 简化布隆过滤器: 多哈希置位, "可能存在"误报, "不存在"必准
import hashlib, math

class Bloom:
    def __init__(self, n, fp=0.01):
        m = math.ceil(-n * math.log(fp) / (math.log(2) ** 2))   # 最优位数组长度
        k = max(1, int(0.693 * m / n))                           # 最优哈希个数
        self.m, self.k, self.bits = m, k, bytearray((m + 7) // 8)
    def _idx(self, key, i):
        h = hashlib.sha256(f"{key}-{i}".encode()).digest()       # 双哈希模拟 k 个哈希
        return int.from_bytes(h[:8], "big") % self.m
    def add(self, key):
        for i in range(self.k):
            p = self._idx(key, i); self.bits[p >> 3] |= 1 << (p & 7)
    def might_contain(self, key) -> bool:
        return all(self.bits[p >> 3] & (1 << (p & 7)) for p in
                   [self._idx(key, i) for i in range(self.k)])

b = Bloom(100_000)
for uid in valid_users: b.add(uid)
if not b.might_contain(user_input): return 404        # 一定不存在 → 直接拦截
# 误报率 fp: 说"存在"但可能其实不存在(去查库确认); 说"不存在"则 100% 不存在 → 挡住穿透
```

布隆的代价：**不能删除**（位共享），计数布隆/布谷鸟过滤器是加强版；06 章提过 LSM 也用它挡无谓的磁盘查找——同一个结构两个舞台。

**② 击穿（某个热点 key 过期瞬间，海量并发同时回源重建）**——秒杀页缓存到期的 1 秒。解法：**单飞（singleflight）思想**——同一 key 的并发回源合并成一次：

```python
# singleflight 思想版(跨进程要上分布式锁, 见第7节)
import threading
_lock, _inflight = threading.Lock(), {}

def get_with_singleflight(key, loader, ttl=60):
    v = r.get(key)
    if v is not None: return v
    with _lock:                                        # 进程内: 谁先到谁回源
        if key not in _inflight: _inflight[key] = loader()
        v = _inflight.pop(key)
    r.setex(key, ttl, v)
    return v
```

跨进程版本：回源前 `SET lock:key <uuid> NX EX 10` 抢锁，抢到的查库回填，没抢到的轮询缓存（或短 sleep 重试）——**一个回源、众人等待**。互斥重建 + 逻辑过期（key 永不过期、value 里带过期时间由单线程异步刷新）是热点更极致的方案。

**③ 雪崩（大量 key 同一时刻过期 / Redis 整体宕机，流量海啸砸向数据库）**。解法（经验做法）：**错峰 TTL**——基础 TTL 加随机抖动 `setex(key, ttl + random.randint(0, 300), v)`，把"同刻过期"摊平；**多级缓存**（本地 LRU + Redis，击穿与雪崩都被第一级垫了一层）；**熔断降级**（库端限流/返回兜底数据，分布式系统 10 章的熔断器在此就位）；Redis 高可用（哨兵/集群，分布式系统 12 章）保整体不死。

## 6. 缓存一致性：先更新库，再删缓存

缓存与数据库是两个系统，**没有免费的强一致**，工程要的是"最终一致 + 窗口尽量小"。主流方案的推演：

```
先删缓存, 再更新库(坏): 删完还没更新库, 并发读把旧值又灌回缓存 → 旧值长期驻留
先更新库, 再删缓存(推荐): 读写并发的最坏交错 = 读到旧值的短窗口
   T1 读: miss → 读库得 v1 → (T2 更新库为 v2 并删缓存) → T1 把 v1 写入缓存
   → 缓存里是旧值 v1 直到 TTL。概率低(需要读慢于写), 且 TTL 兜底。
先更新库, 再更新缓存(不推荐): 并发双写乱序可让缓存停在旧值且无 TTL 之外的自愈
```

为什么是"删"不是"更新"（经验做法）：删除把一致性收敛责任推给下一次读（懒计算），避免"缓存写回的值与下次库值计算口径不一致"这一整类问题；写多读少的数据甚至**不该缓存**（删了也没人来读，白忙）。

**延迟双删**：`删缓存 → 更新库 → sleep(主从延迟估计值) → 再删一次`——第二次删除清掉"在主从延迟窗口里读到的旧值"。它是治标手段：以写路径加一次延迟为代价，缩小读旧值窗口（经验做法：读多写少 + 有从库时用；删除失败要重试队列/订阅 binlog 兜底，如 Canal 消费变更后删缓存——把一致性收敛做成后台任务而非在线路径）。

## 7. 分布式锁：SETNX + 过期 + 看门狗

多个进程互斥（击穿回源、定时任务防重），需要跨进程锁。Redis 方案的演进（每一步都是被事故逼出来的）：

```python
# 1) 基础版: SET NX = 不存在才设置(原子), EX 过期防死锁
ok = r.set("lock:rebuild", token, nx=True, ex=30)
# 2) 释放必须"检查是自己的锁再删", 两步不原子 → 用 Lua 脚本原子化
unlock_lua = """
if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1])
else return 0 end"""
r.eval(unlock_lua, 1, "lock:rebuild", token)   # token=uuid, 防止误删别人的锁
# 3) 业务比 TTL 长怎么办: 看门狗——持锁线程后台定时(如每 10s)续期; 进程崩了续期停止, 锁自然过期
#    (Redission 的 watchdog 即此思想; 手写: 起线程循环 eval 同款 Lua 把 EX 往后推)
```

三件套缺一不可：**NX**（互斥）、**EX**（持有者崩溃后锁能自动释放）、**token+Lua**（只能释放自己的锁）。再看两个进阶问题：

- **锁过期但业务还没跑完**：看门狗续期是主解；兜底是业务幂等（锁失效后另一进程进入，重复执行不能造成数据损坏——分布式系统 09 章的幂等键）。
- **主从切换丢锁（Redlock 争议）**：单实例 Redis 宕机 → 从库升主 → 锁丢失，两个客户端同时持锁。**Redlock**（向 N 个独立实例同时加锁、多数派成功才算持有）试图解决此问题；Kleppmann 与 antirez 有过公开论战（Kleppmann 博客 *How to do distributed locking*，2016）：Redlock 依赖时钟假设、且没法提供类似 fencing token 的安全性。**如实呈现的结论（经验做法）**：Redis 锁是"效率锁"（防重复干活，偶尔双执行无伤大雅）——够用；若是"正确性锁"（双执行会造成资损），用支持 fencing 的共识系统（etcd/ZooKeeper，分布式系统 11 章）或把正确性做进业务幂等，不指望锁本身。

## 常见误区

1. **"Redis 就是缓存，数据丢了无所谓"**：单实例 Redis 还承担着计数器、排行榜、分布式锁这些"有状态"职责。持久化策略（3 节）与淘汰策略（4 节 noeviction）要按角色区分：纯缓存 allkeys-lru+可丢，状态承载 noeviction+AOF。
2. **"KEYS * 看看 key"**：O(N) 阻塞整个单线程，百万 key 的实例直接卡死。生产用 `SCAN`（游标渐进遍历，不阻塞）——同理警惕 O(N) 的大集合命令（HGETALL 大 hash、ZRANGE 全量），先 `TYPE/OBJECT ENCODING` 看体量。
3. **"加锁就安全"**：锁只解决互斥，不解决持有者死亡（EX 补）、误删（token+Lua 补）、主从切换（fencing/幂等补）。每加一把分布式锁都要答完这四问。
4. **"缓存与数据库可以强一致"**：双写永远有窗口（6 节时序推演）。设计目标收敛为"最终一致 + 窗口最小化 + TTL 兜底"，把精力花在收敛机制（binlog 订阅/重试队列）而不是幻想。
5. **"热点 key 过期才击穿，平时没事"**：不设 TTL 的热点 key 在淘汰（allkeys-lru）下同样会"意外消失"。关键路径的缓存要么 TTL+单飞重建，要么逻辑过期+异步刷新，任其自生自灭就是事故预约。

## 本篇小结

- 五结构五场景：String 缓存/计数/锁、List 轻队列、Hash 字段级对象、Set 去重交并、ZSet 排行/延时/限流；key 命名 `业务:实体:id`。
- zset=跳表+哈希表：跳表在内存介质上以实现简单+并发友好胜过 B+树——数据结构选择跟随介质与访问模式（与 06 章 B+树结论互为镜像）。
- 持久化：RDB=checkpoint（快，丢分钟级），AOF=WAL（everysec 丢秒级）；生产双开。
- 过期是惰性+抽样，淘汰 8 种记三个：noeviction（状态）、allkeys-lru（纯缓存）、volatile-ttl（混合）。
- 缓存三问全解：穿透→空值+布隆（"不存在必准"）；击穿→单飞/互斥重建；雪崩→错峰 TTL+多级+熔断。
- 一致性：先更新库再删缓存；延迟双删缩小窗口；binlog 订阅兜底。分布式锁：NX+EX+token+Lua+看门狗；Redlock 是效率锁方案，正确性靠共识系统或业务幂等。

## 练习

1. 用 redis-py 实现滑动窗口限流：ZSET 结构（score=时间戳），`ZREMRANGEBYSCORE` 清窗口外、`ZCARD` 计数判限——写出完整函数并压测验证。
2. 把 5 节 Bloom 的 n=10万/fp=1% 代入公式，算出 m 与 k；故意用 1000 个不存在的 key 测误报率，与理论值对比；再测"add 后误报率随元素数增长"的曲线。
3. 击穿实验：一个 value 较大的 key（模拟慢回源 2s），50 并发在过期瞬间同时请求——对比"裸 get+set"与"singleflight+分布式锁"两版打到的回源次数与总耗时。
4. 复现 6 节"先删缓存再更新库"的旧值回灌：两个线程按文中时序交错（用事件控制步进），打印缓存最终值；再跑"先更新库再删缓存"版本对比窗口概率。
5. 看门狗手写：给 7 节锁加一个续期线程（每 TTL/3 续一次），模拟业务 60s（TTL 30s）运行全程不断锁；kill -9 持锁进程验证锁在 TTL 内自动释放——把"三件套+看门狗"的每个机制用自己的实验钉死。
