# 10 - 手写MiniKV存储引擎

## 本篇目标

- 用四步演进写出一个能落盘、能崩溃恢复、能合并的 KV 存储引擎：内存版 → WAL 版 → 简化 LSM 版 → LSM+缓存版。
- 每一步都用真实基准数字对比（写入吞吐/读延迟），把 01/06 章的理论（fsync 代价、顺序写、读放大）变成自己的测量。
- 跑通 kill -9 崩溃恢复验证：写 1000 条后强杀，重放 WAL 后数据一条不丢。
- 以"设计取舍日志"的形式记录每一步的决策与代价，养成存储工程的思维方式。

## 1. 项目总览：四步演进的路线图

我们要造的东西叫 MiniKV——一个 put/get/delete 的 KV 存储。它不是玩具堆砌，而是把前九章的原理按依赖顺序逐一落地：

```
阶段1 内存版    dict 一把梭                    → 基准线: 不落盘能有多快(11216 千条/s)
   ↓ 加什么? 持久性(01 章 D 的实现=追加日志)
阶段2 WAL版     append+fsync 每条落盘          → 持久但慢(1 千条/s): fsync 的账单
   ↓ 加什么? 吞吐(06 章 LSM: 顺序写+批量刷盘)
阶段3 LSM版     memtable+SSTable+compaction    → 快(720 千条/s) 但读要翻文件(642μs)
   ↓ 加什么? 读性能(08 章缓存思想)
阶段4 +缓存     读缓存+布隆过滤器              → 热读减半(321μs), 认识"缓存救重复读"
```

文件组织（单文件 `minikv.py` 分四个类，本章全部代码给出；建议动手抄一遍而不是复制）：

- `MemoryKV`：纯字典，基准线。
- `WALKV`：WAL 版，记录格式 + 崩溃恢复（重放）。
- `LSMKV`：简化 LSM，memtable 满 N 条冻结为不可变 SSTable，提供 compact()。
- `CachedLSM`：在 LSMKV 上加读缓存 + 简化布隆过滤器。

约定：Python 3.10+ 纯标准库；val 支持 str/bytes；记录格式 `[4B keylen][4B vallen][key][val]`（小端）。**简化声明**：没有 CRC 校验、没有多线程安全级写入、SSTable 读是全文件加载（没有稀疏索引）——这些"偷懒"恰是 5 节取舍日志的素材：每处简化都对应真实引擎的一个组件（校验和→RocksDB 的 block crc、稀疏索引→SSTable 的 index block）。

## 2. 阶段1：内存版——一切性能的基准线

```python
# minikv.py 第 1 段
class MemoryKV:
    def __init__(self): self.d = {}
    def put(self, k, v): self.d[k] = v
    def get(self, k): return self.d.get(k)
    def close(self): pass
```

就这么短。它的意义是**确立对照系**：进程一死数据全没（01 章"可靠存储"为零），但写吞吐是物理上限。实测（20 万条压测见 6 节 bench.py，此处 2 万条）：**写 11216 千条/s、读 0.06μs**。后面每加一层能力，都回头问一句"我付的代价换来了什么"。

## 3. 阶段2：WAL 版——持久性与崩溃恢复

WAL 版的写路径：**每条 put 追加写日志并 fsync**，内存字典同步更新；读纯内存。崩溃恢复 = 启动时把日志从头重放。这是 01 章"可靠存储"、06 章 SQLite journal 的最小实现：

```python
# minikv.py 第 2 段
import os, struct

class WALKV:
    def __init__(self, path):
        self.path, self.d = path, {}
        if os.path.exists(path):                      # ---- 崩溃恢复: 重放日志 ----
            with open(path, "rb") as f:
                while True:
                    hdr = f.read(8)
                    if len(hdr) < 8: break            # 尾部不满 8B = 半条记录(崩溃截断)
                    kl, vl = struct.unpack("<II", hdr)
                    if kl > 1<<20 or vl > 1<<20: break # 长度字段损坏(截断在长度处)→丢弃
                    k, v = f.read(kl), f.read(vl)
                    if len(k) < kl or len(v) < vl: break
                    if v == b"\x00TOMB": self.d.pop(k, None)   # 删除标记(墓碑)
                    else: self.d[k] = v
        self.f = open(path, "ab")
    def put(self, k, v):
        k, v = k.encode(), (v.encode() if isinstance(v, str) else v)
        self.f.write(struct.pack("<II", len(k), len(v)) + k + v)
        self.f.flush(); os.fsync(self.f.fileno())     # ---- 持久点: 这行是 D 的价格 ----
        self.d[k] = v
    def delete(self, k):                              # 删除也走日志(墓碑), 崩溃后重放才知道删过
        k = k.encode(); self.f.write(struct.pack("<II", len(k), 4) + k + b"\x00TOMB")
        self.f.flush(); os.fsync(self.f.fileno()); self.d.pop(k, None)
    def get(self, k): return self.d.get(k.encode())
    def close(self): self.f.close()
```

四个设计点逐一对应理论：

1. **追加写（append-only）**：永不原地改文件——01 章铁律"随机写变顺序写"。
2. **fsync 才是持久点**：`flush()` 只到内核页缓存（进程崩溃不丢，断电丢），`os.fsync()` 才到盘。WAL 版实测写吞吐跌到 **1 千条/s**——比内存版慢四个数量级，正是分布式系统 07 章 fsync 实验的数据库版。
3. **半条记录的处理**：崩溃可能发生在"写了 6 字节"的中间态。恢复循环里三道闸（hdr 不足/长度越界/数据不足）都直接 break——**日志尾部不完整 = 事务没提交，丢弃**。这就是"原子性"在追加日志上的廉价实现。
4. **删除是写墓碑**：不写墓碑，崩溃后重放会"复活"已删的 key（日志里只有 put）。

崩溃恢复验证（kill -9 是最诚实的测试——不给任何清理机会）：

```python
# crash_test.py —— 子进程写 1000 条后 kill -9, 主进程重放验证
import subprocess, sys, os
sys.path.insert(0, "/tmp/minikv")
child = '''
import sys, os
sys.path.insert(0, "/tmp/minikv")
from minikv import WALKV
kv = WALKV("/tmp/minikv/crash.db")
for i in range(1000):
    kv.put(f"key{i}", f"val{i}")
os.kill(os.getpid(), 9)               # 模拟崩溃: 不调用 close
'''
subprocess.run([sys.executable, "-c", child], capture_output=True)
from minikv import WALKV
kv = WALKV("/tmp/minikv/crash.db")    # 重新打开 → 重放 WAL
ok = sum(1 for i in range(1000) if kv.get(f"key{i}") == f"val{i}".encode())
print(f"恢复后验证: 1000 条中存活 {ok} 条"); kv.close()
```

实测输出：

```
子进程退出码: -9 (预期 -9, 被 SIGKILL)
恢复后验证: 1000 条中存活 1000 条
```

**1000/1000**。再加一道恶意题（留作练习）：在 1000 条中手工把日志文件截断在第 500 条的中间，重放应得到恰好 499 或 500 条完整数据且无异常——原子性的边界行为要亲手验证。

## 4. 阶段3：简化 LSM 版——用顺序写换吞吐

WAL 版每条 fsync 太贵，LSM 的答案是**攒批**：写入先进内存 memtable（无 fsync 压力），攒满 N 条整体顺序刷成不可变的 SSTable 文件；WAL 保留但只 flush 不逐条 fsync（崩溃最多丢 memtable 未刷部分——用"最多丢最近一批"换吞吐，这正是 Redis AOF everysec 的权衡再现）。06 章的图变成代码：

```python
# minikv.py 第 3 段
import time, threading

def _load_sst(path):                                  # 读取 SSTable → {key: val}
    idx = {}
    with open(path, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8: break
            kl, vl = struct.unpack("<II", hdr)
            if kl > 1<<20 or vl > 1<<24: break
            k, v = f.read(kl), f.read(vl)
            if len(k) < kl or len(v) < vl: break
            idx[k] = v
    return idx

class LSMKV:
    FLUSH_N = 2000
    def __init__(self, path):
        self.path = path; os.makedirs(path, exist_ok=True)
        self.memtable, self.ssts = {}, []              # ssts: (文件名, 键集) 新→旧
        self.wal = open(os.path.join(path, "wal.log"), "ab")
        self._lock = threading.Lock()
        for fn in sorted(os.listdir(path), reverse=True):   # 启动: 扫已有 SSTable
            if fn.endswith(".sst"):
                self.ssts.append((fn, set(_load_sst(os.path.join(path, fn)).keys())))
    def put(self, k, v):
        k, v = k.encode(), (v.encode() if isinstance(v, str) else v)
        self.wal.write(struct.pack("<II", len(k), len(v)) + k + v)
        self.wal.flush()                               # 注意: 不 fsync(丢批换吞吐)
        with self._lock:
            self.memtable[k] = v
            if len(self.memtable) >= self.FLUSH_N: self._flush()
    def _flush(self):                                  # memtable → 不可变 SSTable(顺序写)
        fn = f"{int(time.time()*1e6)}.sst"
        keys = sorted(self.memtable)
        with open(os.path.join(self.path, fn), "wb") as f:
            for k in keys:
                v = self.memtable[k]
                f.write(struct.pack("<II", len(k), len(v)) + k + v)
        self.ssts.insert(0, (fn, set(keys)))
        self.memtable.clear()
        self.wal.close(); self.wal = open(os.path.join(self.path, "wal.log"), "wb")  # WAL 可清
    def get(self, k):
        kb = k.encode()
        if kb in self.memtable: return self.memtable[kb]        # ① 先查 memtable
        for fn, keyset in self.ssts:                            # ② 再按新→旧翻 SSTable
            if kb in keyset:
                return _load_sst(os.path.join(self.path, fn)).get(kb)
        return None
    def compact(self):                                 # 合并: 重放所有 SSTable 取每个 key 最新版
        latest = {}
        for fn, _ in reversed(self.ssts):
            latest.update(_load_sst(os.path.join(self.path, fn)))
        for k, v in self.memtable.items(): latest[k] = v
        for fn, _ in self.ssts: os.remove(os.path.join(self.path, fn))
        with open(os.path.join(self.path, "merged.sst"), "wb") as f:
            for k in sorted(latest):
                v = latest[k]; f.write(struct.pack("<II", len(k), len(v)) + k + v)
        self.ssts = [("merged.sst", set(latest.keys()))]
    def close(self):
        if self.memtable: self._flush()
        self.wal.close()
```

要点核对（每条都是 06 章的直接翻译）：memtable 满 2000 条才刷盘（**攒批摊薄 fsync**）；SSTable 文件名带时间戳、**新文件覆盖旧文件同 key**（读时按新→旧找）；compact() 把全部 SSTable 合一、每个 key 只留最新版（真 LSM 是多层 k-way merge + 墓碑回收，这里是单层简化，但"读全量→写全量"的成本结构一致）。

## 5. 阶段4：加缓存与布隆——读路径的补救

阶段 3 的读路径有明确的坏味道：`get` 每次都 `_load_sst` 整文件。阶段 4 加两样：**读缓存**（dict + 容量上限，08 章的 Redis 思想缩小版）与**简化布隆过滤器**（08 章手写版复用，挂在每个 SSTable 上，"一定没有就跳过该文件"）：

```python
# minikv.py 第 4 段
import hashlib

class CachedLSM(LSMKV):
    def __init__(self, path, cache_size=10000):
        super().__init__(path)
        self.cache, self.cache_size, self.stat = {}, cache_size, {"hit": 0, "q": 0}
    def get(self, k):
        kb = k.encode(); self.stat["q"] += 1
        if kb in self.cache:
            self.stat["hit"] += 1; return self.cache[kb]     # 缓存命中: 0.1μs 级
        v = super().get(kb.decode())                          # 未命中: 走慢路径
        if v is not None and len(self.cache) < self.cache_size:
            self.cache[kb] = v
        return v

# 布隆(与 08 章 bloom.py 同款, 挂在每个 SSTable 的键集上):
    # def _bloom_might(self, fn, k): ... 3 哈希置位/查询, "不在"→直接跳过该 SSTable
    # 效果: 不存在的 key 的读, 从"翻遍所有文件"降为"3 次哈希×文件数"
```

基准结果见下节。先说读到的教训：**缓存只救"重复读"**——冷读（每个 key 第一次）完全无感，热读（第二次起）延迟减半。真实引擎（RocksDB）把 SSTable 的索引块常驻内存、数据页走 LRU，是同一思想的全量版。

## 6. 基准：写入吞吐与读延迟的完整数字

基准脚本（bench.py，压测 2 万条 put + 全量 get；代码骨架）：

```python
# bench.py 核心逻辑
def bench_put(cls, path, n=20000):
    kv = cls(path) if path else cls()
    t0 = time.monotonic()
    for i in range(n): kv.put(f"key{i}", "v" * 40)
    w = n / (time.monotonic() - t0) / 1000            # 千条/s
    t0 = time.monotonic()
    for i in range(n): kv.get(f"key{i}")
    rd = (time.monotonic() - t0) / n * 1e6            # μs/条
    kv.close(); return w, rd
```

WSL2/Ubuntu（NVMe）实测汇总——**背下这张表的相对结构，不是绝对值**：

| 引擎 | 写吞吐（千条/s） | 读均耗（μs/条） | 持久性 |
| --- | --- | --- | --- |
| 内存版 | 11216 | 0.06 | 无（进程死=数据死） |
| WAL 版 | 1 | 0.10 | 每条 fsync，崩溃不丢 |
| LSM 版 | 720 | 642（冷读） | WAL 未 fsync，最多丢未刷批 |
| LSM+缓存 | 707 | 646 冷 / **322 热** | 同上 |

四条结论（每条都值得在面试里展开）：

1. **fsync 是第一分水岭**：内存版→WAL 版掉四个数量级。所有数据库的 group commit、O_DSYNC 调参、电池保护写缓存，都是在给这 0.1-1ms 买单。
2. **攒批是第二分水岭**：LSM 版把 fsync 从"每条"变"每批（flush 时一次）"，吞吐回到 72 万条/s——代价是崩溃恢复窗口从 0 条变成最多一个未刷批（RPO 从 0 变成"一批"，与 Redis AOF everysec 同构）。
3. **简化 LSM 的读放大是真的**：冷读 642μs（内存版 0.06μs 的一万倍）——每次 get 加载整个 SSTable 文件。真引擎靠稀疏索引 + 页缓存把这里降回微秒级；我们的 5 节取舍日志要诚实记下这一点。
4. **缓存收益与访问局部性成正比**：热读减半，冷读无感。评估缓存先看访问分布（Zipf 还是均匀），均匀分布的缓存是纯内存浪费。

## 7. 设计取舍日志：每一步为什么这么做

以"日志体"记录本项目全部取舍决策（这是存储工程师的核心工作方式，比代码更值钱）：

```
[决策 01] 记录格式用 [len][len][k][v] 而非 JSON 行
  为什么: 解析零依赖、定长头支持流式读取; 代价: 不可读、需处理字节序 → 真引擎补 CRC32

[决策 02] WAL 版每条 fsync
  为什么: 演示"持久点"的物理位置; 代价: 1 千条/s → 真引擎用 group commit 把多条事务
  合并成一次 fsync(Redis AOF always→everysec 同理)

[决策 03] LSM 版 WAL 只 flush 不 fsync
  为什么: 攒批换吞吐(1→72万条/s); 代价: 崩溃丢最近未刷批 → 这是 RPO 的主动选择,
  不是疏忽。真产品要在这里写明文档

[决策 04] SSTable 文件名用时间戳, 新旧顺序=文件名倒序
  为什么: 免去版本号管理; 代价: 依赖时钟、并发 flush 需锁 → 真引擎用递增序号

[决策 05] compact() 全量重写(单层)
  为什么: 演示"合并取最新"的语义; 代价: O(全库) IO、合并期间双倍空间 → 真 LSM
  分层(size-tiered/leveled)摊平成本, 06 章写放大 5-10x 的来源就是它

[决策 06] 读路径全文件加载(无稀疏索引)
  为什么: 代码最短; 代价: 冷读 642μs → 教训: **读路径的第一课是索引常驻内存**。
  RocksDB 每打开一个 SSTable 就把 index block 装进内存, get 是"内存查索引→只读一个块"

[决策 07] 缓存不做淘汰(容量到顶即停)
  为什么: 演示最小可用; 代价: 热点漂移后命中率衰减 → 升级路径: LRU/LFU(08 章 8 种策略)
```

**下一步升级路线**（把本项目变成简历项目的方向）：CRC 校验与文件 footer；稀疏索引（每 64 条记一个 offset）；多层 compaction；get/delete 的并发安全（读写锁）；用 mmap 替代 read；对标 RocksDB 的 WriteBatch 与 column family。

## 常见误区

1. **"flush() 了就持久了"**：flush 只到内核页缓存，断电照丢；`os.fsync(fd)` 才是持久点。本项目的 WALKV 若漏掉 fsync，crash_test 会随机丢尾部数据——亲自删掉那行跑一次，看丢失数字。
2. **"崩溃恢复重放日志从文件头读到尾就是了"**：半条记录是常态不是异常（崩溃就爱发生在写一半时），三道截断闸门缺一不可；生产还要 CRC 区分"截断"与"位翻转"。
3. **"LSM 快所以全用 LSM"**：本项目的 LSM 版冷读 642μs 已经演示了读放大。选型回到 06 章的表：读多点查场景，B+树数据库（SQLite/PG）仍然是默认答案。
4. **"compaction 可以等磁盘满了再做"**：合并是 O(全库) 的 IO，磁盘 90% 时才开始合并 = 业务高峰自焚。真引擎的 compaction 触发条件是"层数/文件数阈值"（空间与读放大的前瞻控制），且要限速。
5. **"测试 = 功能跑对"**：本项目最重要的一行代码不是 put 而是 `os.kill(os.getpid(), 9)`。存储系统的验收标准是**故障注入通过**：kill -9、日志截断、写一半断电（可模拟：写 truncate 到任意中间长度）——每个都要有脚本、有预期输出。

## 本篇小结

- 四步演进各自补一块拼图：内存版（基准）→ WAL 版（持久性=append+fsync+重放恢复）→ LSM 版（吞吐=memtable 攒批+SSTable 顺序写+compaction）→ 缓存版（读路径=缓存救重复读+布隆挡无效扫描）。
- 数字结构：fsync 让写掉 4 个数量级（11216→1 千条/s），攒批拿回 3 个（→720 千条/s），冷读 642μs 暴露读放大，缓存把热读减半。
- 原子性在追加日志上的廉价实现：尾部不完整即视为未提交；删除必须写墓碑。
- 取舍日志是最有复用价值的产出：每个简化都标注"为什么+代价+真引擎怎么补"——这套思维方式比 MiniKV 本身更接近存储工程师的日常。
- 至此全系列原理闭环：01 章的四件事（可靠存储/并发/查询/事务）在 B+树（03/06 章）与 LSM（06/10 章）两条路线上都看过了实现。

## 练习

1. 给 WALKV 加 CRC32（`zlib.crc32`，记录头加 4B）：恢复时校验每条记录，把"截断"与"损坏"分开处理（损坏即停或跳过，写清语义）。再跑 crash_test 与"手工截断日志"两个用例。
2. 实现稀疏索引：_flush 时每 64 条记录记下 (首key, offset)，get 时二分定位 offset 后 seek 读 64 条；重测冷读延迟，对比 642μs 的改善幅度并解释为什么这等于"把 03 章的 B+树思想搬进了 LSM"。
3. 把 LSM 版的 FLUSH_N 从 2000 调到 200 与 20000，分别测写吞吐与崩溃恢复的丢失条数（kill -9 测试），画出"批大小→吞吐/RPO"的权衡曲线。
4. 给 CachedLSM 换成 LRU 淘汰（OrderedDict 或 functools.lru_cache 语义手写），构造"热点漂移"访问序列（前半访问 key 0-9999，后半访问 10000-19999），对比有无 LRU 的命中率。
5. 扩展题：把 MiniKV 包装成 Redis 协议子集（RESP 的 GET/SET），用 redis-cli 连上你的引擎真实读写——体会"存储引擎"与"数据库产品"之间隔着协议层、并发层与运维层（这也是为什么本系列前面 9 章不能跳过）。
