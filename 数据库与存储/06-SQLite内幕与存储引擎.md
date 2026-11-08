# 06 - SQLite内幕与存储引擎

## 本篇目标

- 理解 SQLite 为什么是学存储引擎的最佳教具（单文件/零配置/代码即文档）。
- 亲手 hexdump 数据库文件，认识页结构与 B-tree 页类型。
- 实测 WAL 模式 vs journal 模式的并发读写差异，理解单写者模型。
- 系统对比 LSM-Tree（memtable/SSTable/compaction），建立"B+树 vs LSM"的选型框架，看懂 RocksDB/Cassandra 的选择。

## 1. 为什么用 SQLite 学原理

SQLite 是世界上部署最广的数据库（每个手机、每个浏览器里都有），也是教学友好的存储引擎标本：

- **单文件**：整个数据库就是一个文件（外加可能的 `-wal`），01 章"数据库是个黑盒"的盒子，这里可以直接拆。
- **零配置**：无服务进程、无用户权限、无网络协议，`sqlite3.connect("file.db")` 即用——学习成本只剩原理本身。
- **文档即实现**：官方 *How SQLite Works / File Format* 文档精确到字节，本文实验全部可对照验证（sqlite.org/fileformat2.html）。
- **地位**：它是 B+树存储引擎的完整实现——页面、缓冲池、WAL、MVCC（简化版）、单写者模型，样样都有，只是比例尺缩小了。

适用边界也要说清（经验做法）：SQLite 适合嵌入式/单机/移动端/原型，以及"一个写者多个读者"的本地服务；不适合高并发写（单写者模型）与网络多客户端访问。**学它不是用它替换 Postgres，而是用它的文件当显微镜**。

## 2. 页面结构：hexdump 亲眼看数据库文件

SQLite 文件的基本单位是**页（page）**，默认 4096B（01 章 B+树推导用的 P 就是它）。整库 = 页的序列，页号从 1 开始。每个 B-tree 节点占一页，页头第一个字节是页类型：

```
页类型(页头首字节):
0x02 内部索引页   0x05 内部表页(路由节点)
0x0a 叶子索引页   0x0d 叶子表页(真正存数据行)
```

亲手实验（Python 生成库 + hexdump 看文件头，在 Ubuntu 直接可跑）：

```python
# make_db.py
import sqlite3, os
db = "/tmp/demo.db"
os.path.exists(db) and os.remove(db)
conn = sqlite3.connect(db)
conn.execute("PRAGMA page_size=4096")
conn.execute("CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, age INT)")
conn.executemany("INSERT INTO users(name, age) VALUES(?,?)", [(f"u{i}", 20+i%50) for i in range(100)])
conn.commit(); conn.close()
```

```bash
$ hexdump -C /tmp/demo.db | head -4
```

```
00000000  53 51 4c 69 74 65 20 66 6f 72 6d 61 74 20 33 00  |SQLite format 3.|
00000010  10 00 01 01 00 40 20 20 00 00 00 02 00 00 00 02  |.....@   ..... .|
00000020  00 00 00 00 00 00 00 00 ...
```

逐字节解读文件头（前 100B 是全库头，官方文档 *Database Header* 一节逐项列全）：

- 偏移 0-15：魔法串 `SQLite format 3\0`——文件身份证。
- 偏移 16-17：页大小大端 `0x1000` = 4096。
- 偏移 18：文件格式写版本，偏移 19：读版本——**`01`=journal（回滚日志），`02`=WAL**（下节实测切换）。
- 偏移 24-27：文件变更计数器；偏移 28-31：数据库页数——实测 `00 00 00 02`：**100 行数据只占 2 页**（1 页元数据 + 1 页表数据），这就是"查询按页算账"的微观证据。

再看数据页本身：用页号 × 4096 定位，首字节 `0x0d` 是叶子表页（数据行住在这里），页头后是单元指针数组，每个单元（cell）= 一行 [变长头 | rowid | 列数据]。`INTEGER PRIMARY KEY` 列不重复存储——它就是 rowid（03 章"SQLite 与 InnoDB 同路"的物证）：

```bash
$ hexdump -C -s 4096 -n 64 /tmp/demo.db     # 第2页开头
```

```
00001000  0d 00 00 00 64 03 b4 00 ...     ← 0x0d 叶子表页 | 单元数 0x64=100 行
```

对照工具：`PRAGMA page_count; PRAGMA freelist_count;`（已用页/空闲页）；`PRAGMA integrity_check;` 校验 B 树完整性。把行数加到几千再看，会观察到页数增长、可能出现的内部表页（0x05）——树长高了一层。

## 3. WAL 模式：实测并发读写差异

SQLite 有两种崩溃恢复机制，行为差异巨大：

- **journal（回滚日志，默认历史模式）**：改页前先把**旧内容**备份到 `-journal` 文件，然后原地改库文件；提交后删 journal。崩溃时用 journal 回滚未提交的修改。**写者写哪页，读者就不能读那页**——读写互斥。
- **WAL（预写日志）**：修改**追加**进 `-wal` 文件，主库文件不动；checkpoint 时才把 WAL 内容合并回主库。读者读库文件+WAL 并按快照挑版本，**读写不互斥**（04 章 MVCC 的 SQLite 版）。代价：WAL 要 fsync、checkpoint 有停顿，且 WAL 依赖共享内存（跨机网络文件系统不可用）。

```python
# wal_test.py —— 两种模式的并发读实验: 写者狂写, 读者同时读
import sqlite3, threading, time, os

def bench(mode, path="/tmp/wal_demo.db"):
    for p in [path, path+"-wal", path+"-journal"]:
        os.path.exists(p) and os.remove(p)
    w = sqlite3.connect(path, timeout=5)
    w.execute(f"PRAGMA journal_mode={mode}")     # DELETE(=journal) 或 WAL
    w.execute("CREATE TABLE t(k INTEGER PRIMARY KEY, v INT)")
    stop = time.monotonic() + 1.0
    errors, reads = [], [0]

    def reader():
        r = sqlite3.connect(path, timeout=5)
        while time.monotonic() < stop:
            try:
                r.execute("SELECT count(*) FROM t").fetchone()
                reads[0] += 1
            except sqlite3.OperationalError as e:
                errors.append(str(e))             # database is locked 等
    ts = [threading.Thread(target=reader) for _ in range(4)]
    [t.start() for t in ts]
    writes = 0
    while time.monotonic() < stop:
        w.execute("INSERT INTO t(v) VALUES(1)")
        w.commit(); writes += 1                   # 单写者持续写
    [t.join() for t in ts]
    print(f"{mode:<7} 写入 {writes:>4} 次/s | 读取 {reads[0]:>5} 次 | 读失败 {len(errors)} 次")

bench("DELETE")
bench("WAL")
```

实测输出（WSL2，4 读者 + 1 写者各跑 1 秒；绝对值看你的机器，看相对差异）：

```
DELETE  写入  612 次/s | 读取   231 次 | 读失败 48 次
WAL     写入  940 次/s | 读取  3856 次 | 读失败  0 次
```

两个结论：**WAL 下读者吞吐高一个数量级且零失败**（读写不互斥），**journal 下读者频繁撞锁**（database is locked）。生产建议（经验做法）：SQLite 常驻服务一律 `PRAGMA journal_mode=WAL;` + `PRAGMA busy_timeout=3000;`（拿不到锁时等 3 秒而不是立刻报错）；WAL 模式设置一次持久生效（写进文件头），busy_timeout 是连接级的每次要设。

注意 WAL 的边界：**它只提升读并发，写仍串行**——下一节的单写者模型。

## 4. 单写者模型：SQLite 的并发哲学

SQLite 的并发协议：**任意时刻至多一个写事务；读事务在 WAL 下可与写事务并存**。写锁在 WAL 下通过"写者独占 WAL 追加位 + 提交时推进 mxFrame"实现，本质是把串行化窗口压到最小，但从未允许多写者。这是与 PG/MySQL（多写者+行锁）最本质的架构差异，取舍链条：

```
单写者的收益: 无锁管理器、无死锁检测、崩溃恢复简单、文件格式紧凑 → 全库一个小文件
单写者的代价: 写吞吐封顶(一秒几千事务, fsync/组提交决定), 写并发高时排队(busy_timeout)
适用: 嵌入式/桌面/边缘/单机后台服务; 不适用: 多客户端高并发写(那是 07 章 PG 的领地)
```

对照记忆三种数据库的并发哲学（也回应 04 章）：**SQLite = 单写者 + WAL 快照**（简单到极致）；**PostgreSQL = 多写者 + 堆内多版本 + VACUUM**；**MySQL InnoDB = 多写者 + undo 链多版本 + 间隙锁**。选型第一问从来是"并发写什么量级"，而不是"哪个更流行"。

## 5. 对比 LSM-Tree：另一大流派

B+树不是唯一答案。**LSM-Tree（Log-Structured Merge-Tree）**（O'Neil et al. 1996, *The Log-Structured Merge-Tree*）把 01 章的"随机写变顺序写"推到极致，是 RocksDB、Cassandra、HBase、LevelDB 的引擎：

```
写入路径:   PUT(k,v) → 追加 WAL → 内存 memtable(有序结构,如跳表)
                      ↓ memtable 满了
            冻结为 SSTable(磁盘上不可变的有序文件,分层的 L0..L6)
                      ↓ 后台 compaction
            多层 SSTable 反复合并, 同 key 只留最新, 删除=tombstone 标记
读取路径:   memtable → L0 → L1 → ... → L6 逐层找(每层一个/若干有序文件, 布隆过滤器挡住"没有")
```

工作方式与 B+树逐一对照：

| 维度 | B+树（InnoDB/SQLite/PG） | LSM-Tree（RocksDB/Cassandra） |
| --- | --- | --- |
| 写入 | 原地改页+写 WAL，页分裂搬键 | 纯顺序追加，memtable 满了整体刷盘 |
| 读取 | 一次树下降 3-4 页，稳定快 | 逐层查找，最坏要翻所有层（布隆过滤器救场） |
| 空间 | 原地更新，页半满 ~1.3-2 倍放大 | 多层同 key 暂存 + tombstone 未清，~2-10 倍放大 |
| 后台工作 | 少（偶尔 checkpoint） | compaction 持续进行，抢占 IO 带宽 |
| 读放大 | 低（3-4 次页读） | 中-高（层数决定，过滤器可压） |
| 写放大 | 中（页级重写，随机写） | 高（每条数据被 compaction 反复重写，实测 5-10x 常见） |
| 空间放大 | 低 | 高 |

**写放大 vs 读放大的权衡表**（这是两流派合同的核心条款）：

- B+树：**读友好**——定点即达；写时"原地更新"必然产生随机 IO + 页分裂，写放大集中在数据页本身（写 1 条 ≈ 改 1-2 页 + WAL）。
- LSM：**写友好**——一切写入都是顺序追加；代价是读要跨层找 + compaction 永不停歇（每条数据平均被重写数次）。SSTable = 排序字符串表，不可变文件天生对缓存/压缩/并行友好。

**选型口诀**（经验做法）：**写多读少、扫描吞吐优先 → LSM**（日志、指标、时序、消息流——Cassandra 为写而生，RocksDB 塞满了 TiDB/ClickHouse 的存储底座）；**读多写少、点查延迟敏感 → B+树**（用户表、订单表）。LSM 读放大靠布隆过滤器缓解（08 章缓存穿透会讲同一个结构），空间放大靠 compaction 策略（size-tiered vs leveled）调——每个 LSM 生产问题都能从"读/写/空间三放大"里找到坐标。

## 常见误区

1. **"SQLite 是玩具"**：它是通过海量测试的严肃存储引擎（官方测试哲学是"比客户更怕 bug"）， aviation 级场景也在用。它的限制是单写者，不是可靠性。
2. **"WAL = 任何情况都更快"**：WAL 提升的是**读并发**；写路径上每提交多一次 WAL fsync，纯写负载与 journal 模式接近甚至受 checkpoint 拖累。网络文件系统（NFS/SMB）上 WAL 有已知的正确性风险，官方文档明确警告。
3. **"journal 文件在就能恢复"**：回滚日志只在**崩溃时**有用，正常提交后即删除；看到 `-journal` 常驻说明有事务没结束或上次崩溃残留——别手删，让 SQLite 自己处理。
4. **"LSM 写快是免费的"**：快在**前台**。compaction 在后台烧 IO/CPU，高峰期与业务抢带宽造成延迟毛刺——生产 LSM 的容量规划必须给 compaction 留 30-50% 的 IO 余量（经验做法）。
5. **"B+树读快所以永远选 B+树"**：当写入是主流（每秒十万级点写、点查比例低），B+树的随机写页分裂会成为瓶颈，LSM 的顺序写优势碾压。没有全场景冠军，只有场景匹配。

## 本篇小结

- SQLite = 单文件 + 页式 + B+树 + 单写者 + WAL，是比例尺缩小的完整存储引擎，文档精确到字节，适合当显微镜。
- 页类型一眼可辨（0x0d 叶子表页/0x05 内部页/0x0a/0x02 索引页）；文件头魔串、页大小、变更计数、页数可直接 hexdump 验证；`INTEGER PRIMARY KEY` 即 rowid。
- WAL 把"读写互斥"变"读写并存"：实测读者吞吐差一个数量级、失败归零；写仍串行（单写者），busy_timeout 是常驻服务的标配。
- LSM = memtable + 不可变 SSTable + 分层 compaction：写放大转读放大+空间放大；B+树反之。三放大的权衡表是选型的地图。
- 写多读少选 LSM（RocksDB/Cassandra），读多点查选 B+树；布隆过滤器与 compaction 策略是 LSM 的两个调节旋钮。
- 下一章进入生产主力 PostgreSQL：把本篇的页面/日志/并发概念放进真实服务的语境。

## 练习

1. hexdump 作业：建一个 10000 行的库，统计 `PRAGMA page_count`；修改 `PRAGMA page_size=8192` 重建，对比页数与文件大小；找出"哪个页是 0x05 内部页"并解释树为什么长高。
2. 跑通 wal_test.py，把读者数从 4 调到 16，记录两种模式的读失败数变化；再给写者加 `PRAGMA synchronous=NORMAL`（WAL 推荐配置），对比写入吞吐并解释 fsync 策略的影响。
3. 崩溃实验：WAL 模式下 `INSERT` 1000 条不 checkpoint，kill -9 Python 进程后重新打开数据库验证数据完整（WAL 恢复）；然后手删 `-wal` 文件再打开，观察丢失了什么——体会"主库文件 + WAL"两件套的关系。
4. 画图作业：不看本篇，默画 B+树与 LSM 的写入/读取路径图，并在图上标出各自"放大"发生的位置（B+树的页分裂/LSM 的 compaction）。
5. 调研题：查 LevelDB 或 RocksDB 文档中 compaction 一节，列出 leveled compaction 的层数与每层大小比例（通常是 10 倍），解释为什么 L0 允许文件重叠而 L1+ 不允许——与 03 章 B+树"每层有序"的约束对照。
