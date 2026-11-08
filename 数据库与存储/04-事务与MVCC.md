# 04 - 事务与MVCC

## 本篇目标

- 快速回顾 ACID 与隔离级别（理论细节见分布式系统 07 章，本章只补"实现视角"）。
- 用**同一套实验序列**在 READ COMMITTED 与 REPEATABLE READ 下跑出不同结果，把并发异常从纸面变成可复现。
- 拆开 MVCC：版本链、ReadView/快照的判定规则，配图讲清"读的是哪个版本"。
- 实测 PostgreSQL 的 xmin/xmax 多版本痕迹；理解行锁/间隙锁的角色与长事务对 VACUUM/undo 的拖累。

## 1. ACID 与隔离级别：30 秒回顾 + 实现视角

分布式系统 07 章已从"异常现象"角度讲完 ACID 与四级隔离；本章补**单机实现**这一半。两张表先对齐术语：

| 隔离级别 | 脏读 | 不可重复读 | 幻读 | 典型实现 |
| --- | --- | --- | --- | --- |
| READ UNCOMMITTED | 可能 | 可能 | 可能 | 直接读最新（几乎无人用） |
| READ COMMITTED (RC) | 不可能 | 可能 | 可能 | 每条语句一个快照 |
| REPEATABLE READ (RR) | 不可能 | 不可能 | 标准说可能* | 每个事务一个快照 |
| SERIALIZABLE | 不可能 | 不可能 | 不可能 | SSI / 严格 2PL |

*PostgreSQL 的 RR = 快照隔离（Snapshot Isolation），SELECT 不会看到新行，但存在写偏斜（write skew）异常；MySQL InnoDB 的 RR 用 next-key lock 基本堵住幻读；SQLite 只支持 SERIALIZABLE（且同一时刻只允许一个写者，06 章）。SQL 标准与实现细节见 Berenson et al. 1995（*A Critique of ANSI SQL Isolation Levels*）。

实现视角的一句话总纲：**A 靠 undo/回滚（撤销未提交），D 靠 redo/WAL（重做已提交，第 06/10 章），I 靠锁 + MVCC，C 靠应用不变量**。MVCC 之前，实现 I 的主流是两阶段锁（2PL）：读者也持锁，读写互斥排队；MVCC（多版本并发控制）的思路是**为每个事务保留数据的多个版本，读旧版本、写新版本，读写互不阻塞**——代价是存储要放得下多个版本，且要有人清理死版本。这是"用空间和后台工作换并发"的定价，本章的主线。

## 2. 并发异常三例：同一实验序列，两种隔离级别

写一个可复现实验：两个连接并发，T1 读两次同一行，T2 在中间改它。通过调整 T1 的隔离级别，观察两次读的结果是否一致（SQLite 默认 SERIALIZABLE 无法演示脏读，本节用 PostgreSQL 16 实测；MySQL 的设置方式随文标注）：

```sql
-- 准备
CREATE TABLE acct(id INT PRIMARY KEY, bal INT);
INSERT INTO acct VALUES (1, 100);

-- 会话 T1 (READ COMMITTED)
BEGIN;
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;   -- MySQL: SET SESSION TRANSACTION ISOLATION LEVEL ...
SELECT bal FROM acct WHERE id=1;   -- 第一次读 → 100
-- 此处切换到 T2 执行:  UPDATE acct SET bal=200 WHERE id=1; COMMIT;
SELECT bal FROM acct WHERE id=1;   -- 第二次读 → ???  (不可重复读出现点)
COMMIT;
```

实测（psql 两个终端，或 Python 双连接）：

```
T1 @ READ COMMITTED:  第一次读 100, 第二次读 200   ← 不可重复读: 同事务两次读不一致
T1 @ REPEATABLE READ: 第一次读 100, 第二次读 100   ← 快照冻结了 100, T2 的修改对 T1 不可见
```

三个异常用同一框架归纳（时序约定：T1 是受害者，T2 是干扰者）：

```
脏读:        T1读到 T2 未提交的值     | RC 就禁止(只读已提交)    | 实现上: 读时检查 xmax 是否已提交
不可重复读:  T1两次读同一行, 值变了   | RC 出现, RR 禁止        | 实现上: RC 每语句新快照, RR 事务开头一次快照
幻读:        T1两次同一条件查询, 行数变| RC 出现; RR 视实现而定  | 实现上: 快照锁住"已存在行的旧版本",
             (新 INSERT 的行)                                 但"新出现的行"需要谓词/间隙锁或 SSI 才挡得住
```

实验脚本化（Python + psycopg，双连接并行，可放进 CI 跑）：

```python
# anomaly_demo.py —— READ COMMITTED vs REPEATABLE READ 对比 (需 pip install "psycopg[binary]")
import psycopg

def run(isolation):
    c1 = psycopg.connect("dbname=test", autocommit=True)
    c2 = psycopg.connect("dbname=test", autocommit=True)
    c1.execute("DROP TABLE IF EXISTS acct; CREATE TABLE acct(id INT PRIMARY KEY, bal INT)")
    c1.execute("INSERT INTO acct VALUES (1, 100)")
    c1.execute(f"SET TRANSACTION ISOLATION LEVEL {isolation}")
    c1.execute("BEGIN")
    r1 = c1.execute("SELECT bal FROM acct WHERE id=1").fetchone()[0]   # T1 第一次读
    c2.execute("UPDATE acct SET bal=200 WHERE id=1")                    # T2 修改并提交
    c2.execute("COMMIT")
    r2 = c1.execute("SELECT bal FROM acct WHERE id=1").fetchone()[0]   # T1 第二次读
    c1.execute("COMMIT")
    print(f"{isolation:<18} 第一次读={r1}  第二次读={r2}  {'不一致!' if r1 != r2 else '一致'}")

run("READ COMMITTED")      # 第一次读=100  第二次读=200  不一致!
run("REPEATABLE READ")     # 第一次读=100  第二次读=100  一致
```

注意一个实现细节：psycopg 默认非自动提交模式下 `SET TRANSACTION ISOLATION LEVEL` 必须紧跟 BEGIN，所以脚本用 autocommit 手动开事务——**隔离级别是事务的属性，语句启动事务时才生效**，这是很多人"设置了没效果"的原因（PostgreSQL 文档 13.2.1）。

## 3. MVCC 原理：版本链与快照判定

MVCC 的核心：**每行有隐藏的创建者/删除者标记，修改不覆盖旧数据，而是追加新版本**。各库实现不同但思想一致，以 PostgreSQL 讲（它把版本直接放在堆里，最直观）：

```
堆中的行版本 (PostgreSQL):
┌─────────────────────────────────────┐
│ (id=1, bal=100)  xmin=100  xmax=205 │ ← 旧版本: 被 xid=205 的事务标记删除
│ (id=1, bal=200)  xmin=205  xmax=0   │ ← 新版本: 由 xid=205 创建, 还活着
└─────────────────────────────────────┘
xmin = 创建该版本的事务ID    xmax = 删除/更新该版本的事务ID (0=未删)

快照(snapshot) = "我启动时, 哪些事务算已提交、哪些算进行中"
ReadView/快照判定: 对每个候选版本自老向新检查——
  ① xmin 未提交(进行中/已回滚)        → 看不见
  ② xmin 已提交 且 xmax=0             → 可见
  ③ xmax 已提交且 < 我快照的边缘       → 该版本已被删, 顺链找下一个新版本
  一句话: "在我眼里已经存在且还没被删的最新版本"
```

InnoDB 的实现是**回滚段版本链**：聚簇索引的行里藏 `DB_TRX_ID`（最后修改者）与 `DB_ROLL_PTR`（指向前一版本的 undo 记录），旧版本串成链挂在 undo log 里；MySQL 的 RR 快照在**事务第一次读时**生成（RC 是每条语句生成一次）——这一个差异就是"RC 有不可重复读、RR 没有"的全部实现原因。PostgreSQL 则是**新版本直接插在堆表里**，旧版本原地保留，等 VACUUM 清理（第 5 节）。

两个由此而来的关键语义：

1. **快照读不加锁**。RR 下的普通 SELECT 只是"挑版本"，零锁零等待——这是 MVCC 读写互不阻塞的来源，也是它优于 2PL 的地方（读者密度高时吞吐差一个数量级）。
2. **写写仍然冲突**。两个事务改同一行，后提交者要么覆盖（丢失更新风险）要么被阻塞/报错，取决于是否 SELECT FOR UPDATE 或可串行化隔离。**MVCC 只解决读写并发，不解决写写并发**——这是"用了 MVCC 为什么还丢数据"的答案。

## 4. PostgreSQL 实测：xmin/xmax 与 ctid

PG 的版本标记可以直接 SELECT 出来，这是理解 MVCC 最好的教具（MySQL 要看 `information_schema.innodb_trx` 与 hexdump 才能间接观察）：

```sql
CREATE TABLE t(id INT PRIMARY KEY, v INT);
INSERT INTO t VALUES (1, 10);                       -- 事务A
SELECT ctid, xmin, xmax, * FROM t;
```

```
 ctid  | xmin | xmax | id | v
-------+------+------+----+----
 (0,1) |  752 |    0 |  1 | 10     ← ctid=(页号,页内槽号) 即物理位置; xmin=创建者752; xmax=0 未删
```

```sql
BEGIN; UPDATE t SET v = 20 WHERE id = 1;            -- 事务B, xid=753
SELECT ctid, xmin, xmax, * FROM t;                  -- 事务B自己看
```

```
 ctid  | xmin | xmax | id | v
-------+------+------+----+----
 (0,1) |  752 |  753 |  1 | 10     ← 旧版本: 被本事务(753)标记删除
 (0,2) |  753 |    0 |  1 | 20     ← 新版本: 新物理位置(0,2), 本事务创建
```

同一行两个版本同表共存，这就是"多版本"的字面意思。ROLLBACK 后旧版本被清理逻辑回收；COMMIT 后其他连接按快照规则只见新版本。再验证第 3 节的判定规则——开一个 RR 事务不提交，另一个连接反复 UPDATE 同一行：

```sql
-- 会话1: BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
--        SELECT v FROM t WHERE id=1;   -- 反复执行, 结果永远是启动时的值
-- 会话2: 每次执行 UPDATE t SET v=v+1 WHERE id=1; COMMIT;  -- 连续改 100 次
-- 会话1再查: 还是旧值 —— 堆里已堆了 100 个死版本, 全靠 VACUUM 清理(下一节)
```

经验做法：**调试 MVCC 问题时 `SELECT ctid, xmin, xmax, ...` 是第一工具**——死版本堆积（表虚胖）、长事务冻结旧版本，都能从这里直接看出来。

## 5. 锁：MVCC 时代锁还在干什么

MVCC 消灭的是**读写**冲突，**写写**冲突与表级 DDL 冲突仍靠锁。分层认识（PostgreSQL 文档 13.3 节 / InnoDB 文档 15.7）：

1. **行锁（tuple/record lock）**：UPDATE/DELETE/SELECT FOR UPDATE 给目标行加排他锁，事务结束释放。两个事务抢同一行 = 排队或死锁（检测到环后牺牲一个回滚，应用要有重试）。
2. **SELECT FOR UPDATE / FOR SHARE**：快照读不锁，但"读出来准备改"必须显式上锁——**经典丢失更新场景**（两个会话同时 SELECT bal，各扣 50，后提交覆盖前者）的解法，或改用 `UPDATE acct SET bal = bal - 50 ... WHERE bal >= 50` 的原子自增写法。
3. **间隙锁（gap lock）/ next-key lock（InnoDB）**：锁住索引记录之间的"空隙"，阻止别人往范围里 INSERT——这是 MySQL RR 挡幻读的机制；代价是并发插入在热点范围上大量误伤（死锁高发区）。PostgreSQL 没有 InnoDB 式间隙锁：RR 下靠 SSI（SERIALIZABLE 时）的谓词锁在提交时检测危险结构。
4. **表级锁**：ALTER TABLE 拿 ACCESS EXCLUSIVE（连 SELECT 都排队）——大表加列/加索引要用 `CREATE INDEX CONCURRENTLY`、PG 11+ 加列带常量默认值是元数据操作不重写表。锁等待排查：`pg_locks` 视图 + `pg_stat_activity`（12 章巡检清单）。

一个必须内化的顺序感：**MVCC 让读不阻塞写、写不阻塞读；锁只管写写与模式变更**。两套机制并存不是历史包袱，而是各自覆盖冲突矩阵的一角。

## 6. 长事务的危害：旧版本没人敢收

MVCC 的多版本必须有清道夫。PG 的叫 **VACUUM**（回收死元组），InnoDB 的叫 **purge 线程**（清理 undo）。清道夫有一条铁律：**任何还"活着"的事务的快照，都是别的版本存在的理由**——长事务开始的那一刻，它可见的旧版本就都不能删。

```
时间线:
T0: 长事务 L 开启 (RR 快照: 一切对 L 可见的版本都必须保留)
T1..T100: 其他事务把某行改了 100 次 → 100 个死版本, 全部不能清理
T101: L 终于提交 → 100 个死版本才能被 VACUUM 回收

后果(PG): 表/索引膨胀, 查询要跨过死版本 → 变慢; 事务ID 回卷风险(xid 耗尽, PG 强制停库)
后果(InnoDB): undo log 膨胀 → 回滚段撑爆, RR 长事务读旧链越走越长 → 查询越跑越慢
```

实测感受死版本（PG）：

```sql
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT v FROM t WHERE id=1;          -- 持有快照
-- 另一会话: 连续 UPDATE t SET v=v+1 WHERE id=1  x100, COMMIT
SELECT pg_stat_get_dead_tuples?;     -- 简化示意, 实际用下面这句
SELECT n_dead_tup FROM pg_stat_user_tables WHERE relname='t';   -- ≈ 100
COMMIT;                               -- 长事务结束
VACUUM t;                             -- 现在 n_dead_tup 归零
```

运维口径（经验做法）：**把"事务时长"当 SLO 盯**——应用里禁止"开了事务等 HTTP/等用户输入"；报表/ETL 用只读从库跑 RR 长查询；PG 上对 `pg_stat_activity` 里 `state='idle in transaction'` 的连接设告警（12 章）。一句话：MVCC 的账单会寄给**最慢的那个读者**。

## 常见误区

1. **"RR 就没有幻读了"**：PG 的 RR（快照隔离）看不到新行但挡不住写偏斜（两个事务各改对方读过的集合，提交后整体违反业务不变量——分布式 07 章的值班医生例）；MySQL 靠间隙锁基本挡住但引入死锁面。要真串行化：PG 用 SERIALIZABLE（SSI）+ 应用重试 40001。
2. **"MVCC = 不需要锁了"**：读确实不锁，写写冲突、唯一约束冲突、DDL 全在锁的辖区。SELECT 后再 UPDATE 的"读改写"序列在默认隔离下照样丢更新——要 FOR UPDATE 或原子 UPDATE。
3. **"隔离级别是会话级设置，随时改随时生效"**：隔离级别**只在事务开始时生效**；autocommit 模式下每条语句都是独立事务，设置看起来"生效"其实每条都在新事务里——跨语句一致视图必须显式 BEGIN。
4. **"长事务只是'占着连接'"**：它冻结了所有旧版本的回收资格（VACUUM/undo purge 全被它扣住），一个忘了提交的运维窗口能把整库拖到膨胀甚至 xid 回卷停机。
5. **"死锁是数据库 bug"**：写写并发下的死锁是正常事件，数据库的合同是"检测并牺牲一个，报错给你"——应用必须重试（捕获 40P01/1213 错误码）。没有重试逻辑的转账代码等于随机失败。

## 本篇小结

- 隔离级别是"异常 × 代价"的菜单；实现视角：RC=每语句快照，RR=每事务快照，SERIALIZABLE=SSI 或严格 2PL。
- 同一实验序列在 RC/RR 下跑出不同结果：快照的生成频率决定不可重复读是否出现；隔离级别在事务开始时生效。
- MVCC = 多版本 + 快照判定（xmin/xmax/版本链）：快照读零锁，读写互不阻塞；但写写冲突与唯一性仍归锁管，SELECT-then-UPDATE 仍会丢更新。
- PG 把新版本直接放堆里（ctid/xmin/xmax 可查），InnoDB 挂 undo 链；PG 靠 VACUUM 清死版本，InnoDB 靠 purge 线程。
- 长事务是 MVCC 的账单寄送日：旧版本无人敢收 → 膨胀/回滚段撑爆/xid 回卷；把事务时长当 SLO 盯。
- 下一章看"数据库怎么执行你的 SQL"——优化器与执行计划，索引与统计信息如何被消费。

## 练习

1. 跑通 anomaly_demo.py，把 T2 改成 `INSERT` 一条新行，T1 两次 `SELECT COUNT(*)`——验证 RC 下计数变大（幻读），RR 下不变；再给 T1 加一条 `UPDATE acct SET bal=bal WHERE id=1`（锁住已见行）后重试，体会"快照读 vs 当前读"。
2. 在 PG 里复现 4 节 xmin/xmax 实验：ROLLBACK 一半 UPDATE、COMMIT 另一半，观察 ctid 与 xmax 变化；然后 `VACUUM (VERBOSE) t`，记录死元组回收数字。
3. 写一个"丢失更新"复现脚本：两个连接 SELECT bal → 各自 -50 → UPDATE 覆盖；先用默认隔离验证余额丢失 50，再用 `SELECT ... FOR UPDATE` 修复；最后改成单条原子 UPDATE，比较三者的最终余额与耗时。
4. 制造一个 `idle in transaction` 连接（BEGIN 后不提交 sleep 60s），另一会话连续 UPDATE 热点行，观察 `pg_stat_user_tables.n_dead_tup` 只涨不清；COMMIT 后跑 VACUUM 再观察——给"长事务冻结回收"留下自己的实测数字。
5. 对照阅读：把分布式系统 07 章的 2PL/SSI 小节与本章 3/5 节并排，列出"同一个问题（写偏斜）在 MySQL 间隙锁、PG SSI、应用层重试三种方案下的成本"，用一张三列表格回答。
