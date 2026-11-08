# 07 - PostgreSQL实战

## 本篇目标

- 完成 PG 16 安装与 psql 基础操作，建立日常命令肌肉记忆。
- 会选数据类型：text vs varchar、numeric vs float（钱的教训）、jsonb 的正确用法。
- 理解主键策略：自增 vs UUID 的索引分裂问题，学会实测思路。
- 掌握分区表、连接池（为什么需要）、备份三件套与 Python 接入（psycopg3 + SQL 注入防御）。

## 1. 安装与 psql 基础

Ubuntu/WSL2 上装 PostgreSQL 16（apt 默认源版本可能较旧，走 PGDG 官方源）：

```bash
sudo apt install -y postgresql                  # 基础安装(版本随发行版)
sudo systemctl start postgresql                 # WSL2 也可用 service postgresql start
sudo -u postgres psql                           # 首次以 postgres 用户进入
# -- 建自己的工作库与账号
CREATE ROLE yyl LOGIN PASSWORD 'yourpw';
CREATE DATABASE dev OWNER yyl;
\q
psql "postgresql://yyl@localhost:5432/dev"      # 之后这样进
```

psql 元命令速查（斜杠命令不是 SQL，不加分号）：

```
\l          列出数据库        \c dbname    切换数据库
\dt         列出表            \d 表名      表结构+索引+约束(最常用)
\x          切换展开显示(宽表必备)             \timing     显示每条 SQL 耗时
\copy       客户端导入导出CSV  \i file.sql  执行脚本
SELECT version();             当前版本; SELECT pg_backend_pid();  当前连接的进程号
```

一个必须先建立的模型：**PG = 一个 postgres 服务进程 + 每连接一个后端进程（fork）**（MySQL 是每连接一线程）。这决定了 PG 的内存与调度行为，也直接引出 6 节"为什么需要连接池"。

## 2. 数据类型选择：钱与 JSONB

选类型是建模的第一层决定，错了后期极难改。三条主线：

**text vs varchar(n)**：PG 里 `text` 与 `varchar(n)` 性能完全相同（都变长存储），区别只在 `varchar(n)` 的长度**约束**会报错。经验做法：**内部数据用 text + CHECK 约束（可随时放宽）**；对接外部系统需要长度契约时才 varchar(n)。Oracle 背景的 `varchar2(50)` 直觉（不指定长度会报错）在 PG 不成立。

**numeric vs float——钱的教训**：float 是 IEEE 754 二进制浮点，**不能精确表示 0.1**，累加后出现精度漂移：

```sql
SELECT 0.1::float8 * 3;            -- 0.30000000000000004  ← 钱变成假账
SELECT 0.1::numeric * 3;           -- 0.3                  ← 十进制精确
SELECT sum(x) FROM (VALUES (0.1::float8),(0.2),(0.3)) t(x);   -- 0.6000000000000001
SELECT sum(x) FROM (VALUES (0.1::numeric),(0.2),(0.3)) t(x);  -- 0.6
```

规则只有一条（经验做法）：**凡是钱、股数、手续费，一律 `NUMERIC(p,s)`**（如 NUMERIC(18,6) 容纳加密货币小数位）；float 留给统计近似值（收益率均值、模型特征）。两者互转的隐式截断在报表对账时爆发，返工成本极高。SQLite 无此问题（内部统一 REAL，但同样不该存钱）；MySQL 用 DECIMAL。

**jsonb 用法**：PG 的 jsonb 是二进制解析、可索引的 JSON。典型场景——外部 API 的原始返回先整包落库（溯源），稳定字段再提成列：

```sql
CREATE TABLE api_raw(id BIGINT GENERATED ALWAYS AS IDENTITY, src TEXT,
                     payload JSONB, pulled_at TIMESTAMPTZ DEFAULT now());
INSERT INTO api_raw(src, payload) VALUES ('tencent',
  '{"symbol":"SH600519","price":1688.0,"tags":["white","liquor"]}');
-- 查询: ->> 返回text, -> 返回jsonb; ? 检查键存在
SELECT payload->>'symbol', payload->'tags'->>0 FROM api_raw WHERE src='tencent';
CREATE INDEX ON api_raw USING gin(payload);      -- GIN 索引支持包含查询
SELECT count(*) FROM api_raw WHERE payload ? 'tags';
```

经验做法：**jsonb 存"结构未定/纯透传"的数据，一旦某字段开始参与 WHERE/JOIN，就提为独立列**（可加表达式索引过渡：`CREATE INDEX ON api_raw ((payload->>'symbol'))`）。jsonb 不是"不想建模"的借口——是过渡带。

## 3. 序列与主键策略：自增 vs UUID

03 章埋的问题现在收口。两种主键方案：

```sql
-- 方案A: 身份列(PG 10+ 推荐, 替代已过时的 SERIAL)
CREATE TABLE orders_a(id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, ...);
-- 方案B: UUID
CREATE TABLE orders_b(id UUID PRIMARY KEY DEFAULT gen_random_uuid(), ...);
```

对比维度：

| 维度 | 自增 BIGINT | UUID v4（随机） |
| --- | --- | --- |
| 索引体积 | 8B 键，树矮 | 16B 键，每棵二级索引都变胖 |
| 插入局部性 | 恒定递增 → 永远追加最右叶，缓存友好 | 随机落点 → 频繁页分裂、缓存命中率低 |
| 全局唯一 | 否（分库需号段/雪花） | 是 |
| 信息泄露 | 可被猜出订单量 | 不可 |

**索引分裂实测思路**（在 03 章分裂机理上展开）：建两张表分别用两种主键，各插 100 万行随机外键数据，对比两个数字——插入耗时（随机 UUID 的页分裂显著更多）与索引大小（`pg_relation_size('idx')`，UUID 版通常大 30-50%）。随机写放大随树高增长，数据量越大差距越明显（3 层变 4 层是拐点）。缓解办法（经验做法）：**必须 UUID 就用 UUIDv7（时间有序，PG 18 原生）/ 雪花 ID**——保住全局唯一同时恢复插入局部性；单库无分片需求就 BIGINT，别给自己找分布式问题。SQLite 下同理（rowid 天然自增）。

## 4. 分区表：一张大表按规则拆

单表过亿行后（行数只是信号，真正的判据是"维护动作"：索引重建/删除旧数据/VACUUM 都开始小时级），把表按规则拆成物理子表，查询自动路由：

```sql
-- 范围分区: 行情表按月分区(11 章会完整设计, 这里演示语法骨架)
CREATE TABLE kline (
    symbol TEXT NOT NULL,
    ts     TIMESTAMPTZ NOT NULL,
    o NUMERIC, h NUMERIC, l NUMERIC, c NUMERIC, v NUMERIC,
    PRIMARY KEY (symbol, ts)                    -- 分区键必须进主键
) PARTITION BY RANGE (ts);
CREATE TABLE kline_2025_06 PARTITION OF kline
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');
CREATE TABLE kline_2025_07 PARTITION OF kline
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');
INSERT INTO kline VALUES ('SH600519', '2025-06-05 09:31+08', 1688,1690,1685,1689, 12000);
EXPLAIN SELECT * FROM kline WHERE ts >= '2025-06-01' AND ts < '2025-07-01';
-- 计划只见 kline_2025_06 一个分区 → 分区裁剪(pruning)生效; 跨月查询自动扫描相关分区
```

收益三条：**查询裁剪**（只扫命中的分区，配合 05 章执行计划）；**运维解耦**（删旧数据 = `DROP TABLE kline_2023_01`，秒级且不产生 vacuum 压力，对比 DELETE 千万行的痛苦）；**索引变小**（每分区独立局部索引）。分区类型：RANGE（时间/数值序列，最常用）、LIST（地区/状态等离散值）、HASH（打散无明显边界的数据）。经验做法：**先用时间 RANGE 按月，每分区 1000 万-1 亿行为宜；分区键必须出现在主键/唯一约束里**——这个约束决定了设计分区键前要先想清楚查询形态。

## 5. 连接池：为什么需要

回到 1 节的进程模型：**每个 PG 连接 = 一个后端进程**（~5-10MB 基础内存 + 各类缓存），新建连接要 fork + 初始化，数十毫秒级。应用侧的常见灾难：Web 框架每个请求开一个连接 → 100 QPS 就是 100 个进程，内存爆 + 上下文切换剧增；PG 默认 `max_connections=100`，超了直接拒绝。解法不是调大 max_connections（每个连接都有内存代价，上下文切换是全局税），而是**复用**：

```
无池:  请求→fork后端进程→执行→退出     (每请求数十ms额外开销 + 进程数随QPS爆炸)
有池:  请求→从池里借连接→执行→归还     (连接进程常驻, 数量=池大小, 与QPS解耦)

应用内池(psycopg_pool): 单进程内复用, 适合单体应用
pgbouncer(外部池):      独立中间件, 多个应用进程/多台机器共享一个小连接池
  - session 模式: 连接绑定整个会话(用到事务外的会话特性时)
  - transaction 模式: 事务结束即归还(默认推荐, 吞吐最高; 会话级 SET/临时表不可用!)
```

经验做法：**单进程应用先用 `psycopg_pool.ConnectionPool(min_size=5, max_size=20)`；多进程部署直接上 pgbouncer（transaction 模式）**。给个参照系：一台 16GB 的库机，max_connections 调到 300 就是自找 swap——连接数是**稀缺资源**，池化后应用侧并发 1000 也只需要几十个连接。

## 6. 备份：pg_dump 与 PITR 概念

备份分两个等级（12 章做恢复演练，这里先建立工具坐标）：

- **逻辑备份 `pg_dump`**：导出 SQL/目录归档，单库粒度，跨版本恢复友好，适合中小库与日常例行：

```bash
pg_dump "postgresql://yyl@localhost/dev" -Fc -f dev.dump    # -Fc 自定义压缩格式
pg_restore -d newdb dev.dump                                 # 恢复(可 -j 4 并行)
```

- **物理备份 + PITR（Point-In-Time Recovery）**：基于 WAL 归档，能把库**恢复到任意时间点**（"删库发生在 14:32:17，恢复到 14:32:16"），全库粒度，生产标配。概念模型：基础备份（pg_basebackup）+ 持续归档的 WAL 段文件（`archive_mode=on` + `archive_command`），恢复 = 基础备份 + 重放 WAL 到目标时刻（`recovery_target_time`）。成本：需要备份存储空间与归档盘；收益：RPO（可接受的数据丢失量）可压到秒级。
- 经验做法：**逻辑备份兜底 + 物理备份/PITR 上生产的双轨**；加密与异地存放是底线；备份的有效性 = 12 章的恢复演练说了算（光备份不恢复 = 没备份）。

## 7. Python 接入：psycopg3 与 SQL 注入防御

安装与最小连接（psycopg 3.x，注意不是老的 psycopg2）：

```bash
pip install "psycopg[binary]"      # 版本 3.x
```

```python
# pg_demo.py —— 参数化查询是唯一正确的姿势
import psycopg

with psycopg.connect("postgresql://yyl@localhost:5432/dev") as conn:
    with conn.cursor() as cur:
        # 正确: 参数化(占位符 %s, 参数元组) —— 值永远不被解析为 SQL 代码
        cur.execute("SELECT symbol, c FROM kline WHERE symbol = %s AND ts >= %s",
                    ("SH600519", "2025-06-01"))
        rows = cur.fetchall()

        # 批量插入: executemany 慢(逐条), 用 execute_values/copy
        from psycopg import sql
        with conn.transaction():                      # 显式事务块
            cur.executemany(
                "INSERT INTO kline(symbol, ts, o,h,l,c,v) VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (symbol, ts) DO NOTHING",
                [("SH600519", "2025-06-06 09:31+08", 1689,1695,1687,1692, 9800)])
    conn.commit()                                     # with 块退出自动提交/回滚
```

**SQL 注入：一个完整的攻击演示与防御**。注入的本质：字符串拼接让"数据"越权成为"代码"：

```python
# 危险: 用户输入直接拼接
symbol = input("股票代码: ")           # 攻击者输入:  SH600519'; DROP TABLE kline; --
cur.execute(f"SELECT * FROM kline WHERE symbol = '{symbol}'")
# 实际执行: SELECT * FROM kline WHERE symbol = 'SH600519'; DROP TABLE kline; --'
# → 你的表没了。(PG 默认多语句在一个 simple query 里可执行, 变体还有 ' OR '1'='1 拖库全表)
```

```python
# 安全: 参数化 —— 输入永远是"值", 语法树在占位符处已经定型
cur.execute("SELECT * FROM kline WHERE symbol = %s", (symbol,))
```

防御清单（经验做法）：**所有值走占位符；表名/列名不能参数化时用 `psycopg.sql.Identifier()` 组装（白名单校验双保险）；应用账号最小权限（业务账号不给 DROP/DDL）；报错信息不回显 SQL 原文**。注入与 ORM：ORM 的参数化是默认的，但 `raw()`/`extra()` 里拼字符串照样中招——防线在习惯不在工具。

## 常见误区

1. **"varchar(n) 比 text 快"**：PG 里两者同性能，区别只是约束。反过来，用 `varchar(n)` 当"长度校验工具"会在业务扩容时变成 DDL 负担——约束交给 CHECK。
2. **"float 存金额没事，浮点误差很小"**：小误差在 sum/复利/对账里复利式放大，财务系统的一分钱差异能烧掉一整周。NUMERIC 是起点不是优化项。
3. **"UUID 主键更现代，直接换"**：随机 UUID 破坏 B+树插入局部性（03/本篇 3 节），写放大与索引膨胀是实测可复现的。单库 BIGINT，分布式用 UUIDv7/雪花，按需选择。
4. **"max_connections 调大就能抗并发"**：每连接一个进程的模型下，连接数是内存与调度税。正确路径是池化复用，让 20 个连接服务 1000 并发。
5. **"pg_dump 一下就万事大吉"**：pg_dump 恢复的是"最后一次 dump 的时刻"，不能回到任意时间点；且没演练过的备份等于没有。RPO/RTO 定清楚，再选逻辑/物理/PITR 组合。

## 本篇小结

- psql 元命令 + 进程模型（每连接一个进程）是 PG 的第一课；连接是稀缺资源，池化是刚需。
- 类型三决定：内部 text+CHECK、金额 NUMERIC、透传 jsonb（参与查询就提列）。
- 主键策略的本质是插入局部性：自增 BIGINT 追加友好；随机 UUID 页分裂+膨胀，UUIDv7/雪花是兼顾唯一与有序的折中。
- 分区表的三收益（裁剪/运维解耦/小索引）与铁律（分区键进主键，先想查询形态再定分区键）；时间 RANGE 按月是默认起手式。
- 备份双轨：pg_dump 例行 + WAL 归档 PITR；有效性由恢复演练定义（12 章实测）。
- psycopg3：占位符参数化是注入的唯一防线；表名用 sql.Identifier；批量用 executemany/copy；连接用池。

## 练习

1. 建库实操：创建 test 库，建 kline 分区表（按月 3 个分区），用 Python 插入 10 万条随机 OHLCV，验证 `EXPLAIN` 的分区裁剪；再 `DROP` 一个分区观察耗时，与 `DELETE WHERE ts < ...` 对比。
2. 主键实测：按 3 节思路建自增/UUID 两张表各插 100 万行，记录总耗时与 `pg_relation_size` 的差异，写下你机器上的倍数。
3. 注入靶场：建一张 users 表，故意写一个拼接查询的函数，用 `' OR '1'='1` 演示拖库；改参数化后验证输入原样存取（把攻击串存进表再查出来打印，理解"数据就是数据"）。
4. 连接池实验：写脚本同时开 200 个无池连接（psycopg 直连），观察 `SELECT count(*) FROM pg_stat_activity;` 与内存；换 `ConnectionPool(max_size=20)` 达到同样并发，对比两边数字。
5. 备份初体验：`pg_dump -Fc` 导出 test 库，`DROP` 掉 kline，`pg_restore` 恢复并验证行数一致；把 dump 文件 `pg_restore -l` 看目录结构，思考"如果是误删发生在 3 小时前，这份 dump 能帮你回到什么时刻"（引出 PITR 的必要性）。
