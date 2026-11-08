# 02 - SQL基础

## 本篇目标

- 掌握 DDL/DML 的最小必要集合，全程用 Python 自带的 sqlite3 实操。
- 默写 SELECT 的七步逻辑执行顺序，并解释"为什么别名不能在 WHERE 里用"。
- 用文氏图 + 具体结果表掌握四种 JOIN，弄清"LEFT JOIN 后 WHERE 右表 IS NULL"这类惯用法。
- 会用窗口函数（ROW_NUMBER/LAG）做去重和环比——量化数据的日常操作；绕开 NULL 三值逻辑的全部陷阱。

## 1. 实操环境与 DDL/DML

本篇全程用 Python 内置的 `sqlite3`，零安装（PostgreSQL 语法差异随文标注）。先建一张贯穿全篇的表——模拟量化场景的持仓记录：

```bash
$ python3            # 进入 Python 后逐段执行下文
```

```python
import sqlite3
conn = sqlite3.connect(":memory:")   # 内存库，随开随用
cur = conn.cursor()

# ---- DDL：定义结构 ----
cur.execute("""
CREATE TABLE positions (
    id      INTEGER PRIMARY KEY,          -- SQLite: INTEGER PRIMARY KEY 即 rowid 别名
    symbol  TEXT    NOT NULL,
    qty     INTEGER NOT NULL,
    price   REAL,                          -- 金额用 REAL 演示方便, 生产用 NUMERIC(第07章)
    opened  TEXT                            -- 日期, SQLite 无原生 DATE 类型
)""")
# PostgreSQL 等价: id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, symbol TEXT NOT NULL, ...
# MySQL 等价: id INT AUTO_INCREMENT PRIMARY KEY, opened DATE

# ---- DML：增删改 ----
cur.executemany("INSERT INTO positions(symbol, qty, price, opened) VALUES (?,?,?,?)", [
    ("AAPL",  100, 189.5, "2025-01-06"),
    ("MSFT",   50, 415.2, "2025-01-07"),
    ("NVDA",  200, 610.3, "2025-01-08"),
    ("AAPL",   80, 185.1, "2025-01-15"),
    ("TSLA",   60, 242.7, "2025-01-20"),
])
conn.commit()                            # SQLite 手动提交; 每条 DML 都在一个隐式事务里
```

记住三组动词的分工：**DDL**（CREATE/ALTER/DROP）动结构，**DML**（INSERT/UPDATE/DELETE）动数据，**DQL**（SELECT）只读。`?` 是 SQLite/PostgreSQL 的参数占位符（MySQL 用 `%s`）——**永远不要用 f-string 拼 SQL**，那是注入的天堂（07 章有完整演示）。

UPDATE/DELETE 各一条，注意**不带 WHERE 的 UPDATE/DELETE 作用于全表**，生产环境必须先开事务、确认影响行数再提交：

```python
cur.execute("UPDATE positions SET price = price * 1.02 WHERE symbol = 'NVDA'")
print(cur.rowcount)                      # 1 —— 用 rowcount 自查影响面
cur.execute("DELETE FROM positions WHERE qty = 0")   # 清理零持仓
conn.commit()
```

## 2. SELECT 的七步逻辑执行顺序

SQL 写起来是 `SELECT ... FROM ... WHERE ...`，但数据库**逻辑上**按下面的顺序执行（标准 SQL 的语义定义，各家优化器物理上会重排，但结果必须等价于此顺序——DDIA 附录、PostgreSQL 文档 7.2 节均按此描述）：

```
① FROM / JOIN    确定数据源，展开成笛卡尔积再做连接过滤
② WHERE          行级过滤（此时还没有分组、还没有 SELECT 的别名！）
③ GROUP BY       按列折叠成分组
④ HAVING         组级过滤（针对聚合结果）
⑤ SELECT         计算输出列（表达式、别名在这里才诞生）
⑥ ORDER BY       排序（可以使用 ⑤ 刚诞生的别名，因为它在 ⑤ 之后）
⑦ LIMIT/OFFSET   截取窗口
```

这个顺序解释了三个高频困惑：

1. **为什么 WHERE 里不能用 SELECT 的别名**：`SELECT price*qty AS mv ... WHERE mv > 10000` 报错 `no such column: mv`——WHERE 在第 ② 步执行时，别名到第 ⑤ 步才出生。办法：要么在 WHERE 里重写完整表达式，要么用 HAVING（第 ④ 步，仍在 ⑤ 之前，也不行）——**正确做法是重写表达式**，或者用子查询把 ⑤ 提前。
2. **为什么 ORDER BY 可以用别名**：它排在 ⑤ 之后，别名已经存在。
3. **为什么 WHERE 里不能写聚合函数**（`WHERE qty > AVG(qty)` 报错）：聚合在 ③ 才发生，WHERE 是分组前的行级过滤——组还没形成，哪来的 AVG？这种需求写 `HAVING`（见下）。

```python
# 验证: 别名在 WHERE 中不可用
try:
    cur.execute("SELECT symbol, qty*price AS mv FROM positions WHERE mv > 10000")
except sqlite3.OperationalError as e:
    print("报错:", e)                     # no such column: mv
# 正确写法一: 重写表达式
print(cur.execute("SELECT symbol, qty*price AS mv FROM positions WHERE qty*price > 10000").fetchall())
# 正确写法二: 子查询把 SELECT 提前
print(cur.execute("SELECT * FROM (SELECT symbol, qty*price AS mv FROM positions) WHERE mv > 10000").fetchall())
```

实测输出：

```
报错: no such column: mv
[('NVDA', 124500.0), ('AAPL', 18950.0), ('AAPL', 14808.0), ('MSFT', 20760.0)]
[('NVDA', 124500.0), ('AAPL', 18950.0), ('AAPL', 14808.0), ('MSFT', 20760.0)]
```

补充两个执行顺序的推论：LIMIT 之前必须先 ORDER BY，否则"前 5 条"是未定义的任意 5 条（SQLite 常见按 rowid 返回，不可依赖）；GROUP BY 之后 SELECT 里只能出现"分组列"和"聚合函数"，其他列取哪一行的值是未定义行为（SQLite 宽容地随便取，PostgreSQL 直接报错——严格是优点）。

## 3. JOIN 四种：文氏图与具体结果

两张演示表：持仓表 `positions`（上面建的）和价格表 `quotes`（**故意**让 TSLA 没有行情、BNCE 行情没有持仓，制造两边不齐的行）：

```python
cur.execute("CREATE TABLE quotes(symbol TEXT PRIMARY KEY, close REAL)")
cur.executemany("INSERT INTO quotes VALUES (?,?)",
                [("AAPL", 190.2), ("MSFT", 420.1), ("NVDA", 620.0), ("BNCE", 15.8)])
conn.commit()
```

```
INNER JOIN: 交集                  LEFT JOIN: 左表全保留，右表缺的补 NULL
   positions   quotes                positions   quotes
      ┌────┬────┐                      ┌────┬────┐
      │ ██ │ ██ │                      │ ██ │ ██ │
      │ ██ │ ██ │                      │ ───┘    │
      └────┴────┘                      │  (NULL) │
      只有 AAPL/MSFT/NVDA               └─────────┘
      三家有行情                        TSLA 也保留, close=NULL
FULL OUTER JOIN: 并集(两边都保留)   CROSS JOIN: 笛卡尔积(每行×每行)
   SQLite 不支持 FULL OUTER, 用       4 持仓 × 4 行情 = 16 行
   LEFT JOIN ... UNION ... RIGHT JOIN 没有连接条件, 纯组合
   (PostgreSQL 原生支持 FULL OUTER)
```

```python
q = lambda s: cur.execute(s).fetchall()

print("INNER:", q("SELECT p.symbol, p.qty, qt.close FROM positions p JOIN quotes qt USING(symbol)"))
print("LEFT:  ", q("SELECT p.symbol, qt.close FROM positions p LEFT JOIN quotes qt USING(symbol)"))
print("左反连:", q("SELECT p.symbol FROM positions p LEFT JOIN quotes qt USING(symbol) WHERE qt.symbol IS NULL"))
print("CROSS:", q("SELECT count(*) FROM positions CROSS JOIN quotes"))
```

实测输出（左反连找出"有持仓没行情"的孤儿行——对账场景的经典惯用法）：

```
INNER: [('AAPL', 100, 190.2), ('MSFT', 50, 420.1), ('NVDA', 200, 620.0)]
LEFT:   [('AAPL', 190.2), ('MSFT', 420.1), ('NVDA', 620.0), ('TSLA', None)]
左反连: [('TSLA',)]
CROSS: [(16,)]
```

三个工程要点（经验做法）：

1. **LEFT JOIN 后把右表条件写进 WHERE，LEFT 就退化成 INNER**：`LEFT JOIN quotes qt ON ... WHERE qt.close > 100` 会把 close=NULL 的行过滤掉。右表条件要写进 **ON** 里，或显式容忍 NULL（`WHERE qt.close > 100 OR qt.close IS NULL`）。
2. **JOIN 键要类型一致**：`symbol TEXT` 连 `symbol INTEGER`（隐式转换）会让索引失效（05 章），也悄悄改变结果。
3. **USING(col) 与 ON a.col=b.col**：USING 更短且输出列去重；但 PostgreSQL 下 USING 后同一列不能加表前缀，混用时注意。

## 4. 聚合与窗口函数：从 GROUP BY 到 ROW_NUMBER/LAG

普通聚合（GROUP BY）把多行**折叠成一行**；窗口函数在每行旁边**追加一列**聚合值，行数不变——这是两者最本质的区别（PostgreSQL 文档 4.2 节 / SQLite 窗口函数文档，3.25+ 支持）：

```python
# 聚合: 每个标的一行
print(q("SELECT symbol, COUNT(*) AS n, SUM(qty) AS total_qty FROM positions GROUP BY symbol HAVING COUNT(*) >= 1"))
# HAVING 过滤组, WHERE 过滤行; 实际输出: [('AAPL', 2, 180), ('MSFT', 1, 50), ('NVDA', 1, 200)]

# 窗口: 每个标的内部按时间编号, ROW_NUMBER 找"每个 symbol 最新的一条"
print(q("""
SELECT symbol, opened, price FROM (
    SELECT symbol, opened, price,
           ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY opened DESC) AS rn
    FROM positions
) WHERE rn = 1"""))
```

```
[('AAPL', '2025-01-15', 185.1), ('MSFT', '2025-01-07', 415.2), ('NVDA', '2025-01-08', 610.3)]
```

`ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY opened DESC)` 读作：按 symbol 分窗，窗内按时间倒序编号——"每个组取最新一条"的标准解法（老式 SQL 要用关联子查询，繁琐且慢）。同类窗口函数还有 `RANK()`（并列跳号）、`DENSE_RANK()`（并列不跳号）。

**LAG** 取"窗口内上一行的值"，环比/涨跌幅的标配（量化 04 篇的均线、收益率都建立在它之上）：

```python
cur.execute("CREATE TABLE px(symbol TEXT, day TEXT, close REAL)")
cur.executemany("INSERT INTO px VALUES (?,?,?)", [
    ("AAPL","2025-01-06",189.5), ("AAPL","2025-01-07",191.2),
    ("AAPL","2025-01-08",188.9), ("AAPL","2025-01-09",193.0)])
print(q("""
SELECT day, close,
       LAG(close) OVER (ORDER BY day)                          AS prev,
       ROUND( (close / LAG(close) OVER (ORDER BY day) - 1) * 100, 2) AS pct
FROM px WHERE symbol='AAPL'"""))
```

```
[('2025-01-06', 189.5, None, None),
 ('2025-01-07', 191.2, 189.5, 0.9),
 ('2025-01-08', 188.9, 191.2, -1.2),
 ('2025-01-09', 193.0, 188.9, 2.17)]
```

第一行的 LAG 是 NULL——**这是数据不是错误**，收益率序列天然比价格序列短一天，下游代码要处理这个 NULL（正是下一节的主题）。

## 5. NULL 的三值逻辑陷阱

SQL 的世界是三值逻辑：TRUE / FALSE / **UNKNOWN**。任何与 NULL 的比较（`NULL = NULL`、`NULL != 5`、甚至 `NULL != NULL`）结果都是 UNKNOWN，而 WHERE 只保留 TRUE 的行。四个必踩的坑，全部用实测钉死：

```python
t = sqlite3.connect(":memory:"); tc = t.cursor()
tc.execute("CREATE TABLE s(x INTEGER)")
tc.executemany("INSERT INTO s VALUES (?)", [(1,), (2,), (None,)]); t.commit()

print(tc.execute("SELECT * FROM s WHERE x != 1").fetchall())        # [(2,)] —— NULL 行消失了!
print(tc.execute("SELECT * FROM s WHERE x = NULL").fetchall())      # []     —— 永远空!
print(tc.execute("SELECT * FROM s WHERE x IS NULL").fetchall())     # [(None,)] —— 唯一正确姿势
print(tc.execute("SELECT COUNT(*), COUNT(x) FROM s").fetchone())    # (3, 2)  —— COUNT(*) 数行, COUNT(col) 跳过 NULL
```

推论清单（经验做法，逐条都是真实事故的结晶）：

1. 判空只用 `IS NULL / IS NOT NULL`；`WHERE x = NULL` 永远返回空集且不报错——静默的逻辑错误最难查。
2. `NOT IN` 遇上子查询里的 NULL 会**全表变空**：`x NOT IN (1, NULL)` 对任何 x 都是 UNKNOWN。用 `NOT EXISTS` 或确保子查询列 NOT NULL。
3. 聚合函数忽略 NULL（AVG/COUNT(col)），但 `COUNT(*)` 不忽略——"平均分"统计里缺席者要不要算分母，必须显式决定。
4. 排序时 NULL 的位置各库不同：PostgreSQL 默认 NULL 最大（ASC 时排最后），SQLite 把 NULL 当最小（ASC 时排最前）；用 `NULLS FIRST/LAST` 显式声明。
5. 建表时能 NOT NULL 的列就 NOT NULL（默认值用 `DEFAULT`）——让"必须有值"成为数据库的保证而不是应用的自觉。UNIQUE 约束下多个 NULL 不冲突（NULL 彼此"不相等"），要禁就用 PostgreSQL 的 `UNIQUE NULLS NOT DISTINCT`（PG 15+）。

## 常见误区

1. **"SELECT * 没关系"**：结果列顺序漂移（表结构变更后按序号取值的代码全崩）、无法走覆盖索引（03 章）、传输无用列。写明列名是自查"我真的需要这些列吗"的机会。
2. **"LIMIT 能让查询变快"**：无 ORDER BY 的 LIMIT 只是"任意几条"；有 ORDER BY 但排序列无索引时，数据库仍要先算完整个结果集再排序截断——LIMIT 省传输不省计算。
3. **"JOIN 就是慢，应该拆成循环查询"**：恰恰相反，一次 JOIN 通常远快于 N 次往返（网络/进程开销 dominating）。该问的是"JOIN 键有没有索引、类型对不对"，而不是"要不要 JOIN"。
4. **"窗口函数是 PostgreSQL 的高级玩具"**：SQLite 3.25+（2018 年）与 MySQL 8+ 都支持。去重、环比、Top-N-per-group 都是它的本职，写在应用层循环里才是真玩具。
5. **"NULL 就是 0 或空串"**：三者语义完全不同（缺席/数值零/空字符串）。建模时先想清楚每列三态的语义，再决定 NULL/DEFAULT/NOT NULL——这是建模问题不是语法问题（11 章）。

## 本篇小结

- DDL/DML/DQL 分工明确；参数占位符（`?`）是底线习惯，拼字符串必被注入。
- SELECT 七步顺序（FROM→WHERE→GROUP BY→HAVING→SELECT→ORDER BY→LIMIT）一个顺序解释三个困惑：WHERE 没有别名、WHERE 没有聚合、ORDER BY 有别名。
- 四种 JOIN：INNER 交集、LEFT 保左补 NULL、左反连找孤儿（`WHERE 右表 IS NULL`）、CROSS 笛卡尔积；右表条件进 WHERE 会把 LEFT 退化成 INNER。
- 窗口函数 = "行数不变的聚合"：ROW_NUMBER 解"每组最新"，LAG 解"环比"；聚合折叠行、窗口不折叠。
- NULL 三值逻辑：与 NULL 的任何比较都是 UNKNOWN；`IS NULL` 是唯一判空姿势；`NOT IN (..., NULL)` 全空；COUNT(*) 与 COUNT(col) 语义不同。

## 练习

1. 不看 2 节，默写七步执行顺序，然后向自己解释三个问题：HAVING 能用 SELECT 别名吗？（提示：HAVING 在 ⑤ 之前）GROUP BY 之后的 SELECT 出现非分组列会怎样？为什么数据库"逻辑上"必须按此顺序？
2. 在 positions/quotes 两表上加一张 trades 表（symbol, day, qty, price），写查询：每只股票最近三笔成交及其金额（ROW_NUMBER 或窗口 LIMIT 子句），并让没有成交的持仓也出现在结果里（LEFT JOIN 的位置放哪一层？）。
3. 用 LAG 写"连续两天上涨"的标的筛选（给 px 表多插几只股票的数据）；再用 `NOT IN` 子查询故意构造一个全空的错误结果，改成 `NOT EXISTS` 修复。
4. 把 2 节的 `:memory:` 换成文件库（`sqlite3.connect("/tmp/pos.db")`），用 `EXPLAIN QUERY PLAN` 看每个查询是 SCAN 还是 SEARCH（03 章的主角提前登场），记录哪些查询全表扫描。
5. 把本篇所有 SQL 原样放到 PostgreSQL 里跑（`psql` 或 Python + psycopg），记录三处差异（提示：ROUND 的参数类型、日期比较、NULL 排序）——方言意识从第一天建立。
