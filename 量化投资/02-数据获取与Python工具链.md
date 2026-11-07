# 02 - 数据获取与 Python 工具链

## 本篇目标

1. 理解金融数据的特殊性：为什么它是"脏数据"重灾区；
2. 掌握 pandas 处理时间序列行情的核心操作；
3. 用 akshare 获取 A 股数据，并知道免费数据源的边界。

## 1. 金融数据的四宗"脏"

量化策略的质量上限由数据决定（garbage in, garbage out）。新手常以为难在策略，实际**难在数据**。四类必须处理的问题：

1. **复权问题（最重要的坑）**：分红送股会造成价格"跳崖"，直接用未复权价格算收益率全是假象。例：10 送 10 后股价减半，账面跌 50%，股东实际没亏。因此算收益必须用**后复权**或**前复权**价格：
   - **后复权**：以上市首日为基准向后调整，价格反映真实累计回报（绝对水平大）。**回测计算收益率应使用后复权**；
   - **前复权**：以当前日为基准向前调整，价格形态贴近行情软件显示。注意：前复权价随时间变化（每次除权后历史价都会重算），**存档的前复权数据会失效**——这是前视偏差的一个隐蔽来源（05 篇）；
2. **幸存者偏差**：只拉"当前上市的股票"列表，就漏掉了已退市的公司——用这种数据回测选股策略，收益系统性虚高（退市的差股票被排除了），05 篇详述；
3. **缺失与停牌**：停牌日无行情；数据源可能给 NaN、0 或填充值，策略要显式处理；
4. **API 的不稳定性**：免费数据源的接口字段、频率限制随时会变——**代码要把"数据获取"与"策略逻辑"分层**，数据源变更只改一层。

## 2. pandas 行情时间序列核心操作

先造一段演示数据，统一后续示例：

```python
import numpy as np
import pandas as pd

# 演示数据：100 个交易日的模拟价格（真实数据获取见第 4 节）
rng = np.random.default_rng(42)
dates = pd.bdate_range("2025-01-02", periods=100)          # 工作日索引
price = pd.Series(
    100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 100))),
    index=dates, name="close")
```

五个高频操作：

```python
# ① 收益率（03 篇展开为何用对数收益）
ret = np.log(price / price.shift(1))          # 对数收益率
ret = price.pct_change().apply(np.log1p)      # 等价写法

# ② 滚动统计（技术指标的基础件）
ma20 = price.rolling(20).mean()               # 20 日均线
vol20 = ret.rolling(20).std() * np.sqrt(252)  # 年化波动率（252 个交易日）

# ③ 时区对齐与缺失处理（多资产必做）
df = pd.DataFrame({"A": price, "B": other_price})   # A/B 停牌日不同
df = df.dropna()                              # 只保留共同交易日（或 ffill 按需）

# ④ 重采样：日 → 周/月
weekly = price.resample("W-FRI").last()       # 每周五收盘价
monthly = price.resample("ME").last()         # 每月月末（pandas ≥2.2 用 "ME"，旧版本用 "M"）

# ⑤ 警惕的一个陷阱：rolling 的窗口含当天
signal = price > price.rolling(20).mean()     # 今天的信号用了今天的收盘价
# 明天才能用今天收盘价算出的信号 → 实盘要 shift(1)，05 篇前视偏差的核心案例
tradable_signal = signal.shift(1)             # 检查每一个信号是否都错位了一天
```

操作 ⑤ 值得贴在显示器上：**"用今天的收盘价决定今天收盘时买入"是最常见的回测作弊**。写完任何信号代码，问一句"这个值在交易时刻真的可得吗？"

## 3. akshare 入门

[akshare](https://github.com/akfamily/akshare) 是开源 Python 财经数据接口库，聚合公开数据源（东财、新浪等），免费无需注册。**接口可能随上游变化，以下示例以 akshare 官方文档为准**（版本差异可能导致函数名不同）。

```python
# pip install akshare --upgrade
import akshare as ak

# A 股日线（前复权），代码 600519（示例标的，仅作数据演示）
df = ak.stock_zh_a_hist(symbol="600519", period="daily",
                        start_date="20200101", end_date="20251231",
                        adjust="qfq")          # qfq=前复权, hfq=后复权, ""=不复权
df.index = pd.to_datetime(df["日期"])          # 列名以实际返回为准

# 指数日线（示例：沪深300）
idx = ak.index_zh_a_hist(symbol="000300", period="daily",
                         start_date="20200101", end_date="20251231")

# 上市公司列表（配合 05 篇理解幸存者偏差：这只是"当前存活"的列表）
stock_list = ak.stock_info_a_code_name()
```

**免费源的能力边界**（务必清楚）：

| 维度 | 免费源（akshare 等） | 商业源（Wind/聚宽/米筐等） |
| --- | --- | --- |
| 已退市股票 | 基本拿不到 → 幸存者偏差 | 有完整退市库 |
| 财务/股东数据 | 有但口径与更新时点需自查 | 口径规范、有发布时间戳 |
| 稳定性 | 接口随上游变动 | SLA 保障 |
| 适用 | 学习、方法论验证 | 生产研究 |

结论：**用免费数据学习本系列完全够用**；但当你发现"回测收益好得可疑"时，第一嫌疑人是数据（幸存者偏差/复权错误），第二嫌疑人才是策略。

> 其他可选工具速览：`tushare`（需注册积分制）、`yfinance`（美股，Yahoo 数据，注意其分红拆分处理）、`baostock`（A 股免费，含复权因子）。选型原则同上：与策略逻辑分层。

## 4. 数据层代码的组织

最小工程结构，把"脏"关在一层里：

```python
project/
  data/            # 数据获取与清洗（唯一接触 akshare 的地方）
    fetch.py       # fetch_daily(code, start, end) -> DataFrame(复权价)
    cache.py       # 本地缓存，避免重复请求（也尊重数据源限频）
  strategy/        # 策略逻辑：只吃干净的 DataFrame，不知道数据从哪来
  backtest/        # 回测引擎（05/09 篇）
```

```python
# data/fetch.py 示例骨架
import pandas as pd

CACHE_DIR = Path("cache")

def fetch_daily(code: str, start: str, end: str, adjust="hfq") -> pd.DataFrame:
    """获取后复权日线。策略代码永远调用这个函数，不直接调 akshare。"""
    f = CACHE_DIR / f"{code}_{start}_{end}_{adjust}.parquet"
    if f.exists():
        return pd.read_parquet(f)          # 缓存优先：可复现 + 不骚扰数据源
    df = _call_akshare(code, start, end, adjust)
    df = _validate(df)                      # 检查列齐全、无负价、排序、去重
    df.to_parquet(f)
    return df
```

要点：**缓存 + 校验 + 单一入口**。校验做四件事：列齐全、无 NaN 价格、日期唯一递增、价格非负。这 20 行会在你数据源第三次变更接口时救你。

## 自测

1. 后复权和前复权的区别？回测为什么用后复权？
2. 为什么"当前上市列表 + 前复权存档"是双重偏差？
3. `price.rolling(20).mean()` 与 `price.rolling(20).mean().shift(1)` 的区别，哪个可用于实盘信号？
4. 为什么数据获取要独立成层？

## 下一篇

[03 - 收益率与风险度量](03-收益率与风险度量.md)——量化的"语言"：从对数收益率到夏普比率。

## 参考来源

- akshare 官方文档与仓库: https://akshare.akfamily.xyz/ / https://github.com/akfamily/akshare
- pandas 官方文档（rolling/resample/时间序列）: https://pandas.pydata.org/docs/
- 复权机制说明可见任一主流行情软件帮助文档；本仓库 05 篇将给出前复权存档导致前视偏差的具体案例
