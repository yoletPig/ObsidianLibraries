# 05 - Property-Based 测试：从举例子到陈述性质

## 本篇目标

1. 完成思维转变：测试不是"列输入猜输出"，而是"陈述对一切输入都成立的性质"；
2. 掌握 hypothesis 三件套：`@given`、strategy、`settings`，会写自定义 strategy；
3. 会用 `RuleBasedStateMachine` 做状态ful测试，理解 shrink 如何给出最小反例；
4. 能判断哪些代码适合性质测试：解析器、纯算法、序列化往返。

## 1. 从示例测试到性质测试：排序反例

先看"示例思维"的天花板。有个同事自己实现的降序排序：

```python
# buggy.py
def sort_desc(nums):
    """降序排序。"""
    result = []
    pool = list(nums)
    while pool:
        m = max(pool)
        result.append(m)
        pool = [x for x in pool if x != m]   # ❌ 删掉了所有等于 m 的元素
    return result
```

按直觉写三个用例（这也正是大多数人的真实写法）：

```python
def test_sort_basic():
    assert sort_desc([3, 1, 2]) == [3, 2, 1]
def test_sort_empty():
    assert sort_desc([]) == []
def test_sort_sorted():
    assert sort_desc([5, 4, 3]) == [5, 4, 3]
```

三个全绿——但 bug 真实存在，只有**重复元素**才能触发：

```python
>>> sort_desc([2, 2, 1])
[2, 1]        # 丢了一个 2
```

示例测试的本质是"我猜哪几个输入最危险"，猜不中就漏。性质测试换一个提问方式：**不说具体输入，只陈述对任意输入都成立的规律**——"结果必须等于标准答案"，对降序排序，等价标准答案现成就有：

```python
# test_sort_hypo.py
from hypothesis import given, strategies as st
from buggy import sort_desc

@given(st.lists(st.integers()))
def test_sorted_desc(xs):
    out = sort_desc(xs)
    assert out == sorted(xs, reverse=True)   # 性质：与标准库等价
```

跑起来，hypothesis 自动生成上百组随机输入，撞上重复元素后给出**最小反例**：

```
E       assert [0] == [0, 0]
E       Failing test case:
E           xs=[0, 0],
```

注意它给的不是某个撞大运的十万元素大列表，而是 `[0, 0]`——两个元素的最小重现。这就是性质测试的双重收益：**生成**（你没想到的输入）+ **缩小**（可读的反例）。David MacIver（hypothesis 作者）在 HypothesisWorks 博客里的表述：性质测试的价值不是"更多随机测试"，而是"把'什么算对'从个例提升为普适陈述"。

## 2. hypothesis 快速上手：@given / strategy / settings

```python
from hypothesis import given, settings, strategies as st

@settings(max_examples=200, deadline=None)      # settings 从上往下包住 @given
@given(st.integers(min_value=0, max_value=100), st.integers(min_value=0, max_value=100))
def test_add_commutative(a, b):
    assert a + b == b + a
```

**strategy** 是"一类值的生成器"：`st.integers()`、`st.text()`、`st.lists(...)`、`st.one_of(...)`、`st.sampled_from([...])`……可组合出任意形状；**`@given`** 把策略绑定到参数上，pytest 会为每个参数集各跑一个"子用例"；**`settings`** 控制引擎：`max_examples`（生成数量，默认 100）、`deadline`（单例耗时上限，慢依赖可设 `None`）、`suppress_health_check`（见常见误区）。

最有用的一招是**种子固化**：CI 里失败后，输出里会带 `@given(...)` 之外的一行随机种子提示；把反例写进 `@example(...)` 装饰器，它就永远参与回归：

```python
from hypothesis import example, given, strategies as st

@example([0, 0])                      # 历史 bug 永久驻留，等价回归用例
@given(st.lists(st.integers()))
def test_sorted_desc(xs):
    assert sort_desc(xs) == sorted(xs, reverse=True)
```

hypothesis 还自带**数据库**（默认 `.hypothesis/` 目录）：失败过的反例会在后续运行中**优先重放**。这个目录进 `.gitignore`，但 CI 里可以选择缓存它（08 篇）让历史反例持续回归。

## 3. 自定义 strategy：复杂数据结构

内置策略组合不够时，两个工具造复杂结构：

**`st.builds` + 类型**：构造 dataclass / 具名对象，或直接 `st.from_type(User)`（从类型注解反推策略——注意 `str`/`int` 字段生成的是任意 Unicode 与大整数，常需要显式收窄）：

```python
from dataclasses import dataclass
from hypothesis import strategies as st

@dataclass
class User:
    name: str
    age: int

users = st.builds(
    User,
    name=st.text(min_size=1, max_size=20),
    age=st.integers(min_value=0, max_value=150),
)
```

**`@st.composite`**：需要"后一步依赖前一步"的生成逻辑时，`draw` 手动抽取（例：生成"和为 1 的权重列表"——批量归一化测试的常用素材）：

```python
@st.composite
def weights(draw, n_min=2, n_max=6):
    n = draw(st.integers(min_value=n_min, max_value=n_max))
    raw = [draw(st.floats(min_value=0.01, max_value=1)) for _ in range(n)]
    s = sum(raw)
    return [x / s for x in raw]

@given(weights())
def test_weights_sum_to_one(ws):
    assert abs(sum(ws) - 1.0) < 1e-9
```

**经验做法**：写自定义策略时**收窄生成的值域**（`min_size`/`max_value`/字母表黑白名单）——这不是作弊，而是把"测试空间"对齐到"接口的真实契约"（年龄不会是 10^20）；契约之外的行为本来就属于异常路径测试（那是示例测试的活）。

## 4. 状态ful测试：RuleBasedStateMachine

排序类性质针对**无状态纯函数**。有状态对象（栈、缓存、连接池）的性质是"**任意操作序列下，行为与参考模型一致**"。hypothesis 的 `RuleBasedStateMachine` 随机编排 rule 序列，你只需要声明每步操作与不变量。

一个有 bug 的栈（返回栈顶、却删栈底——单看每个方法都"半对"）：

```python
# stack.py
class Stack:
    def __init__(self):
        self._items = []
    def push(self, x):
        self._items.append(x)
    def pop(self):
        v = self._items[-1]     # 返回栈顶
        self._items.pop(0)      # ❌ 却删掉栈底
        return v
    def __len__(self):
        return len(self._items)
```

状态机测试：维护一个 Python list 作**参考模型**，随机 push/pop，逐步对账：

```python
# test_stack_stateful.py
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, precondition, rule
from stack import Stack

class StackMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.stack = Stack()
        self.expected = []              # 参考模型：正确的栈

    @rule(x=st.integers())
    def do_push(self, x):
        self.stack.push(x)
        self.expected.append(x)

    @precondition(lambda self: self.expected)   # 栈非空才允许 pop
    @rule()
    def do_pop(self):
        v = self.stack.pop()
        assert v == self.expected.pop()  # 返回值必须等于"最后压入的"

TestStack = StackMachine.TestCase
```

注意两个语法细节：`@precondition` 必须与 `@rule` 叠用（单独写会直接 `InvalidDefinition` 报错，实测确认），且叠放顺序是 `precondition` 在上、`rule` 在下。运行结果——引擎自动搜出**最短操作序列**：

```
E       assert 1 == 0
E       Falsifying example: state.do_push(x=0), state.do_push(x=1), state.do_pop(), state.do_pop()
```

四个操作复现：`push(0), push(1), pop()` 正常返回 1；第二次 `pop()` 却又返回 1（栈底早被删走了，栈顶还是 1）——与参考模型的 0 对不上。手写用例几乎不可能想到"连 pop 两次"这个序列，状态机把它当成搜索空间的一部分。

适合状态ful测试的对象：LRU 缓存、限流器、连接池、状态机协议（Agent 13 篇评测 Agent 行为轨迹时用的正是同构思想：**参考模型 + 轨迹对账**）。

## 5. shrink 机制：最小反例如何生成

理解 shrink 才能读懂 hypothesis 的失败报告。原理：生成不是随机的噪声，而是**从随机字节流确定性重建的结构**——同一个种子永远生成同一个值。失败后，引擎对这段字节流做局部简化（整数变小、列表变短、分支变浅），每改一步就重跑断言，保持失败的简化被采纳，直到无法再简化。

于是你看到的效果是：明明撞上的是一个 40 元素的 CSV 表格，报告里只剩 `[[',']]`。一个真实的往返 bug 演示——手写 CSV 编解码：

```python
# test_csv_roundtrip.py
from hypothesis import given, strategies as st

def encode(rows: list[list[str]]) -> str:
    return "\n".join(",".join(r) for r in rows)

def decode(s: str) -> list[list[str]]:
    return [line.split(",") for line in s.split("\n")]

cells = st.text(min_size=0, max_size=6)
rows = st.lists(st.lists(cells, min_size=1, max_size=4), min_size=1, max_size=4)

@given(rows)
def test_roundtrip(rs):
    assert decode(encode(rs)) == rs        # 性质：往返必须无损
```

```
E           rs=[[',']],
```

最小反例一行看懂：**单元格内容含逗号**时往返损坏（"`a,b`" 编码后无法区分是两个字段还是一个字段）。修复方向立即明确——按 CSV 规范给含分隔符的字段加引号（或直接用标准库 `csv` 模块）。示例测试要撞上这个组合需要有人**恰好**想到往字段里塞逗号；性质测试把它当成普通搜索空间，还替你把重现缩到最短。

## 6. 什么代码适合性质测试

按投入产出排序：

1. **解析器 / 序列化**：往返性质（`decode(encode(x)) == x`）与**对偶性质**（合法输入必须解析成功 + 非法输入必须报错不崩溃）。JSON/YAML/配置文件/协议编解码都是首要受益者；
2. **纯算法**：与参考实现等价（如上文的 `sorted`）、**不变量性质**（排序后元素集合不变、字典序关系保持）、**幂等性质**（`sort(sort(x)) == sort(x)`、`dedup(dedup(x)) == dedup(x)`）、**交换/结合律**（数学与位运算类代码）；
3. **有状态组件**：参考模型对账（第 4 节），上限投入——模型本身就是半个实现，写得动再写；
4. **不太适合**：强 I/O 胶水（mock 住依赖后可测的核心逻辑才轮到它）、UI、以及"性质本身只能复述实现"的代码——见常见误区。

## 常见误区

**误区一：同义反复性质。** `assert my_reverse(reverse(xs)) == reverse(my_reverse(xs))`——拿被测函数自己验证自己，什么也证不了。性质必须锚定在**独立事实**上：参考实现、数学定律、往返、不变量。写完性质问一句：如果实现换成随机数，这条性质会不会红？

**误区二：CI 里的随机性恐慌。** "每次跑的输入都不一样，红得没有意义？"——相反，失败必然给出**确定性的最小反例**（shrink 的意义就在这）；再用 `@example` 固化、`.hypothesis/` 缓存重放，回归性完全可控。正确的 CI 姿势是给 property 测试单独加 `property` 标记，夜跑加大 `max_examples`（08 篇分层策略）。

**误区三：health check 报错直接 suppress。** 首次运行报 `FailedHealthCheck`（如 data_too_large、too_slow）说明策略生成了超预期规模的数据——先收窄值域，确实需要大对象再 `suppress_health_check=[HealthCheck.too_slow]`，无脑全 suppress 会把"生成太慢"演化成"测试太慢"。

**误区四：把 property 测试当全部。** 它找的是"你想不到的输入"，不擅长验证**具体业务行为**（"下单后余额减 1"这种精确断言示例测试更直接）。两者是互补：示例测"对的行为"，性质测"没有例外"。

**误区五：浮点性质写死相等。** 生成的浮点经运算后 `==` 几乎必然翻车，用 `math.isclose`/`pytest.approx`，或在策略里限定可精确表示的值。

## 本篇小结

- 性质测试把"什么算对"从个例提升为普适陈述：排序反例证明手写三用例的天花板，hypothesis 用 `sorted` 等价性质当场抓出重复元素丢失；
- 三件套：strategy（值域形状）、`@given`（绑定）、`settings`（引擎）；`@example` 固化历史反例，`.hypothesis/` 缓存重放；
- 自定义策略：`st.builds` 构造对象、`@st.composite` 处理依赖式生成，值域收窄对齐接口契约；
- `RuleBasedStateMachine` = 随机操作序列 + 参考模型对账，`@precondition` 必须叠在 `@rule` 上；缓存、限流器、栈这类组件的正规测法；
- shrink 给出最小确定性反例（`[[',']]` 一眼看懂 CSV 引号缺失），理解它才能读懂失败报告；
- 适用面：解析/往返 > 纯算法不变量 > 有状态模型对账；与示例测试互补而非替代。

## 练习

1. 给 1 节的 `sort_desc` 写修复版，并补一条"元素多重集不变"的性质（`Counter(out) == Counter(xs)`），说明这条性质能抓到哪类修复失误。
2. 为 `strip_html_tags(s)`（自定义实现）写性质测试：输出中不出现 `<` 与 `>`、重复调用幂等。先预测它会发现什么再运行。
3. 给 3 节的 `User` 写 `to_json/from_json`，用往返性质测出"名字里带引号/换行"的 bug，并修复（提示：`json.dumps` vs 手拼字符串）。
4. 实现一个 `LRUCache(capacity)`，用 `RuleBasedStateMachine` 对账参考模型（提示：参考模型用 `OrderedDict`），让引擎找出你的淘汰顺序 bug。
5. 把 5 节的 CSV `encode`/`decode` 修复到往返性质通过（自己实现引号转义，不用标准库 csv），再加大 `max_size` 撞一次引号嵌套的坑。
