# 02 - pytest 基础：从断言到日常工作流

## 本篇目标

1. 理解断言重写机制——为什么 pytest 的裸 `assert` 远强于 `unittest` 的 `assertEqual` 家族；
2. 掌握组织结构：文件命名、目录布局、conftest.py 的角色；
3. 熟练使用参数化与 skip/xfail，把"相似用例"写成数据而不是复制粘贴；
4. 建立运行与选择的工作流（`-k`/`-m`），并用 fixture 写出第一版现代 setup/teardown。

## 1. 断言重写：为什么 pytest 的 assert 比 unittest 强

同一个失败，两种报法。先看 unittest 风格（pytest 可以直接跑 unittest 用例，所以两者输出可以直接对比）：

```python
# test_uni.py —— unittest 风格
import unittest

class T(unittest.TestCase):
    def test_eq(self):
        x, y = 2, 3
        self.assertEqual(x * y + 1, 8)
```

失败输出只有一行结论：

```
E       AssertionError: 7 != 8
```

`7` 从哪来的？你得回脑袋里展开 `x * y + 1`。再看 pytest 原生写法：

```python
# test_demo.py —— pytest 原生
def test_assert_rewrite():
    x, y = 2, 3
    assert x * y + 1 == 8
```

```
>       assert x * y + 1 == 8
E       assert ((2 * 3) + 1) == 8
```

pytest 把**中间表达式的求值结果**直接标了出来：`((2 * 3) + 1)`。对字典、列表、长字符串，重写机制还能把"哪一部分不一致"指出来（比如断言两个 dict 相等时，会单独显示差异的键）。

**机制**：这不是运行时魔法。pytest 在**收集（导入）测试模块时做 AST 级别的改写**（assertion rewriting）——把每条 `assert` 语句重写成一系列带钩子的语句，缓存各子表达式的中间值，失败时按原始源码位置拼出可读的diff。官方文档《How to use assertions》的"Assertion rewriting details"一节给出了完整过程；重写发生在导入时、只对**测试模块与 pytest 注册的模块**生效。

由此推出两个实用结论：

1. **自定义的辅助断言函数不会自动享受重写**——如果你把 `assert` 藏进一个普通 helper 模块，失败信息会退化成普通 AssertionError。解决：在模块顶部用 `pytest.register_assert_rewrite("helpers.assertions")` 注册，让 pytest 在导入它之前也做重写；
2. **不要用 `assert (a == b, "msg")` 当"带消息的断言"**——这是 tuple，永远为真。pytest 8 已经会在收集期警告你（实测）：

```python
def test_gotcha():
    assert (2 == 3, "2 和 3 不相等")   # 期望失败，实际通过！
```

```
PytestAssertRewriteWarning: assertion is always true, perhaps remove parentheses?
============== 1 passed, 1 skipped, 1 xfailed, 1 warning in 0.08s ==============
```

带消息的正确姿势是 `assert a == b, f"a={a} b={b}"`（消息在逗号后）——而且重写后的输出通常已经足够好，多数时候根本不需要自定义消息。

## 2. 组织结构：命名、布局与 conftest.py

**命名规范**（均可配置，但先遵守默认再谈定制）：

| 对象 | 默认规则 | 配置项 |
| --- | --- | --- |
| 测试文件 | `test_*.py` 或 `*_test.py` | `python_files` |
| 测试函数 | `test_` 开头 | `python_functions` |
| 测试类 | `Test` 开头、**无** `__init__` | `python_classes` |

**经验做法**：全库统一 `test_*.py` + `test_` 前缀，`*_test.py` 属于历史兼容，混用会让 `-k` 与脚本选择变得混乱。一个常见翻车现场：把测试写成 `def check_xxx():` 或文件叫 `utils_test.py`（同项目其他文件都是 `test_` 开头），pytest 静默不收集——怀疑"测试没跑"时，第一反应是 `pytest --collect-only -q` 看收集到了什么。

**目录布局**（本系列统一约定，src 布局 + tests 镜像）：

```text
mylib/
├── pyproject.toml
├── src/
│   └── mylib/
│       ├── __init__.py
│       ├── parser.py       # 被测代码
│       └── client.py
└── tests/
    ├── conftest.py         # 本层 fixture 与配置，无 __init__.py
    ├── test_parser.py      # 与被测模块镜像命名
    ├── test_client.py
    └── integration/
        ├── conftest.py     # 子目录私有 fixture
        └── test_client_api.py
```

镜像命名的价值：`grep parser` 时测试与实现成对出现；`pytest tests/test_parser.py` 天然是"一个模块的回归命令"。

**conftest.py** 的两条核心规则：

1. pytest 启动时按**目录层级**自动加载它，路径上所有 conftest（根 → 子目录）依次生效，**不需要、也不应该被 import**；
2. 它是 fixture 的家（03 篇的主角），也常放 `pytest_addoption`、`pytest.ini` 级别的钩子。子目录 conftest 里的同名 fixture 会**覆盖**上层——这是分层复用的机制，也是意外的来源，03 篇 6 节细讲。

## 3. 参数化：把相似用例写成数据

三个参数以上还一个函数一个用例地复制粘贴，是测试代码腐化的第一现场。`parametrize` 用数据表替代复制：

```python
# test_param.py
import pytest
from mylib.parser import retry_delay   # 01 篇修好 ms bug 的版本

@pytest.mark.parametrize("raw, seconds", [
    (30, 30),
    ("2.5s", 2.5),
    ("5m", 300),
    ("1h", 3600),
    pytest.param("500ms", 0.5, id="millis"),   # 单个用例指定 id
])
def test_retry_delay(raw, seconds):
    assert retry_delay({"retry_delay": raw}) == seconds
```

```
$ pytest test_param.py -v
test_param.py::test_retry_delay[30-30] PASSED                    [ 20%]
test_param.py::test_retry_delay[2.5s-2.5] PASSED                 [ 40%]
test_param.py::test_retry_delay[5m-300] PASSED                   [ 60%]
test_param.py::test_retry_delay[1h-3600] PASSED                  [ 80%]
test_param.py::test_retry_delay[millis] PASSED                   [100%]
```

四个要点：

1. **自动 id**：pytest 把参数值拼进用例名（`[30-30]`）。浮点、对象会得到难读的名字（`raw3` 之类），可读场景显式传 `ids=` 列表或用 `pytest.param(..., id=)`——id 可读的直接收益是 `-k` 选择（第 5 节）与失败报告；
2. **一组参数装饰器是笛卡尔积**：叠两个 `@parametrize` 得到 3×4=12 条用例，适合"所有输入形态 × 所有处理模式"的组合测试；
3. **单条用例特殊化**：`pytest.param(值, marks=pytest.mark.xfail(...), id=...)` 可以给数据表里的某一格单独打标记（上面 `millis` 格演示过 xfail 的接法）；
4. **失败时数据在报告里**：参数化失败的报告会展开 raw 与 seconds 的实际值（01 篇的输出就是实例），不用再复现输入。

**经验做法**：参数化表里**故意放边界值**（空串、0、负数、超长、Unicode、`None`）——这些正是手工测试永远懒得点的地方；以及"已知 bug 的回归值"：修过的每个 bug 都把它的输入形态补进表里。

## 4. skip 与 xfail：正确表达"暂时不绿"

两者常被混用，语义完全不同：

- **skip**：*这条测试根本不跑*。适用于条件性平台差异、待修复的环境依赖。测试代码照样会被收集，但函数体不执行；
- **xfail**：*这条测试会跑，且我预期它失败*。适用于"已知 bug 未修""功能未实现"——它和"注释掉"的差别是：**一旦突然变绿（bug 被顺手修好），pytest 会报 XPASS 提醒你解除标记**。

```python
import sys
import pytest

@pytest.mark.skip(reason="等待上游接口修复")          # 无条件跳过
def test_skip_plain(): ...

@pytest.mark.skipif(sys.platform != "linux",
                    reason="仅 Linux 验证文件权限语义")  # 条件跳过
def test_skipif_linux(): ...

@pytest.mark.xfail(reason="issue #42：并发下计数可能少一", strict=True)
def test_increment_race():
    ...
```

`strict=True` 让 xfail 变成"双向断言"：失败=XFAIL（不算红），通过=**直接红**（XFAIL strict），防止"已知 bug"悄悄修好了却没人摘标记。官方文档《skipping》建议：对绑定具体 issue 的 xfail 一律加 strict。

**经验做法**：skip 的 reason 里写清楚"复活条件"（哪个 issue 修复后必须解除），CI 里加一条自定义检查：存在超过 30 天的 skip/xfail 就在流水线里打 warning（08 篇的失败纪律配合使用）。skip 和 xfail 都是**有息负债**，要记账。

## 5. 运行与选择：日常命令工作流

```bash
pytest                          # 全量（默认在当前目录收集）
pytest tests/test_parser.py     # 按文件
pytest tests/test_parser.py::test_retry_delay["5m-300"]   # 精确到用例
pytest -k "millis and not slow" # 按用例名表达式选择（对 id 生效）
pytest -m "not slow"            # 按标记选择（需注册标记，见下）
pytest --collect-only -q        # 只收集不执行：检查命名/收集问题
pytest -x --lf -q               # 首败即停 + 只重跑上次失败（--last-failed）
pytest -s                       # 禁用输出捕获，看 print/stdout
pytest -vv --tb=short           # 详细名 + 短回溯
```

`-k` 的匹配对象是**整个用例名（含参数化 id）**——实测 `"30 or millis"` 会同时命中 `[30-30]` 与 `[5m-300]`（因为后者 id 里含 "30"），id 命名时要有这个意识。`-m` 按 mark 选择，前提是在 `pyproject.toml` 里注册（否则 warning）：

```toml
[tool.pytest.ini_options]
markers = ["slow: 超过 1s 的测试", "integration: 需要真实依赖"]
addopts = "-m 'not slow and not integration'"   # 本地默认只跑单元层
```

**经验做法**：`addopts` 里放"日常 90% 场景的默认集"，CI 里显式 `-m ""`（清空过滤）跑全量——本地快反馈、CI 兜底，两层各司其职。

## 6. fixture 入门：setup/teardown 的现代写法

unittest 时代每个类要写 `setUp`/`tearDown`，类一多，"每个测试各建一份昂贵资源"与"共享一份全局资源"之间没有中间态。pytest 的答案是 fixture：

```python
# test_tmpfile.py
import pytest

@pytest.fixture
def log_file(tmp_path):                 # 依赖 pytest 内置 fixture（03 篇细讲）
    path = tmp_path / "app.log"
    path.write_text("", encoding="utf-8")
    yield path                          # yield 前=setup，后=teardown
    # 收尾代码：teardown 阶段执行（此处 tmp_path 自带清理，无需手写）

def test_append(log_file):              # 测试函数按名声明依赖
    log_file.write_text("hello\n")
    assert "hello" in log_file.read_text()
```

对比 unittest 的三处进步：

1. **按需注入**：只把需要的东西作为参数传进来，不需要的测试零成本（unittest 的 setUp 对全类无差别执行）；
2. **yield 分段**：setup/teardown 写在一个函数里，视线不用上下跳；teardown 即使测试失败也会执行；
3. **可组合可分层**：fixture 可以依赖 fixture、可以按目录覆盖（03 篇整章展开）。

## 常见误区

**误区一：测试函数里写 `print` 调试后忘了删。** pytest 默认捕获 stdout，绿了什么也看不见；但失败回溯里会**回放**捕获的输出——所以 print 不算害群之马，可临时保留；要主动看输出用 `-s` 或 `capsys` 断言（03 篇）。

**误区二：测试类加了 `__init__`。** pytest 见到带 `__init__` 的 Test 类会拒绝收集并警告。需要每用例的初始化，用 fixture；需要共享状态，用类属性 + autouse fixture。

**误区三：参数化 id 用默认浮点串。** `[raw2-0.5]` 这类自动 id 在 `-k` 选择与 CI 报告里难以定位，养成显式 id 的习惯。

**误区四：把断言写进 helper 却没注册重写。** 失败信息只剩 `AssertionError`，等于回到 unittest 时代。记得 `pytest.register_assert_rewrite()`。

**误区五：把 `--lf` 当全量回归用。** `--lf`（只重跑上次失败的）语义实测：上次有失败→只跑失败的；上次全绿或缓存被清→**退化为全量**。它是"本地快速反馈"工具，不是回归保障——CI 里不要依赖它（缓存目录在 CI 上不跨任务持久），CI 永远全量。

## 本篇小结

- 断言重写是 pytest 的核心竞争力：导入时 AST 改写让裸 `assert` 输出中间值；helper 里的断言要 `register_assert_rewrite`；`assert (a, msg)` 是永真陷阱；
- 布局：`test_*.py` + `test_` 前缀 + src/tests 镜像；conftest.py 按层级自动加载、专放 fixture 与钩子；
- 参数化用数据表消灭复制粘贴：显式 id、笛卡尔积、单格 marks、边界值进表；
- skip=不跑，xfail=预期失败且 strict 双向；两者都是要记账的负债；
- 命令工作流：`-k`/`-m` 选择、`--collect-only` 排查收集、`-x --lf` 快反馈；本地默认集与 CI 全量分层；
- fixture 的 yield 模式是现代 setup/teardown——下一章把它讲透。

## 练习

1. 把 01 章的 `retry_delay` 参数化表再扩 5 行：大写单位、末尾空格、复合时长 `"1m30s"`、超长字符串、`None`——先猜每个的行为再跑，对照你猜的输出写 100 字。
2. 写一个 helper 模块 `tests/helpers/assertions.py`（内含 `assert_valid_config(cfg)`），分别用"不注册"与 `register_assert_rewrite` 两种方式跑出失败，对比两条失败信息的差异。
3. 用两个 `@parametrize` 叠出笛卡尔积（3 种输入形态 × 4 种分隔符），显式 id 规则自定，然后用 `-k` 精确选出其中 2 条运行。
4. 把项目里某个"注释掉的测试"改成 `xfail(strict=True)`，体验它变 XPASS 时的报错；在 pyproject 里注册 `slow` 标记并用 `-m` 验证过滤。
5. 用 `yield` fixture 写一个"临时环境变量注入 + 恢复"的 setup/teardown（提示：可先手写，03 篇会介绍内置的 `monkeypatch` 更简写法）。
