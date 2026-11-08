# 03 - fixture 与测试架构：把 setup/teardown 写成体系

## 本篇目标

1. 用实验看清四个作用域的真实生命周期，学会按"资源多贵"选作用域；
2. 理解 fixture 组合与依赖注入——为什么这是测试代码的"构造函数"；
3. 掌握三个最高频的内置 fixture：`tmp_path`、`capsys`、`monkeypatch`；
4. 会用 conftest 分层与工厂 fixture 组织一套可维护的测试架构，并识别 autouse 的滥用。

## 1. 作用域：function / class / module / session

`@pytest.fixture(scope=...)` 的四个作用域决定实例**创建一次、活多久**：

| scope | 创建时机 | 销毁时机 | 典型对象 |
| --- | --- | --- | --- |
| `function`（默认） | 每个测试前 | 每个测试后 | 临时文件、内存对象、fake |
| `class` | 每个测试类前 | 类内用例跑完 | 类级共享的可变状态 |
| `module` | 每个模块前 | 模块跑完 | 模块级只读资源 |
| `session` | 整轮测试前 | 进程退出前 | 启动本地服务器、构建产物、数据库容器 |

**执行顺序实验**（完整可运行，把生命周期写进文件、pytest 退出后查看——比 print 直观）：

```python
# test_scope_order.py
import pytest
from pathlib import Path

LOG = Path(__file__).parent / "order.txt"
LOG.write_text("")          # 每次运行前清空

@pytest.fixture(scope="session")
def s_session():
    LOG.write_text(LOG.read_text() + "session: setup\n"); yield
    LOG.write_text(LOG.read_text() + "session: teardown\n")

@pytest.fixture(scope="module")
def s_module():
    LOG.write_text(LOG.read_text() + "  module: setup\n"); yield
    LOG.write_text(LOG.read_text() + "  module: teardown\n")

@pytest.fixture
def s_func():
    LOG.write_text(LOG.read_text() + "    func: setup\n"); yield
    LOG.write_text(LOG.read_text() + "    func: teardown\n")

def test_a(s_session, s_module, s_func): pass
def test_b(s_session, s_module, s_func): pass
def test_c(s_session, s_module, s_func): pass
```

```
$ pytest test_scope_order.py -q
3 passed in 0.07s
$ cat order.txt
session: setup
  module: setup
    func: setup
    func: teardown
    func: setup
    func: teardown
    func: setup
    func: teardown
  module: teardown
session: teardown
```

三条从实验里读出的规律：① 宽作用域先 setup、**后** teardown（洋葱模型）；② function 级每个测试各走一轮——三个测试三组 setup/teardown，这就是"默认隔离"的物理来源；③ 同一作用域内有多个 fixture 时，按**依赖关系**排序（谁被依赖谁先建），无依赖时按声明顺序。teardown 顺序恰好相反。

**选型纪律**：作用域越宽，隔离越差。默认 function；只有"创建成本明显高且内容只读"（如预构建的测试数据集、本地 mock 服务器）才升到 module/session。session 级 fixture 里**永远不要放可变状态**——它会被一整个测试会话共享，一个用例改了它，后面全是薛定谔的绿。

## 2. 组合与依赖注入：fixture 是测试的构造函数

fixture 最大的能力不是替代 setUp，而是**组合**——一个 fixture 可以声明依赖另一个 fixture，pytest 自动解析依赖图、按正确顺序装配：

```python
# test_compose.py
import pytest

@pytest.fixture
def config():                          # 最底层：纯配置
    return {"retry": 3, "timeout": 1.0}

@pytest.fixture
def fake_llm(config):                  # 中层：依赖配置
    class FakeLLM:
        def __init__(self, cfg): self.calls, self.cfg = 0, cfg
        def chat(self, prompt): self.calls += 1; return "pong"
    return FakeLLM(config)

@pytest.fixture
def agent(fake_llm):                   # 上层：组装被测对象
    from mylib.agent import Agent
    return Agent(llm=fake_llm, **fake_llm.cfg)

def test_agent_retries(agent, fake_llm):
    assert agent.run("ping") == "pong"
    assert fake_llm.calls == 1
```

测试函数只声明它**业务上需要**的东西（`agent`），中间层的 `config` 不用关心——这是标准的**依赖注入**：依赖由外部容器（pytest 的 fixture 解析器）装配，而不是对象自己 `new`。06 篇会把同样的思想用到产品代码上（构造函数注入），两边是同一件事：*可测试性的本质是"依赖从外面给进来"*。

pytest 文档《How to use fixtures》把这个结构称为"fixture 固件层次"。两个工程细节：

1. **测试只声明直接依赖**，不要为了拿 `config` 而多声明一个不用的参数——需要值时直接再注入 `config` 一次（同名 fixture 是单例，同一次装配里是同一个对象）；
2. **依赖图在收集期静态可分析**：`pytest --collect-only` 之外还有 `--setup-plan` 能打印"将要执行的 fixture 装配顺序"，调复杂组合时非常有用。

## 3. autouse：方便与滥用一线之隔

`autouse=True` 让 fixture 无需声明就对作用域内所有测试生效：

```python
@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """每个测试都在临时目录里运行，防止污染仓库。"""
    monkeypatch.chdir(tmp_path)
    yield
```

这是 autouse 的正当用法：**全局环境约定**——隔离工作目录、固定时区、关闭网络、给全量测试注入统一日志配置。它相当于 unittest 的 setUp 全局版，但更透明（能 grep 到、能断言、能分层覆盖）。

滥用长这样：

```python
@pytest.fixture(autouse=True)          # ❌ 反例
def database(db_url):
    db = connect(db_url)
    yield db
```

全量测试被迫连接数据库，纯逻辑的单元测试从毫秒级变成秒级，**测试分层（单元/集成）被悄悄抹平**。判断标准：

| 该用 autouse | 不该用 autouse |
| --- | --- |
| 环境隔离（chdir/时区/env/禁网） | 资源连接（数据库、HTTP、消息队列） |
| 全局确定性设置（固定随机种子） | 只被少数测试用的数据 |
| 统一的计时/清理兜底 | 任何"大多数测试用不上"的东西 |

**经验做法**：autouse 数量保持个位数，且都放在 conftest 里集中可见；任何需要"昂贵资源"的都改成显式声明，让集成测试标签（`-m integration`）来控制启停。

## 4. 内置 fixture 三剑客：tmp_path / capsys / monkeypatch

pytest 自带的 fixture 覆盖了最高频的三类隔离需求，先写对它们，再谈造轮子。

**tmp_path——文件系统隔离**（每测试一个独立的临时目录，`pathlib.Path`，测试后自动清理）：

```python
def test_write_config(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("retry: 3\n", encoding="utf-8")
    from mylib.config import load
    cfg = load(p)
    assert cfg["retry"] == 3
```

绝不要在测试里写死 `/tmp/xxx` 或相对路径——并行跑（`pytest -n`）或 CI 缓存恢复都会撞车。需要跨测试共享的临时目录才升 scope（如 `scope="session"` 的下载缓存）。

**capsys——stdout/stderr 捕获与断言**：

```python
def test_progress_output(capsys):
    from mylib.render import render_progress
    render_progress(done=2, total=3)
    out = capsys.readouterr()
    assert out.out == "[## ] 2/3\n"
    assert out.err == ""
```

`capsys` 让"输出"从print-and-pray 变成可断言行为。它捕的是 pytest 已捕获的流（与 `-s` 互斥）；CLI 库的测试里，这比 `subprocess` 自跑快一个量级。注意 `capsys.readouterr()` 会清空缓冲，断言前的输出都在、之后的归零。

**monkeypatch——环境/属性/字典的临时修改**（teardown 自动还原，这是它相对手写 try/finally 的全部意义）：

```python
def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("MYLIB_API_KEY", "test-key")
    from mylib.client import current_key
    assert current_key() == "test-key"

def test_clock(monkeypatch):
    class FakeTime:
        @staticmethod
        def time(): return 1700000000.0
    monkeypatch.setattr("mylib.timing.time", FakeTime.time)
    ...
```

四个常用方法：`setenv/delenv`（环境变量）、`setattr(目标, "名", 值)`（对象属性/模块函数，字符串形式 `"模块.属性"` 可选，但**推荐写全字符串路径**，patch 失败会立刻报 AttributeError 而不是静默）、`chdir`（切目录）、`setitem`（字典项）。与 04 篇的 `unittest.mock.patch` 相比，monkeypatch 的删除值语义（`delenv`、` raising=True`）和自动还原更省心；mock 的强项在"替换调用行为并记录调用"，04 篇辨析。

## 5. conftest 分层：目录即命名空间

conftest.py 按**目录层级**自动生效，根 conftest 放全局约定，子目录放私有 fixture，子目录同名 fixture **覆盖**上层：

```text
tests/
├── conftest.py              # 根：_isolate_cwd(autouse)、fake_llm、agent
├── unit/
│   └── test_agent.py        # 用根层 fake_llm
└── integration/
    ├── conftest.py          # 覆盖 fake_llm：换成 httpx MockTransport 版
    └── test_agent_api.py    # 同名"agent"fixture，真实 HTTP 栈
```

```python
# tests/integration/conftest.py
import pytest

@pytest.fixture
def fake_llm():
    """集成层的假 LLM：真实 HTTP 协议 + 假后端（07 篇展开）。"""
    import httpx, respx
    with respx.mock(base_url="https://api.example.com") as r:
        r.post("/v1/chat").respond(json={"reply": "pong"})
        yield r
```

同一个测试函数名、同一份测试代码，在不同目录里吃到不同的 `fake_llm`——**同一组用例可以同时测单元层和集成层**，这是 conftest 分层最锋利的用法。反过来的教训是：把所有 fixture 都堆在根 conftest，几周后没人知道某个 fixture 谁在用、能不能改。**经验做法**：fixture 跟着使用者走——只有一个目录用，就放该目录的 conftest；≥2 个目录用，才提升到根。

## 6. 工厂 fixture：make_user 模式

参数化 fixture 解决"这批测试都要同一个对象"，但很多测试需要**多个、且带定制参数**的对象（比如"一个 admin 加两个普通用户"）。工厂 fixture 把"构造 + 记录 + 清理"打包：

```python
# test_factory.py
import pytest

@pytest.fixture
def make_user():
    created = []
    def _make(name: str = "alice", admin: bool = False) -> dict:
        u = {"name": name, "admin": admin, "id": len(created) + 1}
        created.append(u)
        return u
    yield _make
    print(f"清理 {len(created)} 个用户")   # teardown：统一回收

def test_admin(make_user):
    assert make_user(admin=True)["admin"]

def test_two_users(make_user):
    a, b = make_user(), make_user("bob")
    assert a["name"] != b["name"]
```

```
$ pytest test_factory.py -q -s
.
清理 1 个用户
.
清理 2 个用户

2 passed in 0.07s
```

这是 pytest 文档《fixture 工厂模式》一节的标准形态，要点有三：

1. **闭包变量 `created` 记录所有产物**，teardown 统一清理——比"每个测试自己清理"可靠（失败路径也走 teardown）；
2. **默认值参数化**：测试只写差异（`admin=True`），公共配置收敛在工厂里；06 篇的 Builder 模式是它的"产品代码版"；
3. 需要更复杂的构造链时，工厂可以依赖其他 fixture（如 `make_user(config)`）——工厂也是普通 fixture，组合规则照旧。

## 常见误区

**误区一：把昂贵资源放进 function 级。** 数据库容器、模型加载放 function 作用域，全量测试时间从 10 秒变 10 分钟。该 session 的 session，同时守住"只读"底线。

**误区二：session 级 fixture 里改共享状态。** 一个测试把共享 dict/dataset 改了，后面全乱且顺序相关。要共享可变状态？要么每测试深拷贝，要么回 function 级。

**误区三：在 fixture 里写断言。** fixture 失败报错是 `error` 而非 `failed`，会掩盖真正的测试语义；fixture 只负责"准备与回收"，断言留给测试函数。

**误区四：autouse 依赖了昂贵 fixture。** autouse 是"对所有测试强制注入"，它依赖的 fixture 连带全量执行。给 autouse 做 code review 时专看依赖链。

**误区五：同名 fixture 意外覆盖。** 根 conftest 的 `fake_llm` 被子目录同名 fixture 静默替换，单测层跑出了集成层行为。排查工具：`pytest --setup-plan` 打印实际装配，一眼看到用的是哪一层。

## 本篇小结

- 四个作用域的生命周期是洋葱模型：宽的先开后关；默认 function，昂贵的只读资源才升宽；宽作用域禁放可变状态；
- fixture 组合即依赖注入：测试声明直接依赖，pytest 按依赖图装配；`--setup-plan` 是装配排查器；
- autouse 只用于全局环境约定（隔离/确定性/兜底），资源连接永远显式；
- tmp_path/capsys/monkeypatch 覆盖文件、输出、环境三类隔离；monkeypatch 的自动还原是相对手写的全部价值；
- conftest 分层让"同一份测试代码跑单元层与集成层"成为机制，fixture 跟使用者走；
- 工厂 fixture（make_user）= 构造 + 记录 + 统一清理，是多对象测试的标准答案。

## 练习

1. 复现 1 节实验，再把 `s_func` 改成 `scope="module"`，预测 order.txt 的新内容，跑一遍验证你的预测；再故意让 `test_b` 抛异常，观察 teardown 是否仍然完整。
2. 用组合 fixture 搭"config → fake_clock → rate_limiter"三层依赖（限流器按 fake 时钟计窗口），写一个测试验证"窗口内第 N+1 次调用被拒"。
3. 把项目里某个手写的 try/finally 环境恢复代码改写成 monkeypatch 版本，对比代码行数与可读性。
4. 按第 5 节搭一个 tests/unit 与 tests/integration 双层目录，同名 `fake_llm` 分层覆盖，用 `pytest --setup-plan` 截图级确认两层各用各的。
5. 给工厂 fixture 加"批量构造 + 自动分配唯一 id + teardown 批量删除"能力（`make_users(3, role="viewer")`），并让删除顺序与创建顺序相反——想清楚为什么这更安全。
