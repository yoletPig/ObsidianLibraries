# 04 - Mock 与测试替身：边界在哪里

## 本篇目标

1. 用五分类表（Dummy/Stub/Spy/Mock/Fake）精确说出你手里的替身是什么；
2. 熟练使用 `unittest.mock`：MagicMock、`patch` 的三种作用域与各自的坑；
3. 建立 mock 边界原则——知道什么时候该 mock，什么时候 mock 是反模式；
4. 会用 Fake 仓库与 freezegun 处理"Mock 拙劣"的两类难题：状态与时间。

## 1. 测试替身五分类：你用的到底是什么

Gerard Meszaros 在《xUnit Test Patterns》（2007）里把所有替身统称 Test Double，并给出五个子类。日常口语里大家全喊 "mock"，导致讨论永远对不上频道。先对表入座：

| 类型 | 干什么 | 关心什么 | 典型实现 |
| --- | --- | --- | --- |
| **Dummy** | 只为填参数签名，从不被真用 | 什么都不关心 | `None`、`object()`、空 `Mock()` |
| **Stub** | 返回固定数据，让被测代码能跑下去 | 被测代码**读**了什么 | `Mock(return_value=...)`、respx 打桩 |
| **Spy** | Stub + 记录调用，事后查验 | 被测代码**调**了什么 | `Mock()` 的 `call_args_list` |
| **Mock** | 预先设定期望：*必须*被这样调用 | 交互**协议**对不对 | `Mock(spec=...)` + `assert_called_with` |
| **Fake** | 有真实行为的轻量实现 | **状态**变化对不对 | 内存字典当数据库、内存消息总线 |

一句口诀：Stub 回答"输入端给了什么"，Spy/Mock 回答"输出端调了什么"，Fake 回答"用过之后状态对不对"。**Spy 与 Mock 的差别**在检验时机：Spy 是"先放任调用、事后查验"（state verification），Mock 是"事前期望、调用即校验"（behavior verification）——Meszaros 的原书第 6 章专门辨析这对双胞胎。

为什么分类重要？因为**不同类型出错的方式不同**：Stub 的问题通常是"桩数据失真"，Mock 的问题通常是"把实现细节焊死进测试"（过度指定），Fake 的问题通常是"和真实现行为漂移"（第 5 节给对策）。

## 2. unittest.mock 基础：MagicMock 与 patch 的三种作用域

```python
from unittest import mock

llm = mock.MagicMock()          # 万物皆可替换的万能对象
llm.chat.return_value = "pong"  # 当 Stub 用：指定返回
llm.chat("ping")                # 真被调用
assert llm.chat.called and llm.chat.call_args == mock.call("ping")
llm.anything.at.all             # 属性访问永不报错——这是特性也是陷阱
```

`MagicMock` 支持 `return_value`（固定返回）、`side_effect`（序列返回/抛异常/可调用体）、调用记录（`call_count`/`call_args`/`call_args_list`）。`spec=SomeClass` 参数让它只接受真实类已有的属性——**用 mock 必带 spec**，后面讲为什么。

把 mock 放进被测代码要用 `patch`，三种作用域，三种不同的坑：

```python
import mymod
from unittest import mock

# ① 装饰器：函数结束时自动还原；mock 从测试参数传入（名字必须是 mock_xxx 或参数序匹配）
@mock.patch("mymod.get_client", autospec=True)
def test_a(fake_client):
    fake_client.fetch.return_value = []
    assert mymod.sync() == 0

# ② 上下文管理器：作用域最清晰，推荐默认
def test_b():
    with mock.patch("mymod.get_client", autospec=True) as fake_client:
        fake_client.fetch.return_value = []
        assert mymod.sync() == 0

# ③ 手动 start/stop：适合跨多个测试共享（一般放 conftest 里包成 fixture）
def setup_module(module):
    patcher = mock.patch("mymod.get_client", autospec=True)
    patcher.start()
    module.addCleanup(patcher.stop)    # 比手动 stop 可靠：失败也会执行
```

两个必守细节：

1. **`autospec=True`（或 `create_autospec`）**：让替身严格镜像原函数签名——多传、少传、改了参数名都会立刻报错。裸 MagicMock 对签名完全沉默，被测代码改了接口测试还绿，这是 mock 测试腐化的头号来源；
2. **`patch` 的目标必须是"被测代码视角的名字"**——下一节的三大翻车现场全在这。

## 3. mock 边界原则：不要 mock 你不拥有的东西

Steve Freeman 与 Nat Pryce 在《Growing Object-Oriented Software, Guided by Tests》（GOOS，2009）提出的原则：**"Don't mock what you don't own"**——不要 mock 第三方库，为自己不拥有的接口写替身。这里的"拥有"不是"写过它"，而是"它的接口契约由你定义、你能随意改动"。

翻车机理分两层：

**第一层：替身失真。** 第三方库的内部行为你没有契约保证。mock 掉 `requests.get` 并断言 `retry_called_once`，等于断言了**你对库内部实现的猜测**——库一升级，你的测试是绿的，线上是红的。测试替身的价值在"替身与真实现契约一致"，而第三方契约你既没定义、也没法验证。

**第二层：API 泄漏。** mock 第三方意味着测试里到处是 `mock.call("POST", "/v1/x", params=...)` 这类第三方调用细节，库一改 API，散落几十处的 mock 全要改。

**正确姿势：包一层你拥有的端口。** 把第三方收进自家接口：

```python
# src/mylib/ports.py —— 你拥有的接口（抽象）
class LLMClient(Protocol):
    def chat(self, prompt: str) -> str: ...

# src/mylib/adapters.py —— 真实现：薄壳，只在集成层测
class OpenAIClient:
    def __init__(self, base_url: str, key: str): ...
    def chat(self, prompt: str) -> str: ...

# tests/fakes.py —— 假实现：你拥有，行为可验证
class FakeLLMClient:
    def __init__(self, reply: str = "pong"): self.calls: list[str] = []; self._reply = reply
    def chat(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._reply
```

业务测试 mock **你自己的端口**（`FakeLLMClient`），`OpenAIClient` 只在集成测试里对真/伪服务端测（07 篇）。这样 mock 数量从"每个测试各自 patch 第三方"收敛为"每个端口一个 Fake"，且替身契约由你定义、随重构一起走。06 篇的端口-适配器与此互为表里：**可 mock 性是设计出来的，不是 patch 出来的**。

**经验做法**：代码评审时看到 `mock.patch("openai.ChatCompletion.create")` 这类字符串直接打回——要么包一层端口，要么降级为集成测试。

## 4. patch 路径写错的三种现场

patch 的目标解析规则：`patch("a.b.c")` 会 import `a.b` 再 setattr 其 `c`。写错路径的三种典型现场，输出全部实测：

**现场一：patch 了源头，没 patch 使用处（最阴险，不报错）。** 被测模块 `from os import getcwd` 在**导入期**把函数对象绑进了自己的命名空间，之后 patch `"os.getcwd"` 改的是 `os` 模块，对 `mymod` 里的旧引用毫无影响：

```python
import mymod                      # mymod 顶部有 from os import getcwd
from unittest import mock

def test_wrong_target():
    with mock.patch("os.getcwd", return_value="/fake"):
        assert mymod.where_am_i() == "/fake"
```

```
E           AssertionError: assert '/tmp/testlab/ch04' == '/fake'
FAILED test_patch_demo.py::test_patch_wrong_target - AssertionError: assert '...
```

静默返回真实路径。修法：patch **使用处**——`mock.patch("mymod.getcwd", ...)`（实测通过）。反过来，`import os` 后经 `os.getcwd()` 调用的代码，patch `"os.getcwd"` 就能生效——**patch 目标取决于被测模块怎么引用它**，不是"哪里定义 patch 哪里"。

**现场二：属性名写错。** `mock.patch("mymod.get_cwd")`（真实属性是 `getcwd`）：

```
AttributeError: <module 'mymod' from '/tmp/testlab/ch04/mymod.py'> does not have the attribute 'get_cwd'
```

patch 在进入 with 时立即校验属性存在，报错早、好定位。（`create=True` 能绕过校验——除非给不存在属性占位，否则别用，它会掩盖拼写错误。）

**现场三：模块名写错。** `mock.patch("mymd.getcwd")`：

```
ModuleNotFoundError: No module named 'mymd'
```

同样早失败。真正麻烦的是这三种之外的第四种：**patch 对象是值拷贝**——被测模块导入期就执行 `TIMEOUT = load_config().timeout`，之后你 patch `load_config` 或 `TIMEOUT` 都改不了已经拷贝的值。对策不在 patch 层，在代码层：值依赖要么经参数传入，要么保留为可变引用（`settings.timeout`）。

## 5. Fake 比 Mock 好的场景：内存版仓库

凡是"**多次调用之间有状态**"的依赖，Mock 用起来极其别扭：你得用 `side_effect` 写一个"第一次返回 A、第二次返回 B"的状态机，等价于重写了一个劣质实现。此时 Fake 是更诚实的替身——一段有真实行为的轻量实现：

```python
# tests/fakes.py
class FakeUserRepo:
    """内存版用户仓库：与真仓库共享同一个接口。"""
    def __init__(self):
        self._data: dict[int, dict] = {}
        self._next_id = 1

    def add(self, user: dict) -> int:
        uid = self._next_id; self._next_id += 1
        self._data[uid] = {**user, "id": uid}
        return uid

    def get(self, uid: int) -> dict | None:
        return self._data.get(uid)

    def delete(self, uid: int) -> None:
        if uid not in self._data:
            raise KeyError(uid)
        del self._data[uid]

# test_register.py
def test_register_then_fetch():
    repo = FakeUserRepo()
    from mylib.service import register
    uid = register(repo, name="alice")
    assert repo.get(uid)["name"] == "alice"

def test_delete_missing_raises():
    repo = FakeUserRepo()
    import pytest
    with pytest.raises(KeyError):
        repo.delete(999)          # ❗ 顺便测出了 Fake 自身与真库契约的一致性
```

对比用 Mock 写 `test_register_then_fetch`：你得 `repo.get.side_effect = lambda uid: {...}`，等于在测试里手写了仓库实现，且**同一逻辑散落每个测试**。Fake 的额外红利：`KeyError` 这类**真实现的错误契约**可以被测试，Mock 只能 `side_effect=KeyError()` 人工安排。

**Fake 的风险与对策**：Fake 与真实现**行为漂移**（真库支持事务，Fake 不支持，测试全绿上线炸）。对策两条：① Fake 与真实现实现**同一个显式接口**（Protocol/ABC），接口变更时两边一起编译失败/测试失败；② 07 篇给 Fake 与真实现跑**同一套契约测试**——这正是不 mock 第三方原则的落地闭环。

## 6. 冻结时间：freezegun

时间是最常见的隐藏依赖。token 过期、账期计算、"今天"的日报——所有依赖 `datetime.now()` 的代码都不可重复测试。freezegun 冻结全局时钟：

```python
# test_expiry.py
import datetime
from freezegun import freeze_time

def is_expired(issued: str, ttl_days: int) -> bool:
    issued_at = datetime.datetime.fromisoformat(issued)
    return datetime.datetime.now() - issued_at > datetime.timedelta(days=ttl_days)

@freeze_time("2024-03-01")
def test_ttl60_not_expired():
    assert is_expired("2024-01-01", 60) is False    # 恰好 60 天，未超

@freeze_time("2024-03-01")
def test_ttl30_expired():
    assert is_expired("2024-01-01", 30) is True     # 早已过期

def test_unfrozen_still_real():
    assert is_expired("2024-01-01", 60) is True     # 真实时钟早已超过
```

三个实践要点：① `freeze_time` 也支持上下文管理器形式（`with freeze_time(...)`），需要"时间流逝"的场景用 `freeze_time(...).start()` 后 `tick()` 推进或逐段冻结；② 它对 `time.time`、`time.monotonic`、`random` 种子等一并有 hook，覆盖面比手写 monkeypatch 全；③ **教训（实测）**：`@freeze_time` 装饰**类**时不会可靠地穿透 `@staticmethod`——上面若把 `is_expired` 塞进类里用 staticmethod 装饰器套 freeze，冻结会静默失效。冻结逻辑要么挂在模块级函数，要么经 fixture 注入 fake clock（03 篇练习 2 的做法），不要依赖装饰类。

**经验做法**：代码里禁止裸调 `datetime.now()`，统一走 `mylib.clock.now()`——替换点只有一处，freezegun/monkeypatch/fake 三种方案随便挑，这又是"可测性是设计出来的"。

## 常见误区

**误区一：裸 MagicMock 无 spec。** 拼错的属性名、改掉的签名全部静默通过。用 `mock.create_autospec(真函数)` 或 `patch(..., autospec=True)`，让替身自带契约校验。

**误区二：mock 掉了被测逻辑本身。** `mock.patch.object(MyService, "_compute")` 然后测 `run()`——被测对象的核心私有方法被你替换了，测试只剩空壳。mock 只该挂在**依赖边界**（网络/DB/时钟/文件），不挂在被测对象内部。

**误区三：断言过度指定。** `assert_called_once_with("ping", timeout=3, headers={...})` 把实现细节焊进测试，重构必红。断言对外的关键交互（发了一条什么消息、调用了一次），其余用 Fake 的**状态断言**（仓库里多了一条记录）替代调用记录断言——状态断言比交互断言更抗重构。

**误区四：一个测试里 mock 五个依赖。** 这不是单元测试，是用 mock 胶水重写了实现。五个依赖意味着被测对象职责过多——先修设计（06 篇），再谈测试。

**误区五：patch 忘了还原。** 手动 `patch.start()` 不配 `addCleanup`/`stop`，替身泄漏进同进程的其他测试，制造"单独跑绿、全量跑红"的顺序耦合。永远 `module.addCleanup(patcher.stop)` 或用 with/装饰器。

## 本篇小结

- 五分类按"关心什么"记忆：Dummy 填位、Stub 给输入、Spy/Mock 查交互（事后/事前）、Fake 验状态；
- `patch` 三种作用域：装饰器/with/start+addCleanup；必带 autospec；
- 边界原则（GOOS）：不 mock 你不拥有的接口——第三方收进自家端口，替身只造在自己的端口上；
- patch 三种翻车：源头 vs 使用处（from-import 导入期绑定，静默失效）、属性名错（AttributeError）、模块名错（ModuleNotFoundError）；值拷贝类问题要靠设计解决；
- 状态依赖用 Fake（内存仓库），错误契约可测；Fake 漂移风险用共享接口 + 契约测试兜住；
- 时间统一走 `clock.now()`，freezegun 冻结（注意装饰类对 staticmethod 失效的坑）。

## 练习

1. 给 `FakeUserRepo` 补 `update` 方法并写"更新不存在 id 抛 KeyError"的测试；再为它写一段与真 SQLite 仓库对照的契约测试（同一套用例、两个实现各跑一遍）。
2. 把下面代码按"不 mock 你不拥有的东西"重构：`def sync_files(): for f in os.listdir("/data"): s3.upload(f)`——指出端口、真实现、Fake 三部分各放哪个文件，测试怎么写。
3. 复现 4 节现场一，然后把 `mymod` 的 `from os import getcwd` 改成 `import os`，验证 patch `"os.getcwd"` 变得生效——写一段 100 字解释根因。
4. 用 freezegun 写"连续登录失败 5 次、30 分钟后解锁"的测试（时钟经参数注入或 monkeypatch），故意制造一次"快进时间"的场景。
5. 找一个你自己项目里的 `patch("xxx.yyy")`，判断它 mock 的东西你"拥有"吗？不拥有的话，写出端口化改造的三步计划。
