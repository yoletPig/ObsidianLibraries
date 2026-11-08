# 12 - import 系统与模块工程

## 本篇目标

1. 理解 **import 机制**：`sys.modules` 缓存、查找器（finder）与加载器（loader）的分工；
2. 会诊断与根治**循环导入**，说清它的成因（执行到一半的模块）；
3. 掌握包结构、`__init__.py`、相对导入的规则与项目布局（与 uv 系列 02 篇的 src 布局衔接）；
4. 会用 `importlib` 做**动态导入**（插件系统），落地"导入即注册"。

## 1. import 机制：缓存、查找器、加载器

`import foo` 的幕后（官方文档：The import system，Python Language Reference 第 5 章）分三步：

1. **查缓存**：`sys.modules` 是"已导入模块名 → 模块对象"的全局字典，命中直接复用——**模块在整个进程里只执行一次**，后续 import 都是拿同一个对象（01 篇引用语义：两处 import 得到的是同一个模块对象）；
2. **查找**：未命中则由**查找器（finder）**链搜索（内置模块 → `sys.path` 上的文件系统 → 各种扩展点），找到后产生"模块规格"（`ModuleSpec`）；
3. **加载与执行**：**加载器（loader）**创建空模块对象放入 `sys.modules`，然后**从头到尾执行模块代码**——模块的顶层语句（包括定义、注册副作用）在这一刻运行。

```python
>>> import sys
>>> "json" in sys.modules          # 导入前
False
>>> import json
>>> "json" in sys.modules          # 导入后：进了缓存
True
>>> m1 = sys.modules["json"]       # 缓存里的模块对象
>>> import json as j2
>>> m1 is j2                       # 两次 import 是同一个对象（01 篇）
True
>>> sys.path                       # 查找器按此顺序搜索（脚本目录通常在最前）
['/home/you/proj', '/usr/lib/python3.12', ..., '/usr/lib/python3.12/site-packages']
```

三个直接推论（后面反复用到）：

- **import 有副作用**：模块顶层代码在"第一次 import"时执行——03/11 篇的"导入即注册"正是利用这一点；反过来，顶层重活（连网络、读配置）会让 import 变慢（03 篇经验做法的根源）；
- **模块级全局状态天然单例**：连接池、配置对象放模块顶层，全进程共享同一份；
- **`sys.path` 决定一切**：脚本的启动方式（`python a.py` 把脚本目录置顶）直接改变搜索路径——相对导入问题的根源之一（第 3 节）。

## 2. 循环导入：成因与四种解法

**成因**：A 模块 import B，B 又 import A——**模块只执行到一半**。展开时序：`import a` → 执行 a.py → 遇到 `import b` → 执行 b.py → 遇到 `import a` → **a 在 sys.modules 里已存在（半成品）** → b 拿到"a 模块对象"，但 a 的属性还没定义完 → b 若此刻读取 a 的某个名字 → `ImportError`/`AttributeError`（取决于访问时机）。

```python
# ---------- 反例：accounts.py 与 auth.py 互相需要 ----------
# accounts.py
from auth import require_user          # ① 从 auth 拉名字

class Account:
    owner = require_user()             # ② 类定义时就要用

# auth.py
from accounts import Account           # ③ 又回头拉 accounts ← 循环！
# 时序：import accounts → 执行到 ① → import auth → 执行到 ③ →
#        accounts 半成品（Account 未定义）→ ImportError
```

四种解法（按推荐度排序）：

**解法一（根治）：下沉公共依赖**。循环的信号灯是"两个模块互相需要对方的某个东西"——把它抽到第三个底层模块，双方都向下依赖， DAG 恢复：

```python
# users.py（新底层）：放 require_user
# accounts.py：from users import require_user
# auth.py：from accounts import Account（单向，无环）
```

**解法二（架构层）：依赖倒置**。A 需要 B 的功能却不想 import B？让 B 实现 A 定义的协议（06 篇 Protocol），在更高层注入。框架/插件架构的标准做法。

**解法三（局部缓兵）：函数内延迟 import**。import 移到**函数体内**——调用时才执行，此时两个模块都已完成：

```python
# accounts.py
def log_account_event(acc: "Account") -> None:   # 注解也可用字符串前向引用（06 篇）
    from auth import current_user                # ★ 运行时才 import，绕开循环
    print(f"{current_user()} 操作了 {acc}")
```

**解法四（类型专用）：`TYPE_CHECKING` + 字符串注解**。循环只为"类型注解"而存在时（最常见：模型与仓库互相引用类型），用类型检查专用分支——运行时根本不执行：

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:                       # 只有 mypy 看，运行时不 import
    from accounts import Account

def freeze(acc: "Account") -> None:     # 注解写成字符串
    ...
```

**经验做法**：循环导入是**设计问题的症状**而不是 import 的 bug——看到循环先想"哪个模块放错了层"（解法一/二），函数内 import 只用于"第三方懒加载"（如 `matplotlib` 只在画图函数里 import）这种性能优化场景。

## 3. 包结构、__init__.py 与相对导入

**包（package）= 含 `__init__.py` 的目录**（3.3+ 还有无 `__init__.py` 的命名空间包，用于多分发包合并，常规项目不用管）。`__init__.py` 的三件事：

1. 标记"这是包"（也是包的初始化代码执行点）；
2. **定义包的公共 API**：把内部模块的名字提升到包级别；
3. `__all__` 声明 `from pkg import *` 的边界。

```python
# mypkg/__init__.py
from mypkg.reader import read_records          # ★ 提升：用户写 from mypkg import read_records
from mypkg.models import Record
__all__ = ["read_records", "Record"]           # 公共 API 白名单（PEP 8 惯例）
__version__ = "1.0.0"
```

**相对导入规则**：包内模块用 `from . import x`（当前包）、`from .. import y`（父包）——**只能用在包内模块**，顶级脚本（直接 `python file.py` 运行）里用会炸：

```python
# mypkg/reader.py
from mypkg.models import Record     # ✅ 绝对导入（推荐：重构安全、可读）
from .models import Record          # ✅ 相对导入（包内短引用）
from .models import Record as _R    # 内部私有约定：下划线 + 不进 __all__
```

```text
>>> # 反例：直接运行包内文件
$ python mypkg/reader.py
ImportError: attempted relative import with no known parent package
# 原因：直接运行时该文件是顶级脚本 __package__ 为空，"." 无从解析
# 正解：python -m mypkg.reader（以模块身份运行）或统一绝对导入
```

**项目布局**（与 uv 系列 02 篇 `uv init --lib` 产出的 **src 布局**完全一致；打包权威论证见 [Python Packaging User Guide](https://packaging.python.org/)）：

```text
myproject/
├── pyproject.toml          # 项目与依赖声明（uv 系列）
├── src/
│   └── mypkg/
│       ├── __init__.py     # 公共 API 与版本
│       ├── core/
│       │   ├── __init__.py
│       │   └── engine.py
│       ├── plugins/
│       │   ├── __init__.py
│       │   └── search.py
│       └── cli.py
└── tests/                  # 在包外：测试 import 的是"安装后的包"（src 布局的意义）
```

src 布局的核心价值：**测试跑的一定是安装后的包**，杜绝"测试 import 了工作目录里的源码、发布版却是坏的"这类事故。**经验做法**：模块内 import 优先绝对导入（`from mypkg.core import engine`），相对导入只用于紧邻兄弟；`__init__.py` 保持薄（只做提升），重逻辑进子模块——否则 import 包的代价被顶层副作用放大。

## 4. 动态导入：importlib 与插件系统

**`importlib.import_module("pkg.mod")`**：运行时按字符串导入——插件系统的地基（11 篇元类注册的最后一环在这里闭环）：

```python
import importlib

mod = importlib.import_module("mypkg.plugins.search")   # 等价 import mypkg.plugins.search
cls = getattr(mod, "SearchPlugin")
print(cls.name)
```

**完整插件系统**：包目录约定 + `pkgutil.iter_modules` 扫描 + 动态导入触发注册（11 篇元类/`__init_subclass__` 的注册副作用）：

```python
"""plugins/__init__.py —— 插件加载器：导入即注册。"""
import importlib
import pkgutil

from .base import Plugin            # Plugin.__init_subclass__ 负责注册（11 篇）

def load_all() -> dict[str, type]:
    package = importlib.import_module(__name__)          # 当前包
    for info in pkgutil.iter_modules(package.__path__):  # 扫描包下所有模块
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{info.name}")  # ★ 导入即触发 @注册
    return Plugin.registry

# plugins/base.py
class Plugin:
    registry: dict[str, type] = {}
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        Plugin.registry[getattr(cls, "name", cls.__name__.lower())] = cls

# plugins/search.py（第三方或团队任何人加文件即可扩展）
from .base import Plugin
class SearchPlugin(Plugin):
    name = "search"
    def run(self, payload): return {"hits": [...]}

# 用法
>>> from plugins import load_all
>>> load_all()
{'search': <class 'plugins.search.SearchPlugin'>}
```

配套工具：`importlib.resources`（3.9+）读包内数据文件（`importlib.resources.files("mypkg") / "data.json"`，解决"打包后数据文件在哪"的问题）；`importlib.reload(mod)` 重载（开发调试用，生产慎用——旧对象的引用不会更新）。**经验做法**：插件命名空间独立成子包，加载器只认"实现了 Plugin 协议"（06 篇 Protocol 校验），配合 `try/except ImportError` 隔离坏插件，不让一个坏文件拖垮启动。

## 5. 命名规范与工程纪律

PEP 8 的 import 条款（14 篇汇总，这里列 import 专属）：

- **顺序三段式**：标准库 → 第三方 → 本地，段间空行（isort/ruff `I` 规则自动排序）；
- **避免 `from x import *`**（污染命名空间、破坏静态分析）；**避免隐式相对**（3.x 已禁止）；**不要 `import` 未使用的模块**（ruff `F401` 拦截）；
- 模块名小写下划线（`my_module`），包名短小写；`__all__` 与下划线前缀配合表达公共/私有边界。

**经验做法**（工程纪律）：① 顶层模块保持"import 快"——重依赖（matplotlib/torch）下沉到使用它的函数内；② 配置对象放独立 `config.py`（模块级单例 + 类型注解），其他模块 `from mypkg import config`；③ 循环导入先修依赖方向，`TYPE_CHECKING` 只救类型；④ 与 uv 系列衔接——`uv add` 管依赖声明、本篇管代码内 import 关系，两层正交：**包管理器解决"包从哪来"，import 系统解决"名字从哪找"**。

## 常见陷阱

**陷阱一：直接运行包内文件用相对导入**（第 3 节）——`ImportError: attempted relative import with no known parent package`；用 `python -m pkg.mod` 运行或改绝对导入。

**陷阱二：循环导入靠"调换 import 顺序"碰运气**——症状时有时无（取决于哪个模块先被 import），其实环还在；按第 2 节解法根治。

**陷阱三：`__init__.py` 里塞重逻辑/重依赖**——`import mypkg` 变得又慢又有副作用，还容易制造循环（`__init__` 导子模块、子模块又导 `__init__`）。

**陷阱四：shadowing 标准库**——项目里放个 `json.py`/`email.py`，`import json` 导入的是它自己（脚本目录在 `sys.path` 最前），诡异报错排半天；命名避开标准库（14 篇评审清单有此条）。

**陷阱五：动态导入后以为会"自动注册"却没触发**——只 `pkgutil` 扫描没 `import_module`，注册副作用没发生（11 篇模式的三步：扫描 → 导入 → 注册，缺一不可）。

## 本篇小结

- import 三步：`sys.modules` 缓存（模块只执行一次、天然单例）→ 查找器按 `sys.path` 搜 → 加载器执行模块代码（**import 有副作用**）；
- 循环导入 = 执行到一半的模块被引用；根治靠**依赖下沉/倒置**，缓兵用函数内 import 或 `TYPE_CHECKING`；
- 包 = 带 `__init__.py` 的目录；`__init__` 定义公共 API；绝对导入优先；src 布局让测试测的是安装后的包（uv 02 篇 `uv init --lib`）；
- `importlib.import_module` + `pkgutil` 扫描 = 插件系统：**扫描 → 动态导入 → 注册**，11 篇元类在这里闭环；
- 纪律：三段式 import 顺序、禁 `*`、不 shadow 标准库、`__init__` 保持薄。

## 练习

1. 复现循环导入：按第 2 节反例创建两个文件，观察 ImportError；依次应用解法一（下沉）与解法四（TYPE_CHECKING）修复，记录两种修复后的 import 结构图。
2. 构建"计算器插件"项目：`plugins/` 下实现 `add`/`mul`/`fib` 三个插件（各自实现 `run(a, b)`），用 `load_all()` 动态加载并通过注册表调用；故意放入一个语法错误的插件文件，验证"坏插件不拖垮启动"的隔离写法。
3. 解释下面输出的每一行（提示：sys.modules、模块单例、执行次数）：
   ```python
   # counter.py: print("counter 被执行"); count = 0
   >>> import counter
   >>> import counter
   >>> from counter import count
   >>> import sys; sys.modules.pop("counter")
   >>> import counter
   ```
4. 把一个扁平脚本项目（所有 .py 在根目录）重构成 src 布局 + 包（`uv init --lib` 起步），用 `uv run pytest`（或 `python -m pytest`）验证 tests 目录 import 的是包而不是根目录文件；说明 `python -m` 与 `python 路径.py` 两种运行方式对 `sys.path` 的不同影响。
5. 用 `importlib.resources` 给你的包加一个 `data/settings.json` 并写 `load_settings()` 读取；把它安装成 wheel（`uv build`，uv 系列 03 篇）后在新目录测试"数据文件随包走"。
