# 09 - 静态分析与 Fuzzing：让机器在你之前找到 bug

## 本篇目标

1. 把类型检查纳入测试视野：mypy/pyright 各自能拦下哪类"跑起来才知道"的 bug；
2. 建立精简的 Python lint 组合：ruff 的 B/S/PT 规则组与 bandit 安全扫描；
3. 掌握 C++ 侧三板斧：clang-tidy 精选检查、cppcheck、ASan/UBSan/TSan 的 CMake 接法；
4. 入门 fuzzing：理解 AFL++/libFuzzer 的思想，会写最小 harness，看懂一个 fuzz 发现的真实类型 bug。

## 1. 类型检查即测试：mypy/pyright 拦下什么

把类型检查器理解成**零运行成本的自动测试**：它对所有分支做一次"类型层的全量执行"——这是手写测试永远做不到的覆盖（05 篇说 property-based 能自动生成输入，类型检查更进一步：生成**全部类型形态**的输入）。

一个真实的函数，两处潜伏 bug（实测 mypy 输出）：

```python
# demo_mypy.py
def running_total(xs: list[float]) -> list[float]:
    out = []
    total = 0                    # ❌ 初始化成了 int
    for x in xs:
        total += x
        out.append(total)
    return out                   # ❌ 实际是 list[int]

def send_notification(user_id: int | None) -> str:
    return f"notify user {user_id.upper()}"   # ❌ None 没有 .upper()
```

```
$ mypy demo_mypy.py --strict
demo_mypy.py:5: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
demo_mypy.py:7: error: Incompatible return value type (got "list[int]", expected "list[float]")  [return-value]
demo_mypy.py:10: error: Item "int" of "int | None" has no attribute "upper"  [union-attr]
```

第三条 `union-attr` 正是生产事故的经典来源：`None` 在某个没测过的分支里悄悄流进来。运行时测试要专门构造"None 进来"的用例，mypy 对**每一个可能为 None 的调用点**都报警——这就是"零成本全量覆盖"的含义。

工程要点：

1. **分阶段收紧**：`mypy . ` 全仓一次基线（老的报错先 `# type: ignore[码]` 记账），然后 CI 对**改动文件**强制 strict——增量策略与 08 篇的 diff 覆盖率同构；
2. **mypy 与 pyright 的取舍**：规则内核相近，pyright（VS Code/Pylance 默认）与编辑器体验一体化，mypy 是 CI 老牌标准——团队统一其一即可，别两套全开互相打架；
3. 类型检查**替代不了**行为测试（它不管 `total = 0` 该不该从 0 开始求和），但把它从"可选修养"升格为 CI 门禁（08 篇 yaml 已就位）后，你手写测试的数量会肉眼可见地下降。

## 2. Python lint 体系：ruff 规则精选 + bandit

ruff（Rust 实现，单工具替代 flake8/isort/pydocstyle 等数十个插件）的默认规则偏风格；真正抓 bug 的是三组扩展规则（实测输出）：

```bash
$ ruff check --select B,S,PT .
```

| 组 | 来源 | 抓什么 | 代表规则（实测命中的） |
| --- | --- | --- | --- |
| `B` | flake8-bugbear | 真实 bug 模式 | 可变默认参数、`assert (a, msg)` 永真、循环变量泄漏 |
| `S` | flake8-bandit | 安全反模式 | `S307` eval 使用、`S101` 生产代码 assert、硬编码密码 |
| `PT` | flake8-pytest-style | 测试代码一致性 | fixture 参数风格、parametrize 写法、断言写法 |

```
S307 Use of possibly insecure function; consider using `ast.literal_eval`
 --> demo_ruff.py:4:5
  |
4 |     eval(user_input)
  |     ^^^^^^^^^^^^^^^^
```

**配置建议**（`pyproject.toml`）：`select = ["E", "F", "B", "S", "PT"]`，其中测试文件豁免 `S101`（pytest 的断言就是 assert，生产禁用与测试必用冲突）：

```toml
[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101"]
```

bandit 与 `S` 组的关系：bandit 是安全专用扫描器，规则更深（含中危/高危分级、`# nosec` 白名单机制）；ruff 的 `S` 组是它的轻量子集。**经验做法**：日常门禁用 ruff `S` 组挡住低垂果实；发布到 PyPI 的包或处理不可信输入的服务，nightly 加 bandit 全量扫描（08 篇的分层策略再次适用）。

## 3. C++ 侧：clang-tidy 精选 + cppcheck

C++ 的 bug 面更宽（内存、未定义行为、生命周期），静态分析是刚需：

**clang-tidy**（基于真实编译数据库，误报率低）精选检查，经 CMake 接入（08 篇 yaml 里的 `-DCMAKE_CXX_CLANG_TIDY` 一行已生效）：

| 检查 | 抓什么 |
| --- | --- |
| `bugprone-*` | 悬悬引用、整数截断、异常逃逸析构 |
| `cppcoreguidelines-slicing` | 对象切片（多态按值拷贝丢失派生部分） |
| `modernize-use-override` | 该写 `override` 的虚函数 |
| `performance-*` | 不必要的按值拷贝、`*` 传参可改引用 |
| `readability-*` | 命名、`const` 正确性 |

```cmake
# CMakeLists.txt 片段：clang-tidy 接入 + 警告即错误
set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-checks=bugprone-*,performance-*,modernize-use-override;--warnings-as-errors=*")
```

**cppcheck**（免编译、超快，适合当"第二意见"）：`cppcheck --enable=warning,portability --inline-suppr src/`。与 clang-tidy 分工：cppcheck 扫**没编译过的第三方/遗留目录**，clang-tidy 守自家代码（它需要 compile_commands.json，`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` 生成）。

## 4. Sanitizer：运行时 UB 的照妖镜（CMake 接法）

静态分析抓**看得出来的**问题；内存与未定义行为要看**运行时现场**。三大 sanitizer：

| 工具 | 抓什么 | 开销 |
| --- | --- | --- |
| ASan（`-fsanitize=address`） | 越界读/写、use-after-free、内存泄漏 | 慢 ~2x，内存 ~3x |
| UBSan（`-fsanitize=undefined`） | 有符号溢出、除零、错位对齐、空指针解引用 | 慢 ~1.2x |
| TSan（`-fsanitize=thread`） | 数据竞争、死锁 | 慢 5~15x |

CMake 接法（一个构建目录一种 sanitizer，**不要混用 ASan/TSan**）：

```cmake
# 手动配置：Debug 构建挂 ASan+UBSan
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
cmake --build build-asan
ctest --test-dir build-asan --output-on-failure   # gtest 用例照常跑，崩溃即详情
```

**经验做法**：nightly job 用 Sanitizer 构建跑全量 gtest（08 篇的 nightly 分层），PR 主线不用——开销会拖慢反馈；本地排查内存问题时第一反应就切 ASan 构建。

## 5. fuzzing 入门：AFL++ 思想与 libFuzzer 最小 harness

fuzzing = **自动生成海量随机/变异输入，喂给被测函数，监控崩溃**。两大谱系：AFL++（覆盖引导：用插桩记录每个输入到达的分支，优先变异"能走到新代码"的输入，文件/进程级，语言无关）与 libFuzzer（同思想的进程内版本，编译进二进制，与 ASan 是天生一对）。覆盖引导是它们的共同灵魂：**盲随机**几乎撞不出深层路径，"以覆盖率为目标的进化"才能。

libFuzzer 最小 harness（被测对象：一个"长度前缀协议"解析器，故意留一个 bug）：

```cpp
// fuzz_proto.cpp
#include <cstdint>
#include <cstddef>
#include <cstring>

// 长度前缀协议：第 0 字节 = 负载长度 n，随后 n 字节为负载
int parse(const uint8_t* data, size_t size) {
    if (size < 1) return -1;
    int n = data[0];
    if (n > 15) return -1;              // 上界检查
    char buf[16] = {0};
    memcpy(buf, data + 1, (size_t)n);   // ❌ 没检查 size >= n + 1
    return buf[0];
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    parse(data, size);
    return 0;
}
```

编译与运行（clang 工具链：`sudo apt install clang` 即含 libFuzzer）：

```bash
clang++ -g -O1 -fsanitize=fuzzer,address fuzz_proto.cpp -o fuzz_proto
mkdir corpus && echo -n "abc" > corpus/seed1
./fuzz_proto corpus/ -max_total_time=60
```

在本系列写作环境中没有 clang，但**同一 bug 用 g++ + ASan 手工触发**，可以完整看到 fuzzing 命中后你会看到的现场（1 字节输入 `{15}`：声明 15 字节负载、一个都没给）：

```
$ g++ -g -fsanitize=address -o proto proto.cpp && ./proto
==18175==ERROR: AddressSanitizer: stack-buffer-overflow on address 0x7ffc09219fe1 ...
READ of size 15 at 0x7ffc09219fe1 thread T0
    #1 0x6445a305f3e3 in parse(unsigned char const*, unsigned long) proto.cpp:12
```

`#1` 直接指向 `memcpy` 那一行——这就是 fuzzing + sanitizer 组合的标准战果报告：**崩溃输入 + 精确栈**。修复：`if (n > 15 || size < (size_t)n + 1) return -1;`。Python 侧不需要 libFuzzer：hypothesis（05 篇）就是覆盖引导思想在 Python 的实现，Atheris 是 Python 解释器层的 libFuzzer 绑定，处理 C 扩展时值得一用。

## 6. bug 案例走读：fuzz 找到的真实类型 bug

最后一例，bug 的根源是**类型**，代码"看起来全对"。需求：协议头遇到 `0xFF` 终止符停止拷贝：

```cpp
int copy_header(const unsigned char* data, size_t size, char* out) {
    size_t i = 0;
    while (i < size) {
        char c = data[i];            // ❌ char 在 x86 Linux 上是 signed
        if (c == 0xFF)               // 0xFF 是 int 255；c 是 -1 → 永不相等
            break;
        out[i] = c;
        i++;
    }
    return (int)i;
}
```

实测运行（输入第 2 字节是 0xFF，`out` 只有 8 字节）：

```
$ ./typebug
copied 8 bytes: A□BBBBBB□BBBBBB     # 没在终止符停，把 0xFF 后面的也拷了
```

加一个字节的输入就会写越界，ASan 现场：

```
==25255==ERROR: AddressSanitizer: stack-buffer-overflow ...
```

**根因链条**（这就是"fuzz 找到的类型 bug"的标准形态）：`data[i]` 是 `unsigned char`（0~255），赋给 `signed char` 后 0xFF 变成 -1；比较时 `c` 提升为 int 的 -1，与字面量 `0xFF`（int 的 255）永不相等——终止符失效，循环一路拷出去。三层防线各自能拦住它：

1. **编译器警告**：`-Wall -Wextra -Wconversion` 对 `char c = data[i]` 会给 narrowing 警告（所以 C++ 门禁必须开警告即错误，08 篇 yaml 已含）；
2. **类型正确写法**：`uint8_t c = data[i]; if (c == 0xFF) break;`——同一类型域内比较，符号问题不存在；
3. **fuzz + ASan**：构造到 `0xFF` 的输入只要几秒，越界写当场现形。

AFL++/libFuzzer 历史上抓到的同类真实案例大量集中在解析器（图像库、压缩库、协议栈）——**处理不可信字节的代码，fuzzing 不是可选项**。

## 常见误区

**误区一：把静态分析当"全绿工程"。** 首次接入必然海量告警，正确姿势是基线记账 + 增量强制（08 篇策略），一上来 `warnings-as-errors` 只会催生海量 `# type: ignore`/`// NOLINT`。

**误区二：sanitizer 构建只跑冒烟。** ASan 只能抓"被执行路径"上的问题——挂在 nightly 全量测试上才有意义，只跑 hello world 等于没开。

**误区三：harness 里写了业务断言。** `LLVMFuzzerTestOneInput` 里每条 `assert` 都可能被合法输入"证伪"导致误报崩溃。harness 只该检查**无输入能使之崩溃的不变量**（不越界、不悬挂、返回值在域内）。

**误区四：fuzz 种子库为空。** 给几条"典型合法输入"当种子（corpus），引导引擎从合法结构开始变异——比纯随机起步快一个数量级。

**误区五：Python 项目不 fuzz。** 处理网络字节/用户上传的解析器照样 fuzz（hypothesis 收窄策略生成字节流即可），坐标转换、序列化、压缩解压都是高危区。

## 本篇小结

- 类型检查是零成本全量测试：`union-attr` 类 None 流入是它的招牌战果；基线 + 增量 strict 落地；
- lint 组合：ruff `B`（bug 模式）+ `S`（安全）+ `PT`（pytest 风格），测试文件豁免 S101；安全要求高的项目 nightly 加 bandit；
- C++ 静态面：clang-tidy 守自家代码（挂 CMake，警告即错误），cppcheck 扫免编译区域；
- ASan/UBSan/TSan 抓运行时 UB：nightly 全量跑 sanitizer 构建，PR 主线不背开销；ASan/TSan 不可同开；
- fuzzing 灵魂是覆盖引导；libFuzzer harness 极薄（只喂输入、只查崩溃不变量）；种子库决定起步速度；
- 类型 bug 案例：`char` 符号扩展让终止符永不相等——编译警告、正确类型、fuzz+ASan 三层防线缺一不可。

## 练习

1. 给 `running_total` 修两处类型问题后重跑 `mypy --strict` 至零告警；再故意写一个 `def f(x: int) -> int: return x / 2` 看看它能否被抓（答案在第 1 节找）。
2. 在你的项目上跑 `ruff check --select B,S,PT`，对每条告警分类：真 bug / 风格 / 误报（误报写 nosec 或调整配置），记录三类比例。
3. 给本章 `parse` 修复后重新用 ASan 验证：`{15}`、`{15, 'a'}`、`{16, ...}` 三组输入的返回值与退出码。
4. 把 `copy_header` 的 `char c` 改成 `uint8_t c` 重跑实测，再开 `-Wall -Wextra -Wconversion` 看原始版本触发哪几条警告。
5. 为你自己的一个字符串解析函数写 libFuzzer harness（或 hypothesis 字节流策略），准备 5 条种子，fuzz 5 分钟，记录是否发现输入能触发未定义行为或异常。
