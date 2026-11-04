# 11-错误处理optional-variant与异常

> **本篇回答的问题**
> - Python 里函数失败可以返回 `None` 或抛异常，C++ 对应的武器是什么？
> - `std::optional`、`std::variant`、异常，各自的适用场景是什么？
> - throw 之后发生什么（栈展开）？noexcept 又扮演什么角色？
> - 实战：同一个配置解析函数，optional 版和异常版并排对比。

前置：[03-内存模型与RAII](03-内存模型与RAII.md)（RAII、析构）、[06-移动语义与完美转发](06-移动语义与完美转发.md)（noexcept）。本篇所有带"预期输出"的示例都在 `g++ -std=c++17 -Wall -Wextra -pedantic` 下编译运行验证过，零警告。

## 1. C++ 错误处理三流派

Python 的世界很简单：预期内的失败用 `None` / `except`，程序基本靠异常。C++ 没有统一的答案，实际是三个流派并存，你会在任何一个真实代码库里同时见到：

| 流派 | 代表 | 一句话定位 |
|---|---|---|
| 返回错误码 | `int ret = f(); if (ret) ...` | C 语言传统；性能最好，但错误和正常返回混在一条通道 |
| 异常 | `throw` / `try` / `catch` | 错误与正常流程分离；有运行时开销且要求代码"异常安全" |
| expected 类物 | `std::optional`（C++17）、`std::variant`（C++17）、`std::expected`（C++23） | 把"可能失败"写进**返回类型**，编译器和调用方都躲不开 |

选型直觉（后文逐个展开）：**"没有结果"是正常业务 → optional；"结果是几种之一" → variant；"出了问题必须有人负责且无法在本层处理" → 异常**。音频/推理引擎对延迟确定性极其敏感，公开 API 偏向前两类；异常多见于初始化、配置加载等一次性路径。

## 2. std::optional：把"可能没有"写进类型

老代码表达"找不到"惯用**哨兵值**：找不到返回 `-1`、返回 `nullptr`、返回 `""`。问题在于哨兵值混进了合法取值范围，**编译器不会强迫你检查它**——忘了判断就越界/解引用，运行时才炸。

`std::optional<T>`（`<optional>`）是更好的答案：它是"一个 `T`，**或者**什么都没有（`std::nullopt`）"的容器，"可能没有"直接写在函数签名里，想忽略都难。

```cpp
// 11_ex1_optional.cpp
#include <iostream>
#include <optional>
#include <string>
#include <vector>

struct AudioDevice {
    std::string name;
    int channels;
};

// 旧风格：找不到返回 -1。调用方必须记得检查；而且 int 本来就是"下标"，
// -1 是混进合法取值范围里的暗号，编译器不会帮你盯着它。
int find_device_index(const std::vector<AudioDevice>& devs, const std::string& name) {
    for (std::size_t i = 0; i < devs.size(); ++i)
        if (devs[i].name == name) return static_cast<int>(i);
    return -1;
}

// C++17：optional<T> = "一个 T，或者什么都没有"，写进返回类型里。
std::optional<AudioDevice> find_device(const std::vector<AudioDevice>& devs,
                                       const std::string& name) {
    for (const auto& d : devs)
        if (d.name == name) return d;
    return std::nullopt;
}

int main() {
    std::vector<AudioDevice> devs{{"mic-0", 1}, {"speaker-0", 2}};

    if (auto d = find_device(devs, "mic-0"))        // optional 可直接当布尔条件
        std::cout << "找到 " << d->name << "，" << d->channels << " 声道\n";
    else
        std::cout << "没找到\n";

    auto missing = find_device(devs, "headset");
    std::cout << "has_value: " << missing.has_value() << '\n';

    // value_or：拿不到就用默认值，一行兜底
    auto fallback = missing.value_or(AudioDevice{"none", 2});
    std::cout << "声道数(默认2): " << fallback.channels << '\n';

    // 对比：哨兵版返回 -1，调用方忘了判断就越界——编译器不管
    int idx = find_device_index(devs, "headset");
    std::cout << "哨兵版下标: " << idx << "（使用 devs[idx] 会越界！）\n";
}
```

预期输出：

```
找到 mic-0，1 声道
has_value: 0
声道数(默认2): 2
哨兵版下标: -1（使用 devs[idx] 会越界！）
```

API 速查：

- `opt.has_value()` / 直接 `if (opt)`：有没有值。
- `*opt` / `opt->member`：取值（**无值时是未定义行为**，取之前必须检查）。
- `opt.value()`：取值，无值时抛 `std::bad_optional_access`——唯一和异常挂钩的路径。
- `opt.value_or(默认值)`：有值取值、无值取默认，最常用的兜底写法。

这比 Python 的 `dict.get(key)` 语义几乎一样，但 C++ 把"有没有"变成了**类型层面的事实**，忘检查是编译风格问题而非靠自觉。

## 3. 实例：安全解析配置项

推理引擎启动时要读一堆配置：`sample_rate=16000`、`frame_ms=20`。解析会以三种方式失败：键不存在、值不是数字、值带尾巴（`"20ms"`）。全部统一成 `nullopt`：

```cpp
// 11_ex2_config.cpp
#include <charconv>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// 把字符串安全地解析成 int：失败返回 nullopt，而不是魔法数或异常
std::optional<int> parse_int(std::string_view s) {
    int value = 0;
    const char* first = s.data();
    const char* last = s.data() + s.size();
    auto [ptr, ec] = std::from_chars(first, last, value);  // C++17 结构化绑定
    if (ec != std::errc{} || ptr != last)                  // 解析失败 / 有剩余字符
        return std::nullopt;
    return value;
}

// 查配置并解析成 int：键不存在、值非法，统一成"没有值"一种结果
std::optional<int> get_int_config(const std::vector<std::pair<std::string, std::string>>& cfg,
                                  const std::string& key) {
    for (const auto& [k, v] : cfg)
        if (k == key) return parse_int(v);
    return std::nullopt;
}

int main() {
    std::vector<std::pair<std::string, std::string>> cfg{
        {"sample_rate", "16000"},
        {"frame_ms", "abc"},      // 值非法
        {"channels", "1"},
    };

    if (auto sr = get_int_config(cfg, "sample_rate"))
        std::cout << "sample_rate = " << *sr << '\n';

    std::cout << "frame_ms = " << get_int_config(cfg, "frame_ms").value_or(20)
              << "（默认值）\n";

    if (auto missing = get_int_config(cfg, "gpu_id"); !missing)   // if 带初始化语句
        std::cout << "gpu_id 未配置\n";
}
```

预期输出：

```
sample_rate = 16000
frame_ms = 20（默认值）
gpu_id 未配置
```

`std::from_chars`（`<charconv>`）本身就按"错误码"风格设计——返回结构化绑定能拆出的 `ec`，不抛异常、不分配内存，是解析热路径的首选。把它的结果包成 `optional` 上抛，调用方拿到的是统一的"有或没有"。

## 4. std::variant + std::visit：类型安全的"多选一"

`optional` 只能表达"有/没有"。当结果是**几种类型之一**——比如解码器的输出要么是 PCM 帧、要么是错误信息——用 `std::variant<A, B>`：类型安全的联合，它**始终记得**当前装的是哪个类型（`r.index()`），取错类型在运行期有明确报错，而不是像 C 的 union 那样按内存瞎解释。

```cpp
// 11_ex3_variant.cpp
#include <iostream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

struct PcmFrame {                 // 解码成功：一段 PCM 样本
    std::vector<float> samples;
    int sample_rate;
};

struct DecodeError {              // 解码失败：错误码 + 说明
    std::string message;
    int code;
};

// variant<A, B> = "要么是 A，要么是 B" 的类型安全联合。
// 与 C 的 union 不同：它时刻记得当前存的是哪个类型，取错类型会抛异常/拒绝编译期检查。
using DecodeResult = std::variant<PcmFrame, DecodeError>;

DecodeResult decode_frame(int stream_state) {
    if (stream_state == 0)
        return PcmFrame{{0.1f, -0.2f, 0.3f}, 16000};
    return DecodeError{"码流损坏：缺帧头", 2};
}

int main() {
    for (int state : {0, 1}) {
        DecodeResult r = decode_frame(state);

        // holds_alternative：运行时问一句"现在装的是哪种类型？"
        if (std::holds_alternative<PcmFrame>(r))
            std::cout << "index = " << r.index() << "（PcmFrame）\n";
        else
            std::cout << "index = " << r.index() << "（DecodeError）\n";

        // std::visit + 泛型 lambda + if constexpr：一次处理所有备选类型
        std::visit([](auto&& result) {
            using T = std::decay_t<decltype(result)>;
            if constexpr (std::is_same_v<T, PcmFrame>) {
                std::cout << "解码成功: " << result.samples.size() << " 个样本 @ "
                          << result.sample_rate << "Hz\n";
            } else {
                std::cout << "解码失败[" << result.code << "]: "
                          << result.message << '\n';
            }
        }, r);
    }
}
```

预期输出：

```
index = 0（PcmFrame）
解码成功: 3 个样本 @ 16000Hz
index = 1（DecodeError）
解码失败[2]: 码流损坏：缺帧头
```

三个常用接口：`std::holds_alternative<T>(v)` 问"装的是 T 吗"；`std::get<T>(v)` 取出（装错类型抛 `std::bad_variant_access`）；`std::visit(lambda, v)` 对"无论装了什么"统一分派——泛型 lambda 配 `if constexpr`（第 09 篇）给每个备选类型各写一段，**编译器强制你覆盖所有备选**，新增一个错误类型时漏处理的分支会直接编译失败，这是它比"错误码 + if 链"强的根本原因。

（顺带一提 C++23 的 `std::expected<T, E>`：为"值或错误"场景定制的 variant，`tl::expected` 等库在 C++17 下提供同构 API，工程里已广泛使用。）

## 5. 异常基础：throw / try / catch 与栈展开

异常是"另一条退出路径"：错误沿调用链**向上**传播，直到有人接住。C++ 标准异常都以 `std::exception` 为基类，`what()` 给出说明文字：

```cpp
// 11_ex4_exception.cpp
#include <iostream>
#include <stdexcept>
#include <string>

struct Session {                       // RAII 资源守卫（如打开一个音频设备会话）
    explicit Session(std::string name) : name_(std::move(name)) {
        std::cout << "[open] " << name_ << '\n';
    }
    ~Session() { std::cout << "[close] " << name_ << '\n'; }  // 栈展开时也会执行
    void process(int n) {
        if (n <= 0) throw std::invalid_argument("帧长必须为正");
        std::cout << "[process] n = " << n << '\n';
    }
private:
    std::string name_;
};

void pipeline(int n) {
    Session s("device-session");
    s.process(n);
    std::cout << "[pipeline] 完成\n";  // 抛异常时这一行被跳过
}

int main() {
    try {
        pipeline(160);                 // 正常路径
        pipeline(-1);                  // 抛出 std::invalid_argument
        std::cout << "这里永远走不到\n";
    } catch (const std::invalid_argument& e) {  // 先匹配更派生的类型
        std::cout << "[catch] invalid_argument: " << e.what() << '\n';
    } catch (const std::exception& e) {         // 基类兜底，必须写在后面
        std::cout << "[catch] exception: " << e.what() << '\n';
    }
    std::cout << "程序正常结束\n";
}
```

预期输出：

```
[open] device-session
[process] n = 160
[pipeline] 完成
[close] device-session
[open] device-session
[close] device-session
[catch] invalid_argument: 帧长必须为正
程序正常结束
```

前四行是正常路径：`pipeline(160)` 返回时 `s` 正常析构（`[close]`）。盯着第二次调用看懂**栈展开**（stack unwinding）：`pipeline(-1)` 里抛出异常后，`[pipeline] 完成` 被跳过，但**已经构造完成的局部对象 `s` 的析构函数依然被自动调用**（第二个 `[close]` 出现在 `[catch]` 之前）——这就是第 03 篇 RAII 在异常下的回报：只要资源包在 RAII 对象里（`std::vector`、`std::unique_ptr`、`std::fstream`……），异常路径也不会泄漏。这也是"异常安全基本保证"的底线：**不泄漏资源、不变量不被破坏**，成员函数抛异常后对象仍可安全析构。

规则三条：

1. `catch` 按**书写顺序**匹配，派生类 catch 必须写在基类前面，否则基类兜底会把派生类"截胡"（编译器会警告）。
2. 按引用捕获（`const std::exception&`）避免对象切片和多余拷贝。
3. **不要让析构函数抛异常**：栈展开途中再抛会直接 `std::terminate`。

**noexcept 回顾**（第 06 篇）：声明 `noexcept` 的函数抛出异常会直接 `std::terminate`。它的两个作用——移动构造标 `noexcept` 让 `std::vector` 扩容时选移动而非拷贝；同时它是给读者的契约"此函数不产生异常"，也是错误码/optional 风格代码库的常规标注。

## 6. 什么时候用异常，什么时候用 optional/错误码

工程经验，按序自问：

1. **"没有结果"是不是正常情况？** 是（查找设备、查配置、解析用户输入）→ `std::optional`。失败不稀奇就不该走异常这条重路径。
2. **结果是不是"几种类型之一"？** 是（PCM 或错误、解析树或语法错误）→ `std::variant` + `visit`。
3. **这个失败是不是"本层无法处理、必须有人负责"的罕见事故？** 是（配置文件损坏、模型文件缺失、内存耗尽）→ 异常。异常的优势恰恰是**跨层传递不打扰中间层**：十几层调用栈里只有最外层需要 try，中间层一行不用改。
4. **在实时音频回调 / 每帧热循环里吗？** 那里避免抛异常（异常机制的开销和耗时不确定），也避免任何可能抛异常的分配——返回错误码或 optional，调用方立即处理或丢帧。

反模式两个：**用异常做控制流**（循环里靠 catch 退出——又慢又难读）；**吞异常**（`catch (...) {}` 什么都不做——错误被无声抹掉，排查时是灾难）。内部库边界上，常见折中是：底层用错误码/optional，接口层包一层转成异常（或反过来），让调用方按自己的风格消费。

## 7. 实战：同一个解析函数的 optional 版 vs 异常版

把同一逻辑写成两版，并排观察调用方代码的形态差异：

```cpp
// 11_ex5_compare.cpp
#include <charconv>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

// 版本 A：optional——"失败"是正常返回值的一部分，调用方用 if 分支处理。
std::optional<int> parse_int_opt(std::string_view s) {
    int value = 0;
    const char* last = s.data() + s.size();
    auto [ptr, ec] = std::from_chars(s.data(), last, value);
    if (ec != std::errc{} || ptr != last)
        return std::nullopt;
    return value;
}

// 版本 B：异常——失败走另一条退出路径，调用方用 try/catch 处理。
int parse_int_throw(std::string_view s) {
    int value = 0;
    const char* last = s.data() + s.size();
    auto [ptr, ec] = std::from_chars(s.data(), last, value);
    if (ec != std::errc{} || ptr != last)
        throw std::runtime_error("非法整数: " + std::string(s));
    return value;
}

void run(const char* label, const char* input) {
    std::cout << label << " (\"" << input << "\"):\n";

    if (auto v = parse_int_opt(input))
        std::cout << "  optional 版本: " << *v << '\n';
    else
        std::cout << "  optional 版本: 解析失败（nullopt）\n";

    try {
        int v = parse_int_throw(input);      // 先取结果，再打印——避免 throw 打断输出流
        std::cout << "  异常版本: " << v << '\n';
    } catch (const std::exception& e) {
        std::cout << "  异常版本: 抛出异常 —— " << e.what() << '\n';
    }
}

int main() {
    run("采样率配置", "16000");
    run("帧长配置", "20ms");
}
```

预期输出：

```
采样率配置 ("16000"):
  optional 版本: 16000
  异常版本: 16000
帧长配置 ("20ms"):
  optional 版本: 解析失败（nullopt）
  异常版本: 抛出异常 —— 非法整数: 20ms
```

对比要点：

- **正常路径**两版都干净；**失败路径**上 optional 版就是一个 `else`，异常版要多包一层 try/catch——失败越是常态，optional 越省。
- 异常版自带**失败的上下文信息**（"非法整数: 20ms"），optional 只有"没有值"一个比特；要携带原因就得升级成 variant/expected。
- 配置解析这类"失败是常态"的场景，本文选 optional 是有意为之；如果这是模型加载器、失败意味着"整个程序没法启动"，抛异常让最外层 catch 后打日志退出反而更简洁。

## 常见坑

- **不检查就解引用 optional**：`*opt` / `opt->x` 在无值时是未定义行为（不是抛异常）。先用 `if (opt)` 或 `has_value()` 确认；拿不准就 `value_or`。
- **`opt.value()` 抛 `std::bad_optional_access`**：以为 optional "从不抛异常"是误解；不想抛就别用 `.value()`。
- **`std::get<T>` 对 variant 取错类型**：运行期抛 `std::bad_variant_access`。先 `holds_alternative` 或直接 `std::visit`。
- **catch 顺序颠倒**：基类 `catch (const std::exception&)` 写在派生类前面，派生类永远匹配不到（编译器通常给警告，`-Wall` 下可见）。
- **析构函数抛异常**：栈展开途中再抛异常 → `std::terminate`。析构里只做无抛操作。
- **靠 catch 当 goto 用**：异常穿越多层调用有开销且语义是"事故"，不是控制流。
- **吞异常**：`catch (...) {}` 把故障藏起来；至少要记日志或转译成更明确的状态再上抛。

## 小结

- 三流派：错误码（C 传统、最快、易忘查）、异常（跨层传播、要求异常安全）、optional/variant（把"可能失败"写进类型，编译器强制处理）。
- `std::optional<T>`：有值或 `nullopt`；`if (opt)`、`opt->x`、`value_or(默认)` 三板斧；替代哨兵值 -1/nullptr。
- `std::variant<A, B>` + `std::visit`：类型安全的多选一，visit 强制覆盖所有备选；是"值或错误"的 C++17 载体（C++23 的 `std::expected` 是定制版）。
- 异常：`throw` 沿调用链上抛，**栈展开**自动析构沿途局部对象（RAII 兜底）；catch 按序匹配、按引用捕获；析构函数绝不抛异常；`noexcept` 既是优化开关也是契约。
- 选型口诀：常态失败 → optional；多选一 → variant；罕见且必须有人负责的事故 → 异常；实时回调 → 错误码/optional，别抛。

## 自测题

1. `find` 函数用 `-1` 当"找不到"哨兵有什么隐患？optional 如何从根上解决？
2. `std::optional<int> o; std::cout << *o;` 会发生什么？`o.value()` 呢？
3. `std::visit` 的泛型 lambda 里为什么要配 `if constexpr`？换成普通 `if` 会怎样？
4. 栈展开时析构函数被调用吗？这保证了什么？如果析构函数自己也抛异常呢？
5. 音频实时回调（每 10ms 执行一次）里该选哪套错误处理？为什么？
6. optional 版和异常版的 `parse_int`，失败路径和正常路径各发生了什么不同的事？

**答案**

1. `-1` 混在合法下标取值范围内，编译器不强迫检查，忘查就 `v[-1]` 越界，运行时才炸。optional 把"可能没有"写进返回类型：要用值必须先过 `if (opt)` 或显式 `.value()`，忘检查在代码形态上就藏不住。
2. `*o` 在无值时是**未定义行为**（可能崩溃、可能读垃圾，不保证报错）。`o.value()` 有明确定义：抛 `std::bad_optional_access`。
3. visit 会对每个备选类型实例化一次 lambda，`if constexpr` 在编译期把不适用的分支丢弃。换普通 `if`：两个分支都要对两种类型编译，访问错误分支里的成员（如 `DecodeError::code` 对 `PcmFrame`）直接编译失败。
4. 被调用——已构造的局部对象沿调用链逆序析构，保证 RAII 资源（内存、设备句柄、锁）在异常路径也不泄漏，这就是异常安全基本保证。若析构函数在栈展开途中再抛异常，程序直接 `std::terminate`，所以析构必须不抛。
5. 错误码或 optional。回调周期短且不能有不确定耗时：异常机制开销不定，也不该在热路径承担"失败是常态"（缓冲区满/空）的处理成本；用返回值让调用方当场判断、丢帧或重试。
6. optional 版：失败与成功走**同一条返回通道**，失败是返回值的一部分，调用方用一个 `if/else` 分支消费，无额外隐藏路径。异常版：失败走**独立的上抛通道**，跳过当前函数剩余代码沿栈展开向上，直到某个 try/catch 接住；调用方必须显式包 try，且异常携带错误上下文（`what()`），代价是路径外置、开销不确定。

---

到此，模板与错误处理两条线齐了。下一篇 [12-并发编程](12-并发编程.md)——`std::thread`、`std::atomic` 与生产者-消费者（本篇 RingBuffer 的单生产者/单消费者无锁版正是为它铺路）。
