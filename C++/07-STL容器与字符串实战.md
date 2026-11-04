# 07-STL容器与字符串实战

> **本篇回答的问题**
> - vector、map、unordered_map…… 容器这么多，语音引擎里各用在什么地方？
> - `size` 和 `capacity` 什么区别？为什么 `push_back` 会让迭代器失效？
> - `map` 和 `unordered_map` 怎么选、插入/查找/删除分别用哪个 API？
> - `std::array` 比 C 数组好在哪？`std::string` 截取/查找/数字互转怎么写？

前置：[06-移动语义与完美转发](06-移动语义与完美转发.md)。带"预期输出"的示例都在 `g++ -std=c++17 -Wall -Wextra -pedantic` 下编译运行验证过。

## 1. 先有地图：常用容器总览

从 Python 过来可以这样对应：`vector` ≈ list，`map` ≈ dict（有序版），`unordered_map` ≈ dict 本尊，`set` ≈ set。选容器的第一问永远是"**我怎么访问它**"——答案直接锁定容器。

| 容器 | 底层结构 | 随机访问 | 插入/删除（中间） | 查找 | 典型场景 |
|---|---|---|---|---|---|
| `vector` | 动态数组，连续内存 | O(1) | O(n) | O(n) | 采样缓冲、特征向量、**默认首选** |
| `array` | 定长数组，连续内存 | O(1) | 不适用 | O(n) | 固定长度：MFCC 的 13 维系数 |
| `deque` | 分段连续数组 | O(1) | 头尾 O(1)，中间 O(n) | O(n) | 环形队列：滑窗、流式分帧 |
| `list` | 双向链表 | 不支持 | O(1)（有迭代器时） | O(n) | 极少用；频繁中间插删且要稳定地址 |
| `map` | 红黑树（有序） | 不支持 | O(log n) | O(log n) | 要按键有序遍历：词表、配置 |
| `unordered_map` | 哈希表 | 不支持 | 均摊 O(1) | 均摊 O(1) | 只求快：流 ID -> 流对象 |
| `set` / `unordered_set` | 红黑树 / 哈希表 | 不支持 | O(log n) / 均摊 O(1) | 同左 | 去重、判断"在不在" |

**默认 `vector`，有理由才换**——连续内存对缓存友好，语音这种逐样本计算的热点代码尤其吃这个红利。

## 2. vector 深入：size、capacity 与扩容

`size` 是"现在装了多少"，`capacity` 是"已经申请了多少空间"。装满了再 `push_back`，vector 会申请一块更大的新内存、把元素**搬**过去、释放旧的——这就是扩容。搬家之后，所有指向旧内存的迭代器和指针**全部失效**。

```cpp
// 07_ex_vector_capacity.cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<float> frame;
    std::size_t old_cap = frame.capacity();   // 初始都是 0
    for (int i = 1; i <= 6; ++i) {
        frame.push_back(static_cast<float>(i));
        if (frame.capacity() != old_cap) {
            std::cout << "push 第 " << i << " 个后扩容: size=" << frame.size()
                      << " capacity=" << frame.capacity() << '\n';
            old_cap = frame.capacity();
        }
    }

    frame.reserve(1024);
    std::cout << "reserve(1024) 后: size=" << frame.size()
              << " capacity=" << frame.capacity() << '\n';
}
```

预期输出（libstdc++ 按翻倍增长，标准只保证几何级数增长，具体倍数看实现）：

```
push 第 1 个后扩容: size=1 capacity=1
push 第 2 个后扩容: size=2 capacity=2
push 第 3 个后扩容: size=3 capacity=4
push 第 5 个后扩容: size=5 capacity=8
reserve(1024) 后: size=6 capacity=1024
```

两个实用结论：**已知大小时先 `reserve`**——音频一帧固定 512 个样本，一次申请到位，避免反复搬家（注意它只改 capacity 不改 size，装数据仍用 `push_back`/`resize`）；`shrink_to_fit` 可归还多余容量，但只是建议不保证生效，别在热路径频繁调用。

### 失效规则速查

| 操作 | 迭代器/指针 |
|---|---|
| `push_back` 引起扩容 | **全部失效**；不扩容则只有 end 失效 |
| `insert`/`erase` | 插入/删除点**之后**全部失效 |
| `reserve`/`resize`/`clear` | 全部失效 |
| `operator[]` / 范围 for | 当次访问安全，前提是期间没有发生修改 |

```cpp
// 07_ex_invalidate.cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<float> v{1.0f, 2.0f, 3.0f};
    v.reserve(3);
    float* p = &v[0];                 // 指向第一个元素
    std::cout << "扩容前: *p = " << *p << '\n';

    v.push_back(4.0f);                // 超出 capacity，触发扩容搬迁
    std::cout << "capacity 是否变化: "
              << (v.capacity() > 3 ? "是（元素已搬到新内存）" : "否") << '\n';

    // 此刻 p 已经悬空，*p 是未定义行为——真实代码严禁再用 p
    p = &v[0];                        // 正确做法：扩容后重新获取
    std::cout << "重新取指针后: *p = " << *p << '\n';
}
```

预期输出：

```
扩容前: *p = 1
capacity 是否变化: 是（元素已搬到新内存）
重新取指针后: *p = 1
```

最阴险的是这段代码"看起来能跑"——旧内存刚释放往往还读得出旧值，数据量一大就随机崩溃。规则记死：**修改了 vector 之后，之前拿的迭代器/指针一律作废重取**。

## 3. map vs unordered_map：选型与常用 API

两者 API 几乎一样，差别在底层：`map` 是红黑树，**按键有序**，操作 O(log n)；`unordered_map` 是哈希表，**无序**，均摊 O(1)。怎么选：

- 默认（只要快、不关心顺序）：`unordered_map`。比如流 ID 到流对象的映射，每来一帧都要查一次，哈希表最省。
- 需要**按 key 有序遍历**（按字母序输出词表、按时间戳回放事件）或区间查询（`lower_bound`）：`map`。key 是自定义类型时，`map` 只需一个 `<`，`unordered_map` 要自己写哈希——前者省事。

常用 API（`unordered_map` 同名语义一致）：插入 `insert`（已存在**不**覆盖）、`insert_or_assign`（存在则覆盖，C++17）、`try_emplace`（不存在才构造，C++17）；查找用 `find`，删除用 `erase`。**判断"有没有"用 `find`，查询别用 `operator[]`**——它查不到会静默插入默认值（第 8 节实测给你看）。vector 侧高频 API：`push_back`/`emplace_back`、`pop_back`、`insert`、`erase`（按下标 `v.erase(v.begin() + i)`）、`resize`（改 size）、`reserve`（改 capacity）、`clear`、`front`/`back`/`data`。完整可运行演示见第 8 节。

## 4. std::array：带长度信息的 C 数组

```cpp
// 07_ex_array.cpp
#include <algorithm>
#include <array>
#include <iostream>

// C 数组作参数会退化为指针，拿不到长度
void c_version(const float* buf) {
    std::cout << "C 数组版本: sizeof(buf)=" << sizeof(buf)
              << "（只是指针大小，长度信息丢了）\n";
}

// 对比：std::array 按引用传递不退化，size() 可用——用模板函数即可带长度接住
template <std::size_t N>
void modern_version(const std::array<float, N>& buf) {
    std::cout << "std::array 版本: size()=" << buf.size() << '\n';
}

int main() {
    std::array<float, 3> win{0.5f, 1.0f, 0.5f};
    modern_version(win);
    c_version(win.data());             // 需要裸指针时才 .data()

    // 能直接参与 STL 算法
    std::sort(win.begin(), win.end());
    std::cout << "排序后: ";
    for (float x : win) std::cout << x << ' ';
    std::cout << '\n';

    // 拷贝就是真正的元素拷贝：other 改动不影响 win（值语义）
    std::array<float, 3> other = win;
    other[0] = 99.0f;
    std::cout << "win[0]=" << win[0] << " other[0]=" << other[0] << '\n';
}
```

预期输出：

```
std::array 版本: size()=3
C 数组版本: sizeof(buf)=8（只是指针大小，长度信息丢了）
排序后: 0.5 0.5 1 
win[0]=0.5 other[0]=99
```

和 C 数组 `float win[3]` 相比，`std::array` 栈上分配、零额外开销，但补上三块短板：知道自己的 `size()`、可以拷贝赋值（值语义）、可以直接返回和传参不退化。**长度编译期固定的数据（滤波器系数、MFCC 的 13 维），一律 `std::array`**；把它写成函数返回值也是安全的（C 数组做不到）。

## 5. std::string 常用操作

```cpp
// 07_ex_string.cpp
#include <iostream>
#include <string>

// C++20 才有 s.starts_with(p) / s.ends_with(p)，C++17 自己写：
bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() &&
           s.compare(0, prefix.size(), prefix) == 0;
}
bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int main() {
    std::string path = "/data/voice/test_16k.wav";

    std::size_t slash = path.find_last_of('/');
    std::string name = path.substr(slash + 1);          // 从位置截到末尾
    std::cout << "文件名: " << name << '\n';

    std::size_t dot = name.find('.');
    std::string stem = name.substr(0, dot);             // [0, dot)
    std::cout << "stem: " << stem << '\n';

    std::cout << "是 wav / 在 data 目录: " << std::boolalpha
              << ends_with(name, ".wav") << " / " << starts_with(path, "/data/") << '\n';

    std::string s = "noise gate";
    s.replace(s.find("noise"), 5, "vad");               // 替换子串
    std::cout << "replace 后: " << s << '\n';
}
```

预期输出：

```
文件名: test_16k.wav
stem: test_16k
是 wav / 在 data 目录: true / true
replace 后: vad gate
```

要点：`substr(pos, len)` 的 len 省略表示"到末尾"；`find` 系列找不到返回 `std::string::npos`（`size_t` 最大值），**判断只能和 npos 比较**。两个易错细节：`s.substr(s.find('x'))` 找不到时传入 npos，而 `substr` 要求 `pos <= size()`，会抛 `std::out_of_range`；而 `path.substr(path.find_last_of('/') + 1)` 这种 `+1` 写法在找不到时 `npos+1` 溢出为 0，恰好返回整个字符串——惯用法安全，但要明白为什么。另外 `starts_with`/`ends_with` 是 C++20 的，C++17 用上面的 `compare` 写法。

## 6. 字符串与数字互转

```cpp
// 07_ex_numstr.cpp
#include <charconv>
#include <iostream>
#include <string>
#include <system_error>

int main() {
    // 数字 -> 字符串
    int sr = 16000;
    float energy = 0.03125f;
    std::string tag = "sr=" + std::to_string(sr) + " energy=" + std::to_string(energy);
    std::cout << tag << '\n';   // 注意 float 会转成 6 位定点小数

    // 字符串 -> 数字
    std::cout << "conf=" << std::stod("0.873") << '\n';

    // C++17 的 from_chars：不抛异常、不依赖 locale、无开销
    const char* text = "-0.5 0.25";
    float a = 0.0f;
    auto [ptr, ec] = std::from_chars(text, text + 4, a);
    if (ec == std::errc()) {
        std::cout << "a=" << a << " 停在偏移 " << (ptr - text) << '\n';
    }
}
```

预期输出：

```
sr=16000 energy=0.031250
conf=0.873
a=-0.5 停在偏移 4
```

三档工具按需选：`to_string`/`stoi`/`stod` 最常用，但 `stoi` 系列解析失败会**抛异常**（`stoi("16000Hz", &pos)` 的位置参数可告诉你消耗了几个字符），`to_string` 的浮点格式固定为 `%f`；`from_chars`（`<charconv>`，C++17）不抛异常、不认 locale、解析到哪一目了然，适合逐字段解析配置行和特征文件，是高性能路径的首选。

## 7. 实战：统计词频并排序输出

处理 ASR 日志、统计解码候选时，"分词 -> 计数 -> 排序"是标准动作。`map` 负责计数，但它只能按 key 排序；想按出现次数排，得把结果搬到 `vector<pair>` 里再 `sort`：

```cpp
// 07_ex_wordfreq.cpp
#include <algorithm>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

int main() {
    const std::string text = "the frame is silent and the frame is short";

    // 1. 分词并统计：map 的 value 是 int，operator[] 会自动初始化为 0
    std::map<std::string, int> freq;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        ++freq[word];
    }

    // 2. map 本身按 key 字典序排列
    std::cout << "按字母序（map 天然有序）:\n";
    for (const auto& [w, n] : freq) {
        std::cout << "  " << w << ": " << n << '\n';
    }

    // 3. 想按出现次数排序：map 不支持，搬到 vector<pair> 里再 sort
    std::vector<std::pair<std::string, int>> by_count(freq.begin(), freq.end());
    std::sort(by_count.begin(), by_count.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;  // 次数多的在前
        return a.first < b.first;                              // 并列按字母序
    });

    std::cout << "按次数降序:\n";
    for (const auto& [w, n] : by_count) {
        std::cout << "  " << w << ": " << n << '\n';
    }
}
```

预期输出：

```
按字母序（map 天然有序）:
  and: 1
  frame: 2
  is: 2
  short: 1
  silent: 1
  the: 2
按次数降序:
  frame: 2
  is: 2
  the: 2
  and: 1
  short: 1
  silent: 1
```

## 8. 音频场景：帧缓冲与多路流管理

引擎里最典型的两个容器用法：**一帧音频是 `vector<float>`**（长度 = 采样点数，连续内存方便逐样本计算）；**多路音频流是 `map`/`unordered_map`**（流名/会话 ID 到流状态的映射）。第 3 节的 API 四件套在这里一次演示：

```cpp
// 07_ex_streams.cpp
#include <iostream>
#include <map>
#include <string>
#include <vector>

struct AudioStream {
    int sample_rate = 16000;
    std::vector<float> samples;      // 该路的采样
};

int main() {
    std::map<std::string, AudioStream> streams;   // 流名 -> 流

    // 插入：emplace 原地构造；try_emplace 仅在 key 不存在时才构造
    streams.emplace("mic", AudioStream{16000, {0.1f, -0.2f, 0.3f}});
    streams.try_emplace("ref", AudioStream{16000, {0.0f, 0.0f}});
    streams.try_emplace("mic", AudioStream{8000, {9.0f}});   // 已存在，被忽略
    streams.insert_or_assign("ref", AudioStream{16000, {0.5f}});  // 存在则覆盖

    // 查找：用 find，别用 operator[] —— 查不到会静默插入空流
    if (auto it = streams.find("mic"); it != streams.end()) {
        std::cout << "mic 有 " << it->second.samples.size() << " 个采样\n";
    }

    for (const auto& [name, st] : streams) {     // 按 key 字典序遍历
        std::cout << name << ": " << st.sample_rate << "Hz, "
                  << st.samples.size() << " samples\n";
    }

    // 挂断一路
    streams.erase("ref");
    std::cout << "erase 后剩 " << streams.size() << " 路\n";

    // operator[] 的坑：拼错流名会凭空造出一个空流
    (void)streams["mci"];            // 拼错了！
    std::cout << "拼错流名后 size=" << streams.size()   // 变成 2 了
              << "，空流 sample_rate=" << streams["mci"].sample_rate << '\n';
}
```

预期输出：

```
mic 有 3 个采样
mic: 16000Hz, 3 samples
ref: 16000Hz, 1 samples
erase 后剩 1 路
拼错流名后 size=2，空流 sample_rate=16000
```

流式处理中 `deque<float>` 也很常用：麦克风不断从尾部推入新样本、分帧逻辑从头部取固定窗口，两端都是 O(1)，比 `vector` 头部 `erase` 的 O(n) 划算（13 篇实战的环形缓冲区是它的手动版）。

## 常见坑

1. **扩容后还用旧迭代器/指针**：`for (auto it = v.begin(); ...; ) { v.push_back(x); ++it; }` 是未定义行为。要么先 `reserve`，要么用下标，要么改完再遍历。
2. **`map[key]` 当查询用**：key 不存在会**插入**默认值。只查询用 `find` 或 `at`（不存在时抛异常）。另注意 `vector<bool>` 是位压缩特化，`v[i]` 返回代理对象，取引用会编译错。
3. **在范围 for 中往容器里加元素**：修改 size 可能使迭代器失效，同样未定义行为。
4. **`find` 返回值没判 npos 就继续运算**：`s.substr(s.find('x'))` 找不到时传入 npos，直接抛 `std::out_of_range`；`path.substr(path.find_last_of('/') + 1)` 的 `+1` 写法则恰好安全（`npos+1` 溢出为 0）。
5. **erase 之后的循环迭代器**：`c.erase(it);` 之后 `++it` 操作的是已失效的 it。正确写法 `it = c.erase(it);` 或用 remove+erase 惯用法（08 篇细讲）。

## 小结

- 容器选型一句话：**默认 `vector`；要 key 有序用 `map`，只求快用 `unordered_map`；定长数据用 `std::array`；两端进出用 `deque`**。
- `size` 是内容量，`capacity` 是已申请空间；扩容 = 搬家 = 迭代器/指针全失效，已知规模先 `reserve`。
- map 四件套：`insert`（不覆盖）/ `insert_or_assign`（覆盖）/ `try_emplace`（不存在才构造）/ `find`+`erase`；查询别用 `operator[]`。字符串：`substr`/`find`/`replace`，数字互转 `to_string`、`stoi/stod`，高性能解析用 `from_chars`，`starts_with`/`ends_with` 在 C++17 用 `compare` 手写。

## 自测题

1. 为什么"查询 map 里某个 key 是否存在"不推荐 `m[key]` 或 `m.count(key)`？更好的写法是什么？
2. `std::string s = "abc"; std::size_t p = s.find('z');` 之后 `s.substr(p)` 会发生什么？那 `s.substr(0, p)` 呢？
3. 想按"出现次数"给 `map<string,int>` 排序，为什么不能直接对 map sort？套路是什么？
4. `std::array<float, 13>` 传参为什么推荐 `const std::array<float, 13>&` 而不是按值传？C 数组 `float[13]` 传参又会退化成什么？

**答案**

1. `m[key]` 在 key 不存在时会**静默插入**默认值；`count` 正确但 `find` 一次查找就能同时拿到"存在性"和元素本身。推荐 `if (auto it = m.find(k); it != m.end()) { ... }`。
2. `p == npos`，而 `substr(pos)` 要求 `pos <= size()`，`npos` 是 `size_t` 最大值必然超出，所以 `substr(p)` **抛出 `std::out_of_range`**。`substr(0, p)` 则不抛：count 会被钳制到 `min(count, size() - pos)`，`npos` 被钳成 `size()`，返回整个 `"abc"`。
3. map 的元素是 `pair<const string, int>`，key 有 const 修饰且 map 迭代器不支持随机访问，`std::sort` 无法工作；套路是拷贝到 `vector<pair<string,int>>` 再用 lambda 按 `second` 排序（并列时给次级键保证输出确定）。
4. 按值传会拷贝全部 13 个 float，引用只传一个地址；长度是模板参数的一部分，引用传递还能在函数内用 `size()`。C 数组传参退化为 `float*`，长度信息丢失（`sizeof` 变成指针大小）。
