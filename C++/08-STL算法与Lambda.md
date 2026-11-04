# 08-STL算法与Lambda

> **本篇回答的问题**
> - 手写 for 循环明明很简单，为什么社区坚持推荐 `<algorithm>`？
> - lambda 的捕获列表 `[=]`、`[&]`、`[x = 0]` 到底各自捕获了什么？悬空引用怎么来的？
> - `remove_if` 删了元素却没缩小容器，`size()` 怎么不变？erase-remove 惯用法是什么？
> - 实战：对一帧 16kHz 音频算 RMS 能量与峰值；用 sort+lambda 做 NMS 式候选去重。

前置：[07-STL容器与字符串实战](07-STL容器与字符串实战.md)。带"预期输出"的示例都在 `g++ -std=c++17 -Wall -Wextra -pedantic` 下编译运行验证过。

## 1. 为什么用 `<algorithm>` 而不是手写循环

先用 Python 类比：你在 Python 里写 `sum(x*x for x in frame)`、`sorted(cands, key=lambda c: -c.score)`，而不会手写下标循环。C++ 的 `<algorithm>` 就是同一件事，只是语法更啰嗦。

三个硬理由：

1. **意图即名字**。`std::count_if` 一眼可读"按条件计数"；手写 `int n = 0; for (...) if (...) ++n;` 读完循环体才能确认没夹带私货。算法名是给半年后的自己看的注释。
2. **不易写错边界**。手写下标循环最容易错在 `<` 还是 `<=`、空容器、`size()-1` 无符号下溢。标准算法自带正确边界，空区间自动安全。
3. **性能白送**。实现可以对连续内存做向量化（SIMD），`-O2` 下 `std::accumulate` 常常比朴素的朴素循环快。你手写的循环编译器不一定敢优化。

写作姿势：**热路径里的简单累加/查找，一律先想 `<algorithm>` 里有没有现成的**。

## 2. lambda：写在调用点的匿名函数

`<algorithm>` 的几乎每个函数都要一个"策略"参数，lambda 就是为它而生的匿名函数对象。语法全貌：

```cpp
[捕获列表](参数列表) -> 返回类型 { 函数体 }
```

最关键的是**捕获列表**：lambda 函数体里想用外层局部变量，必须先"捕获"——它本质是编译器生成一个匿名类，把捕获的变量存成成员。于是：

```cpp
// 08_ex_lambda.cpp
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    float gate = 0.01f;
    std::vector<float> frame{0.5f, 0.001f, -0.02f, 0.9f, 0.005f};

    // 按值捕获 [gate]：lambda 内部拿到 gate 的一份拷贝，之后改 gate 不影响它
    auto is_loud_by_value = [gate](float s) { return std::fabs(s) > gate; };

    // 按引用捕获 [&gate]：内部用的就是外面的 gate 本体
    auto is_loud_by_ref = [&gate](float s) { return std::fabs(s) > gate; };

    gate = 0.3f;   // 改门槛
    std::cout << "按值捕获（旧门槛 0.01）: "
              << std::count_if(frame.begin(), frame.end(), is_loud_by_value) << '\n';
    std::cout << "按引用捕获（新门槛 0.3）: "
              << std::count_if(frame.begin(), frame.end(), is_loud_by_ref) << '\n';

    // mutable：按值捕获的拷贝在 lambda 内默认是 const，想改它必须加 mutable
    auto tick = [count = 0]() mutable {   // [count = 0] 顺带演示 init capture
        return ++count;
    };
    std::cout << tick() << tick() << tick() << '\n';   // count 存在 lambda 对象里

    // [=] 全部按值可用，但显式列出捕获的写法更可读、更不易踩悬空引用
    int lo = 0, hi = 10;
    auto use_both = [lo, &hi](int x) { return x >= lo && x <= hi; };
    std::cout << "15 in range: " << std::boolalpha << use_both(15) << '\n';
}
```

预期输出：

```
按值捕获（旧门槛 0.01）: 3
按引用捕获（新门槛 0.3）: 2
123
15 in range: false
```

逐项拆解：

- **按值 `[gate]`**：拍快照，lambda 寿命可以超过原变量，代价是拷贝。默认 `operator()` 是 const 的，改拷贝要加 `mutable`（如 `tick`——init capture `[count = 0]` 还能捕获一个表达式新建的变量）。
- **按引用 `[&gate]`**：零拷贝、能同步看到外部变化，但 **lambda 可能活得比变量久**——把按引用 lambda 存起来异步回调（音频回调线程、任务队列），原变量早销毁了，就是经典的悬空引用崩溃。跨线程/异步场景默认按值。
- **`[=]`/`[&]` 默认捕获**：图省事，但一眼看不出用了什么、危险在哪。团队代码里建议显式列出（`[gate]`、`[&frame, gate]`），编译器替你揪出漏网的。

## 3. 常用算法：transform / accumulate / count_if / find_if

一帧 16kHz 音频的几个最基本统计量，正好把三个最常用算法串起来：

```cpp
// 08_ex_rms.cpp
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
    // 一帧 16kHz 音频样本（已归一化到 [-1, 1]）
    const std::vector<float> frame{0.5f, -0.25f, 0.1f, -0.9f, 0.3f, 0.0f};

    // RMS 能量：accumulate 累加平方和，再开根号
    float sum_sq = std::accumulate(frame.begin(), frame.end(), 0.0f,
                                   [](float acc, float s) { return acc + s * s; });
    float rms = std::sqrt(sum_sq / static_cast<float>(frame.size()));

    // 峰值：max_element 按绝对值比较，返回迭代器
    auto peak_it = std::max_element(frame.begin(), frame.end(),
                                    [](float a, float b) {
                                        return std::fabs(a) < std::fabs(b);
                                    });

    // 过零率：相邻样本符号不同的次数（常用的清/浊音特征）
    int crossings = 0;
    for (std::size_t i = 1; i < frame.size(); ++i) {
        if ((frame[i - 1] < 0.0f) != (frame[i] < 0.0f)) ++crossings;
    }

    std::cout << "RMS = " << rms << '\n';
    std::cout << "峰值 = " << *peak_it << "（位于第 "
              << std::distance(frame.begin(), peak_it) << " 个样本）\n";
    std::cout << "过零次数 = " << crossings << '\n';
}
```

预期输出：

```
RMS = 0.451387
峰值 = -0.9（位于第 3 个样本）
过零次数 = 4
```

注意三个细节：`accumulate` 的**初值 `0.0f` 同时决定了累加类型是 float**（写 `0` 就是整数累加，全部取整，经典 bug）；`max_element` 这类"查找"算法返回**迭代器**而非值，要值就解引用、要下标就 `std::distance`；过零率依赖相邻关系，不是独立的逐元素变换，这种情况手写循环完全正当——算法库不是教条。

再看门限类判断和逐元素变换：

```cpp
// 08_ex_gate.cpp
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const std::vector<float> frame{0.5f, 0.001f, -0.02f, 0.9f, 0.005f, -0.7f};
    const float gate = 0.01f;   // 噪声门限

    // count_if：超过门限的样本数（响音占比，可用于 VAD 初判）
    int loud = std::count_if(frame.begin(), frame.end(),
                             [gate](float s) { return std::fabs(s) > gate; });
    std::cout << "响样本 " << loud << '/' << frame.size() << '\n';

    // find_if：第一处超过门限的位置，找不到返回 end()
    auto it = std::find_if(frame.begin(), frame.end(),
                           [gate](float s) { return std::fabs(s) > gate; });
    if (it != frame.end()) {
        std::cout << "首个响样本下标 " << (it - frame.begin())
                  << "，值 " << *it << '\n';
    }

    // all_of / any_of：一句话表达"整段/存在"
    bool has_peak = std::any_of(frame.begin(), frame.end(),
                                [](float s) { return std::fabs(s) > 0.5f; });
    bool all_audible = std::all_of(frame.begin(), frame.end(),
                                   [gate](float s) { return std::fabs(s) > gate; });
    std::cout << "有峰值: " << std::boolalpha << has_peak
              << ", 全部可闻: " << all_audible << '\n';

    // transform：int16 采样归一化到 [-1, 1]
    std::vector<int> raw{16000, -32000, 8000};
    std::vector<float> norm(raw.size());
    std::transform(raw.begin(), raw.end(), norm.begin(),
                   [](int s) { return static_cast<float>(s) / 32768.0f; });
    for (float s : norm) std::cout << s << ' ';
    std::cout << '\n';
}
```

预期输出：

```
响样本 4/6
首个响样本下标 0，值 0.5
有峰值: true, 全部可闻: false
0.488281 -0.976562 0.244141 
```

`transform` 相当于 Python 的 `[f(x) for x in seq]`，注意输出区间要提前开好（`norm(raw.size())`，这里用 `resize` 语义构造）；`all_of`/`any_of` 语义即名字，空区间上 `all_of` 返回 true（"没有反例"）、`any_of` 返回 false。

## 4. sort + 自定义比较器：NMS 式候选去重

语音检测流水线常这样收尾：滑窗扫出一堆重叠候选（片段 + 置信度），按分数排序，然后贪心保留——与已保留片段重叠过多的丢弃。这就是目标检测里 NMS 的思想，`sort` 的自定义比较器正好登场：

```cpp
// 08_ex_nms.cpp
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

// 一段候选：检测出的语音片段（起点/终点毫秒 + 置信度）
struct Segment {
    std::string label;
    int start_ms = 0;
    int end_ms = 0;
    float score = 0.0f;
};

int main() {
    std::vector<Segment> cands{
        {"speech", 100, 900, 0.62f},
        {"speech", 150, 850, 0.91f},
        {"speech", 120, 400, 0.45f},
        {"music",  300, 700, 0.88f},
        {"noise",  500, 600, 0.30f},
        {"speech", 2000, 2600, 0.75f},
    };

    // 1. 按分数降序；分数相同按起点升序（结果确定，方便排查）
    std::sort(cands.begin(), cands.end(),
              [](const Segment& a, const Segment& b) {
                  if (a.score != b.score) return a.score > b.score;
                  return a.start_ms < b.start_ms;
              });

    // 2. 贪心保留：与已保留片段时间重叠超过一半的丢弃（NMS 思想）
    std::vector<Segment> kept;
    for (const Segment& c : cands) {
        bool suppressed = false;
        for (const Segment& k : kept) {
            int overlap = std::min(c.end_ms, k.end_ms) -
                          std::max(c.start_ms, k.start_ms);
            int len = std::min(c.end_ms - c.start_ms, k.end_ms - k.start_ms);
            if (overlap > 0 && overlap * 2 > len) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) kept.push_back(c);
    }

    std::cout << "保留的片段:\n";
    for (const Segment& s : kept) {
        std::cout << "  " << s.label << " [" << s.start_ms << ", "
                  << s.end_ms << ") score=" << s.score << '\n';
    }
}
```

预期输出：

```
保留的片段:
  speech [150, 850) score=0.91
  speech [2000, 2600) score=0.75
```

比较器三个规矩：参数用 `const&`（拷贝结构体白花钱）；**"严格弱序"**——a 等于 b 时必须都返回 false，写成 `a.score >= b.score` 是未定义行为，等值时给个次级键（起点）最稳妥；`std::sort` 不保证相等元素原顺序，要稳定用 `std::stable_sort`。

## 5. erase-remove 惯用法：最著名的"陷阱"

需求：删掉一帧里所有静音样本。直觉写法是 `std::remove_if(v, pred)`，但它有反直觉的行为——**`remove_if` 根本不知道容器是谁，改不了 size**。它只做"搬移"：把保留的元素向前挪、压缩到头部，返回新逻辑末尾的迭代器；尾部留下的是"垃圾残骸"。

```cpp
// 08_ex_erase.cpp
#include <algorithm>
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8};

    // remove_if 只做"搬移"：把该删的挪到尾部，返回新逻辑末尾迭代器。
    // 它既不知道容器是谁，也没有权限改容器大小——size() 不变！
    auto new_end = std::remove_if(v.begin(), v.end(),
                                  [](int x) { return x % 2 == 0; });
    std::cout << "remove_if 后 size=" << v.size()      // 还是 8
              << "，尾部残留: ";
    for (auto it = new_end; it != v.end(); ++it) std::cout << *it << ' ';
    std::cout << '\n';

    // 惯用法：erase 接手，真正删除
    v.erase(new_end, v.end());
    std::cout << "erase 后 size=" << v.size() << ": ";
    for (int x : v) std::cout << x << ' ';
    std::cout << '\n';

    // C++17 也可以两步合成一条表达式（易读性取舍自己定）：
    std::vector<int> w{1, 2, 3, 4, 5, 6};
    w.erase(std::remove_if(w.begin(), w.end(),
                           [](int x) { return x < 4; }),
            w.end());
    for (int x : w) std::cout << x << ' ';
    std::cout << '\n';
}
```

预期输出：

```
remove_if 后 size=8，尾部残留: 5 6 7 8 
erase 后 size=4: 1 3 5 7 
4 5 6 
```

记住口诀：**remove 不删，erase 才删；俩得手拉手**。设计成这样是因为算法操作迭代器区间，不接触容器——这也是 C++ 标准库的通用约定（`unique` 同理）。顺带一提，`remove_if` 的搬移对 vector 来说是 O(n) 且只过一遍数据，比"逐个 erase 中间元素"（每次 O(n)，整体 O(n²)）高效得多。

## 6. `<numeric>`：iota 与 partial_sum

数值算法和音频是绝配——前缀和就是"累计能量包络"：

```cpp
// 08_ex_numeric.cpp
#include <iostream>
#include <numeric>
#include <vector>

int main() {
    // iota：生成 0,1,2,... 常用于预生成索引表（倒放、随机打乱前的初值）
    std::vector<int> idx(5);
    std::iota(idx.begin(), idx.end(), 0);
    std::cout << "iota: ";
    for (int i : idx) std::cout << i << ' ';
    std::cout << '\n';

    // partial_sum：前缀和。O(n) 求任意区间的累计能量，做能量包络
    std::vector<float> frame_energy{0.1f, 0.4f, 0.2f, 0.9f, 0.3f};
    std::vector<float> prefix(frame_energy.size());
    std::partial_sum(frame_energy.begin(), frame_energy.end(), prefix.begin());
    std::cout << "prefix: ";
    for (float e : prefix) std::cout << e << ' ';
    std::cout << '\n';
    // 区间 [1, 3) 的能量 = prefix[2] - prefix[0]
    std::cout << "帧 1..2 总能量 = " << prefix[2] - prefix[0] << '\n';

    // partial_sum 也接受自定义运算（这里演示连乘：增益级联）
    std::vector<float> gains{0.5f, 2.0f, 0.1f, 10.0f};
    std::vector<float> cum(gains.size());
    std::partial_sum(gains.begin(), gains.end(), cum.begin(),
                     [](float a, float b) { return a * b; });
    std::cout << "级联增益: ";
    for (float g : cum) std::cout << g << ' ';
    std::cout << '\n';
}
```

预期输出：

```
iota: 0 1 2 3 4 
prefix: 0.1 0.5 0.7 1.6 1.9 
帧 1..2 总能量 = 0.6
级联增益: 0.5 1 0.1 1 
```

前缀和的妙处：建一次 O(n) 表，之后**任意区间和都是 O(1)**（`prefix[r] - prefix[l-1]`）。算一段语音的总能量、能量包络、滑动平均，都可以走这条路。

## 7. C++17 并行算法：一句话认识

C++17 给多数算法加了执行策略重载：`std::sort(std::execution::par, v.begin(), v.end());` 即声明"允许并行执行"。思想上值得知道——标准库从此有了表达"这段计算可以并行"的语言级入口，音频多路处理、大批量特征归一化是潜在受益者。但落地有前提：GCC 需要链接 TBB（`-ltbb`，Ubuntu 下装 `libtbb-dev`），且算法必须真的无数据竞争才安全。**本篇只介绍不要求运行**，等 12 篇并发编程打好基础再回来用它。

## 常见坑

1. **`accumulate` 初值写成 `0`**：整数初值让整个累加按整数进行，`0.5 + 0.5` 会变 0。初值写 `0.0f` 并与元素类型一致。
2. **比较器用 `>=`/`<=`**：破坏严格弱序，`std::sort` 可能越界崩溃。相等元素让比较器返回 false。
3. **以为 `remove_if`/`unique` 缩小了容器**：它们只搬移元素。必须 `v.erase(..., v.end())` 收尾。
4. **按引用捕获的 lambda 跨越了变量生命周期**：存进回调队列、传给别的线程，原变量已销毁——悬空引用。异步场景按值捕获。
5. **`transform` 输出区间没开够**：往 `begin()` 写而容器是空的，未定义行为。先 `resize`/构造到目标大小，或用 `back_inserter`。
6. **在 `sort` 的比较器里比较不相干的浮点 NaN**：NaN 和任何值比较都是 false，可能破坏严格弱序；数据可能含 NaN 时先清洗。
7. **空区间上的 `front()`/`back()`/解引用 `begin()`**：先 `empty()` 判断；`find_if` 的返回值只能和 `end()` 比，不能直接解引用。

## 小结

- `<algorithm>` 的价值是**意图即名字 + 边界不写错 + 白送的性能**；先查算法库，再考虑手写循环。
- lambda 是"捕获了变量的匿名函数对象"：按值拍快照（加 `mutable` 才能改）、按引用共享原变量但可能悬空；跨线程/异步一律按值，日常显式列出捕获。
- 高频算法背五个：`accumulate`（初值定类型）、`count_if`/`find_if`、`transform`（输出区间先开好）、`sort`+比较器（严格弱序）、`remove_if`+`erase`（remove 不删 erase 删）。
- 判断类三兄弟 `all_of`/`any_of`/`none_of` 语义即名字；`<numeric>` 的 `iota` 造索引、`partial_sum` 造前缀和（任意区间和 O(1)）。
- C++17 并行算法（`std::execution::par`）是"可并行"的标准表达，GCC 下需 TBB，本篇只介绍。

## 自测题

1. `std::accumulate(v.begin(), v.end(), 0)` 对 `vector<float>` 求和会得到什么？为什么？
2. 下面比较器错在哪？`[](const Segment& a, const Segment& b) { return a.score >= b.score; }`
3. 只调用 `std::remove_if` 之后，`v.size()` 变吗？`v.end()` 之后的"残骸"能访问吗？
4. 一个 lambda 按引用捕获了局部变量 `gate`，被存进任务队列在 1 秒后执行。会发生什么？怎么修？
5. 已有每帧能量 `std::vector<float> e`，如何用一次 O(n) 预处理让"第 i 到第 j 帧总能量"变成 O(1) 查询？

**答案**

1. 得到整数 0 附近的错误结果（实际是各元素截断成 int 的累加）：初值 `0` 的类型 int 决定了累加全程按整数进行。应写 `0.0f`。
2. 相等元素（`a.score == b.score`）时返回 true，违反严格弱序（等价关系不满足传递性），`std::sort` 未定义行为。相等时应返回 false，可加次级键 `return a.score > b.score;`。
3. 不变（算法只操作迭代器，不知道容器）。尾部到旧 `end()` 之间的值是搬移后的残骸，**可以访问（仍指向有效元素）但语义上已不属于逻辑内容**；实际工程中直接 `erase(new_end, v.end())` 掉。
4. 悬空引用：原变量早已析构，1 秒后 lambda 读它就是未定义行为（通常崩溃或读到垃圾值）。改为按值捕获 `[gate]`。
5. `std::partial_sum` 建前缀和表 `p`（`p[0]=0`，`p[k]` 为前 k 帧之和），区间 `[i, j)` 总能量即 `p[j] - p[i]`，O(1)。
