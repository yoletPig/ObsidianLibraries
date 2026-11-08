# 12 - 实战：用 C++ 写一个高性能 JSON 解析器

## 本篇目标

- 把全系列串起来：模板（第 02 章）、SIMD（第 03 章）、内存序（第 04/05 章）、性能工程（第 06 章）、模糊测试（衔接《测试》系列 09），产出一个 ~600 行、可测可跑的完整项目。
- 理解 JSON 解析的核心设计决策：单遍 vs 两遍、递归下降、`string_view` 零拷贝、arena。
- 用**诚实的三回合优化**走一遍性能工程流程——包括"直觉优化没用上"和"分配计数推翻假设"两个反面案例。
- 学会公平对比 nlohmann/json 与 simdjson 的方法学，以及"堆布局让同一二进制 ±35%"的基准陷阱。

本章所有代码在 g++ 11.4 / Ubuntu 22.04（i5-12600K）实测编译运行；数字为该环境实测，绝对值因机器而异。

## 1. 需求定义

支持 **RFC 8259 的常用子集**：

- 六种值：`null`、`true`/`false`、数字（double 承载）、字符串、数组、对象；
- 字符串转义：`\" \\ \/ \b \f \n \r \t \uXXXX`（`\u` 只做 BMP 的 UTF-8 编码，代理对合并不支持——明确写进文档）；
- 严格模式细节：数字前导零/小数点后必须跟数字（[RFC 8259 §6]），控制字符必须转义；
- 明确**不做**：大整数（>2^53 精度丢失，需要 i64 路径是另一个项目）、键重复检测、流式解析。

输出形态选 **DOM（整棵树）**：调用方拿到 `Value` 随意遍历。放弃 SAX/on-demand 是有意识的取舍——教学目标是"完整可用的解析器"而不是打榜（第 7 节会看到这个取舍的代价）。

## 2. 设计决策：单遍 vs 两遍

高性能 JSON 解析器有两种主流架构：

| 架构 | 流程 | 代表 | 特点 |
| --- | --- | --- | --- |
| 单遍 | 一边扫一边建树（递归下降） | nlohmann、RapidJSON | 简单、错误即停、DOM 友好 |
| 两遍 | 第一遍 SIMD 扫描预解析结构（找引号/花括号位图），第二遍按需取值 | simdjson | 吞吐 GB/s 级，但要先扫全输入 |

我们选**单遍递归下降**：教学清晰、代码量可控，且单遍对小文档的"首 token 延迟"更好（simdjson 的第一遍要扫完全部输入）。simdjson 的优势来自第二遍的"按需"（on-demand）——只解析你实际访问的字段，很多字段根本不建对象，这是它吞吐量级的真正来源（见第 7 节）。

## 3. 核心实现：递归下降解析器

完整头文件如下（`json.hpp`，288 行；测试与基准代码见第 6/8 节）：

```cpp
// json.hpp —— 教学版高性能 JSON 解析器（RFC 8259 子集）
// 设计要点：单遍递归下降 + string_view 零拷贝 + arena 承接字符串
#pragma once
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace minijson {

// ---------- 错误类型：带行列位置 ----------
struct ParseError {
    std::string_view msg; // 错误描述
    size_t offset;        // 在输入中的字节偏移
    size_t line, col;     // 1-based 行列号
};

// ---------- AST：对象用 vector<pair> 而不是 map（更快，允许重复键）----------
class Value;
using Array  = std::vector<Value>;
using Member = std::pair<std::string_view, Value>; // 键是原文/arena 里的 string_view
using Object = std::vector<Member>;

class Value {
public:
    using Rep = std::variant<std::nullptr_t, bool, double, std::string_view, Array, Object>;
    Rep rep;
    Value() : rep(nullptr) {}
    Value(std::nullptr_t) : rep(nullptr) {}
    Value(bool b) : rep(b) {}
    Value(double d) : rep(d) {}
    Value(std::string_view s) : rep(s) {}
    Value(Array a) : rep(std::move(a)) {}
    Value(Object o) : rep(std::move(o)) {}
    bool is_null()   const { return std::holds_alternative<std::nullptr_t>(rep); }
    bool is_bool()   const { return std::holds_alternative<bool>(rep); }
    bool is_number() const { return std::holds_alternative<double>(rep); }
    bool is_string() const { return std::holds_alternative<std::string_view>(rep); }
    bool is_array()  const { return std::holds_alternative<Array>(rep); }
    bool is_object() const { return std::holds_alternative<Object>(rep); }
    const Object& as_object() const { return std::get<Object>(rep); }
    const Array&  as_array()  const { return std::get<Array>(rep); }
    double           as_number() const { return std::get<double>(rep); }
    std::string_view as_string() const { return std::get<std::string_view>(rep); }
};

// ---------- Arena：解析期字符串的内存池（第 06 章同款思想）----------
class Arena {
public:
    explicit Arena(size_t chunk = 1 << 20) : chunk_(chunk) {}
    ~Arena() { for (auto& c : chunks_) ::operator delete(c); }
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    char* allocate(size_t n, size_t align = 16) {
        uintptr_t cur = reinterpret_cast<uintptr_t>(cur_);
        uintptr_t base = (cur + align - 1) & ~(uintptr_t)(align - 1); // 对齐
        if (n > static_cast<size_t>(reinterpret_cast<uintptr_t>(end_) - base)) {
            grow(n);                                                   // 先扩块再取新基址
            base = reinterpret_cast<uintptr_t>(cur_);
        }
        cur_ = reinterpret_cast<char*>(base) + n;
        return reinterpret_cast<char*>(base);
    }
private:
    void grow(size_t need) {
        size_t cap = chunk_;
        while (cap < need) cap *= 2;
        char* c = static_cast<char*>(::operator new(cap));
        chunks_.push_back(c);
        cur_ = c;
        end_ = c + cap;
    }
    size_t chunk_;
    std::vector<char*> chunks_;
    char* cur_ = nullptr;
    char* end_ = nullptr;
};

// ---------- 解析器 ----------
class Parser {
public:
    // 输入必须比返回的 Value 活得久（string_view 只引用不拷贝）
    // 出错时返回 null 并填 err_out；约定：错误走 ParseError，不向外抛
    static Value parse(std::string_view input, Arena& arena, ParseError* err_out = nullptr) {
        Parser p(input, arena);
        try {
            Value v = p.parse_value();
            p.skip_ws();
            if (p.pos_ != input.size()) p.fail("文档结束后有多余字符");
            return v;
        } catch (const ParseError& e) {
            if (err_out) *err_out = e;
            return Value{};
        }
    }

private:
    Parser(std::string_view s, Arena& a) : s_(s), arena_(a) {}

    std::string_view s_;
    Arena& arena_;
    size_t pos_ = 0;

    [[noreturn]] void fail(const char* msg) const {
        size_t line = 1, col = 1;                 // 行列号：从头数换行（错误路径不追求快）
        for (size_t i = 0; i < pos_ && i < s_.size(); ++i) {
            if (s_[i] == '\n') { ++line; col = 1; } else ++col;
        }
        throw ParseError{msg, pos_, line, col};   // 内部用异常做回退，边界处转成错误码
    }

    // 跳过空白：常见情形是 0~4 个，先快路径再循环
    void skip_ws() {
        if (pos_ < s_.size()) {
            char c = s_[pos_];
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') return;
            ++pos_;
        }
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos_;
            else break;
        }
    }

    char peek() const { return s_[pos_]; }
    char next() {
        if (pos_ >= s_.size()) fail("输入提前结束");
        return s_[pos_++];
    }
    void expect(char c) {
        if (next() != c) { --pos_; fail("期望字符不匹配"); }
    }

    Value parse_value() {
        skip_ws();
        if (pos_ >= s_.size()) fail("缺少值");
        switch (s_[pos_]) {
            case '{': return parse_object();
            case '[': return parse_array();
            case '"': return Value(parse_string());
            case 't': parse_lit("true",  Value(true));  return Value(true);
            case 'f': parse_lit("false", Value(false)); return Value(false);
            case 'n': parse_lit("null",  Value(nullptr)); return Value(nullptr);
            default:  return Value(parse_number());
        }
    }

    void parse_lit(const char* lit, Value) {
        const size_t n = std::char_traits<char>::length(lit);
        if (s_.compare(pos_, n, lit) != 0) fail("字面量拼写错误");
        pos_ += n;
    }

    Value parse_object() {
        expect('{');
        Object members; members.reserve(8);   // 启发式：小对象最常见（见第 6 节剖析）
        skip_ws();
        if (peek() == '}') { ++pos_; return Value(std::move(members)); }
        while (true) {
            skip_ws();
            std::string_view key = parse_string();
            skip_ws();
            expect(':');
            members.emplace_back(key, parse_value());
            skip_ws();
            char c = next();
            if (c == '}') break;
            if (c != ',') fail("对象成员后应为 , 或 }");
        }
        return Value(std::move(members));
    }

    Value parse_array() {
        expect('[');
        Array items; items.reserve(4);
        skip_ws();
        if (peek() == ']') { ++pos_; return Value(std::move(items)); }
        while (true) {
            items.push_back(parse_value());
            skip_ws();
            char c = next();
            if (c == ']') break;
            if (c != ',') fail("数组元素后应为 , 或 ]");
        }
        return Value(std::move(items));
    }

    // 快路径：不含转义时直接引用原文（零拷贝）
    std::string_view parse_string() {
        expect('"');
        size_t start = pos_;
        while (true) {
            if (pos_ >= s_.size()) fail("字符串未闭合");
            char c = s_[pos_];
            if (c == '"') {
                std::string_view raw = s_.substr(start, pos_ - start);
                ++pos_;                                            // 越过结束引号
                if (raw.find('\\') == std::string_view::npos) return raw;
                return unescape(start - 1, pos_);                  // 有转义：整段交给解码
            }
            if (c == '\\') { pos_ += 2; continue; }                // 转义对整体跳过，\" 不误判
            if (static_cast<unsigned char>(c) < 0x20) fail("字符串中出现未转义控制字符");
            ++pos_;
        }
    }

    // 把 [begin,end) 内含转义的原文解码进 arena
    std::string_view unescape(size_t begin, size_t end) {
        char* out = arena_.allocate(end - begin);   // 上界即原文长度，一次预留
        size_t o = 0;
        for (size_t i = begin + 1; i < end - 1; ++i) {             // 跳过首尾引号
            char c = s_[i];
            if (c != '\\') { out[o++] = c; continue; }
            char e = s_[++i];
            switch (e) {
                case '"':  out[o++] = '"';  break;
                case '\\': out[o++] = '\\'; break;
                case '/':  out[o++] = '/';  break;
                case 'b':  out[o++] = '\b'; break;
                case 'f':  out[o++] = '\f'; break;
                case 'n':  out[o++] = '\n'; break;
                case 'r':  out[o++] = '\r'; break;
                case 't':  out[o++] = '\t'; break;
                case 'u': {                        // \uXXXX：仅 BMP，不做代理对合并
                    if (i + 4 >= end) fail("\\u 转义不完整");
                    unsigned cp = 0;
                    for (int k = 1; k <= 4; ++k) {
                        char h = s_[i + k];
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= static_cast<unsigned>(h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= static_cast<unsigned>(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= static_cast<unsigned>(h - 'A' + 10);
                        else fail("\\u 中有非法十六进制字符");
                    }
                    i += 4;
                    if (cp < 0x80) out[o++] = static_cast<char>(cp);
                    else if (cp < 0x800) {
                        out[o++] = static_cast<char>(0xC0 | (cp >> 6));
                        out[o++] = static_cast<char>(0x80 | (cp & 0x3F));
                    } else {
                        out[o++] = static_cast<char>(0xE0 | (cp >> 12));
                        out[o++] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                        out[o++] = static_cast<char>(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default: fail("非法转义字符");
            }
        }
        return {out, o};
    }

    double parse_number() {
        size_t start = pos_;
        if (peek() == '-') ++pos_;
        if (pos_ >= s_.size()) fail("数字不完整");
        if (peek() == '0') ++pos_;                              // 前导零：只允许单独一个 0
        else if (peek() >= '1' && peek() <= '9') { while (pos_ < s_.size() && isdigit(s_[pos_])) ++pos_; }
        else fail("非法数字起始");
        if (pos_ < s_.size() && peek() == '.') {
            ++pos_;
            if (pos_ >= s_.size() || !isdigit(s_[pos_])) fail("小数点后必须有数字");
            while (pos_ < s_.size() && isdigit(s_[pos_])) ++pos_;
        }
        if (pos_ < s_.size() && (peek() == 'e' || peek() == 'E')) {
            ++pos_;
            if (pos_ < s_.size() && (peek() == '+' || peek() == '-')) ++pos_;
            if (pos_ >= s_.size() || !isdigit(s_[pos_])) fail("指数后必须有数字");
            while (pos_ < s_.size() && isdigit(s_[pos_])) ++pos_;
        }
        // strtod 的正确用法：endptr 校验整段被吃掉 + errno 查溢出（第 5 节）
        char* endp = nullptr;
        std::string_view num = s_.substr(start, pos_ - start);
        errno = 0;
        double d = std::strtod(num.data(), &endp);
        if (endp != num.data() + num.size()) fail("数字解析意外截断");
        if (errno == ERANGE) fail("数字溢出/下溢");
        return d;
    }
};

} // namespace minijson
```

结构一目了然：`parse → parse_value → {parse_object, parse_array, parse_string, parse_number, parse_lit}`，文法与 [RFC 8259 §2] 的 ABNF 一一对应——**递归下降的本质是"文法即代码"**。三个实现要点：

1. **`string_view` 零拷贝快路径**：不含转义的字符串（绝大多数）直接引用原文，不分配不拷贝——这是本实现相对朴素版的第一支柱。
2. **对象/数组用 `vector` 而不是 `map`**：查找 O(n) 但 n 是小常数（JSON 对象平均 5-10 键），换来连续内存与 3 倍少的分配；还天然允许重复键（与 RFC 一致：重复键行为未定义，由调用方处理）。
3. **`Value` 用 `std::variant`**（第 06 章）：闭集类型的正确工具；`variant` 大小 = 最大成员（Object 的 vector 24 字节）+ 4 字节索引。

## 4. 字符串转义：两处容易写错的细节

第 3 节代码里有两处值得单独讲：

- **转义对的原子跳过**：快路径扫描里 `if (c == '\\') { pos_ += 2; continue; }`——没有这行，`"a\"b"` 会在第一个 `\"` 的引号处误判为字符串结束，剩余 `b"` 变成"文档结束后有多余字符"（这是本项目开发中真实踩过并修掉的 bug，也正是模糊测试最擅长抓的类别）。
- **`\uXXXX` 只做 BMP**：D800–DFFF 是代理对（surrogate pair），正确做法是检测高位代理后与下一个 `\uXXXX` 合成 4 字节 UTF-8。教学版遇到代理对会编码成 3 字节 CESU-8 式错误结果——这是**已声明的功能边界**，模糊测试用例生成器不应把代理对当作"必须正确"的输入。真实项目（如 simdjson）会补全这一步，工作量约 20 行。

## 5. 数值解析：strtod 的正确用法

自己写 float 解析是著名深坑（正确处理舍入需要大数运算，参考 Clinger/ Gay 的论文与 fast_float 库）。**正确姿势是让 `strtod` 干活，自己把三件配套事做对**：

1. **先手动扫描出数字的完整词素**（`-12.5e2` 的字符范围）：strtod 会"贪婪吃到非数字为止"，若不先扫边界，`12abc` 会被它吃成 12 而不报错；
2. **`endptr` 校验**：`endp != num.data() + num.size()` 说明词素内有意外的解析中断—— fail；
3. **`errno == ERANGE`**：溢出（±inf）与下溢（0）按规范报错。

一个隐蔽细节：`strtod` 的输入必须以 `'\0'` 结尾或"后面恰好不是数字"——我们传 `string_view` 的 `data()` 直接指原文，若原文是 `123456` 且后面紧跟另一个数字 token，strtod 可能多读一位。本实现不会踩到，因为**词素扫描保证了 num 末尾必是 `,`/`}`/`]`/空白等非数字字符**；但把 `num.data()` 指向无后续保护内存时要有这个意识（经验做法：给 `parse_number` 的单元测试专门覆盖"文档末尾是数字"用例）。

## 6. 性能优化三回合（真实数字）

基准：20 万条记录的合成文档 18.0 MB（紧凑格式，每条 `{"id":..,"score":..,"label":..,"flag":..,"sub":{..}}`），每回合跑 10 轮取 best 与 median。

**回合 1：朴素实现**（`std::string` 值、`std::map` 对象、逐字符 append、`substr` 喂 strtod）：

```text
回合1 朴素: best=177.3  median=173.5 MB/s
```

**回合 2：`string_view` 零拷贝 + `vector<pair>` + arena**（即第 3 节最终代码）：

```text
回合2 最终版: best=191.4  median=190.5 MB/s     （+9%）
```

**回合 3：SIMD 跳空白**（第 03 章 SSE2 手法：`_mm_cmpeq_epi8` 四路比较 + `movemask` + `ctz`，16 字节一批）：

```text
回合3 SIMD（紧凑文档）: best≈189  median≈190 MB/s   —— 没有收益！
```

为什么没收益：统计一下就知道，紧凑 JSON 里空白只占 ~0.4% 字节，跳空白本来就不是热点。换 36 空格深缩进文档再测：

```text
标量 skip_ws: best=226.0 MB/s
SSE2  skip_ws: best=243.4 MB/s     （+7.7%——仍然不是数量级）
```

**剖析驱动的补测：分配计数**。直觉说"回合 2 的大头是消灭字符串拷贝"，实测只 +9%——用全局 `operator new` 计数器找真相：

```cpp
// bench_alloc.cpp 节选
static std::atomic<long> g_allocs{0};
void* operator new(std::size_t n) {
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(n ? n : 1);
}
// 解析一次 18MB 文档：
// new 次数=1200020, 累计字节=180.8 MB        ← 输入的 10 倍堆流量！
```

120 万次 new 来自哪：20 万条记录 × 每条对象的 member vector 扩容链（cap 1→2→4→8，4 次分配）。加启发式 `reserve(8)/reserve(4)`（已并入第 3 节代码）后分配降到 ~40 万次，但吞吐只到：

```text
回合2+reserve: best=195.5  median=193.2 MB/s     （再 +2.5%）
```

**分配次数 −67% 只换来 +2.5%**——瓶颈既不是字符串拷贝、也不是分配次数，而是 **DOM 树在堆上的散布本身**（180MB 工作集对缓存/TLB 的冲击）。三个"直觉失效"连在一起，正是第 06 章"没有剖析数据不动手"的最好注脚。

## 7. 与 nlohmann/json、simdjson 对比：公平基准方法学

同机、同文档、同口径（10 轮取 median）：

```text
回合1 朴素          : median ≈174 MB/s
nlohmann/json (DOM) : median ≈176 MB/s   （best 183；官方 -O2 无特殊编译选项）
本实现 (DOM)        : median ≈190 MB/s
simdjson (按需)     : 公开数据 GB/s 量级（非本机复测，见下）
```

三条方法学纪律（经验做法）：

1. **同一进程内先跑正确性测试再跑基准会污染堆布局**——本项目实测：同一二进制，干净进程 258 MB/s，先跑过 tests() 的进程 190 MB/s（**±35%！**）。原因：120 万次分配的堆地址分布决定缓存/TLB 行为。所以对比必须在**同一 harness、同一启动条件**下进行，报 median 并做多次进程级重跑。
2. **nlohmann 打平不是"它慢"**：它同样做 DOM+map/vector 混合，且内置了 SSE2 空白跳过；我们略胜来自零拷贝字符串与更小的节点。真实项目选它毫无问题——易用性、完备性、异常安全远比 8% 重要。
3. **simdjson 的 GB/s 不是"解析得快"，是"少解析"**：on-demand 模式只解析被访问的字段，未访问的数组元素根本不建对象。拿我们的"全量 DOM"和它的"按需"比吞吐是不公平的；正确对比要么它开 DOM 模式，要么我们也做 on-demand（那是另一个项目）。它的公开基准（ simdjson README / benchmarks 目录）报告解析速度 2-4 GB/s 量级（x86 AVX2），作为方向参考即可。

## 8. 模糊测试验证（衔接《测试》系列 09）

两个工具。**libFuzzer harness**（需要 clang，GCC 不带 libFuzzer）：

```cpp
// fuzz_harness.cpp —— clang++ -std=c++20 -O1 -g -fsanitize=fuzzer,address fuzz_harness.cpp
#include "json.hpp"
#include <cstdint>
#include <string_view>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size > (1u << 20)) return 0;                 // 限制单输入大小
    std::string_view in{reinterpret_cast<const char*>(data), size};
    minijson::Arena arena(1 << 16);
    minijson::ParseError err{};
    minijson::Value v = minijson::Parser::parse(in, arena, &err);
    (void)v;                                          // 契约：任何输入都不得崩溃/挂起/逃逸异常
    return 0;
}
```

GCC 环境的替代——**随机变异循环 + ASan**（已跑）：

```cpp
// rnd.cpp 节选：以合法文档为种子，随机替换/删除/插入片段，20 万次
// g++ -std=c++20 -O1 -g -fsanitize=address -I. rnd.cpp -o rnd && ./rnd
```

```text
20 万次变异: 异常逃逸 0 次        （ASan 全程无越界/UAF 报告）
```

契约明确才测得动：`parse` 的约定是"错误以 `ParseError` 形式返回（不抛）、任何输入不崩溃"。变异循环里出现的 `catch (...)` 命中即 bug。开发过程中这套循环真实抓到过一个 bug——第 4 节的转义误判（修复前 `"a\"b"` 类输入直接解析错乱）。

## 9. 错误处理与位置报告

错误路径的三个设计点：

- **边界把异常转成值**：`parse` 内部用 C++ 异常做栈回退（递归下降报错最省事的机制），边界捕获后填 `ParseError` 返回 null——调用方零异常负担，也兼容 `-fno-exceptions` 的上层（第 10 章）。
- **行列号在错误时才算**：`fail()` 里从头数换行，O(错误位置)——错误是罕见路径，热路径零开销。行号算法：遇到 `'\n'` 行 +1 列归 1，否则列 +1。
- **错误信息分层**：`msg` 指静态字符串（"字面量拼写错误"），`offset` 给精确字节位置，`line/col` 给人看——日志里三样都打：

```cpp
minijson::ParseError err{};
auto v = minijson::Parser::parse("{\n  \"k\": tru,\n}", arena, &err);
// err.msg="字面量拼写错误", err.line=2, err.col=8
```

## 10. 复盘：从这 ~600 行学到什么

行数账：`json.hpp` 288 + `test_json.cpp` 58 + `naive_json.cpp` 144（回合 1 对照）+ `bench_alloc.cpp` 37 + `rnd.cpp` 32 + libFuzzer harness ~15 + `CMakeLists.txt` ~30 ≈ **600 行**。

1. **文法即代码**：递归下降把 ABNF 直接翻译成函数调用关系，正确性靠"结构对齐文法"而不是小心；改需求（加一种类型）= 加一个 case。
2. **性能直觉三连败**：string_view 拷贝消除 +9%（SSO 早把短字符串兜住了）、reserve 省 67% 分配只 +2.5%、SIMD 跳空白在紧凑文档上零收益——**每一步都必须测量，直觉只是假设**。
3. **真瓶颈藏在分配的"布局"里**：180MB 堆流量的缓存/TLB 冲击 + 堆地址分布让同一二进制 ±35%。DOM 型解析器要再上台阶，路只有两条：arena 上挂 vector（彻底消灭散布，参考 RapidJSON 的 MemoryPoolAllocator）或转 on-demand（simdjson 路线）。
4. **正确性是测出来的**：20 万次变异 0 逃逸 + 转义误判案例说明，解析器这类"输入即攻击面"的代码，模糊测试不是可选项。
5. **与成熟库的差距是设计哲学差距**：nlohmann 赢在易用性，simdjson 赢在按需解析，我们 600 行赢在"每一步都知道为什么"。读 simdjson 源码前先写过自己的解析器，收获翻倍。

## 常见误区

1. **拿 libFuzzer 硬上 GCC**：`-fsanitize=fuzzer` 是 clang 专属；GCC 用户用 AFL++ 或本章的随机变异循环， harness 结构完全一样。
2. **strtod 不查 endptr/errno**：`12abc` 静默变 12、`1e999` 静默变 inf——两行校验省掉一类整类 bug。
3. **递归下降无深度限制**：`[[[[[...` 深嵌套会打爆调用栈（MCU 上就是第 10 章的栈溢出）。生产版加 `depth` 计数上限（如 128 层）——留作练习 4。
4. **基准忘了控制堆状态**：第 7 节的 ±35% 案例。基准程序应尽量在"冷启动后固定流程"下多次重跑，报中位数，警惕首轮。
5. **`\u` 转义当小事**：代理对、UTF-8 校验是 JSON 解析器正确性的重灾区；功能边界要写进文档而不是留给用户发现。
6. **错误信息只给"invalid json"**：没有 offset/line/col 的解析错误在生产环境等于没有错误信息——调试配置文件的人会恨你。

## 本篇小结

- 单遍递归下降 + `string_view` 零拷贝 + arena + `vector<pair>` DOM，~600 行完成 RFC 8259 子集解析器，median 190 MB/s（18MB 合成文档）。
- 转义对的原子跳过、`\u` 的 BMP 边界、strtod 的 endptr/errno 配套——三处细节决定解析器"能不能上生产"。
- 三回合优化实测 +9% / +0% / +2.5%，分配计数揭示真瓶颈是 DOM 树的堆散布；堆布局让同一二进制 ±35%，基准方法学与优化本身同等重要。
- 模糊测试（libFuzzer 或随机变异 + ASan）是解析器的标配验证；20 万次变异 0 异常逃逸是本实现的及格线。
- 与 nlohmann（易用性）和 simdjson（按需解析）的对比要"同口径"，差距是设计哲学而非代码质量。

## 练习

1. 给 `Parser` 加 `size_t depth_` 与上限 128，超限 `fail("嵌套过深")`；用 1000 层 `[[[...` 输入验证修复前后行为（修复前大概率段错误——注意用 ASan 构建）。
2. 把 `Object`/`Array` 的内存改为从 Arena 分配（给 `std::vector` 传自定义 allocator，复用第 06 章 Arena），重测吞吐与 `operator new` 计数，验证"消灭堆散布"假设。
3. 补全 `\u` 代理对：检测 D800–DBFF 后读取下一个 `\uXXXX` 合成码点，UTF-8 编码 4 字节；为合法/非法代理对各写 3 个测试用例。
4. 用 libFuzzer（clang）跑 harness 10 分钟，把 crash 输入保存下来写成回归测试；对比随机变异循环 20 万次与 libFuzzer 同时间 discovering 的输入数。
5. 实现"按需"接口的雏形：`find(doc, "records[3].label")` 路径直取、跳过无关子树，测量"只取一个字段"场景下与全量 DOM 的吞吐比，解释与 simdjson 的差距还剩什么。
