# 03 - 内存模型与 RAII

> 本篇回答的问题：
> - C++ 程序的内存分几块？栈和堆各管什么？对象从出生到死亡经历了什么？
> - 裸 `new`/`delete` 有哪四大坑？RAII 是什么？为什么说它是 C++ 最重要的思想？
> - "所有权"是什么意思？（为下一篇智能指针铺路）
> - `const std::string& s = make_something();` 绑定临时对象，安全吗？
> - 怎么用 ASan 抓内存泄漏？

前置要求：已读第 01、02 篇；知道对象有构造和析构（下一篇系统讲类）。

---

## 1. 栈 vs 堆：两种内存，两种人生

C++ 程序的内存大致分三块，本篇关心前两块：

- **栈（stack）**：函数调用时自动分配、返回时自动回收。局部变量都住这里，分配/回收就是移动栈指针，快到可忽略；但空间有限（默认几 MB），且**生命周期严格等于所在作用域**。
- **堆（heap）**：全局大内存池，由你在运行期显式申请（`new`/`malloc`），**大小只受物理内存限制，生命周期由你手动控制**——麻烦就麻烦在这。
- **静态/全局区**：全局变量和 `static` 变量的家，程序启动创建、退出销毁，本篇不展开。

一个对象的完整人生，由它住在哪决定：

```
栈对象：  进入作用域 { ──构造── 使用 ──离开作用域自动析构── }   死期写进了户口本，编译器准时执行
堆对象：  new ──使用── delete                                   死期全靠手动操作；忘了 delete 是泄漏，delete 早了是悬垂
```

看构造/析构的真实时间线（这里用到最简单的类：构造函数、析构函数，下一篇细讲）：

```cpp
// lifetime.cpp
#include <cstdio>

struct Frame {
    explicit Frame(int id) : id_(id) {
        std::printf("  Frame(%d) 构造\n", id_);
    }
    ~Frame() {
        std::printf("  Frame(%d) 析构\n", id_);
    }
    int id_;
};

Frame make_frame(int id) {
    Frame f{id};              // 栈对象：函数结束就销毁
    std::printf("  return 前的 f\n");
    return f;                 // 值返回（编译器会做拷贝消除，不必担心拷贝开销）
}

int main() {
    std::printf("-- 进入 main --\n");
    {
        Frame a{1};                 // 出了这对大括号，a 立刻销毁
        std::printf("  块内使用 a\n");
    }                               // <-- a 在这里析构
    std::printf("-- 离开块 --\n");

    Frame b = make_frame(2);        // 返回值被"接住"，活到 main 结束
    std::printf("main 末尾\n");
    return 0;                       // <-- b 在这里析构（后构造的先析构）
}
```

预期输出：

```
-- 进入 main --
  Frame(1) 构造
  块内使用 a
  Frame(1) 析构
-- 离开块 --
  Frame(2) 构造
  return 前的 f
main 末尾
  Frame(2) 析构
```

读两遍这个输出，你就理解了 C++ 内存模型的一半：`a` 的析构精确发生在**离开大括号的那一行**（栈对象生命周期=作用域，编译器保证，无一例外）；同一作用域多个栈对象**构造与析构顺序相反**（想象一摞盘子）；`make_frame` 里的局部 `f` 没有立刻析构——它被值返回"接住"了，这是编译器的拷贝消除（C++17 起对这种写法是**强制**的，不是玄学）。

什么时候需要堆？——**大小运行期才知道**（用户上传了一段 3 秒的 wav）、**活得比当前作用域久**（把结果交出去）、**太大**（几百 MB 的模型权重不适合塞栈上）。

---

## 2. 裸 new / delete 的四大坑

堆对象必须手动管理，"手动"意味着四种经典死法：

1. **内存泄漏**：`new` 了没 `delete`。长跑的推理服务每秒漏一点，几天后 OOM 被运维叫醒。
2. **双重释放（double free）**：同一块内存 `delete` 两次。堆结构被破坏，崩溃点可能离案发现场十万八千里，极难查。
3. **悬垂指针（dangling pointer）**：`delete` 之后还在用那个指针——内存还回去了，指针还指着它。读到垃圾是好事，写到被复用的内存就是安全漏洞。
4. **异常不安全**：`new` 之后、`delete` 之前抛了异常（或提前 `return`），`delete` 那行被跳过——泄漏。第 02 篇的 `std::string` 为什么随便抛异常也不泄漏？因为它内部替你管住了。

> 这四种死法可以同时出现在一个函数里：`new` 完提前 `return`（泄漏）、两条路径各 `delete` 一次（双重释放）、delete 后继续用（悬垂）。下面用 RAII 一次性解决。

---

## 3. RAII：让作用域替你管理资源

### 3.1 核心思想

C++ 给了你一个其他语言少见的保证：**栈对象离开作用域时，析构函数百分之百会被调用**（哪怕中间抛了异常）。RAII（Resource Acquisition Is Initialization，资源获取即初始化）就把它变成资源管理的基石：

> **把资源绑到栈对象的生死上：构造函数里获取资源，析构函数里释放资源。** 之后你只管声明对象，释放交给作用域，永远不会忘。

资源远不止内存：文件句柄、互斥锁、音频设备、GPU 显存、socket……凡是"要获取、要归还"的都适用。第 02 篇的 `std::string`/`std::vector` 就是标准库里的 RAII 类——构造时按需向堆要内存，析构时归还，你从来没为它们写过 `delete`。**你已经在用 RAII 了，本篇只是教你给任意资源手写一个。**

### 3.2 完整示例：RAII 包装 FILE*

以语音工程最日常的资源——音频文件为例，把 `FILE*` 管起来：

```cpp
// raii_file.cpp
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// RAII：构造 = 获取资源，析构 = 释放资源。
// FILE* 交给这个类管理之后，就再也不用手写 fclose。
class WavFile {
public:
    explicit WavFile(const std::string& path)
        : handle_(std::fopen(path.c_str(), "rb")) {
        if (handle_ == nullptr) {
            throw std::runtime_error("cannot open " + path);
        }
        std::printf("[WavFile] opened %s\n", path.c_str());
    }

    ~WavFile() {
        if (handle_ != nullptr) {
            std::fclose(handle_);
            std::printf("[WavFile] closed\n");
        }
    }

    // 禁止拷贝：两个对象管同一个 FILE* 会导致双重 fclose（第 04 篇细讲）
    WavFile(const WavFile&) = delete;
    WavFile& operator=(const WavFile&) = delete;

    bool read_samples(std::vector<float>& out, int n) {
        out.resize(static_cast<size_t>(n));
        return std::fread(out.data(), sizeof(float),
                          static_cast<size_t>(n), handle_) ==
               static_cast<size_t>(n);
    }

private:
    std::FILE* handle_;    // 资源句柄藏在这里
};

int main() {
    // 先造一个测试文件
    {
        std::FILE* f = std::fopen("test.f32", "wb");
        const float samples[4] = {0.1f, 0.2f, 0.3f, 0.4f};
        if (f != nullptr) {
            std::fwrite(samples, sizeof(float), 4, f);
            std::fclose(f);
        }
    }

    try {
        WavFile wav("test.f32");
        std::vector<float> buf;
        if (wav.read_samples(buf, 4)) {
            std::printf("read ok:");
            for (const auto& s : buf) {
                std::printf(" %g", s);
            }
            std::printf("\n");
        }
    }   // <-- 不管中间发生什么（哪怕抛异常），离开作用域都会 fclose
    catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
    }

    try {
        WavFile bad("not_exist.f32");                      // 打开失败 -> 抛异常
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());              // 没有泄漏，因为没构造成功就不用析构
    }
    return 0;
}
```

预期输出：

```
[WavFile] opened test.f32
read ok: 0.1 0.2 0.3 0.4
[WavFile] closed
error: cannot open not_exist.f32
```

逐点品味：`main` 里**没有出现任何 fclose**，但句柄都被正确关闭——包括打开失败那次（构造函数抛异常前没拿到资源，析构自然无事可做，这正是异常安全的标准答案）。对比裸指针版本：`fopen` 后每条路径（正常、提前 return、抛异常）都要记得 `fclose`，漏一条就是泄漏，而 RAII 版只有**一个析构函数**管释放，逻辑集中、无法遗漏。`= delete` 删除拷贝是 RAII 类的标准动作（资源句柄只有一份，"拷贝管理权"必须显式设计，见下一节）。装资源的 `std::FILE* handle_;` 是裸的，但被**封装在类内**，外界碰不到——RAII 不是消灭裸指针，而是把裸指针圈进牢笼。

### 3.3 再来一个：RAII 计时器

RAII 不只管"释放"，一切"进入时做 X、退出时必须做 Y"的成对操作都适合，比如性能剖析：

```cpp
// raii_timer.cpp
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

// 作用域计时器：构造时开始计时，析构时自动打印耗时。
// 用法：在想知道耗时的作用域里放一行 ScopedTimer t("名字"); 就完了。
class ScopedTimer {
public:
    explicit ScopedTimer(std::string name)
        : name_(std::move(name)),
          start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        const auto end = std::chrono::steady_clock::now();
        const auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start_)
                .count();
        std::printf("[timer] %s took %lld us\n", name_.c_str(),
                    static_cast<long long>(us));
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

void fake_inference() {
    ScopedTimer t("mel_spectrogram");
    std::this_thread::sleep_for(std::chrono::milliseconds(12));   // 假装在算
}

int main() {
    fake_inference();
    std::printf("main done\n");
    return 0;
}
```

预期输出（耗时数字每次不同）：

```
[timer] mel_spectrogram took 12066 us
main done
```

想测哪段代码，就在作用域第一行放个计时器，退出时自动出结果——Python 里 `with` 语句/装饰器的体验，C++ 用 RAII 原生实现。同样的模式能写锁守卫（`std::lock_guard`，第 12 篇会见到）。

---

## 4. 所有权（ownership）：下一篇的铺垫

上面 `WavFile` 的设计里藏着一个关键概念：**所有权**——**谁负责释放这个资源，谁就拥有它。**

- `handle_` 这个 `FILE*` 归 `WavFile` 对象所有：析构时由它 fclose，别的代码不许碰。所以必须 `= delete` 删除拷贝——两个对象同时"拥有"就等于谁都不负责，必然双重释放。
- 裸指针时代，所有权只存在于程序员脑子里（注释里那句 "// 调用方负责 delete" 是 C++ 史上最不可靠的文档）。现代 C++ 把所有权写进**类型系统**：`std::unique_ptr<T>` 独占所有权、出作用域自动释放（= RAII 的指针形态，**以后你的默认选择**）；`std::shared_ptr<T>` 共享所有权，最后一个放手的负责释放。

第 04 篇系统讲智能指针。本篇带走一句话：**对每个堆资源问一句"谁拥有它？"，答案应该是一个明确的类型，而不是"大家商量着来"。**

---

## 5. 生存期延伸：const 引用绑定临时对象

第 02 篇说过引用必须绑定已存在的对象，那临时对象（函数返回值）能不能绑？规则微妙：

- `const T&` 绑定临时对象：**合法**，且标准保证临时对象的寿命**延长到引用的作用域结束**；
- 非 const 的 `T&` 绑定临时对象：编译错误（"都说是临时的了，凭什么允许别人改"）。

```cpp
// constref.cpp
#include <cstdio>
#include <string>

std::string make_tag() {
    return "frame_001";     // 返回一个临时 string
}

int main() {
    // const 引用绑定临时对象：临时对象的生命周期被延长到引用的作用域结束，
    // 这是合法且安全的，标准明确保证。
    const std::string& tag = make_tag();
    std::printf("tag=%s\n", tag.c_str());   // 完全没问题

    // 而非 const 的引用绑不了临时对象，下面这行编译不过：
    // std::string& bad = make_tag();   // error: cannot bind non-const lvalue ref
    return 0;
}
```

预期输出：

```
tag=frame_001
```

**重要限定**：延长只发生在"**局部 const 引用直接绑定临时对象**"这一种场景。临时对象被传进函数、存进类的引用成员、或经返回引用转了一手，延长不生效——那就是悬垂。工程上更省心的做法是按值接（`std::string tag = make_tag();`，有移动语义兜底，通常并不贵）。

---

## 6. 用 ASan 抓泄漏

第 01 篇用 ASan 抓了越界；它的子模块 **LeakSanitizer** 会在程序退出时盘点所有未释放的堆内存。看一个标准泄漏：

```cpp
// leak_asan.cpp
#include <cstdio>

int main() {
    auto* buffer = new float[16000];   // 1 秒的 16kHz 音频缓冲
    buffer[0] = 0.0f;
    std::printf("forgot to delete...\n");
    // 没有 delete[] buffer;  —— 泄漏！
    return 0;
}
```

```bash
g++ -std=c++17 -g -Wall -Wextra -pedantic -fsanitize=address -o leak_asan leak_asan.cpp
./leak_asan
```

真实输出（关键行）：

```
=================================================================
==4245==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 64000 byte(s) in 1 object(s) allocated from:
    #0 ... in operator new[](unsigned long)
    #1 ... in main leak_asan.cpp:5

SUMMARY: AddressSanitizer: 64000 byte(s) leaked in 1 allocation(s).
```

报告直接给出：泄漏多少字节、在**哪一行**分配的。16000 个 float 正好 64000 字节，数字对得上。

双重释放同样当场抓获（`delete p; delete p;`）：

```
==30142==ERROR: AddressSanitizer: attempting double-free on 0x502000000010 ...
    #1 ... in main double_free_asan.cpp:7
freed by thread T0 here:
    #1 ... in main double_free_asan.cpp:6
```

第一次 free 和第二次 free 的行号、分配位置全在完整报告里——手工查这种问题可能要一整天，ASan 几秒抓完。把 `-fsanitize=address` 加进你的测试构建习惯里（比如单元测试永远带它跑）。

---

## 常见坑

1. **`new[]` 配 `delete[]`，`new` 配 `delete`**：写错组合是未定义行为。好消息：学了第 04 篇后你几乎不再手写任何 delete。
2. **以为"程序退出了泄漏就没事"**：短命脚本或许没事；长跑服务、循环里分配、给别人的库做回调——泄漏会滚雪球。
3. **悬垂引用假装是延长生存期**：只有"局部 const 引用直接绑临时"这一种延长；放进容器、成员、函数返回值里的引用一律不延长。
4. **RAII 类忘了处理拷贝**：默认拷贝逐成员复制，两个对象会管同一个句柄。要么 `= delete`，要么实现移动（第 06 篇），要么干脆存智能指针。
5. **构造函数里抛异常前留了资源**：构造函数抛异常时析构函数**不会**被调用（对象没出生完），先获取的资源会漏掉——资源应全部交给 RAII 成员托管（成员会自动析构）。
6. **栈上开大数组爆栈**：`float buf[16000 * 60 * 10];`（10 分钟音频）约 38 MB，默认栈只有 8 MB，直接段错误。大缓冲用 `std::vector`（住堆上，自动 RAII）。

---

## 小结

- 栈：分配快、生命周期=作用域、编译器全自动；堆：大小与寿命自由、代价是手动管理。
- 裸 new/delete 四大坑：泄漏、双重释放、悬垂、异常不安全。
- RAII：构造获取、析构释放，把资源绑到作用域上；文件句柄、锁、计时器都能用同一模式包装。
- 所有权：谁负责释放谁拥有；所有权应写进类型（第 04 篇 `unique_ptr`/`shared_ptr`），而不是注释。
- `const T&` 绑临时对象会延长其寿命到引用作用域结束，但仅限局部直接绑定这一种场景。
- `-fsanitize=address` 同时抓越界、双重释放、泄漏（LeakSanitizer），报告带行号和分配栈。

## 自测题

1. `struct T { ~T() { puts("bye"); } };` 配上 `int main() { T a; { T b; puts("mid"); } puts("end"); }`，输出顺序是什么？为什么？
2. 裸 new/delete 的四大坑分别是什么？各举一句话场景。
3. RAII 用一句中文概括。为什么 RAII 类在异常存在时也安全，而裸 new/delete 不安全？
4. RAII 类为什么要 `= delete` 拷贝构造？不删会发生什么？
5. `const std::string& s = std::string("hi");` 合法吗？把 `s` 换成某个类的引用成员呢？
6. ASan 打印 `Direct leak of 64000 byte(s) ... app.cpp:12`，你该去查什么？

<details><summary>答案</summary>

1. 输出顺序是 `mid`、`bye`、`end`、`bye`：先打印 `mid`；随后内层块结束，`b` 立刻析构（第一个 `bye`）；接着打印 `end`；main 结束时轮到 `a` 析构（第二个 `bye`）。考点：析构精确发生在离开作用域处。
2. ① 泄漏：异常/提前 return 跳过 delete，长跑服务内存滚雪球；② 双重释放：两条路径都 delete 同一指针，堆结构损坏；③ 悬垂：delete 后继续用指针，读写已归还的内存；④ 异常不安全：new 与 delete 之间任何一条抛异常的路径都造成泄漏。
3. "把资源的获取和释放绑到栈对象的构造和析构上，释放交给作用域自动完成。" 异常安全：栈展开时析构函数保证被调用，即使中途抛异常释放代码也必然执行；裸 delete 是普通语句，异常路径直接跳过。
4. 默认拷贝逐成员复制，拷贝出的对象和原对象持有**同一个**资源句柄，两个析构都会释放它 → 双重释放。`= delete` 在编译期禁止拷贝，逼使用者明确管理权转移（移动）。
5. 第一句合法：局部 const 引用直接绑定临时对象，寿命延长到作用域结束。第二句危险：延长不适用于"经参数/成员转手"的场景，构造函数结束后临时实参消亡，成员里留下悬垂引用。
6. 查 app.cpp 第 12 行的分配（16000 个 float = 64000 字节，正是那段音频缓冲）在哪条路径上没有对应的释放。修复方式是改用 `std::vector` 或智能指针，而不是手动补一个 delete。
</details>
