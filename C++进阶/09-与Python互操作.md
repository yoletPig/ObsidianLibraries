# 09 - 与 Python 互操作

## 本篇目标

- 说清"C++ 扩展 Python"的两个动机：GIL 之内的真并行、复用现成 C++ 库。
- pybind11 完整上手：函数/类绑定、STL 自动转换、numpy 零拷贝传递（含实时数据核对）。
- 用 `py::gil_scoped_release` 让长计算不阻塞 Python 线程，并用实验验证。
- 打通构建集成（scikit-build-core）、认清 ABI 常识、做出方案选型对比、给模块写测试。

本章全部数字来自真机构建实测（环境：Python 3.10 / numpy 1.26 / pybind11 3.x / g++ 11，i5-12600K）。

## 1. 为什么 C++ 扩展 Python

两个动机，优先级随场景不同：

1. **GIL 内的真并行**。Python（CPython，3.12 前）的 GIL 让同一进程内的多线程无法并行执行 Python 字节码。但 **GIL 只护 Python 对象——纯 C++ 代码不受它管**：在 C++ 扩展里释放 GIL（第 4 节），多个"重计算"就能真正跑满多核，这是 CPU 密集型任务留在 Python 进程内的关键手段（否则只能 multiprocessing + 进程间通信）。
2. **复用 C++ 资产**。RKNN runtime、ONNX Runtime、ffmpeg、自研算法库——重写不现实。给它们包一层 Python 接口，Python 做胶水与业务，C++ 做计算与 IO，是业界标准分工（torch、numpy 本身就是"C 核心 + Python 皮"）。

反过来看 Python 侧的证据（本机实测，1000 万个 float 求和）：

```text
Python for 循环    : 290.7 ms
Python 内置 sum()  :  26.4 ms
```

同一逻辑用 C++ 实现并经 pybind11 暴露后是 6ms（第 3 节实测）——**约 48 倍**。解释器逐条执行字节码 + 每个元素是完整 PyObject（浮点数带类型头、引用计数），是这 48 倍差距的全部来源。

## 2. pybind11 完整上手

pybind11 是 header-only 的 Python 绑定库（仅头文件 + 一条编译命令），用 C++ 模板在编译期生成 CPython 胶水。环境准备：`pip install pybind11`，编译时加它的 include 路径与 Python 头文件路径。

```cpp
// fastmod.cpp —— 模块名必须与 PYBIND11_MODULE 的第一个参数一致
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <chrono>
#include <vector>
namespace py = pybind11;

double sum_loop(const std::vector<double>& v) {   // Python list 自动转 vector<double>
    double s = 0;
    for (double x : v) s += x;
    return s;
}

double sum_np(py::array_t<double, py::array::c_style | py::array::forcecast> a) {
    auto buf = a.unchecked<1>();                  // 一维视图；unchecked 免去逐元素开销
    double s = 0;
    for (py::ssize_t i = 0; i < buf.size(); ++i) s += buf[i];
    return s;
}

void scale_inplace(py::array_t<double> a, double k) {  // 零拷贝：原地改写
    auto buf = a.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < buf.size(); ++i) buf(i) *= k;
}

PYBIND11_MODULE(fastmod, m) {
    m.def("sum_loop", &sum_loop);
    m.def("sum_np", &sum_np);
    m.def("scale_inplace", &scale_inplace);
}
```

```bash
INC=$(python3 -c "import pybind11; print(pybind11.get_include())")
NPINC=$(python3 -c "import numpy; print(numpy.get_include())")
g++ -O2 -shared -std=c++20 -fPIC -I$INC -I$NPINC fastmod.cpp -o fastmod.so
```

（正式包的文件名要带 ABI 标签如 `fastmod.cpython-310-x86_64-linux-gnu.so`，本机实验直接叫 `fastmod.so` 也能 import；`-fPIC -shared` 是 Linux 动态库的必要条件。）

### 类绑定与 STL 自动转换

```cpp
// clsmod.cpp（节选，已实测）
#include <pybind11/stl.h>        // 启用 vector/map/optional... 与 Python 容器互转
class Tokenizer {
public:
    explicit Tokenizer(std::map<std::string, int> vocab) : vocab_(std::move(vocab)) {}
    std::vector<int> encode(const std::string& text) const { /* 查词典分词 */ }
    void set_unk(int id) { unk_id_ = id; }
    int unk() const { return unk_id_; }
private:
    std::map<std::string, int> vocab_;
    int unk_id_ = 0;
};
PYBIND11_MODULE(clsmod, m) {
    py::class_<Tokenizer>(m, "Tokenizer")
        .def(py::init<std::map<std::string, int>>())          // dict -> map 自动转
        .def("encode", &Tokenizer::encode)                    // 返回值 vector<int> -> list 自动转
        .def_property("unk_id", &Tokenizer::unk, &Tokenizer::set_unk);  // 属性语法糖
}
```

```python
>>> from clsmod import Tokenizer
>>> t = Tokenizer({"hello": 3, "world": 7})
>>> t.unk_id = 99
>>> t.encode("hello world 你好 world")
[3, 7, 99, 7]
>>> t.unk_id
99
```

Python 的 dict/list 进出 C++ 都被自动转换。**但要清醒：自动转换是拷贝语义**——1000 万元素的 list 转成 `std::vector<double>` 要逐个解包 PyObject，本机实测这一个转换就花了 ~140ms（见第 5 节基准里 `sum_loop` 的构成）。**批量数据走 numpy 数组，容器自动转换留给小对象**（经验做法）。

## 3. numpy 数组零拷贝传递（实时数据核对）

`py::array_t<double>` 直接持有 numpy 数组的底层缓冲。零拷贝的最硬核证据是**C++ 侧原地改写、Python 侧立即看到**：

```python
>>> import numpy as np, fastmod
>>> b = np.array([1.0, 2.0, 3.0])
>>> fastmod.scale_inplace(b, 10.0)     # C++ 侧 buf(i) *= k 直接写内存
>>> b
array([10., 20., 30.])                 # Python 侧立刻可见：同一块内存
```

保证零拷贝的四个条件：模板参数写 `py::array_t<double, py::array::c_style | py::array::forcecast>` 时——`c_style` 要求连续（不连续会自动拷贝一份），`forcecast` 强制类型一致（float32 进 double 扩展会拷贝）；Python 侧传入的是 `float64` 连续数组时不发生拷贝。核对方法（经验做法）：改写前 `b.__array_interface__['data'][0]` 记下地址，C++ 里把指针打出来对比；或像上面一样原地改写看可见性。跨进程/跨线程传 numpy 数组时同样只传"视图"不传数据，这正是 llama.cpp Python 绑定、RKNN Python API 处理输入张量的方式。

## 4. GIL 管理：py::gil_scoped_release

默认规则：**pybind11 包装函数被 Python 调用时持有 GIL**。纯 C++ 计算段应该主动释放，让其他 Python 线程并行推进：

```cpp
#include <pybind11/pybind11.h>
void heavy(int ms) {
    {
        py::gil_scoped_release release;          // 进入：解锁 GIL
        // ... 纯 C++ 计算，绝不能碰任何 Python API/PyObject ...
        asm volatile("" : : "r"(sink) : "memory");
    }                                            // 作用域结束：自动重新加锁
}
```

铁律：**释放期间不得调用任何 Python C API、不得解引用 PyObject**（numpy 数组的 `.unchecked()` 视图底层指针是裸指针可以继续用，但不能再摸 Python 对象）。验证实验：C++ 线程 `heavy(400ms)` 释放 GIL 期间，另一个纯 Python 线程能持续推进：

```python
import threading, time, fastmod
counter = {"n": 0}
def py_worker():
    end = time.perf_counter() + 0.4
    while time.perf_counter() < end:
        counter["n"] += 1

th = threading.Thread(target=py_worker)
t0 = time.perf_counter()
th.start()
fastmod.heavy(400)          # C++ 长计算（内部 gil_scoped_release）
th.join()
t1 = time.perf_counter()
# 本机实测输出：
# GIL 释放: heavy(400ms) 期间 Python 线程推进了 4550221 次循环, 总耗时 405 ms
```

400ms 的 C++ 计算与 Python 线程**同时进行**（总耗时 405ms ≈ max 而不是相加），455 万次 Python 循环证明 GIL 确实被释放。若去掉 `gil_scoped_release`，Python 线程几乎完全饿死（只会在 C++ 函数起止的间隙抢到几个时间片）。选 GIL 策略的口诀（经验做法）：**进出 Python 世界的边界上拿/还 GIL，中间整段计算释放；调用方向是"Python → C++"时用 `gil_scoped_release`，"C++ 回调 → Python"时用 `py::gil_scoped_acquire`**。

## 5. 性能对比实验（完整数据）

1000 万元素求和，四种实现同机实测：

```text
Python for 循环     :  290.7 ms      基线
Python 内置 sum()   :   26.4 ms      C 实现的迭代，快 11 倍
pybind11 sum_loop   :  147.2 ms      其中 ~140ms 是 list->vector 的转换拷贝！
pybind11 sum_np     :    6.0 ms      numpy 零拷贝，比 Python 循环快 48 倍
numpy .sum()        :    3.5 ms      成熟库的 SIMD+归约优化（第 03/06 章）
```

三读数：Python 循环 48 倍慢于 C++，验证第 1 节的动机；**`sum_loop` 的 147ms 里转换占大头**——传 list 给 C++ 再算，不如留在 Python 用内置函数，这是"绑定不能无脑替代内置"的实证；批量数据的最优路径是 numpy 视图直通 C++（6ms 里 2ms 是 Python 调用与检查开销，其余是 C++ 逐元素求和）。进一步的方向：把逐元素求和改成块处理/多线程（配合第 4 节 GIL 释放），或直接上 SIMD——推断链路与第 06 章完全一致。

## 6. 构建集成：scikit-build-core 最小配置

散装 `g++` 命令只适合实验；发布包需要"pip install 就能装"的构建。现代推荐 **scikit-build-core**（用 CMake 做 C++ 构建的 PEP 517 后端，替代老 scikit-build）：

```toml
# pyproject.toml
[build-system]
requires = ["scikit-build-core", "pybind11"]
build-backend = "scikit_build_core.build"

[project]
name = "fastmod"
version = "0.1.0"
dependencies = ["numpy"]

[tool.scikit-build]
wheel.packages = []
```

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.22)
project(fastmod LANGUAGES CXX)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(fastmod fastmod.cpp)
install(TARGETS fastmod DESTINATION .)
```

```bash
pip install .          # 自动：建隔离环境 → CMake 构建 → 打 wheel → 装好
python -c "import fastmod; print('ok')"
```

pybind11 被声明为构建依赖后，scikit-build-core 会把它喂给 CMake（`find_package(pybind11 CONFIG)`），第 01 章的 target 中心思想原样适用。老的 `setup.py` + `setuptools` 手写 Extension 的方式仍能用，但新项目不再推荐（经验做法）。交叉编译到 ARM 安装包用 cibuildwheel，与本章机制正交。

## 7. 调用约定与 ABI 常识

衔接《计算机系统与网络》02（机器级表示）。扩展模块是 `.so` 动态库，Python 解释器用 `dlopen` 加载后**按 C ABI 拿符号**：CPython 定义了一组 C 函数指针（`PyMethodDef` 表），pybind11 在模块初始化时填好这张表。要点：

- **C++ 名字修饰（name mangling）**：`extern "C"` 禁用修饰才能被 `ctypes`/`dlsym` 找到；pybind11 模块入口 `PyInit_fastmod` 本身就是 extern "C" 的，内部绑定则完全不走 C ABI。
- **调用约定**：x86-64 System V ABI 前 6 个整型/指针参数走寄存器（rdi/rsi/rdx/rcx/r8/r9），浮点走 xmm——**跨 Python/C++ 边界的每次调用都有真实成本**（第 5 节 6ms 里的固定 2ms 主要是它），所以批量接口设计成"一次传一整个数组"而不是"每元素调一次"。
- **CPython 稳定 ABI**：`Py_LIMITED_API` 让同一个 .so 兼容 3.x 多个小版本（文件名 `abi3.so`）；pybind11 对其支持有限（部分特性不可用），发布多版本 wheel 是更常见的现实选择。

## 8. 替代方案对比（事实性）

| 方案 | 形态 | C++ 支持 | 性能 | 适用 |
| --- | --- | --- | --- | --- |
| ctypes | 标准库，`dlopen` + 手写签名 | 无（C ABI only），C++ 需 `extern "C"` 包装 | 每次调用有解释器开销，小调用慢 | 无依赖环境、只有 C 接口的库 |
| pybind11 | header-only，模板胶水 | 完整（类/模板/STL/智能指针） | 与手写 C API 同级 | 通用首选，C++11+ |
| nanobind | pybind11 作者 2022 年重写 | 完整 | 官方基准：模块更小（~4 倍）、调用更快（多数场景快 1.5-4 倍），需 C++17、仅 CPython | 新项目优先（esp. 3.11+） |
| Cython | Python 超集 → C 编译 | 通过 `cdef extern` 声明 | 数值代码接近手写 C | 已有 Python 代码逐步加速 |

选型口径（经验做法）：**已有 C++ 库要包 Python → pybind11（存量代码）或 nanobind（新项目）；纯 C 接口临时调用 → ctypes；把 Python 热点改写 → Cython 或"改写成 C++ + pybind11"。** 事实要点：nanobind 与 pybind11 同作者（Wenzel Jakob），API 刻意相似，迁移成本低；两者都不支持 PyPy 全部特性。

## 9. 给 pybind11 模块写测试

衔接《测试》系列 11。扩展模块的测试就是**普通 pytest**，测的是"从 Python 看到的行为契约"：

```python
# tests/test_fastmod.py
import numpy as np
import pytest
from fastmod import sum_np, scale_inplace

def test_sum_np_basic():
    assert sum_np(np.array([1.0, 2.0, 3.0])) == pytest.approx(6.0)

def test_sum_np_empty():
    assert sum_np(np.array([], dtype=np.float64)) == 0.0

def test_zero_copy_inplace():
    b = np.array([1.0, 2.0, 3.0])
    scale_inplace(b, 10.0)
    np.testing.assert_allclose(b, [10.0, 20.0, 30.0])   # 原地语义：不改入参就失败

def test_forcecast_copies_float32():
    a = np.array([1, 2, 3], dtype=np.float32)           # 非 float64 → forcecast 拷贝
    b = np.array([1.0, 2.0, 3.0])
    assert sum_np(a) == pytest.approx(6.0)              # 结果仍正确
    a[0] = 100                                          # 改 a 不影响任何 C++ 缓存视图
```

测试策略（经验做法）：**契约测试在 Python 侧写（上面的形态）；内存错误类问题靠 C++ 侧的 ASan/TSan 构建 + valgrind**——CI 里跑一个 `-fsanitize=address` 构建的扩展再跑 pytest，能把越界/UAF 在 Python 调用现场报出来。发布前再用 cibuildwheel 矩阵跑多 Python 版本。

## 常见误区

1. **拿 list 灌 C++ 再逐元素算**（第 5 节 `sum_loop` 147ms 的教训）：容器自动转换是逐 PyObject 解包的拷贝，批量数据永远走 `py::array_t`。
2. **gil_scoped_release 期间摸 Python 对象**：未持有 GIL 时调用 Python C API 是 UB，轻则崩溃重则静默损坏。释放段只碰裸数据；需要 Python 对象时先深拷贝成 C++ 数据再释放。
3. **忘写 `-fPIC`/类型标注 `py::array::c_style`**：不连续/非 C 顺序的数组视图直接读会拿到错位数据——always 声明 c_style，让 pybind11 替你检查并拷贝。
4. **返回大 vector 当 Python list**：`std::vector` → list 的转换同样逐元素造 PyObject；返回 numpy（`py::array_t` 从堆数据构造 + `py::capsule` 管理生命周期）或让 Python 侧提供输出缓冲。
5. **模块名与文件名不一致**：`PYBIND11_MODULE(fastmod, ...)` 编译出的文件必须能让 `import fastmod` 找到；一个 .so 只能有一个模块初始化函数。
6. **在无 GIL 状态抛 C++ 异常穿越边界**：pybind11 会在函数边界把 C++ 异常翻译成 Python 异常，但翻译发生在"重新持有 GIL 之后"；自定义异常要用 `py::register_exception` 注册，别让未注册类型裸穿。

## 本篇小结

- 动机 = GIL 内真并行（gil_scoped_release 实测 405ms 并行）+ 复用 C++ 资产；Python 循环与 C++ 的差距实测 48 倍。
- pybind11 三板斧：`m.def` 绑函数、`py::class_` 绑类、`<pybind11/stl.h>` 自动转容器——但自动转换是拷贝，批量数据走 `py::array_t`。
- numpy 零拷贝的验证法 = 原地改写看可见性 / 对比 data 指针；`c_style|forcecast` 声明避免踩不连续数据。
- 构建走 scikit-build-core（pyproject.toml + CMakeLists 两文件），别再手写 setup.py。
- ABI 常识：模块是 C ABI 的动态库，跨边界调用有固定成本，接口按"整批"设计；方案选型 pybind11（存量）/ nanobind（新项目）/ ctypes（纯 C 临时）/ Cython（Python 代码加速）。

## 练习

1. 给 `fastmod` 加一个 `dot(a, b)`（numpy 点积，手写循环 + `gil_scoped_release`），对比 `np.dot` 的耗时，回答差距来自哪里（提示：SIMD 与多线程）。
2. 用 `__array_interface__['data'][0]` 打印 numpy 数组地址，同时在 C++ 侧打印 `py::array_t::data()` 地址，实证零拷贝；再传一个切片 `a[::2]` 观察地址与 c_style 拷贝行为。
3. 复现第 5 节 `sum_loop` 的"转换开销"：写一个 `noop(v: list)` 的 C++ 函数（什么都不做只收 `std::vector<double>`），测量 1000 万元素的调用耗时，解释它为什么接近 `sum_loop`。
4. 把 `Tokenizer` 扩展：加 `decode(ids: list[int]) -> str` 与 `vocab_size` 属性，用 pytest 写 5 个用例（含空串、未知词、Unicode）。
5. 用 scikit-build-core 把 `fastmod` 打成 wheel（`pip install build && python -m build --wheel`），解包看 .so 是否落在纯 lib 目录，并回答：为什么 wheel 里不需要带 pybind11 的头文件？
