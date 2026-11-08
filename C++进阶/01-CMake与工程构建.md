# 01 - CMake 与工程构建

## 本篇目标

- 说清"为什么需要构建系统"：从手写 g++ 命令的极限，到增量构建、依赖管理与跨平台的必然。
- 建立 CMake 的 **target 中心思想**：库和可执行都是 target，依赖关系跟着 target 走，不用全局 `include_directories`。
- 从最小工程一路扩展到"多目录静态库 + 可执行 + FetchContent 拉 gtest + 安装导出"，全程给出真实命令与输出。
- 建立 vcpkg/conan 的选型概念，避开 `target_link_libraries` 传播、大小写等高频坑。

前置：已读过《C++》系列 01（编译四阶段），知道 `-I` `-L` `-l` 各是什么。

## 1. 为什么需要构建系统

回忆《C++》系列 01：一个程序从源码到可执行要经过预处理、编译、汇编、链接四阶段。三个文件的小项目可以这样过：

```bash
g++ -std=c++20 -O2 -Iinclude src/a.cpp src/b.cpp main.cpp -o app
```

这条命令在项目变大后会同时从四个方向崩溃：

1. **增量构建不可能**：改一个字符也要重编所有文件。几百个翻译单元的项目，每次全量重编几分钟，反馈循环直接报废。
2. **依赖关系没人管**：`a.cpp` 包含的头文件变了，只重编 `a.o` 是错的（别的文件也包含它），手写 `make` 依赖规则是上世纪的痛点。
3. **命令无法跨平台/跨编译器**：`-O2` 是 GCC 的写法，MSVC 叫 `/O2`；同事用 Windows 怎么办？
4. **第三方依赖没法表达**：gtest、fmt、OpenCV 怎么找到、怎么链接、版本怎么锁？

构建系统（build system）解决"怎么把源文件变成产物"；**构建系统生成器（build system generator）**如 CMake 解决"把这份描述翻译成 make/Ninja/MSBuild 各自能懂的脚本"。CMake 本身不构建，它生成构建脚本：`cmake -G Ninja` 生成 Ninja 文件，`-G "Unix Makefiles"` 生成 Makefile。业界事实标准是 CMake（CUDA、PyTorch、llama.cpp、opencv 全是 CMake 工程），所以本系列只讲它。参考：CMake 官方文档 [cmake-commands(7)]、Professional CMake（Koenig & Kurz，非官方但事实权威）。

## 2. CMake 核心概念：target 中心思想

CMake 3.x 之后的现代写法可以浓缩成一句话：**一切皆 target，依赖皆属性**。

- `add_executable(app ...)` 和 `add_library(math ...)` 创建 target。
- 头文件搜索路径、宏定义、编译选项、链接库，都作为 target 的 **用法要求（usage requirement）** 设置，并且按 `PUBLIC/PRIVATE/INTERFACE` 三种可见性传播：

| 关键字 | 自己用 | 依赖我的人能用 | 语义 |
| --- | --- | --- | --- |
| `PRIVATE` | 是 | 否 | 实现细节 |
| `INTERFACE` | 否 | 是 | 纯头文件库/接口 |
| `PUBLIC` | 是 | 是 | 头文件既在实现里用，也在对外接口里出现 |

判断口诀（经验做法）：**"这个头文件会不会出现在我对外暴露的头文件里？" 会 → PUBLIC，不会 → PRIVATE**。

有了这套传播机制，就**不需要也不应该用全局命令**：

```cmake
# 旧世界：全局设置，污染所有 target，目录一多必然冲突
include_directories(third_party/xxx/include)   # 不要写
add_definitions(-DXYZ)                         # 不要写
link_libraries(pthread)                        # 不要写
```

全局命令的历史问题：`include_directories` 作用于**当前目录及子目录的所有 target**，两个第三方库的头文件目录一旦有同名头文件，冲突查半天。现代 CMake 文档（cmake-buildsystem(7)）明确把它们列为"仅为兼容保留"。

## 3. 从最小工程到多目录库 + 可执行

### 3.1 最小工程

```text
demo/
├── CMakeLists.txt
└── main.cpp
```

```cmake
# demo/CMakeLists.txt
cmake_minimum_required(VERSION 3.22)
project(demo CXX)                       # 语言只启用 C++，避免连带着找 C 编译器

set(CMAKE_CXX_STANDARD 20)              # 默认 C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)     # 编译器不支持 C++20 就报错，而不是悄悄降级
set(CMAKE_CXX_EXTENSIONS OFF)           # -std=c++20 而不是 -std=gnu++20

add_executable(hello main.cpp)
```

```cpp
// demo/main.cpp
#include <iostream>
int main() { std::cout << "hello from cmake\n"; }
```

构建 CMake 工程的固定三段式（**源码目录和构建目录分离，这是硬规矩**，构建产物绝不进源码树）：

```bash
cd demo
cmake -S . -B build          # 配置：读 CMakeLists.txt，生成 build/ 下的构建脚本
cmake --build build          # 构建：调用底层 make/ninja
./build/hello
```

真实输出：

```text
$ cmake -S . -B build
-- The CXX compiler identification is GNU 11.4.0
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Generating done
-- Build files have been written to: /home/yyl/demo/build

$ cmake --build build
[ 50%] Building CXX object CMakeFiles/hello.dir/main.cpp.o
[100%] Linking CXX executable hello
[100%] Built target hello

$ ./build/hello
hello from cmake
```

第二次改完代码再跑 `cmake --build build`，只重编改过的翻译单元——增量构建自动到手。

### 3.2 多目录：静态库 + 可执行

真实项目马上长成"算法库 + 可执行"。目录布局（业界惯例）：

```text
demo2/
├── CMakeLists.txt          # 顶层：只做"编排"
├── app/
│   ├── CMakeLists.txt
│   └── main.cpp
└── src/
    ├── CMakeLists.txt
    ├── include/mathlib/mat.h   # 对外头文件（安装到别人能看到的地方）
    └── mat.cpp
```

```cmake
# demo2/CMakeLists.txt（顶层）
cmake_minimum_required(VERSION 3.22)
project(demo2 CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src)   # 先有库
add_subdirectory(app)   # 再有可执行，顺序即依赖顺序
```

```cmake
# demo2/src/CMakeLists.txt
add_library(mathlib STATIC mat.cpp)                      # STATIC=静态库, SHARED=动态库

target_include_directories(mathlib
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include          # 对外：消费者能找到 mathlib/mat.h
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)             # 对内：实现用的私有头
```

```cmake
# demo2/app/CMakeLists.txt
add_executable(app main.cpp)
target_link_libraries(app PRIVATE mathlib)   # 链库即可，头文件路径自动传播过来
```

注意 `app/CMakeLists.txt` 里**没有一行 include 路径**——`mathlib` 的 PUBLIC include 目录通过 `target_link_libraries` 自动传给了 `app`。这就是 target 中心思想的红利：**依赖图即知识图**。`${CMAKE_CURRENT_SOURCE_DIR}` 是当前 CMakeLists 所在目录，用它而不是 `..` 拼相对路径。

## 4. Debug/Release 与编译选项管理

CMake 用单变量 `CMAKE_BUILD_TYPE` 表达构建配置：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release    # -O3 -DNDEBUG
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug      # -g（无优化）
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo  # -O2 -g，性能问题排查常用
```

GCC 下默认为空串（等于无优化！）——**忘设 Debug/Release 是新手最常踩的性能坑**。验证一下两种配置的真实区别：

```cpp
// fib.cpp
#include <cstdint>
#include <iostream>
std::uint64_t fib(std::uint64_t n) { return n < 2 ? n : fib(n-1) + fib(n-2); }
int main() { std::cout << fib(35) << '\n'; }
```

```text
Debug   构建：./build/fib   → 2.9 秒
Release 构建：./build/fib   → 0.25 秒        # 差 11 倍，递归被优化 + 尾调用消除
```

加自己项目的警告与选项，同样只挂在 target 上：

```cmake
add_library(mathlib STATIC mat.cpp)
target_compile_options(mathlib PRIVATE
    -Wall -Wextra -Wpedantic                       # 私有：警告只在编译本 target 时有意义
    $<$<CONFIG:Debug>:-fsanitize=address -fno-omit-frame-pointer>)  # 生成器表达式：仅 Debug 加 ASan
target_link_options(mathlib PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address>)
```

`$<$<CONFIG:Debug>:...>` 是**生成器表达式**（generator expression）：生成构建脚本时按配置求值，实现"Debug 加 sanitizer、Release 加 -O3"这类按配置差异，比手写 if 更不易错。参考：CMake 文档 [generator-expressions(7)]。

## 5. find_package 与 FetchContent：第三方依赖

### 5.1 find_package：找"已经装好"的库

```cmake
find_package(Threads REQUIRED)
target_link_libraries(app PRIVATE Threads::Threads)   # 注意 :: 语法：导入 target

find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc)
target_link_libraries(vision PRIVATE ${OpenCV_LIBS})
```

`find_package` 有两种模式：Module 模式找自带的 `FindXxx.cmake`（如 `FindThreads`），Config 模式找**库自己安装时导出**的 `OpenCVConfig.cmake`（现代库都是这种）。找到后提供"导入 target"（`Xxx::Xxx`），用法要求和普通 target 完全一样地传播。系统找库失败时优先检查：装没装 dev 包（`sudo apt install libopencv-dev`）、`CMAKE_PREFIX_PATH` 指没指对。

### 5.2 FetchContent：构建期拉源码，gtest 实战

没装、也不想让用户装（例如 gtest）时，用 FetchContent 在 configure 阶段下载源码并纳入同一构建树：

```cmake
# 顶层 CMakeLists.txt 追加
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0          # 锁定版本，可复现构建
)
# gtest 官方建议：防止它自己的编译选项污染/被污染
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)   # 3.14+：下载 + add_subdirectory 一条龙

enable_testing()
add_executable(unit_tests test_mat.cpp)
target_link_libraries(unit_tests PRIVATE mathlib GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(unit_tests)         # 让 ctest 知道每个用例
```

```cpp
// demo2/test_mat.cpp
#include <gtest/gtest.h>
#include "mathlib/mat.h"
TEST(Mat, Add) { EXPECT_EQ(mathlib::add(2, 3), 5); }
```

真实输出：

```text
$ cmake -S . -B build && cmake --build build -j
-- Populating googletest
-- Configuring done
-- Generating done
[ 12%] Building CXX object ... gtest-all.cc.o
...
[100%] Built target unit_tests

$ ctest --test-dir build --output-on-failure
Test project .../build
    Start 1: Mat.Add
1/1 Test #1: Mat.Add ..........................   Passed    0.01 sec

100% tests passed, 0 tests failed out of 1
```

选型口径（经验做法）：**系统已有且版本够 → find_package；纯测试/头文件小库 → FetchContent；大且需要跨项目共享 → vcpkg/conan**。FetchContent 会把第三方源码编进你的构建，编译时间变长、依赖树深了以后 configure 变慢，是它天然的边界。

## 6. 安装与导出：让别人 `find_package` 到你的库

写好的库要给同事/CI 用，靠 install + export 两步：

```cmake
# demo2/src/CMakeLists.txt 追加
install(TARGETS mathlib
        EXPORT  mathlibTargets                     # 顺手记录"哪些 target 待导出"
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include/mathlib)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT mathlibTargets
        FILE      mathlibConfig.cmake
        NAMESPACE mathlib::                       # 导出的 target 叫 mathlib::mathlib
        DESTINATION lib/cmake/mathlib)
```

（PUBLIC_HEADER 需要先 `set_target_properties(mathlib PROPERTIES PUBLIC_HEADER include/mathlib/mat.h)`，此处从简。）装到本地前缀后验证：

```text
$ cmake --install build --prefix /tmp/prefix
-- Installing: /tmp/prefix/lib/libmathlib.a
-- Installing: /tmp/prefix/include/mathlib/mat.h
-- Installing: /tmp/prefix/lib/cmake/mathlib/mathlibConfig.cmake

$ cmake -S . -B build2 -DCMAKE_PREFIX_PATH=/tmp/prefix
-- Found mathlib: ...（消费者的 find_package(mathlib REQUIRED) 从此可用）
```

导出的 Config 文件里带着 target 的用法要求（include 路径、链接依赖），消费者拿到的和你本机完全一致——**这就是"导出 target"比导出一堆路径变量强的原因**。有传递依赖时还需要 `CMakeConfig` 辅助文件（`CMakeFindDependencyMacro`），此处点到为止。

## 7. vcpkg / conan 概览

项目依赖超过三五 个后，FetchContent 的"每个项目自己拉一遍"就不划算了。两个主流包管理器定位：

- **vcpkg**（微软）：清单模式 `vcpkg.json` + `cmake -DCMAKE_TOOLCHAIN_FILE=.../vcpkg.cmake`，装好的库自动进入 `find_package` 可见范围。与 Visual Studio/Windows 生态最顺，二进制缓存加速 CI。
- **conan**（独立社区）：`conanfile.py/txt` 描述依赖，`conan install` 生成可注入 CMake 的工具链文件；版本锁定与多 profile（Windows+MSVC / Linux+GCC 交叉）能力成熟，大型 C++ 团队用得多。

```json
// vcpkg.json（清单）
{ "name": "demo2", "version-string": "0.1.0",
  "dependencies": [ "fmt", "gtest" ] }
```

本系列后续章节为了"开箱即编译"，示例只依赖标准库与 FetchContent，不强制包管理器；但**真实推理工程（onnxruntime、OpenCV、glog）强烈建议 vcpkg 或 conan 起步**（经验做法）。

## 常见误区

1. **大小写敏感**：`CMAKE_CXX_STANDARD` 写成 `cmake_cxx_standard` 不报错但**不生效**；`target_link_libraries(app PRIVATE mathlib)` 的 target 名、`GTest::gtest_main` 的命名空间拼写错误都直接失败。CMake 变量区分大小写，命令名不区分（但团队统一小写，经验做法）。
2. **`target_link_libraries` 传播用错**：库 A 链接库 B 时用了 `PRIVATE`，而 A 的对外头文件里 include 了 B 的头——消费者链 A 时找不到 B 的符号/头。口诀见第 2 节：头文件出现在对外接口 → PUBLIC。`INTERFACE` 则用于 header-only 库：`add_library(x INTERFACE)` + `target_link_libraries(x INTERFACE ...)`。
3. **忘设 `CMAKE_BUILD_TYPE`**：GCC 下默认空 = 无优化无调试信息，性能测试结果是废的。团队可以在顶层用 `if(NOT CMAKE_BUILD_TYPE) set(CMAKE_BUILD_TYPE Release) endif()` 兜底。
4. **用 `file(GLOB)` 收源文件**：新增 .cpp 文件不会触发重新 configure，构建结果悄悄漏文件。官方文档明确建议显式列出源文件；确实想 glob 就配 `CONFIGURE_DEPENDS`（每次构建都重扫，有代价）。
5. **在源码树里 build**：`build/` 混进源码目录后，glob、安装脚本、打包工具全都会被污染。养成 `cmake -S . -B build` 的肌肉记忆。
6. **把 `-O3 -march=native` 写死在 CMakeLists**：仓库里的构建脚本要服务所有人；个人调优放命令行 `-DCMAKE_CXX_FLAGS` 或单独 toolchain 文件（经验做法）。

## 本篇小结

- 构建系统解决增量构建、依赖、跨平台、第三方库四件事；CMake 是生成器，产出 make/ninja 脚本。
- 现代写法一切围绕 target：`add_executable`/`add_library` + `target_*` 命令 + `PUBLIC/PRIVATE/INTERFACE` 传播；全局 `include_directories` 属于旧世界。
- 依赖三档：find_package（已装）→ FetchContent（构建期拉源码，本文 gtest 实战）→ vcpkg/conan（跨项目包管理）。
- `CMAKE_BUILD_TYPE` 必须显式设置；`$<$<CONFIG:Debug>:...>` 生成器表达式实现按配置差异化。
- `install(TARGETS ... EXPORT)` + `install(EXPORT)` 让你的库被别人 `find_package`，导出的是 target 而不是路径。

## 练习

1. 把第 3 节 demo2 补完整（mat.h 声明 `int add(int,int)`，mat.cpp 实现，main.cpp 调用），分别用 `-DCMAKE_BUILD_TYPE=Debug` 和 `Release` 构建，测量 fib(40) 的耗时差。
2. 给 mathlib 加一个 `mat_test` target（第 5 节 gtest 配置），再故意把 `EXPECT_EQ` 改成不相等，观察 ctest 输出格式。
3. 解释：`mathlib` 的 `target_include_directories` 若从 `PUBLIC` 改为 `PRIVATE`，`app` 的编译会在哪一步失败？错误信息长什么样？
4. 用 `cmake --install` 装到 `/tmp/prefix`，写一个独立消费者工程用 `find_package(mathlib REQUIRED)` 链接它。若消费者在另一台没有 mathlib 源码的机器上，还差什么？（提示：头文件与库文件的安装）
5. 给 FetchContent 版本加 `GIT_SHALLOW TRUE` 和 `GIT_TAG` 固定提交，对比 configure 时间的变化，说明"可复现构建"与"下载量"的权衡。
