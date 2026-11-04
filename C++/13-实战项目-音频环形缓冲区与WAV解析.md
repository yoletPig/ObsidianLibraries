# 13-实战项目-音频环形缓冲区与WAV解析

## 本篇回答的问题

- 前面 12 篇的知识，怎么组装成一个真正的语音工程组件？
- 如何从零写一个线程安全、定容、零运行期分配的音频 `RingBuffer`（生产者写样本，消费者读定长帧）？
- WAV 的 44 字节头怎么解析出采样率/位深/声道数并算 RMS？结构体直接 `read` 进内存有什么对齐坑，`static_assert(sizeof(...) == 44)` 为什么能救命？
- 多文件 C++ 工程怎么用 CMake 组织和编译？

这是整个系列的收官实战。两个项目都不依赖任何第三方库，全部代码在 Ubuntu 22.04 + g++ 11 下真实编译运行，输出均为实测结果。

## 项目 A：线程安全音频环形缓冲区 RingBuffer

### 需求与设计

音频流水线里，采集线程和推理线程之间需要一个缓冲区，特征是：

- **生产者**写入任意长度的样本块（采集回调每次给 256/512 个样本，长度不定）；
- **消费者**按**固定帧长**读取（比如 480 样本 = 30ms @ 16kHz，这是 ASR 特征提取的标准节奏）；
- 缓冲满/空时**阻塞等待**而不是丢数据；
- 容量固定，**运行期零内存分配**（实时音频线程里分配内存会造成卡顿）。

"定长数组 + head/tail 下标取模"就是环形缓冲区；"多线程阻塞"就是第 12 篇的 mutex + condition_variable；"模板"让 `float` 样本、`int16_t` 样本都能用同一个类。四样合起来：

### 完整实现

```cpp
// ring_buffer.cpp
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(std::size_t capacity)
        : buf_(capacity) {} // RAII：构造即拥有资源，无需手动 new/delete

    // 生产者接口：写入一批样本。缓冲满时阻塞，直到有空位或生产者端关闭。
    bool push(const T* data, std::size_t n) {
        std::unique_lock<std::mutex> lock(mtx_);
        can_push_.wait(lock, [&] { return size_ + n <= buf_.size() || closed_; });
        if (closed_) return false;
        for (std::size_t i = 0; i < n; ++i) {
            buf_[(head_ + i) % buf_.size()] = data[i];
        }
        head_ = (head_ + n) % buf_.size();
        size_ += n;
        lock.unlock();
        can_pop_.notify_one(); // 拿着锁也可以 notify，但锁外 notify 通常更高效
        return true;
    }

    // 消费者接口：读取一个定长帧。样本不够时阻塞，直到凑够或生产者端关闭。
    bool pop_frame(T* out, std::size_t n) {
        std::unique_lock<std::mutex> lock(mtx_);
        can_pop_.wait(lock, [&] { return size_ >= n || closed_; });
        if (size_ < n) return false; // 已关闭且剩余不足一帧
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = buf_[tail_];
            tail_ = (tail_ + 1) % buf_.size();
        }
        size_ -= n;
        lock.unlock();
        can_push_.notify_one();
        return true;
    }

    // 生产者收工后调用：唤醒所有等待者，让消费者把余量取完再退出
    void close() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            closed_ = true;
        }
        can_push_.notify_all();
        can_pop_.notify_all();
    }

private:
    std::vector<T> buf_;                // 底层存储，容量固定，零运行期分配
    std::size_t head_ = 0;              // 下一个写入位置
    std::size_t tail_ = 0;              // 下一个读取位置
    std::size_t size_ = 0;              // 当前元素个数（head/tail 之差的另一面）
    bool closed_ = false;
    std::mutex mtx_;
    std::condition_variable can_push_, can_pop_;
};

constexpr int kSampleRate = 16000;
constexpr int kFrameSize = 480;   // 30 ms @ 16kHz
constexpr int kTotalSamples = 16000; // 模拟 1 秒音频

int main() {
    RingBuffer<float> rb(4096);
    std::atomic<int> frames_read{0};

    // 消费者：像推理线程一样按帧取数据
    std::thread consumer([&] {
        std::vector<float> frame(kFrameSize);
        while (rb.pop_frame(frame.data(), frame.size())) {
            ++frames_read;
        }
    });

    // 生产者：像音频采集回调一样持续写入 1 秒 440Hz 正弦波
    std::thread producer([&] {
        std::vector<float> chunk(1024);
        for (int sent = 0; sent < kTotalSamples; sent += static_cast<int>(chunk.size())) {
            std::size_t n = std::min(chunk.size(),
                                     static_cast<std::size_t>(kTotalSamples - sent));
            for (std::size_t i = 0; i < n; ++i) {
                chunk[i] = 0.25f * std::sin(2.0f * 3.14159265f * 440.0f *
                                            static_cast<float>(sent + i) / kSampleRate);
            }
            rb.push(chunk.data(), n);
        }
        rb.close();
    });
    producer.join();
    consumer.join();
    std::cout << "总样本 " << kTotalSamples << "，共读取 "
              << frames_read.load() << " 个完整帧（"
              << frames_read.load() * kFrameSize << " 个样本）\n";
    std::cout << "预期帧数 = " << kTotalSamples / kFrameSize << "\n";
    return 0;
}
```

编译运行（多次运行输出稳定）：

```bash
g++ -std=c++17 -Wall -Wextra -O2 -pthread ring_buffer.cpp -o ring_buffer
./ring_buffer
```

```
总样本 16000，共读取 33 个完整帧（15840 个样本）
预期帧数 = 33
```

16000 / 480 = 33.33，所以是 33 个完整帧，剩下的 160 个样本不足一帧，被 `close()` 协议安全丢弃——真实引擎里这正是"最后一帧不完整就丢弃/补零"的决策点。

### 设计要点

- **为什么维护 `size_` 而不用 `head_ == tail_` 判空/满**：环形缓冲区的经典歧义是"空"和"满"时 head 可能等于 tail。单独维护 `size_` 让判空判满都是 O(1) 且无歧义，代价是 push/pop 都要同步更新它（反正都要持锁，没有额外成本）。**谓词是 lambda 捕获 `this` 成员**：wait 每次醒来在锁内重新检查，防虚假唤醒与通知丢失（第 12 篇第六节）。
- **`push` 里整块拷贝 + 取模**：演示版按样本逐个取模，简单正确，工业实现会做"分两段 memcpy 绕回"的优化。**`close()` 是收工协议**：只改标志不唤醒，等待中的线程永远睡死；先拿锁改标志，再 `notify_all`，两个方向的等待者都能退出。

## 项目 B：最小 WAV 解析器

### WAV 文件的 44 字节头

标准 PCM WAV = 44 字节头 + 原始样本数据。字段全部**小端序**（x86 恰好也是小端，这既是巧合也是坑——见后文）：

| 偏移 | 大小 | 字段 | 本例值 |
|---|---|---|---|
| 0 | 4 | "RIFF" | `52 49 46 46` |
| 4 | 4 | 文件大小 - 8 | 32036 |
| 8 | 4 | "WAVE" | `57 41 56 45` |
| 12 | 4 | "fmt " | — |
| 20 | 2 | 音频格式（1 = PCM，fmt 长度固定 16） | 1 |
| 22 | 2 | 声道数 | 1 |
| 24 | 4 | 采样率 | 16000 |
| 28 | 4 | 字节率 = 采样率×声道×位深/8 | 32000 |
| 32 | 2 | 块对齐 = 声道×位深/8 | 2 |
| 34 | 2 | 位深 | 16 |
| 36 | 4 | "data" | — |
| 40 | 4 | 数据字节数 | 32000 |

用 `xxd -l 44 test.wav` 看到的正是这张表（对照 24 偏移的 `80 3e 00 00` = 0x3E80 = 16000）。

### 生成测试 WAV

不依赖任何音频文件，一条 python3 命令生成 1 秒 440Hz 正弦波（16kHz / 16bit / 单声道，教程读者可直接复现）：

```bash
python3 -c "
import wave, struct, math
with wave.open('test.wav','wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    frames = b''.join(struct.pack('<h', int(8000*math.sin(2*math.pi*440*i/16000))) for i in range(16000))
    w.writeframes(frames)
"
```

正弦振幅 8000 / 32768 ≈ 0.244，RMS 理论值 = 0.244/√2 ≈ **0.1726**——解析器的输出将精确验证这一点。

### 字节对齐的坑：为什么不能闭眼 `read` 结构体

最直觉的写法是定义一个和文件布局一模一样的结构体（完整定义见下面的 wav_parser.cpp），一次 `file.read(&hdr, sizeof(hdr))` 灌进去。

坑在哪：**编译器为了让 4 字节成员落在 4 的倍数地址上，会在成员之间塞填充字节**。比如把 `fmt_size`（4 字节）和两个 `uint16_t` 的排列稍作调整，`sizeof` 就可能从 44 变成 48——一次读 48 字节，后面所有字段整体错位 4 字节，采样率读出来是个天文数字。

三种对策（按推荐程度）：1. **逐字节组装**（本项目采用，完全可移植）——`read_u32_le(p) = p[0] | p[1]<<8 | p[2]<<16 | p[3]<<24`，自己控制每一个字节，顺带把大小端问题一并解决；2. **`#pragma pack(1)`**——告诉编译器禁止填充，`sizeof` 强制为 44，跨编译器需配合 `#pragma pack(push,1) ... #pragma pack(pop)` 使用，且某些架构上非对齐访问有性能或崩溃代价；3. **`memcpy` 逐字段搬运**——读原始字节缓冲区再 `memcpy` 到各字段，本质和方案 1 相同。

本项目两个都用：结构体映射做"教学演示"，逐字节组装做"可移植复核"，并让 `static_assert(sizeof(WavHeader) == 44)` 把"我的编译器没有塞填充"这件事从假设变成编译期事实——**一旦某天换编译器/换平台出现填充，程序直接编译不过，而不是运行时读出乱码**。

### 完整实现

```cpp
// wav_parser.cpp
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

// 44 字节标准 PCM WAV 头。
// 关键点：这个结构体在 x86-64 上恰好没有内部填充，sizeof == 44；
// 但可移植性不能靠运气，下面用 static_assert 把"sizeof 必须是 44"钉死。
struct WavHeader {
    char     riff[4];        // "RIFF"
    uint32_t chunk_size;     // 文件大小 - 8
    char     wave[4];        // "WAVE"
    char     fmt[4];         // "fmt "
    uint32_t fmt_size;       // PCM 固定为 16
    uint16_t audio_format;   // 1 = PCM
    uint16_t num_channels;   // 声道数
    uint32_t sample_rate;    // 采样率，如 16000
    uint32_t byte_rate;      // = sample_rate * num_channels * bits/8
    uint16_t block_align;    // = num_channels * bits/8
    uint16_t bits_per_sample;// 位深，如 16
    char     data[4];        // "data"
    uint32_t data_size;      // 样本数据字节数
};
static_assert(sizeof(WavHeader) == 44, "WAV 头必须是 44 字节，出现意外填充");

// 小端整数读取：跨平台最稳的方式不是结构体映射，而是逐字节组装
uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
uint16_t read_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "用法: " << argv[0] << " <test.wav>\n";
        return 1;
    }

    std::ifstream file(argv[1], std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件: " << argv[1] << '\n';
        return 1;
    }

    // 方式一（教学演示）：一次读入整个头，逐字段检查
    WavHeader hdr{};
    file.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (file.gcount() != static_cast<std::streamsize>(sizeof(hdr))) {
        std::cerr << "文件太短，连 44 字节头都凑不齐\n";
        return 1;
    }
    if (std::memcmp(hdr.riff, "RIFF", 4) != 0 || std::memcmp(hdr.wave, "WAVE", 4) != 0) {
        std::cerr << "不是 WAV 文件（RIFF/WAVE 标记不符）\n";
        return 1;
    }
    if (hdr.audio_format != 1) {
        std::cerr << "只支持 PCM（audio_format 应为 1，实际 " << hdr.audio_format << "）\n";
        return 1;
    }

    std::cout << "声道数     : " << hdr.num_channels << '\n'
              << "采样率     : " << hdr.sample_rate << " Hz\n"
              << "位深       : " << hdr.bits_per_sample << " bit\n"
              << "字节率     : " << hdr.byte_rate << " B/s\n"
              << "块对齐     : " << hdr.block_align << " B/帧\n"
              << "数据字节数 : " << hdr.data_size << '\n';

    // 方式二（更可移植）：用逐字节组装复核关键字段，结果应与方式一一致
    file.seekg(0);
    uint8_t raw[44];
    file.read(reinterpret_cast<char*>(raw), 44);
    uint32_t sr2 = read_u32_le(raw + 24);
    uint16_t ch2 = read_u16_le(raw + 22);
    std::cout << "逐字节复核 : " << ch2 << " 声道, " << sr2 << " Hz\n";

    // 读取 16-bit 样本并计算 RMS（语音能量特征的起点）
    std::size_t num_samples = hdr.data_size / (hdr.bits_per_sample / 8 * hdr.num_channels);
    std::vector<int16_t> samples(num_samples);
    file.read(reinterpret_cast<char*>(samples.data()),
              static_cast<std::streamsize>(hdr.data_size));
    if (file.gcount() != static_cast<std::streamsize>(hdr.data_size)) {
        std::cerr << "警告：实际读到 " << file.gcount() << " 字节（头声称 "
                  << hdr.data_size << "）\n";
        return 1;
    }

    double sum_sq = 0.0;
    for (int16_t s : samples) {
        double norm = static_cast<double>(s) / 32768.0; // 归一化到 [-1, 1)
        sum_sq += norm * norm;
    }
    double rms = std::sqrt(sum_sq / static_cast<double>(samples.size()));
    std::cout << "样本数     : " << samples.size() << '\n'
              << "RMS        : " << rms << "（正弦波理论值 ≈ 0.1726）\n";
    return 0;
}
```

真实运行输出：

```
声道数     : 1
采样率     : 16000 Hz
位深       : 16 bit
字节率     : 32000 B/s
块对齐     : 2 B/帧
数据字节数 : 32000
逐字节复核 : 1 声道, 16000 Hz
样本数     : 16000
RMS        : 0.172622（正弦波理论值 ≈ 0.1726）
```

RMS 0.172622 与理论值 0.1726 吻合——解析、归一化、能量计算整条链路正确。

## 用 CMake 组织两个项目

工程有两个可执行文件、一个要用线程库一个不用，手写 g++ 命令开始变得啰嗦，这是引入 CMake 的时机。目录结构：`audio_demo/` 下放 `CMakeLists.txt`、`ring_buffer.cpp`、`wav_parser.cpp` 和 `test.wav`（由上面的命令生成）。

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(audio_demo CXX)

set(CMAKE_CXX_STANDARD 17)          # 对应教程的 C++17 标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)       # 用 -std=c++17 而不是 GNU 扩展

find_package(Threads REQUIRED)      # 线程库：Linux 上对应 -pthread

add_executable(ring_buffer ring_buffer.cpp)
target_link_libraries(ring_buffer PRIVATE Threads::Threads)
target_compile_options(ring_buffer PRIVATE -Wall -Wextra)

add_executable(wav_parser wav_parser.cpp)   # 纯文件解析，无需线程库
target_compile_options(wav_parser PRIVATE -Wall -Wextra)
```

真实构建过程与运行（在项目目录下）：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/ring_buffer
./build/wav_parser test.wav
```

输出与前两节完全一致（`ring_buffer` 33 帧、`wav_parser` RMS 0.172622）。注意 `find_package(Threads REQUIRED)` + `Threads::Threads` 就是"`-pthread`"的 CMake 写法——第 12 篇强调的坑在这里被框架接管了。

## 知识点回溯：本篇用到了前面哪些内容

| 本篇用到的 | 来自 |
|---|---|
| `explicit` 构造、成员初始化、RAII 资源管理（构造即拥有 `vector` 存储） | 03-类与对象（RAII 思想贯穿 04/12） |
| `std::vector` 预分配、`std::deque`、`std::min`；`std::move(frame)` 移动大块样本、lambda 按引用捕获 | 08-STL、05-移动语义、09-lambda |
| `template <typename T>` 类模板：`RingBuffer<T>` 对 `float`/`int16_t` 通用；`static_assert` 编译期检查 `sizeof` | 10-函数模板、11-类模板 |
| `std::thread` / `mutex` / `condition_variable` / `atomic` / 生产者-消费者 | 12-并发编程 |
| `std::ifstream` 二进制读取、`reinterpret_cast` 的边界使用 | 07-文件与流 |
| 常量约定 `kFrameSize`、无 `using namespace std`、RAII 锁 | 全系列代码规范 |

两个项目合起来，正是一个最小语音引擎的雏形：**RingBuffer 是采集→推理的交接带，WAV 解析器是离线数据的入口**，把它们拼在一个 main 里（WAV → RingBuffer → 按帧读出），你就有了第一个能跑通数据流的 C++ 语音前端。

## 常见坑

1. **忘了 `close()` 就结束生产者**：消费者在"样本不够"的 wait 里永久睡死。收工协议（标志 + notify_all）必须成对出现。**`RingBuffer` 容量小于帧长**同理：`size_ + n <= capacity` 永远不成立，双方互相等待，容量至少是帧长的若干倍（本例 4096 >> 480）。
2. **实时线程里 `new`/`vector` 扩容**：音频回调中堆分配可能带来毫秒级卡顿，`RingBuffer` 构造时一次分配固定容量的意义就在于此。
3. **把 WAV 头结构体 `read` 进去就信了**：填充字节、字节序、字段缺失（有些 WAV 的 fmt 块不是 16 字节、头不止 44 字节）都会让解析错位。生产代码要循环解析 chunk（本教程的最小版只处理 44 字节标准头），并保留逐字节组装这条后路。删掉 `static_assert` 更糟——错误被推迟到运行时的乱码。
4. **16-bit 样本忘记归一化**：直接把 `int16_t` 当浮点特征喂给模型，数值范围差 3 万倍。先除以 32768。
5. **CMake 忘记链接线程库**：链接期报 `pthread_create` 未定义。`find_package(Threads REQUIRED)` + `Threads::Threads` 是标准解法。

## 小结

- `RingBuffer` 把第 12 篇的 mutex + condition_variable 生产者-消费者模式模板化，加上固定容量与 head/tail 环形下标，成为可复用的音频组件；收工靠 `close()` 的"标志 + 唤醒"协议。
- WAV 头是逐字段定义的二进制格式；解析二进制的两条正路是"逐字节组装（最可移植）"和"`#pragma pack(1)` / memcpy（要防填充）"，`static_assert(sizeof == 44)` 把布局假设钉死在编译期。
- RMS = 样本平方和均值的开方，是语音能量的第一特征；16-bit 样本先除以 32768 归一化。CMake 用 `CMAKE_CXX_STANDARD 17` 和 `Threads::Threads` 把标准与 `-pthread` 的坑系统性解决。
- 至此系列收官：从变量与指针，到 RAII 与移动语义，到 STL 与模板，到并发，最终落成两个能编译、能运行、输出可验证的真实组件。

## 自测题

1. `RingBuffer` 里如果删掉 `size_`，改用 `head_ == tail_` 判断，会遇到什么经典问题？
2. `pop_frame` 的 wait 谓词是 `size_ >= n || closed_`。如果写成只有 `size_ >= n`，生产者 `close()` 后消费者会发生什么？
3. 为什么 WAV 头结构体在某些编译器下 `sizeof` 会大于 44？本项目用哪两个手段防御？
4. `read_u32_le` 为什么要逐字节组装，而不能 `*reinterpret_cast<const uint32_t*>(p)` 一把读？
5. 16kHz、16-bit、双声道的 WAV，`byte_rate` 和 `block_align` 各是多少？
6. 本例 16000 个样本只读出 33 帧共 15840 个样本，剩下的去哪了？真实引擎里通常怎么处理？

### 答案

1. 空和满时 head 都可能等于 tail，无法区分；要么牺牲一个槽位，要么像本项目一样额外维护 `size_`。
2. `close()` 只会唤醒等待者，但谓词仍为 false，消费者会继续睡死（新样本永远不会来了），程序挂住不退出。谓词里的 `closed_` 加上"唤醒后重查"才是完整退出条件。
3. 编译器会在成员间插入填充字节使成员对齐到自身大小的倍数地址。防御：`static_assert(sizeof(WavHeader) == 44)`（编译期哨兵）+ 逐字节组装 `read_u16_le/read_u32_le`（不依赖结构体布局）。
4. 逐字节组装不受结构体填充、编译器对齐策略影响，还显式定义了字节序（WAV 是小端）；直接指针重解释假定"对齐正确 + 主机是小端"，换平台就可能出错甚至崩溃。
5. `block_align` = 2 声道 × 2 字节 = 4 字节/帧；`byte_rate` = 16000 × 4 = 64000 字节/秒。
6. 最后 160 个样本不足一个 480 样本的帧，`pop_frame` 在 `closed_` 后因 `size_ < n` 返回 false 被丢弃。真实引擎通常补零凑满一帧送模型，或在结果侧丢弃该不完整帧。
