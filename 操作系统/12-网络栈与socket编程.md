# 12 - 网络栈与socket编程

## 本篇目标

- 建立 TCP/IP 分层的实用心智模型：每一层加什么、socket API 卡在哪一层，并写出完整可跑的 TCP client/server。
- 讲清三次握手与四次挥手的"为什么"，看懂 `ss -tan` 里的状态机，回答 TIME_WAIT 为什么必然大量出现、要不要怕。
- 解决粘包问题：理解"TCP 是字节流不是消息"的根源，掌握定长/分隔符/长度前缀三种方案并用代码实现。
- 对比三种并发服务器模型（每进程/每线程/事件驱动），知道各自的天花板与适用场景，并连接 09 篇的 epoll。

前置：第 09 篇的 IO 模型与 epoll、第 02 篇的进程/线程；环境 Ubuntu 22.04 / WSL2（本篇代码与 Windows 侧互通需注意 WSL2 的 localhost 转发特性，见正文）。

## 1. TCP/IP 分层与 socket API

分层不是背口诀，是**分工表**（TCP/IP 协议族的经典参考：Stevens《TCP/IP 详解 卷1》第 1 章；本篇的 socket 语义以 `man 7 socket`、`man 7 ip`、`man 7 tcp` 为准）：

```
应用层   HTTP / 你的推理协议          ┐ 应用关心：数据长什么样
传输层   TCP(可靠流) / UDP(数据报)     │ socket(SOCK_STREAM/SOCK_DGRAM)
网络层   IP(寻址路由)                 │ socket(AF_INET)
链路层   以太网帧 / WSL2 虚拟网卡      ┘ 内核+网卡操心
```

每层收发时做一件事：**封装**（加头）与**解封装**（剥头）。应用写下的 100 字节，向下依次加 TCP 头（端口/序号）、IP 头（地址）、以太网头（MAC），物理层变比特；接收端反向剥开。socket API 是内核给应用在**传输层与网络层之间开的口子**：

```
socket()          造一个通信端点（fd 家族成员，08 篇"一切皆文件"）
bind()            给端点绑定 地址+端口
listen()/accept() 服务端：进入监听、逐个接客
connect()         客户端：发起连接（三次握手在这里发生）
send()/recv()     读写字节流（socket 进 epoll，09 篇）
close()           发起挥手（四次挥手的开始）
```

完整的 client/server（客户端 30 行、服务端 60 行，可直接编译对跑；这是后面所有实验的骨架）：

```c
// tcp_server.c —— 完整 TCP echo 服务器（单连接版，模型对比见第 4 节）
// 编译：gcc tcp_server.c -o tcp_server && ./tcp_server
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

int main(void) {
    int lfd = socket(AF_INET, SOCK_STREAM, 0);          // 1. 端点
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));  // 见第 2 节 TIME_WAIT
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);           // 0.0.0.0：所有网卡
    addr.sin_port = htons(9001);                        // 端口号要网络字节序
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    if (listen(lfd, 128) < 0) { perror("listen"); return 1; }
    printf("监听 9001 ...\n");

    while (1) {
        int cfd = accept(lfd, NULL, NULL);              // 2. 阻塞等连接（握手完成才返回）
        char buf[1024];
        ssize_t n;
        while ((n = recv(cfd, buf, sizeof(buf), 0)) > 0) {  // 3. 收发循环
            send(cfd, buf, n, 0);                       // echo 回去
        }
        close(cfd);                                     // 4. 对端关闭/出错则断开
    }
}
```

```c
// tcp_client.c —— 配套客户端
// 编译：gcc tcp_client.c -o tcp_client && ./tcp_client
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

int main(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(9001);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);    // 字符串 → 二进制地址
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("connect"); return 1; }

    char msg[] = "hello tcp";
    send(fd, msg, strlen(msg), 0);                      // 写
    char buf[1024];
    ssize_t n = recv(fd, buf, sizeof(buf) - 1, 0);      // 读
    buf[n] = 0;
    printf("echo 返回: %s\n", buf);
    close(fd);
    return 0;
}
```

```
$ gcc tcp_server.c -o tcp_server && gcc tcp_client.c -o tcp_client
$ ./tcp_server &
监听 9001 ...
$ ./tcp_client
echo 返回: hello tcp
$ ss -tanp | grep 9001                     # 看连接状态（第 2 节的主角）
LISTEN  0  128  0.0.0.0:9001  0.0.0.0:*  users:(("tcp_server",pid=...))
ESTAB   0  0  127.0.0.1:9001 127.0.0.1:52840 users:(("tcp_server",pid=...))
```

**WSL2 注**：WSL2 是独立虚拟机，有自己的 IP；从 Windows 访问 WSL2 内的服务用 localhost（WSL2 自动转发），从 WSL2 访问 Windows 服务则要用 Windows 宿主的 IP（`ip route | grep default`）。防火墙拦不住时先检查这一层。

## 2. 三次握手、四次挥手与 TIME_WAIT

握手挥手不是仪式，是**两端各自确认对方能力**的状态机协议（TCP 状态机见 RFC 793 与 Stevens 卷1 第 18 章）：

```
三次握手（建立）                     四次挥手（关闭）
客户端            服务端             客户端            服务端
  │── SYN(seq=x) ──►│                 │── FIN ─────────►│
  │                 │                 │◄──────── ACK ───│
  │◄─ SYN+ACK ──────│                 │◄──────── FIN ───│   服务端可能还有数据要发
  │── ACK ─────────►│                 │── ACK ─────────►│
  ESTABLISHED       ESTABLISHED       TIME_WAIT         CLOSE_WAIT→LAST_ACK
```

为什么是三次不是两次：**双方都要确认"我说的你收到了、你说的我收到了"**；两次握手无法防止历史旧连接的 SYN 迟到重放（旧的重复 SYN 建立幽灵连接——RFC 793 的原始动机）。为什么挥手四次：TCP 全双工，两个方向各自独立关闭，FIN+ACK 常常不能合并（被动方可能还有数据要发，CLOSE_WAIT 就是"我收到了你的关闭但我还没发完"）。

**TIME_WAIT 是主动关闭方的终末状态**：发完最后一个 ACK 后停留 **2MSL**（通常 60 秒，`sysctl net.ipv4.tcp_fin_timeout` 相关）才消失。为什么要等：①确保最后的 ACK 若丢了、对方重发的 FIN 还能被应答；②让旧连接的报文在网络中自然死亡，不污染复用同四元组的新连接。实测它：

```bash
$ ./tcp_client & sleep 0.1; ss -tan | grep 9001 | head -3
ESTAB   0  0  127.0.0.1:52842 127.0.0.1:9001
FIN-WAIT-2 0  0  127.0.0.1:52842 127.0.0.1:9001    ← 客户端关闭后的中间态
（1 分钟内再跑几次 client，可见成批 TIME_WAIT 挂在客户端侧）
```

工程判断（**经验做法**）：TIME_WAIT 挂在**主动关闭的客户端**侧是健康现象，百万级短连接压测时堆积属正常；只有**服务端**大量 TIME_WAIT（服务主动关闭）才要警惕——通常意味着连接被服务端提前 close 了（比如 HTTP/1.0 行为或超时逻辑），先修代码再谈调参。`SO_REUSEADDR` 的作用正是让服务端重启时能绑定仍处 TIME_WAIT 的地址（不与 2.6.12+ 的 SO_REUSEPORT 混淆，后者是负载均衡用）。

## 3. 粘包问题与解决

先破题：**TCP 没有"包"的概念，粘包是应用层的误会**。TCP 是字节流：你的三次 send 在对端可能是 1 次 recv 全收到（合并），也可能是 3 次 recv 各收到一部分（拆开）——内核只保证**字节顺序与完整性**，不保留你的发送边界（MSS、Nagle、缓冲区水位都会切分）。复现它：

```python
# sticky_demo.py —— 复现"三次发送两次收到"
import socket, time

def client():
    s = socket.socket()
    s.connect(("127.0.0.1", 9002))
    for m in ("msg1|", "msg2|", "msg3|"):
        s.sendall(m.encode())            # 三次 send，紧挨着
        time.sleep(0.001)
    s.close()

if __name__ == "__main__":
    import threading, subprocess
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 9002)); srv.listen(1)
    threading.Thread(target=client).start()
    c, _ = srv.accept()
    time.sleep(0.05)                     # 等三条消息在网络缓冲里汇合
    print("一次 recv 收到:", c.recv(1024).decode())   # 可能一口气收到三条
    c.close(); srv.close()
```

```
$ python3 sticky_demo.py
一次 recv 收到: msg1|msg2|msg3|          ← 粘包现场：边界没了
```

三种标准解法（按"协议设计"优先级排序）：

1. **长度前缀**（首选，二进制协议标配）：每条消息前 4 字节表示正文长度，读满长度为止。实现：

```c
// framed_read.c —— 长度前缀拆包：recv_all 读满 n 字节才算一条
// 编译：gcc framed_read.c -o framed_read && ./framed_read
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/socket.h>

static int recv_all(int fd, void *buf, size_t n) {   // 循环读满 n 字节
    size_t got = 0;
    while (got < n) {
        ssize_t r = recv(fd, (char *)buf + got, n - got, 0);
        if (r <= 0) return -1;                       // 0=对端关闭 -1=错误
        got += r;                                    // 关键：recv 返回"这次拿到多少"，不是"你要多少"
    }
    return 0;
}

int read_frame(int fd, char *out, size_t cap) {
    uint32_t len_net;
    if (recv_all(fd, &len_net, 4) < 0) return -1;    // 先读 4 字节长度（网络字节序）
    uint32_t len = ntohl(len_net);
    if (len > cap) return -1;                        // 防恶意超大长度
    return recv_all(fd, out, len);                   // 再读正文，读满为止
}
```

2. **分隔符**：如 HTTP 头的 `\r\n\r\n`、Redis 协议的 CRLF。适合文本协议；正文含分隔符需转义。
3. **定长消息**：如固定 64 字节帧。最简单但浪费/不灵活。

注意 UDP 没有这个问题（数据报天然有边界，`SOCK_DGRAM` 一次 recv 一条消息）——代价是不可靠、不保序。**经验做法**：推理服务的请求/响应协议一律长度前缀（vLLM 的 OpenAI 兼容层之下、gRPC 的 HTTP/2 帧全是这个思想：**帧 = 长度 + 类型 + 负载**）。

## 4. 并发服务器模型：每进程 / 每线程 / 事件驱动

单连接服务器（第 1 节的 tcp_server）一次只能服务一个客户端。三种主流解法（宏观对比；事件驱动的实现细节在 09 篇）：

**模型 A：每连接一个进程**（Apache 经典 prefork 模式）：

```c
while (1) {
    int cfd = accept(lfd, NULL, NULL);
    if (fork() == 0) {              // 子进程处理这一条连接
        close(lfd);                 // 子进程不需要监听 socket
        handle(cfd);
        return 0;                   // 处理完退出（父进程要 wait 或 SIG_IGN SIGCHLD，02 篇）
    }
    close(cfd);                     // 父进程立刻关掉自己的副本，继续 accept
}
```

隔离最强（一个崩不死全家）、内存开销最大、切换成本最高（07 篇）。适合：连接数少、处理重、要求故障隔离的服务（传统 CGI 时代的选择）。

**模型 B：每连接一个线程**（Java/Python 服务的默认形态）：

```c
pthread_t tid;
pthread_create(&tid, NULL, handler, (void *)(intptr_t)cfd);   // 每连接一个线程
```

内存与切换开销比进程小一个量级，共享地址空间方便缓存模型/连接池；线程数要控（07 篇：CPU 密集 ≈ 核数；IO 密集按等待比放大）。适合：中等并发、逻辑有阻塞调用（数据库查询、调外部 API）的业务服务——**Agent 生产部署的 FastAPI 多 worker 就是这个模型的变体**。

**模型 C：事件驱动（epoll + Reactor）**：

```c
int ep = epoll_create1(0);
epoll_ctl(ep, EPOLL_CTL_ADD, lfd, ...);
while (1) {                         // 单线程（或极少数线程）处理全部连接
    int n = epoll_wait(ep, events, MAXEV, -1);
    for (...) { /* 就绪的 accept/读/写，handler 快进快出（09 篇三条纪律） */ }
}
```

一个线程扛数万连接，切换开销近零；但 handler 一阻塞全员冻结，CPU 密集任务要外挂工作池（09 篇的混合架构）。Redis/Nginx/Node 的选择，推理服务的 HTTP 前端（如 vLLM 的 uvicorn/asyncio）同款。

| | 每进程 | 每线程 | 事件驱动 |
| --- | --- | --- | --- |
| 万级连接 | 勉强（内存炸） | 勉强（切换贵） | 轻松 |
| 阻塞调用影响 | 只挂自己 | 只挂自己线程 | **挂所有连接** |
| 故障隔离 | 强 | 中（进程级） | 弱（一崩全崩） |
| 编程复杂度 | 低 | 低 | 高（状态机/回调） |
| 代表 | Apache prefork | 传统 Java/Python 服务 | Redis/Nginx/Node/uvicorn |

选型口诀（**经验做法**）：连接多而逻辑轻 → 事件驱动；逻辑重且要阻塞调库 → 线程池（连事件驱动外壳 + 内部线程池的混合体最常见）；要强隔离 → 进程。生产推理服务几乎都是混合体：asyncio 收发 + 线程池执行阻塞的引擎调用。

## 常见误区

1. **"send 成功 = 对方收到了"**：send 只是把数据放进本地发送缓冲；对端是否收到由 TCP 重传机制决定，应用层要"确认"必须自己设计 ACK/响应协议。可靠传输 ≠ 确认送达（RAS 问题）。
2. **"recv 返回的字节数等于发送方的单次 send"**：字节流无边界，recv 按缓冲区有多少给多少；粘包不是 bug 是 TCP 的本质（第 3 节）。
3. **"TIME_WAIT 多 = 有问题"**：短连接压测的客户端侧 TIME_WAIT 成千上万是常态；要消灭它先考虑连接复用（HTTP keep-alive、连接池）——大多数"TIME_WAIT 调优"的正确答案是改用长连接。
4. **"端口号随进程走"**：端口是 TCP 四元组（源IP/源端口/目的IP/目的端口）的成员，属于连接不属于进程；同一进程可以监听 0.0.0.0 或具体网卡、同一端口可以服务成千连接（每连接独立四元组）。
5. **"并发服务器线程越多越好"**：见 07 篇切换成本与 09 篇不对称性；连接数 ≠ 线程数，事件驱动让这等式作废。

## 本篇小结

- 分层分工：应用定协议、传输定可靠与端口、网络定路由、链路定介质；socket API 是内核在传输层开的口，fd 身份（08 篇）让它无缝进 epoll（09 篇）。
- 三次握手防旧 SYN 重放，四次挥手因全双工独立关闭；TIME_WAIT 是主动方的安全气囊（2MSL），客户端侧堆积是常态，服务端堆积查代码；SO_REUSEADDR 服务重启必备。
- 粘包 = 字节流无边界 + 应用误以为有包；长度前缀是二进制协议正解（recv_all 读满为一条），分隔符适合文本，UDP 天然有界。
- 并发模型三选一：进程（隔离强开销大）、线程（默认形态，控数量）、事件驱动（万连接轻逻辑，handler 不阻塞）；生产系统多为"事件外壳 + 线程池内核"混合体。

## 练习

1. 编译对跑 tcp_server/tcp_client；用 `ss -tanp | grep 9001` 观察握手建立后的 ESTAB 状态与双方端口号；连续跑 5 次 client，观察 TIME_WAIT 出现在哪一侧、60 秒后是否消失。
2. 给 tcp_client 加"连接后只 send 不 recv"再 close，观察服务端 recv 返回 0 的路径；再实现"服务端检测半关闭"（recv==0 时优雅收尾），并在注释里写下你验证 FIN 序列的方法（tcpdump -i lo port 9001 可选）。
3. 跑 sticky_demo.py 多次，记录"一次收到/分两次收到"的比例；把 time.sleep(0.001) 换成 0.2，观察现象如何变化，用发送缓冲与到达时机解释。
4. 把 framed_read.c 接到 tcp_server.c 上：客户端用 Python 按"4 字节大端长度 + 正文"发消息（struct.pack），服务端逐帧 echo；发送 3 条不同长度消息验证边界恢复。
5. 用三种模型各写一个版本处理"每请求 sleep(50ms) 后回显"的负载：单进程串行、线程池（pthread）、epoll 事件驱动 + 计时；对比 20 并发下的总吞吐，把"哪种模型赢、为什么"写成 5 行结论（提示：IO 等待主导的负载是事件驱动与线程池的主场）。
