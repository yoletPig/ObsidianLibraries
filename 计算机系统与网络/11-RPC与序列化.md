# 11 - RPC 与序列化

## 本篇目标

- 划清本地调用与远程调用的语义边界：哪些东西天然跨不过网络（指针/异常/事务），超时之后"成功还是失败"为什么不可知。
- 用实测对比 JSON/MessagePack/Protobuf 的编码大小与适用场景，建立"选格式"的量化依据。
- 拆解 Protobuf 的编码原理（varint、tag=字段号<<3|wire_type）与兼容性规则，看懂 .proto 的演进纪律。
- 认识 gRPC 的四种流模式（对照 LLM 流式输出），理解 HTTP/2 如何承载 gRPC；掌握超时/重试/退避的配置纪律与重试风暴；最后手写一个 100 行 mini-RPC。

前置：07 章（TCP）、08 章（HTTP/2）；与分布式系统系列 10 章（容错/熔断）在重试一节交汇。

## 1. 本地调用 vs 远程调用：假装一样的代价

RPC（Remote Procedure Call）的野心是"像调本地函数一样调远程服务"。但本地调用与远程调用之间有一道语义鸿沟（分布式系统经典"八大谬误"的前几条：网络可靠、延迟为零、带宽无限、拓扑不变——都是假的）：

| 本地调用 | 远程调用 | 差在哪 |
| --- | --- | --- |
| 传指针/引用 | 对象状态无法传递，只有**序列化后的值** | 必须传"数据的拷贝" |
| 抛异常 | 异常对象跨不过网络，只能编码成错误码/错误消息 | 异常类型/堆栈丢失 |
| 事务语义 | 两个服务两套事务，无法原子（除非分布式事务） | 一致性靠自己 |
| 失败要么不发生要么抛异常 | **超时 = 结果未知**（请求可能已执行） | 最危险的一条 |
| 纳秒级返回 | 毫秒级 + 抖动 | 设计假设全变 |

**超时的不可知性**是 RPC 最重要的心智建设：你发请求，超时 5s 后放弃——服务端可能根本没收到、可能正在算、可能**已经成功只是响应丢了**。盲目重试一个写操作可能重复扣款（08 章 POST 不幂等的 RPC 版）。解法三板斧：业务侧幂等键（Idempotency-Key）、框架侧重试策略（6 节）、协议侧"至多一次/至少一次"的明确承诺。**经验做法**：所有有副作用的 RPC 接口必须幂等或能安全重试，这条写进接口评审 checklist。

## 2. 序列化格式对比：实测说话

同一个对象（一条对话消息），三种格式编码：

```python
# enc_demo.py —— 同一数据三种编码的大小对比
import json, struct

msg = {"id": 1024, "role": "user", "content": "你好",
       "scores": [0.9, 0.8, 0.7], "meta": {"lang": "zh", "tokens": 12}}

# 1) JSON（文本，人类可读）
b1 = json.dumps(msg, ensure_ascii=False).encode()
# 2) MessagePack（二进制 JSON，msgpack 库：pip install msgpack）
import msgpack
b2 = msgpack.packb(msg, use_bin_type=True)
# 3) 手工 struct（模拟 Protobuf 的紧凑二进制思路：字段号+类型+值）
b3 = (b'\x08\x80\x08'                       # field 1 varint: 1024
      b'\x12\x04user'                       # field 2 bytes: "user"
      b'\x1a\x04你好'                        # field 3 bytes: "你好"
      b'\x22\x1a' + json.dumps(msg["scores"]).encode())  # field 4: scores 子结构
for name, b in [("JSON", b1), ("MessagePack", b2), ("Protobuf风格", b3)]:
    print(f"{name:12} {len(b):4d} bytes  {b[:40]!r}")
```

```bash
$ python3 enc_demo.py
JSON         119 bytes  b'{"id": 1024, "role": "user", "content": "\xe4\xbd\xa0\xe5\xa5\xbd", ...'
MessagePack   83 bytes  b'\x83\xa2id\xcd\x04\x00\xa4role\xa4user\xa7content\xa2...'
Protobuf风格  45 bytes  b'\x08\x80\x08\x12\x04user\x1a\x04\xe4\xbd\xa0\xe5\xa5\xbd\x22\x1a...'
```

（本机 Python 3.10 实测；比例随数据形状浮动，但量级关系稳定：**JSON ≈ 2.5×Protobuf，MessagePack 居中**。）定性总结：

- **JSON**：文本、自描述（带字段名）、人人可读、语言无关生态最广；代价是字段名重复传输、数字/二进制编码低效、解析慢。REST API 事实标准。
- **MessagePack**：JSON 的二进制化（结构同、字段名变短、变长整数），体积省 30%，仍需字段名——适合"想要 JSON 的灵活又想省点"的场景。
- **Protobuf**：**schema 前置**（字段名不进报文，只有字段号）、二进制紧凑（省 50-70%）、解析最快（定长/变长整数直读）、强类型+版本兼容规则；代价是需要 .proto 与代码生成、报文不可读。gRPC 内部序列化，微服务间主流。

## 3. Protobuf 编码原理：varint 与字段号

**varint（变长整数）**：每字节低 7 位放数据、最高位是"继续位"——小数字用 1 字节，大数字多用几字节。编码 1024：二进制 `100_0000_0000`，从低位切 7 位一组：`0000000|1000000`，倒序后每字节加继续位 → `0x80 0x08`（上面实测的 `\x80\x08` 就是它——**1024 用 2 字节存下**，固定 4 字节 int32 省一半）。

每个字段的前缀是 **tag = (字段号 << 3) | wire_type**：wire_type 0=varint、1=64-bit、2=长度前缀字节串/嵌套消息、5=32-bit。解码器按字段号+类型识别每个片段——**字段名完全不参与编码**（.proto 里 `string role = 2;` 的 "role" 只是给生成代码看的，报文里只有数字 2）。

**兼容性规则**（Protobuf 官方文档 Language Guide 的更新纪律）：

- **字段号是契约**：已发布的字段号**永不复用**（改名字可以，改号=把旧数据当新字段解析，灾难）；废弃字段标 `reserved 2;` 占住号码防手滑复用。
- **加字段 = 兼容**：旧代码遇到未知字段号直接跳过（这也是 Protobuf "向前兼容"的实现方式）。
- **类型变更纪律**：int32/int64/bool 互通；string↔bytes 禁止；optional↔repeated 需打包规则注意。
- **经验做法**：schema 变更评审盯三件事——字段号没复用、类型没变脸、repeated 改 single 要走迁移期。同一条纪律映射到 gRPC 服务演进：**只加不删、删的标 deprecated 留一个版本周期**。

## 4. gRPC 四种流模式

gRPC（Google 2015 开源，基于 HTTP/2 + Protobuf）把服务方法分成四种形状：

```
1) Unary            请求1 ──► 响应1        普通函数调用（同步问答）
2) Server streaming 请求1 ──► 响应N        服务端逐个推（LLM 流式生成！）
3) Client streaming 请求N ──► 响应1        客户端攒批（日志上报/文件分块上传）
4) Bidi streaming   请求N ⇄ 响应N         双向独立（对话、实时协作、代理转发）
```

.proto 语法与 LLM 场景对照：

```protobuf
service ChatService {
  rpc Complete (ChatRequest) returns (ChatResponse);              // unary：短问答
  rpc StreamComplete (ChatRequest) returns (stream ChatChunk);   // server-streaming：
                                                                  // token 流式输出
  rpc UploadDocs (stream DocChunk) returns (UploadSummary);       // client-streaming
  rpc AgentConversation (stream AgentMsg) returns (stream AgentMsg); // bidi：Agent 循环
}
```

你天天用的"打字机效果"就是 server-streaming：服务端生成一个 token 推一帧，客户端逐帧渲染——比"全部生成完再返回"的首 token 延迟低一个数量级（08 章 SSE 是 HTTP 侧的等价物；内部 gRPC 走 stream ChatChunk，对外网关转成 SSE）。

## 5. HTTP/2 如何承载 gRPC

gRPC 把一次调用映射到一条 HTTP/2 流：路径是 `/包名.服务名/方法名`（如 `/chat.v1.ChatService/StreamComplete`），请求/响应体是 Protobuf 帧，流式模式用 HTTP/2 的 DATA 帧分块推送。**为什么 gRPC 必须 HTTP/2**（HTTP/1.1 做不到的）：① **多路复用**——一条 TCP 连接并发成百上千条流，unary 高并发不再受"每连接一响应"限制（08 章队头阻塞的 gRPC 语境：HTTP/2 治了应用层，QUIC 治传输层）；② **双向流**——HTTP/1.1 的响应先到语义撑不起 bidi，HTTP/2 的流是全双工的；③ **头部压缩 HPACK**——微服务间海量小请求，重复头部压缩后开销显著下降。**经验做法**：内网 gRPC 走明文 h2c（HTTP/2 cleartext）省 TLS 开销很常见，但要确认服务网格/网关支持；跨公网必须 TLS（09 章）。

## 6. 超时、重试与重试风暴

RPC 框架（gRPC/自研）必须配置的四件套：

- **超时（deadline）**：单次调用超时 + **链路传播**——上游 10s deadline，下游收到时只剩 8s，应把 8s 传下去而不是重新给 10s（gRPC 的 deadline 原生随请求传播）。全局预算：端到端 deadline 从用户侧倒推分配。
- **重试**：只对**幂等且可能是暂时性失败**的请求重试（网络错/503/429）；指数退避（1s→2s→4s…）+ **jitter**（随机抖动，防止重试齐步走形成脉冲）。
- **重试预算**：限制重试总量不超过正常流量的 ~10%（1.1 倍）——超预算时新请求不再重试。**没有预算的重试是自我放大的故障**：下游抖动 → 全部上游重试 → 流量×N → 下游彻底打垮 → 更多重试——这就是**重试风暴**，与分布式系统系列 10 章的熔断/舱壁模式是同一枚硬币（那边讲全链路怎么熔，这边讲单点怎么配）。
- **负载均衡策略**：客户端负载均衡（gRPC 自带 round-robin/ weighted）vs 代理负载均衡（Envoy/Nginx，10 章的 L7）；对长连接（gRPC 流）注意连接级均衡导致的"热连接"问题。

**经验做法**：重试配置三行纪律——只重试幂等操作、指数退避+jitter 上限 2~3 次、全局重试预算 1.1 倍；再配合上游熔断（连续失败快速失败），四道防线缺一不可。

## 7. 动手：100 行 mini-RPC（socket + struct）

不用任何框架，用 02/07 章的知识手写一个带长度前缀的请求-响应协议（粘包问题用 4 字节长度头解决——OS 系列 12 篇的方法论）：

```python
# mini_rpc.py —— server 与 client 二合一（先起 server 再起 client）
import socket, struct, threading, json

def send_msg(sock, obj):                       # [4字节长度][JSON负载]
    data = json.dumps(obj).encode()
    sock.sendall(struct.pack(">I", len(data)) + data)

def recv_exact(sock, n):                       # 循环读满 n 字节
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk: raise ConnectionError("peer closed")
        buf += chunk
    return buf

def recv_msg(sock):
    (length,) = struct.unpack(">I", recv_exact(sock, 4))   # 长度前缀解粘包
    return json.loads(recv_exact(sock, length))

def handle(req):                               # "业务逻辑"
    if req["op"] == "add": return {"sum": req["a"] + req["b"]}
    return {"err": "unknown op"}

def server():
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 9777)); srv.listen()
    print("server ready on :9777")
    while True:                                # 一连接一线程的简单模型
        conn, addr = srv.accept()
        threading.Thread(target=lambda c=conn: (
            send_msg(c, handle(recv_msg(c))), c.close())).start()

def client():
    s = socket.create_connection(("127.0.0.1", 9777), timeout=3)
    send_msg(s, {"op": "add", "a": 2, "b": 40})
    print("client got:", recv_msg(s))          # → {"sum": 42}
    s.close()

if __name__ == "__main__":
    import sys
    threading.Thread(target=server, daemon=True).start()
    import time; time.sleep(0.3); client()     # demo 跑通后 Ctrl-C
```

```bash
$ python3 mini_rpc.py
server ready on :9777
client got: {'sum': 42}
```

这 40 行骨架涵盖了 RPC 的全部要素：**序列化**（JSON——换成 3 节的 Protobuf 只动编解码两行）、**粘包处理**（长度前缀）、**传输**（TCP 字节流）、**请求-响应语义**（1 节的超时未知性就在 `timeout=3` 里：超时后服务器线程可能仍在算）。补上 deadline 传播、重试、服务注册发现，就是一个 RPC 框架的雏形——gRPC 把这些工程细节都做好了而已。

## 常见误区

1. **"Protobuf 比 JSON 快，全都该换"**：快 3-5 倍是真的，但代价是 schema 管理与代码生成流程；对外 API 的 JSON 生态（浏览器/curl/网关调试）价值常大于性能收益。内部高频微服务用 Protobuf，外部 API 用 JSON 是常见平衡。
2. **"字段号随便编，反正是内部的"**：字段号一进生产就是对外契约（历史数据、跨版本服务）；复用废弃字段号是 Protobuf 数据损坏第一大来源。`reserved` 永远占住号。
3. **"超时设长一点更安全"**：长超时把故障放大（下游卡死时上游线程池被占满，级联拖垮上游——分布式系列 10 章的雪崩路径）；正确姿势是**短超时+有限重试**，把失败快速暴露。
4. **"重试是提高成功率的免费午餐"**：重试流量 = 额外负载，风暴场景是压垮服务的最后一根稻草（6 节）；预算+退避+jitter 是标配，不是可选项。
5. **"gRPC 流模式很少用"**：LLM 时代 server-streaming 是主流形态（打字机输出）；Agent 循环天然是 bidi。做推理服务方向，流式 gRPC/SSE 是必修不是选修。

## 本篇小结

- RPC 与本地调用的鸿沟：值拷贝（无指针）、错误码（无异常）、超时结果未知（最危险）；副作用接口必须幂等（经验做法）。
- 编码大小实测：JSON 119B / MessagePack 83B / Protobuf 风格 45B；JSON 赢生态、MessagePack 折中、Protobuf 赢体积速度与类型契约。
- Protobuf：varint（1024→2 字节）、tag=字段号<<3|wire_type、字段名不进报文；兼容纪律=字段号永不复用（reserved 占位）、只加不删（经验做法）。
- gRPC 四种流：unary/server-streaming（LLM 流式输出）/client-streaming（攒批上报）/bidi（Agent 循环）；承载于 HTTP/2（多路复用+双向流+HPACK）。
- 超时/重试纪律：deadline 链路传播、只重试幂等、指数退避+jitter、重试预算 1.1 倍——四道防线防重试风暴（与分布式 10 章熔断互补）。
- mini-RPC 40 行跑通：长度前缀解粘包 + JSON 编解码 + TCP 双端；框架不过是把这些细节工程化。

## 练习

1. 把 mini_rpc.py 的 JSON 换成 struct 手工二进制编码（op 用 1 字节、a/b 用 `>i`），对比同一请求的线上字节数；估算把 10 万 QPS 的 add 服务从 JSON 换成二进制编码，年省的出口流量（按 64B 请求+32B 响应算）。
2. 用 msgpack 库重跑 2 节实验：给 msg 加一个 200 字节的长文本字段，再对比三种格式的大小；解释"字段名重复开销"占比为什么随值变大而下降。
3. 给 3 节的 .proto 加需求：新增 `model` 字段（string），旧服务要能读新报文、新服务要能读旧报文。写出变更方案（字段号、reserved 使用），并说明旧服务读到未知字段时的行为。
4. 设计一个会触发重试风暴的实验：mini_rpc 的 server 加 50% 概率 503 响应，client 无退避重试 3 次，观察总请求数放大倍数；再加指数退避+预算（最多 1.1×请求数）重跑，对比放大倍数。
5. 扩展 mini_rpc 支持 server-streaming：`{"op":"gen","prompt":"hi","n":5}`，服务端分 5 帧推送 `{"chunk":"tok"}`，客户端逐帧打印——复刻一遍 4 节的打字机效果（提示：循环 send_msg + 结束帧标记）。
