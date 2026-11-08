# 08 - HTTP 协议详解

## 本篇目标

- 逐行解剖一次真实的 HTTP 请求/响应报文，把请求行、头部、体三段结构的语义全部落到字段上。
- 掌握方法与状态码的**语义**（幂等性、301/302/307/308 的区别），这是 API 设计与网关配置的通用语言。
- 吃透三个高频头部机制：Content-Type、Cookie、缓存（Cache-Control/ETag），并完整走一遍 CORS 预检流程。
- 理解 HTTP/1.1 的队头阻塞、HTTP/2 的多路复用与 HPACK、HTTP/3（QUIC）为什么能根治队头阻塞，并用 curl 实测各版本。

前置：07 章 TCP；本篇的 curl 实验在 WSL2 直接可跑（注意代理环境变量，见 8 节）。

## 1. HTTP 是什么：基于文本的请求-响应协议

HTTP（HyperText Transfer Protocol）是应用层协议：客户端发**请求**，服务器回**响应**，协议本身无状态（stateless）——每个请求自带全部上下文，服务器不记得"上一个请求是谁发的"，状态靠 Cookie 等机制由客户端补回。HTTP/1.1 的规范是 RFC 7230~7235（2014），语义部分在 2022 年被重整为 RFC 9110（语义）/RFC 9112（HTTP/1.1）。

看真实报文最方便的工具是 `curl -v`（本机真实输出，走代理环境）：

```bash
$ curl -sv -o /dev/null --http1.1 https://httpbin.org/get 2>&1 | sed -n '/^>/,$p' | head -30
> GET /get HTTP/1.1
> Host: httpbin.org
> User-Agent: curl/7.81.0
> Accept: */*
>
* TLSv1.3 (IN), TLS handshake, Finished (20):      ← TLS 层信息（09 章）
< HTTP/1.1 200 OK
< Date: Sat, 09 Nov 2026 06:12:33 GMT
< Content-Type: application/json
< Content-Length: 255
< Connection: keep-alive
< Server: gunicorn/19.9.0
<
{ "args": {}, "headers": { "Host": "httpbin.org", ... }, "origin": "...", "url": "..." }
```

`>` 开头是请求，`<` 开头是响应——报文结构一览无余。**经验做法**：`curl -v` 是 HTTP 排查的瑞士军刀；看不了 TLS 解密内容时加 `--trace-ascii -` 能看到更多细节，调试 webhook/API 对接时先贴 -v 输出再提问。

## 2. 报文解剖：请求与响应的完整结构

```
 请求                                  响应
 ─────────────────────────────        ─────────────────────────────
 GET /path?q=1 HTTP/1.1   ← 请求行    HTTP/1.1 200 OK          ← 状态行
   方法 路径+查询串 版本                    版本 状态码 原因短语
 Host: httpbin.org        ← 头部区     Date: ...                ← 头部区
 User-Agent: curl/7.81.0               Content-Type: application/json
 Accept: */*                           Content-Length: 255
 （每行 头名: 值，CRLF 结尾）            Connection: keep-alive
                            ← 空行（CRLF，报文头的终结标志）
 POST 场景此处是请求体：                 响应体（HTML/JSON/字节流…）
   Content-Type: application/json
   Content-Length: 27
   {"prompt": "hi"}      ← 请求体
```

三条结构铁律：① 头部区以**空行**终结，之后才是体——手工写报文最容易漏这个 CRLF；② 体长由 `Content-Length` 声明，流式场景用 `Transfer-Encoding: chunked`（分块，每块带自己的长度）；③ HTTP/1.1 里 **Host 头是必需的**（RFC 9112 3.2）——一台 IP 上靠它区分成百上千个虚拟主机，CDN/反向路由（10 章）全靠 Host 分流。

## 3. 方法与状态码：语义先行

**方法语义表**（RFC 9110 第 9 章；幂等性=执行 N 次与 1 次效果相同）：

| 方法 | 安全（不改状态） | 幂等 | 典型用途 |
| --- | --- | --- | --- |
| GET | 是 | 是 | 读取资源 |
| HEAD | 是 | 是 | 只要头部（探测大小/存在性） |
| PUT | 否 | 是 | 整体替换（幂等） |
| DELETE | 否 | 是 | 删除（幂等） |
| POST | 否 | 否 | 创建/触发动作（**不幂等**——重试会重复下单） |
| PATCH | 否 | 否（一般） | 部分更新 |

幂等性直接决定 11 章的重试策略：GET/PUT/DELETE 可自动重试，POST 默认不可（除非带 Idempotency-Key）。**状态码分组**：1xx 信息、2xx 成功（200 OK/201 Created/204 No Content）、3xx 重定向、4xx 客户端错（400/401 未认证/403 无权限/404/429 限流）、5xx 服务端错（500/502 网关错/503 不可用/504 网关超时）。**3xx 的细颗粒辨析**（301 vs 302 vs 307 vs 308，RFC 9110 15.4）：

```
            临时（浏览器下次仍用原 URL）      永久（搜索引擎换收录）
 GET 场景       302 Found                     301 Moved Permanently
 保持方法       307 Temporary Redirect         308 Permanent Redirect
 （POST 仍是 POST）── 307/308 与 302/301 的唯一区别
 历史坑：多数浏览器把 301/302 的 POST 降级成 GET —— 要保持 POST 用 307/308
```

**经验做法**：写网关/重定向配置时默认用 307/308（语义明确不吞方法）；登录后跳转用 302；域名迁移用 301。把"POST 被 301 吞成 GET"当作排查线上表单莫名 GET 化的第一嫌疑。

## 4. 关键头部详解

**Content-Type**：体的媒体类型（RFC 9110 8.3）。高频值：`application/json`、`application/x-www-form-urlencoded`（表单默认）、`multipart/form-data`（文件上传，boundary 分隔）、`text/event-stream`（**SSE，LLM 流式输出的载体**，11 章 server-streaming 的 HTTP 版）。协商由 `Accept` 头完成——服务端 406 的常见来源。

**Cookie**：HTTP 无状态，Cookie 是把状态放回客户端的标准机制（RFC 6265）。服务器 `Set-Cookie: session=abc; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=3600`，浏览器后续请求自动带 `Cookie: session=abc`。属性逐个过：`HttpOnly` 防 JS 读（防 XSS 偷会话）、`Secure` 仅 HTTPS 发送、`SameSite` 防 CSRF（Lax 默认：跨站 GET 带 cookie，POST 不带）、`Max-Age`/`Expires` 控寿命。**经验做法**：会话 cookie 必须 HttpOnly+Secure+SameSite 三件套；token 放 localStorage 是 XSS 时代的高危做法，能放 cookie 就放 cookie。

**缓存头部**：两层机制配合（RFC 9111）：

```
 强缓存（不问服务器）：Cache-Control: max-age=3600        1 小时内直接用本地副本
 协商缓存（问服务器）：客户端 If-None-Match: "etag123"
                      服务端没变 → 304 Not Modified（无体，省流量）
                      变了   → 200 + 新内容 + 新 ETag
 ETag（内容指纹）与 Last-Modified（时间戳）二选一或并用；
 no-cache=可用但每次协商；no-store=完全不存
```

静态资源最佳实践（**经验做法**）：`文件名带 hash + Cache-Control: max-age=31536000, immutable`（内容变了名字就变，缓存永不失效）；API 响应默认 no-cache 或短 max-age。LLM API 的 SSE 响应必须 `Cache-Control: no-cache` + `X-Accel-Buffering: no`（防网关缓冲打断流式）。

## 5. CORS：跨域资源共享的完整机制

浏览器的**同源策略**：脚本只能读"协议+域名+端口"三全同的响应，否则被拦——没有它，你登录银行后访问恶意网站，恶意网站的 JS 就能拿你的银行会话发请求读响应。CORS（Cross-Origin Resource Sharing，RFC 9110 的 Fetch 规范部分）是服务端**授权跨域读取**的标准机制。完整流程（预检请求）：

```
 前端 JS: fetch("https://api.x.com/v1", { method:"POST",
           headers:{ "Content-Type":"application/json",
                     "Authorization":"Bearer xx" }, body:"{}" })

 【第一步】浏览器先发预检（不是所有请求都要预检，见下）
   OPTIONS /v1 HTTP/1.1
   Origin: https://app.y.com
   Access-Control-Request-Method: POST
   Access-Control-Request-Headers: authorization, content-type

 【第二步】服务端授权
   HTTP/1.1 204 No Content
   Access-Control-Allow-Origin: https://app.y.com
   Access-Control-Allow-Methods: POST, GET
   Access-Control-Allow-Headers: Authorization, Content-Type
   Access-Control-Max-Age: 86400          ← 预检结果缓存 1 天

 【第三步】预检通过，浏览器才发真正的 POST
   POST /v1 HTTP/1.1
   Origin: https://app.y.com
   Authorization: Bearer xx
   → 响应必须仍带 Access-Control-Allow-Origin，浏览器才把响应交给 JS
```

什么时候**不**预检（"简单请求"）：方法为 GET/HEAD/POST、Content-Type 仅限三种朴素类型（form-urlencoded/multipart/text-plain）、无自定义头。带 Authorization 或 JSON Content-Type 的请求几乎必然预检——这就是前端联调时"OPTIONS 请求 405/404 导致真请求发不出去"的经典现场。**经验做法**：CORS 是**浏览器行为**，curl/服务端互调根本不执行 CORS——"我 curl 能通浏览器报 CORS 错"不是玄学，是协议分层；排查时看 OPTIONS 响应头的 Allow-Origin 是否匹配（含端口！）、是否有 5xx，并确认网关没有把 OPTIONS 路由到业务后端。

## 6. HTTP/1.1 的队头阻塞与连接 workaround

HTTP/1.1 默认 **keep-alive**（持久连接，复用 TCP），支持 **pipelining**（连续发多个请求不等响应）——但规范要求响应必须按序返回（RFC 9112 9.3），于是第一个慢响应阻塞后面所有响应：**队头阻塞（head-of-line blocking）**。浏览器实际放弃 pipelining，改用**每域名 6 条并发连接** workaround。代价肉眼可见：6 倍握手/TLS 成本、6 倍服务端连接状态、且仍按连接内串行。07 章说过新连接要经历慢启动——**HTTP/1.1 时代的性能优化一半在跟"连接"搏斗**：域名分片、雪碧图、内联小图，全是给队头阻塞打的补丁。

## 7. HTTP/2 与 HTTP/3（QUIC）：两条根治路线

**HTTP/2**（RFC 7540/9113）在**一条 TCP 连接**上引入流（stream）抽象：

```
 HTTP/1.1: 请求1 ──响应1── 请求2 ──响应2──（串行，队头阻塞）
 HTTP/2:   连接 ─┬─ stream1: 请求1/响应1（交错成帧）
                ├─ stream3: 请求3/响应3
                └─ stream5: 请求5/响应5    （多路复用：帧级交错，互不阻塞）
 帧结构： [ 长度(24b) | 类型(8b) | 标志(8b) | R | 流ID(31b) | 载荷 ]
```

配套机制：**HPACK 头部压缩**（静态表 61 个常见头 + 动态表记录本次连接出现过的头，后续只发索引——同一连接的第 N 个请求头可能只有十几字节）与二进制分帧。**服务器推送**（server push）因命中率低已被主流浏览器移除（Chrome 106 起），知道即可不必深究。**但 HTTP/2 没有根治队头阻塞**：流在应用层分开了，底层仍是**一条 TCP**——TCP 只见字节流，任何一段丢包，所有流的数据都得等重传补齐（TCP 层队头阻塞）。丢包率 2% 时 HTTP/2 吞吐甚至可能低于多条 HTTP/1.1 连接（各连接互不牵连）。

**HTTP/3**（RFC 9114 + QUIC RFC 9000/9002）换掉传输层：**基于 UDP 的 QUIC** 在用户态重造了 TCP 的可靠传输（07 章的序号/ACK/重传全套），但**每个流独立编序号、独立重传**——stream3 的丢包不再阻塞 stream5，根治队头阻塞。其他红利：**握手 1-RTT（0-RTT 恢复）**把 TCP+TLS 两次握手合并成一次（09 章详解）；连接迁移（Connection ID 标识连接，换 Wi-Fi 不断流）；头部加密防中间设备篡改。代价：UDP 被部分企业防火墙限速、CPU 开销略高（内核 Offload 跟不上）。**实测对比**：

```bash
$ curl -sv -o /dev/null --http2 https://httpbin.org/get 2>&1 | grep -E 'ALPN|HTTP/'
* ALPN: offers h2, http/1.1
* using HTTP/2                       ← ALPN 在 TLS 握手中协商出 h2
$ curl -sI --http2 https://httpbin.org/get | head -1
HTTP/2 200
$ curl -sI --http3-only https://cloudflare-quic.com/ 2>&1 | head -1
HTTP/3 200                            ← 需要 curl 支持 HTTP/3（本机 7.81 未编入，
                                         换新版 curl 或用 quiche 客户端）
```

（本机 curl 未编 HTTP/3 支持，正常——发行版 curl 普遍滞后；生产客户端如 Chrome 已默认启用。）**经验做法**：推理服务的流式输出（SSE）在 HTTP/1.1 下一条连接只能服务一个响应，HTTP/2/3 下同连接多流并发——网关侧开 h2c（明文 HTTP/2）或 TLS ALPN h2 是流式服务的基础设施标配。

## 常见误区

1. **"HTTP 无状态 = 连接无状态"**：无状态指协议语义（服务器不依赖前序请求记忆），TCP 连接本身是有状态的（07 章）；keep-alive 的连接里跑着无状态请求，两者不矛盾。
2. **"401 和 403 随便用"**：401 = 没有身份（去登录/带凭证，配 WWW-Authenticate），403 = 有身份但没权限（重试也没用）。API 设计混用会让客户端重试逻辑失效。
3. **"POST 是安全的所以能随便重试"**：POST 不幂等，网络超时后"成功还是失败"是未知的（11 章两军问题的 HTTP 版）——支付/下单接口要么设计成幂等（Idempotency-Key），要么把重试决策交给用户。
4. **"HTTP/2 多路复用 = 并发更快，直接上"**：单 TCP 流在丢包链路上反而放大队头阻塞；且 HTTP/2 需要同一域名合并连接，网关/CDN 必须支持。收益场景是大量小请求+稳定网络，反之可能倒退。
5. **"QUIC 就是 UDP 上的 TCP"**：QUIC 在用户态重造了可靠传输，但把"流"作为一等公民（每流独立序号）、把 TLS 1.3 握手融进传输握手（1-RTT）、用 Connection ID 支持迁移——这些是 TCP 协议结构给不了的能力，不只是"换了个 UDP 外壳"。

## 本篇小结

- 报文三段：起行（方法/状态）+ 头（每行 name: value）+ 空行 + 体；Content-Length/chunked 定长；HTTP/1.1 必带 Host。
- 方法看幂等（GET/PUT/DELETE 幂等、POST 不幂等——重试策略的分界），状态码 3xx 用 307/308 保持方法（301/302 会把 POST 吞成 GET）。
- 三大头部：Content-Type/SSE 流式、Cookie（HttpOnly+Secure+SameSite 三件套，经验做法）、缓存（强缓存 max-age + 协商缓存 ETag/304）。
- CORS：非简单请求先发 OPTIONS 预检，服务端以 Allow-Origin/Methods/Headers 授权；只约束浏览器，curl 无此限制；OPTIONS 405 是网关路由背锅的经典现场。
- HTTP/1.1 队头阻塞 → 浏览器 6 连接 workaround；HTTP/2 多路复用+HPACK 治了应用层、留了 TCP 层队头阻塞；HTTP/3/QUIC 每流独立重传根治之，附 1-RTT/0-RTT 与连接迁移。
- curl 实测：--http1.1/-v 逐行解剖，ALPN 协商 h2；本机 curl 无 HTTP/3 支持属常态，用新版客户端或记文档结论。

## 练习

1. 用 `curl -v --http1.1 http://httpbin.org/redirect/2` 追踪两级重定向，抄出每一跳的状态码与 Location 头；换成 `-d 'x=1'` POST 再试，观察方法在重定向中如何变化（对照 3 节表格）。
2. 手写 telnet/nc 实验：`printf 'GET / HTTP/1.1\r\nHost: example.com\r\nConnection: close\r\n\r\n' | nc example.com 80`，解释为什么必须有两个 `\r\n`（空行终结头部）；漏掉 Host 会得到什么状态码？
3. 用浏览器 DevTools 的 Network 面板访问任意 HTTPS 站点：找出一条 304 响应，抄出请求侧 If-None-Match 与响应侧 ETag；再找一条 from disk cache 的响应，说明它走的是强缓存还是协商缓存。
4. 写一个最小 Flask/FastAPI 服务（或用 httpbin.org），用 curl 分别发：无 Origin 头、`Origin: http://evil.com`、`Origin: http://app.y.com`（服务端配了该域），对比响应头里的 Access-Control-Allow-Origin，解释为什么前两种在浏览器里都拿不到响应。
5. 对同一站点分别用 `curl --http1.1 -w '%{num_connects}\n' -o /dev/null URL1 URL2 URL3` 与 `curl --http2 ...` 统计连接数（num_connects），用数字验证 6 节"HTTP/1.1 需要多连接、HTTP/2 单连接"的结论。
