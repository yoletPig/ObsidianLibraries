# 09 - TLS 与安全通信

## 本篇目标

- 讲清对称与非对称加密的分工逻辑，理解"混合加密"为什么是 HTTPS 的必然设计。
- 画出 TLS 1.2 与 TLS 1.3 的握手时序并说清差异（2-RTT → 1-RTT → 0-RTT 及其重放风险）。
- 用 openssl s_client 解析一张真实证书：证书链、校验逻辑（域名/有效期/信任根/吊销）全部落到命令输出上。
- 完成一个自签证书 + 本地 TLS 服务 + 抓握手的完整实验闭环；纠正"HTTPS=绝对安全"的常见误解。

前置：08 章（HTTP）；07 章（握手的概念）。

## 1. 明文的世界：TLS 要解决什么

HTTP 明文传输，链路上任何设备（运营商、Wi-Fi 提供者、代理）都能读、能改、能冒充。TLS（Transport Layer Security，TLS 1.2 = RFC 5246，TLS 1.3 = RFC 8446）要在不可信信道上同时解决三件事：

- **机密性（confidentiality）**：窃听者看不懂内容；
- **完整性（integrity）**：篡改会被立刻发现（MAC/AEAD 认证）；
- **真实性（authenticity）**：你连的确实是 example.com，而不是中间人的服务器。

第三点最容易被轻视——加密本身防不了冒充（攻击者可以和你握手他自己的密钥再转发给你），所以 TLS 的设计核心其实是**"密钥协商 + 身份认证"绑在一起**：证书证明"这把密钥属于这个域名"。

## 2. 对称 vs 非对称：为什么要"混合"

| | 对称加密（AES-128-GCM、ChaCha20） | 非对称加密（RSA、ECDHE） |
| --- | --- | --- |
| 密钥 | 一把，双方共用 | 公钥加密/私钥解密（或反之签名） |
| 速度 | 快（硬件指令 AES-NI，GB/s 级） | 慢 3~4 个数量级（RSA-2048 每秒千次级） |
| 密钥分发 | **死结**：怎么把密钥安全地告诉对方？ | 公钥可公开，天然解决分发 |
| TLS 用途 | 会话数据加密（bulk data） | 身份认证（证书签名验证）+ 密钥交换（ECDHE） |

两边各有死穴：对称加密快但**密钥分发无解**；非对称解决分发但太慢。唯一出路是**混合加密**：用非对称手段（ECDHE 密钥交换 + 证书认证）**协商出一个对称会话密钥**，之后全部流量用对称加密——非对称只用一两次，对称承担海量数据。这就是 HTTPS 的整体结构。

**前向保密（PFS, Perfect Forward Secrecy）**：密钥交换用 **ECDHE**（Ephemeral，临时椭圆曲线密钥）——每次连接生成全新临时密钥对，会话密钥由双方临时私钥+对方公钥推得，**服务器的长期私钥（RSA 证书私钥）不参与会话密钥推导**。效果：攻击者今天录下全部密文，十年后拿到服务器私钥（长期保存的，泄露概率高），**仍然解不开**过去的会话——过去每条会话的密钥随连接销毁。RSA 密钥交换（TLS 1.2 允许）没有这个性质（拿私钥可解全部历史密文），这是 TLS 1.3 直接删除 RSA 密钥交换、只留 (EC)DHE 的原因（RFC 8446 2 节的设计说明）。

## 3. TLS 1.2 握手全流程（2-RTT）

```
 客户端                                     服务端
  │ ──① ClientHello──────────────────────► │   RTT 1
  │    支持的密码套件列表、随机数 Rc、SNI      │
  │ ◄─② ServerHello──────────────────────  │
  │    选定套件、随机数 Rs                    │
  │ ◄─③ Certificate（服务器证书链）───────── │
  │ ◄─④ ServerKeyExchange（ECDHE 参数+签名） │
  │ ◄─⑤ ServerHelloDone ──────────────────  │
  │ ──⑥ ClientKeyExchange（ECDHE 公钥）───► │   RTT 2
  │ ──⑦ ChangeCipherSpec, Finished ──────► │  （Finished 已加密，验证握手完整性）
  │ ◄─⑧ ChangeCipherSpec, Finished ─────── │
  ▼ 应用数据（对称加密）                      ▼
```

会话密钥推导：双方各出临时 ECDHE 私钥，用对方公钥算出**预主密钥**（这条信道上只有双方能算出它），再混入两个随机数派生出对称密钥。证书（③）的作用是**给 ECDHE 参数签名**——中间人可以发自己的 ECDHE 参数，但签不出 example.com 的证书私钥，客户端验证签名失败即终止。

## 4. TLS 1.3：1-RTT 与 0-RTT

TLS 1.3（RFC 8446）把握手砍掉一轮：**ClientHello 直接带上 key_share**（客户端的 ECDHE 公钥），服务端一个来回就能算出会话密钥并开始发送加密数据——1-RTT 完成握手+开始传数据：

```
 TLS 1.2：CH →(RTT1) SH+Cert+SKE → CKX+CCS+Fin →(RTT2) SH+Fin → 数据
 TLS 1.3：CH(+key_share) →(RTT1) SH(+key_share)+{EncryptedExtensions,
          Cert, CertVerify, Finished} → {Finished} → 数据     ← 花括号 = 已加密
```

变化清单：① 握手第三段起**全部加密**（EncryptedExtensions/Certificate 都在加密信道里——证书也藏起来了，中间设备看不到你访问哪个域名，也正因为如此 SNI 加密/ECH 才有必要）；② 删除 RSA 密钥交换与全部弱算法（强制 PFS，见 2 节）；③ **0-RTT**：复用旧会话（PSK）时，客户端可以在第一个飞行包里就带上加密的应用数据——省一个 RTT，但 **0-RTT 数据没有防重放保护**（攻击者录下 0-RTT 包可原样重发，服务端无法区分），所以规范要求 0-RTT 只用于幂等请求（GET），POST/支付类禁止进 0-RTT。**实测**（08 章的 curl 输出里已经现形）：

```
* TLSv1.3 (OUT), TLS handshake, Client hello (1):
* TLSv1.3 (IN), TLS handshake, Server hello (2):
* TLSv1.3 (IN), TLS handshake, Encrypted Extensions (8):   ← TLS 1.3 特有：已加密
* TLSv1.3 (IN), TLS handshake, Certificate (11):
* TLSv1.3 (IN), TLS handshake, CERT verify (15):           ← 证书对握手签名
* TLSv1.3 (IN), TLS handshake, Finished (20):
* TLSv1.3 (OUT), TLS handshake, Finished (20):
```

ClientHello 后紧跟 Encrypted Extensions/Certificate——对比 TLS 1.2 的明文 Certificate，一眼识别版本。

## 5. 证书链与信任根：openssl 实测解析

证书（X.509）是一个"域名 ↔ 公钥"的绑定声明，由 CA 私钥签名。浏览器/OS 内置了一批**根 CA 证书**（信任锚，trust store）；站点证书通常由**中间 CA** 签发，形成链：`根 CA ──签──► 中间 CA ──签──► 站点证书`。**链式验证**：用中间 CA 的公钥验证站点证书签名，再用根 CA 公钥验证中间证书——每一环的签名验证通过且各环都在有效期内，链才成立。

```bash
$ openssl s_client -connect httpbin.org:443 -servername httpbin.org < /dev/null 2>/dev/null \
  | openssl x509 -noout -subject -issuer -dates -ext subjectAltName
subject=CN=*.httpbin.org
issuer=C=US, O=Let's Encrypt, CN=R12
notBefore=Oct 27 09:37:38 2026 GMT
notAfter =Jan 25 09:37:37 2027 GMT
X509v3 Subject Alternative Name: DNS:*.httpbin.org
```

再看重整条链与校验结果（真实输出）：

```bash
$ openssl s_client -connect httpbin.org:443 -servername httpbin.org < /dev/null 2>&1 \
  | grep -E 'depth|verify return|Verify'
depth=2 C=US, O=Internet Security Research Group, CN=ISRG Root X1
verify return:1
depth=1 C=US, O=Let's Encrypt, CN=R12                     ← 中间 CA
verify return:1
depth=0 CN=*.httpbin.org                                  ← 站点证书
Verify return code: 0 (ok)                                 ← 全链验证通过
```

`depth=2→1→0` 自根向叶逐环验证，`Verify return code: 0 (ok)` 是信任链闭合的判定。注意 `R12`（中间 CA）不在系统信任库里，它作为**链的一部分**由服务器下发——这就是为什么"服务器漏配中间证书"是经典事故：桌面浏览器可能靠 AIA 补链救回来，大量移动端/curl 环境直接报 `unable to get local issuer certificate`。

## 6. 证书校验逻辑：客户端到底检查什么

一个 TLS 客户端验证证书的完整清单（RFC 8446 4.4.2 + 各浏览器策略）：

1. **域名匹配**：访问的域名必须命中 SAN（subjectAltName）条目；通配符 `*.httpbin.org` 只匹配一层子域。
2. **有效期**：当前时间 ∈ [notBefore, notAfter]——**系统时间错乱是最隐蔽的证书错误**（WSL2 休眠恢复后时钟漂移，curl 报 `certificate is not yet valid` 的经典现场）。
3. **信任链**：从叶到某个信任锚逐环签名验证（5 节）。
4. **密钥用法/基本约束**：中间 CA 的 BasicConstraints 必须是 CA=true，防"叶子证书冒充 CA"。
5. **吊销（概览）**：CRL（证书吊销列表，定期下载）与 OCSP（在线查询）——实践中覆盖率一般（软失败普遍），新方案 OCSP Stapling（服务器把吊销状态"钉"进握手）与短有效期证书（Let's Encrypt 90 天）是现在的现实防线。

**经验做法**：生产环境证书监控盯三样——到期日（提前 30 天告警，自动化续期如 certbot/cert-manager）、中间链完整性（用 `openssl s_client` 从一台干净机器验）、SAN 域名覆盖（新加域名别忘续签）。

## 7. 常见误解：HTTPS ≠ 绝对安全

- **证书配错仍是常态**：自签证书没有可信链（curl 报 `self-signed certificate`）、域名不匹配、中间链缺失——HTTPS 图标不代表"校验通过"，`-k/--insecure` 跳过校验等于关闭真实性检查（**经验做法**：测试环境宁可导入自签 CA 到信任库，也不要养成 -k 习惯）。
- **TLS 版本老旧**：TLS 1.0/1.1 已废弃（RFC 8996），PCI-DSS 明令禁用；老版本存在 POODLE/BEAST 等攻击面。服务器配置基线：TLS 1.2 最低，TLS 1.3 优先。
- **HTTPS 不隐藏元数据**：SNI（明文）、DNS 查询（明文，除非 DoH/DoT）、IP 与流量大小/时序（可做流量指纹分析）都在 TLS 之外——中间人虽然读不了内容，但仍知道"你在连哪个域名、传了多大流量"。ECH（加密 SNI）与 DoH 是补这些洞的新标准，未完全普及。
- ** HTTPS 保护的是传输，不是终端**：证书配在你的域名上但私钥泄露、服务器本身被入侵、浏览器恶意扩展——TLS 管不到。

## 8. 实验室：自签证书 + 本地 TLS 服务 + 抓握手

```bash
# 1) 生成自签证书（一条命令：生成 RSA 私钥 + 自签证书，SAN 写 localhost）
$ openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -keyout server.key -out server.crt \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
Generating a RSA private key
........+++++
writing RSA key                                        # server.key / server.crt 就绪

# 2) 起 TLS 服务（直接用 openssl 的测试服务，或 python http.server 的 SSL 包装）
$ openssl s_server -accept 8443 -cert server.crt -key server.key -www &
# 3) 不校验访问（对照实验 A：-k 跳过校验）
$ curl -sk https://localhost:8443 | head -3
<!DOCTYPE html>...
# 4) 校验访问（对照实验 B：报告自签不可信）
$ curl -s https://localhost:8443
curl: (60) SSL certificate problem: self-signed certificate
# 5) 把自签证书导入信任库后再访问（对照实验 C：校验通过）
$ sudo cp server.crt /usr/local/share/ca-certificates/mytest.crt && sudo update-ca-certificates
$ curl -s https://localhost:8443 | head -2
<!DOCTYPE html>...
```

实验 C 通过的瞬间就是 5 节"信任链"的具象化：**自签证书不是不能用，是"谁来把它放进信任库"的问题**——企业内网自建 CA 的原理与这一步完全相同（分发根 CA 即可）。抓握手（对照 4 节流程图）：

```bash
$ curl -sv https://localhost:8443 -o /dev/null 2>&1 | grep -E 'TLSv1.3|subject|SSL connection'
* TLSv1.3 (OUT), TLS handshake, Client hello (1):
* TLSv1.3 (IN), TLS handshake, Server hello (2):
* TLSv1.3 (IN), TLS handshake, Certificate (11):
* TLSv1.3 (IN), TLS handshake, CERT verify (15):
* TLSv1.3 (IN), TLS handshake, Finished (20):
* SSL connection using TLSv1.3 / TLS_AES_256_GCM_SHA384
```

`TLS_AES_256_GCM_SHA384` 是协商出的密码套件：AES-256-GCM（对称，AEAD 同时管机密与完整性）+ SHA384（握手哈希）——2 节"混合加密"落到一行套件名上。

## 常见误区

1. **"HTTPS 就是加密，加了就安全"**：TLS 只保传输三性（机密/完整/真实），证书配错、老协议版本、SNI/DNS 明文、终端失守都在保护范围外。正确的心智模型：HTTPS 消除的是"链路窃听与篡改"这一类威胁，不是全部威胁。
2. **"RSA 用来加密数据"**：生产 TLS 里 RSA 只用于证书签名验证（1.3 里连密钥交换都不用了）；数据加密永远是对称算法。看到"RSA 加密了你的请求"的科普可以直接划走。
3. **"0-RTT 白赚一个 RTT，开了就好"**：0-RTT 数据可被重放攻击，规范明确只允许幂等请求进 0-RTT；对支付/写操作开 0-RTT 是引入漏洞不是优化（RFC 8446 2.3 与 8 节专门警告）。
4. **"自签证书 = 不安全"**：自签只是"不在公共信任库"；企业内网自建 CA + 信任库分发是标准做法（8 节实验 C）。真正危险的是**跳过校验**（-k）这个习惯本身。
5. **"证书过期几天没事"**：浏览器全屏拦截页、各 SDK 直接拒绝连接、App 白屏——证书过期是可用性事故（著名案例：多家大厂因证书集中同日过期引发大面积故障）。自动化续期 + 监控是唯一解（经验做法）。

## 本篇小结

- TLS 三目标：机密（对称加密）、完整（AEAD 认证）、真实（证书+签名）；第三点是防冒充的关键，加密本身不防冒充。
- 混合加密分工：ECDHE 协商（非对称，一次性）+ AES-GCM 传输（对称，海量数据）；ECDHE 临时密钥带来前向保密——TLS 1.3 删除 RSA 密钥交换正是强制 PFS。
- TLS 1.2 握手 2-RTT；1.3 砍到 1-RTT（ClientHello 带 key_share、第三段起加密），0-RTT 只许幂等请求（重放风险）。
- 证书链：根 CA（信任锚）→中间 CA→站点；openssl s_client 的 depth=2/1/0 与 Verify return code: 0 是链验证的直接观察面；漏配中间证书是经典事故。
- 校验五查：SAN 域名、有效期（注意系统时钟）、信任链、CA 约束、吊销（OCSP/CRL 概览）；curl 60 号错误自检清单（经验做法）。
- 实验闭环：自签证书 → s_server → curl 三态（-k 通过/报错/导入信任库通过）→ 抓 TLS 1.3 握手对照 4 节时序。

## 练习

1. 用 `openssl s_client -connect 你常访问的网站:443 -servername 同域名` 抓出证书的 subject/issuer/有效期/SAN，画出这条证书链（根→中间→叶），并判断它是否启用了 OCSP Stapling（`-status` 选项）。
2. 对同一网站分别强制 `curl --tlsv1.2` 与 `curl --tlsv1.3`，用 `-w '%{time_appconnect}\n'` 对比握手耗时差（多次取样），用 3/4 节的 RTT 差解释数字。
3. 修改 8 节实验：把 server.crt 换成一张 `CN=example.com` 的自签证书访问 https://localhost:8443，记录 curl 的具体报错（hostname mismatch），说出它对应 6 节清单的第几条。
4. 解释题：为什么 TLS 1.3 的 Certificate 消息是加密的，而 TLS 1.2 的是明文？这对中间设备（企业网关、防火墙）各意味着什么？（提示：SNI 可见性与 ECH 的动机。）
5. 用 Wireshark/tshark（或 tcpdump -A）抓一次 `curl https://httpbin.org/get`，确认：TLS 记录层之外你能看到什么（TCP 头、SNI、证书长度）、看不到什么（HTTP 路径、头部、响应体）——把"HTTPS 不隐藏元数据"落到抓包证据上。
