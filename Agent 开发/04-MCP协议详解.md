# 04 - MCP 协议详解

## 本篇目标

1. 理解 MCP 解决的问题：M×N 集成难题；
2. 掌握 MCP 的三层架构（Host / Client / Server）与核心原语（tools / resources / prompts）；
3. 了解传输方式（stdio / Streamable HTTP）与安全要点；
4. 写一个最小 MCP Server 并在客户端使用。

## 1. MCP 是什么、为什么需要它

没有 MCP 之前，每个 AI 应用要接入 M 个数据源/工具，每个工具要适配 N 个应用——**M×N 种集成**。MCP 把它变成 **M+N**：工具方实现一次 MCP Server，应用方实现一次 MCP Client，即可互相连通。

**Model Context Protocol（MCP）**是 Anthropic 于 **2024 年 11 月 25 日**发布并开源的协议，用于标准化 LLM 应用与外部数据源、工具之间的连接。技术上构建在 **JSON-RPC 2.0** 之上，并定义了能力协商、订阅等语义。它很快成为事实标准：OpenAI 于 2025 年 3 月宣布支持（Agents SDK、ChatGPT 桌面版等），Google DeepMind 于 2025 年 4 月宣布在 Gemini 中支持。2025 年 12 月，Anthropic 将 MCP 捐给 Linux Foundation 旗下的 **Agentic AI Foundation（AAIF）**，进入厂商中立的开放治理。

> 心智模型：**MCP 之于 AI 应用 ≈ USB-C 之于外设**。这是 Anthropic 官方公告中使用的类比：一个协议口，插上"文件系统、数据库、GitHub、浏览器"等各种外设。

## 2. 三层架构：Host、Client、Server

MCP 采用客户端-服务器架构，注意是**三层**而不是两层：

```
┌──────────────────────────────────────────────┐
│  MCP Host（宿主应用，如 Claude Desktop / IDE）  │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │MCP Client│  │MCP Client│  │MCP Client│   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└───────┼──────────────┼─────────────┼─────────┘
        │ 1:1          │             │
   ┌────▼─────┐  ┌─────▼────┐  ┌────▼─────┐
   │MCP Server│  │MCP Server│  │MCP Server│
   │(文件系统) │  │(GitHub)  │  │(数据库)   │
   └──────────┘  └──────────┘  └──────────┘
```

- **Host**：用户使用的 LLM 应用，内部管理多个 Client。LLM 本体运行在 Host 侧；
- **Client**：Host 内的连接器，与一个 Server 保持 **1:1** 连接并管理协议会话；
- **Server**：独立的进程/服务，暴露某一类能力（文件、API、数据库……）。Server 可以是本地进程，也可以是远程服务。

## 3. 三大原语 + 两个反向能力

Server 可以向 Client 暴露三类东西（协议规范原文称 primitives）：

| 原语 | 由谁控制 | 含义 | 典型例子 |
| --- | --- | --- | --- |
| **Tools**（工具） | **模型控制**（model-controlled） | 可被 LLM 决定调用的函数，有副作用 | `create_issue`、`run_query` |
| **Resources**（资源） | **应用控制**（application-controlled） | 只读上下文数据，由应用决定注入什么 | 文件内容、数据库 schema、日志 |
| **Prompts**（提示模板） | **用户控制**（user-controlled） | 预制的提示词模板，用户主动选择 | 斜杠命令、快速操作菜单 |

区分记忆法：**Tools = 手，Resources = 参考资料，Prompts = 话术模板**。控制权归属决定了交互方式——工具调用由模型自主发起；资源由应用决定是否放进上下文；提示由用户显式触发。

除这三个原语外，协议还有 **Client → Server 反向的能力**（由 Host 实现并向 Server 提供）：

- **Sampling（采样）**：Server 可以请 Host 的 LLM 补充生成一次（例如审查类 Server 需要用 LLM 摘要结果），模型调用权仍在用户控制的 Host 手里；
- **Roots**：Client 告知 Server 其被允许操作的文件系统边界；
- **Elicitation**（2025-06-18 规范新增）：Server 在操作中途向**用户**请求补充信息（如缺失参数）。规范明确禁止用它索要密码、API 密钥、令牌和支付信息。

## 4. 传输方式

| 传输 | 场景 | 说明 |
| --- | --- | --- |
| **stdio** | 本地 Server | Host 以子进程启动 Server，通过标准输入/输出交换 JSON-RPC 消息。最简单，本地首选 |
| **Streamable HTTP** | 远程 Server | **2025-03-26 规范**引入，取代早期的 HTTP+SSE 双端点方案：单一 HTTP 端点，既支持普通 HTTP 响应也支持按需升级为 SSE 流 |
| stdio 之外的本地选项 | 桌面场景 | 部分宿主还支持 WebSocket 等扩展传输（非核心规范） |

远程 MCP Server 的鉴权在 2025-06-18 规范中基于 **OAuth 2.1**（Server 可作为 OAuth 资源服务器）。

## 5. 动手：用 Python SDK 写一个最小 MCP Server

官方维护多语言 SDK（Python、TypeScript、Java、Kotlin、C# 等）。Python SDK 提供 `FastMCP` 高层接口：

```python
# server.py — 最小 MCP Server（pip install "mcp[cli]" 后运行）
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """两数相加。当需要精确算术时使用，不要心算。"""
    return a + b

@mcp.tool()
def failing_tool() -> str:
    """演示错误的返回方式。"""
    raise ValueError("这个工具总是失败——错误信息会作为调用结果回传")

if __name__ == "__main__":
    mcp.run()          # 默认 stdio 传输
```

函数签名和 docstring 会被 SDK 自动转成 JSON Schema（03 篇的准则在这里依然全部适用）。

在 Claude Desktop 中注册（`claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "demo": {
      "command": "python",
      "args": ["/绝对路径/server.py"]
    }
  }
}
```

也可以直接用 SDK 自带的调试工具做协议级测试，不依赖任何宿主：

```bash
mcp dev server.py   # 启动 MCP Inspector，可在浏览器中查看/调用工具
```

> 自己实现客户端时（不用现成宿主），握手顺序是：`initialize` 请求（声明协议版本与能力）→ 服务端响应 → 客户端发 `notifications/initialized` → 之后才能 `tools/list`、`tools/call`。工具调用结果里的 `isError: true` 表示业务级失败——和 03 篇"错误回喂"原则一致，失败信息要交回模型处理。

## 6. 安全：MCP 带来的新攻击面

MCP 方便了工具接入，也**集中了风险**。开发与使用 MCP Server 时必须记住：

1. **第三方 Server 不可信任为"插件商店级"安全**。安装一个 MCP Server 意味着把它连接的数据暴露给宿主 LLM、把它的工具能力交给模型——等于给它信任。审核来源，最小化授权（OAuth scope、只读凭据）。
2. **工具描述本身是注入载体**（详见 09 篇）。恶意 Server 可以在工具描述里藏提示词，诱导模型滥用**其他**工具。这被称为跨 Server 的间接注入风险，社区有所谓 "tool poisoning" 的讨论；规范也在持续强化相关安全要求。防御：人审启用新工具时的描述、宿主侧做工具描述审查。
3. **混淆代理问题（confused deputy）**：宿主拿着用户的凭据干活，攻击者通过注入内容操纵宿主的行为。OAuth 2.1 资源服务器分离等规范演进就是针对这类问题的加固。
4. **能力协商要显式**：Host 应该向用户展示将要连接的 Server 及其工具清单，而不是静默连接。

## 自测

1. M×N → M+N 的说法中，M 和 N 各指什么？
2. Tools / Resources / Prompts 三原语分别由谁控制？这个控制权差异如何影响交互？
3. 本地接入用哪种传输？远程推荐哪种？Streamable HTTP 取代了什么方案？
4. 为什么说"安装第三方 MCP Server"是有安全含义的动作？说出两条缓解措施。

## 下一篇

[05 - 记忆与上下文管理](05-记忆与上下文管理.md)——会话状态、记忆压缩与上下文窗口工程。

## 参考来源

- Anthropic, *Introducing the Model Context Protocol*（2024-11-25）: https://www.anthropic.com/news/model-context-protocol
- MCP 官方文档与规范（含 2025-03-26、2025-06-18 等修订与 changelog）: https://modelcontextprotocol.io / https://modelcontextprotocol.io/specification/2025-06-18/changelog
- OpenAI, *New tooling to help developers build agents*（2025-03-26，宣布支持 MCP）: https://openai.com/index/new-tooling-to-help-developers-build-agents/
- Google 对 MCP 的支持：2025-04-09 Demis Hassabis 宣布 Gemini 模型与 SDK 将支持 MCP（TechCrunch 报道: https://techcrunch.com/2025/04/09/google-says-itll-embrace-anthropics-standard-for-connecting-ai-models-to-data ）；2025 年 5 月 Google I/O 宣布 Gemini SDK 原生支持（The New Stack: https://thenewstack.io/google-embraces-mcp ）
- Anthropic, *Donating the Model Context Protocol…Agentic AI Foundation*（2025-12，Linux Foundation）: https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation
- Python SDK（FastMCP）: https://github.com/modelcontextprotocol/python-sdk
