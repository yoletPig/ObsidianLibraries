# 06 - 多 Agent 协作

## 本篇目标

1. 先回答最关键的问题：**什么时候才需要多 Agent**（答案是：比你以为的少得多）；
2. 掌握三种主流协作模式的取舍；
3. 知道多 Agent 特有的成本与失败模式；
4. 对主流框架建立事实性认知，不迷信。

## 1. 什么时候需要多 Agent

先说结论（与 Anthropic《Building Effective Agents》《How we built our multi-agent research system》的立场一致）：

> **单 Agent + 好工具 能解决的，就不要拆多 Agent。** 多 Agent 不是"更先进"，而是**权衡**。

满足以下条件之一，才值得拆：

1. **子任务天然并行、互不依赖**：调研 10 个竞品——每个调研线程独立，可并行，互不污染；
2. **上下文隔离有收益**：某个子任务需要读海量资料（几十万 token），会"撑爆/污染"主对话——把它隔离到子 Agent，只把结论带回来；
3. **需要不同专业配置**：不同子任务需要不同的系统提示/工具集（写代码的 Agent 不需要挂简历解析工具）。

反之，一个子任务串行依赖另一个、或者单 Agent 十几步就能做完的，拆开只会增加协调成本与失败点。

## 2. 三种协作模式

### 模式一：Orchestrator–Worker（编排者-执行者）

```
        ┌────────────┐
        │ Orchestrator│ 决定拆分 → 分派 → 汇总
        └──┬───┬───┬──┘
      ┌────▼┐ ┌▼────┐ ┌▼────┐
      │Wkr 1│ │Wkr 2│ │Wkr 3│   各自独立执行，返回结果
      └─────┘ └─────┘ └─────┘
```

- 主 Agent 负责拆任务、分派、整合；Worker 只干自己的活；
- **这是默认推荐模式**：控制权集中，行为可预测，好调试；
- Anthropic 的多 Agent 研究系统（Claude 的深度研究功能）就是"主 Agent + 并行子 Agent"的结构。

### 模式二：Pipeline（流水线）

```
输入 → Agent A（起草） → Agent B（审查） → Agent C（改写） → 输出
```

- 固定顺序、各环节职责单一；本质上是 01 篇说的"工作流 + 每步一个 Agent"；
- 适合**角色对峙**型任务：生成者与审查者分开，比一个 Agent 自查效果好；
- 缺点：不灵活——环节固定，无法应对需要动态分支的任务。

### 模式三：自由对话/群聊（peer-to-peer）

```
Agent A ⇄ Agent B ⇄ Agent C     （大家都能说话，谁组织不定）
```

- 直觉上"像人类团队"，实践中**最难控制**：容易进入互相恭维、话题漂移、没人负责收口的局面；
- 只在需要模拟多方博弈/辩论（如红蓝对抗、多视角评审）时考虑，且仍要指定一个"主席"负责终止。

**推荐顺序：模式一 → 模式二 →（很少）模式三。**

## 3. 多 Agent 的真实成本

Anthropic 在多 Agent 研究系统的工程复盘（2025-06-13）中给出的量化观察值得所有团队记取：

- **token 成本：单 Agent 约为普通聊天的 4 倍，多 Agent 系统约为普通聊天的 15 倍**——成本对任务复杂度高度敏感；
- 有效性数据（同文）：在他们的内部研究评测上，以 Claude Opus 4 为主 Agent、Sonnet 4 为子 Agent 的多 Agent 系统**比单 Agent Opus 4 好 90.2%**；同时 token 用量本身能解释 BrowseComp 成绩方差的约 80%——"多花 token"与"多 Agent 架构"的效果是绑定的，**小任务盲目套多 Agent 只会白花 15 倍的钱**；
- 该文通篇强调效果差异来自工程细节（子 Agent 的提示词、探索边界、工具接口、评测），这是复现效果时真正要花力气的地方。

此外，多 Agent 引入了单 Agent 没有的新失败模式：

| 失败模式 | 表现 | 缓解 |
| --- | --- | --- |
| 电话游戏（信息衰减） | 主 Agent 的指令经转述后失真，Worker 干错活 | 分派消息写明"目标/边界/交付物格式"，避免口口相传 |
| 职责重叠 | 两个 Worker 做了同一件事或互相等待 | 编排者显式划分边界与依赖 |
| 协调开销 | 拆分/汇总本身花掉的 token 与步数超过收益 | 任务开始前先估："单 Agent 做会怎样？" |
| 级联失败 | 一个 Worker 崩了，整体卡死 | Worker 级超时 + 降级（该子任务失败不拖垮全局）+ 可恢复 |

## 4. 框架生态：事实性速览

以下只列"是什么"，不评高下（各项目迭代很快，以官方文档为准）：

- **LangGraph**（LangChain 团队）：把 Agent 建模为**图/状态机**，节点是步骤、边是控制流，支持 checkpoint 与人审节点。适合需要精细控制流程的团队；
- **OpenAI Agents SDK**（2025-03 发布，前身实验框架 Swarm）：轻量，核心概念是 Agent、handoff（把控制权移交给另一个 Agent）、guardrail。OpenAI 2025 年发布的 A2A 相关支持亦围绕互操作展开；
- **CrewAI**：基于"角色扮演团队"心智模型（role/goal/backstory），上手快，适合编排风格明确的任务；
- **AutoGen**（微软，开源研究框架，AutoGen v0.4 起重写为事件驱动架构）：支持多 Agent 对话式协作，学术与原型场景常用；
- **A2A 协议**（Agent2Agent，Google 于 2025 年 4 月提出，2025 年 6 月捐给 Linux Foundation）：定位于**不同厂商 Agent 之间的互操作**，与 MCP（Agent↔工具/数据）互补——MCP 连接"手脚"，A2A 连接"同事"。

选型建议：先想清楚你要的是 02 篇那种手写循环、还是图编排、还是跨组织互操作；框架解决的是工程效率问题，**不解决 07 篇要讲的可靠性问题**。

## 自测

1. 说出三条"该拆多 Agent"的条件，再说出一条"不该拆"的信号。
2. 三种协作模式各自的适用场景？哪种最应该谨慎？
3. 多 Agent 的 token 成本大约是普通聊天的多少倍？单 Agent 呢？

## 下一篇

[07 - Agent 开发常见问题与陷阱](07-Agent开发常见问题与陷阱.md)——把前面所有环节的故障模式一次讲透。

## 参考来源

- Anthropic, *Building Effective Agents*（2024-12）: https://www.anthropic.com/research/building-effective-agents
- Anthropic Engineering, *How we built our multi-agent research system*（2025-06-13，token 4×/15×、评测 90.2%、token 解释 80% 方差均出自此文）: https://www.anthropic.com/engineering/built-multi-agent-research-system
- OpenAI Agents SDK 文档: https://openai.github.io/openai-agents-python/
- LangGraph 文档: https://langchain-ai.github.io/langgraph/
- A2A 官方站点（协议规范与治理信息）: https://a2a-protocol.org ；2025-04 由 Google 发布、2025-06 捐入 Linux Foundation（Linux Foundation 公告: https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-enable-secure-intelligent-communication-between-ai-agents ）
