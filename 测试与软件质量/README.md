# 测试与软件质量系列教程：从"能跑就行"到质量工程

> 面向写得出功能、但对"怎么证明它一直对"心里没底的学习者：改一处坏三处的恐惧、不敢重构的祖传代码、CI 红了就点重跑的侥幸——本系列把**测试**当作软件质量的核心工程实践系统讲一遍：pytest 生态（fixture/mock/property-based/集成测试）、CI 质量门禁（静态分析/覆盖率/fuzzing/性能回归）、C++ 侧的 GoogleTest 与 Sanitizer，最后用一个真实项目从零补齐全部测试。
> 事实基准：Kent Beck《Test-Driven Development: By Example》定义了测试驱动的心智模型，Michael Feathers《Working Effectively with Legacy Code》定义了"可测试性"，pytest 与 GoogleTest 的细节以各自[官方文档](https://docs.pytest.org/)为准——工具会变，质量思想不变。

## 学习路径

| 篇 | 主题 | 你将学会 |
| --- | --- | --- |
| [01-为什么测试是顶级工程师的加速器](01-为什么测试是顶级工程师的加速器.md) | 不测试的成本、测试金字塔 | 算清测试的经济账，用 FIRST 原则判断什么是好测试 |
| [02-pytest基础](02-pytest基础.md) | 断言重写、组织结构、参数化 | pytest 8.x 日常用法全覆盖，告别 unittest 手稿 |
| [03-fixture与测试架构](03-fixture与测试架构.md) | 作用域、依赖注入、conftest 分层 | 用 fixture 搭出可维护的测试架构 |
| [04-Mock与测试替身](04-Mock与测试替身.md) | 五类替身、mock 边界原则 | 知道什么时候该 mock、什么时候 Fake 更好 |
| [05-Property-Based测试](05-Property-Based测试.md) | hypothesis、自定义 strategy、状态测试 | 从"举例子"升级到"陈述性质" |
| [06-测试策略与可测试性设计](06-测试策略与可测试性设计.md) | seam、依赖注入、characterization | 给遗留代码加测试，设计天生可测的模块 |
| [07-集成测试与外部依赖](07-集成测试与外部依赖.md) | 数据库/HTTP/文件系统测试 | 边界层测试稳、快、可重复 |
| [08-CI流水线与质量门禁](08-CI流水线与质量门禁.md) | GitHub Actions、覆盖率门禁 | Python+C++ 矩阵流水线与失败纪律 |
| [09-静态分析与Fuzzing](09-静态分析与Fuzzing.md) | mypy/ruff/clang-tidy/libFuzzer | 让机器在你之前找到 bug |
| [10-性能测试与回归](10-性能测试与回归.md) | timeit、pytest-benchmark、Google Benchmark | 慢也是 bug：可重复的基准与回归防线 |
| [11-C++测试与双语言工程](11-C++测试与双语言工程.md) | GoogleTest、CMake、pybind11 | C++ 侧测试体系与双语言项目布局 |
| [12-实战-给真实项目补测试](12-实战-给真实项目补测试.md) | 综合项目：Agent 工具库补测试 | 全流程演练 + 面试高频 20 题 |

### 阶段划分

1. **为什么与工具地基（01–02）**：先算账再动手；pytest 从断言到 fixture 的日常主线。
2. **测试架构（03–06）**：fixture 分层、测试替身边界、property-based、可测试性设计——这四篇决定你写的测试是资产还是负债。
3. **边界与防线（07–10）**：集成测试的稳定性、CI 门禁、静态分析与 fuzzing、性能回归——把质量从个人习惯变成团队机制。
4. **C++ 与综合（11–12）**：双语言工程收口，再用一个真实项目把全系列串成一次完整演练。

## 与本仓库其他方向的衔接

- **Python 语言精通**：语言地基——03 装饰器、05 上下文管理器、06 类型注解在测试代码里是日常工具；本系列默认你已掌握 pytest 之外的 Python 语法；
- **Agent 开发 13（Agent 评测体系）**：本系列思想在 LLM 场景的延伸——单测断言的是"函数输出"，Agent 评测断言的是"行为质量"，12 篇的实战对象正是一个 Agent 工具库，两篇对照读收获最大；
- **uv 08（迁移 CI 与团队落地）**：CI 的工具链细节——`uv sync --frozen`、缓存、发布；本系列 08 篇讲质量门禁"装什么门"，uv 08 讲"用什么水管"，互为表里；
- **C++ 系列**：姊妹系列——11 篇的 GoogleTest/CMake/Sanitizer 建立在 C++ 01–05 的语法与 RAII 地基上，双语言项目的测试布局是两个系列的汇合点。

## 约定

- Python 3.10+（用 `match`、`X | Y` 等现代语法），pytest 8.x，示例输出取自 pytest 8.4 实测；
- C++ 用 C++17 与 GoogleTest 1.14+，构建用 CMake 3.22+；
- 所有 shell 命令在 Ubuntu 22.04（WSL2）可直接执行；项目布局统一为"src 布局 + tests/ 镜像目录"，两套语言并存时见 11 篇的目录约定。
