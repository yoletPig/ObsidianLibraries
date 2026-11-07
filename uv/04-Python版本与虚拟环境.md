# 04 - Python 版本与虚拟环境

## 本篇目标

把"本机到底有几个 Python、项目用哪个、环境建在哪"三个经典玄学问题变成**确定性操作**。核心事实：uv 自己管理 Python 解释器（基于 python-build-standalone 独立构建发行版），不依赖也不扰动系统 Python。

## 1. 解释器从哪来：uv 的托管 Python

```bash
uv python list                    # 列出：已装的 + 可安装的版本
uv python install 3.12 3.13       # 安装（可多版本并存）
uv python install --preview pypy  # 甚至 PyPy（文档示例支持）
```

关键机制事实：

1. uv 安装的解释器来自 **python-build-standalone** 项目（Astral 参与维护的独立 Python 发行版）——**不碰系统 Python**（macOS 的 /usr/bin/python3、Linux 发行版的 python、Windows 的商店版都原样不动）；
2. 多版本并存，安装位置在 uv 的数据目录（可用 `uv python dir` 查看）；
3. 这解决了 pyenv 时代"要编译、要装构建依赖、切换靠 shim"的三重麻烦：预编译、即装即用、按项目声明切换（无全局切换概念——**切换单位从"机器"变成"项目"**）。

> 系统升级/IDE 识别：IDE（如 VS Code 的 Python 插件）可以指定 `.venv` 里的解释器路径，与 uv 完全兼容。

## 2. 项目绑定版本：.python-version

```bash
uv python pin 3.12       # 写入 .python-version 文件：3.12
cat .python-version      # → 3.12
```

这份小文件（02 篇已见）是项目解释器要求的**唯一声明处**：

- `uv venv` / `uv sync` / `uv run` 都会读它，自动找到（或下载）匹配的解释器；
- **提交进 Git**：团队成员与 CI 自动获得一致的解释器版本（03 篇"必提交"清单的一员）；
- 兼容性：pyenv 用户同样认识这个文件名——混用团队不会互踩。

**requires-python 与 .python-version 的分工**（易混点）：

| | `pyproject.toml` 的 `requires-python` | `.python-version` |
| --- | --- | --- |
| 语义 | 本项目**兼容的最低**解释器（发布用，约束下游） | 本项目**开发用**的精确版本（团队一致） |
| 例子 | `>=3.10`（库要兼容 3.10+） | `3.12`（开发者都用 3.12 开发） |

库项目常见组合：requires-python 写宽（>=3.10），开发用 3.12 锁定 + CI 里测 3.10/3.11/3.12 矩阵（08 篇）。

## 3. 虚拟环境：从仪式到实现细节

### 3.1 默认行为：你几乎不用碰

- `uv sync` / `uv add` / `uv run` 在项目根自动创建/维护 **`.venv/`**（不存在则建，存在则对齐锁）；
- `.venv` 用项目绑定解释器创建（读 `.python-version`）。

### 3.2 手动场景：uv venv

```bash
uv venv                       # 在当前目录建 .venv
uv venv env310 --python 3.10  # 指定解释器与目录（临时/特殊用途）
uv venv --python 3.12.4       # 精确到补丁版本；该版本没装会自动下载
source .venv/bin/activate     # 需要时照常激活（Windows: .venv\Scripts\activate）
```

`uv venv` vs `python -m venv` 的差别：**快一个数量级 + 解释器自动就位**（不够会下载）。创建出的目录结构是标准 venv，任何工具（IDE、老脚本）都认。

### 3.3 环境坏了怎么办：删了重建

uv 世界里环境是**可再生品**（03 篇可重建的不入库原则）：

```bash
rm -rf .venv && uv sync       # 万能修复法（或 uv sync --reinstall）
```

对比旧世界"环境坏了修半天"：环境的价值含量低了——**事实都锁在 `uv.lock` 里**，环境只是锁的一次实例化。

## 4. 常见疑问（FAQ 形式的排错手册）

**Q1：`uv run` 用的是哪个 Python？**
按顺序：`--python` 参数 → 项目 `.python-version` → `pyproject.toml` requires-python 匹配的托管/系统解释器。排查用 `uv run python -c "import sys; print(sys.executable)"` 看实际路径。

**Q2：为什么我 `pip install` 装的包不见了？**
`uv sync` 按锁对齐环境，把锁外的包移除了（03 篇：这是特性）。装包请走 `uv add`（进锁，可复现）或明确知道自己在做什么时用 `uv pip install`（不走锁的兼容层，01 篇的角色区分）。

**Q3：uv 下载 Python/包很慢？**
配置镜像（`UV_DEFAULT_INDEX` 环境变量或 `[[tool.uv.index]]` 配置，08 篇细讲）；Python 发行版的下载地址也可用 `UV_PYTHON_INSTALL_MIRROR` 替换（国内环境常见需求，见官方文档 ENV 页）。

**Q4：全局工具脚本要用特定 Python？**
用 05 篇的 `uvx --python 3.12 <tool>`。

## 5. 把全流程串起来（新人入职 3 分钟）

```bash
# 机器上什么都不用装，除了 uv 本身
git clone <公司项目> && cd <项目>
uv sync                       # 自动：下载 .python-version 指定的 Python → 建 .venv → 按锁装包
uv run pytest                 # 开工
```

对照传统入职文档（装系统依赖 → pyenv → virtualenvwrapper 配置 → requirements → 手动排错），这一段是 uv 价值最直观的地方。

## 自测

1. uv 的托管 Python 来自哪个项目？为什么不碰系统 Python？
2. `requires-python` 与 `.python-version` 各管什么？库项目的推荐组合？
3. "环境是可再生品"——这个观点如何改变排错方式？
4. `pip install` 装的包被 `uv sync` 移除，是 bug 吗？正确装包方式？

## 下一篇

[05 - uvx 与工具管理](05-uvx与工具管理.md)——一次性工具与常驻 CLI。

## 参考来源

- uv 官方文档·Python versions / projects（含 python-build-standalone 说明）: https://docs.astral.sh/uv/concepts/python-versions/
- python-build-standalone 项目: https://github.com/astral-sh/python-build-standalone
- uv 官方文档·Environment variables（UV_PYTHON_INSTALL_MIRROR 等）: https://docs.astral.sh/uv/reference/environment/
