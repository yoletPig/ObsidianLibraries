# 05 - uvx 与工具管理

## 本篇目标

Python 生态的 CLI 工具（ruff、black、httpie、pdm…）一直有"装在哪、污染谁的环境"的经典问题。pipx 的答案——**每个工具一个隔离环境**——被 uv 以 `uv tool` / `uvx` 完整继承并提速。本章讲清两种使用模式与脚本场景。

## 1. 两种模式：临时运行 vs 常驻安装

```bash
# 模式一：uvx（= uv tool run）——临时运行，用完即弃
uvx ruff check .               # ruff 环境不存在？自动建临时环境、安装、运行、缓存备用
uvx cowsay -t "hello"          # 任何 PyPI 上的 CLI 都可以这样"白嫖"一次

# 模式二：uv tool install——常驻安装
uv tool install ruff           # 装进独立环境，命令进 ~/.local/bin（PATH 内）
ruff check .
uv tool list                   # 已装工具清单
uv tool upgrade ruff / --all   # 升级（工具环境与项目无关，互不影响）
uv tool uninstall ruff
```

选择判据：**用一次/偶尔用 → uvx；每天用 → tool install**（省去每次的启动检查开销）。uvx 对已运行过的工具有缓存，第二次也快——实际差异比想象小，拿不准就先 uvx。

### 与"全局 pip install"的对比（为什么这是正确姿势）

| | `pip install`（全局/项目内） | `uv tool install` |
| --- | --- | --- |
| 依赖隔离 | ❌ 与其他包抢同一环境（版本冲突温床） | ✅ 每工具独立环境 |
| 项目污染 | ❌ 工具依赖混进项目锁 | ✅ 与项目锁无关 |
| 卸载干净度 | ❌ 留残骸 | ✅ 一条命令 |

经典事故重现：项目要 `requests==2.20`，全局 CLI 工具要 `requests>=2.30`——传统全局安装必炸；tool/uvx 模式下两者互不相识。

## 2. uvx 的进阶用法

```bash
uvx --from ruff ruff check .          # 指定来源包（命令名≠包名时）
uvx --from "ruff==0.6.9" ruff --version   # 锁定工具版本
uvx --python 3.10 old-tool            # 指定工具的运行解释器（04 篇 Q4）
uvx black --diff file.py              # 参数照常透传
```

**锁版本是可复现 CI 的关键**（08 篇）：`uvx black` 今天与明年可能执行不同版本——流水线里用 `uvx --from "black==24.8.0"` 或把工具声明进项目 dev 组。

## 3. 与项目开发依赖的边界

"ruff 该 `uv tool install` 还是 `uv add --dev`？"——按**一致性与用法**分：

| 场景 | 推荐 | 理由 |
| --- | --- | --- |
| 团队代码规范工具（ruff/pytest/mypy） | `uv add --dev`（进锁） | **版本进锁 = 全团队与 CI 同一版本**；工具版本不同，检查结果就不同 |
| 个人效率工具（httpie、cowsay、jupyter lab） | `uv tool install` / `uvx` | 与项目无关，锁文件不该被个人偏好污染 |
| 临时试用一个包的 CLI | `uvx` | 零承诺 |

一句话：**影响团队产出一致性的进锁（dev 组），个人用的走工具环境**。

## 4. uv run 的兄弟场景：--with 与 --no-project

```bash
# 临时带一个依赖跑脚本（不改任何项目文件）：
uv run --with rich python -c "import rich; rich.print('hi')"
uv run --with "flask==3.0" app.py

# 在项目环境之外跑（明确脱离项目锁）：
uv run --no-project python script.py
```

`--with` 是"一次性依赖"的最轻表达：适合临时实验、CI 里跑个检查脚本。它与 07 篇的 PEP 723 是一条光谱的两端：`--with` 适合一两行的临时需求；**依赖多了、脚本要进 Git 分享了，就写成 PEP 723 内嵌元数据**。

## 5. 场景演练

**演练 A：换新电脑第一天**

```bash
uv tool install ruff black httpie    # 三个常用 CLI 一次装齐，无任何环境烦恼
```

**演练 B：读别人的 Notebook，缺个包**

```bash
uv run --with polars jupyter lab     # 不动任何已装环境，临时补上 polars
```

**演练 C：CI 里的 lint 步骤**

```bash
uvx --from "ruff==0.6.9" ruff check src/    # 锁版本（可复现）
```

## 自测

1. uvx 与 uv tool install 的选择判据？
2. "全局 pip install 工具"的两类事故？tool 模式如何免疫？
3. 团队规范工具为什么该进 dev 组而不是 tool install？（一致性视角）
4. `--from` 什么时候必须用？
5. `uv run --with` 与 PEP 723 脚本的分工？

## 下一篇

[06 - 工作区与 monorepo](06-工作区与monorepo.md)——Cargo 式多包仓库。

## 参考来源

- uv 官方文档·Tools concept: https://docs.astral.sh/uv/concepts/tools/
- pipx 文档（隔离安装模式的鼻祖，uv tool 的对照物）: https://pipx.pypa.io/
- uv 官方文档·uv run（--with/--no-project）: https://docs.astral.sh/uv/reference/cli/
