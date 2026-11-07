# 06 - 工作区与 monorepo

## 本篇目标

当项目从一个包长成"多个相互依赖的包"（核心库 + API 服务 + CLI），每个包独立建项目会出现"改了库要手动重装到服务里"的裂脑问题。uv 的 **workspaces**（官方文档明确标注 "Cargo-style workspaces"——借鉴 Rust 的成熟设计）用一个根锁文件统一管理多包。

## 1. 认识 workspace：三要素

一个 workspace = **一个根项目（root）+ 若干成员（members）+ 一份共享锁文件（uv.lock）**：

```text
monorepo/
├── pyproject.toml          ← 根项目（也是 workspace 定义处）
├── uv.lock                 ← 全 workspace 共享的唯一锁
├── packages/
│   ├── core/               ← 成员 1：核心库
│   │   └── pyproject.toml
│   ├── api/                ← 成员 2：API 服务（依赖 core）
│   └── cli/                ← 成员 3：命令行（依赖 core）
```

根 `pyproject.toml`：

```toml
[project]
name = "monorepo"           # 根自身可以是个"空壳"项目，也可以有实际代码
version = "0.1.0"
requires-python = ">=3.12"

[tool.uv.workspace]
members = ["packages/*"]    # 成员清单（支持 glob）

[tool.uv.sources]
core = { workspace = true }  # 关键：core 指向 workspace 内的包（而非 PyPI）
```

成员 `packages/api/pyproject.toml`：

```toml
[project]
name = "api"
dependencies = ["core"]     # 声明依赖 core（正常写法）

[tool.uv.sources]
core = { workspace = true } # 且标记：core 来自 workspace 内部
```

## 2. `workspace = true` 是整个设计的钥匙

普通依赖写法 `"core"` 的语义是"从 PyPI 装 core"——monorepo 里这会装到**发布版**而不是你刚改的工作区代码（裂脑问题的根源）。`[tool.uv.sources]` 的 `workspace = true` 把依赖**重定向到工作区成员的本地路径**：

- `uv sync` 时，api 的环境里装的 core 是**本地可编辑引用**（editable install）——改 core 源码，api 立即生效，零重装；
- 对外发布时不受影响：构建 api 的 wheel 时，`core` 依赖仍是正常的 PyPI 依赖名（sources 只是本地解析行为，不写入发布元数据）。

这套"声明 + 源重定向"的分离，与 Rust Cargo 的 `[workspace.dependencies]` 设计同构——Rust 社区已用这套模式支撑了超大型 monorepo，uv 直接移植其经验。

## 3. 一把锁的价值

全 workspace **只有一份 `uv.lock`**（在根目录）：

1. **一致性**：core 被三个包依赖，三方环境里的传递依赖版本**必然相同**——"在我包里好好的，在他服务里坏了"的版本漂移类 bug 被结构性消灭；
2. **效率**：一次解析（所有成员的需求一起解）、一次 `uv sync`（各成员环境共享缓存去重，04 篇缓存机制在这里放大收益）；
3. **原子性**：改 core 的依赖 → 根目录 `uv lock` → 一把锁覆盖所有下游的重新校准。

日常操作全在根目录进行：

```bash
uv sync                    # 根 + 所有成员环境就绪
uv run --package api ...   # 在 api 包的环境里执行命令（--package 指定成员）
uv add --package cli "rich>=13"    # 给指定成员加依赖
```

## 4. 什么时候用 workspace（什么时候不用）

判别问题只有一个：**多个包是否同源、同节奏演进？**

| 情形 | 建议 |
| --- | --- |
| 核心库 + 多个消费它的服务/CLI，同仓库同节奏 | ✅ workspace 的主场 |
| 多个完全独立的产品放一个仓库 | 单独项目更简单（workspace 强加的统一锁对独立产品是负担） |
| 依赖**别人**维护的包 | 普通 `uv add`，与 workspace 无关 |

与 Git 侧的衔接（Git 07 篇）：monorepo 大到需要 partial clone / sparse-checkout 时，workspace 照常工作——两者解决的是不同维度（依赖管理 vs 检出范围）。

## 5. 最小可运行示例（动手 5 分钟）

```bash
mkdir mono && cd mono && uv init
mkdir -p packages/core packages/cli
uv init --lib packages/core
uv init --app packages/cli
# 手工按 §1 在根与 cli 的 pyproject.toml 里加 [tool.uv.workspace] 与 sources
uv sync
# 验证：改 packages/core/src/core/__init__.py 的一行
#      uv run --package cli python -c "import core; ..." → 立即看到改动
```

## 自测

1. workspace 三要素？锁文件为什么必须只有一把？
2. `workspace = true` 解决的"裂脑"具体是什么？发布时它如何不影响元数据？
3. "多个独立产品"为什么不该用 workspace？
4. 改 core 的源码后，api 环境里的 core 为什么立即生效？（editable install 机制）

## 下一篇

[07 - 脚本与单文件程序](07-脚本与单文件程序.md)——PEP 723：把依赖写进脚本本身。

## 参考来源

- uv 官方文档·Workspaces: https://docs.astral.sh/uv/concepts/workspaces/
- Cargo Book·Workspaces（设计源头）: https://doc.rust-lang.org/cargo/reference/workspaces.html
- PEP 660（editable install 的现代标准，workspace 依赖的底层机制之一）: https://peps.python.org/
