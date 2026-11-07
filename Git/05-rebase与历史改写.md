# 05 - rebase 与历史改写

## 本篇目标

1. 理解 rebase 的机制：**摘下提交、换个基底重放**——以及它为什么必然"改写历史"；
2. 掌握交互式 rebase 这个历史整理的瑞士军刀；
3. 内化那条著名的**黄金法则**，并理解它的技术依据。

## 1. rebase 机制：重放

`git rebase main`（在 feature 分支上执行）做的事：

```
        A ── B ── D            ← main
         \       \
          C ── F                ← feature

rebase 后：
        A ── B ── D                      ← main
                 \
                  C' ── F'               ← feature（换了个新基底）
```

步骤拆解（每一步都是 01 篇对象模型的操作）：

1. 找到两分支的共同祖先（A）；
2. 把 feature 独有的提交（C、F）**暂存为一组 patch**；
3. 把 feature 指针重置到 main 顶（D）；
4. 依次重放 C、F——**每个重放的提交都是新对象**（C'、F'：内容可能相同，但因父提交变了，commit id 必然不同）；
5. 冲突时：逐个提交停下处理（`git rebase --continue` / `--abort`）。

**核心推论**：rebase 之后，原 C、F 仍在对象库里（06 篇 reflog 可找回），但**没有任何分支指向它们**——"历史被改写"的本质是"指针挪走，旧对象成孤儿"。

## 2. 黄金法则：不要 rebase 已共享的提交

> **不要对"已经推送到共享分支的提交"做 rebase。**

技术依据（不是教条）：你在本地把 C、F 重放成 C'、F' 后推送，远程会发现"main 上已经有 C、F，而你居然要推一个不含它们的历史"——只能 force push。此后**其他同事本地还停在 C、F 上**，他们 pull 后会得到两套平行历史，合并噩梦开始。

判别流程（每次 rebase 前过一遍）：

```
这些提交推送过吗？
 ├─ 没有 → 随便 rebase（包括 amend），完全安全
 └─ 推送过 → 这个分支只有我一个人在用吗？
              ├─ 是（个人 PR 分支）→ rebase 后用 --force-with-lease 推（04 篇）
              └─ 否（共享分支）→ 禁止 rebase，用 merge（03 篇）
```

`--force-with-lease` 在这里是"改写了远程快照的个人分支"的安全通道（04 篇的语义：远程引用与你 fetch 时一致才执行）。

## 3. 交互式 rebase：历史整理台

```bash
git rebase -i HEAD~4        # 打开最近 4 个提交的编辑列表（旧的在下面）
```

```
pick 8a1c2fe feat: 导出 JSON 支持
pick d92b3aa 修 typo
pick 4f01e77 feat: 压缩管线
pick a77c190 fix: 压缩在加密前
```

每个提交前的 `pick` 可改为其他动作（编辑器里有完整菜单）：

| 命令 | 作用 | 典型用途 |
| --- | --- | --- |
| `pick` | 保留原样 | — |
| `reword` | 保留改动，改消息 | 提交信息写坏了 |
| `edit` | 停在该提交，允许修改内容 | 想补一个文件进去（改完 `--continue`） |
| `squash`/`fixup` | 与上一个提交**合并**（fixup 丢弃本条消息） | "修 typo""fix fix"并进正经提交 |
| `drop` | 删除该提交 | 撤销误提交 |
| 调整行序 | 重排提交 | 让逻辑顺序合理 |

最常用的组合——**把"过程垃圾"整理成干净提交**：开发时十几个 "wip/fix/typo" 提交，发 PR 前交互式 rebase 把它们 squash 成 2~3 个语义完整的提交。**在个人 feature 分支上做，完全符合黄金法则**（未共享）。

## 4. 其他历史改写工具

| 工具 | 场景 |
| --- | --- |
| `git commit --amend` | 改最后一个提交（消息/内容）；本质是"重放最后一提交"，同样遵守黄金法则 |
| `git commit --fixup <commit>` + `git rebase -i --autosquash` | 日常开发中标记"这是给某提交的补丁"，rebase 时自动归位合并——大特性分支的卫生习惯 |
| `git cherry-pick <commit>` | 把**别的分支上的单个提交**摘到当前分支（重放单提交，热修复的常用通道） |
| `git reset --hard <commit>` | 粗暴的"历史回退"（06 篇详述，与 rebase 的区别是保留不了提交序列） |
| `git filter-repo` | 批量改写**全部历史**（删大文件/敏感信息）；旧工具 `filter-branch` 已被官方文档标记不推荐 |

## 5. rebase 的实用模式：上游同步

个人分支长期开发时，把 main 的进展"接上来"的两种方式：

```bash
git merge main        # 分叉图 + 合并节点（真实但杂乱）
git rebase main       # 线性：你的提交总是基于最新 main（整洁，但每接一次都改写一次自己的历史）
```

选择：**已推送的个人分支用 merge 接上游（避免反复 force push）；未推送/重放成本低时用 rebase**。团队约定优先于个人偏好（09 篇把这项写进协作文档）。

## 6. rebase 事故与救援

rebase 中途乱了（冲突处理错误、顺序搞砸）：

```bash
git rebase --abort          # ① 最常用：整个 rebase 回滚，回到开始前
git reflog                  # ② 已 continue 完才发现错了：reflog 找 rebase 前的位置
git reset --hard HEAD@{2}   #    回到那时（06 篇详解）
```

**只要记住"旧对象还在库里"，rebase 就不是危险操作**——危险的是在共享分支上改写（那不是技术问题，是纪律问题）。

## 自测

1. 复述 rebase 的五步机制。为什么重放后 commit id 必然不同？
2. 黄金法则的技术依据是什么（不是"老师说的"）？
3. squash 与 fixup 的区别？`--autosquash` 配合哪个日常命令使用？
4. 个人已推送分支 rebase 后如何安全推送？
5. rebase 到一半想全部放弃，用什么？"完成才发现错了"呢？

## 下一篇

[06 - 撤销的艺术](06-撤销的艺术.md)——Git 的后悔药体系。

## 参考来源

- Pro Git 第 3.6 节（rebase，含黄金法则）: https://git-scm.com/book
- git-rebase 官方文档（-i 动作菜单、--autosquash）: https://git-scm.com/docs/git-rebase
- git-filter-repo（官方文档推荐的历史改写工具）: https://github.com/newren/git-filter-repo
