# 02 - SOLID 原则

## 本篇目标

SOLID 是五个面向对象设计原则的首字母缩写，由 Robert C. Martin（Uncle Bob）在 2000 年前后整理并提出（Michael Feathers 建议了 SOLID 这个缩写）。本篇逐个讲透五原则：动机、反例、正例、以及**过度应用**的代价。

先说清定位：**SOLID 是"尺子"，模式是"答案"**。模式是针对特定问题的固定解法；原则是评价任何设计好坏的通用标准。学完本篇，你应该能对任何一段代码说出"它违反了哪条原则、违反带来的具体伤害"。

## S — 单一职责原则（Single Responsibility Principle）

> **一个类应该只有一个引起它变化的原因。**（Martin 的表述）

注意：SRP 的"职责"不是"只干一件事"（那是函数层面的最小化），而是**只有一个变化的理由**。变化的理由来自"为谁而变"——Martin 的经典例子：一个 `Employee` 类同时被 CFO（算薪酬）、COO（汇报工时）、CTO（存数据库）三方要求修改，它就有三个变化原因，应拆开。

```python
# 反例：三个变化原因挤在一个类里
class OrderProcessor:
    def process(self, order):
        ...                       # 业务规则（产品经理驱动变化）
        html = render(order)      # 报表格式（运营驱动变化）
        db.save(order)            # 持久化方式（DBA/架构驱动变化）

# 正例：按"变化原因"拆开
class OrderService:      ...   # 只懂业务规则
class OrderRepository:   ...   # 只懂存储
class OrderReport:       ...   # 只懂报表
```

**伤害清单**（违反 SRP 时）：改一处报表格式，要回归测试整个业务逻辑；类名与内容脱节；合并冲突高发。

**过度应用的代价**：拆成几十个只有一行的小类，跳转成本飙升。经验阈值：一个类**叙述它的职责**时若需要用"和"字，就该警惕；但两三个内聚紧密的小职责合在一起也正常。

## O — 开闭原则（Open-Closed Principle）

> **软件实体应当对扩展开放，对修改关闭。**（Bertrand Meyer 1988 年提出，Martin 引入 OOP 语境）

含义：新增一种行为时，**加新代码**（扩展），而不是**改老代码**。老代码被改，就可能被改坏——所有依赖它的地方都要重新测试。

```python
# 反例：每加一种支付方式都要改这个函数
def pay(order, method):
    if method == "alipay":    ...
    elif method == "wechat":  ...

# 正例：新支付方式 = 新增一个类，pay 不动
class Payment(ABC):
    @abstractmethod
    def charge(self, order): ...

def pay(order, method: Payment):
    method.charge(order)
```

判断标准：**新增一种情况时，你动了几个已有文件？** 动 0 个、新增 1 个，符合 OCP；往 if-else 里再加一个分支，违反 OCP。

**关键提醒**：OCP 不是"永远不改代码"——修 bug 当然要改。它约束的是**可预期的扩展方向**：预测不到的方向，先留简单实现，等第一种真实变化出现时再抽象（正是 01 篇 YAGNI 的平衡点）。

## L — 里氏替换原则（Liskov Substitution Principle）

> **所有引用父类的地方，必须能透明地替换为子类对象，而程序行为不变。**（Barbara Liskov 1987 年提出，1994 年与 Jeannette Wing 形式化）

这是继承的正确性标准。经典反例——"正方形不应继承长方形"：

```python
class Rectangle:
    def set_width(self, w): self.w = w
    def set_height(self, h): self.h = h

class Square(Rectangle):
    def set_width(self, w): self.w = self.h = w  # 保持正方形约束
    def set_height(self, h): self.w = self.h = h

def stretch(rect: Rectangle):
    rect.set_width(2)
    rect.set_height(3)
    assert rect.w * rect.h == 6   # 对 Square 失败！
```

`Square` 在数学上"是一个" `Rectangle`，但在**程序行为**上不能替换父类——它悄悄强化了约束（改宽会连带改高）。违反 LSP 的继承是**假继承**，应改用组合。

实践守则：

1. 子类**不得强化前置条件**（父类允许的输入子类也必须接受）；
2. 子类**不得弱化后置条件**（父类保证的行为子类必须至少保证）;
3. 父类不变式子类必须维持。

## I — 接口隔离原则（Interface Segregation Principle）

> **客户端不应被迫依赖它用不到的方法。**

```python
# 反例：胖接口
class Worker(ABC):
    def work(self): ...
    def eat(self): ...          # 机器人被迫实现"吃饭"

class Robot(Worker):
    def eat(self): pass         # ← 空实现 = 依赖了用不到的东西

# 正例：按客户需求拆小接口
class Workable(ABC):
    def work(self): ...
class Eatable(ABC):
    def eat(self): ...
```

胖接口的伤害：实现者被迫写空方法（谎言代码）；调用者面对一列无关方法，认知负担大。判断标准：**看实现类有没有被迫空实现**——有就说明接口该拆了。

## D — 依赖倒置原则（Dependency Inversion Principle）

> **高层模块不应依赖低层模块，两者都应依赖抽象。**（Martin, 1996）

"倒置"指依赖**箭头方向**的倒置：

```
传统: 业务逻辑 ──依赖──→ MySQL 存储          （高层绑死低层）
倒置: 业务逻辑 ──依赖──→ Repository 接口 ←── MySQL 实现
```

```python
class OrderRepository(ABC):          # 抽象由高层定义
    @abstractmethod
    def save(self, order): ...

class MySQLRepository(OrderRepository): ...   # 低层实现抽象
class InMemoryRepository(OrderRepository): ... # 测试用假实现

class OrderService:                  # 高层只认抽象
    def __init__(self, repo: OrderRepository):
        self.repo = repo
```

两个常被混淆的点：

1. DIP ≠ 依赖注入（DI）。DIP 是**设计原则**（谁依赖谁），DI 是实现该原则的**技术手段**（把依赖从外部注入构造函数）。本例的构造函数注入就是最朴素的 DI；
2. DIP 是**测试性的地基**：正因为 `OrderService` 只依赖抽象，才能注入 `InMemoryRepository` 做单元测试，不需要真数据库。

## 五原则的内在关系

它们不是并列的五条，而是有主线的：

```
SRP: 职责要单一（变化的源头要隔离）
OCP: 靠什么做到"加不改" → 靠抽象 + 多态
LSP: 抽象能成立的前提 → 子类可替换
ISP: 抽象要合适 → 小而专，不胖
DIP: 这一切的结构性表达 → 高层低层都依赖抽象
```

一句话总纲：**用小而正确的抽象，隔离变化，让依赖指向稳定**。

## 自测

1. SRP 的"职责"指什么？判断标准是什么？
2. OCP 与 YAGNI 如何平衡？
3. Square/Rectangle 违反 LSP 的具体表现？修复方向？
4. DIP 和 DI 的区别？

## 下一篇

[03 - 创建型模式](03-创建型模式.md)——对象创建的解耦：单例、工厂、建造者、原型。

## 参考来源

- Robert C. Martin, *Design Principles and Design Patterns*（2000，SOLID 各原则的原始阐述）: https://web.archive.org/web/20150906155800/http://www.objectmentor.com/resources/articles/Principles_and_Patterns.pdf
- Robert C. Martin, *Agile Software Development: Principles, Patterns, and Practices*, Prentice Hall, 2002
- Bertrand Meyer, *Object-Oriented Software Construction*, 1988（OCP 出处）
- Barbara Liskov & Jeannette Wing, *A Behavioral Notion of Subtyping*, ACM TOPLAS, 1994（LSP 形式化）
