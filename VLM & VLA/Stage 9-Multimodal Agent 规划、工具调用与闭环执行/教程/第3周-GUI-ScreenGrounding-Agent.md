# 第 3 周教程：视觉 GUI / Web / OS 操作控制 Agent（Screen Grounding Agent）

> **本周要回答的三个问题**
> 1. 截图到动作（Click/Type/Scroll）的映射管线怎么搭？
> 2. Set-of-Mark（SoM）为什么能把"连续坐标回归"变成"离散选择题"？它的失效面是什么？
> 3. Playwright 自动化环境怎么与 Agent Loop 安全地接起来？

对应学习计划：第 3 周。交付物：Playwright + SoM/Grounding 的 Web GUI Agent——自然语言指令驱动，实时截图解析、生成点击/输入操作，连续完成 ≥5 步网页操作。

---

## 1. 第一性原理：GUI 操作是"感知粒度"与"动作精度"的双重考验

### 1.1 根本矛盾：VLM 的坐标精度 vs 交互元素的像素要求

点一个按钮需要 **±5 像素级**的定位精度，而 VLM（Stage 1/3 的结论）对绝对坐标的预测能力薄弱：视觉 Token 的粒度（Stage 1 的 patch 网格）、空间编码的先天薄弱、训练数据里坐标约定混乱，都让"直接预测 (x, y)"成为 Agent 成功率的最大短板。

两条技术路线直击这个矛盾：

| 路线 | 思路 | 代表 | 精度来源 |
| --- | --- | --- | --- |
| **专用 Grounding 模型** | 用大规模"截图-元素框"数据专门训练 GUI 元素定位 | SeeClick（*SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents*, ACL 2024）、OmniParser（微软，截图→结构化 UI 元素解析器） | 专项训练的定位头 |
| **SoM 提示工程** | 在截图上叠加编号标记，让模型"选编号"而非"算坐标" | Set-of-Mark（SoM prompting） | **把回归问题变成分类问题** |

两者常组合使用：OmniParser 解析出所有可交互元素 → SoM 叠加编号 → VLM 选编号 → 程序取编号对应元素的精确中心坐标执行点击。

### 1.2 SoM 的第一性原理：回归 → 选择的难度坍缩

**为什么"选编号"远易于"算坐标"**：

1. **输出空间的性质**：坐标回归要求模型从连续空间精确输出数值（且要求与视觉 Token 网格对齐——Stage 1 的 14px patch 粒度决定了原生精度极限）；而选编号是一个**离散分类**，模型只需在语义层面回答"哪个标记是目标"——这正是 LLM/VLM 最擅长的"指认"能力。
2. **感知辅助**：编号标记本身就是视觉锚点——模型不需要在密集 UI 里"内部定位"，标记替它完成了"关注点显式化"。这与 Stage 3 的发现同源：模型对"图中有什么"尚可，对"哪里"很弱——SoM 把"哪里"外包给了程序画上去的标记。
3. **精度兜底**：编号对应的可点击区域由程序（OmniParser/浏览器 DOM）给出**精确几何坐标**——点击精度与 VLM 无关，100% 对齐。

**SoM 的失效面**（同样重要）：标记过密时编号拥挤/遮挡 UI（标记本身干扰感知）；元素语义不清时编号选择仍靠模型（SoM 只解决定位不解决语义理解——"哪个是搜索按钮"仍要模型判断）；动态页面标记过期（标记基于旧截图，DOM 已变——必须"截图→标记→决策→执行"原子化，禁止跨轮复用标记）。

### 1.3 GUI Agent 的动作空间与安全边界

动作空间（从 Web 到 OS 递增危险等级）：

| 动作 | Web（Playwright） | 桌面（PyAutoGUI） | 危险等级 |
| --- | --- | --- | --- |
| Click (x,y) | `page.click` / 元素引用 | 坐标点击 | 中（点错按钮可能不可逆） |
| Type | `page.fill` | 键盘模拟 | 中（输错文本可撤回，但可能触发提交） |
| Scroll | `page.scroll` | 滚轮模拟 | 低 |
| **提交类**（下单/删除/发送） | 表单提交 | 回车/按钮 | **高——必须有确认闸** |

安全边界三原则：**白名单域**（Agent 只在指定网站活动）、**破坏性动作人工确认**（下单/删除/支付类动作必须停下来等人批准）、**全程审计日志**（每个动作的截图/参数/结果可回放——Stage 4 轨迹审计的地基）。

---

## 2. 实现与验证

### 2.1 Playwright + SoM 的 Web Agent

```python
"""
Web GUI Agent: 截图 -> SoM 标记 -> VLM 选编号 -> Playwright 执行, 含确认闸。
运行方式: python stage9_week3_gui_agent.py --task "在搜索框输入 laptop 并搜索"
依赖: playwright (pip install playwright && playwright install chromium), openai
"""
import base64
import json
import re
from playwright.sync_api import sync_playwright


# ---------- SoM 标记: 截图 + 可交互元素 -> 编号标注图 ----------
def som_annotate(screenshot_png: bytes, elements: list[dict]):
    """elements: [{'id': 1, 'center': (x, y), 'bbox': [...], 'desc': 'button'}]
       真实实现: OmniParser 或 DOM 提取 (Playwright 可直接拿元素几何!)"""
    from PIL import Image, ImageDraw
    import io
    img = Image.open(io.BytesIO(screenshot_png)).convert("RGB")
    d = ImageDraw.Draw(img)
    for e in elements:
        x1, y1, x2, y2 = e["bbox"]
        d.rectangle([x1, y1, x2, y2], outline="red", width=3)
        d.rectangle([x1 - 2, y1 - 18, x1 + 26, y1 + 2], fill="red")
        d.text((x1 + 4, y1 - 15), str(e["id"]), fill="white")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def interactive_elements(page) -> list[dict]:
    """从 DOM 提取可交互元素 (比视觉解析更精确; OmniParser 用于无 DOM 场景)"""
    els = []
    for i, h in enumerate(page.query_selector_all(
            "button, a, input, [role=button], select"), 1):
        try:
            bb = h.bounding_box()
            if bb and bb["width"] > 5 and bb["height"] > 5:
                els.append({"id": i, "bbox": [bb["x"], bb["y"],
                            bb["x"] + bb["width"], bb["y"] + bb["height"]],
                            "center": (bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2),
                            "desc": (h.text_content() or h.get_attribute("aria-label")
                                     or h.evaluate("e=>e.tagName"))[:40].strip()})
        except Exception:
            continue
    return els


# ---------- Agent 决策 (VLM 选编号) ----------
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["click", "type", "scroll", "done", "fail"]},
        "element_id": {"type": "integer"},
        "text": {"type": "string"},
        "note": {"type": "string"},
    },
    "required": ["action"],
}


def vlm_decide(screenshot_annotated: bytes, elements: list[dict], task: str, history) -> dict:
    """在线=多模态 VLM; 离线=规则桩 (按 desc 关键词匹配演示决策能力)"""
    menu = "\n".join(f"{e['id']}: [{e['desc']}]" for e in elements)
    # 真实调用: 把 som_annotate 的图 + menu + task 发给 VLM, 约束解码出 DECISION_SCHEMA 的 JSON
    # 离线规则桩: 模拟"选编号"决策
    low = task.lower()
    for e in elements:
        dl = e["desc"].lower()
        if "search" in dl and ("搜索" in low or "search" in low):
            return {"action": "click", "element_id": e["id"]}
    for e in elements:
        if elements and e["desc"] == "" or "textbox" in e["desc"].lower() or "input" in e["desc"].lower():
            if "输入" in low or "type" in low:
                return {"action": "type", "element_id": e["id"], "text": "laptop"}
    if history and len(history) >= 2:
        return {"action": "done", "note": "任务步骤已完成(桩)"}
    return {"action": "fail", "note": "桩未找到匹配元素"}


def run_gui_agent(task: str, url: str, max_steps: int = 8, headless: bool = True):
    DANGEROUS = ("pay", "checkout-confirm", "delete", "order")   # 确认闸关键词
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(url)
        history = []
        for step in range(max_steps):
            els = interactive_elements(page)
            shot = page.screenshot()
            annotated = som_annotate(shot, els)
            decision = vlm_decide(annotated, els, task, history)
            act = decision.get("action")
            history.append({"step": step, "decision": decision, "n_elements": len(els)})
            if act == "done":
                browser.close()
                return history, "DONE"
            if act == "fail":
                browser.close()
                return history, f"FAILED: {decision.get('note')}"
            el = next((e for e in els if e["id"] == decision.get("element_id")), None)
            if el is None:
                history[-1]["error"] = "element_id 不存在 (标记过期?)"
                continue
            # 确认闸: 危险动作需人工确认 (自动化验收时用 --auto-confirm)
            if any(k in (el["desc"].lower()) for k in DANGEROUS):
                if input(f"⚠️ 即将执行可能不可逆操作 [{el['desc']}]，确认? (y/N) ") != "y":
                    history[-1]["error"] = "user_vetoed"; continue
            if act == "click":
                page.mouse.click(*el["center"]); page.wait_for_load_state("networkidle")
            elif act == "type":
                page.mouse.click(*el["center"])
                page.keyboard.type(decision.get("text", ""), delay=30)
            elif act == "scroll":
                page.mouse.wheel(0, 400)
        browser.close()
        return history, "FAILED: max_steps exceeded"


def _offline_check():
    """不启动浏览器的逻辑自检: SoM 标记与决策映射"""
    els = [{"id": 1, "bbox": [10, 10, 60, 40], "center": (35, 25), "desc": "Search"},
           {"id": 2, "bbox": [80, 10, 140, 40], "center": (110, 25), "desc": "textbox"}]
    shot = som_annotate(b"" or __import__("io").BytesIO(
        __import__("PIL.Image", fromlist=["Image"]).new("RGB", (200, 60), "white").tobytes()
    ) if False else _png_stub(), els)
    assert len(shot) > 0, "SoM 标注图生成失败"
    d = vlm_decide(shot, els, "在搜索框输入 laptop 并搜索", history=[])
    assert d["action"] in ("click", "type"), f"决策动作异常: {d}"
    print(f"✅ SoM 标注 + 决策桩自检通过; 首个决策: {d}")


def _png_stub():
    import io
    from PIL import Image
    b = io.BytesIO(); Image.new("RGB", (200, 60), "white").save(b, "PNG")
    return b.getvalue()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="在搜索框输入 laptop 并搜索")
    ap.add_argument("--url", default="https://demo-playwright.example")   # 换成你的测试站
    ap.add_argument("--headless", type=bool, default=True)
    args = ap.parse_args()
    _offline_check()
    history, status = run_gui_agent(args.task, args.url, headless=args.headless)
    for h in history:
        print(h["step"], h["decision"], h.get("error", ""))
    print("状态:", status)
```

**验收口径**（学习计划：连续 ≥5 步网页操作）：跑在真实测试站（如 playwright 官方 demo 站或自建电商 mock）上，以"搜索→点击商品→查看评价→排序→加入购物车"的 5 步链路为验收任务；每步的 SoM 标注图与决策 JSON 留档（这就是可审计轨迹）。`_offline_check` 保证"截图标注→决策映射"的核心逻辑在无浏览器环境也可自检。

---

## 3. 工程权衡与失效模式

### 3.1 决策表：元素获取的三条路线

| 路线 | 精度 | 通用性 | 成本 | 适用 |
| --- | --- | --- | --- | --- |
| DOM 提取（Playwright） | **精确（几何+语义）** | 仅 Web | 低 | Web Agent 首选 |
| OmniParser（视觉解析） | 高 | 任意截图（含 OS/远程桌面） | 需额外模型推理 | 无 DOM 场景（桌面/游戏/VNC） |
| 纯 VLM 坐标预测 | 低（Stage 1/3 结论） | 最通用 | 无 | 仅作兜底/非交互定位 |

**经验规则**：有 DOM 用 DOM，无 DOM 用 OmniParser 级解析器，SoM 做人机/模型接口，裸坐标预测是最后手段——**精度从结构里来，不从模型里硬抠**。

### 3.2 三个代表性失效模式

**失效 1：标记过期（Stale Annotation）——点错位置**
- **症状**：点击执行后落点与预期元素无关；轨迹里出现"element_id 不存在"。
- **根因**：页面在标记生成与执行之间发生了变化（动画/异步加载/广告轮播）——SoM 的编号与当前 DOM 已错位。
- **定位**：执行前重截屏 diff（像素差或元素集 diff）。
- **修复**："截图→标注→决策→执行"严格原子化，执行前校验元素几何未变（重查 bounding_box 一致性）；不稳定页面在动作间加 `wait_for_load_state`。

**失效 2：SoM 标记遮挡与编号爆炸**
- **症状**：密集 UI（后台管理系统）上模型选错编号，或干脆"看不见"某些编号。
- **根因**：可交互元素过多（>40 个）时编号重叠、字体过小；标记遮挡了目标元素本身的视觉特征。
- **定位**：人眼看标注图——你自己认不清编号，模型更不行。
- **修复**：元素过滤（只标可交互且可见的）；分区域标注（先选区域再局部标注——层次化 SoM）；编号字体/位置优化（外部角标而非覆盖中心）。

**失效 3：破坏性动作的自动化事故**
- **症状**：验收跑得很顺，直到 Agent 在真实站点上点了"确认支付"/清空了测试数据。
- **根因**：缺少危险动作确认闸与白名单域；验收环境与生产环境未隔离。
- **定位**：审计日志里的事故动作回放。
- **修复**：三原则落地（白名单/确认闸/审计）；测试账号 + 测试环境隔离；对"提交类"DOM 事件做静态拦截（Playwright route 拦截支付接口）。

---

## 4. 延伸思考题

1. **SoM 的信息论视角**：把"选编号"与"预测坐标"作为两个通信信道比较——前者传递 $\log_2 N$ bit（N 个元素），后者要求连续数值精确到像素。从"信道容量与任务需求匹配"的角度，论述为什么 UI 元素语义分割（OmniParser 类）+ 离散选择是更优的编码方式，以及它牺牲了什么（自由落点、画布类应用）。
2. **Grounding 的跨域泛化**：SeeClick 类模型在 Web 上训练，到 OS 桌面（图标/菜单栏风格完全不同）泛化会掉多少？设计一个跨域评测协议（Web/OS/移动端三域同任务族），并预测掉分模式。（提示：布局先验与元素形态先验是域特定的——这正是 OmniParser 走"通用 UI 解析"路线的动机。）
3. **动手实验**：给 2.1 的 Agent 加上"第 2 周的 Reflection"：点击后用 OCR/元素文本校验"预期效果是否发生"（如点击搜索按钮后页面出现结果列表），未发生则进入反思重试——把三周的积木拼成第一个完整的自纠错 GUI Agent。

---

*下一篇：[第 4 周：Agent 轨迹合成、Sanity Check 与清洗](第4周-轨迹合成与清洗.md)*
