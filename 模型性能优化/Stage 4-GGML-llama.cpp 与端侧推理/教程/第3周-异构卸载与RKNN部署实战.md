# 第 3 周教程：异构卸载与 RKNN 部署实战——把模型送上你的硬件

> **本周要回答的问题**
> 1. llama.cpp 的 `-ngl` 层卸载、RPC 后端、`ggml-backend` 多后端抽象各解决什么？
> 2. 没有 NVIDIA GPU 的机器上，llama.cpp 还能怎么加速？（Vulkan / RPC）
> 3. RKNN-Toolkit2 的「ONNX → RKNN → 板端」全流程是什么？量化掉点怎么逐层定位？
> 4. RK3588 三核绑定与零拷贝内存为什么重要？

对应学习计划：第 3-4 周。交付物：① `llama-server` 并发压测；② 跑通一次跨后端推理（Vulkan/RPC）；③ RK3588 上完成语音模型（DeepFilterNet 或 VAD）的 ONNX→RKNN→板端三连 + 逐层精度报告；④ `rknn-llm` 跑通 1-3B LLM。

---

## 1. 第一性原理：模型跑不动时，把层"搬家"

### 1.1 llama.cpp 的异构三件套

- **`-ngl N`（层卸载）**：把前 $N$ 层（或全部 `-ngl 99`）放 GPU，其余留 CPU。分界点在残差流上切一刀——GPU 吃算力密集的大头，CPU 兜底放不下的尾巴；
- **RPC 后端**：把若干层拆到**远程机器**执行——局域网拼出"分布式大显存"。协议层就是逐层的激活张量搬运，瓶颈变成网络带宽（故适合层间激活小的位置切分）；
- **`ggml-backend` 抽象**：统一的"后端接口 + 调度器"，每个算子图节点按设备能力分派（CPU / CUDA / Vulkan / Metal / **CANN**）。新增硬件 = 实现一个 backend——**你的 310P 联动就靠 CANN 后端，Stage 6 深挖，本周先读接口**。

### 1.2 llama-server：端侧服务化的最小闭环

OpenAI 兼容 API + continuous batching：多请求共享一次前向（decode 批量化），把 CPU/GPU 的带宽利用率从单请求的谷底拉起来。端侧"服务化"从这里起步，云端版本（Stage 7）换 vLLM。

### 1.3 RKNN 部署的根本逻辑

NPU 的算子集有限、只吃量化模型，所以部署链 = **框架模型 → 中间格式（ONNX）→ 厂商工具链转换+量化 → 板端运行时**：

$$
\text{PyTorch} \xrightarrow{\text{export}} \text{ONNX} \xrightarrow[\text{量化(可选)}]{\text{RKNN-Toolkit2}} \text{.rknn} \xrightarrow{\text{RKNN Runtime}} \text{NPU}
$$

三个必知概念：

1. **量化选项**：非对称量化 + 校准数据集；敏感层可指定回退高精度（混合精度）；
2. **逐层余弦对比**：工具链提供"每层输出与浮点参考的余弦相似度"——**量化掉点定位的标准手段**（本周交付的精度报告核心）；
3. **零拷贝**：`rknn_create_mem` 分配 NPU 可直接访问的物理连续内存，输入输出不再经过一次 `memcpy`——推理延迟的隐形杀手就是这些多余拷贝。

---

## 2. 实现与验证（本周交付核心）

### 2.1 llama-server 压测（x86 机器）

```bash
# 起服务（CPU 推理；有 Vulkan 设备加 --split-mode 相关参数）
./build/bin/llama-server -m qwen2.5-1.5b-Q4_K_M.gguf \
    --host 0.0.0.0 --port 8080 -c 4096 -t 8

# 单请求基准
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"介绍 RKNN"}],"max_tokens":256}'
```

```python
# bench_server.py —— 并发 1/4/8 压测（交付表格的数据来源）
import time, requests, concurrent.futures as cf

URL = "http://localhost:8080/v1/chat/completions"
def one(i):
    t0 = time.time()
    r = requests.post(URL, json={
        "model": "qwen",
        "messages": [{"role": "user",
                      "content": f"用一句话介绍第 {i} 种排序算法"}],
        "max_tokens": 128}, timeout=600)
    d = time.time() - t0
    n = r.json()["usage"]["completion_tokens"]
    return d, n, (time.time() - t0 - d)   # 总时长、输出 token 数、(首 token 需流式)

for conc in (1, 4, 8):
    t0 = time.time()
    with cf.ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(one, range(conc)))
    wall = time.time() - t0
    tps = sum(n for _, n, _ in res) / wall
    print(f"并发 {conc}: 墙钟 {wall:.1f}s, 聚合 {tps:.1f} tokens/s, "
          f"单请求均值 {sum(d for d,_,_ in res)/conc:.1f}s")
# 预期：聚合吞吐随并发上升（连续批处理），单请求延迟缓慢劣化
```

### 2.2 跨后端跑通一次（无 NVIDIA GPU 的路径）

```bash
# 方案 A：Vulkan（有核显/AMD 卡即可）——编译时打开
cmake -B build-vulkan -DGGML_VULKAN=ON && cmake --build build-vulkan -j
./build-vulkan/bin/llama-cli -m qwen2.5-1.5b-Q4_K_M.gguf -ngl 99 -p "你好" -n 32

# 方案 B：RPC——两台局域网机器拼内存
# 远端：./build/bin/rpc-server -p 50052
# 本机：./build/bin/llama-cli -m <大模型>.gguf --rpc 远端IP:50052 -ngl 99
# 记录：切分位置、网络带宽、tokens/s——验证"层间激活搬运"的开销模型
```

### 2.3 RKNN 三连：语音模型上 RK3588（本周重头戏）

```python
# onnx2rknn.py —— 在装有 rknn-toolkit2 的 x86 主机上运行（注意官方版本矩阵）
from rknn.api import RKNN

ONNX = "deepfilternet3.onnx"      # 或你的 VAD 模型
RKNN_MODEL = "deepfilternet3.rknn"
DATASET = "calib_list.txt"        # 每行一个校准输入文件路径（16 条即可）

rknn = RKNN(verbose=True)
rknn.config(mean_values=[[0]], std_values=[[1]], target_platform="RK3588",
            quantized_algorithm="normal")          # 敏感任务可试 "mmse"
rknn.load_onnx(model=ONNX)
rknn.build(do_quantization=True, dataset=DATASET)  # INT8 量化
rknn.export_rknn(RKNN_MODEL)

# ---- 逐层精度对比（交付核心）----
rknn.init_runtime(target=None)                     # 模拟器
acc = rknn.accuracy_analysis(inputs=["calib_0.npy"], output_dir="./acc_out",
                             target=None)          # 输出每层余弦相似度
rknn.release()
```

```bash
# 板端推理（RK3588）：用官方 rknn_runtime + C API demo 或 python
# 三核绑定与零拷贝是 C API 侧的开关：
#   核绑定：rknn.init_runtime(core_mask=RKNN_NPU_CORE_0_1_2)   ← 三核并行
#   零拷贝：rknn_create_mem() 分配 DMA 内存 → rknn_set_io_mem()
```

逐层精度报告模板（交付）：

| 层 | 余弦相似度 | 判定 |
| --- | --- | --- |
| conv/encoder 各层 | _实测（预期 > 0.99）_ | 正常 |
| 量化敏感层（通常：首个大动态范围层、Sigmoid/Softmax 邻近层） | _实测_ | < 0.95 → 候选回退 |

**判定规则**：余弦 < 0.95 的层列入"回退清单"，用混合精度（该层保高精度）重转，直到端到端指标（如降噪 PESQ / VAD F1）达标——这是"NPU 部署精度崩了怎么排查"的标准答案。

### 2.4 rknn-llm：板上跑 1-3B LLM

```bash
git clone https://github.com/airockchip/rknn-llm && cd rknn-llm
# 按 README 转换（支持 Qwen/Phi/Gemma 等 1-3B，INT4/INT8 权重）
# 记录三件事：
#   ① tokens/s（prefill 与 decode 分开记）
#   ② 内存占用（RK3588 通常 8/16G，预算表必须含系统开销）
#   ③ 生成质量抽查（与第 2 周 CPU 版对比同一提示词）
```

---

## 3. 工程权衡与失效模式

### 3.1 权衡

- **`-ngl` 的切点**：全卸载最快，但显存不足时要在"层数换显存"上找平衡；切在中间层时，层间激活的 CPU↔GPU 搬运是隐形成本。
- **RPC 的适用面**：网络 < 10 Gbps 时只适合切"激活小"的位置（层数少、序列短）；它是"拼内存"方案，不是"提速"方案。
- **NPU 量化的激进程度**：INT8 稳妥、INT4 省内存但敏感层概率大增——端侧语音模型（降噪/ASR）对数值误差敏感，先保守再逐步激进。

### 3.2 失效模式

1. **ONNX 导出算子不被 RKNN 支持**：症状——转换报错列出算子名；根因——自定义算或新版本算子；修复——改写为等价结构，或走第 4 周的自定义算子路线。
2. **逐层余弦都高但端到端崩**：症状——单层 > 0.99、任务失败；根因——误差在串联中复合（与量化评测同款）；修复——以端到端指标为准，对复合敏感的层提前回退。
3. **零拷贝没做对**：症状——板端推理延迟是理论值的 2-3 倍；根因——输入输出仍走 `memcpy`（每次推理两趟大块拷贝）；修复——`rknn_create_mem` + `rknn_set_io_mem` 全链路零拷贝。
4. **三核绑定反而变慢**：症状——开三核比单核慢；根因——模型太小，核间同步/拆分开销超过并行收益；修复——按模型规模选核（大模型三核、小模型单核）。

---

## 4. 延伸思考题（含解析）

**Q1**：`ggml-backend` 的抽象对厂商（如华为）意味着什么？为什么这对你重要？
**A**：意味着接入 llama.cpp 生态只需实现一组后端接口，整个模型层与应用层免费复用——CANN 后端就是这么进社区的。对你的意义：310P/Atlas 上跑 LLM 的路径已经被铺好（Stage 6 深挖），且你本周读接口时积累的认知届时直接复用。

**Q2**：RPC 后端和真正的分布式推理（张量并行）区别在哪？
**A**：RPC 是**层间流水**（不同机器算不同层，串行传递激活），简单但受网络延迟约束；张量并行是**层内切分**（同一层的矩阵切开多卡同时算，每层都要集合通信），需要高带宽互联但并行度高。前者是穷人拼内存，后者是富人堆算力。

**Q3**：为什么量化敏感层常常是"第一个大动态范围层"和"非线性激活邻近层"？
**A**：前者：输入动态范围大 → 量化网格被撑大 → 相对误差大，且它是误差进入网络的第一站（无前置平滑）；后者：Sigmoid/Softmax 在饱和区导数近零、在跳变区又极陡，量化抖动被非线性放大。逐层对比就是要把这两类层找出来。

**Q4**：RK3588 跑 1.5B LLM，为什么 NPU（rknn-llm）通常比 CPU（llama.cpp）快但灵活性差？
**A**：NPU 有专用 INT 乘累加阵列与更高有效带宽，decode GEMV 直接受益；但算子集固定（新结构要等工具链、或自定义算子）、上下文长度与采样逻辑受运行时限制。灵活与高效的经典互换——选型报告（第 5-6 周）就是量化这次互换。

**Q5**：你的语音结课项目里，哪些组件适合本周的 RKNN 路线、哪些适合 llama.cpp CPU？
**A**：VAD/降噪/小型 ASR：固定输入形状、纯数值前向 → RKNN NPU（实时、低功耗）；对话 LLM：变长序列、采样逻辑复杂 → llama.cpp CPU 或 rknn-llm（按第 5-6 周横评定）。异构分工正是端侧助手的标准架构。

---

## 5. RKNN 部署排错手册（板端实战沉淀）

### 5.1 转换期错误速查

| 错误症状 | 常见根因 | 处置 |
| --- | --- | --- |
| 算子不支持列表 | 自定义结构/新算子 | 等价改写，或走自定义算子（第 4 周） |
| 量化后模拟器结果全错 | 校准数据格式不符（如未归一化） | 核对 `mean_values/std_values` 与训练时预处理一致 |
| 动态形状报错 | 输入尺寸未固定 | 固定输入（语音帧数/频谱维度），或按官方 `dynamic_shape` 示例配置 |
| 转换极慢 | 模型含大量小算子 | 先在 ONNX 侧用 `onnx-simplifier` 折叠常量与冗余节点 |

### 5.2 板端运行期排错流程

```
推理结果异常
├─ 模拟器上正常？ → 是：板端 runtime 版本与工具链不匹配 → 对齐版本
├─ 模拟器也异常？
│    ├─ 量化前（FP 模式）正常？ → 量化掉点 → 逐层余弦对比定位
│    └─ FP 模式也异常？ → 算子实现差异或前后处理错误 → 逐层输出对比
└─ 偶发异常？ → 内存对齐/多线程竞争 → 检查零拷贝缓冲生命周期
```

### 5.3 语音模型转换的三个专属注意点

1. **帧级输入的时间维**：降噪/ASR 模型常有"帧循环"结构，导出时把时间步固定（如 1 帧或 chunk），板端自己维护循环——别让时间维进图（动态形状是 NPU 的天敌）。
2. **复数与 STFT**：若模型含 STFT/iSTFT，优先把它移出图外（CPU 上做或预计算），NPU 对复数运算支持极差。
3. **状态张量（RNN/GRU 类）**：hidden state 要作为显式输入/输出导出，板端循环喂回——这类模型正是"自定义算子 + 混合执行"的高发区。

---

## 本周交付清单

- [ ] `llama-server` 压测表：并发 1/4/8 的聚合吞吐与单请求延迟。
- [ ] 跑通一次跨后端推理（Vulkan 或 RPC），记录配置与数字。
- [ ] RK3588 完成语音模型三连（ONNX→RKNN→板端），输出逐层余弦报告并标注回退清单。
- [ ] `rknn-llm` 跑通 1-3B，记录 tokens/s 与内存。
- [ ] 读懂 `ggml-backend` 接口头文件一遍，记录 5 个核心函数（Stage 6 预习）。
