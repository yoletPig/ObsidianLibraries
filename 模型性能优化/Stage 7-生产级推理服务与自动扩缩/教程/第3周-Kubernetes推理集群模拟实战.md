# 第 3 周教程：Kubernetes 推理集群模拟实战

> **本周要回答的三个问题**
> 1. 用 Kind 在单台主机上能模拟出什么、模拟不出什么？边界在哪里？
> 2. 推理负载在 K8s 里怎么建模（Deployment、GPU 资源、滚动更新）？
> 3. 为什么推理服务的自动扩缩更适合用"队列长度"驱动（KEDA）而不是 CPU 利用率（HPA）？

对应学习计划：第 3 周。交付物：单台云主机用 Kind 搭 1 master + 2 worker 模拟集群，部署一个推理 InferenceService（小模型代替 7B），配 HPA 与基于队列的 KEDA 扩缩，用压测流量触发扩缩容并录屏展示副本变化。

> ⚠️ **模拟环境声明**：本教程全部在单主机 Kind 上完成。能验证：K8s 资源建模、调度、扩缩控制回路、滚动更新逻辑。**不能验证**：真实多机的网络延迟、跨节点故障、真实 GPU 调度性能。压测数据仅作控制回路演练，不代表生产容量。

---

## 1. 第一性原理：为什么推理负载要交给编排系统

### 1.1 编排系统解决的四件事

手工管理 N 个推理实例要处理：进程拉起/重启、资源分配、流量接入、版本更新、故障恢复。K8s 把这些标准化：

| 需求 | K8s 抽象 |
| --- | --- |
| 跑 N 个副本 | Deployment（`replicas: N`） |
| 声明算力需求 | resources（CPU/GPU/memory） |
| 接入流量 | Service / Ingress |
| 版本更新 | 滚动更新（RollingUpdate） |
| 故障自愈 | 控制器持续对齐期望状态 |
| 自动扩缩 | HPA / KEDA |

### 1.2 为什么用 Kind 模拟

真实多机集群成本高、搭建重。**Kind（Kubernetes IN Docker）**用 Docker 容器模拟节点，单机就能拉起一个完整控制平面 + 多"节点"，足以演练**资源建模、调度、扩缩控制回路**这些与机器数量无关的逻辑。这正是你约束下"把部署架构写出来"的最优载体。

---

## 2. Kind 集群搭建（交付核心一）

### 2.1 配置 1 master + 2 worker

```yaml
# kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1beta1
nodes:
- role: control-plane
- role: worker
- role: worker
```

```bash
# 安装 kind 与 kubectl（若未装）
# 创建集群
kind create cluster --name infer-cluster --config kind-config.yaml

# 验证：应看到 1 control-plane + 2 worker
kubectl get nodes
```

**预期输出**：

```
NAME                          STATUS   ROLES           AGE
infer-cluster-control-plane   Ready    control-plane   30s
infer-cluster-worker          Ready    <none>          20s
infer-cluster-worker2         Ready    <none>          20s
```

### 2.2 模拟环境的局限说明

Kind 的"节点"是同机的 Docker 容器：
- **没有真实隔离**：三"节点"共享同一台主机的 CPU/内存；
- **没有真实网络**：跨"节点"通信走本机网络栈；
- **没有真实 GPU**（默认）：用 CPU 跑小模型模拟推理负载。

所以本教程用**轻量模型**（如一个小分类器或 echo 服务）代替 7B，重点演练**编排逻辑**而非推理性能。

---

## 3. 推理负载建模（交付核心二）

### 3.1 Deployment（小模型推理服务）

```yaml
# inference-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: infer-service
spec:
  replicas: 2
  selector:
    matchLabels: {app: infer-service}
  template:
    metadata:
      labels: {app: infer-service}
    spec:
      containers:
      - name: infer
        image: your-registry/tiny-infer:latest    # 小模型推理镜像
        ports:
        - containerPort: 8080
        resources:
          requests: {cpu: "200m", memory: "256Mi"}
          limits:   {cpu: "500m", memory: "512Mi"}
        readinessProbe:            # 就绪探针：模型加载完才接流量
          httpGet: {path: /health, port: 8080}
          initialDelaySeconds: 5
```

**要点**：
- `readinessProbe`：模型加载慢，必须等就绪才接流量，否则请求打到未加载完的副本；
- `resources`：声明请求与上限，调度器据此分配；
- 真实场景加 `nvidia.com/gpu` 资源（需 device plugin，模拟环境省略）。

### 3.2 Service 暴露

```yaml
# inference-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: infer-service
spec:
  selector: {app: infer-service}
  ports:
  - port: 80
    targetPort: 8080
```

```bash
kubectl apply -f inference-deployment.yaml
kubectl apply -f inference-service.yaml
kubectl get pods -w          # 观察副本起来
```

---

## 4. 自动扩缩：HPA vs KEDA（交付核心三）

### 4.1 HPA：CPU 驱动

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: infer-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: infer-service
  minReplicas: 2
  maxReplicas: 6
  metrics:
  - type: Resource
    resource:
      name: cpu
      target: {type: Utilization, averageUtilization: 60}
```

### 4.2 为什么推理更适合队列驱动（KEDA）

HPA 用 CPU 利用率扩缩，但推理负载的真实压力信号是**排队请求数**：
- GPU 推理时 CPU 未必高（算力在 GPU），CPU 指标失真；
- 排队长度直接反映"处理不过来"，是更准的扩容信号。

**KEDA** 支持按自定义指标（如消息队列长度、请求队列）扩缩：

```yaml
# keda-scaledobject.yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: infer-keda
spec:
  scaleTargetRef:
    name: infer-service
  minReplicaCount: 2
  maxReplicaCount: 8
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: request_queue_length
      query: sum(infer_pending_requests)
      threshold: "10"        # 队列每满 10 个请求加一个副本
```

（KEDA 需先 `helm install` 部署到集群。）

### 4.3 触发扩缩演练

```bash
# 装 metrics-server（HPA 依赖）
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# 压测打流量，观察副本变化
kubectl autoscale ... # 或已用 HPA yaml
# 另开终端压测
while true; do curl -s http://localhost:PORT/infer -d '{"x":1}'; done

# 观察
kubectl get hpa -w            # HPA 当前副本/目标
kubectl get pods -w           # 副本增减
```

**交付**：录屏展示"压力上升 → 副本从 2 涨到 N → 压力下降 → 副本缩回"的完整闭环。这是扩缩控制回路成立的直接证据。

---

## 5. 工程权衡与失效模式

### 5.1 权衡

- **HPA vs KEDA**：HPA 简单通用但信号失真；KEDA 精准但要额外组件与自定义指标。
- **模拟深度**：Kind 够练编排，但真实容量/网络要真机验证。
- **扩缩速度**：副本冷启动慢（模型加载）→ 扩容滞后于流量。

### 5.2 失效模式

1. **扩容滞后**：流量突增，新副本还在加载模型。修复：预热副本、调低触发阈值提前扩、或保留最小副本。
2. **抖动扩缩**：指标在阈值附近震荡，副本频繁增减。修复：加冷却时间（`--horizontal-pod-autoscaler-downscale-stabilization`）。
3. **探针配置错**：模型没加载完就接流量 → 请求失败。修复：`readinessProbe` + 足够 `initialDelaySeconds`。
4. **模拟误判**：把 Kind 的吞吐当生产容量。修复：明确标注"模拟数据"，生产前真机压测。

---

## 6. 延伸思考题（含解析）

**Q1**：Kind 模拟能验证什么、不能验证什么？
**A**：能：资源建模、调度、滚动更新、扩缩控制回路这些与机器数无关的逻辑。不能：真实多机网络延迟、跨节点故障、真实 GPU 调度与吞吐。故模拟数据只演练逻辑，容量要真机测。

**Q2**：为什么推理服务要配 readinessProbe？
**A**：模型加载慢（几秒到几十秒）。没有就绪探针，流量会打到还在加载的副本导致失败。探针确保副本真正就绪才接入流量。

**Q3**：为什么推理扩缩更适合队列驱动而非 CPU 驱动？
**A**：GPU 推理时 CPU 利用率未必高，CPU 信号失真；而排队长度直接反映"处理不过来"，是更准的扩容触发信号。KEDA 支持按队列/自定义指标扩缩。

**Q4**：扩容滞后（新副本加载模型慢）怎么缓解？
**A**：① 预热——保持最小副本数兜底；② 提前扩——调低触发阈值让扩容早于打满；③ 加快加载——模型缓存、更小镜像、分片加载。

**Q5**：滚动更新如何做到不中断服务？
**A**：RollingUpdate 逐批替换：先起新副本、等其就绪、再把流量切过去、再停旧副本，`maxUnavailable`/`maxSurge` 控制节奏，全程有就绪副本在服务。

---

## 本周交付清单

- [ ] Kind 搭 1 master + 2 worker，`kubectl get nodes` 验证。
- [ ] 部署小模型推理 Deployment + Service，副本就绪。
- [ ] 配 HPA（CPU）与 KEDA（队列）两种扩缩。
- [ ] 压测触发扩缩容，录屏展示副本变化闭环。
- [ ] 明确标注模拟边界，不把模拟数据当生产容量。
