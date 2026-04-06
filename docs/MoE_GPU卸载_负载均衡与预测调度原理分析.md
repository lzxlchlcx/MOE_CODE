# MoE GPU卸载 负载均衡与预测调度原理分析

## 一、核心调度算法原理

### 1.1 调度决策公式

代码中的调度决策基于以下公式 (`deepseek.py:1092-1136`):

```python
e = 1.11   # 专家搬运开销 (ms)
tg = 0.20  # GPU计算开销 (ms)
TG = (1 + i) * e + tg  # 第i个专家的GPU累计开销
TC = sum(cpu_time_table[tokens])  # 剩余CPU计算时间
```

**调度规则** (decode阶段):
```
if TG < TC:
    → 分配给CPU (ondemand)
else:
    → 分配给GPU
```

**原理**: 当GPU搬运开销TG小于CPU计算时间TC时，说明该专家不值得等待GPU传输，直接在CPU计算反而更快。

### 1.2 预取机制 (Prefetch)

```python
# 预测下一层热点专家
next_predicted_experts, next_routing_weights, _ = next_layer.mlp.gate(inps)
top3_experts = [expert[0] for expert in sorted_experts[:self.cache]]
```

**原理**: 利用当前层的输出提前预测下一层的热点专家，实现流水线式预加载。

### 1.3 专家分类

```python
gpu_experts      # 直接在GPU上的专家
experts_in_placeholder  # 已预取到占位专家的
experts_loading        # 正在加载的
experts_remaining      # 需要ondemand的
cpu_experts            # 分配给CPU的
```

---

## 二、实验数据分析

### 2.1 Log数据解读

```
Layer 26: GPU 12.45ms, CPU 15.44ms, 并行度 1.77x
```

| 指标 | 值 | 含义 |
|------|-----|------|
| GPU Thread Time | 12.45ms | GPU线程总耗时 |
| CPU Thread Time | 15.44ms | CPU线程总耗时 |
| Parallel Time | 15.79ms | 实际并行执行时间 |
| Parallel Degree | 1.77x | 并行度 = (12.45+15.44)/15.79 |

**理想并行度**: 2.0x (GPU和CPU完全并行)
**实际并行度**: 1.77x → **存在17.5%的效率损失**

---

## 三、存在的问题与矛盾

### 3.1 核心问题: GPU/CPU负载不均衡

| 问题 | 表现 | 根因 |
|------|------|------|
| **调度偏向GPU** | CPU空闲，GPU繁忙 | TG计算低估了实际搬运开销 |
| **GIL阻塞** | CPU线程等待 | GPU线程处理ondemand时持有Python GIL |
| **串行化执行** | 并行度<2.0 | `process_experts_remaining`中每处理一个专家就join |

### 3.2 矛盾分析

**矛盾1: 理论TG vs 实际开销**

```python
e = 1.11  # 假设的搬运开销
```

但实际测量显示搬运开销约 **1.39-1.5ms**，代码中使用的是 **1.11ms**，低估了约 **25%**。

这导致更多专家被分配到GPU队列，而实际上CPU计算可能更快。

**矛盾2: 理论并行 vs 实际串行**

理论上GPU和CPU线程并行执行，但日志显示 **并行度只有1.77x** 而非理想的2.0x。

**根因**: GPU线程在处理ondemand专家时需要先执行 `_async_ondemand()` 进行权重传输，这段时间CPU线程处于等待状态。

**矛盾3: 预取预测准确率**

代码在第1145行预测下一层专家:
```python
next_predicted_experts, next_routing_weights, _ = next_layer.mlp.gate(inps)
```

但 `self.cache` 设置的预取数量有限 (batch_size=1时cache=15)，且预测基于单层输出，未考虑多层协同。

---

## 四、改进建议

### 4.1 修正调度参数

```python
# 实际测量搬运开销
e = 1.39   # 改为实测值
# 或动态测量
e = measure_copy_time()
```

### 4.2 优化并行度

将 `process_experts_remaining` 中的串行join改为批量处理:

```python
# 当前 (串行)
for t in threads:
    t.join()  # 每处理完一个专家才处理下一个

# 改进 (并行)
for t in threads:
    t.start()
for t in threads:
    t.join()  # 等待所有线程完成
```

### 4.3 增加异步预取

在CPU线程执行时，GPU线程可以异步预取下一批专家，实现真正的流水线重叠。

### 4.4 动态调整cache

```python
# 根据实际命中率动态调整
if hit_rate < 0.5:
    self.cache = min(self.cache + 5, 64)
else:
    self.cache = max(self.cache - 2, 3)
```

---

## 五、总结

| 方面 | 当前状态 | 理想状态 |
|------|---------|---------|
| 并行度 | 1.77x | 2.0x |
| 调度准确性 | 低估搬运开销25% | 精确测量 |
| 预取命中率 | 未统计 | >80% |
| CPU利用率 | 低 | GPU/CPU均衡 |

**核心矛盾**: 理论模型与实际执行存在偏差，特别是调度参数e和tg的静态设定，无法适应动态负载变化。
