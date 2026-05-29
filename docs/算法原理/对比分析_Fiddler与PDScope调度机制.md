# Fiddler vs PDScope 专家调度机制对比分析

> 对比对象：`10-fiddler-main/src/fiddler/mixtral.py`（793行）vs `20-PDScope/mixtral.py`（1572行）
> 模型：Mixtral-8x7B（32层 × 8专家）

---

## 一、整体架构差异

| 维度 | Fiddler | PDScope |
|------|---------|---------|
| 代码量 | 793 行（单文件） | 1572 行（单文件） |
| Placeholder 数量 | 1 个 | 6 个（4主 + 2 ondemand） |
| 执行方式 | 串行（GPU→CPU） | 三线程并行（GPU + CPU + 预取） |
| 调度算法 | O(2^8) 暴力枚举 | O(n log n) TG/TC 贪心 |
| Batch 支持 | 单条 / beam search | 多 batch（4/8/16/32/64） |
| 预取机制 | 无 | Gate 预测 + 异步预取 |
| 延迟模型 | 粗粒度静态值 | 实测 cpu_time_table 查表 |
| 专家分类 | 2 级（GPU / CPU） | 5 级（常驻 / 已预取 / 加载中 / ondemand / CPU） |

---

## 二、调度算法对比

### 2.1 Fiddler：暴力枚举 256 种方案

**位置**：`mixtral.py:686-718`

```python
# 统计每个专家的 CPU/GPU 延迟
cost_per_expert[i_expert, 0] = top_2.shape[0] * self.latency_cpu  # CPU延迟 ∝ token数
cost_per_expert[i_expert, 1] = self.latency_gpu                    # GPU延迟 = 固定值

# 遍历 2^8 = 256 种分配方案
for config in range(1 << len(experts)):      # 0 ~ 255
    sum_cost = 0
    for i_expert in range(len(experts)):
        if (config >> i_expert) & 1:         # 第 i 位 = 1 → CPU
            sum_cost += cost_per_expert[i_expert, 0]
        else:                                # 第 i 位 = 0 → GPU
            sum_cost += cost_per_expert[i_expert, 1]
    if sum_cost < best_cost:
        best_cost = sum_cost
        best_config = config
```

**特点**：
- 每个专家独立决定 CPU/GPU，共 2^8=256 种组合
- GPU 延迟与 token 数无关（固定 `latency_gpu=70`），CPU 延迟与 token 数成正比
- 目标函数是最小化 `sum(CPU延迟 + GPU延迟)`，**不是** `max(CPU, GPU)` 负载均衡
- 仅适用于 8 专家（Mixtral），对 DeepSeek 64/256 专家不可扩展（2^256 不可枚举）

**固定参数**：
```python
self.latency_cpu = 7    # CPU 上每 token 的延迟（单位未指定）
self.latency_gpu = 70   # 专家权重从 CPU 拷贝到 GPU 的固定延迟
```

### 2.2 PDScope：TG/TC 贪心分割

**位置**：`mixtral.py:894-938`

```python
# 过滤掉已在GPU（常驻/预取/placeholder）的专家，只对剩余专家排序
sorted_experts = sort_by_token_count(filtered_experts, descending)

e = 27.0    # 专家 CPU→GPU 传输延迟
tg = 4.22   # GPU 计算延迟
TA = sum(cpu_time_table[min(tokens, 1498)] for each expert)  # 全 CPU 总时间
TC = TA

for i in range(n-1):
    TG = (1 + i) * e + tg                          # 累积 GPU 成本
    TC = TC - cpu_time_table[min(token_count, 1498)]  # 递减 CPU 成本

    if self.is_decode:
        if TG < TC:              # GPU 时间 < CPU 时间 → 送 GPU
            if token_count > 1:
                ondemand_experts.append(expert_id)
            else:
                experts_in_placeholder.append(expert_id)
        else:
            break                # 后续专家全送 CPU
    else:  # Prefill
        if TG < TC + cpu_time_table[min(token_count, 1498)]:  # 更宽松条件
            ondemand_experts.append(expert_id)
        else:
            break
```

**特点**：
- 先过滤已驻留 GPU 的专家（常驻 + 已预取 + placeholder 中的），只对剩余专家决策
- GPU 成本 `TG = (1+i)*e + tg` 是**累积**的（搬运 i+1 个专家的串行时间）
- CPU 成本 `TC` 随分配逐步递减（移除的专家不再占 CPU）
- 基于实测 `cpu_time_table` 查表，而非固定系数
- Decode 额外条件：`token_count > 1` 才 ondemand，单 token 专家只进 placeholder
- 时间复杂度 O(n log n)（排序 + 单遍扫描），可扩展到任意专家数

---

## 三、专家分类对比

### 3.1 Fiddler：二分类

```python
# 暴力搜索后，按 bit 位分两组
for i_expert in range(8):
    if (best_config >> i_expert) & 1:
        cpu_experts.append(i_expert)    # CPU 组
    else:
        gpu_experts.append(i_expert)    # GPU 组
```

两类专家，每组内串行处理。

### 3.2 PDScope：五分类

```python
# 1. GPU 常驻专家
if self.is_expert_in_gpu_now(i_layer, i_expert):
    gpu_experts.append(i_expert)
    experts_in_gpu.append(i_expert)

# 2. 已预取到 placeholder 的专家
if expert_in_placeholder:
    experts_in_placeholder.append(i_expert)

# 3. 正在加载中的专家
elif i_expert in self.prefetching_list[i_layer]:
    experts_loading.append(i_expert)

# 4. Ondemand 实时加载的专家
elif i_expert in ondemand_experts:
    experts_remaining.append(i_expert)

# 5. CPU 执行的专家
else:
    cpu_experts.append(i_expert)
```

---

## 四、执行模式对比

### 4.1 Fiddler：串行执行

```python
# 先执行所有 GPU 专家（含 ondemand 加载）
for i_expert in gpu_experts:
    if self.is_expert_in_gpu(i_layer, i_expert):
        current_state = experts[i_expert](current_state, weights)  # 直接执行
    else:
        self.expert_placeholder.load_state_dict(experts[i_expert].state_dict())
        current_state = self.expert_placeholder(current_state, weights)  # 加载后执行
    inps_after_experts.index_add_(0, top_2, current_state)

# 再执行所有 CPU 专家
for i_expert in cpu_experts:
    current_state = self.run_expert_at_cpu(i_layer, i_expert, current_state.to("cpu"), weights.to("cpu"))
    inps_after_experts.index_add_(0, top_2, current_state.to(self.dev))
```

**问题**：GPU 和 CPU **串行**执行，总延迟 = GPU 时间 + CPU 时间，无并行加速。

### 4.2 PDScope：三线程并行

```python
# 预取线程：异步加载下一层热门专家到 placeholder
prefetch_thread = threading.Thread(target=prefetch_experts)

# GPU 线程：处理常驻 + 已预取 + 加载中 + ondemand 专家
gpu_thread = threading.Thread(target=process_gpu_experts)

# CPU 线程：处理 CPU 专家
cpu_thread = threading.Thread(target=process_cpu_experts)

# 启动并行
gpu_thread.start()
cpu_thread.start()
gpu_thread.join()    # 等待 GPU 完成
cpu_thread.join()    # 等待 CPU 完成
```

GPU 线程内部还分 4 个子线程并行处理不同类别的专家：
```python
threads.append(threading.Thread(target=process_experts_in_gpu))
threads.append(threading.Thread(target=process_experts_in_placeholder))
threads.append(threading.Thread(target=process_experts_loading))
threads.append(threading.Thread(target=process_experts_remaining))
```

**理论并行度**：总延迟 ≈ max(GPU 时间, CPU 时间)，而非 sum。

---

## 五、Placeholder 管理对比

### 5.1 Fiddler：单个 placeholder

```python
# 只预分配 1 个占位专家
self.expert_placeholder = copy.deepcopy(
    self.model.layers[0].block_sparse_moe.experts[0]
).to(self.dev)

# 每次只加载 1 个专家，用完即被下一个覆盖
self.expert_placeholder.load_state_dict(experts[i_expert].state_dict())
current_state = self.expert_placeholder(current_state, weights)
```

**限制**：同一时刻 GPU 上最多只有 1 个非驻留专家，无法并行处理多个 ondemand。

### 5.2 PDScope：6 个 placeholder + 奇偶层分离

```python
# 4 个主 placeholder（按奇偶层分离）
self.expert_placeholder  → 偶数层
self.expert_placeholder2 → 偶数层
self.expert_placeholder3 → 奇数层
self.expert_placeholder4 → 奇数层

# 2 个 ondemand 专用
self.expert_placeholder5
self.expert_placeholder6
```

**分配策略**（`_async_load_expert`）：
```python
if layer_idx % 2 == 1:    # 奇数层
    if not self.expert_placeholder3_inused:
        target = self.expert_placeholder3
    elif not self.expert_placeholder4_inused:
        target = self.expert_placeholder4
else:                     # 偶数层
    if not self.expert_placeholder_inused:
        target = self.expert_placeholder
    elif not self.expert_placeholder2_inused:
        target = self.expert_placeholder2
```

奇偶层分离的原因：预取线程在处理第 N 层时加载第 N+1 层的专家，相邻两层不能争用同一个 placeholder。

---

## 六、预取机制对比

### 6.1 Fiddler：无预取

每层独立决策，不考虑下一层。每次遇到非驻留 GPU 专家都要同步加载权重。

### 6.2 PDScope：Gate 预测 + 异步预取

**步骤 1：预测下一层专家**（`mixtral.py:946-960`）
```python
if i_layer < self.n_layer - 1:
    # 用下一层的 gate 预测路由
    next_router_logits = next_layer.block_sparse_moe.gate(inps)
    _, next_predicted_experts = torch.topk(routing_weights, 2, dim=-1)

    # 按 token 数排序取 top-k
    top3_experts = sorted_experts[:self.cache]
```

**步骤 2：异步预取**（`mixtral.py:1274-1327`）
```python
def prefetch_experts():
    layer_hot_experts = hot_experts.get(self.prefetch_layers, [])
    for expert_id in layer_hot_experts:
        if not self.is_expert_in_gpu_now(self.prefetch_layers, expert_id):
            self._async_load_expert(self.prefetch_layers, expert_id)
```

**步骤 3：标记管理**
```python
self.prefetching_list[layer] = [expert_id]   # 加载中
time.sleep(0.027)                             # 等待传输完成
self.prefetch_list[layer] = [expert_id]       # 加载完成
self.prefetching_list[layer] = []             # 清除加载中标记
```

**注意**：PDScope 使用 `time.sleep()` 等待预取完成，这是一个简化实现（应使用 CUDA Event 同步）。

---

## 七、GPU 常驻专家数量

### 7.1 Fiddler：自动计算

```python
def calc_n_expert_on_gpu(self):
    n_param = sum(p.numel() for p in expert.parameters())
    total_mem = torch.cuda.get_device_properties(self.dev).total_memory
    free_mem = total_mem * 0.95 - torch.cuda.memory_allocated(self.dev)
    return int(free_mem // (n_param * 2))  # bfloat16 每参数 2 字节
```

根据实际剩余显存动态计算。

### 7.2 PDScope：按 batch_size 硬编码

```python
def calc_n_expert_on_gpu(self):
    if self.batch_size == 64:
        return 62
    elif self.batch_size == 32:
        return 70
    else:
        return 74
```

不同 batch_size 需要不同大小的 KV Cache，硬编码为经验值。

---

## 八、专家状态检测对比

### 8.1 Fiddler：只看静态标记

```python
def is_expert_in_gpu(self, i_layer, i_expert):
    return self.expert_loc[i_layer, i_expert] == 1
```

`expert_loc` 在初始化时设置，运行期间不变。

### 8.2 PDScope：动态检测实际位置

```python
def is_expert_in_gpu_now(self, i_layer, i_expert):
    expert = self.model.layers[i_layer].block_sparse_moe.experts[i_expert]
    return next(expert.parameters()).is_cuda
```

通过检查参数张量是否在 CUDA 上来判断实际位置，能反映预取/ondemand 的动态变化。

---

## 九、已知 Bug 对比

### PDScope 特有 Bug（Fiddler 不存在）

| # | Bug | 位置 | 影响 |
|---|-----|------|------|
| 1 | placeholder 硬编码专家 `run_expert_at_gpu(11, 2, ...)` | line 1147 | 推理结果完全错误 |
| 2 | loading 硬编码专家 `run_expert_at_gpu(11, 2, ...)` | line 1173 | 推理结果完全错误 |
| 3 | 预取用 `time.sleep(0.027)` 模拟传输 | line 1321 | 预取不精确 |
| 4 | `release_placeholder()` 只释放 4/6 个 placeholder | line 406-413 | placeholder 5/6 永不释放 |
| 5 | `expert_placeholder_inused` 未定义（应为 `self.expert_placeholder_inused`） | line 54 vs 56 | 预取时可能 AttributeError |

### Fiddler 的局限性（非 Bug）

| # | 局限性 | 影响 |
|---|--------|------|
| 1 | 暴力枚举 O(2^n) | 仅适用于 8 专家模型 |
| 2 | sum 而非 max 目标 | 不保证 CPU/GPU 负载均衡 |
| 3 | GPU/CPU 串行执行 | 总延迟 = GPU + CPU |
| 4 | 单个 placeholder | 同一时刻只能有 1 个非驻留专家在 GPU |
| 5 | 无预取机制 | 每次 ondemand 都要同步等待 |
| 6 | 固定延迟参数 `latency_cpu=7, latency_gpu=70` | 不随硬件自适应 |

---

## 十、关键差异流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Fiddler 执行流程                              │
│                                                                     │
│  Gate路由 → 暴力256种方案 → [GPU组串行执行] → [CPU组串行执行] → 合并   │
│                          │                                          │
│                          └─ 唯一1个placeholder                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        PDScope 执行流程                              │
│                                                                     │
│  Gate路由 → 过滤已驻留专家 → TG/TC贪心分割 → ┬─ GPU线程(4子线程) ─┐  │
│          │                                  │                     │  │
│          ├─ 下一层Gate预测 → 预取线程 ───────┤                     │  │
│          │                                  └─ CPU线程 ───────────┤  │
│          │                                                        │  │
│          └─ 6个placeholder(奇偶分离)                              │  │
│                                     结果合并 ←────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 十一、总结

| 评估维度 | Fiddler | PDScope |
|----------|---------|---------|
| **设计定位** | 概念验证（proof-of-concept） | 工程优化原型 |
| **核心贡献** | 展示"激活值传CPU而非权重传GPU"的思路 | 在 Fiddler 基础上增加负载均衡+预取+并行 |
| **调度质量** | 暴力枚举最优解但目标函数不当 | 贪心近似但目标更合理（TG vs TC） |
| **执行效率** | 串行，无并行加速 | 多线程并行，理论加速 max(GPU,CPU) |
| **代码质量** | 简洁清晰，无已知 Bug | 功能更多但存在硬编码 Bug |
| **可扩展性** | 仅 Mixtral-8x7B | 理论可扩展到更多专家 |
| **实用性** | 适合理解基本原理 | 适合作为性能优化起点 |
