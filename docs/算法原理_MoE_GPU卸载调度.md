# MoE GPU 卸载调度算法原理

本文档整合了 MoE 混合专家模型 GPU 卸载系统的完整算法原理，包括三级调度策略、负载均衡分析、核心机制和优化方向。

---

# 第一部分：系统概述

## 涉及文件

| 文件 | 功能 |
|------|------|
| `deepseek.py` / `qwen.py` / `moon.py` / `mixtral.py` | 模型主类，加载和推理逻辑 |
| `config.py` | 统一配置管理，提供 `e`/`tg`/CPU时间表等参数 |
| `logger.py` | 统一日志和报告系统 |
| `scheduler.py` | ondemand 调度算法和预取策略 |

## 初始化阶段核心步骤

1. **加载配置**: 从 `config.py` 或 `args` 初始化
2. **加载模型**: 使用 `transformers.AutoModelForCausalLM.from_pretrained()` 加载完整模型
3. **非专家组件固定到 GPU**: Embedding 层、Attention 层、LayerNorm 层、Gate 投影层、共享专家
4. **计算可容纳专家数**: `calc_n_expert_on_gpu()` — 根据显存和 buffer_factor 计算
5. **设置专家位置**: `set_expert_loc()` — 优先加载热点专家
6. **加载专家到 GPU**: `bring_expert_to_gpu()` — 根据 `expert_loc` 矩阵移动

---

# 第二部分：三级调度策略

## 2.1 第一级：热点策略（静态规划）

### 目标
将高频专家放在 GPU 常驻，提升命中率。

### 算法流程
```python
# 1. 统计热点专家
expert_hot_data = {(layer, expert): token_count}

# 2. 按token数排序
hot_experts = sorted(expert_hot_data, by=token_count, descending)

# 3. 计算GPU可容纳专家数
n_gpu = calc_n_expert_on_gpu():
    可用显存 = gpu_memory * 0.95 * buffer_factor(0.7)
    单专家大小 = gate_proj + up_proj + down_proj
    n_gpu ≈ 830

# 4. 将热点专家放在GPU
for (layer, expert) in hot_experts[:n_gpu]:
    expert_loc[layer, expert] = 1  # 标记为GPU
```

### 效果
- GPU常驻 830/1664 = 49.9% 专家
- 命中率 70-86%（top-6 路由使热点专家高频命中）
- `buffer_factor=0.7` 预留30%显存给 KV cache 和激活值

### buffer_factor 的作用
预留显存空间，避免推理时 OOM：
```
可用显存(95%): 19.12 GB
无buffer可容纳: 1186个
buffer_factor: 0.7
最终: 830个 (1186 × 0.7)
```
可根据实际情况调整（0.7-0.9），修改位置如 `deepseek.py:616`。

---

## 2.2 第二级：Ondemand 决策（动态调度）

### 目标
动态决定是否将 CPU 专家搬运到 GPU，不是"尽可能多搬"，而是根据成本动态决策。

### 关键参数

| 参数 | 含义 | 示例值 |
|------|------|--------|
| `e` | 专家 CPU→GPU 搬运时间 | 1.09ms |
| `tg` | GPU 上计算一个专家的时间 | 0.14ms |
| `cpu_time_table` | 不同 token 数下 CPU 计算时间 | 0.07~5.87ms |

### 决策算法
```python
# 按token数降序排序CPU上的专家
sorted_experts = sort_by_token_count(cpu_experts)

TA = sum(cpu_time_table[token_count] for each expert)  # 全CPU总时间
TC = TA           # 剩余CPU时间
total_ondemand = 0  # 累计搬运成本

for i in range(n-1):
    TG = total_ondemand + e + tg    # 搬运i+1个专家的成本
    TC -= cpu_time_table[token_count]  # 减去当前专家的CPU时间

    if TG < TC:          # 成本最优：搬运+GPU < CPU
        ondemand.append(expert_id)
    elif TC > total_ondemand:  # 负载均衡：CPU不会过载
        ondemand.append(expert_id)
    else:
        break            # 停止搬运
```

### 决策三重目标

| 目标 | 条件 | 含义 |
|------|------|------|
| 成本最优 | `TG < TC` | 搬运+GPU计算 < CPU计算 → 搬运划算 |
| 负载均衡 | `TC > total_ondemand_cost` | CPU工作量 > 累计搬运成本 → CPU不过载 |
| 资源限制 | `max_ondemand ≤ 4` | 与 placeholder 数量匹配 |

### 决策示例
```
场景: batch_size=4, decode阶段
sorted_experts = [(exp1, 4), (exp2, 3), (exp3, 2), (exp4, 1)]

TA = 0.296 + 0.222 + 0.148 + 0.074 = 0.74ms

专家1: TG=1.23ms, TC=0.444ms → TG>TC 但 TC>0 → 搬运(负载均衡)
专家2: TG=2.46ms, TC=0.222ms → 不满足 → break
```

---

## 2.3 第三级：Prefetch 预取（异步预测）

### 目标
预测下一层需要的专家，提前异步加载。

### 算法流程
```python
# 1. 预测下一层热点专家
next_predicted_experts, _, _ = next_layer.mlp.gate(inps)
top_experts = sorted_experts[:self.cache]

# 2. 异步预取到空闲placeholder
for (layer+1, expert_id) in top_experts:
    if not is_expert_in_gpu_now(layer+1, expert_id):
        placeholder = find_available_placeholder()
        with torch.cuda.Stream(self._prefetch_stream):
            for name in ["gate_proj", "up_proj", "down_proj"]:
                dst.copy_(src, non_blocking=True)
```

### 当前效果
- 预取占比: 3.8%（2773次命中）
- 效果有限，因为热点专家已在 GPU 常驻，预取目标空间有限

### 预取局限性
1. GPU 已放置49.9%专家，热点策略覆盖大部分高频专家
2. 预取检查跳过已在GPU的专家
3. 只有CPU上的冷门专家被预取，但通常不是热点
4. `self.cache` 预取数量有限（batch_size=1时cache=15）

---

# 第三部分：并行执行架构

## 执行流程
```python
for layer in range(n_layer):
    # 1. 专家分类（4类）
    experts_in_gpu          # GPU常驻
    experts_in_placeholder  # 已预取
    experts_loading         # 正在加载
    cpu_experts             # CPU处理

    # 2. Ondemand决策
    ondemand_experts = scheduler.decide_ondemand(sorted_experts, is_decode)

    # 3. GPU线程和CPU线程并行
    gpu_thread = Thread(target=process_gpu_experts)
    cpu_thread = Thread(target=process_cpu_experts)
    gpu_thread.start(); cpu_thread.start()
    gpu_thread.join(); cpu_thread.join()

    # 4. 合并结果
    outputs = merge_results()
```

## GPU 线程处理
```python
def process_gpu_experts():
    gpu_time = 0
    for exp in experts_in_gpu:
        output = run_expert_at_gpu(exp)
        gpu_time += tg
    for exp in experts_in_placeholder:
        output = placeholder(inps)
        gpu_time += tg
    for exp in ondemand_experts:
        _async_ondemand(exp, placeholder)  # 搬运约1.09ms
        output = placeholder(inps)
        gpu_time += e + tg
    synchronize()
    return gpu_time
```

## CPU 线程处理
```python
def process_cpu_experts():
    cpu_time = 0
    for exp in cpu_experts:
        output = run_expert_at_cpu(exp)
        cpu_time += cpu_time_table[token_count]
    synchronize()  # 等待GPU完成（可能阻塞）
    return cpu_time
```

---

# 第四部分：日志数据分析

## 4.1 expert_stats.txt — 层处理时间
```
层 1: 93.39ms (中间层: 38.48ms, 最终层: 54.91ms)
```
- **中间层时间**: 专家FFN计算时间
- **最终层时间**: 路由决策 + 输出合并 + 残差+LayerNorm

瓶颈定位：
- 中间层占比高 → 专家计算是瓶颈
- 最终层占比高 → 路由/调度/合并开销大

## 4.2 线程时间统计
```
Layer 1: GPU 52.85ms, CPU 37.41ms, 并行 53.58ms, 并行度 1.68x
```
- 并行度 = (GPU时间 + CPU时间) / 并行时间
- 理想: 2.0x（完全并行）

---

# 第五部分：热点专家更新机制

### 流程
```python
# 记录当前迭代
self.current_iter_expert_stats[i_layer] = {
    'expert_ids': [...], 'token_counts': [...]
}

# get_hot_expert() 按token数降序排序
sorted_experts = sorted(zip(ids, counts), key=lambda x: x[1], reverse=True)
hot_experts[layer] = [expert[0] for expert in sorted_experts]

# 备份到last_iter，清空current_iter
```

### 重要提示
代码中只有**统计**逻辑，没有**保存到文件**逻辑。如需持久化到 `hot/deep.txt`，需手动添加保存代码。

---

# 第六部分：性能数据与负载分析

## 6.1 当前性能表现（batch_size=4）

| 指标 | 数值 | 占比 |
|------|------|------|
| GPU总工作时间 | 109997ms | 91.7% |
| CPU总工作时间 | 84499ms | 70.4% |
| 并行总时间 | 119963ms | 100% |
| GPU空闲时间 | 9966ms | 8.3% |
| CPU空闲时间 | 35463ms | 29.6% |

**GPU/CPU负载比**: 1.30:1（GPU过载）
**并行度**: 1.62x（理论最优1.77x，损失8.5%）

## 6.2 负载不均衡根因

### 原因1：ondemand搬运增加GPU负担
- 4548次ondemand，每次~1.09ms，累计约5秒加到GPU线程
- CPU工作量反而减少

### 原因2：GPU处理更多专家
- GPU: 70%常驻 + 6.2%ondemand + 3.8%prefetch ≈ 80%
- CPU: 20%

### 原因3：ondemand搬运是同步阻塞的
- `_async_ondemand` 实际是同步的
- GPU线程阻塞，CPU快速完成后等待GPU

### 原因4：CPU线程包含 synchronize() 等待GPU
- `run_expert_at_cpu` 调用 `torch.cuda.synchronize()`
- CPU线程阻塞等待，无法真正并行

---

# 第七部分：关键矛盾与优化方向

## 7.1 测量与实际开销矛盾
- `cpu_time_table` 测量纯CPU计算时间（理想环境）
- 实际运行包含 `synchronize()` 等待GPU（并发环境）
- 实测值(6.30ms) vs 表中值(0.30ms)，相差21倍
- Scheduler 决策依据不准确

**改进**：修改测量方式模拟并发环境，或在 scheduler 中加入 `synchronize_overhead`。

## 7.2 GPU过载矛盾
- Ondemand搬运在GPU线程同步执行
- GPU承担80%专家，CPU只承担20%

**改进**：
1. Ondemand搬运真正异步（CUDA Stream）
2. 减少 `buffer_factor`（从0.7降到0.6），让更多专家走CPU
3. 动态负载均衡，实时调整 ondemand 决策

## 7.3 Prefetch效果有限矛盾
- 预取占比仅3.8%
- 热点专家已在GPU，预取目标空间有限

**改进**：
1. 减少 GPU 常驻专家，让部分热点走预取而非常驻
2. 改进预取预测（基于历史路由数据）
3. 增加 placeholder 数量（从6个到8-10个）

## 7.4 理论TG vs 实际开销矛盾
- 代码中 `e = 1.11ms`，实际测量约1.39-1.5ms，低估25%
- 导致更多专家被分配到GPU队列

---

# 第八部分：优化路径总结

### 短期优化（低难度）
- 修改 scheduler 决策，考虑 `synchronize()` 等待开销
- 减少 `buffer_factor`，增加CPU工作量
- 使用实测搬运开销替代静态 `e` 值

### 中期优化（中等难度）
- Ondemand搬运真正异步（CUDA Stream）
- 改进 microbench 测量（模拟并发环境）

### 长期优化（高难度）
- 动态负载均衡（实时调整 ondemand 决策）
- 增加 placeholder 数量（需更多显存）
- 改进预取预测算法

### 瓶颈优先级

| 优先级 | 瓶颈 | 影响 | 难度 |
|--------|------|------|------|
| **P0** | GPU线程同步阻塞 | 严重 | 中 |
| **P1** | CPU时间表不准确 | 中等 | 低 |
| **P2** | 热点策略占用GPU过多 | 中等 | 高 |
| **P3** | placeholder数量有限 | 低 | 高 |

---

# 第九部分：关键提醒

1. **硬编码参数必须重新测量**: `e` 和 `tg` 与硬件强相关，换机器必须先跑 `microbench.py`
2. **热点专家不会自动保存**: 代码只统计不保存，需手动添加保存逻辑
3. **buffer_factor 是必要的**: 预留空间对稳定运行很重要，可在0.7-0.9之间调整
