# MoE GPU卸载算法完整流程

## 一、测试什么数据（Microbench）

### 1.1 硬件性能参数测量

**测试目标**：量化硬件的基本性能特征，为调度决策提供依据

**测试内容**：

#### e (专家传输时间) ≈ 1.09ms
- **测量方法**：`expert.to("cuda") + synchronize()`
- **含义**：CPU->GPU搬运一个专家权重的完整时间
- **用途**：ondemand/prefetch搬运开销估算

#### tg (GPU计算时间) ≈ 0.14ms
- **测量方法**：`expert(inps) + synchronize()`（固定token数）
- **含义**：GPU上计算一个专家的时间
- **用途**：GPU专家计算开销（与token数无关，GPU并行）

#### cpu_time_table[1~64] ≈ 0.07~5.87ms
- **测量方法**：遍历token=1~64，测量`expert(inps)`在CPU上
- **含义**：CPU上计算不同token数量的专家时间
- **用途**：CPU专家计算时间表（与token数线性相关）

**测试环境**：
- 单专家预热后的稳定值
- 无并发干扰的理想环境
- 测量纯计算时间（不含数据传输）

**关键假设**：
- GPU计算与token数无关（GPU并行处理）
- CPU计算与token数线性相关（串行处理）
- 搬运成本固定（每个专家约1.09ms）

**实测数据示例**（DeepSeek-V2-lite, RTX 4090D）：
```
Token=1:  CPU计算 0.074ms
Token=4:  CPU计算 0.296ms
Token=10: CPU计算 0.762ms
Token=64: CPU计算 5.87ms

e = 1.09ms（搬运时间）
tg = 0.14ms（GPU计算时间）
```

---

## 二、怎么使用这些数据

### 2.1 配置加载与存储

**存储结构**：
```json
// configs/system_config_deepseek.json
{
  "gpu_name": "NVIDIA GeForce RTX 4090 D",
  "gpu_memory_gb": 23.98,
  "transfer_time_ms": 1.09,  // 搬运时间（毫秒）
  "gpu_compute_time_ms": 0.14,  // GPU计算时间（毫秒）
  "cpu_time_table": [0.074, 0.148, ...]  // CPU时间表（毫秒）
}
```

**加载流程**：
```python
# 1. 启动时加载配置
config = ModelConfig.from_args(args)
scheduler = ExpertScheduler(config)

# 2. Scheduler初始化
self.e = config.e  # 1.09ms
self.tg = config.tg  # 0.14ms
self.cpu_time_table = config.cpu_time_table  # [0.074, 0.148, ...]

# 3. latency.py自动检测配置
if not os.path.exists(config_path):
    print("Config not found, running microbench...")
    run_microbench_and_generate_config()
```

**关键参数说明**：
- `e`：单位毫秒，搬运一个专家的成本
- `tg`：单位毫秒，GPU计算一个专家的成本
- `cpu_time_table`：单位毫秒，索引为token数，值为计算时间

---

## 三、怎么调度（三级调度策略）

### 3.1 第一级：热点策略（静态规划）

**目标**：将高频专家放在GPU常驻，提升命中率

**算法流程**：
```python
# 1. 统计热点专家（训练数据或历史运行）
expert_hot_data = {(layer, expert): token_count}

# 2. 按token数排序，选top热点专家
hot_experts = sorted(expert_hot_data, 
                     by=token_count, 
                     descending)

# 3. 计算GPU可容纳专家数
n_gpu = calc_n_expert_on_gpu():
  - 可用显存 = gpu_memory * 0.95 * buffer_factor(0.7)
  - 单专家大小 = gate_proj + up_proj + down_proj
  - n_gpu = 可用显存 / 单专家大小 ≈ 830

# 4. 将热点专家放在GPU
for (layer, expert) in hot_experts[:n_gpu]:
  model.layers[layer].mlp.experts[expert].to("cuda")
  expert_loc[layer, expert] = 1  # 标记为GPU
```

**效果**：
- GPU常驻830/1664 = 49.9%专家
- 命中率70-86%（top-6路由使热点专家高频命中）
- buffer_factor=0.7预留30%显存给KV cache和激活值

**热点策略原理**：
- MoE每层选top-6专家（路由网络决策）
- 热点专家在各层高频被选中
- 放在GPU上可避免ondemand搬运开销

---

### 3.2 第二级：Ondemand决策（动态调度）

**目标**：动态决定是否将CPU专家搬运到GPU

**算法输入**：
```python
sorted_experts = [(expert_id, token_count), ...]  # 按token数降序
is_decode = True/False  # 是否decode阶段
e = 1.09ms  # 搬运成本
tg = 0.14ms  # GPU计算成本
cpu_time_table = [...]  # CPU计算时间表
```

**决策逻辑详解**：

#### 核心思想
不是"尽可能多搬"，而是根据成本动态决策：
- 当搬运+GPU计算成本 < CPU计算成本时，搬运
- 考虑负载均衡，保证CPU有足够工作量
- 限制ondemand数量（与placeholder匹配）

#### 算法步骤
```python
# 1. 计算TA（如果所有专家都在CPU的总时间）
TA = sum(cpu_time_table[token_count] for each expert)

# 2. 遍历专家，逐步决策
TC = TA  # 剩余CPU时间
total_ondemand_cost = 0  # 累计搬运成本

for i in range(n-1):
  expert_id, token_count = sorted_experts[i]
  
  # 3. 计算搬运成本TG_i
  TG_i = e + tg = 1.09 + 0.14 = 1.23ms
  TG = total_ondemand_cost + TG_i
  
  # 4. 计算留在CPU的成本TC_i
  TC -= cpu_time_table[token_count]
  
  # 5. 决策（decode阶段）
  if TG < TC:
    # 成本最优：搬运收益 > 成本
    ondemand_experts.append(expert_id)
    total_ondemand_cost += TG_i
    
  elif TC > total_ondemand_cost:
    # 负载均衡：保证CPU有工作量
    ondemand_experts.append(expert_id)
    total_ondemand_cost += TG_i
    
  else:
    # 不满足条件，停止搬运
    break
```

#### 决策目标
1. **成本最优**：`TG < TC`
   - 搬运+GPU计算 < CPU计算 → 搬运划算

2. **负载均衡**：`TC > total_ondemand_cost`
   - 剩余CPU工作量 > 累计搬运成本 → CPU不会过载

3. **资源限制**：`max_ondemand=4`
   - 与placeholder数量匹配（6个placeholder，最多用4个）

#### 典型决策示例
```
场景：batch_size=4, decode阶段
sorted_experts = [(exp1, 4), (exp2, 3), (exp3, 2), (exp4, 1)]

TA = cpu_table[4] + cpu_table[3] + cpu_table[2] + cpu_table[1]
    = 0.296 + 0.222 + 0.148 + 0.074 = 0.74ms

专家1决策：
  TG = 1.23ms
  TC = 0.74 - 0.296 = 0.444ms
  TG > TC → 不搬运（成本不划算）
  但 TC(0.444) > total_ondemand_cost(0) → 搬运（负载均衡）
  ondemand_experts.append(exp1)
  total_ondemand_cost = 1.23ms

专家2决策：
  TG = 1.23 + 1.23 = 2.46ms
  TC = 0.444 - 0.222 = 0.222ms
  TG > TC → 不搬运
  TC(0.222) < total_ondemand_cost(1.23) → 不搬运
  break（停止决策）
```

---

### 3.3 第三级：Prefetch预取（异步预测）

**目标**：预测下一层需要的专家，提前异步加载

**算法流程**：
```python
# 1. 预测下一层热点专家（基于当前层路由）
top4_experts = predict_next_layer_hot_experts(current_routing)

# 2. 异步预取
for (layer+1, expert_id) in top4_experts:
  if not is_expert_in_gpu_now(layer+1, expert_id):
    # 找可用placeholder
    placeholder = find_available_placeholder()
    
    # 异步加载
    _async_load_expert(layer+1, expert_id, placeholder)

# 3. CUDA Stream异步拷贝
with torch.cuda.Stream(self._prefetch_stream):
  for name in ["gate_proj", "up_proj", "down_proj"]:
    dst = getattr(placeholder, name).weight.data
    src = getattr(experts[expert_id], name).weight.data
    dst.copy_(src, non_blocking=True)

# 4. 下一层开始时等待完成
torch.cuda.synchronize(self._prefetch_stream)
```

**效果**：
- 预取占比3.8%（热点专家已在GPU，预取空间有限）
- 异步执行，与当前层计算并行
- 使用CUDA Stream + non_blocking实现真正异步

**预取局限性**：
- GPU已放置49.9%专家，热点策略覆盖大部分高频专家
- 预取检查`is_expert_in_gpu_now`跳过已在GPU的专家
- 只有CPU上的冷门专家被预取，但通常不是热点

---

### 3.4 并行执行架构

**执行流程**：
```python
for layer in range(n_layer):
  # 1. 专家分类（4类）
  experts_in_gpu = [已在GPU常驻的专家]
  experts_in_placeholder = [已预取的专家]
  experts_loading = [正在加载的专家]
  cpu_experts = [需ondemand或走CPU的专家]
  
  # 2. Ondemand决策
  sorted_experts = sort_by_token_count(cpu_experts)
  ondemand_experts = scheduler.decide_ondemand(sorted_experts, is_decode)
  
  # 3. 启动GPU线程和CPU线程
  gpu_thread = Thread(target=process_gpu_experts)
  cpu_thread = Thread(target=process_cpu_experts)
  
  # 4. GPU线程处理
  def process_gpu_experts():
    gpu_time = 0
    
    # 处理GPU常驻专家
    for exp in experts_in_gpu:
      output = run_expert_at_gpu(layer, exp, inps)
      gpu_time += tg
    
    # 处理预取专家
    for exp in experts_in_placeholder:
      output = placeholder(inps)  # 用预取的权重
      gpu_time += tg
    
    # 处理ondemand专家（同步搬运+计算）
    for exp in ondemand_experts:
      _async_ondemand(layer, exp, placeholder)  # 搬运约1.09ms
      output = placeholder(inps)
      gpu_time += e + tg
    
    synchronize()
    return gpu_time
  
  # 5. CPU线程处理
  def process_cpu_experts():
    cpu_time = 0
    
    for exp in cpu_experts:
      output = run_expert_at_cpu(layer, exp, inps)
      cpu_time += cpu_time_table[token_count]
    
    synchronize()  # 等待GPU完成（阻塞开销）
    return cpu_time
  
  # 6. 并行执行
  gpu_thread.start()
  cpu_thread.start()
  gpu_thread.join()
  cpu_thread.join()
  
  # 7. 合并结果
  outputs = merge_gpu_and_cpu_results()
  
  # 8. 计算并行度
  parallel_time = wall_time
  parallel_degree = (gpu_time + cpu_time) / parallel_time
```

**关键机制**：
- **占位专家**：GPU上预先创建6个placeholder，用于ondemand搬运
- **Pinned Memory**：CPU专家权重锁定到pin_memory()，加速传输
- **专家分类**：每层开始时分类专家，决定处理方式
- **并行线程**：GPU和CPU线程并行处理不同专家

---

## 四、达到什么目标

### 4.1 性能目标

**吞吐量提升**：
- **目标**：从2.26 token/s（单GPU）提升到3.5+ token/s（并行）
- **实现**：GPU/CPU并行处理，资源利用率从50%提升到91.7%
- **效果**：batch_size=4时吞吐3.48 token/s，提升55%

**延迟降低**：
- **目标**：热点专家在GPU快速计算（0.14ms vs CPU 6.30ms）
- **实现**：热点策略使命中率70-86%，大部分专家在GPU
- **效果**：避免ondemand搬运开销（1.09ms）

**并行度最大化**：
- **目标**：并行度接近理论最优2x
- **实现**：当前1.62x（GPU/CPU负载比1.30:1）
- **差距**：性能损失8.5%（8.3%并行度未达到）

---

### 4.2 负载均衡目标

**理想状态**：
```
GPU工作量 ≈ CPU工作量
GPU时间 ≈ CPU时间
并行度 = 2x（完美并行）
无空闲等待
```

**当前状态**（batch_size=4, 平衡ondemand）：
```
GPU总工作时间: 109997ms (91.7%)
CPU总工作时间:  84499ms (70.4%)
并行总时间:     119963ms (100%)

GPU空闲时间:  9966ms (8.3%)
CPU空闲时间: 35463ms (29.6%)  ← GPU过载，CPU等待

并行度: 1.62x
理论最优: 1.77x
性能损失: 8.5%

GPU/CPU负载比: 1.30:1（GPU过载）
```

**负载不均衡根因**：
1. **Ondemand搬运在GPU线程同步执行**
   - 每次搬运约1.09ms
   - 4548次ondemand累计约5秒
   - 增加GPU线程负担

2. **CPU线程包含`synchronize()`等待GPU**
   - `run_expert_at_cpu`调用`torch.cuda.synchronize()`（deepseek.py:1988）
   - 等待GPU线程处理其他专家完成
   - CPU线程阻塞等待，无法真正并行

3. **Scheduler决策不考虑GPU等待开销**
   - `cpu_time_table`测量纯CPU计算时间（0.30ms）
   - 实测包含`synchronize()`等待（6.30ms，相差21倍）
   - 决策依据与实际开销不匹配

---

### 4.3 决策目标总结

**Scheduler决策的三重目标**：

| 目标 | 条件 | 含义 |
|------|------|------|
| 成本最优 | `TG < TC` | 搬运+GPU计算 < CPU计算 → 搬运划算 |
| 负载均衡 | `TC > total_ondemand_cost` | 剩余CPU工作量 > 累计搬运成本 → CPU不会过载 |
| 资源限制 | `max_ondemand ≤ 4` | 与placeholder数量匹配，防止资源耗尽 |

**决策优先级**：
1. 成本最优（节省总时间）
2. 负载均衡（最大化并行度）
3. 资源限制（硬件约束）

---

## 五、算法流程总结图

```
┌─────────────────────────────────────────────────┐
│  Microbench测试硬件性能                           │
│  e=1.09ms, tg=0.14ms, cpu_time_table=[...]      │
│  （单专家预热，无并发干扰）                        │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  配置加载到ModelConfig/SystemConfig              │
│  latency.py自动检测配置不存在时运行microbench     │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  热点策略（静态规划）                             │
│  830个高频专家 → GPU常驻                          │
│  命中率70-86%（top-6路由）                        │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  每层处理：                                       │
│  1. 专家分类（gpu/cpu/ondemand/prefetch）        │
│  2. Scheduler决策（TG vs TC + 负载均衡）         │
│  3. Prefetch预取（异步CUDA Stream）              │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  GPU线程 + CPU线程并行执行                        │
│  GPU: 处理70%专家 + ondemand搬运                  │
│  CPU: 处理20%专家 + synchronize等待GPU           │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  结果合并 → 下一层                                │
│  并行度1.62x，吞吐3.5 token/s                    │
│  GPU过载（负载比1.30:1），CPU空闲29.6%           │
└─────────────────────────────────────────────────┘
```

**关键数据流**：
```
测试数据 → Scheduler决策 → 并行执行 → 负载均衡
```

---

## 六、关键矛盾与改进方向

### 6.1 测量与实际开销矛盾

**问题描述**：
- `cpu_time_table`测量纯CPU计算时间（理想环境）
- 实际运行包含`synchronize()`等待GPU（并发环境）
- 实测值（6.30ms）远高于表中值（0.30ms），相差21倍

**影响**：
- Scheduler决策依据不准确
- 认为`TC=0.30ms < TG=1.23ms`，不该搬运
- 但实际CPU线程等待GPU完成，总时间远大于预期
- 导致负载不均衡（GPU过载，CPU空闲）

**改进方向**：
1. **修改microbench测量方式**
   - 模拟实际并发环境
   - 包含`synchronize()`等待开销
   - 测量真实CPU线程时间

2. **修改scheduler决策算法**
   - `TC = cpu_time_table[idx] + synchronize_overhead`
   - 动态调整`synchronize_overhead`（根据GPU负载）

3. **移除`synchronize()`**
   - 检查CPU专家计算是否真的需要等待GPU
   - 如果纯CPU操作，移除以减少等待

---

### 6.2 GPU过载矛盾

**问题描述**：
- Ondemand搬运在GPU线程同步执行
- 每次搬运约1.09ms，累计约5秒
- GPU承担80%专家，CPU只承担20%
- GPU线程工作量 > CPU线程工作量

**影响**：
- GPU线程成为瓶颈
- CPU线程快速完成，等待GPU
- 并行度无法达到理论最优2x

**改进方向**：
1. **Ondemand搬运真正异步**
   - 使用CUDA Stream异步搬运
   - 不在GPU线程内阻塞

2. **调整GPU常驻专家数量**
   - 减少`buffer_factor`（从0.7降到0.6）
   - 让更多专家走CPU，增加CPU工作量

3. **动态负载均衡**
   - 根据实时GPU/CPU负载调整ondemand决策
   - GPU负载高时减少ondemand，CPU负载高时增加ondemand

---

### 6.3 Prefetch效果有限矛盾

**问题描述**：
- Prefetch占比仅3.8%
- 热点专家已在GPU常驻，预取目标空间有限
- 只有CPU上的冷门专家被预取，但通常不是热点

**影响**：
- Prefetch对吞吐影响微弱
- CUDA Stream异步机制未充分利用

**改进方向**：
1. **减少GPU常驻专家**
   - 降低热点策略覆盖范围
   - 让部分热点专家走预取而非常驻

2. **改进预取预测**
   - 基于历史路由数据预测下一层热点
   - 提高预取命中率

3. **增加placeholder数量**
   - 当前只有6个，限制ondemand和prefetch并发
   - 增加到8-10个（需更多显存）

---

## 七、总结

### 7.1 算法核心思想

**三级调度策略**：
1. **热点策略**：静态规划，高频专家常驻GPU
2. **Ondemand决策**：动态调度，成本最优+负载均衡
3. **Prefetch预取**：异步预测，提前加载下一层专家

**决策目标**：
- 成本最优：搬运+GPU计算 < CPU计算
- 负载均衡：GPU工作量 ≈ CPU工作量
- 资源限制：placeholder数量约束

**关键机制**：
- 占位专家：GPU上预先创建，用于ondemand搬运
- Pinned Memory：加速CPU->GPU传输
- CUDA Stream：异步执行prefetch
- 并行线程：GPU和CPU同时处理不同专家

### 7.2 当前性能表现

**已实现目标**：
- ✓ 吞吐提升55%（2.26 → 3.48 token/s）
- ✓ 命中率70-86%（热点策略有效）
- ✓ 并行度1.62x（GPU/CPU并行）

**未实现目标**：
- ✗ 并行度未达理论最优（1.62x vs 1.77x，损失8.5%）
- ✗ 负载不均衡（GPU过载1.30:1，CPU空闲29.6%）
- ✗ Prefetch效果有限（占比3.8%）

**关键瓶颈**：
- P0: GPU线程同步阻塞（ondemand搬运）
- P1: CPU时间表不考虑并发等待
- P2: 热点策略占用GPU过多
- P3: Placeholder数量限制

### 7.3 优化路径

**短期优化**（低难度）：
- 修改scheduler决策，考虑`synchronize()`等待开销
- 减少`buffer_factor`，增加CPU工作量

**中期优化**（中等难度）：
- Ondemand搬运真正异步（CUDA Stream）
- 改进microbench测量（模拟并发环境）

**长期优化**（高难度）：
- 动态负载均衡（实时调整ondemand决策）
- 增加placeholder数量（需更多显存）
- 改进预取预测算法

---

## 附录：关键文件位置

| 文件 | 功能 | 关键代码 |
|------|------|----------|
| `microbench.py` | 硬件性能测试 | 第170-177行：CPU时间测量 |
| `scheduler.py` | Ondemand决策算法 | 第26-95行：decide_ondemand() |
| `deepseek.py` | 模型推理主类 | 第1704-1772行：并行执行流程 |
| `deepseek.py` | CPU专家计算 | 第1968-2002行：run_expert_at_cpu() |
| `config.py` | 配置管理 | 第134-143行：SystemConfig定义 |

---

## 参考资料

- `docs/performance_analysis_2026-04-15.md`：性能分析报告
- `docs/deepseek_pytorch_transformers_guide.md`：PyTorch & Transformers函数指南
- `实验日志.md`：完整实验记录
- `实验数据.md`：batch_size=4实验数据汇总