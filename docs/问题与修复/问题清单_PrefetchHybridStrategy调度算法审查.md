# 问题清单：PrefetchHybridStrategy 调度算法审查

## 背景

审查对象：`40-myself/src/expert_scheduling.py` 中的 `PrefetchHybridStrategy`。

该策略用于 MoE 专家 CPU/GPU 混合卸载场景，主要包含两部分：

1. 当前层活跃专家的 CPU/GPU 执行位置决策。
2. 基于下一层 gate 预测结果的专家预取决策。

上层调用位置：`40-myself/src/deepseek.py`。

整体设计方向合理，但当前实现存在若干维度假设、代价模型、预取策略和工程一致性问题，需要在后续实验前优先核查。

---

## 一、总体结论

`PrefetchHybridStrategy` 试图实现类似 PDScope AdaptSched 的调度逻辑：

1. Prefill 阶段使用“三步调度法”：全局排序、局部重排、置信度预取。
2. Decode 阶段使用基于 `n_g^rho` 的 ABC 分支策略。
3. 当前层 GPU/CPU 执行与下一层专家预取通过线程并行重叠。

但当前实现更接近“论文思想的启发式移植”，还没有完全形成稳定、可验证的工程实现。主要风险集中在：

1. `gate()` 输出 shape 与 `_build_expert_mask()` 的维度假设可能不一致。
2. 代价模型中 CPU/GPU 延迟使用方式不完全合理。
3. Prefill 三步法的公式和论文算法存在偏差，且含多个硬编码经验参数。
4. Decode 分支只按专家数量判断，未考虑 token 负载和路由权重。
5. 预取结果缺少命中率、收益和误预取统计，难以验证实际有效性。

---

## 二、高优先级问题

### P0-1：`selected_experts` 维度假设可能错误

位置：`40-myself/src/expert_scheduling.py:9-19`

当前实现：

```python
return torch.nn.functional.one_hot(selected_experts, num_classes=n_expert).permute(2, 1, 0)
```

问题：

1. 注释写 `selected_experts` 形状为 `[batch_size, seq_len, 2]`。
2. 文档中已有记录显示 DeepSeek gate 常见输出是 `[num_tokens, top_k]`。
3. 如果 `selected_experts` 实际是 `[num_tokens, top_k]`，`one_hot()` 后是 `[num_tokens, top_k, n_expert]`，`permute(2, 1, 0)` 才会得到 `[n_expert, top_k, num_tokens]`。
4. 如果实际是 `[batch_size, seq_len, top_k]`，`one_hot()` 后是 4D，当前 `permute(2, 1, 0)` 会直接报错。

影响：

1. 活跃专家统计可能错误。
2. token 到 expert 的 assignment 可能错位。
3. routing weight 可能和 token 索引不匹配。
4. 后续 CPU/GPU 调度和预取决策全部基于错误输入。

建议：

1. 在 `decide_and_prepare()` 入口处明确支持的 shape。
2. 将 `selected_experts` 统一 reshape 为 `[num_tokens, top_k]`。
3. 将 `routing_weights` 同样统一为 `[num_tokens, top_k]`。
4. 更新 `_build_expert_mask()` 注释和断言。

建议检查项：

```python
assert selected_experts.dim() == 2
assert routing_weights.dim() == 2
assert selected_experts.shape == routing_weights.shape
```

---

### P0-2：`_organize_token_assignments()` 的 routing weight 索引风险

位置：`40-myself/src/expert_scheduling.py:46-62`

当前实现：

```python
idx, top_2 = torch.where(expert_mask[i_expert])
routing_weight_subset = routing_weights[top_2, idx, None]
expert_assignments[i_expert] = (top_2, routing_weight_subset)
```

问题：

1. `idx` 实际表示 top-k 维度中的位置。
2. `top_2` 实际表示 token 维度中的位置。
3. 变量名 `top_2` 容易误导，实际上它是 token index。
4. 如果 `routing_weights` 不是 `[num_tokens, top_k]`，该索引会错。

影响：

1. 专家输出乘错 routing weight。
2. `index_add_()` 位置可能正确但权重错误，推理结果会偏移且不容易立即报错。

建议：

1. 重命名变量：`topk_idx, token_idx = torch.where(...)`。
2. 明确 `routing_weights[token_idx, topk_idx]`。
3. 增加 shape 断言。
4. 添加小型单元测试，构造固定 `selected_experts/routing_weights` 验证每个专家拿到的 token 和权重。

---

### P0-3：当前层 CPU/GPU 最优搜索目标函数不符合并行执行模型

位置：`40-myself/src/expert_scheduling.py:240-259`

当前实现：

```python
sum_cost += cost_cpu[bit]
sum_cost += cost_gpu[bit]
```

问题：

1. 当前策略通过 `sum(CPU cost + GPU cost)` 找最小总代价。
2. 但 `deepseek.py` 中 CPU 专家和 GPU 专家是通过 `ThreadPoolExecutor` 并行执行的。
3. 并行执行下更合理的目标应接近 `min max(T_cpu, T_gpu)`，而不是 `min sum(T_cpu + T_gpu)`。

影响：

1. 调度可能倾向总和最小，但造成 CPU/GPU 负载不均衡。
2. 实际 wall time 可能不是最优。
3. 与 PDScope 负载均衡思想不完全一致。

建议：

1. 将搜索目标改为：

```python
cpu_total = sum(cpu costs assigned to CPU)
gpu_total = sum(gpu costs assigned to GPU)
objective = max(cpu_total, gpu_total)
```

2. 如需要兼顾总工作量，可用加权目标：

```python
objective = max(cpu_total, gpu_total) + lambda_penalty * (cpu_total + gpu_total)
```

---

## 三、中高优先级问题

### P1-1：CPU 代价可能被重复乘以 token 数

位置：`40-myself/src/expert_scheduling.py:233`

当前实现：

```python
cost_cpu[bit] = tc * _lookup_latency(self.latency_cpu_table, tc)
```

问题：

1. `_lookup_latency(table, token_count)` 的表项名称是 `avg_time_ms`。
2. 如果 benchmark 表记录的是“该 token_count 下完整专家 forward 的总耗时”，则这里再乘 `tc` 会把 CPU 代价放大 `tc` 倍。
3. 只有当表项表示“每 token 平均耗时”时，乘 `tc` 才合理。

影响：

1. 高 token 数专家会被过度倾向分配到 GPU。
2. CPU 路径被系统性低估可用性。
3. Prefetch/ondemand 决策会偏激进。

建议：

1. 检查 micro benchmark JSON 中 `expert_cpu[].avg_time_ms` 的语义。
2. 如果它已经是总耗时，应改为：

```python
cost_cpu[bit] = _lookup_latency(self.latency_cpu_table, tc)
```

3. 如果它是 per-token 耗时，应将字段名改成 `avg_time_per_token_ms`，避免误用。

---

### P1-2：GPU 代价没有包含非驻留专家的搬运成本

位置：`40-myself/src/expert_scheduling.py:234-238`

当前实现：

```python
cost_gpu[bit] = _lookup_latency(self.latency_gpu_table, tc)
if self.is_expert_in_gpu(i_layer, i_expert):
    cost_gpu[bit] = 0
```

问题：

1. 对非 GPU 常驻专家，GPU 执行不仅包含计算时间，还包含 CPU→GPU 权重搬运时间 `t_io`。
2. 当前 `cost_gpu` 只使用 `latency_gpu_table`，看起来更像 GPU compute time。
3. 对已在 GPU 的专家，代价直接置 0，也忽略了 GPU compute time。

影响：

1. GPU 路径可能被严重低估。
2. 非驻留专家可能被过多分配到 GPU。
3. 已驻留专家虽然无需搬运，但仍有计算时间，不应完全为 0。

建议：

1. 常驻 GPU 专家：`cost_gpu = gpu_compute(tc)`。
2. 非驻留 GPU 专家：`cost_gpu = t_io + gpu_compute(tc)`。
3. 如果要模拟“常驻专家计算与其他操作重叠”，也应单独建模，而不是直接置 0。

---

### P1-3：`predicted_next_weights` 参数没有被使用

位置：`40-myself/src/expert_scheduling.py:218-219, 276-360`

问题：

1. 接口传入了 `predicted_next_weights`。
2. `_prefill_schedule()` 和 `_decode_schedule()` 中没有实际使用该权重。
3. 当前下一层专家只按出现次数统计，不考虑路由强度。

影响：

1. 高权重专家和低权重专家被同等对待。
2. 预取优先级不够准确。
3. 弱预测专家可能浪费 placeholder 和传输带宽。

建议：

1. 对下一层专家计算加权热度：

```python
score[eid] += predicted_next_weights[token_idx, topk_idx]
```

2. 预取排序可综合 `token_count` 和 `routing_weight_sum`。
3. 可设置置信度阈值，只预取高置信专家。

---

### P1-4：Prefill 三步法实现与 Algorithm 1 存在偏差

位置：`40-myself/src/expert_scheduling.py:276-355`

问题：

1. 当前 Step 1 使用固定公式：

```python
T_all_G = alpha + n_transfer * self.t_io + _lookup_latency(self.latency_gpu_table, 1)
```

2. 这里的 `alpha = 0.1` 是硬编码，来源不明确。
3. `T_all_G` 使用 `n_transfer = total - i`，不是增量累积式。
4. `T_all_C` 对前缀求和，但当前排序和成本定义可能与论文不同。

影响：

1. 分割点可能偏离论文算法。
2. 当前层 ondemand 和下一层 prefetch 的集合都可能不稳定。
3. 不同 batch、不同硬件上的效果难以解释。

建议：

1. 参考 `docs/PDScope_opensource_refactor_report.md` 中 P5 版本描述，改为增量 `T_G_cum += t_io`。
2. 删除或配置化 `alpha`。
3. 将 Step 1、Step 2、Step 3 拆成更清晰的内部函数，便于单独测试。

---

## 四、中优先级问题

### P2-1：Decode 策略只按专家数量判断，未考虑 token 负载

位置：`40-myself/src/expert_scheduling.py:357-406`

当前实现：

```python
k = len(cpu_experts) + len(gpu_experts)
cur_gpu_resident = sum(1 for e in gpu_experts if self.is_expert_in_gpu(i_layer, e))
next_gpu_resident = ...
```

问题：

1. `k` 是专家数，不是 token 负载。
2. `cur_gpu_resident` 和 `next_gpu_resident` 也只是专家数量。
3. Decode 阶段虽然通常每步 token 少，但 batch/beam 情况下不同专家 token 数仍可能不同。

影响：

1. 多 token 专家和单 token 专家被等价看待。
2. 负载均衡可能失真。

建议：

1. Decode 策略中引入 `token_counts`。
2. `n_g^rho` 可以按预计 GPU/CPU 时间求解，而不是按专家数求解。
3. 下一层也应统计预测 token count 或 weighted score。

---

### P2-2：`i_layer + 1 < 100` 是硬编码

位置：`40-myself/src/expert_scheduling.py:262`

当前实现：

```python
if predicted_next_experts is not None and i_layer + 1 < 100:
```

问题：

1. `100` 与模型层数无关。
2. 上层实际模型层数是 `self.n_layer`。
3. 这里依赖外部预测器不传最后层，缺少明确边界。

影响：

1. 换模型后容易产生越界假设。
2. 可读性差。

建议：

1. 将 `n_layer` 作为策略初始化参数传入。
2. 使用 `i_layer + 1 < self.n_layer`。
3. 或由上层保证不传 `predicted_next_experts`，并删除该条件。

---

### P2-3：`idxs` 变量未使用

位置：`40-myself/src/expert_scheduling.py:223`

当前实现：

```python
active_experts, idxs, top_2s = _collect_active_experts(expert_mask, n_expert)
```

问题：

1. `idxs` 后续没有使用。
2. 同类问题也存在于 `HybridCPUGPUStrategy`。

影响：

1. 增加理解成本。
2. 容易误判 top-k index 和 token index 的含义。

建议：

1. 如果不需要，改成 `_`。
2. 如果需要，将变量名改成 `topk_indices_by_expert`。

---

### P2-4：预取没有去重和容量感知

位置：`40-myself/src/expert_scheduling.py:349-355, 400-404`

问题：

1. `_prefill_schedule()` 和 `_decode_schedule()` 返回的 `prefetch` 只是 list。
2. 没有显式去重。
3. 没有感知 placeholder 当前可用数量。
4. 上层 `_prefetch_next_layer_experts()` 遇到无 placeholder 会 `break`，但策略层不知道容量约束。

影响：

1. 预取列表可能超过实际容量。
2. 后面的高价值专家可能因为前面的低价值专家占位失败。
3. 预取收益不可控。

建议：

1. 策略层传入 `max_prefetch` 或 placeholder 空闲数量。
2. 返回前做去重：`list(dict.fromkeys(prefetch))`。
3. 按预期收益排序后截断。

---

## 五、低优先级问题

### P3-1：`F` 导入未使用

位置：`40-myself/src/expert_scheduling.py:6`

当前实现：

```python
import torch.nn.functional as F
```

问题：当前文件实际使用的是 `torch.nn.functional.one_hot`，没有使用别名 `F`。

建议：

1. 改成 `F.one_hot(...)`。
2. 或删除 `import torch.nn.functional as F`。

---

### P3-2：变量命名不准确

位置：`40-myself/src/expert_scheduling.py:38-41, 59-61`

问题：

1. `top_2` 实际表示 token index。
2. `idx` 实际表示 top-k slot index。
3. 对 DeepSeek top-k 可能是 6，不一定是 top-2。

建议：

1. `idx` 改为 `topk_idx`。
2. `top_2` 改为 `token_idx`。
3. 注释中避免写死 top-2，改成 top-k。

---

### P3-3：命中率统计粒度不足

位置：`40-myself/src/expert_scheduling.py:208-209, 235-238`

当前统计：

```python
self.cnt_expert_hit
self.cnt_expert_all
```

问题：

1. 只能统计 GPU 常驻命中。
2. 无法区分 hot expert 命中、prefetch 命中、ondemand 加载。
3. 无法统计误预取。

建议新增指标：

1. `hot_hit_count`
2. `prefetch_hit_count`
3. `prefetch_request_count`
4. `prefetch_used_count`
5. `ondemand_count`
6. `cpu_execution_count`

---

## 六、建议修复顺序

### 第一阶段：先保证正确性

1. 修正 `selected_experts/routing_weights` shape 契约。
2. 修正 `_organize_token_assignments()` 的变量命名和索引逻辑。
3. 增加最小单元测试，验证专家 token 分配和 routing weight 匹配。

### 第二阶段：修正代价模型

1. 确认 benchmark 表中 `avg_time_ms` 是总耗时还是 per-token 耗时。
2. GPU 代价加入 `t_io` 和 compute time。
3. 当前层 CPU/GPU 搜索目标从 `sum` 改为 `max(cpu_total, gpu_total)`。

### 第三阶段：优化预取策略

1. 使用 `predicted_next_weights` 做加权优先级。
2. 预取列表去重并加入容量约束。
3. 将 `alpha/t_attn/R_hit` 参数化。
4. 增加预取命中率和误预取统计。

### 第四阶段：对齐 PDScope Algorithm 1

1. Prefill Step 1 改为增量式 `T_G_cum`。
2. Prefill Step 2 明确 `L_global ∩ E_cur`。
3. Prefill Step 3 使用逐专家期望收益和自然回退。
4. Decode 分支引入 token 负载和预测权重。

---

## 七、建议实验验证

### 正确性验证

1. 构造 toy `selected_experts` 和 `routing_weights`，检查每个专家拿到的 token index 与权重。
2. 对比 `GPUOnlyStrategy` 和 `PrefetchHybridStrategy` 在强制全 GPU 情况下的输出差异。
3. 检查 top-k 为 6 时是否仍能正确工作。

### 性能验证

1. 记录每层 `cpu_experts/gpu_experts/prefetch_experts` 数量。
2. 记录实际 CPU/GPU/prefetch wall time。
3. 对比 `sum` 目标和 `max` 目标下的端到端延迟。
4. 统计预取专家是否在下一层真正被使用。

### 消融实验

1. 关闭预取，仅保留 CPU/GPU 混合调度。
2. 开启预取但不使用 routing weight。
3. 开启 routing weight 加权预取。
4. 对比不同 `R_hit`、`t_attn`、`max_prefetch` 参数。

---

## 八、关联文件

| 文件 | 说明 |
|------|------|
| `40-myself/src/expert_scheduling.py` | `PrefetchHybridStrategy` 主实现 |
| `40-myself/src/deepseek.py` | 策略调用、CPU/GPU/Prefetch 并行执行 |
| `40-myself/src/expert_predictor.py` | 下一层 gate 预测器 |
| `40-myself/src/placeholder_manager.py` | GPU placeholder 管理 |
| `docs/PDScope_opensource_refactor_report.md` | PDScope Algorithm 1 对齐参考 |
| `docs/算法原理_MoE_GPU卸载调度.md` | MoE GPU 卸载调度背景 |
| `docs/性能分析_Microbench与调优.md` | benchmark 与调度参数背景 |

---

## 九、记录时间

2026-05-24
