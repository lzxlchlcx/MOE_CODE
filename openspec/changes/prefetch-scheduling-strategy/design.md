## Context

`40-myself/src/deepseek.py` 是当前项目自研的 MoE 推理引擎，使用策略模式 (`ExpertSchedulingStrategy`) 分离调度逻辑。当前 `HybridCPUGPUStrategy` 通过穷举 2^n_active 种配置找到最小代价的 CPU/GPU 分配方案，但仅返回当前层调度结果，缺乏跨层预取能力。

参考实现 `20-PDScope/deepseek.py` 已实现三层并行模型（GPU 专家执行 + CPU 专家执行 + 预取线程），其关键机制：
1. 利用下一层 gate 网络预测下一层活跃专家
2. 根据当前层 CPU/GPU 执行时间差计算 I/O 气泡
3. 在气泡窗口内通过独立线程将下一层专家权重加载到 GPU placeholder

当前代码已有 `ExpertPredictor`（GatePredictor）和 `ExpertPlaceholderManager`（12 个 placeholder），基础设施完备。

## Goals / Non-Goals

**Goals:**
- 新增 `PrefetchHybridStrategy`，在 `decide_and_prepare` 中计算并返回预取专家列表
- 在 `mixtral_forward` 中实现预取线程与 `_execute_gpu_experts`/`_execute_cpu_experts` 并行执行
- 利用 `latency_cpu`/`latency_copy`/`latency_gpu` 三个延迟参数做调度决策

**Non-Goals:**
- 不实现 CUDA Graph 静态计算图优化
- 不实现 CUDA Stream 多流并发传输

## Decisions

### D1: `decide_and_prepare` 返回 4-tuple

**决策**: 扩展返回值为 `(cpu_experts, gpu_experts, prefetch_experts, expert_assignments)`

**理由**: 最小改动原则。在现有 3-tuple 基础上增加 `prefetch_experts` 列表，而非引入新的调度接口。`prefetch_experts` 是 `List[int]`，表示下一层需要预取到 GPU placeholder 的专家 ID。

**替代方案**: 引入独立的 `PrefetchDecider` 类。被否决——增加了不必要的类间协调复杂度，当前策略类已有延迟参数和专家信息，直接计算更内聚。

### D2: 预取决策算法——PDScope AdaptSched 完整调度

**决策**: 实现完整的 PDScope AdaptSched 调度算法，区分 Prefill 和 Decode 两个阶段：

**Prefill 阶段（三步调度法）：**

1. **Step 1 全局排位**：将当前层 L 和预测的下层 L+1 的非 GPU 常驻专家放入池子 `E_all`，按 token 数升序排列。对每个候选边界 i 计算全局 GPU 成本 `T_all^G` 和 CPU 成本 `T_all^C`，选出全局队列 `L_global`。

2. **Step 2 局部重排**：在 `L_global` 中仅筛选当前层专家，寻找最小边界 i' 使局部 GPU 成本 `T_G` < 局部 CPU 成本 `T_C`，得到按需加载集合 `L_on`。

3. **Step 3 置信度感知预取**：利用 `L_on` 确定后的 I/O 气泡 `T_gap = T_C - T_G`，计算可容纳预取数量 `f`，通过期望效用 `xi` 决定是否预取。

**Decode 阶段（ABC 策略）：**

每个激活专家只处理 1 个 Token，调度变为纯负载均衡。计算最优 GPU 专家数 `n_g^rho`，根据当前层和下一层的 GPU 驻留专家数与 `n_g^rho` 的关系触发三种模式：
- **Mode A**：当前层 < `n_g^rho`，暂停预取保当前层
- **Mode B**：当前层 > `n_g^rho`，释放资源预取下层
- **Mode C**：两层都 > `n_g^rho`，只卸载不预取
- 两层都 < `n_g^rho` 时降级回 Prefill 三步逻辑

**理由**: 完整实现 AdaptSched 可以最大化跨层预取收益。延迟参数使用 benchmark JSON 的查找表模型：

| AdaptSched 符号 | 含义 | JSON 数据来源 | 特征 |
|---|---|---|---|
| `t_io` | 专家权重 CPU→GPU PCIe 传输 | `expert_weight_copy.avg_ms` ≈ 1.67ms | 固定值，与 token 数无关 |
| `t_c(tokens)` | CPU 计算一个专家 | `expert_cpu[].avg_time_ms`（按 `token_count` 索引） | 随 token 数线性增长（1→0.14ms, 128→11.58ms） |
| `t_g(tokens)` | GPU 计算一个专家 | `expert_gpu[].avg_time_ms`（按 `token_count` 索引） | 基本恒定（1→0.093ms, 128→0.118ms） |

**与现有代码的差异**：当前 `HybridCPUGPUStrategy` 仅使用 `expert_cpu[0]`（1 token）和 `expert_gpu[0]`（1 token）的标量值。`PrefetchHybridStrategy` 需要加载完整的 `expert_cpu[]` 和 `expert_gpu[]` 查找表，因为 Prefill 阶段不同专家处理的 token 数差异大（1~128），`t_c` 差异高达 80 倍。

**替代方案**: 简化版 I/O 气泡模型（`n_prefetch = floor(T_bubble / latency_copy)`）。被否决——过于粗糙，无法区分 Prefill/Decode 阶段的不同优化目标，也不具备 ABC 策略的负载均衡能力。

### D3: 预取执行——独立线程并行

**决策**: 在 `mixtral_forward` 中，使用 `ThreadPoolExecutor` 同时启动三个任务：`_execute_gpu_experts`、`_execute_cpu_experts`、`_prefetch_next_layer_experts`。

**理由**: 与 `20-PDScope` 的实现一致（`threading.Thread` 并行执行 GPU/CPU/预取）。`_prefetch_next_layer_experts` 通过 `ExpertPlaceholderManager.acquire_placeholder` + `load_weights` 将下一层专家权重提前加载到 placeholder。

### D4: placeholder_manager 扩展——保持同步接口

**决策**: 不修改 `ExpertPlaceholderManager` 接口。预取线程直接调用现有的 `acquire_placeholder` + `load_weights`，这些操作是线程安全的（因为预取线程和执行线程操作不同层的 placeholder）。

**理由**: `release_by_layer` 已按层释放，预取操作针对 `i_layer+1`，当前执行操作针对 `i_layer`，不会冲突。保持同步接口简化了实现和调试。

## Risks / Trade-offs

- **[线程安全]** 预取线程与执行线程并发访问 `ExpertPlaceholderManager` → `ExpertPlaceholderManager` 的 `_available`/`_occupied`/`_reverse_map` 非线程安全，预取线程的 acquire/release 可能与执行线程冲突 → 需要在 `ExpertPlaceholderManager` 中添加 `threading.Lock` 保护
- **[预取命中率]** GatePredictor 预测的下一层专家可能不准确 → 预取了错误的专家浪费 PCIe 带宽和 placeholder → 最差情况退化为无预取，性能不低于当前基线
- **[placeholder 不足]** 12 个 placeholder 可能不够同时容纳当前层 ondemand 加载和下一层预取 → 预取操作优先级低于当前层执行，当 placeholder 不足时放弃预取
