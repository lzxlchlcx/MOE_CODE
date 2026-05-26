## ADDED Requirements

### Requirement: 专家调度策略接口
系统 SHALL 提供统一的专家调度策略接口 `ExpertSchedulingStrategy`，定义 `decide_and_prepare` 方法，接收层索引、专家模块列表、路由信息，返回 CPU/GPU 专家列表和每个专家的 token 分配信息。

#### Scenario: 策略接口被 mixtral_forward 调用
- **WHEN** `mixtral_forward` 处理 MoE 层（layer > 0）并需要调度专家
- **THEN** 通过当前调度策略对象的 `decide_and_prepare` 方法获取分配信息，`mixtral_forward` 不再包含内联的调度逻辑

### Requirement: 策略返回值格式
`decide_and_prepare` SHALL 返回一个元组 `(cpu_experts, gpu_experts, expert_assignments)`，其中：
- `cpu_experts`: CPU 上执行的专家索引列表
- `gpu_experts`: GPU 上执行的专家索引列表
- `expert_assignments`: 字典，键为专家索引，值为元组 `(top_2_tensor, routing_weight_subset)`

#### Scenario: 返回值被 mixtral_forward 使用
- **WHEN** `mixtral_forward` 从策略获取返回值
- **THEN** 遍历 `gpu_experts` 和 `cpu_experts`，从 `expert_assignments` 取对应分配信息，执行专家计算并聚合结果

### Requirement: 纯 GPU 调度策略
系统 SHALL 提供 `GPUOnlyStrategy` 策略实现，将所有专家分配到 GPU 上执行。

#### Scenario: 所有专家分配到 GPU
- **WHEN** 使用 `GPUOnlyStrategy`
- **THEN** 返回的 `cpu_experts` 为空，`gpu_experts` 包含所有有 token 分配的专家
- **AND** `expert_assignments` 包含所有活跃专家的 token 分配信息

### Requirement: CPU-GPU 混合代价优化调度策略
系统 SHALL 提供 `HybridCPUGPUStrategy` 策略实现，对每个 MoE 层的活跃专家构建代价表，通过穷举 2^n_active 种配置找到最低代价的 CPU/GPU 分配方案。

#### Scenario: 代价最优的 CPU/GPU 分配
- **WHEN** 使用 `HybridCPUGPUStrategy` 且某层有 n_active 个活跃专家
- **THEN** 对每个活跃专家计算 `cost_cpu = token_count * latency_cpu` 和 `cost_gpu = latency_gpu`（已驻留 GPU 的专家 cost_gpu=0），穷举所有 2^n_active 种分配方案选择总代价最低的配置

#### Scenario: 专家命中率统计
- **WHEN** `HybridCPUGPUStrategy` 执行专家调度
- **THEN** 对已驻留 GPU 的专家统计其处理的 token 数到 `cnt_expert_hit`，所有活跃专家的 token 数统计到 `cnt_expert_all`

### Requirement: 公共辅助函数提取
系统 SHALL 提供以下模块级辅助函数，供策略实现复用：
- `_build_expert_mask`: 从 `selected_experts` 构建 one-hot expert mask
- `_collect_active_experts`: 从 expert mask 中收集有 token 分配的专家索引
- `_organize_token_assignments`: 组织每个专家的 token 分配和路由权重

#### Scenario: 策略实现复用辅助函数
- **WHEN** 任意策略实现需要构建 expert mask 或收集活跃专家
- **THEN** 调用对应的辅助函数完成操作，不重复实现相同逻辑

### Requirement: 策略选择与注入
系统 SHALL 在 `FiddlerDeepSeek.__init__` 中根据 `args.cpu_offload` 参数选择并实例化对应的调度策略，存储为 `self.expert_strategy`。`cpu_offload == 0` 时使用 `GPUOnlyStrategy`，否则使用 `HybridCPUGPUStrategy`。prefill 阶段（`force_gpu=True`）时 SHALL 统一使用 `GPUOnlyStrategy`。

#### Scenario: cpu_offload 为 0 时选择纯 GPU 策略
- **WHEN** `args.cpu_offload == 0`
- **THEN** `self.expert_strategy` 为 `GPUOnlyStrategy` 实例

#### Scenario: cpu_offload 非 0 时选择混合策略
- **WHEN** `args.cpu_offload != 0`
- **THEN** `self.expert_strategy` 为 `HybridCPUGPUStrategy` 实例，但 prefill 阶段仍使用 `GPUOnlyStrategy`

### Requirement: mixtral_forward 执行逻辑
`mixtral_forward` SHALL 从策略获取分配信息后，负责：
1. 遍历 GPU 专家：对驻留 GPU 的专家直接调用，否则通过 `expert_placeholder` 执行
2. 遍历 CPU 专家：调用 `run_expert_at_cpu` 执行
3. 对每个专家的结果乘以对应的 `routing_weights`
4. 通过 `index_add_` 聚合所有专家的结果

#### Scenario: mixtral_forward 执行专家计算
- **WHEN** `mixtral_forward` 获取策略返回的分配信息
- **THEN** 按照分配列表在对应设备上执行专家计算，应用权重并聚合结果

### Requirement: 行为等价性
重构后系统的前向传播数值输出 SHALL 与重构前完全一致（在浮点精度范围内），`generate` 接口的输入输出行为不变。

#### Scenario: 数值输出一致
- **WHEN** 使用相同的模型权重、输入 tokens 和调度配置运行重构前后的代码
- **THEN** 每层输出的 hidden states 在 `torch.allclose(atol=1e-5)` 意义下一致
