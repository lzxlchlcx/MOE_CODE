## ADDED Requirements

### Requirement: 预取调度策略初始化
系统 SHALL 支持 `PrefetchHybridStrategy` 类，接收延迟查找表和固定参数进行初始化。

初始化参数：
- `dev`: 计算设备
- `is_expert_in_gpu`: 判断专家是否 GPU 常驻的回调函数
- `t_io`: 专家权重 CPU→GPU PCIe 传输时间（固定值），对应 benchmark JSON 的 `expert_weight_copy.avg_ms`（约 1.67ms）
- `latency_cpu_table`: CPU 专家计算延迟查找表（dict: token_count → time_ms），对应 benchmark JSON 的 `expert_cpu[]`，按 `token_count` 索引（1→0.142ms, 8→0.754ms, 128→11.58ms）
- `latency_gpu_table`: GPU 专家计算延迟查找表（dict: token_count → time_ms），对应 benchmark JSON 的 `expert_gpu[]`，按 `token_count` 索引（1→0.093ms, 8→0.092ms, 128→0.118ms）

#### Scenario: 初始化预取调度策略
- **WHEN** 创建 `PrefetchHybridStrategy` 实例，传入延迟查找表
- **THEN** 系统保存 `t_io`、`latency_cpu_table`、`latency_gpu_table`，并初始化命中统计计数器 `cnt_expert_hit` 和 `cnt_expert_all`

#### Scenario: 延迟查表按 token 数索引
- **WHEN** 调度算法需要某个专家的 CPU/GPU 计算延迟
- **THEN** 系统根据该专家的 token 数从查找表中获取对应延迟值，超出表范围的 token 数使用表中最后一个条目

### Requirement: 调度决策输入
`decide_and_prepare` 方法 SHALL 接收当前层专家信息、下一层预测专家信息、阶段标志和延迟参数，统一入口处理 Prefill 和 Decode 两个阶段。

#### Scenario: 传入完整调度参数
- **WHEN** 调用 `decide_and_prepare` 时传入 `i_layer`、`experts`、`selected_experts`、`routing_weights`、`n_expert`、`predicted_next_experts`、`predicted_next_weights`、`is_prefill`
- **THEN** 系统根据 `is_prefill` 标志分别走 Prefill 三步调度法或 Decode ABC 策略

#### Scenario: 未传入下一层预测专家
- **WHEN** `predicted_next_experts` 为 None
- **THEN** 系统退化为无预取的 CPU/GPU 混合调度（`prefetch_experts` 返回空列表）

### Requirement: 调度决策输出
`decide_and_prepare` 方法 SHALL 返回 4-tuple `(cpu_experts, gpu_experts, prefetch_experts, expert_assignments)`。

#### Scenario: 输出包含预取列表
- **WHEN** 调度决策完成
- **THEN** 返回值中 `prefetch_experts` 是下一层 `(i_layer+1)` 需要预取到 GPU placeholder 的专家 ID 列表，列表中的专家 SHALL 不在目标层 GPU 常驻集合中

### Requirement: Prefill 阶段三步调度法
在 Prefill 阶段（`is_prefill=True`），系统 SHALL 执行 PDScope AdaptSched 的三步调度：全局排位 → 局部重排 → 置信度感知预取。

#### Scenario: Step 1 全局排位得到 L_global
- **WHEN** 将当前层 L 和预测的下层 L+1 的非 GPU 常驻专家放入同一个池子 `E_all`，按每个专家处理的 token 数量升序排列
- **THEN** 对每个候选边界 i 计算全局 GPU 总成本 `T_all^G = alpha + (n + n' - i + 1) * t_io + t_g` 和全局 CPU 总成本 `T_all^C = sum(t_c(tokens) for E_all[0:i]) + t_attn`，其中 `t_c(tokens)` 和 `t_g(tokens)` 通过延迟查找表按 token 数索引，`t_io` 为固定值。当 `T_all^G < T_all^C` 时边界右侧专家入选 `L_global`

#### Scenario: Step 2 局部重排得到 L_on
- **WHEN** 从 `L_global` 中筛选仅属于当前层的专家
- **THEN** 寻找最小边界 i' 使局部 GPU 成本 `T_G = max[n_g * t_g(tokens), alpha + (n - i' + 1) * t_io] + t_g(tokens)` 小于局部 CPU 成本 `T_C = sum(t_c(tokens) for E_cur[0:i'-1])`，边界右侧的当前层专家构成按需加载集合 `L_on`

#### Scenario: Step 3 置信度感知预取
- **WHEN** `L_on` 确定后，计算 I/O 气泡 `T_gap = T_C - T_G`
- **THEN** 计算气泡可容纳的预取数量 `f = floor((T_gap + t_attn) / t_io)`，计算期望效用 `xi = R_hit * (f - |f| + 1) * t_io - (1 - R_hit) * (|f| - f) * t_io`，取 R_hit 为0.8，仅当 `xi > 0` 时从 `L_global` 中剩余的下层专家（非当前层）选入 `prefetch_experts`

### Requirement: Decode 阶段 ABC 策略
在 Decode 阶段（`is_prefill=False`），系统 SHALL 执行显式负载均衡，目标是使 GPU 和 CPU 同时完工。

#### Scenario: 计算最优 GPU 专家数 n_g^rho
- **WHEN** 进入 Decode 调度，Decode 阶段每个专家只处理 1 个 token
- **THEN** 使用 `t_c(1)` 和 `t_g(1)` 从查找表取值，计算 `n_g^rho = argmin max(n_g^rho * t_g(1), (k - n_g^rho) * t_c(1))`，其中 k 为激活专家总数

#### Scenario: Mode A 保当前层
- **WHEN** 当前层 GPU 驻留专家数 < `n_g^rho` 且下一层 >= `n_g^rho`
- **THEN** 暂停预取，将所有带宽用于当前层按需加载，`prefetch_experts` 为空

#### Scenario: Mode B 保下一层预取
- **WHEN** 当前层 GPU 驻留专家数 > `n_g^rho` 且下一层 < `n_g^rho`
- **THEN** 当前层多余专家交给 CPU 执行，释放 GPU 资源和 PCIe 带宽用于预取下一层专家，`prefetch_experts` 包含下一层需要预取的专家

#### Scenario: Mode C 双超标清理
- **WHEN** 当前层和下一层 GPU 驻留专家数都 > `n_g^rho`
- **THEN** 两层都只卸载不预取，`prefetch_experts` 为空

#### Scenario: 两层都低于配额时降级
- **WHEN** 当前层和下一层 GPU 驻留专家数都 < `n_g^rho`
- **THEN** 降级回 Prefill 的三步调度逻辑计算预取列表

### Requirement: GPU 常驻专家跳过
在所有调度阶段，已被标记为 GPU 常驻的专家 SHALL 不出现在 `cpu_experts` 或 `prefetch_experts` 中。

#### Scenario: GPU 常驻专家不参与调度
- **WHEN** 专家已标记为 GPU 常驻（`is_expert_in_gpu(i_layer, i_expert)` 返回 True）
- **THEN** 系统将其 GPU 代价设为 0，直接加入 `gpu_experts` 列表，不参与预取计算
