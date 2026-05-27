## ADDED Requirements

### Requirement: 阶段感知调度计划
系统 MUST 根据当前阶段（Prefill 或 Decode）生成不同的专家调度计划。

#### Scenario: Prefill 阶段生成三类计划
- **WHEN** 当前阶段为 Prefill，且提供当前层专家需求与下一层预测需求
- **THEN** 系统 MUST 返回当前层 CPU 专家列表、当前层 GPU 专家列表以及下一层 preload 专家列表

#### Scenario: Decode 阶段生成负载均衡计划
- **WHEN** 当前阶段为 Decode，且提供当前层专家需求与下一层预测需求
- **THEN** 系统 MUST 返回符合 Decode 负载均衡目标的 CPU 专家列表、GPU 专家列表以及 preload 专家列表

### Requirement: 基于占位符快照的决策
系统 MUST 基于占位符快照判断专家是否已在 GPU 上，而不是直接依赖静态位置标记。

#### Scenario: 专家已加载到占位符
- **WHEN** 某专家已由占位符管理器加载到 GPU
- **THEN** 调度器 MUST 将该专家视为 GPU 可用专家

#### Scenario: 专家未加载到 GPU
- **WHEN** 某专家既不在静态 GPU 常驻集合，也未加载到占位符
- **THEN** 调度器 MUST 将该专家视为需要 CPU 或 preload 决策的候选项

### Requirement: 调度结果包含预取与回收
系统 MUST 在调度结果中显式表达 preload 与 eviction 计划。

#### Scenario: 预取下一层专家
- **WHEN** 调度器判断下一层存在可预取专家
- **THEN** 调度结果 MUST 包含该层需要 preload 的 `(layer, expert_id)` 列表

#### Scenario: 释放多余专家
- **WHEN** 调度器判断当前占位资源需要回收
- **THEN** 调度结果 MUST 包含需要 eviction 的 `(layer, expert_id)` 列表
