## MODIFIED Requirements

### Requirement: 专家预测器抽象接口
系统应提供 `ExpertPredictor` 抽象基类，定义统一的预测接口。

#### Scenario: 调用预测方法
- **WHEN** 调用 `predictor.predict(hidden_states, model, current_layer, lookahead)`
- **THEN** 系统返回候选专家需求列表，列表中的每个元素 MUST 包含目标层 `layer` 和 `expert_id`

### Requirement: GatePredictor 实现
系统应提供基于下一层 gate 网络的预测策略 `GatePredictor`。

#### Scenario: 使用下一层 gate 预测
- **WHEN** 调用 `GatePredictor.predict(hidden_states, model, current_layer, lookahead)` 且目标层存在
- **THEN** 系统使用 `model.layers[current_layer + lookahead].mlp.gate(hidden_states)` 进行预测，并返回带层号的候选专家需求

#### Scenario: 最后一层无需预测
- **WHEN** 目标层超出模型层数范围
- **THEN** 系统返回空预测结果

### Requirement: 与 attention 并行执行
系统应支持预测与 attention 计算并行执行。

#### Scenario: 并行预测下一层专家
- **WHEN** 当前层执行 attention 计算
- **THEN** 系统同时使用 hidden_states 预测未来层的活跃专家

#### Scenario: 预测结果在策略决策前可用
- **WHEN** attention 计算完成
- **THEN** 下一层或未来多层的预测结果应已就绪，可供调度策略使用

### Requirement: 预测器可替换
系统应支持通过配置选择不同的预测策略。

#### Scenario: 使用自定义预测器
- **WHEN** 实现继承自 `ExpertPredictor` 的自定义预测器并传入 mDeepSeek
- **THEN** 系统使用自定义预测器进行专家预测
