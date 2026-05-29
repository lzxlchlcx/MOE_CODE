## ADDED Requirements

### Requirement: 专家预测器抽象接口
系统应提供 `ExpertPredictor` 抽象基类，定义统一的预测接口。

#### Scenario: 调用预测方法
- **WHEN** 调用 `predictor.predict(hidden_states, model)`
- **THEN** 系统返回 `(predicted_experts, routing_weights)` 元组，其中 `predicted_experts` 为预测的专家索引张量，`routing_weights` 为对应的路由权重

### Requirement: GatePredictor 实现
系统应提供基于下一层 gate 网络的预测策略 `GatePredictor`。

#### Scenario: 使用下一层 gate 预测
- **WHEN** 调用 `GatePredictor.predict(hidden_states, model)` 且指定了 `next_layer_idx`
- **THEN** 系统使用 `model.layers[next_layer_idx].mlp.gate(hidden_states)` 进行预测，返回该 gate 的输出

#### Scenario: 最后一层无需预测
- **WHEN** `next_layer_idx` 超出模型层数范围
- **THEN** 系统返回空预测结果

### Requirement: 与 attention 并行执行
系统应支持预测与 attention 计算并行执行。

#### Scenario: 并行预测下一层专家
- **WHEN** 当前层执行 attention 计算
- **THEN** 系统同时使用 hidden_states 预测下一层的活跃专家

#### Scenario: 预测结果在策略决策前可用
- **WHEN** attention 计算完成
- **THEN** 下一层的预测结果应已就绪，可供调度策略使用

### Requirement: 预测器可替换
系统应支持通过配置选择不同的预测策略。

#### Scenario: 使用自定义预测器
- **WHEN** 实现继承自 `ExpertPredictor` 的自定义预测器并传入 mDeepSeek
- **THEN** 系统使用自定义预测器进行专家预测
