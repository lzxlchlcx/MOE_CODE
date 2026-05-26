## Why

当前专家权重从 CPU 加载到 GPU 占用了推理延迟。通过在 attention 计算的同时预测下一层的活跃专家，可以提前触发权重搬运，实现计算与传输的 overlap，减少 decode 阶段的端到端延迟。

## What Changes

- 创建 `ExpertPredictor` 抽象基类，定义预测接口：输入 `(hidden_states, model)`，输出 `(predicted_experts, routing_weights)`
- 实现基于下一层 gate 的预测策略 `GatePredictor`：直接用下一层的 gate 网络对当前 hidden_states 做路由预测
- 在 `mixtral_forward` 的 MoE 层循环中，将预测与 attention 计算并行执行
- 预测结果在策略决策（`decide_and_prepare`）之前完成，供调度策略或占位符管理器使用

## Capabilities

### New Capabilities
- `expert-predictor`: 专家预测器抽象及实现，支持与 attention 并行预测下一层活跃专家

### Modified Capabilities

（无现有 spec 需要修改）

## Impact

- 新增文件 `40-myself/src/expert_predictor.py`
- 修改 `40-myself/src/deepseek.py` 的 `mixtral_forward` 方法，集成预测器调用
- 不影响现有调度策略逻辑
