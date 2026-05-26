## Context

当前 `mixtral_forward` 的执行流程是严格串行的：attention → gate → 专家执行。每一层的专家只有在 gate 计算完成后才知道哪些需要加载。对于 CPU offload 模式，权重搬运延迟直接叠加在推理延迟上。

关键观察：当前层的 attention 计算使用的是 post-layernorm 的 hidden states，而下一层的 gate 也接受相同维度的输入。如果能在当前层 attention 计算的同时，用 hidden_states 预测下一层的活跃专家，就能提前触发占位符加载，实现 overlap。

## Goals / Non-Goals

**Goals:**
- 定义可扩展的专家预测器抽象接口
- 实现基于下一层 gate 的预测策略（GatePredictor）
- 在 `mixtral_forward` 中与 attention 并行执行预测
- 预测结果供后续调度策略和占位符管理器使用

**Non-Goals:**
- 本次不实现基于预测结果的自动预取加载（仅产出预测结果）
- 不修改现有调度策略的决策逻辑
- 不修改占位符管理器

## Decisions

### 决策 1: 策略模式设计
**选择**: 定义 `ExpertPredictor` 抽象基类，`GatePredictor` 作为默认实现
**替代方案**: 直接在 `mixtral_forward` 中硬编码 gate 预测逻辑
**理由**:
- 未来可能添加基于历史统计、学习模型等预测策略
- 与项目现有的策略模式风格一致（`ExpertSchedulingStrategy`、`EvictionStrategy`）

### 决策 2: 预测接口签名
**选择**: `predict(hidden_states, model) -> (predicted_experts, routing_weights)`
**替代方案**: 只返回 expert_id 列表
**理由**:
- routing_weights 包含路由权重信息，可用于排序和优先级决策
- 与 `layer.mlp.gate(inps)` 的返回值格式对齐

### 决策 3: 与 attention 并行
**选择**: 使用 ThreadPoolExecutor 在 attention 计算线程之外并行执行预测
**替代方案**: 串行执行（先 attention 后预测）
**理由**:
- attention 是 GPU 计算，预测是轻量级 GPU 操作，可以流水线化
- 使用已有的 ThreadPoolExecutor 模式

### 决策 4: 预测时机
**选择**: 在 post_attention_layernorm 之后、gate 之前执行下一层预测
**替代方案**: 在层循环开始前预测
**理由**:
- post_attention_layernorm 的输出是下一层 gate 的近似输入
- 与 attention 并行时使用 layernorm 前的 hidden_states 作为输入（近似）

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| gate 预测有误差（用当前层输入预测下一层） | 误差可接受，仅用于预取优化而非精确调度 |
| 并行增加代码复杂度 | 保持预测逻辑独立封装，不影响主流程 |
| 额外 GPU 计算开销 | gate 计算很轻量（单层 MLP），开销可忽略 |
