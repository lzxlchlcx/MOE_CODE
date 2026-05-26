## Why

`deepseek.py` 的 `mixtral_forward` 方法中，专家执行逻辑（第 411-533 行）是一个 120+ 行的 `if/else` 代码块，直接嵌套在前向传播的层循环里。该代码块包含两种调度策略（纯 GPU 执行 vs CPU-GPU 混合代价优化调度），逻辑复杂且高度耦合，导致难以阅读、测试和扩展新的调度策略。

## What Changes

- 将 `mixtral_forward` 中第 411-533 行的专家执行逻辑抽取为独立的策略类
- 策略负责：决策 + 张量预处理（expert_mask 构建、活跃专家收集、token 分配组织）
- `mixtral_forward` 负责：遍历策略返回的分配列表、调用专家执行、应用 routing weights、聚合结果
- 将"纯 GPU 执行"分支重构为独立的调度策略实现
- 将"CPU-GPU 混合代价优化"分支重构为独立的调度策略实现
- 公共逻辑（如 `expert_mask` 构建、活跃专家收集）提取为共享辅助函数供策略复用
- `mixtral_forward` 中的 MoE 层处理代码通过调用解耦后的策略接口获取分配信息

## Capabilities

### New Capabilities
- `expert-scheduling-strategy`: 专家调度策略接口与多种实现（GPU-only、CPU-GPU hybrid），策略负责决策和张量预处理，执行在 mixtral_forward 完成

### Modified Capabilities

## Impact

- 文件：`40-myself/src/deepseek.py` — 主要修改对象，`FiddlerDeepSeek` 类结构变更
- `mixtral_forward` 方法体简化，专家执行逻辑分为策略决策+前向执行两步
- 无 API 层面变更，`generate()` 等外部接口保持不变
- 无新增依赖
