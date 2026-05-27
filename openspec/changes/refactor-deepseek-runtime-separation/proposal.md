## Why

当前 `src/model/deepseek.py` 将预测、调度、占位符管理、CPU/GPU 执行和线程同步全部耦合在 `forward` 流程中，导致算法难以替换、调度逻辑难以验证，也无法按 PDScope 思路清晰落地阶段感知的 Prefill/Decode 调度。现在需要把运行时拆成可组合模块，以支持更准确的策略迭代和更可靠的并行执行。

## What Changes

- **BREAKING** 重构 DeepSeek MoE 运行时调用链，将 `forward` 中的预测、策略、占位符、执行和同步逻辑拆分到独立模块。
- 引入统一的预测器接口，支持输出下一层或未来多层的候选专家需求，且结果携带 `(layer, expert_id)` 信息。
- 引入统一的策略器接口，基于当前层真实专家、未来层预测专家和占位符快照，计算 CPU、GPU、preload 和 eviction 计划。
- 扩展占位符管理能力，使其同时反映 GPU、CPU、SSD、loading 与 placeholder 的状态，并支持预取与回收。
- 引入线程管理器，统一负责 preload、CPU 计算、GPU 计算与结果合并，保证并发执行正确性。
- 规范运行时数据结构，减少 `deepseek.py` 中的临时状态和散乱分支。

## Capabilities

### New Capabilities
- `expert-scheduling`: 提供阶段感知的 MoE 调度能力，按 Prefill/Decode 输出 CPU、GPU、preload 与 eviction 计划。
- `expert-execution`: 提供统一的线程管理与并行执行能力，负责 preload、CPU、GPU 计算和结果同步。

### Modified Capabilities
- `expert-predictor`: 预测结果从“仅返回下一层 gate 输出”扩展为“返回带层号的候选专家需求”，并支持未来多层 lookahead。
- `placeholder-management`: 从“只管理 GPU 占位符”扩展为“管理 GPU/CPU/SSD/loading/placeholder 状态，并支持预取、回收和快照查询”。

## Impact

- 影响 `src/model/deepseek.py` 的 MoE forward 流程，需要把当前内联逻辑迁移到独立组件。
- 影响 `src/expert_predictor.py`、`src/expert_scheduling.py`、`src/placeholder_manager.py`，需要调整接口和职责边界。
- 需要新增执行与调度相关模块，统一专家需求、调度计划和执行上下文的数据结构。
- 需要更新与深度学习推理、预取和线程同步相关的测试与调试日志输出。
