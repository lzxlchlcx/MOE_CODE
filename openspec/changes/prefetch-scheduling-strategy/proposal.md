## Why

当前 `40-myself/src/deepseek.py` 的 `HybridCPUGPUStrategy` 调度策略仅返回 CPU/GPU 专家列表，缺乏跨层预取能力。参考 PDScope 论文的 AdaptSched 调度器和 `20-PDScope/deepseek.py` 的预取实现，当前系统在 Decode 阶段存在 I/O 气泡——GPU 等待 CPU 完成计算时 PCIe 带宽空闲。通过引入 latency-aware 预取调度策略，利用 CPU/GPU/copy 延迟参数计算 I/O 气泡窗口，并在气泡期间并行预取下一层专家权重到 GPU placeholder，可以有效隐藏 PCIe 传输延迟。

## What Changes

- **新增预取调度策略类** `PrefetchHybridStrategy`：继承现有 `ExpertSchedulingStrategy`，在 `decide_and_prepare` 中除了返回 `cpu_experts`/`gpu_experts` 外，新增返回 `prefetch_experts` 列表（下一层需要预取的专家 ID 列表）
- **利用延迟参数做调度决策**：使用已有的 `latency_cpu`、`latency_copy`（即 `latencu_copy`）、`latency_gpu` 三个基准延迟值，计算当前层 CPU/GPU 执行时间差（I/O 气泡），在气泡窗口内安排预取
- **新增并行预取线程**：在 `mixtral_forward` 中，当专家执行线程运行时，同时启动预取线程将下一层专家权重提前加载到 GPU placeholder 中
- **修改 `decide_and_prepare` 返回值**：从 3-tuple 扩展为 4-tuple `(cpu_experts, gpu_experts, prefetch_experts, expert_assignments)`

## Capabilities

### New Capabilities
- `prefetch-scheduling`: 基于 latency 的专家预取调度策略，包含预取决策算法和并行预取执行框架

### Modified Capabilities
- `placeholder-management`: 需要支持跨层预取场景下的 placeholder 异步获取与释放（当前 `ExpertPlaceholderManager` 的 `acquire_placeholder` 和 `load_weights` 是同步的，需要增加异步预加载接口）

## Impact

- **受影响文件**: `40-myself/src/expert_scheduling.py`（新增策略类）、`40-myself/src/deepseek.py`（修改 `mixtral_forward` 调用逻辑）、`40-myself/src/placeholder_manager.py`（可能需要新增异步加载接口）
- **API 变更**: `decide_and_prepare` 返回值从 3-tuple 变为 4-tuple（**BREAKING**，需更新所有调用方）
- **依赖**: 使用已有的 `latency_cpu`/`latency_copy`/`latency_gpu` 基准数据，无新外部依赖
