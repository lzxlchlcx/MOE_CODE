## Why

当前 deepseek.py 的 444-467 行中，GPU 专家和 CPU 专家是串行执行的，无法利用 CPU 和 GPU 的并行能力，导致推理延迟较高。通过将两者抽象为并行线程，可以同时执行，显著提升性能。

## What Changes

- 将 mixtral_forward 中 GPU 和 CPU 专家的串行执行重构为并行执行
- 使用 Python threading 或 concurrent.futures 实现并行
- 两个线程分别处理 GPU 和 CPU 专家，最后合并结果到 inps_after_experts
- 保持现有的路由权重、专家调度、显存管理逻辑不变

## Capabilities

### New Capabilities
- `parallel-expert-execution`: CPU 和 GPU 专家并行执行，提升推理吞吐量和延迟

### Modified Capabilities

## Impact

- 文件：40-myself/src/deepseek.py - 主要修改 mixtral_forward 中的专家执行部分
- 新增导入：threading 或 concurrent.futures
- 无 API 层面变更，generate 等外部接口保持不变
