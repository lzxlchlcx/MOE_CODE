## 1. 运行时数据结构

- [x] 1.1 新增统一类型定义，包含 `ExpertKey`、`ExpertDemand`、`ExpertSchedule`、`ExpertLayerRequest`、`ExpertLayerContext` 和专家 assignment 类型
- [x] 1.2 将当前层真实路由结果转换为 `ExpertDemand` 与 assignment 映射，保持现有 token 索引与 routing weight 行为不变
- [x] 1.3 为未来层预测结果提供统一的 demand 构建函数，支持 `(layer, expert_id)` 和 token count 聚合

## 2. 预测器重构

- [x] 2.1 更新 `ExpertPredictor` 接口，使其接收 `current_layer` 与 `lookahead` 参数
- [x] 2.2 更新 `GatePredictor`，使其返回带目标层信息的 `ExpertDemand` 列表
- [x] 2.3 保留兼容路径或适配层，确保 `deepseek.py` 可从新预测结果构建调度请求
- [x] 2.4 增加最后一层或越界 lookahead 返回空预测的测试或检查

## 3. 占位器与位置快照

- [x] 3.1 扩展占位符管理器，提供 `snapshot()` 方法返回 GPU、placeholder、loading、CPU、SSD 与空闲占位符状态
- [x] 3.2 增加 `is_on_gpu(layer, expert_id)` 查询，使静态常驻专家和 placeholder 专家都被视为 GPU 可用
- [x] 3.3 增加 loading 状态记录，避免 preload 与 ondemand 对同一专家重复加载
- [x] 3.4 确保 preload 无可用占位符时可以安全跳过，不影响当前层计算

## 4. 延迟模型与策略器

- [x] 4.1 新增 `ExpertLatencyModel`，封装 CPU、GPU compute 和 transfer 成本查询
- [x] 4.2 新增 `ExpertScheduler` 抽象接口，输入 request、placement snapshot 和 latency model，输出 `ExpertSchedule`
- [x] 4.3 实现 PDScope Prefill 调度路径，输出当前层 CPU/GPU 列表和未来层 preload 列表
- [x] 4.4 实现 PDScope Decode 调度路径，基于 `n_g^rho` 输出当前层 CPU/GPU 列表和 preload 列表
- [x] 4.5 移除 `PrefetchHybridStrategy` 中的 `2^n_active` 穷举路径，改为调用阶段感知调度逻辑
- [x] 4.6 保留 GPU-only 路径作为正确性基线

## 5. 执行器与线程管理

- [x] 5.1 新增 `ExpertExecutionManager`，统一执行 CPU、GPU 与 preload 任务
- [x] 5.2 将 `_execute_gpu_experts` 从 `deepseek.py` 移入执行器，并保持当前 GPU 输出合并语义
- [x] 5.3 将 `_execute_cpu_experts` 从 `deepseek.py` 移入执行器，并保持 dtype/device 转换语义
- [x] 5.4 将 `_prefetch_next_layer_experts` 从 `deepseek.py` 移入执行器或占位器协作路径
- [x] 5.5 明确线程同步顺序，确保 preload 失败或跳过不会影响当前层 CPU/GPU 输出

## 6. DeepSeek forward 集成

- [x] 6.1 在 `FiddlerDeepSeek.__init__` 中初始化 predictor、scheduler、placement manager、latency model 和 execution manager
- [x] 6.2 重写 MoE 层 forward 片段，使其只负责 gate、构建 request、调用 scheduler、调用 executor 和合并 shared expert 输出
- [x] 6.3 删除或收敛 `mixtral_forward` 中直接管理线程、preload 和 placeholder 的内联逻辑
- [x] 6.4 保持 `force_gpu` 或 `cpu_offload=0` 时走 GPU-only 基线

## 7. 验证与回归

- [x] 7.1 增加轻量单元测试或脚本，验证调度器返回的 CPU/GPU/preload 列表不重叠且层号正确
- [x] 7.2 增加占位器快照测试，验证 placeholder 中的专家会被识别为 GPU 可用
- [ ] 7.3 运行 DeepSeek 小输入推理，验证重构前后输出可用且无 dtype/device 错误
- [ ] 7.4 对比 GPU-only 与 CPU-offload 路径，确认调度重构不破坏基本推理流程
- [x] 7.5 记录 preload 请求数、成功数、跳过数和命中数，为后续性能调优提供数据
