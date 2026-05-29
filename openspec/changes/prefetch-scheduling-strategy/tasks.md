## 1. PlaceholderManager 线程安全改造

- [x] 1.1 在 `ExpertPlaceholderManager.__init__` 中添加 `self._lock = threading.Lock()`
- [x] 1.2 在 `acquire_placeholder`、`load_weights`、`release_placeholder`、`release_by_layer`、`get_placeholder_for_expert` 方法中添加 `with self._lock` 保护
- [ ] 1.3 验证现有测试（如有）或手动验证 placeholder 分配/释放功能正常

## 2. PrefetchHybridStrategy 实现

- [x] 2.1 在 `expert_scheduling.py` 中新增 `PrefetchHybridStrategy` 类，继承 `ExpertSchedulingStrategy`
- [x] 2.2 实现 `__init__` 方法，接收 `dev`、`is_expert_in_gpu`、`latency_cpu`、`latency_copy`、`latency_gpu` 参数
- [x] 2.3 实现 `decide_and_prepare` 方法的 CPU/GPU 调度逻辑（复用 `HybridCPUGPUStrategy` 的穷举代价模型）
- [x] 2.4 在 `decide_and_prepare` 中新增 `predicted_next_experts` 可选参数，默认 None
- [x] 2.5 实现 I/O 气泡计算逻辑：`T_cpu = len(cpu_experts) * latency_cpu`，`T_gpu = len(gpu_experts) * latency_gpu + len(gpu_experts_not_in_gpu) * latency_copy`，`T_bubble = max(0, T_cpu - T_gpu)`
- [x] 2.6 实现预取专家选择逻辑：`n_prefetch = floor(T_bubble / latency_copy)`，从 `predicted_next_experts` 中选取非 GPU 常驻的前 `n_prefetch` 个专家
- [x] 2.7 返回 4-tuple `(cpu_experts, gpu_experts, prefetch_experts, expert_assignments)`

## 3. deepseek.py 调用层改造

- [x] 3.1 修改 `__init__` 中的策略初始化：当 `cpu_offload != 0` 时使用 `PrefetchHybridStrategy` 替代 `HybridCPUGPUStrategy`
- [x] 3.2 修改 `mixtral_forward` 中 `decide_and_prepare` 的调用，传入 `predicted_next_experts` 参数（来自 `GatePredictor` 预测结果）
- [x] 3.3 修改 `decide_and_prepare` 返回值解包，从 3-tuple 改为 4-tuple
- [x] 3.4 在 `mixtral_forward` 中新增 `_prefetch_next_layer_experts` 方法：遍历 `prefetch_experts`，通过 `placeholder_manager.acquire_placeholder` + `load_weights` 预加载下一层专家权重
- [x] 3.5 修改 `ThreadPoolExecutor` 并行执行逻辑：从 2 个 worker（GPU+CPU）扩展为 3 个 worker（GPU+CPU+Prefetch），仅当 `prefetch_experts` 非空时启动预取线程
- [x] 3.6 更新 `_execute_gpu_experts` 和 `_execute_cpu_experts` 的 result 收集逻辑，兼容 3 线程并行

## 4. 验证与调试

- [x] 4.1 运行现有的 `40-myself/src/infer_deepseek.py` 验证模型推理功能正常
- [x] 4.2 添加日志输出预取命中情况：打印每层预取的专家列表和预取耗时
- [x] 4.3 对比 `PrefetchHybridStrategy` 和 `HybridCPUGPUStrategy` 的 Decode 阶段延迟数据
