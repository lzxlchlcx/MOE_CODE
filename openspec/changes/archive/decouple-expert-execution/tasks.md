## 1. 公共辅助函数提取

- [x] 1.1 实现 `_build_expert_mask` 辅助函数
- [x] 1.2 实现 `_collect_active_experts` 辅助函数
- [x] 1.3 实现 `_organize_token_assignments` 辅助函数

## 2. 调度策略接口与基类

- [x] 2.1 定义 `ExpertSchedulingStrategy` 抽象基类与 `decide_and_prepare` 接口
- [x] 2.2 在 `FiddlerDeepSeek.__init__` 中准备策略上下文（`expert_placeholder`、`dev`、`is_expert_in_gpu` 方法引用等）

## 3. GPUOnlyStrategy 实现

- [x] 3.1 实现 `GPUOnlyStrategy` 类，包含所有必要的状态注入
- [x] 3.2 实现 `GPUOnlyStrategy.decide_and_prepare`，复用辅助函数收集活跃专家并全部分配到 GPU

## 4. HybridCPUGPUStrategy 实现

- [x] 4.1 实现 `HybridCPUGPUStrategy` 类，包含代价表构建所需的状态（`latency_cpu`、`latency_gpu`、`cnt_expert_hit`、`cnt_expert_all` 等）
- [x] 4.2 实现活跃专家收集与代价表构建
- [x] 4.3 实现穷举搜索最优配置（2^n_active）
- [x] 4.4 实现 `decide_and_prepare` 返回 CPU/GPU 分配列表和 token 分配信息

## 5. FiddlerDeepSeek 集成与解耦

- [x] 5.1 在 `__init__` 中根据 `args.cpu_offload` 选择并实例化策略对象
- [x] 5.2 修改 `mixtral_forward`，删除第 411-533 行的内联专家执行代码
- [x] 5.3 在 `mixtral_forward` 中，调用策略的 `decide_and_prepare` 方法获取分配信息
- [x] 5.4 在 `mixtral_forward` 中实现专家执行逻辑：遍历 GPU/CPU 专家列表，调用专家，应用权重，聚合结果
- [x] 5.5 prefill 阶段（`force_gpu=True`）时确保使用 `GPUOnlyStrategy`

## 6. 验证与测试

- [x] 6.1 运行现有代码路径，确保功能正常，输出文本质量不变
- [x] 6.2 在调试模式下对比重构前后同一层输出的 hidden states，验证数值一致性
- [x] 6.3 检查专家命中率统计、显存占用与性能指标无退化
