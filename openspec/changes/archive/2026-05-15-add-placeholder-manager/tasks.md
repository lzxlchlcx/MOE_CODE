## 1. 创建淘汰策略抽象和实现

- [x] 1.1 在 40-myself/src/ 下创建 eviction_strategy.py，定义 EvictionStrategy 抽象基类
- [x] 1.2 实现 LRUEvictionStrategy（最近最少使用）
- [x] 1.3 实现 FIFOEvictionStrategy（先进先出）

## 2. 创建 ExpertPlaceholderManager 类

- [x] 2.1 在 40-myself/src/ 下创建新文件 placeholder_manager.py
- [x] 2.2 实现 ExpertPlaceholderManager 类的初始化方法（支持传入淘汰策略）
- [x] 2.3 实现占位符分配方法（无空闲时触发淘汰策略）
- [x] 2.4 实现权重加载方法（更新淘汰策略的访问记录）
- [x] 2.5 实现占位符释放和按层释放方法
- [x] 2.6 实现状态查询方法

## 3. 集成到 FiddlerDeepSeek 类

- [x] 3.1 在 deepseek.py 中导入 ExpertPlaceholderManager 和淘汰策略
- [x] 3.2 在 __init__ 方法中初始化占位符管理器（配置淘汰策略）
- [x] 3.3 更新 _execute_gpu_experts 方法使用管理器

## 4. 更新策略类

- [x] 4.1 修改 expert_scheduling.py 中的策略类，适配新接口
- [x] 4.2 确保向后兼容性

## 5. 测试验证

- [x] 5.1 测试基本推理功能是否正常
- [x] 5.2 验证 LRU 淘汰逻辑正确性
- [x] 5.3 验证 FIFO 淘汰逻辑正确性
- [x] 5.4 检查性能无明显回归
