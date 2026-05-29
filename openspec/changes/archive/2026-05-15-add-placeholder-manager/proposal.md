## Why

当前 `40-myself/src/deepseek.py` 中的占位符管理逻辑比较简单，只有单个 `expert_placeholder`，且占位符的管理逻辑直接耦合在主类中。随着需要引入更复杂的占位符管理（如多个占位符、预取机制等），需要将占位符管理抽象为独立的类，提高代码的可维护性和扩展性。

## What Changes

- 创建新的 `ExpertPlaceholderManager` 类，封装所有占位符管理逻辑
- 支持多个占位符的管理和分配
- 提供占位符加载、释放、状态查询等接口
- 引入淘汰策略抽象（`EvictionStrategy`），支持 LRU、FIFO 等算法，便于扩展
- 修改 `mDeepSeek` 类，使用新的占位符管理器替代原有的 `self.expert_placeholder`
- 更新相关策略类，适配新的占位符管理接口

## Capabilities

### New Capabilities
- `placeholder-management`: 专家占位符管理功能，包括多个占位符的创建、分配、加载权重、释放、淘汰等操作
- `eviction-strategy`: 占位符淘汰策略抽象，支持 LRU、FIFO 等算法，可通过策略模式扩展

### Modified Capabilities
（无现有功能需要修改）

## Impact

- 主要影响 `40-myself/src/deepseek.py`
- 可能影响 `40-myself/src/expert_scheduling.py` 中的策略类
- 保持 API 向后兼容
