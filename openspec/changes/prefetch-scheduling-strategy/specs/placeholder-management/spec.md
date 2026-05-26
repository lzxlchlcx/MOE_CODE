## MODIFIED Requirements

### Requirement: 占位符分配
系统应支持分配可用的占位符给指定的专家，并在并发访问时保证线程安全。

#### Scenario: 分配空闲占位符
- **WHEN** 调用 `acquire_placeholder(layer_id, expert_id)` 且有空闲占位符
- **THEN** 系统分配一个占位符并记录该占位符当前服务的 (layer_id, expert_id)

#### Scenario: 无空闲占位符
- **WHEN** 调用 `acquire_placeholder(layer_id, expert_id)` 且无空闲占位符
- **THEN** 系统根据淘汰策略选择一个已占用的占位符进行淘汰，返回该占位符

#### Scenario: 无空闲占位符且禁用淘汰
- **WHEN** 调用 `acquire_placeholder(layer_id, expert_id)` 且无空闲占位符且淘汰策略为 None
- **THEN** 系统返回 None

#### Scenario: 并发分配占位符
- **WHEN** 多个线程同时调用 `acquire_placeholder`
- **THEN** 系统通过内部锁保证同一占位符不会被重复分配给不同的专家

## ADDED Requirements

### Requirement: 占位符管理器线程安全
`ExpertPlaceholderManager` 的所有公共方法 SHALL 使用 `threading.Lock` 保护内部状态（`_available`、`_occupied`、`_reverse_map`），以支持预取线程和执行线程的并发访问。

#### Scenario: 预取线程与执行线程并发访问
- **WHEN** 预取线程为 `layer+1` 调用 `acquire_placeholder` 的同时，执行线程为 `layer` 调用 `release_by_layer`
- **THEN** 系统通过锁保证两个操作互斥执行，不会出现数据竞争
