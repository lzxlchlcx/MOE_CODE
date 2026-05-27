## MODIFIED Requirements

### Requirement: 占位符管理器初始化
系统应支持根据模板专家创建多个占位符，并初始化在指定设备上。

#### Scenario: 初始化占位符管理器
- **WHEN** 创建 ExpertPlaceholderManager 实例，传入模板专家、设备和占位符数量
- **THEN** 系统创建指定数量的占位符副本，并都移动到指定设备

### Requirement: 占位符分配
系统应支持分配可用的占位符给指定的专家。

#### Scenario: 分配空闲占位符
- **WHEN** 调用 `acquire_placeholder(layer_id, expert_id)` 且有空闲占位符
- **THEN** 系统分配一个占位符并记录该占位符当前服务的 `(layer_id, expert_id)`

#### Scenario: 无空闲占位符
- **WHEN** 调用 `acquire_placeholder(layer_id, expert_id)` 且无空闲占位符
- **THEN** 系统根据淘汰策略选择一个已占用的占位符进行淘汰，返回该占位符

#### Scenario: 无空闲占位符且禁用淘汰
- **WHEN** 调用 `acquire_placeholder(layer_id, expert_id)` 且无空闲占位符且淘汰策略为 None
- **THEN** 系统返回 None

### Requirement: 占位符权重加载
系统应支持将指定专家的权重加载到已分配的占位符。

#### Scenario: 加载专家权重到占位符
- **WHEN** 调用 `load_weights(placeholder, expert)`
- **THEN** 系统将专家的 state_dict 加载到占位符中

### Requirement: 占位符释放
系统应支持释放不再使用的占位符，使其可被重新分配。

#### Scenario: 释放占位符
- **WHEN** 调用 `release_placeholder(placeholder)`
- **THEN** 系统清空该占位符的 `(layer_id, expert_id)` 记录，并标记为空闲

#### Scenario: 按层释放占位符
- **WHEN** 调用 `release_by_layer(layer_id)`
- **THEN** 系统释放所有服务于该层的占位符

### Requirement: 占位符状态查询
系统应支持查询占位符的当前状态。

#### Scenario: 查询占位符是否空闲
- **WHEN** 调用 `is_available(placeholder)`
- **THEN** 系统返回该占位符是否空闲

#### Scenario: 查询占位符当前服务的专家
- **WHEN** 调用 `get_expert_for_placeholder(placeholder)`
- **THEN** 系统返回该占位符当前服务的 `(layer_id, expert_id)` 或 None

#### Scenario: 查询指定专家是否已加载到某个占位符
- **WHEN** 调用 `get_placeholder_for_expert(layer_id, expert_id)`
- **THEN** 系统返回该专家对应的占位符对象或 None

### Requirement: 统一专家位置快照
系统 MUST 能够提供专家在 GPU、CPU、SSD、loading 和占位符中的统一快照。

#### Scenario: 获取专家位置快照
- **WHEN** 调用位置快照接口
- **THEN** 系统返回包含 GPU、CPU、SSD、loading 和 placeholder 状态的快照对象

#### Scenario: 预取状态可见
- **WHEN** 某专家正在被 preload
- **THEN** 系统 MUST 在快照中将该专家标记为 loading

### Requirement: 淘汰策略抽象
系统应支持可插拔的占位符淘汰策略，通过策略模式实现。

#### Scenario: 指定淘汰策略
- **WHEN** 创建 ExpertPlaceholderManager 时传入淘汰策略（如 LRU、FIFO）
- **THEN** 系统使用该策略决定无空闲占位符时淘汰哪个

#### Scenario: 自定义淘汰策略
- **WHEN** 实现继承自 EvictionStrategy 的自定义策略类并传入管理器
- **THEN** 系统在淘汰时调用自定义策略的 `select_victim` 方法

### Requirement: LRU 淘汰策略
系统应支持 LRU（最近最少使用）淘汰算法。

#### Scenario: LRU 淘汰最近最少使用的占位符
- **WHEN** 需要淘汰且策略为 LRU
- **THEN** 系统淘汰最后一次访问时间最早的占位符

#### Scenario: LRU 访问时间更新
- **WHEN** 已分配的占位符被访问（acquire 或 load_weights）
- **THEN** 系统更新该占位符的最后访问时间

### Requirement: FIFO 淘汰策略
系统应支持 FIFO（先进先出）淘汰算法。

#### Scenario: FIFO 淘汰最早分配的占位符
- **WHEN** 需要淘汰且策略为 FIFO
- **THEN** 系统淘汰最早被分配的占位符
