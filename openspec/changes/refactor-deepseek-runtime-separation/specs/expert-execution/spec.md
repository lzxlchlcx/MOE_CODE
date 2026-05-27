## ADDED Requirements

### Requirement: 统一并行执行管理
系统 MUST 通过统一的线程管理器协调 CPU、GPU 和 preload 三类任务。

#### Scenario: 并行执行专家计算与预取
- **WHEN** 调度器返回 CPU、GPU 和 preload 计划
- **THEN** 系统 MUST 并行启动对应任务，并在全部必要任务完成后再合并结果

#### Scenario: preload 不影响当前层结果正确性
- **WHEN** preload 任务失败或被跳过
- **THEN** 系统 MUST 仍能完成当前层 CPU/GPU 专家计算并返回正确结果

### Requirement: 结果合并正确性
系统 MUST 正确合并 CPU 与 GPU 分支的专家输出。

#### Scenario: 合并 CPU 与 GPU 输出
- **WHEN** CPU 与 GPU 专家都完成计算
- **THEN** 系统 MUST 按 token 索引和 routing weight 合并两个分支的输出张量

#### Scenario: 结果张量类型一致
- **WHEN** CPU 输出需要回传 GPU
- **THEN** 系统 MUST 在合并前将输出转换为与目标张量一致的 dtype 和 device 语义

### Requirement: 预取执行隔离
系统 MUST 将 preload 执行与当前层专家计算隔离，避免占用当前层必要的资源。

#### Scenario: 占位符不足时回退
- **WHEN** preload 所需占位符不足
- **THEN** 系统 MUST 放弃多余 preload 而不影响当前层计算
