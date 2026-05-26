## ADDED Requirements

### Requirement: GPU 和 CPU 专家并行执行
系统 SHALL 使用多线程实现 GPU 和 CPU 专家的并行执行，同时运行，最后合并结果。

#### Scenario: 同时存在 GPU 和 CPU 专家
- **WHEN** 某层同时有分配到 GPU 的专家和分配到 CPU 的专家
- **THEN** 两个执行线程并行工作，总执行时间约为两者中的最大值

#### Scenario: 只有 GPU 专家
- **WHEN** 某层只有分配到 GPU 的专家，没有 CPU 专家
- **THEN** 只有 GPU 线程工作，执行逻辑与之前一致

#### Scenario: 只有 CPU 专家
- **WHEN** 某层只有分配到 CPU 的专家，没有 GPU 专家
- **THEN** 只有 CPU 线程工作，执行逻辑与之前一致

### Requirement: 数值一致性
并行执行的结果 SHALL 与串行执行的结果数值一致（在浮点误差范围内）。

#### Scenario: 前后对比
- **WHEN** 使用相同的输入和配置，分别运行串行版本和并行版本
- **THEN** 输出的 logits 完全一致（或 torch.allclose 满足）

### Requirement: 线程安全
并行执行 SHALL 线程安全，避免数据竞争和不确定的行为。

#### Scenario: 多次执行结果一致
- **WHEN** 多次运行相同的输入
- **THEN** 每次结果完全一致
