## ADDED Requirements

### Requirement: ShareGPT 热专家分析入口
系统 SHALL 提供一个显式入口，用于基于 ShareGPT JSON 数据集运行热专家分析，并允许用户指定模型路径、数据集路径和分析样本数量。

#### Scenario: 指定样本数量运行分析
- **WHEN** 用户使用热专家分析入口传入 `--dataset` 和 `--num-samples 10`
- **THEN** 系统 MUST 从 ShareGPT 数据集中选择不超过 10 条 human prompt 参与分析

#### Scenario: 样本数量超过数据集大小
- **WHEN** 用户传入的 `--num-samples` 大于数据集中可用 human prompt 数量
- **THEN** 系统 MUST 使用全部可用 human prompt，且不得报错

### Requirement: 专家路由频次统计
系统 SHALL 基于 MoE gate 的真实 `selected_experts` 统计每个 `(layer, expert_id)` 在分析运行中的出现次数。

#### Scenario: 记录单层专家选择
- **WHEN** 某一 MoE 层 gate 为 token 选择了专家 `expert_id`
- **THEN** 系统 MUST 将对应 `(layer, expert_id)` 的计数增加该专家在该层被选中的次数

#### Scenario: 不统计调度位置
- **WHEN** 调度器将某个专家安排到 CPU、GPU 或 preload
- **THEN** 系统 MUST NOT 将该执行位置作为热度来源替代 gate 选择统计

### Requirement: 热专家文本输出
系统 SHALL 将热专家排序结果保存到 `hot/` 目录下的文本文件，默认文件可被现有 `FiddlerDeepSeek.set_expert_loc()` 读取。

#### Scenario: 生成默认 hot 文件
- **WHEN** 热专家分析完成且用户未指定自定义输出文件名
- **THEN** 系统 MUST 写入 `hot/deep.txt`，每行格式为 `layer,expert`

#### Scenario: 热度排序
- **WHEN** 系统写入热专家文本文件
- **THEN** 文件中的专家 MUST 按统计频次从高到低排序

### Requirement: 结构化统计输出
系统 SHALL 保存结构化统计文件，用于记录热专家分析的输入参数和统计结果。

#### Scenario: 生成统计 JSON
- **WHEN** 热专家分析完成
- **THEN** 系统 MUST 在 `hot/` 目录下写入统计 JSON，包含数据集路径、实际样本数量、总路由次数和每个 `(layer, expert_id)` 的计数

#### Scenario: 记录复现信息
- **WHEN** 用户指定随机种子或样本数量
- **THEN** 结构化统计文件 MUST 记录对应 seed 和样本数量

### Requirement: 可复现抽样
系统 SHALL 支持通过 seed 控制 ShareGPT prompt 抽样，使相同输入参数产生相同样本集合。

#### Scenario: 相同 seed 重复运行
- **WHEN** 用户使用相同数据集、相同 `--num-samples` 和相同 `--seed` 重复运行热专家分析
- **THEN** 系统 MUST 选择相同的 prompt 集合用于分析
