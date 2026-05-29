## Context

当前 `src/scripts/infer_deepseek.py` 已支持 `--dataset` 从 ShareGPT JSON 中抽样 prompt，并调用 `FiddlerDeepSeek.generate()` 运行推理。`FiddlerDeepSeek.set_expert_loc()` 已经尝试读取 `./hot/deep.txt`，格式为每行 `layer,expert`，用于优先把热门专家放入 GPU。因此系统已有数据入口和热专家消费入口，但缺少中间的“采集 gate 路由统计并生成 hot 文件”的能力。

现有 MoE 路由发生在 `src/model/deepseek.py` 的每层 gate 计算附近，调度器随后基于 `selected_experts` 构造当前层需求。热专家分析最自然的数据源是 gate 输出的真实 `selected_experts`，而不是调度器最终分配到 CPU/GPU 的结果；后者会受到当前 GPU 常驻状态、placeholder、预取和 offload 策略影响，不适合作为“数据集真实热度”。

## Goals / Non-Goals

**Goals:**

- 提供一个可重复运行的 ShareGPT 热专家分析入口。
- 支持指定运行样本数量，默认可用于小规模验证。
- 基于真实 gate 选择统计 `(layer, expert_id)` 的出现次数。
- 按热度降序生成 `hot/deep.txt`，兼容现有 `set_expert_loc()` 读取格式。
- 额外保存结构化统计结果，用于审计样本数、token 数、专家频次和运行参数。

**Non-Goals:**

- 不改变专家调度算法、预取策略或 CPU/GPU 分配策略。
- 不要求在分析过程中评估模型输出质量。
- 不新增 ShareGPT 数据格式转换依赖；仅支持现有 JSON conversation 格式。
- 不在本 change 中优化分析性能，例如并行多进程或离线缓存 hidden states。

## Decisions

### D1: 以 gate 真实选择作为热度来源

**决策**: 在推理或分析 forward 过程中记录每个 MoE 层 gate 选出的 `selected_experts`，按 `(layer, expert_id)` 计数。

**理由**: 热专家应反映数据集触发的路由分布，而不是当前调度策略造成的执行位置。CPU/GPU/offload/preload 都是运行时策略，可能掩盖真实专家热度。

**替代方案**: 统计调度器的 `gpu_experts` 或 `cpu_experts`。被否决，因为这些集合受显存容量、placeholder 状态和策略参数影响，不适合生成长期可复用的 hot 文件。

### D2: 使用专用分析入口，而不是复用普通 infer 输出

**决策**: 新增专用脚本或在现有脚本中新增互斥模式，使用户显式运行热专家分析命令，并传入 `--dataset`、`--num-samples`、`--output-dir`、`--output-name` 等参数。

**理由**: 普通推理关注延迟和生成结果，热专家分析关注统计产物。专用入口可以避免污染 infer 脚本输出，也便于将来扩展 JSON 报告、top-k 截断和 deterministic sampling。

**替代方案**: 在 `infer_deepseek.py` 中每次运行都自动写 hot 文件。被否决，因为普通 benchmark 或调试推理不一定希望覆盖 hot 配置。

### D3: 输出两类文件

**决策**: 至少生成以下文件：

- `hot/deep.txt`：每行 `layer,expert`，按热度降序排列，用于现有加载逻辑。
- `hot/deep_stats.json`：包含数据集路径、样本数、生成 token 数、总路由次数、每个专家计数和排序结果。

**理由**: 文本文件保证与现有 `set_expert_loc()` 兼容；JSON 文件保证统计过程可追溯，也便于后续对比不同数据集或样本规模。

**替代方案**: 只输出 `deep.txt`。被否决，因为缺少元信息后难以确认这个 hot 文件来自哪个数据集、多少样本、是否只跑了 smoke test。

### D4: 样本选择默认可复现

**决策**: 分析入口支持 `--num-samples` 和 `--seed`。当 `--num-samples` 小于数据集大小时，使用固定 seed 抽样；也可以支持顺序读取模式作为后续扩展。

**理由**: 热专家统计需要可重复，否则不同运行会产生难以解释的 hot 列表差异。

**替代方案**: 直接沿用 `random.sample()` 的全局随机行为。被否决，因为不利于复现实验。

## Risks / Trade-offs

- **[统计开销]** 完整生成会比只跑 gate 更慢 → 先复用现有 `generate()` 路径确保正确，再保留将来引入轻量 gate-only 分析的空间。
- **[覆盖 hot 文件]** 用户可能误覆盖已有 `hot/deep.txt` → CLI 应提供输出路径参数，并在日志中明确写入位置；必要时支持 `--output-name` 避免覆盖。
- **[样本偏差]** 小样本统计的热专家可能不稳定 → JSON 元信息必须记录样本数和 seed，用户可逐步增大 `--num-samples`。
- **[路径语义]** `set_expert_loc()` 当前默认读取 `./hot/deep.txt`，相对路径依赖运行目录 → 分析输出默认写入项目根目录下的 `hot/`，文档和日志需明确。
