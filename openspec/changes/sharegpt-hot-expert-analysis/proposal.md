## Why

当前推理脚本可以从 ShareGPT 数据集中抽样运行，但缺少面向 MoE 专家热度的统计产物，无法用真实对话分布指导 `hot/deep.txt` 这类 GPU 常驻专家配置。需要新增一个可重复运行的分析能力：指定 ShareGPT 样本数量执行推理或路由统计，汇总热专家，并将结果保存到 `hot/` 目录供后续加载使用。

## What Changes

- 新增 ShareGPT 热专家分析能力，支持从 ShareGPT JSON 数据集中按指定数量抽取 prompt。
- 统计推理过程中各 `(layer, expert_id)` 被 gate 选中的频次，并按热度排序输出。
- 支持限制运行样本数，便于先跑小规模 smoke test，再跑完整测试集。
- 将热专家结果保存到 `hot/` 目录，包含可被现有 `set_expert_loc()` 消费的简单文本格式。
- 可选输出包含统计元信息的结构化结果，方便后续对比不同数据集、样本数和模型配置。

## Capabilities

### New Capabilities
- `hot-expert-analysis`: 定义基于 ShareGPT 数据集运行 MoE 路由统计、识别热专家并保存结果的能力。

### Modified Capabilities

无。

## Impact

- 影响推理/脚本入口，例如 `src/scripts/infer_deepseek.py` 或新增专用分析脚本。
- 影响 `src/model/deepseek.py` 中 gate 路由或推理过程的可观测性，需要暴露或记录专家选择统计。
- 影响 `hot/` 目录输出约定，需确保生成文件可被现有 `FiddlerDeepSeek.set_expert_loc()` 的 `layer,expert` 行格式读取。
- 不引入新的模型依赖；复用现有 ShareGPT JSON 加载、DeepSeek 推理和 MoE gate 结果。
