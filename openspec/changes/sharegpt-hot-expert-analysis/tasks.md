## 1. 路由统计基础设施

- [x] 1.1 在模型运行时增加热专家计数器，按 `(layer, expert_id)` 累积 gate 的真实 `selected_experts` 次数
- [x] 1.2 提供重置统计、读取排序统计和导出统计数据的方法，避免多次运行之间状态污染
- [x] 1.3 确保统计来源为 gate 选择结果，而不是 CPU/GPU/preload 调度结果

## 2. ShareGPT 分析入口

- [x] 2.1 新增专用热专家分析脚本或在现有推理脚本中新增显式分析模式
- [x] 2.2 支持 `--model`、`--dataset`、`--num-samples`、`--seed`、`--n-token`、`--output-dir` 和 `--output-name` 参数
- [x] 2.3 从 ShareGPT JSON 中提取 human prompt，并在 `--num-samples` 超过可用数量时安全使用全部 prompt
- [x] 2.4 使用本地随机实例和 seed 执行可复现抽样

## 3. 输出文件生成

- [x] 3.1 分析完成后创建 `hot/` 输出目录（如不存在）
- [x] 3.2 生成兼容 `set_expert_loc()` 的文本文件，默认写入 `hot/deep.txt`，每行格式为 `layer,expert`
- [x] 3.3 按专家频次从高到低写入文本文件，频次相同时使用稳定排序规则
- [x] 3.4 生成结构化 JSON 统计文件，包含数据集路径、seed、请求样本数、实际样本数、生成 token 数、总路由次数和专家计数

## 4. 验证与回归

- [x] 4.1 添加或更新轻量测试，验证 ShareGPT prompt 抽样数量、超量样本处理和 seed 可复现性
- [x] 4.2 添加或更新测试，验证 hot 文本文件格式与排序结果
- [x] 4.3 使用小样本 ShareGPT 输入运行分析入口，确认能够生成 `hot/deep.txt` 和统计 JSON
- [x] 4.4 验证生成的 `hot/deep.txt` 可被 `mDeepSeek.set_expert_loc()` 正常读取
