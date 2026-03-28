# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## 项目概述
这是一个**MoE (Mixture of Experts) 混合专家模型GPU卸载基准测试系统**，用于测试多种MoE架构大语言模型的推理性能。

## 支持的模型
- DeepSeek-V2-lite: `/mnt/g/Models/DeepSeek-v2-lite-chat`
- Qwen3-30B-A3B: `/mnt/g/Models/Qwen3-30B-A3B/`
- Mixtral-8x7B: `/home/share/bz/model/Mixtral-8x7B-v0.1`
- Moonlight-16B-A3B-Instruct: `/home/share/bz/model/Moonlight-16B-A3B-Instruct`

## 关键命令
- **主要测试**: `cd opensource && python latency.py --model "/mnt/g/Models/DeepSeek-v2-lite-chat" --batch_size 1`
- **微基准测试**: `cd opensource && python microbench.py --model <model_path>`
- **批量测试**: `cd opensource && ./run_benchmark.sh`

## 项目结构（非显而易见）
- **工作目录**: 所有操作必须在 `opensource/` 目录下执行
- **结果保存**: 详细结果文件保存到 `opensource/result/` 目录
- **数据集**: `sharegpt_v3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json`
- **热点专家**: `hot/` 目录下的 `deep.txt`, `qwen.txt`, `moon.txt`, `mix.txt`
- **模型实现**: 每个模型有独立文件 - [`deepseek.py`](opensource/deepseek.py), [`qwen.py`](opensource/qwen.py), [`moon.py`](opensource/moon.py), [`mixtral.py`](opensource/mixtral.py)

## 关键发现
- 没有 `requirements.txt` 或 `setup.py`，依赖需要手动安装（PyTorch, Transformers, NumPy等）
- 默认模型路径硬编码为 `/mnt/g/Models/DeepSeek-v2-lite-chat`
- 吞吐量计算需注意：需要乘以 `batch_size` 和样本数
- 模型加载可能需要较长时间（数十秒）
- 乱码问题可能与tokenizer解码有关，`.tolist()` 有时可以解决

## 性能优化要点
- 删除 `deepseek.py` 中的print语句可显著提升性能
- batch_size=4 时吞吐量约为 batch_size=1 的2倍
- 热点专家预取机制对性能有重要影响

## 代码修改流程
当用户觉得优化完成时，按照以下步骤进行：

### Git Commit 规范
当代码修改后提交到本地git仓库，commit message格式：
```
<类型>: <简短描述>

<详细说明（可选）>
- 修改点1
- 修改点2
```

**类型**:
- `Fix`: 修复bug
- `Feat`: 新功能
- `Perf`: 性能优化
- `Refactor`: 重构
- `Docs`: 文档更新

### 实验记录编写
完成实验后，应在 [`实验记录.md`](实验记录.md) 中记录：

```markdown
# 日期
## 目标
目标是什么，希望达到什么性能，复现什么现象，实验什么东西
## 预期
预计会产生什么结果
## 执行
怎么做的，执行什么命令
## 结果
结果如何，有什么不符合预期的内容
## 反思
为什么不符合预期，如果符合预期则说明什么猜想是对的，下一步该做什么
```

**规则**:
- 使用中文编写
- 记录关键数值变化（性能提升/下降百分比）
- 记录代码修改位置（文件:行号）
- 保持简洁准确，不要冗余
