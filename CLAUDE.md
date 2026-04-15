# CLAUDE.md

## 项目概述
这是一个**MoE (Mixture of Experts) 混合专家模型GPU卸载基准测试系统**，用于测试多种MoE架构大语言模型的推理性能。

## MoE 模型 CPU-GPU 加载运行原理

### 核心架构

整个框架通过 `Fiddler{ModelName}` 类（如 `FiddlerDeepSeekV2`）实现 MoE 模型的 GPU-CPU 混合推理，核心流程如下：

---

### 初始化阶段

#### 涉及文件
| 文件 | 功能 |
|------|------|
| `deepseek.py` / `qwen.py` / `moon.py` / `mixtral.py` | 模型主类，包含加载和推理逻辑 |
| `config.py` | 统一配置管理，提供 `e`/`tg`/CPU时间表等参数 |
| `logger.py` | 统一日志和报告系统 |
| `scheduler.py` | ondemand 调度算法和预取策略 |

#### 核心步骤
1. **加载配置**: 从 `config.py` 或 `args` 初始化
2. **加载模型**: 使用 `transformers.AutoModelForCausalLM.from_pretrained()` 加载完整模型
3. **非专家组件固定到 GPU**: `bring_non_expert_to_gpu()`
   - Embedding 层: 词嵌入
   - Attention 层: 自注意力机制
   - LayerNorm 层: 层归一化
   - Gate 投影层: 路由决策门
   - 共享专家: Shared Experts（每层都参与，不卸载）
4. **计算可容纳专家数**: `calc_n_expert_on_gpu()`
   - 根据单个专家参数量和 GPU 显存大小
   - 预留 buffer 给 KV cache 和激活值
   - 最终确定 GPU 上可容纳的专家总数
5. **设置专家位置**: `set_expert_loc()`
   - 优先加载热点专家（从 `hot/{model}.txt`）
   - 加载到 `expert_loc` 矩阵标记：`1=GPU`, `0=CPU`
6. **加载专家到 GPU**: `bring_expert_to_gpu()`
   - 根据 `expert_loc` 矩阵将标记为 GPU 的专家从 CPU 移动到 GPU

---

### 占位专家机制

#### 原理
由于专家参数较大，无法在 GPU 上保留所有专家，使用**占位专家**（Placeholder）来实现 ondemand 动态加载：

1. **预先创建占位专家**: 在初始化时，预先在 GPU 上创建几个空壳专家（通常是第一层的前几个专家的深度拷贝）
   - `expert_placeholder`、`expert_placeholder2`、...、`expert_placeholder6`
   - 占位专家常驻 GPU，用于临时存储从 CPU 拷贝来的专家权重
2. **Pinned Memory 加速**: 在拷贝前，将 CPU 上的专家权重锁定到 `pin_memory()`，加速 CPU->GPU 传输
3. **权重拷贝**: 运行时需要某个专家时，用 `_async_ondemand()` 将其权重从 CPU 拷贝到某个占位专家
4. **占位专家到真实专家映射**: 通过 `placeholder_to_expert` 和 `expert_to_placeholder` 字典跟踪哪个占位专家对应哪个真实专家
5. **占位专家复用**: 处理完一层后，通过 `release_placeholder()` 释放占位专家，供下一层使用

---

### 推理阶段

#### 核心流程 (`mixtral_forward()`)
1. **Prefill 阶段**: 处理完整输入序列
   - Gate 网络路由: 计算每个 token 选哪几个专家
   - 专家统计: 按 token 数降序排序专家（热点优先）
   - **ondemand 动态调度** (`scheduler.decide_ondemand()`): 比较搬运成本和 CPU 计算成本，动态决定哪些专家从 CPU 搬运到 GPU
     - **TA**: 如果所有专家都在 CPU 上的总计算时间
     - **TC**: 剩余专家在 CPU 上的计算时间（随着专家被搬运到 GPU，TC 逐渐减小）
     - **TG**: `(1+i)*e + tg` - 搬运 i+1 个专家的成本 + GPU 计算成本
     - **决策逻辑**: 如果 `TG < TC`，说明搬运到 GPU 更划算，将该专家加入 ondemand 列表
     - **目标**: 通过动态平衡 CPU-GPU 负载，最大化整体吞吐量
2. **Decode 阶段**: 逐 token 生成
   - 每个 token 重复上述路由和调度过程
   - GPU 专家和 CPU 专家并行处理
   - 并行度计算: `(gpu_time + cpu_time) / wall_time`
3. **统计记录**: 通过 `logger.py` 记录各项指标

#### ondemand 调度的核心思想
- **不是简单的"尽可能多搬"**，而是根据成本动态决策
- 当搬运成本 + GPU 计算成本 < 剩余 CPU 计算成本时，才搬运
- 这样可以达到 CPU 和 GPU 的最优负载平衡，避免某一侧空闲
- 关键参数 `e`（搬运时间）、`tg`（GPU计算时间）、`cpu_time_table`（CPU计算时间表）必须通过 microbench 实测，不能随意修改

---

### 性能参数来源

#### 实测参数 (`e`/`tg`/CPU时间表)
这些参数由 `microbench.py` 实测生成，而非硬编码：

| 参数 | 来源 | 用途 |
|------|------|------|
| `e` (transfer_time_ms) | `microbench.py` | 专家从 CPU 拷贝到 GPU 的时间 |
| `tg` (gpu_compute_time_ms) | `microbench.py` | 专家在 GPU 上计算的时间 |
| `cpu_time_table` | `microbench.py` | 不同 token 数下专家在 CPU 计算的时间表（64个token的测量值） |

---

### 预取策略

通过 `--prefetch enabled/disabled` 控制：
- **disabled (默认)**: 不使用预取，完全依赖 ondemand
- **enabled**: 启用下一层预测预取，使用 `scheduler.should_prefetch()` 控制

---

## 支持的模型
- DeepSeek-V2-lite: `/mnt/g/Models/DeepSeek-v2-lite-chat`
- Qwen3-30B-A3B: `/mnt/g/Models/Qwen3-30B-A3B/`
- Mixtral-8x7B: `/home/share/bz/model/Mixtral-8x7B-v0.1` _(未下载)_
- Moonlight-16B-A3B-Instruct: `/home/share/bz/model/Moonlight-16B-A3B-Instruct` _(未下载)_

## 项目结构（非显而易见）

### 工作目录
所有操作必须在 `opensource/` 目录下执行

### 新增文件（本次重构）
| 文件 | 功能 |
|------|------|
| `opensource/config.py` | 统一配置管理（ModelConfig/SystemConfig） |
| `opensource/logger.py` | 统一日志系统 + 自动分析 + 调优建议 |
| `opensource/scheduler.py` | 专家调度策略（ondemand决策+预取开关） |
| `opensource/configs/` | 配置文件目录（JSON格式） |

### 结果保存
- **详细结果文件**: `opensource/result/` 目录
- **分析报告**: `opensource/log/` 目录（JSON + TXT）
- **配置文件**: `opensource/configs/` 目录
- **热点专家**: `hot/` 目录下的 `deep.txt`, `qwen.txt`, `moon.txt`, `mix.txt`
- **微基准数据**: `micro*.txt`, `ioreal_*.png`

### 模型实现
每个模型有独立文件：
- `deepseek.py`: DeepSeek-V2 模型
- `qwen.py`: Qwen 模型
- `moon.py`: Moonlight 模型
- `mixtral.py`: Mixtral 模型

## 关键发现

### 重构改进
- **统一配置**: `e`/`tg`/`cpu_time_table` 现在通过 `config.py` 管理，不再硬编码在模型文件中
- **闭环自动化**: `latency.py` 找不到 config 时自动运行 `microbench.py` 生成
- **预取策略**: 通过 `--prefetch enabled/disabled` 灵活控制
- **智能诊断**: 自动分析命中率/并行度/吞吐率/层时间并给出调优建议
- **容错性**: 模型路径不存在时 warning 并跳过，而非 crash
- **日志优化**: 默认保存到 `./log/`，进度条可视化，JSON+TXT 双格式

## 文档位置
- DeepSeek-V2 代码详解: `docs/deepseek_pytorch_transformers_guide.md`
  - 内容: PyTorch & Transformers 函数指南、占位专家核心原理等

## 约定事项
1. 更新 `CLAUDE.md` 文档
2. 将用户询问的问题整理为文档，保存到 `docs/` 目录下

## 实验记录
完成实验后，在 `实验日志.md` 中记录：

### 格式
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

### 规则
- 使用中文编写
- 记录关键数值变化（性能提升/下降百分比）
- 记录代码修改位置（文件:行号）
- 保持简洁准确，不要冗余
