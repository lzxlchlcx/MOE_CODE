# CLAUDE.md

## 项目概述
MoE 混合专家模型 GPU 卸载推理框架，基于 PDScope 论文实现 CPU-GPU 混合调度。

## 工作目录
所有操作在 `opensource/` 目录下执行。

## 核心文件
| 文件 | 功能 |
|------|------|
| `deepseek.py` / `qwen.py` / `moon.py` / `mixtral.py` | 模型主类（`Fiddler{ModelName}`），包含加载和推理 |
| `scheduler.py` | 专家调度：Decode 四模式负载均衡 + Prefill 三步调度法 |
| `config.py` | 统一配置管理（`e`/`tg`/`tc`/CPU时间表） |
| `logger.py` | 日志系统 + 自动分析 + 调优建议 |
| `latency.py` | Benchmark 入口 |
| `microbench.py` | 硬件参数实测（`e`/`tg`/`cpu_time_table`） |

## 调度架构 (scheduler.py)

### Decode 阶段 — `decide_decode(r_cur, r_next, k)`
- 计算最优 GPU 专家数：`n_g = argmin max(n_g·tg, (k-n_g)·tc)`，floor/ceil 精确比较
- 四种模式基于 `r_cur`/`r_next` 与 `n_g` 的关系：

| 模式 | 条件 | 动作 |
|------|------|------|
| A | cur < n_g, next ≥ n_g | 全力 ondemand 当前层 |
| B | cur ≥ n_g, next < n_g | offload 多余 + prefetch 下一层 |
| C | 都 ≥ n_g | 只 offload |
| default | 都 < n_g | 按预算分配 ondemand/prefetch |

返回：`(mode, ondemand_count, prefetch_count, offload_count)`

### Prefill 阶段 — `decide_prefill_schedule(cur, next, ...)`
Algorithm 1 三步法：
1. **全局排序**: 合并两层非驻留专家，增量 TG 累积找 GPU/CPU 边界
2. **局部重排**: `L_global ∩ E_cur`，防止当前层被饿死
3. **置信度预取**: 逐专家计算 ξ（R_hit 衰减），ξ ≤ 0 时 break（f-1 回退）

## mixtral_forward 流程 (deepseek.py)
1. Gate 路由 → 共享专家计算 → 专家 token 统计
2. Section 7: 下一层预测（gate 预测 + 过滤非驻留专家）
3. Section 8: 调度决策（Decode 用 `decide_decode()`，Prefill 用 `decide_prefill_schedule()`）
4. Section 9: 专家分类（GPU驻留 → prefetch → loading → ondemand → CPU）
5. Section 9.5: 释放未使用的预取专家（回收占位槽）
6. Section 9.6: Mode B/C offload 执行（占位专家降级到 CPU）
7. Section 10: 下一层预取（独立线程 + CUDA Stream 异步）
8. Section 11: GPU+CPU 并行专家处理
9. Section 12: 结果合并

## 占位专家机制
6 个占位专家（`_placeholders[6]`），通过列表管理：
- `_ph_in_use[6]`: 使用状态
- `_ph_mapping[6]`: 映射 `(layer, expert)` 
- `_get_available_placeholder()` → `(idx, placeholder)`
- `_release_placeholder_by_index(idx)` 精确释放
- Multi-stream: 3 个 CUDA Stream 并行传输 gate/up/down_proj

## 命中率统计（三项）
- `hot_hit_rate`: 热表常驻命中
- `prefetch_hit_rate`: 预取/占位专家命中
- `gpu_available_rate`: 总 GPU 可用率（= hot + prefetch）

## 性能参数
由 `microbench.py` 实测，通过 `config.py` 管理：
- `e`: 专家 CPU→GPU 传输时间
- `tg`: 专家 GPU 计算时间
- `tc`: 专家 CPU 计算时间（从 `cpu_time_table[1]` 提取）
- `cpu_time_table`: 不同 token 数下的 CPU 计算时间表

## 项目结构
```
opensource/
├── deepseek.py / qwen.py / moon.py / mixtral.py  # 模型实现
├── scheduler.py   # 调度策略
├── config.py      # 配置管理
├── logger.py      # 日志系统
├── latency.py     # Benchmark 入口
├── microbench.py  # 硬件参数测量
├── configs/       # JSON 配置文件
├── result/        # 详细结果
├── log/           # 分析报告 (JSON + TXT)
hot/               # 热点专家文件 (deep.txt 等)
```

## 约定
1. 实验记录写入 `实验日志.md`
2. 文档保存到 `docs/` 目录
