## Context

当前 `FiddlerDeepSeek.mixtral_forward` 方法的 MoE 层处理中，专家执行逻辑以 120+ 行 `if/else` 块形式硬编码在层循环内。两种调度策略（纯 GPU 执行、CPU-GPU 混合代价优化）交织在一起，公共操作（`expert_mask` 构建、活跃专家收集）重复出现在两个分支中。

## Goals / Non-Goals

**Goals:**
- 将专家调度决策逻辑从 `mixtral_forward` 中解耦
- 策略负责决策和张量预处理（expert_mask、活跃专家、token 分配）
- `mixtral_forward` 负责遍历执行、应用权重、聚合结果
- 建立可扩展的调度策略接口，支持新增调度算法
- 保持现有功能行为不变

**Non-Goals:**
- 不引入新的调度策略算法
- 不修改 `mixtral_forward` 之外的接口
- 不改变专家权重加载/放置逻辑
- 不引入第三方依赖

## Decisions

### Decision 1: 使用策略模式（Strategy Pattern）

**选择**: 将每种调度算法封装为独立的策略类，通过统一接口被 `mixtral_forward` 调用。

**替代方案**:
- 简单方法抽取：改动最小但不利于扩展
- 回调函数：无法封装状态

**理由**: 策略模式适合后续增加新的调度算法（如 PDScope）。

### Decision 2: 策略接口设计（选项 A）

**选择**: 策略接口分为两步：

```python
# 第一步：策略决策 + 张量预处理
def decide_and_prepare(
    self,
    i_layer: int,
    experts: ModuleList,
    selected_experts: Tensor,
    routing_weights: Tensor,
) -> Tuple[List[int], List[int], Dict[int, Tuple[Tensor, Tensor]]]
# 返回：cpu_experts, gpu_experts, expert_assignments
# expert_assignments: { expert_id: (top_2_indices, routing_weight_subset) }

# 第二步：mixtral_forward 自己遍历执行、应用权重、聚合
```

共享上下文（`expert_placeholder`、`dev`、`is_expert_in_gpu` 等）通过构造函数注入。

### Decision 3: 公共逻辑提取为辅助函数

**选择**: 以下操作提取为模块级辅助函数：
- `_build_expert_mask(selected_experts, n_expert)` — 构建 one-hot expert mask
- `_collect_active_experts(expert_mask, n_expert)` — 收集有 token 分配的专家及其索引
- `_organize_token_assignments(expert_mask, routing_weights)` — 组织每个专家的 token 分配和路由权重

### Decision 4: 策略类放在同一文件内

**选择**: 策略类定义在 `deepseek.py` 文件中，位于 `FiddlerDeepSeek` 类之前。

**理由**: 项目结构简单，避免过度拆分；策略类与 `FiddlerDeepSeek` 强耦合。

## Risks / Trade-offs

- **[重构引入 bug]** → 通过对比重构前后 `mixtral_forward` 在相同输入下的输出张量来验证
- **[接口参数过多]** → 当前参数是必要的；如果不稳定，可以考虑封装为 `ExpertContext` dataclass
- **[性能开销]** → 策略模式引入的方法调用开销可忽略，专家计算本身在 GPU 上耗时远大于 Python 调度开销
