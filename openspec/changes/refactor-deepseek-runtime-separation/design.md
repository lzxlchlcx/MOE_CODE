## 目标

将 `src/model/deepseek.py` 中耦合的 MoE 运行时拆解为四个清晰职责层：预测器、策略器、占位器、线程管理器。重构后，`forward` 只保留 Transformer 主干流程和模块编排，不再包含具体的调度搜索、预取执行和线程同步细节。

## 设计原则

- **单一职责**：每个模块只负责一类问题，不跨层直接操作其他模块内部状态。
- **数据驱动**：模块之间通过结构化数据对象通信，避免继续传递散乱的列表和临时张量。
- **阶段感知**：Prefill 与 Decode 使用不同策略，不共享穷举式分配逻辑。
- **可替换**：预测器、策略器、淘汰策略和执行器都应支持替换实现。
- **正确优先**：preload 是机会型优化，不能影响当前层输出正确性。

## 模块划分

### 1. 预测器 `ExpertPredictor`

职责：只负责预测未来层可能激活的专家。

输出建议使用统一数据结构：

```python
@dataclass(frozen=True)
class ExpertKey:
    layer: int
    expert_id: int

@dataclass
class ExpertDemand:
    key: ExpertKey
    token_count: int
    score: float = 1.0
    source: str = "predicted"
```

支持：

- `lookahead=1` 的下一层预测
- 未来多层 lookahead
- 返回 `[(layer, expert_id), ...]` 或更结构化的 `ExpertDemand`

当前 `GatePredictor` 可作为 baseline 实现，后续可替换为 PhasePer 或历史驱动预测器。

### 2. 策略器 `ExpertScheduler`

职责：只根据 request + placement snapshot + latency model 生成计划，不执行加载和计算。

建议输入对象：

```python
@dataclass
class PlacementSnapshot:
    gpu_resident: set[tuple[int, int]]
    placeholder_resident: set[tuple[int, int]]
    loading: set[tuple[int, int]]
    cpu_resident: set[tuple[int, int]]
    ssd_resident: set[tuple[int, int]]
    free_placeholders: int

@dataclass
class ExpertLayerRequest:
    layer: int
    phase: str  # prefill / decode
    current: list[ExpertDemand]
    future: list[ExpertDemand]
    assignments: dict[int, ExpertAssignment]
```

输出对象：

```python
@dataclass
class ExpertSchedule:
    cpu: list[ExpertDemand]
    gpu: list[ExpertDemand]
    preload: list[ExpertDemand]
    evict: list[ExpertDemand]
```

PDScope 风格策略应实现为两条路径：

- `schedule_prefill(...)`
- `schedule_decode(...)`

Prefill 使用跨层排序 + 局部重排 + 预取收益判断。
Decode 使用 `n_g^rho` 负载均衡判断，决定当前层和下一层的资源分配。

### 3. 占位器 `ExpertPlacementManager`

职责：管理 GPU 占位专家和专家位置状态，提供快照、分配、加载、释放和回收能力。

需要同时管理：

- 静态 GPU 常驻专家
- placeholder 中的动态专家
- 正在 loading 的专家
- CPU/SSD 可回退状态

建议增加统一状态接口：

```python
class ExpertLocation(Enum):
    GPU_STATIC = "gpu_static"
    GPU_PLACEHOLDER = "gpu_placeholder"
    CPU = "cpu"
    SSD = "ssd"
    LOADING = "loading"
```

占位器应提供：

- `snapshot()`：返回不可变快照
- `is_on_gpu(layer, expert_id)`：查询专家是否可被视为 GPU 可用
- `acquire_placeholder(layer, expert_id)`
- `load_weights(placeholder, expert)`
- `release_by_layer(layer)`
- `get_placeholder_for_expert(layer, expert_id)`

### 4. 线程管理器 `ExpertExecutionManager`

职责：执行调度计划，负责 CPU、GPU、preload 三类任务的并发启动、等待和结果合并。

执行语义：

- GPU 计算和 CPU 计算并行
- preload 仅使用空闲资源，不影响当前层正确性
- 所有必要任务结束后再做结果合并

建议接口：

```python
class ExpertExecutionManager:
    def execute(self, schedule: ExpertSchedule, context: ExpertLayerContext) -> torch.Tensor:
        ...
```

`ExpertLayerContext` 建议包含：

```python
@dataclass
class ExpertLayerContext:
    layer: int
    experts: nn.ModuleList
    inps_flat: torch.Tensor
    hidden_dim: int
    assignments: dict[int, ExpertAssignment]
```

## DeepSeek forward 的新数据流

建议 `mixtral_forward()` 重构为以下步骤：

1. 计算 attention 和 current hidden states。
2. 通过 `ExpertPredictor` 获取未来层候选专家需求。
3. 构建 `ExpertLayerRequest`。
4. 通过 `ExpertScheduler` 生成 `ExpertSchedule`。
5. 由 `ExpertExecutionManager` 执行当前层 CPU/GPU/preload 任务。
6. 合并 expert output 与 shared expert output，继续后续层。

这样 `deepseek.py` 只负责主流程编排，不再承载具体调度和线程细节。

## Prefill 调度设计

Prefill 阶段按照 PDScope 思路分为三步：

1. **全局排位**：将当前层非驻留专家与未来层预测专家合并，按 token 数或效用排序。
2. **局部重排**：只保留当前层专家，保证当前层不会被未来层挤占。
3. **置信度预取**：利用 I/O 气泡和预测命中率决定是否 preload 下一层专家。

输出必须显式包含：

- 当前层 CPU 专家
- 当前层 GPU 专家
- 下一层 preload 专家

## Decode 调度设计

Decode 阶段应以负载均衡为中心：

1. 计算理想 GPU 专家数 `n_g^rho`。
2. 根据当前层和下一层的 GPU 驻留数量判断模式。
3. 决定当前层是否需要额外 ondemand 加载。
4. 决定下一层是否应优先 preload。

Decode 阶段先不实现强制 eviction 的硬动作，只保留 eviction 计划输出，以便后续接入。

## 延迟模型

建议引入独立的 `ExpertLatencyModel`，封装以下成本：

- `cpu(token_count)`
- `gpu_compute(token_count)`
- `transfer(layer, expert_id)`

策略器只调用 latency model，不直接访问 benchmark JSON。

## 迁移方案

### 阶段 1：抽离执行器

把 `_execute_gpu_experts`、`_execute_cpu_experts`、`_prefetch_next_layer_experts` 从 `deepseek.py` 移出，形成独立执行器。

### 阶段 2：抽离 schedule 与 latency model

把当前策略中的 cost 计算、current/future 请求对象、调度结果对象抽象出来。

### 阶段 3：升级占位器快照

补齐动态位置查询，确保预取到 placeholder 的专家能被策略器识别为 GPU 可用。

### 阶段 4：重写 Prefill/Decode 策略

删除 `2^n_active` 穷举，改成 PDScope 风格的阶段感知调度。

## 风险与约束

- 需要确保现有推理输出正确性优先于优化收益。
- preload 必须是机会型，不得破坏当前层结果。
- 线程同步必须明确，不能依赖隐式 `future.result()` 顺序。
- 重构期间要保留最小可运行路径，方便对比新旧实现。
