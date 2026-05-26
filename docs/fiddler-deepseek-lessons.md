# Fiddler DeepSeek 移植踩坑记录

> 日期：2026.5.7
> 上下文：将 Fiddler（原支持 Mixtral-8x7B）扩展支持 DeepSeek-V2-lite-chat

---

## 1. 2^N 枚举只适用于小 N

**场景**：Fiddler 核心算法穷举 2^N 种 CPU/GPU 专家分配方案，选总延迟最小的。

**问题**：Mixtral 有 8 个专家 → 2^8=256 可行。DeepSeek 有 **64** 个专家 → 2^64 ≈ 1.8×10^19，永远跑不完。表现为程序打印 "Model is ready." 后完全挂起。

**修复**：只枚举被 gate 实际选中的活跃专家（通常 6-10 个），`2^10=1024` 可瞬间完成：

```python
# 错误：枚举所有 64 个专家
for config in range(1 << len(experts)):  # 2^64 永远跑不完

# 正确：只枚举活跃专家
active_experts = [i for i in range(len(experts)) if token_count[i] > 0]
for config in range(1 << len(active_experts)):  # 2^6~2^10
```

**教训**：暴力搜索前必须评估搜索空间。当 N>20 时 2^N 就不现实了。

---

## 2. Expert forward() 签名不同：Mixtral vs DeepSeek

**场景**：MoE 层需要逐专家调用 forward。

**差异**：

| 模型 | Expert 类 | forward 签名 | routing_weights 处理 |
|------|-----------|-------------|---------------------|
| Mixtral | `MixtralSparseMoeBlock.expert` | `forward(x, routing_weight)` | 内部乘 |
| DeepSeek | `DeepseekV2MLP` | `forward(x)` | 不处理 |

**问题**：直接调用 `experts[i](x, routing_weights)` 在 DeepSeek 上报 `TypeError: takes 2 positional arguments but 3 were given`。

**修复**：DeepSeek 需要外部乘 routing_weights：

```python
# Mixtral 方式（内部乘）
output = expert(input, routing_weights)

# DeepSeek 方式（外部乘）
output = expert(input)
output = output * routing_weights
```

---

## 3. BFloat16 × Float32 = Float32 类型提升

**场景**：expert 输出是 BFloat16，routing_weights 是 Float32，相乘后类型自动提升为 Float32。

**问题**：`index_add_()` 要求 self 和 source 类型一致，报 `RuntimeError: self (BFloat16) and source (Float) must have the same scalar type`。

**修复**：

```python
current_state = current_state * routing_weights[top_2_list, idx_list, None]
inps_after_experts.index_add_(
    0, top_2,
    current_state.to(inps_after_experts.dtype),  # 显式 cast 回 BFloat16
)
```

**教训**：PyTorch 的类型提升规则（type promotion）在混合精度场景中经常导致隐藏的类型不匹配。在 `index_add_`、`scatter_` 等原地操作前，必须确保 source 类型与 self 一致。

---

## 4. DeepSeek MLA 的 Attention Mask 处理

**场景**：generate 循环中，prefill 后进入 decode 阶段，attention mask 需要增量更新。

**差异**：

| 模型 | Decode 时 mask 期望 |
|------|-------------------|
| Mixtral | `(batch, 1, 32, cumulative_seq_len)` — 覆盖完整 KV cache |
| DeepSeek MLA | `(batch, 1, 1, current_input_len)` — 只匹配当前输入 |

**问题**：decode 阶段传了 `(1, 1, 1, 7)` 的 mask（累积长度），但 DeepSeek MLA 要求 `(1, 1, 1, 1)`（当前输入长度=1），报 `ValueError: Attention mask should be of size (1,1,1,1), but is torch.Size([1,1,1,6])`。

**原因**：DeepSeek-V2 使用 MLA（Multi-head Latent Attention），KV cache 压缩为低秩表示，attention 内部自行管理 cache 的 KV 长度，不需要外部 mask 指定累积长度。

**修复**：decode 阶段直接传 `attention_mask=None`：

```python
if is_decode:
    attention_mask = None
```

---

## 5. CUDA 错误是异步且不可恢复的

**场景**：一次运行中遇到了 CUDA index out of bounds 错误，修复代码后重新运行仍报 `CUBLAS_STATUS_EXECUTION_FAILED`。

**原因**：CUDA 操作是异步提交的。当 `index_add_` 发生越界写入时，错误不会立即报告，而是破坏了 GPU 显存内容。之后的 CUDA 操作（包括不相关的 matmul）都会使用损坏的数据，产生级联错误。

**关键点**：
- **同一个 Python 进程内无法恢复** — GPU 状态已被破坏
- **必须重启进程** — 杀掉 Python 重新启动，GPU 状态才会重置
- **不要被后续错误误导** — CUBLAS 错误通常是前一个 CUDA 错误的后果，不是根因

---

## 6. DynamicCache API 版本差异与 Monkey-patch

**场景**：DeepSeek 模型通过 `trust_remote_code=True` 加载本地模型代码，本地代码调用 `past_key_value.get_usable_length()`，但 transformers 4.55 的 `DynamicCache` 只有 `get_seq_length()`。

**修复**：在类级别 monkey-patch：

```python
import transformers.cache_utils

def _patch_dynamic_cache():
    dc = transformers.cache_utils.DynamicCache
    if not hasattr(dc, 'get_usable_length'):
        def get_usable_length(self, new_seq_len=None, layer_idx=0):
            return self.get_seq_length(layer_idx)
        dc.get_usable_length = get_usable_length
```

**注意**：必须 patch 到**类**上（`dc.get_usable_length = ...`），而不是实例上。因为 DeepSeek 内部创建的 cache 对象可能不是同一个实例。

---

## 7. Monkey-patch 参数顺序映射

**场景**：上一步的 monkey-patch 一开始映射错误。

**调用签名对比**：

```python
# DeepSeek 模型代码调用方式：
kv_seq_len += cache.get_usable_length(kv_seq_len, self.layer_idx)
# → 第1个参数: new_seq_len (当前序列长度)
# → 第2个参数: layer_idx

# DynamicCache.get_seq_length 签名：
cache.get_seq_length(layer_idx=0)
# → 第1个参数: layer_idx
```

**错误映射**（导致 layer_idx 收到 kv_seq_len=5，超出 cache 层数，返回错误值）：

```python
# 错误：直接赋值，参数顺序不匹配
cache.get_usable_length = cache.get_seq_length
# 调用时 get_seq_length(layer_idx=5) 超出范围
```

**正确映射**：

```python
def get_usable_length(self, new_seq_len=None, layer_idx=0):
    return self.get_seq_length(layer_idx)  # 只取 layer_idx
```

**教训**：monkey-patch 时必须确认两侧的参数语义和顺序一致，不能简单赋值。

---

## 附录：DeepSeek-V2 vs Mixtral 架构差异速查表

| 维度 | Mixtral-8x7B | DeepSeek-V2-lite |
|------|-------------|------------------|
| 模型类 | `MixtralForCausalLM` | `AutoModelForCausalLM` + `trust_remote_code` |
| 总层数 | 32（全 MoE） | 27（layer 0 dense, 1-26 MoE） |
| 路由专家数 | 8 | 64 |
| 共享专家数 | 0 | 2 |
| Top-K 选择 | 2 | 6 |
| 专家路径 | `layer.block_sparse_moe.experts[i]` | `layer.mlp.experts[i]` |
| Gate 路径 | `layer.block_sparse_moe.gate` | `layer.mlp.gate` |
| Gate 返回 | `router_logits`（需手动 softmax+topk） | `(selected, weights, _)` 三元组 |
| 共享专家 | 无 | `layer.mlp.shared_experts` |
| 权重名 | w1, w2, w3 | gate_proj, up_proj, down_proj |
| Expert forward | `expert(x, routing_weight)` | `expert(x)` |
| Attention 类型 | 标准 MHA | MLA (Multi-head Latent Attention) |
| Attention mask | `(batch, 1, 32, seq_len)` | `(batch, 1, 1, seq_len)` |
| KV cache | 标准 K/V | 压缩低秩表示 |
