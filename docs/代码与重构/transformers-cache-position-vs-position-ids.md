# Transformers 4.55+ DynamicCache API 变化：cache_position vs position_ids

## 问题背景

在将 DeepSeek 模型从 `trust_remote_code=True`（使用模型仓库自带代码）迁移到 transformers 内置模型时，遇到 `self_attn.forward()` 接口变化：

```python
# 旧版 (trust_remote_code)
attn_output = layer.self_attn(
    hidden_states=inps,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_value=self.past_key_value,
    use_cache=True,
)

# 新版 (transformers 4.55+ 内置模型)
attn_output = layer.self_attn(
    hidden_states=inps,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_value=self.past_key_value,
    use_cache=True,
    cache_position=cache_position,          # ← 新增
    position_embeddings=position_embeddings, # ← 新增
)
```

## cache_position 与 position_ids 的区别

### 基本对比

| 属性 | `position_ids` | `cache_position` |
|------|---------------|------------------|
| **形状** | 2D: `(batch_size, seq_len)` | 1D: `(seq_len,)` |
| **用途** | 计算 RoPE 位置编码 | KV Cache 索引更新 |
| **batch 维度** | 需要（每个 batch 独立） | 不需要（全局共享，自动广播） |

### 为什么 cache_position 只需要 1D？

关键在于 KV Cache 的存储结构和 `index_copy_` 操作：

```python
# KV Cache 是 4D 张量
self.keys = torch.zeros(batch_size, num_heads, max_seq_len, head_dim)

# index_copy_ 操作
target.index_copy_(dim=2, index=cache_position, source=key_states)
```

**写入时 batch 自动广播**：

```
cache_position = [5]  # 1D，全局共享

# 写入操作等价于:
keys[0, :, 5, :] = new_key[0]  # batch 0
keys[1, :, 5, :] = new_key[1]  # batch 1
keys[2, :, 5, :] = new_key[2]  # batch 2
keys[3, :, 5, :] = new_key[3]  # batch 3
```

所有 batch 同时更新到相同的序列位置，这正是自回归生成的标准行为。

### 图示说明

```
keys[batch, head, seq, dim]

position_ids = [[10], [10], [10], [10]]  # 2D，每个 batch 可能有不同位置
cache_position = [10]                     # 1D，全局共享

# RoPE 计算（需要 batch 维度）
cos, sin = rotary_emb(position_ids)  # 每个 batch 可能有不同位置编码

# KV 写入（batch 自动广播）
keys[:, :, cache_position, :] = key_states  # 所有 batch 同时更新
```

## 代码示例

### 正确的参数构建

```python
# position_ids: 2D (batch_size, seq_len)
new_position_ids = torch.arange(
    past_key_values_length,
    past_key_values_length + seq_len,
    dtype=torch.long,
    device=device
).unsqueeze(0).expand(batch_size, -1)  # ← 扩展到 2D

# cache_position: 1D (seq_len,)
cache_position = torch.arange(
    past_key_values_length,
    past_key_values_length + seq_len,
    dtype=torch.long,
    device=device
)  # ← 保持 1D

# 调用 self_attn
attn_output = layer.self_attn(
    hidden_states=hidden_states,
    attention_mask=attention_mask,
    position_ids=new_position_ids,
    past_key_value=past_key_value,
    use_cache=True,
    cache_position=cache_position,
    position_embeddings=position_embeddings,
)
```

## 常见错误

### 错误 1：cache_position 使用 2D

```python
# ❌ 错误：cache_position 不需要 batch 维度
cache_position = new_position_ids  # 2D (batch, seq)

# 会导致 index_copy_ 失败或行为异常
```

### 错误 2：混淆两个参数

```python
# ❌ 错误：用 position_ids 代替 cache_position
attn_output = layer.self_attn(
    ...,
    cache_position=position_ids,  # 错误：传入了 2D tensor
)
```

### 错误 3：忘记传递 position_embeddings

```python
# ❌ 错误：缺少必需的 position_embeddings
attn_output = layer.self_attn(
    ...,
    cache_position=cache_position,
    # 缺少 position_embeddings
)
```

## 相关源码参考

### DynamicCache.update() 实现

```python
# transformers/cache_utils.py
if cache_position is not None:
    # Generation phase. Update specific positions.
    self.keys.index_copy_(2, cache_position, key_states)
    self.values.index_copy_(2, cache_position, value_states)
```

### DeepseekV2Attention.forward() 签名

```python
# transformers/models/deepseek_v2/modeling_deepseek_v2.py
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, ...]:
```

## 迁移建议

1. **初始化 rotary_emb**：在 `__init__` 中创建 `DeepseekV2RotaryEmbedding` 实例
2. **计算 position_embeddings**：在 forward 中调用 `self.rotary_emb(hidden_states, position_ids)`
3. **构建 cache_position**：1D tensor，范围与 position_ids 相同但无 batch 维度
4. **完整传递参数**：确保 `cache_position` 和 `position_embeddings` 都传入

## 参考链接

- [Transformers Cache Utils](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py)
- [DeepSeek V2 Modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py)
