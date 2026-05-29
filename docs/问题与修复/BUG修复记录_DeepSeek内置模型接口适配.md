# BUG 修复记录：DeepSeek 内置模型接口适配

## 问题描述

将 DeepSeek 从 `trust_remote_code=True`（使用模型仓库自带代码）切换到 transformers 内置模型后，运行时报错：

```
File ".../modeling_deepseek_v2.py", line 366, in forward
    q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, position_embeddings.to(q_pe.device))
AttributeError: 'NoneType' object has no attribute 'to'
```

## 根本原因

transformers 4.55.0 内置的 `DeepseekV2Attention` 与旧版模型仓库代码的接口存在差异：

| 组件 | 旧版 (模型仓库) | 新版 (transformers 内置) |
|------|----------------|-------------------------|
| RoPE 计算 | `self_attn` 内部自动计算 | 需外部传入 `position_embeddings` |
| Cache 位置 | 内部使用 `past_key_values_length` | 需传入 `cache_position` 参数 |

### 核心差异分析

**旧版代码（模型仓库自带）：**
```python
# modeling_deepseek.py (来自模型仓库)
class DeepseekV2Attention(nn.Module):
    def forward(self, hidden_states, attention_mask, position_ids, past_key_value):
        # RoPE 在内部计算
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
```

**新版代码（transformers 内置）：**
```python
# modeling_deepseek_v2.py (transformers 4.55.0)
class DeepseekV2Attention(nn.Module):
    def forward(self, hidden_states, attention_mask, position_ids, past_key_value,
                cache_position, position_embeddings,  # ← 新增必需参数
                **kwargs):
        # position_embeddings 必须由调用者传入
        q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, position_embeddings.to(q_pe.device))
```

## 修复方案

### 1. 导入 RotaryEmbedding

```python
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2RotaryEmbedding
```

### 2. 初始化 rotary_emb

在 `__init__` 中添加：
```python
self.rotary_emb = DeepseekV2RotaryEmbedding(config=self.model.config, device=self.dev)
```

### 3. 修改 mixtral_forward 方法签名

添加 `cache_position` 参数：
```python
def mixtral_forward(self, input_ids, position_ids, attention_mask, cache_position, is_prefill=False):
```

### 4. 计算 position_embeddings

在 `mixtral_forward` 内部添加：
```python
# 计算 position_embeddings (cos, sin tuple)
position_embeddings = self.rotary_emb(inps, position_ids)
```

### 5. 更新 self_attn 调用

传入新参数：
```python
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

### 6. 生成 cache_position

在 `generate` 方法中构建 `cache_position`：
```python
# 构建 cache_position
cache_position = torch.arange(
    self.past_key_values_length,
    self.past_key_values_length + input_ids.shape[1],
    dtype=torch.long,
    device=self.dev
)

logits = self.mixtral_forward(input_ids, new_position_ids, attention_mask, cache_position, is_prefill=not is_decode)
```

## 影响文件

- `10-fiddler-main/src/fiddler/deepseek.py`
- `20-PDScope/deepseek.py`（同样需要修复）

## 相关版本信息

- transformers: 4.55.0
- Python: 3.12
- PyTorch: 2.x

## 修复时间

2026-05-07
