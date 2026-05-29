# DeepSeek Attention Mask 修复详解

## 问题描述

`deepseek.py` 输出乱码，生成的文本重复或无意义，例如：
```
输入: "Hello, how are you?"
输出: "I hope you are you are you are you are you are..." （重复乱码）
```

## 根本原因

`tokenize()` 函数将 2D attention_mask 错误地扩展为 4D padding mask（全True），而不是 causal mask。这导致模型可以看到未来的 token，破坏了自回归生成的因果性。

---

## 背景知识

### 1. 自回归语言模型

语言模型生成文本是**自回归**的：每次只生成一个token，且只能依赖**已生成的内容**，不能看到未来。

```
输入: "Hello, how"
      ↓
模型预测下一个token → "are"
      ↓
输入: "Hello, how are"
      ↓
模型预测下一个token → "you"
      ↓
输入: "Hello, how are you"
      ↓
模型预测下一个token → "?"
```

### 2. Attention 机制

Transformer的Self-Attention让每个token都能看到**所有**其他token。对于生成任务，需要**Causal Mask**阻止模型看到未来的token。

```
         你    好    吗
你     [0.5,  0,    0  ]  ← "你"只能看到自己
好     [0.2, 0.6,   0  ]  ← "好"能看到"你"和"好"
吗     [0.2, 0.3,  0.5]  ← "吗"能看到所有（已生成的）

用矩阵表示（True=允许看，False/负无穷=禁止）：
       [[True, False, False],
        [True, True,  False],
        [True, True,  True ]]
```

这称为**上三角掩码**（Upper Triangular Mask），只有下三角和对角线是True。

### 3. 两种 Mask 的区别

| Mask 类型 | 形状 | 作用 | 值 |
|-----------|------|------|-----|
| **Padding Mask** | [batch, seq_len] | 标记哪些是真实token，哪些是填充 | 1=真实token, 0=填充 |
| **Causal Mask** | [batch, 1, seq_len, seq_len] | 阻止看到未来的token | 下三角=True, 上三角=False |

---

## 代码分析

### 修复前：错误的 tokenize()

```python
def tokenize(self, text, input_token=None):
    encodings = self.tokenizer(text, padding=True, return_tensors="pt")
    input_ids = encodings.input_ids.to(self.dev)           # [batch, seq_len]
    attention_mask = encodings.attention_mask.bool().to(self.dev)  # [batch, seq_len]
    
    seq_length = input_ids.shape[1]
    
    # ❌ 错误：手动扩展成全True的4D矩阵
    if attention_mask.dim() == 2:
        attention_mask = attention_mask.unsqueeze(1)      # [batch, 1, seq_len]
        attention_mask = attention_mask.unsqueeze(-1)     # [batch, 1, seq_len, 1]
        attention_mask = attention_mask.expand(-1, -1, -1, seq_length)  # [batch, 1, seq_len, seq_len]
    
    return input_ids, position_ids, attention_mask
```

**错误结果**：
```
输入: "Hello, how are you?" (7个token)
attention_mask 形状: [1, 1, 7, 7]

[[[True, True, True, True, True, True, True],   ← 位置0能看到所有位置（包括未来！）
  [True, True, True, True, True, True, True],   ← 位置1能看到所有位置
  [True, True, True, True, True, True, True],   ← ...
  ...
  [True, True, True, True, True, True, True]]]  ← 位置6能看到所有位置
```

这是一个**padding mask**的广播结果，不是**causal mask**！

### 修复后：正确的 tokenize()

```python
def tokenize(self, text, input_token=None):
    encodings = self.tokenizer(text, padding=True, return_tensors="pt")
    input_ids = encodings.input_ids.to(self.dev)           # [batch, seq_len]
    attention_mask = encodings.attention_mask.to(self.dev)  # [batch, seq_len]
    
    # ✅ 修复：保持2D，让 mixtral_forward 中创建 causal mask
    return input_ids, position_ids, attention_mask
```

---

## generate() 中的关键修复

### 修复前：decode 阶段不更新 attention_mask

```python
for i_token in range(output_token):
    if is_decode:
        # ❌ 错误：没有更新 attention_mask！
        pass
    
    logits = self.mixtral_forward(input_ids, new_position_ids, attention_mask, ...)
```

### 修复后：decode 阶段正确更新 attention_mask

```python
for i_token in range(output_token):
    if is_decode:
        # ✅ 关键修复：更新 attention_mask 包含新token
        # 此时 input_ids 是 [batch, 1]，表示生成一个新token
        # attention_mask 需要更新为 [batch, past_key_values_length + 1]
        past_seq_len = self.past_key_values_length
        attention_mask = torch.ones(
            input_ids.shape[0], past_seq_len + 1,
            dtype=torch.long, device=self.dev
        )
    
    logits = self.mixtral_forward(input_ids, new_position_ids, attention_mask, ...)
```

---

## mixtral_forward() 中的关键修复

### 修复前：使用错误的 mask

```python
def mixtral_forward(self, input_ids, position_ids, attention_mask, ...):
    # attention_mask 已经是错误的4D全True矩阵
    
    for layer in self.model.layers:
        attn_output = layer.self_attn(
            hidden_states=inps,
            attention_mask=attention_mask,  # ← 错误的mask！
            ...
        )
```

### 修复后：使用 create_causal_mask()

```python
from transformers.masking_utils import create_causal_mask

def mixtral_forward(self, input_ids, position_ids, attention_mask, ...):
    inps = self.model.embed_tokens(input_ids)
    
    # ✅ 关键：使用 transformers 官方的 create_causal_mask 创建正确的 causal mask
    causal_mask = create_causal_mask(
        config=self.config,
        input_embeds=inps,
        attention_mask=attention_mask,  # 2D mask
        cache_position=cache_position,
        past_key_values=self.past_key_value,
        position_ids=position_ids,
    )
    
    for layer in self.model.layers:
        attn_output = layer.self_attn(
            hidden_states=inps,
            attention_mask=causal_mask,  # ← 正确的causal mask！
            ...
        )
```

---

## create_causal_mask() 内部逻辑

`create_causal_mask()` 是 transformers 库的函数，根据配置自动创建正确的 causal mask：

### SDPA 模式 (config._attn_implementation=None)

```python
# 返回 None，SDPA 会自动使用 is_causal=True
# PyTorch 原生支持 causal masking，不需要显式 mask
```

### Eager 模式

```python
# 创建下三角矩阵（上三角为True，要mask掉）
causal_mask = torch.triu(
    torch.ones(seq_len, seq_len, dtype=torch.bool),
    diagonal=1
)
causal_mask = ~causal_mask  # 反转：下三角为True（保留的位置）

# 转换为 attention 分数的加法 mask
# True → 0（保留），False → -inf（mask掉）
causal_mask = causal_mask.float().masked_fill(~causal_mask, float('-inf'))

# 如果有 past_key_values，扩展 mask
if past_key_values is not None:
    past_len = past_key_values.get_seq_length()
    causal_mask = torch.cat([
        torch.zeros(seq_len, past_len),  # past部分全可见
        causal_mask                       # 当前部分用causal
    ], dim=-1)

return causal_mask[None, None, :, :]  # [1, 1, seq_len, total_len]
```

---

## 实际数据流对比

### Prefill 阶段 (seq_len=7)

**错误 mask**：
```
[[[True, True, True, True, True, True, True],   ← 位置0看到所有（包括未来！）
  [True, True, True, True, True, True, True],   ← 位置1看到所有
  ...
  [True, True, True, True, True, True, True]]]
```

**正确 causal mask**：
```
[[[  0, -inf, -inf, -inf, -inf, -inf, -inf],   ← 位置0只看自己
  [  0,   0, -inf, -inf, -inf, -inf, -inf],   ← 位置1看0,1
  [  0,   0,   0, -inf, -inf, -inf, -inf],   ← 位置2看0,1,2
  [  0,   0,   0,   0, -inf, -inf, -inf],
  [  0,   0,   0,   0,   0, -inf, -inf],
  [  0,   0,   0,   0,   0,   0, -inf],
  [  0,   0,   0,   0,   0,   0,   0]]]      ← 位置6看全部
```

### Decode 阶段 (生成第1个新token)

```
input_ids: [新生成的token]  ← 只有1个token
cache_position: [7]          ← 位置索引是7
attention_mask: [1, 8]       ← [prefill的7个 + 新的1个]

create_causal_mask 返回:
[[[0, 0, 0, 0, 0, 0, 0, 0]]]  ← [1, 1, 1, 8]
                              新token可以看到所有历史token
```

---

## 修复效果对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| **输出** | `I hope you are you are you are...` | `Sure, here's a light-hearted joke for you: Why don` |
| **连贯性** | 重复乱码 | 连贯英文 |
| **因果性** | ❌ 破坏 | ✅ 保证 |

---

## 关键知识点总结

1. **Padding Mask vs Causal Mask**：
   - Padding Mask：标记填充位置，所有真实token互相可见
   - Causal Mask：阻止看到未来token，下三角可见

2. **手动扩展的危险**：
   ```python
   # 危险！会创建全True矩阵
   attention_mask.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, seq_len)
   ```

3. **使用官方 API**：
   ```python
   from transformers.masking_utils import create_causal_mask
   causal_mask = create_causal_mask(...)
   ```

4. **Decode 阶段必须更新 mask**：
   - 每生成一个token，attention_mask长度+1
   - 让模型知道新的KV cache长度

5. **SDPA vs Eager**：
   - SDPA：返回None，内部处理causal
   - Eager：返回显式4D causal mask

---

## 相关文件

- `10-fiddler-main/src/fiddler/deepseek.py` - 修复后的实现
- `实验日志.md` - 修复记录
