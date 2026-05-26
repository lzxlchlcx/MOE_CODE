# DynamicCache.get_usable_length AttributeError 问题分析

## 问题描述

运行 DeepSeek 模型时出现以下错误：

```
AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'. Did you mean: 'get_seq_length'?
```

错误发生在：
- 文件：`~/.cache/huggingface/modules/transformers_modules/DeepSeek-v2-lite-chat/modeling_deepseek.py`
- 行号：853
- 代码：`kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)`

## 根因分析

### 1. 存在两个不同的 modeling_deepseek.py

| 来源 | 路径 | Cache API |
|------|------|-----------|
| **模型仓库自带** | `~/.cache/huggingface/modules/transformers_modules/DeepSeek-v2-lite-chat/modeling_deepseek.py` | `get_usable_length()` |
| **transformers 库内置** | `site-packages/transformers/models/deepseek_v2/modeling_deepseek_v2.py` | `get_seq_length()` |

### 2. 为什么加载的是仓库自带的版本？

模型的 `config.json` 配置了 `auto_map`：

```json
"auto_map": {
    "AutoConfig": "configuration_deepseek.DeepseekV2Config",
    "AutoModel": "modeling_deepseek.DeepseekV2Model",
    "AutoModelForCausalLM": "modeling_deepseek.DeepseekV2ForCausalLM"
}
```

代码加载时使用了 `trust_remote_code=True`：

```python
self.model = transformers.AutoModelForCausalLM.from_pretrained(
    args.model,
    trust_remote_code=True,  # 启用远程代码
)
```

**`trust_remote_code=True` + `auto_map`** → HuggingFace 优先使用**模型仓库自带**的 `modeling_deepseek.py`，而非 transformers 库内置的实现。

### 3. 版本不匹配

- 模型仓库自带的 `modeling_deepseek.py` 是 DeepSeek 团队早期发布的，调用 `get_usable_length(kv_seq_len, layer_idx)`
- transformers **4.55.0** 的 `DynamicCache` 只有 `get_seq_length()`，没有 `get_usable_length()`
- `deepseek.py` 第 12-25 行的 `_patch_dynamic_cache()` 本应修补此问题，但可能因执行顺序等原因未生效

### 4. 为什么库内置版本用 `get_seq_length`？

transformers 4.55.0 采用模块化架构重构（modular architecture），新版内置的 `modeling_deepseek_v2.py` 已适配新的 Cache API，直接使用 `get_seq_length()` 而非 `get_usable_length()`。

## 总结

> **根本原因**：`trust_remote_code=True` 导致加载了 DeepSeek 模型仓库中旧版的 `modeling_deepseek.py`，该文件调用 `get_usable_length()`，但 transformers 4.55.0 的 `DynamicCache` 没有此方法。

## 可能的修复方案

1. **修复 patch 使其生效**：检查 `_patch_dynamic_cache()` 的执行时机和作用域
2. **替换为库内置模型**：设置 `trust_remote_code=False` 使用 transformers 内置的 `modeling_deepseek_v2.py`
3. **删除缓存中的旧文件**：强制使用新版实现

---

*记录时间：2026-05-07*
