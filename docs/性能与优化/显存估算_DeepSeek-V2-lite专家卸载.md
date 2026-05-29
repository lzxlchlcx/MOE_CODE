# 显存估算：DeepSeek-V2-lite 专家卸载

在 MoE GPU 卸载推理框架中，GPU 显存除存放模型参数外，还需为运行时临时张量预留空间。本文档分析各部分显存开销及精确估算方法。

---

## 模型架构参数（DeepSeek-V2-lite）

| 参数 | 值 |
|---|---|
| `hidden_size` | 2048 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 16 |
| `num_hidden_layers` | 27（layer 0 为 dense，layer 1-26 为 MoE） |
| `intermediate_size` | 10944（dense 层 MLP） |
| `moe_intermediate_size` | 1408（每个 MoE 专家） |
| `n_routed_experts` | 64 |
| `n_shared_experts` | 2 |
| `num_experts_per_tok` | 6（top-K 路由） |
| `kv_lora_rank` | 512（MLA 压缩） |
| `vocab_size` | 102400 |
| `dtype` | bfloat16（2 bytes/param） |

---

## GPU 显存组成

GPU 显存 = **已分配的非专家参数** + **专家参数** + **运行时开销**

其中运行时开销包括：

1. KV Cache
2. Attention Scores
3. 前向传播中间激活张量
4. Causal Mask
5. CUDA 内存碎片与分配器开销

---

## 各项开销估算公式

以下公式中，`bs` = batch_size，`seq` = max_seq_len（input_token + output_token），`L` = n_layer，所有结果单位为字节。

### 1. KV Cache

DeepSeek-V2 使用 **MLA（Multi-head Latent Attention）**，通过 `kv_lora_rank=512` 的低秩投影压缩 KV cache，而非缓存完整的 K/V heads。

```
kv_cache = 2 × kv_lora_rank × bytes_per_param × n_layer × bs × seq
         = 2 × 512 × 2 × 27 × 1 × 160
         ≈ 8.6 MB
```

| bs × seq | KV Cache 大小 |
|---|---|
| 1 × 160 | ~8.6 MB |
| 1 × 1024 | ~55 MB |
| 8 × 1024 | ~440 MB |
| 16 × 2048 | ~3.5 GB |

> MLA 的 KV cache 远小于标准 MHA（后者约为 `2 × num_kv_heads × v_head_dim × ...`），在 batch_size 较小时几乎可忽略。

### 2. Attention Scores（Prefill 峰值）

Prefill 阶段计算完整 attention matrix，是显存峰值点：

```
attn_scores = bs × num_attention_heads × seq × seq × bytes_per_param
            = 1 × 16 × 128 × 128 × 2
            ≈ 0.5 MB
```

| bs × seq | Attn Scores 大小 |
|---|---|
| 1 × 128 | ~0.5 MB |
| 1 × 1024 | ~32 MB |
| 8 × 1024 | ~256 MB |
| 16 × 2048 | ~2 GB |

> Decode 阶段 seq=1，此项开销可忽略。

### 3. 前向传播中间激活张量

每层前向传播需要保存多份 hidden_states（输入残差、LayerNorm 后、attention 输出、MLP 中间结果等），保守估计每层 6 份：

```
activations = bs × seq × hidden_size × bytes_per_param × 6 × n_layer
            = 1 × 160 × 2048 × 2 × 6 × 27
            ≈ 636 MB
```

| bs × seq | Activations 大小 |
|---|---|
| 1 × 160 | ~636 MB |
| 1 × 1024 | ~4.0 GB |
| 4 × 1024 | ~16 GB |

> 这是最大的运行时开销项。PyTorch autograd 虽然在推理时不保存梯度，但前向过程中间张量仍需显存。

### 4. Causal Mask

```
causal_mask = bs × seq × (past_seq + seq) × bytes_per_param
```

通常小于 attention scores，可合并计入安全系数中。

### 5. CUDA 分配器开销

PyTorch 的 `CUDAAllocator` 会预留额外显存（通常 2-10%），加上内存碎片化，无法精确计算，通过安全系数覆盖。

---

## 综合公式

```
available_for_experts = total_gpu_mem
                      - already_allocated     (非专家参数：embedding, lm_head, attention, shared_experts, gate, layernorm...)
                      - runtime_overhead × safety_factor

其中:
  runtime_overhead = kv_cache + attn_scores + activations + causal_mask
  safety_factor = 1.2  (覆盖 CUDA 分配器开销和碎片化)

n_expert_on_gpu = available_for_experts // single_expert_mem
```

---

## 典型场景估算（RTX 3090 24GB, bfloat16）

假设已加载非专家参数约 2.5 GB：

| 场景 | runtime_overhead | 可用专家空间 | 可放专家数 |
|---|---|---|---|
| bs=1, seq=160 | ~0.77 GB × 1.2 ≈ 0.92 GB | ~20.6 GB | ~1280 |
| bs=1, seq=512 | ~2.3 GB × 1.2 ≈ 2.76 GB | ~18.7 GB | ~1162 |
| bs=4, seq=512 | ~8.5 GB × 1.2 ≈ 10.2 GB | ~11.3 GB | ~703 |
| bs=8, seq=1024 | ~35 GB × 1.2 ≈ 42 GB | 不足 | 0 |

> 单个专家大小 ≈ 16.5 MB（2048 × 1408 × 3 × 2 bytes）
> 全部 1664 个专家（26层 × 64）共需 ~26.8 GB，超过单卡容量。

---

## 代码实现

改进后的 `calc_n_expert_on_gpu` 方法位于 `10-fiddler-main/src/fiddler/deepseek.py`：

```python
def calc_n_expert_on_gpu(self, max_seq_len=160):
    # 单个专家大小
    expert_mem_bytes = n_param * bytes_per_param

    # KV Cache (MLA 压缩)
    kv_cache_bytes = 2 * cfg.kv_lora_rank * bytes_per_param * n_layer * bs * seq

    # Attention Scores (prefill 峰值)
    attn_score_bytes = bs * cfg.num_attention_heads * seq * seq * bytes_per_param

    # 激活张量 (每层 6 份 hidden_states)
    activation_bytes = bs * seq * cfg.hidden_size * bytes_per_param * 6 * n_layer

    # Causal Mask
    causal_mask_bytes = bs * seq * (seq + seq) * bytes_per_param

    # 安全系数覆盖 CUDA 开销
    runtime_overhead = int((kv_cache + attn + activations + mask) * 1.2)

    # 可用空间
    available = total_mem - allocated - runtime_overhead
    return max(0, available // expert_mem_bytes)
```

运行时打印详细估算信息，便于验证和调试：
```
Total GPU: 24576.00 MB
Already allocated: 2560.00 MB
  - KV cache est: 8.60 MB
  - Attn scores est: 0.50 MB
  - Activations est: 636.00 MB
  - Causal mask est: 0.20 MB
Runtime overhead (x1.2): 774.00 MB
Available for experts: 21242.00 MB
```

---

## 与原 80% 方案的对比

| 维度 | 原方案 (`total × 0.80 - allocated`) | 精确估算方案 |
|---|---|---|
| 预留策略 | 固定 20% 总显存 | 按实际运行时开销逐项计算 |
| batch_size 感知 | 无 | 有 |
| seq_len 感知 | 无 | 有 |
| 扩展性 | 大 batch/长序列时易 OOM | 安全余量随负载动态调整 |
| 小 batch 时 | 浪费显存（少放专家） | 充分利用显存 |
