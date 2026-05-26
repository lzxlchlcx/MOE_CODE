# PyTorch、Transformers 与代码知识点参考

本文档整合了项目代码中涉及的 PyTorch API、Transformers 库用法、Python 语法技巧以及核心代码机制（占位专家、微基准测试等）。

---

# 第一部分：PyTorch 基础函数

## 1.1 张量操作

### `torch.bfloat16`
```python
self.dtype = torch.bfloat16
```
- 16位浮点数，比 float32 节省一半显存
- 适合深度学习推理，对精度损失不敏感
- DeepSeek 等模型常用此精度

### `torch.device()`
```python
self.dev = torch.device("cuda:0")
```
- 指定计算设备，`"cuda:0"` 使用第1块GPU，`"cpu"` 使用CPU
- 后续通过 `.to(self.dev)` 将张量/模型移动到指定设备

### `torch.tensor()` / `torch.arange()`
```python
row_idx = torch.tensor([i * self.beam_width for i in range(...)])
new_position_ids = torch.arange(self.past_key_values_length,
    self.past_key_values_length + input_ids.shape[1],
    dtype=torch.long, device=self.dev)
```
- `torch.tensor()`: 从Python列表创建张量
- `torch.arange(start, end, dtype, device)`: 创建等差数列张量

### `.unsqueeze()` / `.expand()` / `.view()`
```python
x = torch.tensor([1, 2, 3])    # shape: [3]
x = x.unsqueeze(0)              # shape: [1, 3]  — 增加维度
x = x.expand(5, -1)             # shape: [5, 3]  — 扩展维度（不复制数据）
x = x.view(-1)                  # shape: [15]    — 改变形状
```

### `.to()`
```python
self.expert_placeholder = copy.deepcopy(...).to(self.dev)
```
- `tensor.to(device)` / `tensor.to(dtype)` / `tensor.to(device, dtype)`
- 移动设备或转换数据类型

### `.pin_memory()`
```python
pinned = src_weight_data_tensor.pin_memory()
```
- 将张量锁定在页锁定内存（Pinned Memory），防止被 swap 到磁盘
- GPU可以直接DMA访问，CPU→GPU传输速度提升30%以上

### `.copy_()`
```python
dst.copy_(src)
```
- 原地拷贝，将源张量值复制到目标张量（不改变目标内存地址）

---

## 1.2 张量索引与掩码操作

### `.any(dim=1)` — 维度归约
```python
mask = (selected_experts == expert_id).any(dim=1)
```
沿指定维度检查是否有任何元素为 True。

### `.nonzero()` / `.squeeze()`
```python
mask_index = mask.nonzero().squeeze(1)
```
- `.nonzero()`: 返回非零元素的索引
- `.squeeze()`: 移除大小为1的维度

### `.gather()`
```python
weights = routing_weights[flat_mask].gather(1, index_tensor)
```
沿指定维度收集值。

### `.index_add_()`
```python
inps_after_experts.index_add_(0, mask_index, expert_output)
```
在指定索引位置累加值（in-place操作）。

### 布尔掩码索引
```python
expert_input = inps[batch_mask].view(-1, hidden_dim)
```
使用布尔张量选择元素。

---

## 1.3 数学与神经网络函数

| 函数 | 用途 |
|------|------|
| `F.softmax(logits, dim=-1)` | 计算 softmax，转换为概率分布 |
| `torch.topk(logits, k, dim)` | 返回前 k 大的元素及其索引 |
| `torch.argmax(probs, dim=-1)` | 返回最大值的索引 |
| `F.one_hot(x, num_classes)` | 将索引转换为 one-hot 编码 |
| `.permute(2, 1, 0)` | 重新排列张量维度 |

---

## 1.4 GPU 相关函数

| 函数 | 用途 |
|------|------|
| `torch.cuda.get_device_properties(dev).total_memory` | 获取GPU总显存（字节） |
| `torch.cuda.memory_allocated(dev)` | 获取当前已分配显存 |
| `torch.cuda.synchronize()` | 等待GPU上所有任务完成 |
| `next(expert.parameters()).is_cuda` | 检查参数是否在GPU上 |

### 性能分析：`torch.profiler.profile()`
```python
prof = torch.profiler.profile(
    activities=[...],
    schedule=torch.profiler.schedule(...),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True, profile_memory=True, with_stack=True
)
```

### `@torch.no_grad()`
推理阶段禁用梯度计算，节省显存和计算资源。

---

## 1.5 模型与参数操作

```python
n_param = sum(p.numel() for p in fine_expert.parameters())
```
- `.parameters()`: 返回模型所有参数（生成器）
- `.numel()`: 返回张量元素总数

---

# 第二部分：Transformers 库函数

## 2.1 模型加载

### `AutoModelForCausalLM.from_pretrained()`
```python
self.model = transformers.AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=self.dtype, use_cache=True, trust_remote_code=True
)
```
- 自动加载因果语言模型
- `use_cache=True`: 使用KV缓存加速自回归生成
- `trust_remote_code=True`: 信任远程代码（自定义模型需要）

### `AutoTokenizer.from_pretrained()`
```python
self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
self.tokenizer.pad_token = self.tokenizer.eos_token  # 用eos代替pad
```

## 2.2 分词器操作

```python
encodings = self.tokenizer(text, padding=True, truncation=True,
    max_length=input_token, return_tensors="pt")
```
返回 `BatchEncoding`，包含 `input_ids`、`attention_mask` 等字段。

```python
self.tokenizer.decode(input_ids[i, :].tolist())  # token ID → 文本
```

## 2.3 model 和 model.model 的结构

`model` 是 **Fiddler 包装类实例**（如 `FiddlerMixtral`），不是 HuggingFace 原始模型：

| 属性 | 类型 | 说明 |
|------|------|------|
| `model.tokenizer` | `AutoTokenizer` | 分词器 |
| `model.model` | HuggingFace Model | Transformer backbone（含 `.layers`、`.embed_tokens`） |
| `model.lm_head` | `nn.Linear` | 语言模型头（输出层） |

加载流程：
```python
self.model = transformers.MixtralForCausalLM.from_pretrained(...)
self.lm_head = self.model.lm_head
self.model = self.model.model  # 剥离出 transformer backbone
```

## 2.4 hidden_states 维度

`hidden_states = model.model.embed_tokens(input_ids)` 返回 **`[batch_size, seq_len, hidden_dim]`**。

## 2.5 gate 返回值

```python
selected_experts, routing_weights, _ = gate(hidden_states)
```

| 返回值 | 形状 | 说明 |
|--------|------|------|
| `selected_experts` | `[num_tokens, top_k]` | 每个 token 选中的专家 ID |
| `routing_weights` | `[num_tokens, top_k]` | 对应的路由权重 |

## 2.6 KV缓存

```python
self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
```
自回归生成时缓存之前的 key 和 value，避免重复计算。

---

# 第三部分：Python 语法技巧

### 列表推导式与生成器
```python
popular_experts = [tuple(map(int, line.strip().split(','))) for line in f if line.strip()]
n_param = sum(p.numel() for p in fine_expert.parameters())
```

### 字典推导式
```python
self.cpu_expert_time_per_layer = {i: 0.0 for i in range(1, 27)}
```

### `lambda` + `sorted`
```python
sorted_experts = sorted(expert_data, key=lambda x: x[1], reverse=True)
```

### `zip()` / `enumerate()`
```python
expert_data = list(zip(expert_ids, token_counts))
for i_layer, layer in enumerate(self.model.layers):
```

### `nonlocal` 关键字
```python
def process_gpu_experts():
    nonlocal gpu_time  # 在嵌套函数中修改外层变量
```

### `threading` 多线程
```python
t = threading.Thread(target=process_single_expert, args=(i_expert,))
threads.append(t)
t.start()
for t in threads:
    t.join()  # 等待完成
```

---

# 第四部分：核心代码机制

## 4.1 占位专家（Expert Placeholder）

### 背景痛点
1. **显存不足**: MoE模型专家太多，全部加载显存不够
2. **加载延迟高**: 按需从CPU加载有数十毫秒延迟
3. **显存碎片化**: 频繁分配/释放产生碎片

### 核心思路
提前在GPU上预分配固定数量的"空壳专家"（结构与真实专家完全相同），循环复用，只替换权重。

### 初始化阶段：预分配占位壳
```python
first_layer_mlp = self.model.layers[1].mlp
self.expert_placeholder = copy.deepcopy(first_layer_mlp.experts[0]).to(self.dev)
# ... 共创建6个占位专家
```

### 映射机制
```python
self.expert_to_placeholder = {}         # (layer_idx, expert_id) → 占位专家
self.placeholder_to_expert = {           # 占位专家名 → 真实专家坐标
    'expert_placeholder': None, ...
}
self.expert_placeholder11_inused = False  # 使用状态标记
```

### 核心流程：权重替换
```python
def _async_ondemand(self, layer_idx, expert_id, target_placeholder):
    for name in ['gate_proj', 'up_proj', 'down_proj']:
        w = getattr(self.model.layers[layer_idx].mlp.experts[expert_id], name)
        pinned = w.weight.data.pin_memory()  # 锁页加速
        w.weight.data = pinned
    for name in ['gate_proj', 'up_proj', 'down_proj']:
        dst = getattr(target_placeholder, name).weight.data
        src = getattr(self.model.layers[layer_idx].mlp.experts[expert_id], name).weight.data
        dst.copy_(src)  # 直接覆盖预分配显存
```

### 复用机制
```python
def release_placeholder(self, layer_idx, expert_id):
    # 处理完一层后释放占位专家，供下一层使用
    if stored_expert and stored_expert[0] < layer_idx:
        setattr(self, f"{placeholder_name}_inused", False)
```

### 技术优势对比

| 对比项 | 普通加载方式 | 占位专家方式 |
|--------|--------------|--------------|
| 显存分配 | 每次加载都重新分配 | 仅初始化时分配一次 |
| 加载延迟 | 分配+拷贝时间 | 仅拷贝时间 |
| 显存碎片 | 频繁分配释放 | 固定位置，无碎片 |
| 并发能力 | 容易显存不足 | 固定数量，可控 |

---

## 4.2 weight.data vs weight

| 表达式 | 类型 | 说明 |
|--------|------|------|
| `w.weight` | `Parameter` | 包含 grad_fn、元数据，可训练 |
| `w.weight.data` | `Tensor` | 纯粹底层数据，无梯度追踪 |

项目用 `.data` 做数值拷贝，不需要梯度追踪。现代 PyTorch 推荐用 `.detach()` 代替。

---

## 4.3 non_blocking 异步传输

```python
dst.copy_(src, non_blocking=True)  # 异步拷贝
```

| 方式 | 行为 |
|------|------|
| 同步（默认） | 阻塞直到传输完成 |
| `non_blocking=True` | 立即返回，后台传输 |

必须配合 `pin_memory()` 使用，之后需 `torch.cuda.synchronize()` 确认完成。

---

## 4.4 标准时间测量模式

```python
torch.cuda.synchronize()  # 清空队列
tick = time.time()
operation()
torch.cuda.synchronize()  # 等待完成
elapsed = time.time() - tick  # 真实耗时
```

GPU操作是异步的，不 `synchronize()` 的话测量结果接近0。

---

## 4.5 mask 操作详解（专家token统计）

```python
mask = (selected_experts == expert_id).any(dim=1)
count = mask.sum().item()
```

步骤：
1. `selected_experts == expert_id` → 布尔矩阵 `[num_tokens, top_k]`
2. `.any(dim=1)` → 该 token 是否选中此专家 `[num_tokens]`
3. `.sum()` → 选中该专家的 token 数
4. `.item()` → 转为 Python int（字典键、累加需要）

---

## 4.6 expert_hot_data 格式

```python
expert_hot_data = {
    (layer, expert_id): 调用次数,
    ...
}
```
- 键: `tuple(int, int)` — (层索引, 专家ID)
- 值: `int` — 该专家被所有采样 token 选中的总次数
- 用于 `set_expert_loc()` 决定哪些专家常驻 GPU

---

## 4.7 为什么 microbench 只测量专家 #2

```python
for i in range(2, 3):  # i 始终为 2
```

MoE 中所有专家是**同构的**（相同结构的 MLP），测量一个具有代表性即可，结果适用于所有专家。

---

## 4.8 常用模式总结

### 设备迁移模式
```python
tensor = torch.tensor([1, 2, 3], device=self.dev)  # 创建时指定
tensor = tensor.to(self.dev)                         # 后迁移
if tensor.is_cuda: ...                               # 检查设备
```

### 掩码-提取-计算模式
```python
mask = (selected_experts == expert_id).any(dim=1)
expert_input = inps[mask].view(-1, hidden_dim)
expert_output = expert(expert_input)
weights = routing_weights[mask].gather(...)
expert_output = expert_output * weights
inps_after_experts.index_add_(0, mask.nonzero().squeeze(1), expert_output)
```

### copy 库
```python
copy.deepcopy(obj)   # 深拷贝，递归复制所有层级，完全独立
```

### getattr 动态获取属性
```python
for name in ["gate_proj", "up_proj", "down_proj"]:
    w = getattr(expert, name)  # 等价于 expert.gate_proj
```
