# Microbench 知识点整理

## 1. model.tokenizer 和 model.tokenizer.input_ids

`model.tokenizer` 是 Hugging Face Transformers 模型自带的**分词器**对象（例如 `AutoTokenizer`），用于将文本转换为模型可理解的数值 ID。

`model.tokenizer(...)` 调用返回一个 `BatchEncoding` 对象，包含多个字段：
- `input_ids`：分词后的 token ID 序列
- `attention_mask`：指示哪些位置是真实 token（1），哪些是填充（0）
- 其他可能的字段（如 `token_type_ids`）

`.input_ids` 提取其中的张量，然后 `.to(model.dev)` 将其移动到模型所在的设备（GPU/CPU）。

**示例**：对于 `"Hello world"`，分词后可能得到 `[101, 7592, 2088, 102]`（BERT 风格），这就是 `input_ids`。

---

## 2. model.model 和 model 的结构

`model` 是一个 **Fiddler 包装类实例**（如 `FiddlerMixtral`、`FiddlerDeepSeekV2` 等），不是 HuggingFace 原始模型。

### model 包含的核心对象

| 属性 | 类型 | 说明 |
|------|------|------|
| `model.tokenizer` | `AutoTokenizer` | 分词器，将文本转为 token ID |
| `model.model` | HuggingFace 模型 | **实际的Transformer模型**（已剥离lm_head） |
| `model.dev` | `torch.device` | 设备（通常是 `cuda:0`） |
| `model.lm_head` | `nn.Linear` | 语言模型头（输出层） |

### 模型加载代码（以 mixtral.py 为例）

```python
# 1. 加载完整模型
self.model = transformers.MixtralForCausalLM.from_pretrained(...)  
# 此时 self.model 是 MixtralForCausalLM 实例，包含 .model 和 .lm_head

# 2. 提取 lm_head
self.lm_head = self.model.lm_head

# 3. 剥离出实际模型结构（transformer backbone）
self.model = self.model.model  
# 此时 self.model 是 MixtralModel，包含 .layers、.embed_tokens 等
```

### 调用关系

```python
model.tokenizer(...)           # → AutoTokenizer.__call__()
# 返回 BatchEncoding，包含 input_ids 张量

model.model.embed_tokens(input_ids)  # → 调用模型的词嵌入层
# embed_tokens 是 transformer 的 embedding 层（nn.Embedding）
```

**总结**: `model.model` 才是原始的 HuggingFace Transformer 模型，它包含 `layers`（Transformer层列表）、`embed_tokens`（词嵌入层）等核心组件。

---

## 3. hidden_states 的维度

`hidden_states = model.model.embed_tokens(input_ids)` 返回的维度是 **`[batch_size, seq_len, hidden_dim]`**

| 维度 | 值 | 说明 |
|------|-----|------|
| `batch_size` | 3 | `sample_texts` 里有 3 个样本 |
| `seq_len` | 最大 token 长度 | 3 个样本分词后的最大序列长度（padding 后统一） |
| `hidden_dim` | 模型隐藏层维度 | 例如 DeepSeek-V2 是 4096，Qwen3-30B 是 5120 |

**具体例子**：
- `input_ids` 形状：`[3, 10]`（batch=3，每个序列 10 个 token）
- `hidden_dim` = 4096
- 则 `hidden_states.shape = [3, 10, 4096]`

---

## 4. gate 返回值和维度

根据 `microbench.py:67` 的代码：

```python
selected_experts, routing_weights, _ = gate(hidden_states)
```

`gate` 返回 3 个值，维度如下：

| 返回值 | 形状 | 说明 |
|--------|------|------|
| `selected_experts` | `[num_tokens, top_k]` | 每个 token 选中的 top-k 个专家 ID |
| `routing_weights` | `[num_tokens, top_k]` | 每个 token 对选中专家的路由权重 |
| `_`（丢弃） | - | 可能是辅助信息（如注意力权重） |

**具体例子**：
- `hidden_states` 形状：`[3, 10, 4096]`（batch=3, seq_len=10）
- `top_k` = 2（默认配置）

则：
```
selected_experts.shape    = [30, 2]   # 30 = 3*10 个 token，每个选 2 个专家
routing_weights.shape     = [30, 2]   # 对应的路由权重
```

---

## 5. mask 操作详解

```python
mask = (selected_experts == expert_id).any(dim=1)
```

这是一个 **PyTorch 布尔索引 + 维度归约** 操作：

### 步骤分解

#### 步骤 1: `selected_experts == expert_id`

假设 `selected_experts` 形状 `[30, 2]`（30 tokens × top_k=2），值为：
```
tensor([[0, 3],
        [1, 0],
        [2, 1],        ...
       ])
```

与 `expert_id=0` 比较后得到布尔张量：
```
tensor([[True, True],   # 0 == 0 为 True
        [False, True],  # 1 == 0 为 False, 0 == 0 为 True
        [False, False], ...
       ])
```

#### 步骤 2: `.any(dim=1)`

在 `dim=1`（最后一个维度）上做**或运算**，相当于"该 token 是否选中了此专家"：

```
tensor([True,   # [True, True] → 任意为 True
        True,   # [False, True] → 任意为 True
        False,  # [False, False] → 全为 False
        ...])
```

#### 步骤 3: `.sum()`

统计 `True` 的数量（True=1, False=0），得到选中共 0 个专家的 token 数。

**一句话总结**：`mask.sum()` = **选中了指定专家的 token 数量**。

---

## 6. .item() 的作用

`.item()` 将 **PyTorch 张量** 转换为 **Python 标量（int/float）**。

### 是否可以使用？

**可以不用**，但会有区别：

| 方式 | 类型 | 用途 |
|------|------|------|
| `mask.sum().item()` | Python `int` | 纯数值，用于普通计算 |
| `mask.sum()` | `torch.Tensor` | 仍是张量，用于 GPU/张量运算 |

### 实际影响

```python
expert_counts[expert_id.item()] = mask.sum().item()
```

这里 `.item()` 是因为：
1. **字典键需要 Python 标量**：`expert_counts` 的 key 必须是 `int`，不能用 `torch.Tensor`
2. **后续累加操作**：`expert_hot_data[key] += count` 中的 `count` 必须是 Python 数值

### 如果不用 .item()？

```python
# 这样会报错：字典键不能是 Tensor
expert_counts[expert_id.item()] = mask.sum()  
# TypeError: unhashable type: 'torch.Tensor'

# 或者后续累加时：
expert_hot_data[key] += mask.sum()  
# 不会报错，但 expert_hot_data[key] 会变成 Tensor 类型
```

**总结**：`.item()` 是 **从 GPU 张量到 Python 标量的转换**，在需要纯数值（字典键、数值累加、打印等）时必须使用。如果后续操作都在 PyTorch 张量层面进行，则可以省略。

---

## 7. expert_hot_data 的格式

```python
expert_hot_data = {
    (layer, expert_id): 调用次数,
    ...
}
```

### 具体结构

| 键 | 值 |
|---|---|
| `(1, 0)` | 64 | 第1层专家0被调用64次 |
| `(1, 1)` | 60 | 第1层专家1被调用60次 |
| `(1, 2)` | 56 | 第1层专家2被调用56次 |
| ... | ... | ... |
| `(4, 7)` | 36 | 第4层专家7被调用36次 |

### 数据类型

- **键**: `tuple` (int, int) — 即 `(层索引, 专家ID)`
- **值**: `int` — 该专家被所有采样 token 选中的总次数

### 使用场景

该字典会被用于 `set_expert_loc()` 决定哪些专家常驻 GPU（热点专家排序靠前，优先加载到 GPU）。

---

## 8. 后续处理涉及的 Python 语法

```python
expert_counts = {}                                    # 字典初始化
for expert_id in selected_experts.unique():           # for 循环 + PyTorch方法
    mask = (selected_experts == expert_id).any(dim=1) # PyTorch 布尔索引
    expert_counts[expert_id.item()] = mask.sum().item() # 字典赋值

for expert_id, count in expert_counts.items():         # 字典 .items() 遍历
    key = (i_layer, expert_id)                         # 元组作为键
    if key not in expert_hot_data:                     # 成员检查 "not in"
        expert_hot_data[key] = 0                      # 字典初始化
    expert_hot_data[key] += count                     # 增量赋值
```

### 语法点总结

| 语法 | 说明 |
|------|------|
| `{}` | 空字典字面量 |
| `dict[key] = value` | 字典赋值 |
| `dict.items()` | 遍历键值对 |
| `key not in dict` | 成员检查 |
| `dict[key] += value` | 增量赋值 |
| `(a, b)` | 元组作字典键 |
| `for x in iterable:` | for 循环 |
| `.item()` | PyTorch 张量转 Python 标量 |
| `.unique()` | PyTorch 获取唯一值 |
| `.any(dim=1)` / `.sum()` | PyTorch 维度操作 |

---

## 9. 占位专家（Expert Placeholder）

在 `microbench.py` 中使用 `copy.deepcopy` 创建占位专家：

```python
expert_placeholder = copy.deepcopy(first_layer_experts[0]).to(dev)
```

### 作用

通过预分配 GPU 显存，避免每次拷贝时重新分配显存，从而**只测量真实拷贝时间**。

### 原理对比

| 方式 | 说明 | 问题 |
|------|------|------|
| 直接拷贝（无预分配） | 每次 `.to(dev)` 都需先分配显存 | 分配时间算入拷贝时间，测量不准确 |
| 使用占位专家（预分配） | expert_placeholder 已在 GPU 分配好显存 | 直接覆盖，只测真实拷贝 |

### 步骤

1. **创建占位符**：通过 `deepcopy` 复制专家结构并移动到 GPU（显存一次性分配）
2. **复用显存**：后续拷贝时，`dst` 使用已分配的 `expert_placeholder`，直接覆盖数据
3. **避免分配开销**：不调用 `.to(dev)`，只用 `.copy_()` 写入已有显存

### 代码示例

```python
# 1. 创建占位专家（预分配）
expert_placeholder = copy.deepcopy(first_layer_experts[0]).to(dev)

# 2. 后续拷贝复用预分配的显存
for name in ["gate_proj", "up_proj", "down_proj"]:
    dst = getattr(expert_placeholder, name).weight.data  # 已在GPU分配
    src = getattr(first_layer_experts[i], name).weight.data  # CPU上的权重
    dst.copy_(src)  # 直接覆盖，无分配开销
```

---

## 10. copy 库

`import copy` 是 Python 标准库，用于对象拷贝。

### 核心函数

| 函数 | 说明 | 嵌套对象 |
|------|------|----------|
| `copy.copy(x)` | 浅拷贝，只拷贝顶层 | 共享引用 |
| `copy.deepcopy(x)` | 深拷贝，递归拷贝所有层级 | 完全独立 |
| `copy(x)` | `copy.copy()` 的简写 | 同上 |

### 浅拷贝 vs 深拷贝

```python
original = [[1, 2, 3], [4, 5, 6]]

# 浅拷贝：外层独立，嵌套列表共享引用
shallow = copy.copy(original)
shallow[0][0] = 999  # 会影响 original

# 深拷贝：完全独立
deep = copy.deepcopy(original)
deep[0][0] = 888  # 不会影响 original
```

### 在本项目中的应用

```python
# 复制专家模块的完整结构（权重 + 层结构）
expert_placeholder = copy.deepcopy(first_layer_experts[0]).to(dev)
```

---

## 11. getattr 动态获取属性

`getattr(obj, name)` 是 Python 内置函数，通过**字符串名称**动态访问对象成员。

### 基本用法

```python
class Expert:
    def __init__(self):
        self.gate_proj = "gate_weight"
        self.up_proj = "up_weight"
        self.down_proj = "down_weight"

expert = Expert()

# 两种写法等价
expert.gate_proj                    # "gate_weight"
getattr(expert, "gate_proj")       # "gate_weight"

# 属性不存在时返回默认值
getattr(expert, "nonexistent", "default")  # "default"
```

### 在本项目中的应用

```python
# 循环遍历多个权重
for name in ["gate_proj", "up_proj", "down_proj"]:
    w = getattr(first_layer_experts[i], name)  # 动态获取
```

### 能获取的内容

| 类型 | 示例 | 结果 |
|------|------|------|
| 属性 | `getattr(obj, "x")` | `obj.x` |
| 方法 | `getattr(obj, "forward")` | `obj.forward` |
| 子模块 | `getattr(obj, "layer1")` | `obj.layer1` |
| 权重 | `getattr(weight, "weight")` | 权重 Tensor |

---

## 12. weight.data vs weight

在 PyTorch 中，层的权重访问有两种方式：

| 表达式 | 类型 | 说明 |
|--------|------|------|
| `w.weight` | `Parameter` | 包含 grad_fn、元数据，可训练 |
| `w.weight.data` | `Tensor` | 纯粹底层数据，无梯度追踪 |

### 区别

```python
type(w.weight)        # torch.nn.Parameter
w.weight.grad_fn     # 有，记录梯度来源
w.weight.requires_grad  # True

type(w.weight.data)  # torch.Tensor
w.weight.data.grad_fn  # None
w.weight.data.requires_grad  # False
```

### 为什么用 `.data`？

在只需要数值拷贝、不需要梯度追踪时，使用 `.data` 更直接：

```python
src = w.weight.data        # 获取纯粹数据
pinned = src.pin_memory()  # 锁定内存
```

**注意**：`.data` 是历史 API，现代 PyTorch 推荐使用 `.detach()` 代替。

---

## 13. pin_memory 锁定内存

`tensor.pin_memory()` 将 Tensor 锁定到**页锁定内存**（pinned memory），防止被 swap 到磁盘。

### 内存类型对比

| 类型 | 说明 | CPU→GPU 传输 |
|------|------|-------------|
| 普通内存 | 可分页，能被 swap | 需要中转，速度慢 |
| Pinned Memory | 锁定在物理 RAM | DMA 直传，速度快 |

### 传输路径对比

```
普通传输（无 pin_memory）：
CPU Memory → Pageable Buffer → Pinned Buffer → GPU (多一步)

Pinned 传输（使用 pin_memory）：
CPU Memory (pinned) → DMA → GPU (直接传输)
```

### 使用示例

```python
# 普通 Tensor
x = torch.randn(1000)
x.is_pinned()  # False

# 锁定到页锁定内存
pinned_x = x.pin_memory()
x.is_pinned()  # True

# 传输到 GPU（已锁定的更快）
y = pinned_x.to("cuda:0")
```

### 注意事项

- **占用物理内存**：pinned 内存无法被 swap，会消耗物理 RAM
- **只对 CPU tensor 有效**：GPU tensor 调用会报错
- **配合 `non_blocking=True` 使用**：`tensor.to(device, non_blocking=True)` 可进一步加速

---

## 14. non_blocking 异步传输

`non_blocking=True` 启用**异步传输**，让 CPU 调用后立即返回，传输在后台进行。

### 同步 vs 异步

```python
# 同步传输（默认）
tensor.to("cuda")  # 阻塞直到传输完成

# 异步传输
tensor.to("cuda", non_blocking=True)  # 立即返回，传输在后台进行
```

### 工作流程对比

```
同步：
CPU ──→ [发起 DMA] ──→ 等待 ──→ DMA 完成

异步 (non_blocking=True)：
CPU ──→ [发起 DMA] ──→ 立即返回
                        ↓
                   DMA 在后台进行
```

### 常见组合

```python
# 最佳实践：pin_memory + non_blocking
src = tensor.pin_memory()
dst = src.to("cuda", non_blocking=True)
```

### 注意事项

| 事项 | 说明 |
|------|------|
| **必须配合 `pin_memory()`** | 异步传输只对 pinned tensor 有效 |
| **需要手动同步** | 使用后需 `torch.cuda.synchronize()` 确保完成 |
| **小数据传输可能更慢** | 调度开销占比增加 |

---

## 15. torch.cuda.synchronize() 同步

`torch.cuda.synchronize()` 等待所有已提交的 CUDA 任务完成，确保时间测量准确。

### 为什么需要？

CUDA 操作是**异步**的：

```python
tick = time.time()
dst.copy_(src)  # 启动后立即返回（不等完成）
tock = time.time()
print(tock - tick)  # 几乎为0！实际拷贝在后台进行
```

### 使用后

```python
torch.cuda.synchronize()  # 等待之前所有操作完成
tick = time.time()
dst.copy_(src)           # 启动拷贝
torch.cuda.synchronize()  # 等待拷贝完成
tock = time.time()
print(tock - tick)  # 真实的拷贝时间
```

### 标准微基准测量流程

```python
torch.cuda.synchronize()  # 1. 清空队列（等待所有之前操作完成）
tick = time.time()        # 2. 开始计时
operation()               # 3. 执行待测操作
torch.cuda.synchronize()  # 4. 等待操作真正完成
elapsed = time.time() - tick  # 5. 记录真实耗时
```

### 其他同步方式

| 方法 | 作用 |
|------|------|
| `torch.cuda.synchronize()` | 同步当前设备所有流 |
| `torch.cuda.Event()` | 记录/等待特定时间点 |
| `tensor.item()` | 隐式同步（取值时） |

---

## 16. copy_ 原地拷贝

`dst.copy_(src)` 是 PyTorch 的**原地拷贝**方法，将源张量数据复制到目标张量。

### 调用形式

| 用法 | 说明 |
|------|------|
| `dst.copy_(src)` | 同步拷贝（当前设备） |
| `dst.copy_(src, non_blocking=True)` | 异步拷贝（需 src 已 pin_memory） |

### 核心行为

- **原地操作**：修改 `dst`，不创建新张量
- **跨设备**：支持 CPU↔GPU、GPU↔GPU 拷贝
- **DMA 条件**：src 在 pinned memory 时自动使用 DMA 加速

### DMA 使用条件

| 条件 | 是否使用 DMA |
|------|-------------|
| dst 在 GPU，src 在 CPU（普通内存） | ❌ 需要中转，慢 |
| dst 在 GPU，src 在 CPU（**pinned**） | ✅ DMA 直传，快 |
| dst 在 GPU，src 在 GPU | ✅ GPU 内拷贝 |

### 与其他拷贝方式对比

| 方法 | 是否原地 | 是否异步 | 典型用途 |
|------|---------|----------|----------|
| `a.copy_(b)` | 是 | 否（可搭配 non_blocking） | 覆盖写入 |
| `a = b.clone()` | 否 | 否 | 生成新副本 |
| `a[:] = b` | 是 | 否 | 原地赋值 |

---

## 17. 为什么只测量专家 #2

在 microbench.py 的传输时间测量循环中：

```python
for _ in token_counts:
    for i in range(2, 3):  # i 始终为 2，只测专家 #2
        ...
```

### 原因

| 原因 | 解释 |
|------|------|
| **所有专家结构相同** | MoE 中每个专家都是相同结构的 MLP，权重形状完全一致 |
| **测量一个具有代表性** | 专家 #2 的传输/计算时间 ≈ 其他专家的时间 |
| **节省时间** | 避免 8 个专家 × 64 种 token 数 = 512 次重复测量 |

### 结论

MoE 中所有专家是**同构的**（identical），只需测量一个有代表性的即可，结果适用于所有专家。这是一种**采样策略**，而非遗漏。
