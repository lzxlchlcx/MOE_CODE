# Microbench dtype 选择与一致性

## 问题背景

在使用 `transformers.AutoModelForCausalLM.from_pretrained()` 加载模型时，`torch_dtype` 参数决定了模型权重在内存中的精度格式。Microbenchmark 测量的三个关键参数（e、tg、cpu_time_table）直接用于 ondemand 调度器的决策，因此 **测量时的 dtype 必须与实际推理时的 dtype 一致**。

## 为什么必须指定 dtype

不指定 `torch_dtype` 时，Transformers 使用模型仓库 `config.json` 中的默认值，大多数 LLM 默认为 **`float32`**（4 bytes/param），而实际推理通常用 **`bfloat16`**（2 bytes/param）。

| 维度 | 不指定 (float32) | 指定 (bfloat16) |
|------|------------------|-----------------|
| 内存占用 | 4 bytes/param | 2 bytes/param |
| 传输时间 (e) | 偏大约 2x | 准确 |
| GPU 计算时间 (tg) | 无 Tensor Core 加速，严重偏大 | 走 Tensor Core 管线，准确 |
| CPU 计算时间 | 计算量翻倍，偏大 | 准确 |

用 float32 测出的参数代入调度器，会导致 `TG < TC` 决策严重偏差，专家搬运策略完全次优。

## bf16 存储但 fp16 加载的影响

模型权重以 bf16 存储、但加载时指定 `torch_dtype=torch.float16`：

```python
# from_pretrained 内部等价于
weight_fp16 = weight_bf16.to(torch.float16)  # 隐式转换，不会报错
```

| 维度 | 影响 |
|------|------|
| 正确性 | 几乎无影响，bf16→fp16 数值差异在 MoE 推理中可忽略 |
| 权重体积 | 相同（都是 2 bytes/param） |
| 传输时间 (e) | 相同（总字节数一样） |
| GPU 计算时间 (tg) | **有差异** — fp16 和 bf16 在 GPU 上的 Tensor Core 吞吐不同 |
| CPU 计算时间 | **有差异** — CPU 对 fp16 和 bf16 的计算路径不同 |
| 调度一致性 | **不一致** — 项目所有 Fiddler 类统一用 bf16 推理 |

即使 bf16 和 fp16 都是 2 bytes，同一 GPU 上两者的计算延迟也有微小差别（CUDA kernel 编译路径不同、中间累加精度舍入行为不同），导致测量数据不能准确反映实际推理性能。

## 正确做法：从 config.json 读取 torch_dtype

模型的 `config.json` 中有 `torch_dtype` 字段，记录了模型训练/保存时使用的精度。加载时应读取该字段，保证测量环境与模型原始精度一致：

```python
import json
import os
import torch
import transformers

def load_model(model_path):
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        dtype_str = cfg.get("torch_dtype", "bfloat16")  # fallback to bf16
        dtype = getattr(torch, dtype_str)                # "bfloat16" → torch.bfloat16
    else:
        dtype = torch.bfloat16

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="cpu", trust_remote_code=True
    )
    return hf_model, dtype
```

### 各模型实际 dtype

| 模型 | config.json 中 torch_dtype | Fiddler 类中硬编码 |
|------|---------------------------|-------------------|
| DeepSeek-V2-lite | `bfloat16` | `torch.bfloat16` (`deepseek.py:62`) |
| Qwen3-30B-A3B | `bfloat16` | `torch.bfloat16` (`qwen.py:33`) |
| Moonlight-16B-A3B | `bfloat16` | `torch.bfloat16` (`moon.py:31`) |
| Mixtral-8x7B | `bfloat16` | `torch.bfloat16` (`mixtral.py:31`) |

## 关联文件

- `myself/microbench_v2.py` — 独立 microbench 实现，已采用从 config.json 读取 dtype
- `opensource/microbench.py` — 原始 microbench，通过 Fiddler 类间接指定 bf16
- `opensource/config.py` — 配置管理
- `opensource/deepseek.py` / `qwen.py` / `moon.py` / `mixtral.py` — 各模型 Fiddler 类
