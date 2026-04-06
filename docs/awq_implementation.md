# AWQ量化实现设计方案

## 一、修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `opensource/deepseek.py` | 1. 添加AWQ量化类 2. 修改`__init__`初始化量化配置 3. 修改`_async_load_expert()`实现量化传输 4. 修改`calc_expert_mem()`内存计算 |
| `opensource/qwen.py` | 同上（DeepSeek逻辑迁移） |
| `opensource/moon.py` | 同上（DeepSeek逻辑迁移） |
| `opensource/mixtral.py` | 同上（DeepSeek逻辑迁移） |

## 二、整体流程变化

### 修改前流程
```
CPU专家权重(FP16) → weight.copy_() → GPU占位专家(FP16) → 执行计算
```

### 修改后流程
```
CPU专家权重(FP16) → [AWQ量化] → INT8权重 + scale → [传输] → GPU(INT8存储) → [计算前自动反量化] → 执行计算
```

### 新增关键函数
1. `AWQQuantizer` 类 - 负责AWQ量化算法实现
2. `quantize_expert_weights()` - 量化单个专家权重
3. `load_quantized_expert()` - 加载量化权重到GPU
4. `calc_quantized_mem()` - 计算量化后的内存占用

## 三、AWQ算法实现步骤

### 步骤1：计算activation范围（保护显著权重）
```python
def get_activation_range(layer, calib_data):
    """使用校准数据获取每层activation的范围"""
    # 收集forward过程中的中间activation
    # 记录每个channel的最大值
```

### 步骤2：计算最优量化scale
```python
def compute_awq_scale(weight, activation_range):
    """
    AWQ核心：根据activation范围计算最优scale
    公式：scale = max(activation_range) / (max_quant_value * weight.max())
    """
```

### 步骤3：应用量化
```python
def quantize_tensor(tensor, scale, zero_point=0, bits=4):
    """将FP16 tensor量化为INT4/INT8"""
    quantized = torch.round(tensor / scale + zero_point)
    clipped = torch.clamp(quantized, min=-max_val, max=max_val)
    return clipped.to(torch.int8), scale
```

### 步骤4：反量化（算子自动处理）
```python
# 使用torch.nn.quantized.Linear替代普通Linear
# 计算时自动执行：dequant -> matmul -> quant
```

## 四、量化配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `quant_bits` | 4 | 量化位数（4或8） |
| `group_size` | 128 | 量化分组大小 |
| `zero_point` | True | 是否使用zero_point |
| `calib_samples` | 512 | 校准样本数 |

## 五、内存计算调整

```python
# 修改 calc_expert_mem()
quantization_factor = 1.0 / (quant_bits / 16)  # INT4=0.25, INT8=0.5
expert_mem = original_expert_mem * quantization_factor
```

## 六、注意事项

1. **先在单个模型测试** - 建议从DeepSeek开始
2. **校准数据准备** - 需要准备少量prompt用于校准
3. **验证精度损失** - 对比量化前后输出差异
4. **兼容现有卸载逻辑** - 保持CPU/GPU卸载机制不变