# MoE混合专家模型GPU卸载关键知识点

## 1. 日志文件数据分析

### 1.1 expert_stats.txt - 层处理时间统计
- **内容**: 每一层的总处理时间、中间层时间、最终层时间
- **用途**: 分析每层的性能瓶颈，找出处理时间异常的层
- **格式**:
  ```
  层 1: 93.39ms (, 中间层: 38.48ms, 最终层: 54.91ms, %)
  ```
- **时间定义**:
  | 时间项 | 含义 |
  |--------|------|
  | **中间层时间** | 专家FFN计算时间：token被路由到选定专家后，在CPU/GPU上执行实际前馈网络计算的总时间 |
  | **最终层时间** | 路由+合并时间：包含① 路由决策时间（Gate层选择哪些专家处理哪些token）；② 输出合并时间（各专家计算结果加权合并 + 残差连接 + LayerNorm等收尾操作） |

  MoE单层执行流程：`输入 → 路由计算 → 分发token → 专家FFN计算 → 合并输出 → 残差+LayerNorm → 输出`
  - 中间层时间 ≈ 专家FFN计算总耗时
  - 最终层时间 ≈ 路由决策 + 输出合并 + 收尾操作耗时

  瓶颈定位规则：
  - 中间层占比高 → 专家计算是瓶颈
  - 最终层占比高 → 路由/调度/合并开销较大

### 1.2 linshi.txt - 线程时间统计
- **内容**: 每层的GPU线程时间、CPU线程时间、并行时间、并行度
- **用途**: 评估GPU-CPU并行处理效率，并行度越接近2越好
- **格式**:
  ```
  Layer 1 Thread Time Stats:
  GPU Thread Time: 52.85ms
  CPU Thread Time: 37.41ms
  Parallel Time: 53.58ms
  Parallel Degree: 1.68x
  ```

---

## 2. 负载不均衡调整机制

### 2.1 核心思想
基于时间成本动态决定哪些专家从CPU搬运到GPU

### 2.2 关键参数
| 参数 | 含义 | 说明 |
|------|------|------|
| `e` | 搬运开销 | 专家从CPU→GPU的时间(ms) |
| `tg` | GPU计算开销 | 单个专家在GPU上的计算时间(ms) |
| `TA` | 总CPU时间 | 所有专家都在CPU上的总时间 |
| `TG` | 搬运+GPU时间 | 搬运前i+1个专家的时间 + GPU计算 |
| `TC` | 剩余CPU时间 | 剩余专家在CPU上的时间 |

### 2.3 决策逻辑
```python
# 按token数量降序排序专家
sorted_experts = sorted(expert_token_counts.items(), key=lambda x: x[1], reverse=True)

# 逐个判断
for i in range(n-1):
    TG = (1 + i) * e + tg
    TC = TC - cpu_time_table[token_count]
    
    if TG < TC:
        # 搬运到GPU更快
        ondemand_experts.append(expert_id)
    else:
        # 不满足条件，停止
        break
```

---

## 3. ondemand专家计算方式

### 3.1 完整流程
1. **统计token数**: 统计每个专家处理的token数量
2. **按token排序**: 按token数量降序排序，处理多的优先
3. **过滤GPU专家**: 去掉已经在GPU上的专家
4. **动态决策**: 基于时间成本判断哪些需要搬运
5. **加入列表**: 将需要搬运的专家加入`ondemand_experts`

### 3.2 关键点
- **只考虑CPU上的专家**: 已在GPU上的专家不参与计算
- **优先处理token多的**: token多的专家搬运收益更大
- **提前终止**: 一旦搬运不再有利，立即停止

---

## 4. 热点专家更新机制

### 4.1 热点专家定义
处理token数量最多的专家

### 4.2 更新流程
```python
# 第一步：记录当前迭代（每层处理时）
self.current_iter_expert_stats[i_layer] = {
    'expert_ids': [e[0] for e in sorted_experts],
    'token_counts': [e[1] for e in sorted_experts]
}

# 第二步：get_hot_expert()更新
def get_hot_expert(self):
    # 从current_iter获取统计
    expert_ids = self.current_iter_expert_stats[layer_id]['expert_ids']
    token_counts = self.current_iter_expert_stats[layer_id]['token_counts']
    
    # 按token数量降序排序
    sorted_experts = sorted(zip(expert_ids, token_counts), key=lambda x: x[1], reverse=True)
    
    # 保存到hot_experts
    hot_experts[layer_id] = [expert[0] for expert in sorted_experts]
    
    # 备份到last_iter，清空current_iter
    self.last_iter_expert_stats[layer_id] = {...}
    self.current_iter_expert_stats[layer_id]['expert_ids'].clear()
```

### 4.3 重要提示
⚠️ **热点专家不会自动保存到 `hot/deep.txt`**
- 代码中只有**统计**逻辑，没有**保存**逻辑
- 如需持久化，需手动添加保存代码

---

## 5. microbench.py使用和参数测量

### 5.1 运行命令
```bash
cd opensource
python microbench.py --model "/mnt/g/Models/DeepSeek-v2-lite-chat"
```

### 5.2 测量内容
| 测量项 | 说明 | 输出文件 |
|--------|------|----------|
| 专家搬运开销 | CPU→GPU专家权重拷贝时间 | microdeepseekreal.txt |
| GPU计算开销 | 单个专家在GPU上的计算时间 | microdeepseekreal.txt |
| CPU计算开销 | 不同token数下的CPU专家计算时间 | microdeepseek.txt |
| 性能图表 | 可视化对比 | ioreal_deepseek.png |

### 5.3 测量结果示例
```
Token Count,Expert Transfer Time(ms),GPU Computation Time(ms),CPU Computation Time(ms)
1,1.1141,0.1974,NaN
2,1.1141,0.1090,NaN
...
```

### 5.4 参数更新位置
`deepseek.py` 第1092-1093行：
```python
e = 1.11   # 搬运开销(m秒) - 测量值: 1.1141
tg = 0.20   # GPU计算开销(m秒) - 测量值: ~0.1-0.25
```

---

## 6. buffer_factor的作用

### 6.1 为什么需要buffer_factor？
预留显存空间，避免推理时OOM（Out Of Memory）

### 6.2 buffer_factor = 0.7的含义
只使用70%的可用显存，预留30%

### 6.3 预留空间的用途
| 用途 | 说明 |
|------|------|
| **KV缓存** | 自回归生成时缓存key和value |
| **中间计算结果** | 注意力计算、MLP计算的临时显存 |
| **防止OOM** | 避免推理过程中突然显存溢出 |

### 6.4 计算示例
```
可用显存(95%): 19.12 GB
无buffer可容纳专家: 1186个
buffer_factor: 0.7
最终专家数量: 830个 (1186 × 0.7)
```

### 6.5 如何调整
修改 `deepseek.py` 第616行：
```python
# 原来是：
buffer_factor = 0.7  # 使用70%的可用显存

# 可以改为：
buffer_factor = 0.85  # 使用85%的可用显存
```

---

## 7. 占位专家核心原理

### 7.1 背景痛点
1. 显存不足：专家太多，全部加载到GPU显存不够
2. 加载延迟高：按需从CPU加载有数十毫秒延迟
3. 显存碎片化：频繁分配/释放产生碎片

### 7.2 核心思路
提前在GPU上预分配固定数量的"空壳专家"，循环复用，只替换权重

### 7.3 关键技术
| 技术 | 作用 |
|------|------|
| **预分配占位壳** | 显存分配只做一次，避免碎片 |
| **Pinned Memory** | 锁定CPU内存，加速DMA传输 |
| **in-place拷贝** | 直接写预分配地址，无分配开销 |
| **双向映射** | expert_to_placeholder / placeholder_to_expert |

### 7.4 配合预取
- 用当前层输出预测下一层需要的专家
- 后台异步将专家拷贝到占位专家
- 隐藏加载延迟

---

## 8. 重要提醒

### 8.1 硬编码参数必须重新测量
- `e` 和 `tg` 与硬件强相关
- 换机器必须先跑 `microbench.py`
- 不要直接使用原代码中的值

### 8.2 热点专家不会自动保存
- 代码只统计不保存
- 如需持久化需手动添加保存逻辑

### 8.3 buffer_factor是必要的
- 虽然看似浪费显存
- 但预留空间对稳定运行很重要
- 可根据实际情况调整（0.7-0.9之间）
