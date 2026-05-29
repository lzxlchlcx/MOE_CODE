# 性能分析：Microbench 测量与调优

本文档整合了 Microbench 测量方法、数据准确性分析、dtype 选择以及系统性能瓶颈分析。

---

# 第一部分：Microbench dtype 选择与一致性

## 为什么必须指定 dtype

不指定 `torch_dtype` 时，Transformers 使用模型仓库 `config.json` 中的默认值，大多数 LLM 默认为 **float32**（4 bytes/param），而实际推理通常用 **bfloat16**（2 bytes/param）。

| 维度 | 不指定 (float32) | 指定 (bfloat16) |
|------|------------------|-----------------|
| 内存占用 | 4 bytes/param | 2 bytes/param |
| 传输时间 (e) | 偏大约 2x | 准确 |
| GPU 计算时间 (tg) | 无 Tensor Core 加速，严重偏大 | 走 Tensor Core 管线，准确 |
| CPU 计算时间 | 计算量翻倍，偏大 | 准确 |

用 float32 测出的参数代入调度器，会导致 `TG < TC` 决策严重偏差。

## bf16 存储但 fp16 加载的影响

即使 bf16 和 fp16 都是 2 bytes，同一 GPU 上两者的计算延迟也有微小差别（CUDA kernel 编译路径不同、中间累加精度舍入行为不同），导致测量数据不能准确反映实际推理性能。

## 正确做法：从 config.json 读取 torch_dtype

```python
import json, os, torch, transformers

def load_model(model_path):
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        dtype_str = cfg.get("torch_dtype", "bfloat16")
        dtype = getattr(torch, dtype_str)
    else:
        dtype = torch.bfloat16

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="cpu", trust_remote_code=True
    )
    return hf_model, dtype
```

## 各模型实际 dtype

| 模型 | config.json 中 torch_dtype | Fiddler 类中硬编码 |
|------|---------------------------|-------------------|
| DeepSeek-V2-lite | `bfloat16` | `torch.bfloat16` |
| Qwen3-30B-A3B | `bfloat16` | `torch.bfloat16` |
| Moonlight-16B-A3B | `bfloat16` | `torch.bfloat16` |
| Mixtral-8x7B | `bfloat16` | `torch.bfloat16` |

---

# 第二部分：Microbench 测量方法

## 运行命令
```bash
cd opensource
python microbench.py --model "/mnt/g/Models/DeepSeek-v2-lite-chat"
```

## 测量内容

| 测量项 | 方法 | 含义 |
|--------|------|------|
| e (传输时间) | `expert.to("cuda") + synchronize()` | CPU→GPU搬运一个专家权重的时间 |
| tg (GPU计算) | `expert(inps) + synchronize()` | GPU上计算一个专家的时间 |
| cpu_time_table | 遍历token=1~64测量 | CPU上不同token数的专家计算时间 |

## 实测数据示例（DeepSeek-V2-lite, RTX 4090D）

| Token数 | CPU计算时间 |
|---------|------------|
| 1 | 0.074ms |
| 4 | 0.296ms |
| 10 | 0.762ms |
| 64 | 5.87ms |
| - | e = 1.09ms, tg = 0.14ms |

## 关键假设
- GPU计算与token数无关（GPU并行处理）
- CPU计算与token数线性相关（串行处理）
- 搬运成本固定（每个专家约1.09ms）

## 配置存储结构
```json
{
  "gpu_name": "NVIDIA GeForce RTX 4090 D",
  "gpu_memory_gb": 23.98,
  "transfer_time_ms": 1.09,
  "gpu_compute_time_ms": 0.14,
  "cpu_time_table": [0.074, 0.148, ...]
}
```

---

# 第三部分：数据准确性问题分析

## 问题1：单位转换错误（严重bug）

| 环节 | 值 |
|------|-----|
| microbench测量 | token=1 时 CPU = 0.074ms |
| JSON存储 | cpu_time_table[0] = 0.074021 |
| scheduler使用 | 认为 = 0.074秒 = 74ms |

**根因**: `microbench.py:282` 将秒乘1000转毫秒，但 scheduler 按秒使用。

**影响**: scheduler认为CPU需74ms >> 搬运成本1.09ms，导致ondemand过度激进。

## 问题2：前6项数据不可靠

- `cpu_time_table[0~5]` 使用 `window_size=7` 卷积平滑，前6个token填0
- batch_size=4 decode时，每个专家处理1-4个token，依赖的前6项是估算值

## 问题3：测量环境与实际不一致

- microbench 使用单个专家预热后的稳定值
- 实际每层多个CPU专家并发，存在线程竞争和内存带宽瓶颈
- 实测CPU专家平均6.30ms vs 表中值0.30ms

## 问题4：实测值包含额外开销

- 实测 expert_compute-cpu: 平均6.30ms（2594次）
- 表中值: token=4时0.30ms
- 差异原因：GPU→CPU→GPU数据搬运开销 + 并发竞争 + 可能处理更多token

---

# 第四部分：性能分析报告（2026.4.15）

## 4.1 预取调度策略有效性

| 类别 | 占比 | 次数 |
|------|------|------|
| GPU常驻 | 69.8% | - |
| Prefetch | 3.8% | 2773次 |
| Ondemand | 6.2% | 4548次 |
| CPU | 20.2% | - |

预取效果不显著原因：
1. 热点专家已在GPU常驻，预取跳过
2. 预取预测基于单层输出，准确率有限
3. 预取数量太少（3.8%），对整体影响微弱

## 4.2 CPU-GPU负载均衡

**当前负载分布（batch_size=4）**：

| 指标 | 数值 | 占比 |
|------|------|------|
| GPU总工作时间 | 109997ms | 91.7% |
| CPU总工作时间 | 84499ms | 70.4% |
| GPU空闲 | 9966ms | 8.3% |
| CPU空闲 | 35463ms | 29.6% |

**GPU/CPU负载比**: 1.30:1（GPU过载）
**并行度**: 1.62x（理论最优1.77x，损失8.5%）

## 4.3 吞吐量

| 指标 | 值 |
|------|-----|
| 单GPU吞吐 | 2.26 token/s |
| 并行吞吐 | 3.48 token/s |
| 提升 | 55% |

---

# 第五部分：瓶颈定位

## 瓶颈1：GPU线程同步阻塞（P0）

**位置**: `deepseek.py:380-417` `_async_ondemand`

- ondemand搬运在GPU线程内同步执行
- 4548次 × ~1.09ms ≈ 5秒加到GPU线程
- CPU线程快速完成后等待GPU

## 瓶颈2：CPU计算时间表不准确（P1）

**位置**: `microbench.py` cpu_time_table

- 前6项是线性插值估算值
- 实测CPU时间远高于表中值（6.30ms vs 0.30ms）

## 瓶颈3：热点策略占用GPU过多（P2）

- GPU上放置830个专家（49.9%）
- batch_size=1时命中率86%，CPU几乎空闲
- batch_size=4时命中率70%，负载仍不均衡

## 瓶颈4：placeholder数量有限（P3）

- 只有6个placeholder可用于ondemand搬运
- 限制ondemand的并行度

## 瓶颈优先级

| 优先级 | 瓶颈 | 影响 | 优化难度 |
|--------|------|------|----------|
| **P0** | GPU线程同步阻塞 | 严重（GPU过载） | 中等 |
| **P1** | CPU时间表不准确 | 中等（决策错误） | 低 |
| **P2** | 热点策略占用GPU过多 | 中等（负载不均衡） | 高 |
| **P3** | placeholder数量有限 | 低（架构限制） | 高 |

---

# 第六部分：改进建议

## 修正调度参数
```python
e = 1.39  # 使用实测值替代 1.11
# 或动态测量
e = measure_copy_time()
```

## 优化并行度
将串行join改为批量处理：
```python
# 当前(串行): for t in threads: t.join()
# 改进: 先全部start，再全部join
```

## 动态调整预取cache
```python
if hit_rate < 0.5:
    self.cache = min(self.cache + 5, 64)
else:
    self.cache = max(self.cache - 2, 3)
```

---

# 第七部分：关联文件

| 文件 | 说明 |
|------|------|
| `myself/microbench_v2.py` | 独立microbench，已采用从config.json读取dtype |
| `opensource/microbench.py` | 原始microbench，通过Fiddler类间接指定bf16 |
| `opensource/config.py` | 配置管理 |
| `opensource/scheduler.py` | ondemand决策算法 |
| `opensource/deepseek.py` | 模型推理主类 |
