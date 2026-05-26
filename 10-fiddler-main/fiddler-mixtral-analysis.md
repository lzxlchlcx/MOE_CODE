# Fiddler 项目分析报告

## 一、项目概述

**Fiddler** 是一个针对 Mixture-of-Experts (MoE) 架构大语言模型的快速推理系统，核心论文：[Fiddler: CPU-GPU Orchestration for Fast Local Inference of MoE Models](https://arxiv.org/abs/2402.07033)。

核心目标：在单张 24GB GPU 上运行未量化的 Mixtral-8x7B（>90GB 参数），达到 >3 token/s 的推理速度。

---

## 二、核心模块：`src/fiddler/mixtral.py` — `FiddlerMixtral` 类

### 2.1 核心思想

传统 offloading 方法（如 Mixtral Offloading）将 **专家权重** 从 CPU 内存拷贝到 GPU 执行，权重体积大（每个专家 `3 × 4096 × 14336` 参数），数据搬运开销极高。

Fiddler 的创新：将 **激活值** 从 GPU 传到 CPU，直接在 CPU 上执行专家计算，再将输出传回 GPU。激活值体积远小于权重（`batch_size × 4096` vs `3 × 4096 × 14336`），大幅降低通信开销。

### 2.2 类结构与方法

#### `__init__(self, args)`
- 加载 Mixtral-8x7B 模型（bfloat16 精度）
- 创建一个 GPU 上的 `expert_placeholder`，用于临时加载不在 GPU 上的专家权重
- 初始化 tokenizer、KV Cache（`DynamicCache`）
- 计算当前 GPU 显存能容纳多少个专家（`calc_n_expert_on_gpu`）
- 按预分析的"热度"优先级将最常用的专家常驻 GPU（`set_expert_loc`）

#### `bring_non_expert_to_gpu(self)`
将非专家层全部加载到 GPU：embedding、self-attention、layer norm、gate、lm_head。

#### `calc_n_expert_on_gpu(self)`
根据单个专家参数量和 GPU 剩余显存（95% 总显存 - 已占用），计算可容纳的专家数量。

#### `set_expert_loc(self, n_expert_on_gpu, popular_experts=None)`
维护 `expert_loc[n_layer, n_expert]` 矩阵（0=CPU, 1=GPU）。内置一个按使用频率排序的专家列表（来自预分析 profile），按优先级从高到低将专家标记为 GPU 驻留，直到达到 GPU 容量上限。

#### `mixtral_forward(self, input_ids, position_ids, is_decode)` — 核心前向传播
逐层执行 Mixtral 模型，对每一层：
1. Self-Attention（在 GPU 上执行）
2. Router Gate 计算（在 GPU 上执行），得到 top-2 路由专家及权重
3. **专家执行**（两种模式）：

**模式 A：`cpu_offload=0`（基线模式）**
- 所有专家在 GPU 上执行
- 不在 GPU 上的专家先通过 `expert_placeholder.load_state_dict()` 加载权重到 GPU，执行完毕后移回 CPU

**模式 B：`cpu_offload=1`（Fiddler 核心模式）**
- 对每层 8 个专家，预估每个专家在 CPU 和 GPU 上的延迟：
  - CPU 延迟 = token 数 × `latency_cpu`
  - GPU 延迟 = `latency_gpu`（已驻留 GPU 的专家延迟为 0）
- 遍历所有 `2^8 = 256` 种 CPU/GPU 分配方案，选总延迟最小的配置
- GPU 专家和 CPU 专家并行执行，结果汇总后传回

4. LayerNorm + lm_head 得到 logits

#### `run_expert_at_cpu(self, i_layer, i_expert, inps, routing_weights)`
在 CPU 上执行指定专家的前向计算。

#### `generate(self, text, output_token, input_token)`
自回归生成循环：
- 支持 beam search（通过 `beam_width` 参数）
- 第一轮为 prefill 阶段（处理完整输入），后续为 decode 阶段（逐 token 生成）
- 使用 `DynamicCache` 管理 KV Cache
- 返回 prefill_time、decode_time、专家命中率（hit_rate）

#### `tokenize(self, text)`
将文本 tokenize，并按 `beam_width` 复制为多份 batch。

### 2.3 关键设计决策

| 设计点 | 决策 | 原因 |
|--------|------|------|
| 激活值传输 vs 权重传输 | 传输激活值 | 激活值大小远小于权重 |
| 专家驻留策略 | 按热度优先级预加载 | 热门专家命中率更高 |
| CPU/GPU 负载均衡 | 穷举 256 种方案 | 只有 8 个专家，穷举代价可忽略 |
| CPU 延迟估计 | token 数 × 单位延迟 | CPU 计算时间与 token 数线性相关 |
| GPU 延迟估计 | 固定值 | GPU 计算速度极快，延迟近似常数 |

---

## 三、推理入口：`src/fiddler/infer.py`

命令行入口脚本，解析参数后实例化 `FiddlerMixtral` 并调用 `generate()` 进行推理。

参数说明：
- `--model`：模型路径，默认 `mistralai/Mixtral-8x7B-v0.1`
- `--cpu-offload`：0=纯 GPU 基线，1=Fiddler CPU 卸载模式
- `--input`：输入提示文本
- `--n-token`：生成 token 数
- `--beam-width`：beam search 宽度

---

## 四、基准测试：`benchmarks/`

### 4.1 `benchmarks/microbench.py` — 微基准测试

测量 CPU-GPU 系统中各项基础操作耗时，用于验证 Fiddler 的设计假设：

| 测试项 | 测量内容 |
|--------|----------|
| 权重拷贝 CPU→GPU | `load_state_dict` 将专家权重加载到 GPU placeholder |
| 权重拷贝 GPU→CPU | 反向操作 |
| 激活拷贝 CPU→GPU | `(1, 4096)` 张量传输 |
| 激活拷贝 GPU→CPU | 反向操作 |
| GPU 专家执行 | 不同 batch size 下的专家计算耗时 |
| CPU 专家执行 | 不同 batch size 下的专家计算耗时 |

**设计假设验证**：证明激活值传输耗时远小于权重传输耗时，这是 Fiddler 方案的理论基础。

### 4.2 `benchmarks/latency.py` — 端到端延迟测试

使用 ShareGPT 数据集对 Fiddler 进行端到端性能评测：
- 对 `input_token`(16/32/64/128) × `output_token`(16~512) 的各种组合
- 每组采样 10 次取平均
- 记录 prefill_time、decode_time、hit_rate、token/s

### 4.3 `benchmarks/eval-baseline.py` — 基线对比评测

对两个竞争框架进行公平对比：

**DeepSpeed-MII 基线**：
- 使用 ZeRO-3 offloading 将参数卸载到 CPU
- 通过 `deepspeed.initialize()` 初始化推理引擎

**Mixtral Offloading 基线**：
- 调用修改版的 `mixtral_offloading/` 库
- 支持可选 4-bit/2-bit 量化（HQQ）

评测流程与 `latency.py` 类似，使用 ShareGPT 数据集，测量不同输入输出长度下的 token/s。

### 4.4 `benchmarks/mixtral_offloading/` — Mixtral Offloading 基线库

[Mixtral Offloading](https://github.com/dvmazur/mixtral-offloading) 的修改版副本，作为基线依赖：

| 文件 | 功能 |
|------|------|
| `src/build_model.py` | 构建带 offloading 的 Mixtral 模型 |
| `src/expert_cache.py` | 专家缓存策略（LRU 等） |
| `src/expert_wrapper.py` | 专家层包装器，处理 GPU/CPU 调度 |
| `src/custom_layers.py` | 自定义 MoE 层实现 |
| `src/triton_kernels.py` | Triton GPU kernel 加速 |
| `src/packing.py` | 张量打包工具 |
| `src/utils.py` | 辅助工具函数 |

---

## 五、整体架构关系

```
Fiddler 推理流程:
                                  GPU                        CPU
                              ┌─────────┐
  Input ─→ Tokenize ─→ Embed ─→ Attention ─→ LayerNorm ─→ Gate (路由)
                              │                                    │
                              │   ┌─── 热门专家 (GPU驻留) ──→ GPU计算
                              │   │
                              ├───┤
                              │   │
                              │   └─── 冷门专家 ──→ 传激活值到CPU ──→ CPU计算 ──→ 传回GPU
                              │                                    │
                              └─── 汇总结果 ←──────────────────────┘
                                      │
                                      ▼
                              LayerNorm → lm_head → 输出 token
```

## 六、性能表现（论文数据）

| 环境 | Fiddler | DeepSpeed-MII | Mixtral Offloading |
|------|---------|---------------|---------------------|
| RTX 6000 (24GB) + 48核 CPU | **~3 token/s** | ~0.15 token/s | ~0.37 token/s |
| L4 (24GB) + 32核 CPU | **~3 token/s** | ~0.13 token/s | ~0.30 token/s |

相比基线方法，Fiddler 平均加速 **8-22 倍**。

## 七、Fiddler 与 Mixtral Offloading 实现差异对比

### 7.1 核心策略差异

| 维度 | Mixtral Offloading | Fiddler |
|------|-------------------|---------|
| **GPU 缺专家时的做法** | 把专家**权重**从 CPU 拷贝到 GPU，在 GPU 上计算 | 把**激活值**从 GPU 传到 CPU，在 CPU 上计算 |
| **单次数据搬运量** | 权重：`3×4096×14336` 参数/专家（~330MB bf16） | 激活值：`batch_size×4096`（~8KB） |
| **CPU 角色** | 仅作**存储**（存放放不下的专家权重） | **存储 + 计算**（CPU 直接执行专家前向） |
| **量化支持** | 支持 HQQ 2/4-bit 量化 | 不支持，保持 16-bit 原精度 |
| **数据流方向** | 权重 CPU→GPU（单向搬运） | 激活值 GPU→CPU→GPU（双向传输） |

### 7.2 缓存/调度机制差异

**Mixtral Offloading**（`expert_cache.py`）：
- 维护一个 GPU 上的**专家池**（`main_modules`），大小固定
- 使用 **LRU 缓存**策略管理 GPU 上的专家
- 当所需专家不在 GPU 时，通过 `_swap()` 将 LRU 专家换出、目标专家换入
- 换入换出操作为**权重搬运**（CPU→GPU 的 `storage.copy_`）
- 利用 `buffer` 做**异步重叠**：在处理当前专家时，提前加载下一个专家的权重
- 按层分组驱逐（`eviction_group`），同一层的专家互相替换

**Fiddler**（`mixtral.py`）：
- 根据预 profile 的**热度排名**静态决定哪些专家常驻 GPU
- 无动态缓存/驱逐机制，GPU 上的专家在推理过程中**不变化**
- 运行时通过**穷举 256 种分配方案**（`2^8`）决定当前层的 8 个专家分别在 CPU 还是 GPU 上执行
- 目标是**最小化单层最大延迟**（CPU 和 GPU 并行，取 max）

### 7.3 模型加载差异

**Mixtral Offloading**：
- 将专家权重存入连续的 `UntypedStorage`，通过偏移量管理
- `MixtralExpertWrapper` 将 w1/w2/w3 权重打包到一块连续内存中
- 支持 safetensors 格式，逐个专家按需加载
- 可选 HQQ 量化，使用 Triton kernel 加速量化矩阵乘

**Fiddler**：
- 直接使用 HuggingFace Transformers 原生 `MixtralForCausalLM` 加载
- 不修改模型结构，只通过 `expert_loc` 矩阵控制专家位置
- 用一个 `expert_placeholder`（深拷贝第一个专家）在 GPU 上临时加载权重

### 7.4 前向传播差异

**Mixtral Offloading**（`SparseMoEWrapper.forward`）：
- 通过 `expert_cache.load_experts()` 迭代获取专家
- 缓存系统自动处理换入换出，对上层透明
- 专家统一在 GPU 上执行

**Fiddler**（`mixtral_forward`）：
- 手动展开 MoE 的路由逻辑（gate → topk → 逐专家执行 → index_add）
- GPU 专家和 CPU 专家**并行**：GPU 专家先启动，CPU 专家同时在 CPU 侧计算
- 使用 `run_expert_at_cpu()` 在 CPU 上执行完整的 MLP（w1→SiLU×w3→w2）

### 7.5 性能差异根因

Fiddler 快一个数量级的根本原因：**避免了专家权重的 CPU↔GPU 传输**。在单批次推理场景下，激活值传输（~8KB）比权重传输（~330MB）快约 4 个数量级，虽然 CPU 计算比 GPU 慢，但通信节省远超计算差异。

```
Mixtral Offloading 数据流:
  GPU 缺专家 → 搬运权重 CPU→GPU (~330MB) → GPU 计算 → 结果留在 GPU

Fiddler 数据流:
  GPU 缺专家 → 搬运激活 GPU→CPU (~8KB) → CPU 计算 → 搬运结果 CPU→GPU (~8KB)
```

### 7.6 特性对比总结

| 特性 | Mixtral Offloading | Fiddler |
|------|-------------------|---------|
| 缓存策略 | 动态 LRU | 静态热度排名 |
| CPU 计算 | 不使用 | 核心计算资源 |
| 量化 | 支持（HQQ 2/4-bit） | 不支持 |
| 异步重叠 | 权重预加载 | CPU/GPU 并行计算 |
| 模型修改 | 自定义层 + 存储管理 | 原生 Transformers 模型 |
| 适用场景 | 量化模型 + 低显存 | 全精度 + 低显存 + 多核 CPU |
| 实现复杂度 | 高（自定义缓存系统） | 低（手动展开前向传播） |

---

## 八、局限性

- 仅支持 16-bit Mixtral-8x7B 模型
- CPU 专家计算依赖 PyTorch 实现，需要 CPU 支持 AVX512 指令集才能获得较好性能
- 专家热度列表为硬编码，需预先 profile
- CPU/GPU 延迟参数（`latency_cpu=7`, `latency_gpu=70`）为经验值
