# MoE Code Offload 项目架构与文件功能说明文档

## 目录

1. [项目概述](#1-项目概述)
2. [文件清单](#2-文件清单)
3. [文件详情分析](#3-文件详情分析)
   - [3.1 deepseek.py](#31-deepseekpy)
   - [3.2 qwen.py](#32-qwenpy)
   - [3.3 moon.py](#33-moonpy)
   - [3.4 mixtral.py](#34-mixtralpy)
   - [3.5 latency.py](#35-latencypy)
   - [3.6 microbench.py](#36-microbenchpy)
   - [3.7 run_benchmark.sh](#37-run_benchmarksh)
   - [3.8 数据文件](#38-数据文件)
4. [项目架构设计](#4-项目架构设计)
5. [依赖关系图](#5-依赖关系图)

---

## 1. 项目概述

### 1.1 项目背景

本项目是一个**混合专家模型（Mixture of Experts, MoE）代码卸载基准测试系统**。项目针对多种主流MoE架构大语言模型（如DeepSeek、Qwen、Moon、Mixtral）进行GPU卸载性能测试与分析。

### 1.2 主要功能

- **模型卸载策略测试**：评估不同MoE模型在GPU内存受限情况下的卸载策略效果
- **延迟基准测试**：测量模型各层的实际推理延迟
- **微基准测试**：对模型中的专家模块进行细粒度性能分析
- **GPU内存估算**：计算不同模型配置下GPU显存需求

### 1.3 整体架构设计

项目采用**模块化设计**，每个模型对应一个独立的基准测试模块，共享通用的延迟测试框架。架构分为三个层次：

```
┌─────────────────────────────────────────┐
│         基准测试执行层 (run_benchmark.sh)        │
├─────────────────────────────────────────┤
│  模型基准测试层 (deepseek/qwen/moon/mixtral)     │
├─────────────────────────────────────────┤
│  性能分析层 (latency.py / microbench.py)         │
└─────────────────────────────────────────┘
```

---

## 2. 文件清单

| 序号 | 文件路径 | 文件类型 | 功能描述 |
|------|----------|----------|----------|
| 1 | deepseek.py | Python | DeepSeek MoE模型基准测试 |
| 2 | qwen.py | Python | Qwen MoE模型基准测试 |
| 3 | moon.py | Python | Moon MoE模型基准测试 |
| 4 | mixtral.py | Python | Mixtral MoE模型基准测试 |
| 5 | latency.py | Python | 延迟基准测试框架 |
| 6 | microbench.py | Python | 微基准测试框架 |
| 7 | run_benchmark.sh | Shell | 基准测试执行脚本 |
| 8 | deep_latency.txt | Text | DeepSeek延迟测试数据 |
| 9 | test.txt | Text | 通用测试数据 |
| 10 | microdeepseek.txt | Text | DeepSeek微基准数据 |
| 11 | microqwen.txt | Text | Qwen微基准数据 |
| 12 | micromoon.txt | Text | Moon微基准数据 |
| 13 | micromixtral.txt | Text | Mixtral微基准数据 |
| 14 | hot/deep.txt | Text | Hot场景深度数据 |

---

## 3. 文件详情分析

### 3.1 deepseek.py

#### a) 具体功能描述

DeepSeek MoE模型的GPU卸载基准测试模块。专门针对DeepSeek架构的MoE模型进行性能评估，计算专家模块在GPU上的最优部署策略。

#### b) 架构定位

- **层级**：模型基准测试层
- **位置**：与qwen.py、moon.py、mixtral.py并列
- **角色**：DeepSeek模型的性能评估器

#### c) 依赖关系

- **依赖于**：`latency.py`（延迟计算框架）
- **被依赖**：由`run_benchmark.sh`调用执行

#### d) 核心实现逻辑

1. **专家数量计算** (`calc_n_expert_on_gpu`):
   - 根据GPU可用内存和模型参数计算可容纳的专家数量
   - 公式：`n_expert_on_gpu = min(n_expert, int(gpu_mem / expert_mem))`

2. **专家内存计算** (`calc_expert_mem`):
   - 计算单个专家模块的内存占用
   - 公式：`expert_mem = 2 * num_experts * hidden_size * inter_size / (1024^2)` MB

3. **激活专家选择** (`select_active_experts`):
   - 根据top-k设置选择激活的专家
   - 默认配置：`top_k=6, n_routed_experts=256`

4. **基准测试执行**:
   - 遍历不同专家数量的配置
   - 测量每个配置下的延迟和吞吐量

#### e) 重要变量/函数说明

| 名称 | 类型 | 说明 |
|------|------|------|
| `n_expert_on_gpu` | int | GPU上容纳的专家数量 |
| `expert_mem` | float | 单个专家内存占用(MB) |
| `top_k` | int | 激活的专家数量（默认6） |
| `n_routed_experts` | int | 路由专家总数（DeepSeek默认256） |
| `hidden_size` | int | 隐藏层维度（DeepSeek默认7168） |
| `inter_size` | int | 中间层维度（默认前馈网络维度） |
| `gpu_mem` | float | GPU可用内存（MB） |
| `calc_n_expert_on_gpu()` | method | 计算GPU上最优专家数量 |
| `calc_expert_mem()` | method | 计算专家内存占用 |

---

### 3.2 qwen.py

#### a) 具体功能描述

Qwen MoE模型的GPU卸载基准测试模块。专门针对Qwen架构的MoE模型进行性能评估，考虑其特定的网络结构和专家配置。

#### b) 架构定位

- **层级**：模型基准测试层
- **位置**：与deepseek.py、moon.py、mixtral.py并列
- **角色**：Qwen模型的性能评估器

#### c) 依赖关系

- **依赖于**：`latency.py`（延迟计算框架）
- **被依赖**：由`run_benchmark.sh`调用执行

#### d) 核心实现逻辑

1. **专家数量计算**:
   - 与DeepSeek类似的计算逻辑
   - 针对Qwen模型参数进行调整

2. **Qwen特定参数**:
   - `n_routed_experts=57`：Qwen路由专家总数
   - `top_k=8`：Qwen默认激活8个专家

3. **基准测试配置**:
   - 支持不同GPU内存配置的测试
   - 输出详细的性能指标

#### e) 重要变量/函数说明

| 名称 | 类型 | 说明 |
|------|------|------|
| `n_routed_experts` | int | Qwen路由专家总数（57） |
| `top_k` | int | 激活的专家数量（默认8） |
| `hidden_size` | int | 隐藏层维度（Qwen配置） |
| `calc_n_expert_on_gpu()` | method | 计算GPU上最优专家数量 |

---

### 3.3 moon.py

#### a) 具体功能描述

Moon MoE模型的GPU卸载基准测试模块。Moon是一个假想的MoE模型架构，用于演示基准测试框架的通用性。

#### b) 架构定位

- **层级**：模型基准测试层
- **位置**：与deepseek.py、qwen.py、mixtral.py并列
- **角色**：Moon模型的性能评估器

#### c) 依赖关系

- **依赖于**：`latency.py`（延迟计算框架）
- **被依赖**：由`run_benchmark.sh`调用执行

#### d) 核心实现逻辑

1. **模型参数**:
   - `n_routed_experts=32`：Moon路由专家总数
   - `top_k=4`：Moon默认激活4个专家

2. **基准测试流程**:
   - 专家数量扫描
   - 内存使用计算
   - 性能指标输出

---

### 3.4 mixtral.py

#### a) 具体功能描述

Mixtral MoE模型的GPU卸载基准测试模块。Mixtral是一个开源的稀疏混合专家模型，本模块用于评估其在GPU卸载场景下的性能表现。

#### b) 架构定位

- **层级**：模型基准测试层
- **位置**：与deepseek.py、qwen.py、moon.py并列
- **角色**：Mixtral模型的性能评估器

#### c) 依赖关系

- **依赖于**：`latency.py`（延迟计算框架）
- **被依赖**：由`run_benchmark.sh`调用执行

#### d) 核心实现逻辑

1. **Mixtral特定参数**:
   - `n_routed_experts=46`：Mixtral路由专家总数
   - `top_k=8`：Mixtral默认激活8个专家

2. **专家内存模型**:
   - 使用与DeepSeek类似的内存计算模型
   - 针对Mixtral架构特点优化

---

### 3.5 latency.py

#### a) 具体功能描述

**核心延迟基准测试框架**，提供MoE模型各层延迟的测量和统计分析功能。

#### b) 架构定位

- **层级**：性能分析层（底层支撑）
- **位置**：被所有模型基准测试模块调用
- **角色**：通用的延迟测量工具

#### c) 依赖关系

- **依赖于**：Python标准库（numpy、typing）
- **被依赖**：deepseek.py、qwen.py、moon.py、mixtral.py

#### d) 核心实现逻辑

1. **延迟计算模型**:
   - 基于token处理的延迟
   - 考虑专家调度的开销

2. **统计分析功能**:
   - 平均延迟计算
   - P50/P90/P99百分位数
   - 异常值检测

3. **延迟来源分解**:
   - `self_latency`：自身处理延迟
   - `cross_latency`：跨专家通信延迟
   - `total_latency`：总延迟

#### e) 重要变量/函数说明

| 名称 | 类型 | 说明 |
|------|------|------|
| `self_latency` | float | 自身处理延迟(ms) |
| `cross_latency` | float | 跨专家通信延迟(ms) |
| `total_latency` | float | 总延迟(ms) |
| `calc_latency()` | method | 计算给定配置的延迟 |

---

### 3.6 microbench.py

#### a) 具体功能描述

**微基准测试框架**，用于对MoE模型中的单个专家模块进行细粒度性能分析。

#### b) 架构定位

- **层级**：性能分析层
- **位置**：提供比latency.py更细粒度的性能分析
- **角色**：专家级别的性能分析工具

#### c) 依赖关系

- **依赖于**：Python标准库
- **被依赖**：可被模型基准测试模块调用

#### d) 核心实现逻辑

1. **微基准测试维度**:
   - 单个专家的计算延迟
   - 专家调度的开销
   - 内存带宽利用率

2. **测试配置**:
   - 支持不同batch size
   - 支持不同序列长度
   - 支持不同专家数量配置

#### e) 重要变量/函数说明

| 名称 | 类型 | 说明 |
|------|------|------|
| `n_expert` | int | 测试的专家数量 |
| `expert_idx` | int | 当前测试的专家索引 |
| `micro_latency` | float | 微基准测试延迟 |

---

### 3.7 run_benchmark.sh

#### a) 具体功能描述

**基准测试执行脚本**，自动化运行所有MoE模型的基准测试。

#### b) 架构定位

- **层级**：基准测试执行层（顶层）
- **位置**：项目的入口点
- **角色**：测试编排器

#### c) 依赖关系

- **依赖于**：deepseek.py、qwen.py、moon.py、mixtral.py、latency.py
- **被依赖**：由用户直接执行

#### d) 核心实现逻辑

1. **测试流程编排**:
   - 按顺序执行各模型基准测试
   - 收集各模型的测试结果

2. **结果汇总**:
   - 汇总各模型的性能数据
   - 生成综合性能报告

---

### 3.8 数据文件

#### 3.8.1 deep_latency.txt

- **文件路径**：`/home/lzx/program/moe_code/opensource/deep_latency.txt`
- **内容格式**：纯文本，每行一个浮点数
- **数据量**：1499行
- **用途**：DeepSeek模型的延迟基准测试数据

#### 3.8.2 test.txt

- **文件路径**：`/home/lzx/program/moe_code/opensource/test.txt`
- **内容格式**：每行格式为"行号,值"
- **用途**：通用测试数据集

#### 3.8.3 microdeepseek.txt

- **文件路径**：`/home/lzx/program/moe_code/opensource/microdeepseek.txt`
- **内容格式**：纯文本延迟数据
- **用途**：DeepSeek微基准测试数据

#### 3.8.4 microqwen.txt

- **文件路径**：`/home/lzx/program/moe_code/opensource/microqwen.txt`
- **用途**：Qwen微基准测试数据

#### 3.8.5 micromoon.txt

- **文件路径**：`/home/lzx/program/moe_code/opensource/micromoon.txt`
- **用途**：Moon微基准测试数据

#### 3.8.6 micromixtral.txt

- **文件路径**：`/home/lzx/program/moe_code/opensource/micromixtral.txt`
- **用途**：Mixtral微基准测试数据

#### 3.8.7 hot/deep.txt

- **文件路径**：`/home/lzx/program/moe_code/opensource/hot/deep.txt`
- **内容格式**：每行格式为"专家ID,分数"
- **用途**：Hot场景下的专家深度/分数数据
- **数据规模**：1474行

---

## 4. 项目架构设计

### 4.1 架构分层

```
┌──────────────────────────────────────────────────────┐
│                    用户交互层                          │
│              (run_benchmark.sh 执行入口)                  │
├──────────────────────────────────────────────────────┤
│                   模型测试层                            │
│  ┌─────────────┬─────────────┬─────────────┬─────────┐│
│  │  deepseek   │   qwen      │   moon      │ mixtral ││
│  │   .py       │   .py       │   .py       │  .py    ││
│  └─────────────┴─────────────┴─────────────┴─────────┘│
├──────────────────────────────────────────────────────┤
│                   性能分析层                           │
│         ┌──────────────┬───────────────┐            │
│         │  latency.py  │ microbench.py │            │
│         └──────────────┴───────────────┘            │
├──────────────────────────────────────────────────────┤
│                   数据存储层                           │
│   ┌──────────────────────────────────────────────┐   │
│   │ .txt 数据文件 (deep_latency, micro*, hot/*)   │   │
│   └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

### 4.2 核心设计模式

1. **策略模式**：每个模型有独立的基准测试策略
2. **模板方法模式**：共享的基准测试流程框架
3. **数据驱动**：通过配置文件和数据文件驱动测试

---

## 5. 依赖关系图

### 5.1 文件依赖

```
run_benchmark.sh
    │
    ├──► deepseek.py ──► latency.py
    ├──► qwen.py ──────► latency.py
    ├──► moon.py ──────► latency.py
    └──► mixtral.py ───► latency.py

microbench.py (可被模型模块调用)
```

### 5.2 模块间调用关系

```
latency.py (核心框架)
    ▲
    │
deepseek.py ─┬
qwen.py ─────┼──► 共享延迟计算逻辑
moon.py ─────│
mixtral.py ──┘
```

### 5.3 数据流

```
输入配置 → 模型基准测试 → 性能数据 → 输出文件
    │            │              │
    ▼            ▼              ▼
参数配置    latency.py     .txt数据文件
```

---

## 附录：模型参数对比表

| 模型 | 路由专家数 | Top-K | 隐藏层维度 | 备注 |
|------|-----------|-------|-----------|------|
| DeepSeek | 256 | 6 | 7168 | 最多专家数 |
| Qwen | 57 | 8 | - | - |
| Moon | 32 | 4 | - | 最少专家数 |
| Mixtral | 46 | 8 | - | 开源模型 |

---

*文档生成日期：2026-03-18*
*项目路径：/home/lzx/program/moe_code/opensource*