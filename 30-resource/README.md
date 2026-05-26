# MoE Model GPU Offload Benchmark System

A benchmark system for testing inference performance of various Mixture of Experts (MoE) large language models with GPU offload strategies.

## Table of Contents

- [Overview](#overview)
- [Supported Models](#supported-models)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Key Commands](#key-commands)
- [File Descriptions](#file-descriptions)
- [Architecture](#architecture)
- [Model Parameters](#model-parameters)

---

## Overview

This project is a **MoE (Mixture of Experts) GPU Offload Benchmark System** designed to evaluate inference performance of various MoE architecture large language models under different GPU memory configurations.

### Key Features

- **Model Offload Strategy Testing**: Evaluate offload strategy effectiveness for different MoE models
- **Latency Benchmarking**: Measure actual inference latency for each layer
- **Micro-benchmarking**: Fine-grained performance analysis of expert modules
- **GPU Memory Estimation**: Calculate GPU memory requirements for different model configurations

---

## Supported Models

| Model | Path |
|-------|------|
| DeepSeek-V2-lite | `/mnt/g/Models/DeepSeek-v2-lite-chat` |
| Qwen3-30B-A3B | `/mnt/g/Models/Qwen3-30B-A3B/` |
| Mixtral-8x7B | `/home/share/bz/model/Mixtral-8x7B-v0.1` |
| Moonlight-16B-A3B-Instruct | `/home/share/bz/model/Moonlight-16B-A3B-Instruct` |

---

## Project Structure

```
opensource/
├── deepseek.py           # DeepSeek MoE benchmark module
├── qwen.py               # Qwen MoE benchmark module
├── moon.py               # Moon MoE benchmark module
├── mixtral.py            # Mixtral MoE benchmark module
├── latency.py            # Core latency benchmark framework
├── microbench.py         # Micro-benchmark framework
├── run_benchmark.sh      # Batch benchmark execution script
├── result/               # Detailed benchmark results
├── hot/                  # Hot expert data files
│   ├── deep.txt
│   ├── qwen.txt
│   ├── moon.txt
│   └── mix.txt
├── sharegpt_v3_unfiltered_cleaned_split/  # Test dataset
└── README.md             # This file
```

**Note**: All operations must be executed within the `opensource/` directory.

---

## Quick Start

### 1. Install Dependencies

Required packages (must be installed manually):
- PyTorch
- Transformers
- NumPy

### 2. Run Basic Latency Test

```bash
cd opensource
python latency.py --model "/mnt/g/Models/DeepSeek-v2-lite-chat" --batch_size 1
```

### 3. Run Micro-benchmark

```bash
cd opensource
python microbench.py --model <model_path>
```

### 4. Run Batch Benchmark

```bash
cd opensource
./run_benchmark.sh
```

---

## Key Commands

| Command | Description |
|---------|-------------|
| `python latency.py --model <path> --batch_size 1` | Main latency test |
| `python microbench.py --model <model_path>` | Micro-benchmark test |
| `./run_benchmark.sh` | Run all benchmarks in batch |

---

## File Descriptions

### Core Modules

| File | Description |
|------|-------------|
| [`latency.py`](latency.py) | Core latency benchmark framework providing MoE model layer latency measurement and statistical analysis |
| [`microbench.py`](microbench.py) | Micro-benchmark framework for fine-grained performance analysis of individual expert modules |

### Model Benchmark Modules

| File | Description |
|------|-------------|
| [`deepseek.py`](deepseek.py) | DeepSeek MoE model benchmark - 256 routed experts, top-k=6 |
| [`qwen.py`](qwen.py) | Qwen MoE model benchmark - 57 routed experts, top-k=8 |
| [`moon.py`](moon.py) | Moon MoE model benchmark - 32 routed experts, top-k=4 |
| [`mixtral.py`](mixtral.py) | Mixtral MoE model benchmark - 46 routed experts, top-k=8 |

### Scripts

| File | Description |
|------|-------------|
| [`run_benchmark.sh`](run_benchmark.sh) | Automated batch execution script for all MoE model benchmarks |

### Data Files

| Path | Description |
|------|-------------|
| `result/` | Detailed benchmark result files |
| `hot/*.txt` | Hot expert data for each model |
| `sharegpt_v3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json` | Test dataset |

---

## Architecture

### Layered Architecture

```
┌──────────────────────────────────────────────────────┐
│              Benchmark Execution Layer              │
│                  (run_benchmark.sh)                  │
├──────────────────────────────────────────────────────┤
│                  Model Benchmark Layer               │
│  ┌───────────┬───────────┬───────────┬─────────────┐ │
│  │  deepseek │   qwen    │   moon    │   mixtral   │ │
│  │   .py     │   .py     │   .py     │    .py      │ │
│  └───────────┴───────────┴───────────┴─────────────┘ │
├──────────────────────────────────────────────────────┤
│                 Performance Analysis Layer           │
│        ┌──────────────┬───────────────┐              │
│        │  latency.py  │ microbench.py │              │
│        └──────────────┴───────────────┘              │
├──────────────────────────────────────────────────────┤
│                    Data Storage Layer               │
│   ┌──────────────────────────────────────────────┐   │
│   │  .txt data files (results, hot/*, micro*)   │   │
│   └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

### Module Dependencies

```
run_benchmark.sh
    │
    ├──► deepseek.py ──► latency.py
    ├──► qwen.py ──────► latency.py
    ├──► moon.py ──────► latency.py
    └──► mixtral.py ───► latency.py

microbench.py (can be called by model modules)
```

---

## Model Parameters

| Model | Routed Experts | Top-K | Hidden Size | Notes |
|-------|---------------|-------|-------------|-------|
| DeepSeek | 256 | 6 | 7168 | Maximum experts |
| Qwen | 57 | 8 | - | - |
| Moon | 32 | 4 | - | Minimum experts |
| Mixtral | 46 | 8 | - | Open source model |

---

## Performance Optimization Tips

1. **Remove print statements** in model files (e.g., `deepseek.py`) can significantly improve performance
2. **Batch size impact**: batch_size=4 yields approximately 2x throughput compared to batch_size=1
3. **Hot expert prefetching**: The hot expert prefetching mechanism has significant impact on performance
4. **Throughput calculation**: Remember to multiply by `batch_size` and number of samples
5. **Model loading**: May take considerable time (tens of seconds)
6. **Garbled text**: May be related to tokenizer decoding; `.tolist()` sometimes solves the issue

---

## Notes

- No `requirements.txt` or `setup.py` exists; dependencies must be installed manually
- Default model path is hardcoded to `/mnt/g/Models/DeepSeek-v2-lite-chat`
- Results are saved to the `result/` directory with detailed performance metrics

---

*Document generated: 2026-03-28*
*Project path: /home/lzx/program/moe_code/opensource*
