# CLAUDE.md

## 项目概述
MoE 混合专家模型 GPU 卸载推理框架，
`10-fiddler-main` 基于fiddler实现CPU-GPU调度。
`20-PDScope` 基于 PDScope 论文实现 CPU-GPU 混合调度。

## 约定
1. 实验记录写入 `实验日志.md` ， 包括：整个完成的工作
2. 及时记录BUG或有价值的问题（偏知识点），保存到 `docs/` 目录
3. 实验数据写入 `实验数据.md`， 主要包含一些有价值的实验数据，做好版本记录


## 模型路径
1. DeepSeek-V2-lite: `/mnt/g/Models/DeepSeek-v2-lite-chat`
2. Qwen3-30B-A3B: `/mnt/g/Models/Qwen3-30B-A3B/`

## 数据集路径
1. ShareGPT: `./ShareGPT_V3_unfiltered_cleaned_split.json` (位于项目根目录)