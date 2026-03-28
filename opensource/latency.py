"""Microbenchmarking for CPU offloading"""

import argparse
import json
import os
import random
import sys

sys.path.append("./")
from mixtral import FiddlerMixtral
from deepseek import FiddlerDeepSeekV2
from moon import FiddlerMoon
from qwen import FiddlerQwen
def load_dataset(path, dataset_type='sharegpt'):
    """加载数据集，支持多种格式"""
    if dataset_type == 'dpo':
        texts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 使用chosen中的用户提问作为输入
                texts.append(data['chosen'][0]['content'])
        return texts
    elif dataset_type == 'llava':
        texts = []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                # LLaVA格式通常包含conversations字段
                for conv in item.get('conversations', []):
                    if conv['from'] == 'human':  # 只取用户输入
                        texts.append(conv['value'])
        return texts        
    else:  # 默认sharegpt格式
        with open(path, "r") as f:
            data = json.load(f)
        texts = []
        for d in data:
            if len(d.get("conversations", [])) > 0:
                texts.append(" ".join(d["conversations"][0]["value"].split()))
        return texts

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model",
        type=str,
        default="/mnt/g/Models/DeepSeek-v2-lite-chat",
        help="Model path. default `/mnt/g/Models/Qwen3-30B-A3B/`.",
    )
    parser.add_argument(
        "--cpu-offload",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: exeute at GPU (baseline), 1: offload to CPU.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size for inference.",
    )
    parser.add_argument(
        "--cache",
        type=int,
        default=2,
        help="cache size for inference.",
    )
    parser.add_argument("--beam_num", type=int, default=1, help="Beam search number.")
    parser.add_argument("--beam_width", type=int, default=1, help="Beam search number.")
    args = parser.parse_args()
    # dataset_path = "/home/share/bz/dataset/merged_dpo_zh_emoji_for_firefly.jsonl"  # 修改为JSON文件路径
    # dataset_type = 'dpo'  # 新增：数据集类型    
    # output_dir = "emo_predictors"  # 输出目录
    # dataset_path = "/home/share/bz/dataset/llava_instruct_150k.json"  # 修改为JSON文件路径
    # dataset_type = 'llava'  # 新增：数据集类型    
    # output_dir = "lla_predictors"  # 输出目录    
    dataset_path = "/home/lzx/program/moe_code/opensource/sharegpt_v3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json"  # 修改为JSON文件路径
    dataset_type = 'share'  # 新增：数据集类型    
    output_dir = "expert_predictors"  # 输出目录   
    texts = load_dataset(dataset_path, dataset_type)    

    random.seed(0)
    random.shuffle(texts)
    # if args.model=="/home/share/bz/model/Mixtral-8x7B-v0.1":
    #     model = FiddlerMixtral(args)
    #     prefix="mix"
    # elif args.model=="/home/share/bz/model/Qwen3-30B-A3B":
    #     model = FiddlerQwen(args)
    #     prefix="qwen"
    # elif args.model=="/home/share/bz/model/DeepSeek-V2-Lite":
    #     model = FiddlerDeepSeekV2(args)
    #     prefix="deep"
    # else:
    #     model = FiddlerMoon(args)   
    #     prefix="moon" 
    model = FiddlerDeepSeekV2(args)
    prefix="deep"

    n_sample = 3

    for input_token in [128]:
        for output_token in [32]:
            idx_text = 0
            batchid = 0
            prefill_time_sum, decode_time_sum, hit_rate_sum = 0, 0, 0
            for _ in range(n_sample):
                batch_texts = []
                # 允许重复使用文本直到填满batch_size
                while len(batch_texts) < args.batch_size:
                    if idx_text >= len(texts):
                        idx_text = 0  # 循环使用文本
                    text = texts[idx_text]
                    idx_text += 1
                    if len(text.split()) >= input_token:
                        batch_texts.append(text)
                        print("batchid", len(batch_texts))
                
                # 确保batch_texts长度正确
                batch_texts = batch_texts[:args.batch_size]
                # print(f"生成的batch_texts内容: {batch_texts}")    
                # print(f"生成的batch_texts内容: {batch_texts}")                        
                prefill_time, decode_time, hit_rate, stats = model.generate(
                    batch_texts, output_token=output_token, input_token=input_token
                )
                print("\n输入样本:")
                for i, text in enumerate(batch_texts):
                    print(f"样本 {i+1}: {text[:100]}...")  # 截取前100字符避免太长
                
                print("\n生成结果:")
                for i, output in enumerate(stats['outputs']):
                    print(f"结果 {i+1}: {output[:100]}...")  
                if _ >0:
                    prefill_time_sum += prefill_time
                    decode_time_sum += decode_time
                    hit_rate_sum += hit_rate
                print("\n性能开销统计:")
                for stat_name, times in stats['perf_stats'].items():
                    if times:  # 确保有时间记录
                        avg_time = sum(times) / len(times) * 1000  # 转换为毫秒
                        print(f"{stat_name}: 平均 {avg_time:.2f}ms (共 {len(times)} 次)")
                    else:
                        print(f"{stat_name}: 无记录")       
            latency_file = f"{prefix}_latency.txt"
            with open(latency_file, "a") as f:
                f.write(
                    f"batchsize: {args.batch_size}"                    
                    f"input_token: {input_token}, output_token: {output_token}, "
                    f"prefill_time: {prefill_time_sum / n_sample}, "
                    f"decode_time: {decode_time_sum / n_sample}, "
                    f"hit_rate: {hit_rate_sum / n_sample},"
                    f"{output_token *n_sample/ (prefill_time_sum + decode_time_sum):.2f}token/s\n"
                )


                # print("\n专家热度统计:")
                # hot_stats = stats['expert_hot_stats']
                # for layer in range(model.n_layer):
                #     total = hot_stats['hot_experts'][layer]['count']
                #     if total > 0:
                #         hot_pct = hot_stats['hot_experts'][layer]['hot'] / total * 100
                #         veryhot_pct = hot_stats['hot_experts'][layer]['veryhot'] / total * 100
                #         print(f"层 {layer}: hot专家 {hot_pct:.1f}%, veryhot专家 {veryhot_pct:.1f}%")

                # print("\nhot专家数量分布:")
                # for count in [2,3,4,5]:
                #     print(f"{count}个hot专家的概率: {hot_stats['hot_counts'][count] / model.n_layer * 100:.1f}%")                        
"""
                token_dist = {}
                for record in stats['expert_time']:
                    token_count = record['token_count']
                    token_dist[token_count] = token_dist.get(token_count, 0) + 1
                
                print("\n专家处理token数量分布:")
                print(f"最大token数量: {max(token_dist.keys()) if token_dist else 0}")
                print(f"batch_size: {model.batch_size}")
                for token_num, count in sorted(token_dist.items()):
                    print(f"{token_num}个token: {count}个专家")
                
                # 绘制柱状图
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.bar(token_dist.keys(), token_dist.values())
                plt.xlabel('处理的token数量')
                plt.ylabel('专家数量')
                plt.title(f'专家处理token数量分布 (batch_size={model.batch_size})')
                plt.grid(True)
                
                # 保存图表
                os.makedirs('./log', exist_ok=True)
                plt.savefig('./log/expert_token_distribution.png')
                plt.close()

                # 打印专家统计信息
                os.makedirs('./log', exist_ok=True)
               
                # 写入专家统计信息
                with open('./log/expert_stats.txt', 'a') as f:
                    f.write("\n性能开销统计:\n")
                    for stat_name, times in stats['perf_stats'].items():
                        if times:
                            avg_time = sum(times) / len(times) * 1000
                            f.write(f"{stat_name}: 平均 {avg_time:.2f}ms (共 {len(times)} 次)\n")
                        else:
                            f.write(f"{stat_name}: 无记录\n")

                    f.write(f"\n样本 {_+1}/{n_sample} 专家统计:\n")
                    f.write("专家选择统计 (层ID和专家ID):\n")
                    for selection in stats['expert_selection']:
                        f.write(f"层 {selection['layer_id']}: 专家 {selection['expert_ids']}\n")
                    
                    f.write("\n专家处理时间统计:\n")
                    for time_stat in stats['expert_time']:
                        f.write(f"层 {time_stat['layer_id']} 专家 {time_stat['expert_id']}: "
                              f"{time_stat['time']*1000:.2f}ms (设备: {time_stat['device']}, "
                              f"处理token数: {time_stat['token_count']})\n")
""" 
            # 写入延迟统计文件

