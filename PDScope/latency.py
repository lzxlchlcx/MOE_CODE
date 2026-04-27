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
    
    if dataset_type == 'dpo':
        texts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                texts.append(data['chosen'][0]['content'])
        return texts
    elif dataset_type == 'llava':
        texts = []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                
                for conv in item.get('conversations', []):
                    if conv['from'] == 'human':  
                        texts.append(conv['value'])
        return texts        
    else:  
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
        default="/home/share/bz/model/Qwen1.5-MoE-A2.7B",
        help="Model path. default `/home/share/bz/model/Qwen1.5-MoE-A2.7B`.",
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
    
    
    
    
    
    
    dataset_path = "/home/share/bz/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"  
    dataset_type = 'share'  
    output_dir = "expert_predictors"  
    texts = load_dataset(dataset_path, dataset_type)    

    random.seed(0)
    random.shuffle(texts)
    if args.model=="/home/share/bz/model/Mixtral-8x7B-v0.1":
        model = FiddlerMixtral(args)
        prefix="mix"
    elif args.model=="/home/share/bz/model/Qwen3-30B-A3B":
        model = FiddlerQwen(args)
        prefix="qwen"
    elif args.model=="/home/share/bz/model/DeepSeek-V2-Lite":
        model = FiddlerDeepSeekV2(args)
        prefix="deep"
    else:
        model = FiddlerMoon(args)   
        prefix="moon" 

    n_sample = 3

    for input_token in [128]:
        for output_token in [32]:
            idx_text = 0
            batchid = 0
            prefill_time_sum, decode_time_sum, hit_rate_sum = 0, 0, 0
            for _ in range(n_sample):
                batch_texts = []
                
                while len(batch_texts) < args.batch_size:
                    if idx_text >= len(texts):
                        idx_text = 0  
                    text = texts[idx_text]
                    idx_text += 1
                    if len(text.split()) >= input_token:
                        batch_texts.append(text)
                        print("batchid", len(batch_texts))
                
                
                batch_texts = batch_texts[:args.batch_size]
                
                
                prefill_time, decode_time, hit_rate, stats = model.generate(
                    batch_texts, output_token=output_token, input_token=input_token
                )
                
                
                    
                
                
                
                    
                
                
                
                
                
                for stat_name, times in stats['perf_stats'].items():
                    if times:  
                        avg_time = sum(times) / len(times) * 1000  
                        
                    
                        
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


                
                
                
                
                
                
                
                

                
       