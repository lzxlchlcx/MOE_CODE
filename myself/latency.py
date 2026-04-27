"""PDScope MoE Benchmark Entry Point"""

import argparse
import json
import os
import random
import sys
import subprocess
import warnings
from datetime import datetime

sys.path.append("./")
from deepseek import FiddlerDeepSeekV2
from qwen import FiddlerQwen
from moon import FiddlerMoon
from mixtral import FiddlerMixtral

from config import (
    ModelConfig,
    SystemConfig,
    get_default_config,
    merge_system_to_model,
    auto_detect_gpu_info,
)

DEFAULT_LOG_DIR = "./log"


def load_dataset(path, dataset_type="sharegpt"):
    if dataset_type == "dpo":
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                texts.append(data["chosen"][0]["content"])
        return texts
    elif dataset_type == "llava":
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                for conv in item.get("conversations", []):
                    if conv["from"] == "human":
                        texts.append(conv["value"])
        return texts
    else:
        with open(path, "r") as f:
            data = json.load(f)
        texts = []
        for d in data:
            if len(d.get("conversations", [])) > 0:
                texts.append(" ".join(d["conversations"][0]["value"].split()))
        return texts


def detect_model_type(model_path):
    model_path_lower = model_path.lower()
    if "deepseek" in model_path_lower:
        return "deepseek"
    elif "qwen" in model_path_lower:
        return "qwen"
    elif "moonlight" in model_path_lower or "moon" in model_path_lower:
        return "moon"
    elif "mixtral" in model_path_lower:
        return "mixtral"
    return "deepseek"


def get_model_class(model_name):
    classes = {
        "deepseek": FiddlerDeepSeekV2,
        "qwen": FiddlerQwen,
        "moon": FiddlerMoon,
        "mixtral": FiddlerMixtral,
    }
    return classes.get(model_name, FiddlerDeepSeekV2)


def check_model_exists(model_path):
    if not os.path.exists(model_path):
        warnings.warn(f"Model path not found: {model_path}", UserWarning)
        return False
    model_files = ["config.json", "model.safetensors", "pytorch_model.bin"]
    has_model_file = any(
        os.path.exists(os.path.join(model_path, f)) for f in model_files
    )
    if not has_model_file:
        shard_pattern = "model-00001-of-"
        has_shard = any(shard_pattern in f for f in os.listdir(model_path))
        if not has_shard:
            warnings.warn(f"No model files in {model_path}", UserWarning)
            return False
    return True


def find_system_config(model_name, config_dir="./configs"):
    config_path = os.path.join(config_dir, f"system_config_{model_name}.json")
    if os.path.exists(config_path):
        return config_path
    return None


def run_microbench_auto(model_path, model_name):
    print(f"\nNo config for {model_name}, running microbench...")
    try:
        subprocess.run(
            [sys.executable, "microbench.py", "--model", model_path],
            check=True,
            cwd=os.path.dirname(__file__) or ".",
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"microbench failed: {e}")
        return False


def load_or_generate_config(args) -> ModelConfig:
    model_name = detect_model_type(args.model)

    if hasattr(args, "config") and args.config:
        if os.path.exists(args.config):
            print(f"Loading config from {args.config}")
            config = ModelConfig.load(args.config)
            config.model_path = args.model
            return config

    system_config_path = find_system_config(model_name)

    if not system_config_path and getattr(args, "auto_microbench", True):
        success = run_microbench_auto(args.model, model_name)
        if success:
            system_config_path = find_system_config(model_name)

    config = get_default_config(model_name)
    config.model_path = args.model
    config.batch_size = args.batch_size
    config.cache = args.cache
    config.cpu_offload = args.cpu_offload
    config.beam_width = args.beam_width

    if hasattr(args, "prefetch"):
        config.prefetch_enabled = args.prefetch == "enabled"
    if hasattr(args, "log_dir") and args.log_dir:
        config.log_dir = args.log_dir
    else:
        config.log_dir = DEFAULT_LOG_DIR
    if hasattr(args, "schedule"):
        config.schedule_mode = args.schedule

    if system_config_path and os.path.exists(system_config_path):
        print(f"Loading system config from {system_config_path}")
        system_config = SystemConfig.load(system_config_path)
        config = merge_system_to_model(system_config, config)

    return config


def main():
    parser = argparse.ArgumentParser(description="PDScope MoE Benchmark")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model", type=str, default="/mnt/g/Models/DeepSeek-v2-lite-chat"
    )
    parser.add_argument("--cpu-offload", type=int, default=1, choices=[0, 1])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cache", type=int, default=4)
    parser.add_argument("--beam_num", type=int, default=1)
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument(
        "--prefetch", type=str, default="enabled", choices=["disabled", "enabled"]
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="pdscope",
        choices=["legacy", "pdscope"],
        help="Schedule mode: legacy (original) or pdscope (AdaptSched)",
    )
    parser.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--auto-microbench", type=int, default=1, choices=[0, 1])

    args = parser.parse_args()

    if not check_model_exists(args.model):
        print(f"\n[WARNING] Model not found: {args.model}")
        sys.exit(1)

    config = load_or_generate_config(args)

    dataset_path = "/home/lzx/program/moe_code/opensource/sharegpt_v3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json"
    if os.path.exists(dataset_path):
        texts = load_dataset(dataset_path, "share")
        random.seed(0)
        random.shuffle(texts)
    else:
        texts = ["Hello world"] * 10

    model_cls = get_model_class(config.model_name)
    model = model_cls(config)
    prefix = config.model_name[:4]

    n_sample = 5
    input_token = 128
    output_token = 32

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    detailed_result_file = (
        f"./result/result-{prefix}-bs{config.batch_size}-{timestamp}.txt"
    )

    print(
        f"\nTest config: batch_size={config.batch_size}, input_token={input_token}, "
        f"output_token={output_token}, n_sample={n_sample}"
    )
    print(
        f"Schedule: {config.schedule_mode}, Prefetch: {'enabled' if config.prefetch_enabled else 'disabled'}"
    )
    print(f"Results: {detailed_result_file}")

    prefill_time_sum, decode_time_sum, hit_rate_sum = 0, 0, 0
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs("./result", exist_ok=True)

    with open(detailed_result_file, "w", encoding="utf-8") as f_detailed:
        f_detailed.write("=" * 80 + "\n")
        f_detailed.write(f"PDScope {config.model_name} Benchmark Report\n")
        f_detailed.write(f"Schedule: {config.schedule_mode}\n")
        f_detailed.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_detailed.write(
            f"Config: batch_size={config.batch_size}, "
            f"input_token={input_token}, output_token={output_token}\n"
        )
        f_detailed.write(
            f"Params: e={config.e:.2f}ms, tg={config.tg:.2f}ms, tc={config.tc:.2f}ms\n"
        )
        f_detailed.write("=" * 80 + "\n\n")

        idx_text = 0
        for sample_idx in range(n_sample):
            batch_texts = []
            while len(batch_texts) < config.batch_size:
                if idx_text >= len(texts):
                    idx_text = 0
                text = texts[idx_text]
                idx_text += 1
                if len(text.split()) >= input_token:
                    batch_texts.append(text)
            batch_texts = batch_texts[: config.batch_size]

            prefill_time, decode_time, hit_rate, stats = model.generate(
                batch_texts, output_token=output_token, input_token=input_token
            )

            print(f"\n{'=' * 80}")
            print(f"Sample {sample_idx + 1}/{n_sample}")
            print(f"{'=' * 80}")
            print(
                f"Prefill: {prefill_time:.2f}s, Decode: {decode_time:.2f}s, Hit: {hit_rate:.2f}"
            )

            prefill_time_sum += prefill_time
            decode_time_sum += decode_time
            hit_rate_sum += hit_rate

            f_detailed.write(f"\n{'=' * 80}\n")
            f_detailed.write(f"Sample {sample_idx + 1}/{n_sample}\n")
            f_detailed.write(
                f"Prefill: {prefill_time:.2f}s, Decode: {decode_time:.2f}s, Hit: {hit_rate:.2f}\n\n"
            )

    throughput = (
        output_token
        * config.batch_size
        * n_sample
        / (prefill_time_sum + decode_time_sum)
    )
    summary = f"""
{"=" * 80}
PDScope Benchmark Summary (batch_size={config.batch_size}, schedule={config.schedule_mode})
{"=" * 80}
Avg Prefill: {prefill_time_sum / n_sample:.2f}s
Avg Decode:  {decode_time_sum / n_sample:.2f}s
Avg HitRate: {hit_rate_sum / n_sample:.2f}
Throughput:  {throughput:.2f} token/s
{"=" * 80}
"""
    print(summary)

    with open(detailed_result_file, "a", encoding="utf-8") as f_detailed:
        f_detailed.write("\n" + summary)

    print(f"Detailed results: {detailed_result_file}")

    if hasattr(model, "logger"):
        model.logger.set_prefill_decode_times(
            prefill_time_sum / n_sample, decode_time_sum / n_sample
        )
        model.logger.set_output_config(output_token, n_sample)
        report = model.logger.finalize()
        report.print_summary()
        json_path, txt_path = report.save()
        print(f"Report saved: {json_path}")


if __name__ == "__main__":
    main()
