"""MoE 基准测试主入口 - Config 驱动 + 闭环自动检测"""

import argparse
import json
import os
import random
import sys
import subprocess
import warnings
from datetime import datetime

sys.path.append("./")
from mixtral import FiddlerMixtral
from deepseek import FiddlerDeepSeekV2
from moon import FiddlerMoon
from qwen import FiddlerQwen

from config import (
    ModelConfig,
    SystemConfig,
    get_default_config,
    merge_system_to_model,
    auto_detect_gpu_info,
)

DEFAULT_LOG_DIR = "./log"


def load_dataset(path, dataset_type="sharegpt"):
    """加载数据集，支持多种格式"""
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
    """从模型路径检测模型类型"""
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
    """根据模型名获取模型类"""
    classes = {
        "deepseek": FiddlerDeepSeekV2,
        "qwen": FiddlerQwen,
        "moon": FiddlerMoon,
        "mixtral": FiddlerMixtral,
    }
    return classes.get(model_name, FiddlerDeepSeekV2)


def check_model_exists(model_path):
    """检查模型路径是否存在，不存在则 warning 并返回 False"""
    if not os.path.exists(model_path):
        warnings.warn(f"模型路径不存在: {model_path}，跳过该模型", UserWarning)
        return False
    # 检查是否包含模型文件
    model_files = ["config.json", "model.safetensors", "pytorch_model.bin"]
    has_model_file = any(
        os.path.exists(os.path.join(model_path, f)) for f in model_files
    )
    if not has_model_file:
        # 可能是分片模型，检查是否有 shard 文件
        shard_pattern = "model-00001-of-"
        has_shard = any(shard_pattern in f for f in os.listdir(model_path))
        if not has_shard:
            warnings.warn(
                f"模型目录 {model_path} 中未找到模型文件，可能路径有误",
                UserWarning,
            )
            return False
    return True


def find_system_config(model_name, config_dir="./configs"):
    """查找系统配置文件"""
    config_path = os.path.join(config_dir, f"system_config_{model_name}.json")
    if os.path.exists(config_path):
        return config_path
    return None


def run_microbench_auto(model_path, model_name):
    """自动运行 microbench 生成配置"""
    print(f"\n未找到 {model_name} 的配置文件，正在自动运行 microbench...")
    print("这可能需要几分钟时间...\n")

    try:
        subprocess.run(
            [
                sys.executable,
                "microbench.py",
                "--model",
                model_path,
            ],
            check=True,
            cwd=os.path.dirname(__file__) or ".",
        )
        print("\nmicrobench 运行完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nmicrobench 运行失败: {e}")
        print("将使用默认配置继续...\n")
        return False


def load_or_generate_config(args) -> ModelConfig:
    """加载配置或自动生成"""
    model_name = detect_model_type(args.model)

    # 1. 如果指定了 config 文件，直接加载
    if hasattr(args, "config") and args.config:
        if os.path.exists(args.config):
            print(f"从 {args.config} 加载配置")
            config = ModelConfig.load(args.config)
            config.model_path = args.model
            return config

    # 2. 查找已有的 system_config
    system_config_path = find_system_config(model_name)

    # 3. 如果没有且允许自动运行，运行 microbench
    if not system_config_path and getattr(args, "auto_microbench", True):
        success = run_microbench_auto(args.model, model_name)
        if success:
            system_config_path = find_system_config(model_name)

    # 4. 构建 ModelConfig
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

    # 5. 合并 system_config 到 model_config
    if system_config_path and os.path.exists(system_config_path):
        print(f"从 {system_config_path} 加载系统配置")
        system_config = SystemConfig.load(system_config_path)
        config = merge_system_to_model(system_config, config)

    return config


def main():
    parser = argparse.ArgumentParser()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model",
        type=str,
        default="/mnt/g/Models/DeepSeek-v2-lite-chat",
        help="Model path.",
    )
    parser.add_argument(
        "--cpu-offload",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: execute at GPU (baseline), 1: offload to CPU.",
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
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Config file path (JSON).",
    )
    parser.add_argument(
        "--prefetch",
        type=str,
        default="disabled",
        choices=["disabled", "enabled"],
        help="Prefetch strategy: disabled or enabled.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=DEFAULT_LOG_DIR,
        help=f"Log directory (default: {DEFAULT_LOG_DIR}).",
    )
    parser.add_argument(
        "--auto-microbench",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: auto run microbench if config not found, 0: use default.",
    )

    args = parser.parse_args()

    # 检查模型路径
    if not check_model_exists(args.model):
        print(f"\n[WARNING] 模型路径不存在: {args.model}")
        print("请检查模型路径是否正确，或使用 --model 指定有效路径。")
        print("退出。")
        sys.exit(1)

    # 加载或生成配置
    config = load_or_generate_config(args)

    dataset_path = "/home/lzx/program/moe_code/opensource/sharegpt_v3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json"
    dataset_type = "share"
    texts = load_dataset(dataset_path, dataset_type)

    random.seed(0)
    random.shuffle(texts)

    # 获取模型类并初始化
    model_cls = get_model_class(config.model_name)
    model = model_cls(config)
    prefix = config.model_name[:4] if len(config.model_name) > 4 else config.model_name

    n_sample = 5
    input_token = 128
    output_token = 32

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    detailed_result_file = (
        f"./result/result-{prefix}-bs{config.batch_size}-{timestamp}.txt"
    )

    print(
        f"\n测试配置: batch_size={config.batch_size}, input_token={input_token}, "
        f"output_token={output_token}, n_sample={n_sample}"
    )
    print(f"预取策略: {'启用' if config.prefetch_enabled else '禁用'}")
    print(f"详细结果将保存到: {detailed_result_file}")

    prefill_time_sum, decode_time_sum, hit_rate_sum = 0, 0, 0

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs("./result", exist_ok=True)

    with open(detailed_result_file, "w", encoding="utf-8") as f_detailed:
        f_detailed.write("=" * 80 + "\n")
        f_detailed.write(f"{config.model_name} 模型吞吐测试报告\n")
        f_detailed.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_detailed.write(
            f"配置: batch_size={config.batch_size}, input_token={input_token}, "
            f"output_token={output_token}\n"
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
            print(f"样本 {sample_idx + 1}/{n_sample}")
            print(f"{'=' * 80}")

            print("\n输入样本:")
            for i, text in enumerate(batch_texts):
                print(f"样本 {i + 1}: {text[:100]}...")

            print("\n生成结果:")
            for i, output in enumerate(stats["outputs"]):
                print(f"结果 {i + 1}: {output[:100]}...")

            prefill_time_sum += prefill_time
            decode_time_sum += decode_time
            hit_rate_sum += hit_rate

            print("\n性能开销统计:")
            for stat_name, times in stats["perf_stats"].items():
                if times:
                    avg_time = sum(times) / len(times) * 1000
                    print(f"{stat_name}: 平均 {avg_time:.2f}ms (共 {len(times)} 次)")
                else:
                    print(f"{stat_name}: 无记录")

            f_detailed.write(f"\n{'=' * 80}\n")
            f_detailed.write(f"样本 {sample_idx + 1}/{n_sample}\n")
            f_detailed.write(f"{'=' * 80}\n")
            f_detailed.write(
                f"Prefill时间: {prefill_time:.2f}s, Decode时间: {decode_time:.2f}s, "
                f"命中率: {hit_rate:.2f}\n\n"
            )

            f_detailed.write("输入样本:\n")
            for i, text in enumerate(batch_texts):
                f_detailed.write(f"样本 {i + 1}: {text}\n\n")

            f_detailed.write("生成结果:\n")
            for i, output in enumerate(stats["outputs"]):
                f_detailed.write(f"结果 {i + 1}: {output}\n\n")

            f_detailed.write("性能开销统计:\n")
            for stat_name, times in stats["perf_stats"].items():
                if times:
                    avg_time = sum(times) / len(times) * 1000
                    f_detailed.write(
                        f"{stat_name}: 平均 {avg_time:.2f}ms (共 {len(times)} 次)\n"
                    )
                else:
                    f_detailed.write(f"{stat_name}: 无记录\n")
            f_detailed.write("\n")

    throughput = (
        output_token
        * config.batch_size
        * n_sample
        / (prefill_time_sum + decode_time_sum)
    )
    summary = f"""
{"=" * 80}
测试完成汇总 (batch_size={config.batch_size})
{"=" * 80}
平均Prefill时间: {prefill_time_sum / n_sample:.2f}s
平均Decode时间: {decode_time_sum / n_sample:.2f}s
平均命中率: {hit_rate_sum / n_sample:.2f}
总吞吐: {throughput:.2f} token/s
{"=" * 80}
"""
    print(summary)

    with open(detailed_result_file, "a", encoding="utf-8") as f_detailed:
        f_detailed.write("\n" + summary)

    print(f"详细结果已保存到: {detailed_result_file}")

    # 如果有logger，保存最终报告
    if hasattr(model, "logger"):
        model.logger.set_prefill_decode_times(
            prefill_time_sum / n_sample, decode_time_sum / n_sample
        )
        model.logger.set_output_config(output_token, n_sample)
        report = model.logger.finalize()
        report.print_summary()
        json_path, txt_path = report.save()
        print(f"分析报告已保存到: {json_path}")
        print(f"           {txt_path}")


if __name__ == "__main__":
    main()
