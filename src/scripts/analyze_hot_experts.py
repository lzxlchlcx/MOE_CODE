import argparse
import json
import os
import random
from collections import Counter

from model.deepseek import mDeepSeek


def load_sharegpt_prompts(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = []
    for conv in data:
        for turn in conv.get("conversations", []):
            if turn.get("from") == "human":
                prompts.append(turn.get("value", ""))
                break
    return prompts


def sample_prompts(prompts, num_samples, seed):
    sample_size = min(num_samples, len(prompts))
    rng = random.Random(seed)
    return rng.sample(prompts, sample_size)


def sorted_hot_counts(counts):
    return sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))


def write_hot_outputs(
    counts,
    output_dir,
    output_name,
    dataset_path,
    seed,
    requested_samples,
    actual_samples,
    n_token,
):
    os.makedirs(output_dir, exist_ok=True)
    sorted_counts = sorted_hot_counts(counts)
    txt_path = os.path.join(output_dir, output_name)
    stats_name = f"{os.path.splitext(output_name)[0]}_stats.json"
    stats_path = os.path.join(output_dir, stats_name)

    with open(txt_path, "w", encoding="utf-8") as f:
        for (layer, expert), _ in sorted_counts:
            f.write(f"{layer},{expert}\n")

    stats = {
        "dataset_path": dataset_path,
        "seed": seed,
        "requested_samples": requested_samples,
        "actual_samples": actual_samples,
        "n_token": n_token,
        "total_routes": sum(counts.values()),
        "experts": [
            {"layer": layer, "expert": expert, "count": count}
            for (layer, expert), count in sorted_counts
        ],
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return txt_path, stats_path


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze hot MoE experts from ShareGPT prompts.")
    parser.add_argument("--model", type=str, required=True, help="Path to DeepSeek model.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to ShareGPT JSON dataset.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of ShareGPT prompts to analyze.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for prompt sampling.")
    parser.add_argument("--n-token", type=int, default=20, help="Number of tokens to generate per prompt batch.")
    parser.add_argument("--output-dir", type=str, default="hot", help="Directory for hot expert outputs.")
    parser.add_argument("--output-name", type=str, default="deep.txt", help="Hot expert text filename.")
    parser.add_argument("--cpu-offload", type=int, default=1, choices=[0, 1], help="0: GPU baseline, 1: CPU offload.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--beam-width", type=int, default=1, help="Beam search width.")
    return parser.parse_args()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()

    prompts = sample_prompts(load_sharegpt_prompts(args.dataset), args.num_samples, args.seed)
    model = mDeepSeek(args)
    model.reset_hot_expert_stats()

    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start:start + args.batch_size]
        model.generate(batch, output_token=args.n_token)

    counts = Counter(model.hot_expert_counts)
    txt_path, stats_path = write_hot_outputs(
        counts=counts,
        output_dir=args.output_dir,
        output_name=args.output_name,
        dataset_path=args.dataset,
        seed=args.seed,
        requested_samples=args.num_samples,
        actual_samples=len(prompts),
        n_token=args.n_token,
    )
    print(f"hot experts written to: {txt_path}")
    print(f"hot expert stats written to: {stats_path}")


if __name__ == "__main__":
    main()
