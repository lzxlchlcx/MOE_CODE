import argparse
import json
import os
import random

from model.deepseek import FiddlerDeepSeek


def load_prompts_from_dataset(dataset_path, batch_size):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_prompts = []
    for conv in data:
        for turn in conv.get("conversations", []):
            if turn.get("from") == "human":
                all_prompts.append(turn["value"])
                break
    return random.sample(all_prompts, min(batch_size, len(all_prompts)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to DeepSeek model.",
    )
    parser.add_argument(
        "--cpu-offload",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: execute at GPU (baseline), 1: offload to CPU.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Please tell me a joke.",
        help="Input text to generate (ignored if --dataset is set).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to ShareGPT JSON dataset. Overrides --input with random samples.",
    )
    parser.add_argument(
        "--n-token",
        type=int,
        default=20,
        help="Number of tokens to generate.",
    )
    parser.add_argument("--beam-width", type=int, default=1, help="Beam search width.")

    args = parser.parse_args()
    model = FiddlerDeepSeek(args)

    if args.dataset:
        input_text = load_prompts_from_dataset(args.dataset, args.batch_size)
    else:
        input_text = args.input

    prefill_time, decode_time, hit_rate = model.generate(
        input_text, output_token=args.n_token
    )
    print(
        f"prefill_time: {prefill_time:.4f}, decode_time: {decode_time:.4f}, hit_rate: {hit_rate:.4f}"
    )
