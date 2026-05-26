import argparse
import os

from deepseek import FiddlerDeepSeek


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
        help="Input text to generate.",
    )
    parser.add_argument(
        "--n-token",
        type=int,
        default=20,
        help="Number of tokens to generate.",
    )
    parser.add_argument(
        "--input-token",
        type=int,
        default=128,
        help="Max input sequence length (for GPU memory estimation).",
    )
    parser.add_argument("--beam-width", type=int, default=1, help="Beam search width.")

    args = parser.parse_args()
    model = FiddlerDeepSeek(args)
    prefill_time, decode_time, hit_rate = model.generate(
        args.input, output_token=args.n_token
    )
    print(
        f"prefill_time: {prefill_time}, decode_time: {decode_time}, hit_rate: {hit_rate}"
    )
