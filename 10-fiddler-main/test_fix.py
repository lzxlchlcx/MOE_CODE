"""
测试修复后的 deepseek.py 是否正确处理 attention_mask
"""
import argparse
import os
import sys

# 添加 src/fiddler 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/fiddler'))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from deepseek import FiddlerDeepSeek


class Args:
    def __init__(self):
        self.model = "/mnt/g/Models/DeepSeek-v2-lite-chat"
        self.cpu_offload = 0  # 使用 GPU 模式测试
        self.batch_size = 1
        self.beam_width = 1


def debug_mask_values(model, input_ids, position_ids, attention_mask, cache_position):
    """调试: 检查 create_causal_mask 创建的 mask 值"""
    import torch
    from transformers.masking_utils import create_causal_mask

    inps = model.model.embed_tokens(input_ids)

    causal_mask = create_causal_mask(
        config=model.config,
        input_embeds=inps,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=model.past_key_value,
        position_ids=position_ids,
    )

    # create_causal_mask 可能返回 None（例如 SDPA 注意力）
    if causal_mask is None:
        print(f"  [DEBUG] causal_mask is None (SDPA mode, using is_causal=True)")
    else:
        print(f"  [DEBUG] causal_mask shape: {causal_mask.shape}")
        print(f"  [DEBUG] causal_mask dtype: {causal_mask.dtype}")
        print(f"  [DEBUG] causal_mask min: {causal_mask.min().item():.2f}, max: {causal_mask.max().item():.2f}")

        # 打印 mask 内容（如果是 small seq）
        if causal_mask.shape[-1] <= 10:
            print(f"  [DEBUG] causal_mask:\n{causal_mask[0, 0] if causal_mask.dim() == 4 else causal_mask}")

    return causal_mask


def test_logits(model, input_text):
    """测试: 检查 logits 的质量"""
    import torch
    import torch.nn.functional as F
    import transformers

    # Tokenize
    encodings = model.tokenizer(
        [input_text],
        padding=True,
        truncation=True,
        max_length=10,
        return_tensors="pt"
    )
    input_ids = encodings.input_ids.to(model.dev)
    attention_mask = encodings.attention_mask.to(model.dev)
    seq_length = input_ids.shape[1]
    position_ids = torch.arange(seq_length, dtype=torch.long, device=model.dev).unsqueeze(0)
    cache_position = torch.arange(seq_length, dtype=torch.long, device=model.dev)

    print(f"  [DEBUG] input_ids: {input_ids}")
    print(f"  [DEBUG] attention_mask: {attention_mask}")

    # 调试 mask
    causal_mask = debug_mask_values(model, input_ids, position_ids, attention_mask, cache_position)

    # 使用 transformers 官方方式前向传播
    model.model.eval()
    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=transformers.cache_utils.DynamicCache(),
            use_cache=False,
        )
        official_logits = model.lm_head(outputs.last_hidden_state)

        # 使用我们的方式
        official_logits2 = model.mixtral_forward(
            input_ids, position_ids, attention_mask, cache_position, is_prefill=True
        )

    print(f"  [DEBUG] Official logits shape: {official_logits.shape}")
    print(f"  [DEBUG] Our logits shape: {official_logits2.shape}")

    # Top-k tokens
    official_topk = torch.topk(official_logits[0, -1], 5)
    our_topk = torch.topk(official_logits2[0, -1], 5)

    print(f"  [DEBUG] Official top-5 tokens: {official_topk.indices.tolist()}")
    print(f"  [DEBUG] Official top-5 probs: {official_topk.values.tolist()}")
    print(f"  [DEBUG] Our top-5 tokens: {our_topk.indices.tolist()}")
    print(f"  [DEBUG] Our top-5 probs: {our_topk.values.tolist()}")

    # 解码 top-1 token
    official_token = official_logits[0, -1].argmax().item()
    our_token = official_logits2[0, -1].argmax().item()
    print(f"  [DEBUG] Official top-1 token: {official_token} -> '{model.tokenizer.decode([official_token])}'")
    print(f"  [DEBUG] Our top-1 token: {our_token} -> '{model.tokenizer.decode([our_token])}'")

    return official_logits, official_logits2


def test_fix():
    """测试修复后的模型输出"""
    args = Args()

    print("=" * 60)
    print("测试修复后的 deepseek.py")
    print("=" * 60)

    print("\n正在加载模型...")
    model = FiddlerDeepSeek(args)

    # 测试输入
    test_inputs = [
        "Please tell me a joke.",
        "What is the capital of France?",
        "Hello, how are you today?",
    ]

    print("\n" + "=" * 60)
    print("开始测试 logits 对比")
    print("=" * 60)

    for i, input_text in enumerate(test_inputs):
        print(f"\n--- 测试 {i+1}/{len(test_inputs)} ---")
        print(f"输入: {input_text}")
        official_logits, our_logits = test_logits(model, input_text)

    print("\n" + "=" * 60)
    print("开始测试生成")
    print("=" * 60)

    for i, input_text in enumerate(test_inputs):
        print(f"\n--- 测试 {i+1}/{len(test_inputs)} ---")
        print(f"输入: {input_text}")

        try:
            prefill_time, decode_time, hit_rate = model.generate(
                input_text,
                output_token=20,
            )
            print(f"统计: prefill={prefill_time:.3f}s, decode={decode_time:.3f}s, hit_rate={hit_rate:.2%}")
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_fix()
