"""
调试脚本：验证 deepseek.py 中的 attention_mask 处理
"""
import argparse
import os
import torch
import torch.nn.functional as F
import transformers

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def debug_attention_mask():
    """验证 attention_mask 的创建和处理逻辑"""

    model_path = "/mnt/g/Models/DeepSeek-v2-lite-chat"

    # 加载 tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 测试输入
    text = ["Please tell me a joke."]
    max_length = 10

    # Tokenize
    encodings = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = encodings.input_ids
    attention_mask = encodings.attention_mask

    print("=" * 60)
    print("1. Tokenizer 原始输出")
    print("=" * 60)
    print(f"input_ids shape: {input_ids.shape}")
    print(f"input_ids: {input_ids}")
    print(f"attention_mask shape: {attention_mask.shape}")
    print(f"attention_mask: {attention_mask}")

    # 转换为 bool
    attention_mask_bool = attention_mask.bool()
    print(f"\nattention_mask (bool): {attention_mask_bool}")

    seq_length = input_ids.shape[1]
    print(f"\nseq_length: {seq_length}")

    # ========== 模拟 deepseek.py 中的处理 ==========
    print("\n" + "=" * 60)
    print("2. 模拟 deepseek.py tokenize() 中的处理")
    print("=" * 60)

    # 原始处理逻辑
    if attention_mask_bool.dim() == 2:
        attention_mask_4d = attention_mask_bool.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, seq_length)

    print(f"After unsqueeze(1) and unsqueeze(-1) + expand:")
    print(f"attention_mask shape: {attention_mask_4d.shape}")
    print(f"Expected: [batch=1, 1, seq_len={seq_length}, seq_len={seq_length}]")

    # 查看 mask 的内容
    print(f"\nattention_mask[0, 0, :, :]:")
    print(attention_mask_4d[0, 0, :, :])

    # ========== 正确的 causal mask 应该是什么 ==========
    print("\n" + "=" * 60)
    print("3. 正确的 Causal Mask 应该是什么")
    print("=" * 60)

    # 方法1: 使用 transformers 的 attention_mask 转换
    try:
        from transformers.modeling_utils import AttentionMaskConverter
        causal_mask = AttentionMaskConverter._create_4d_causal_attention_mask(
            seq_length,
            seq_length,
            torch.float32,
            device=input_ids.device,
            past_key_values_length=0
        )
    except ImportError:
        # 手动创建 causal mask
        causal_mask = torch.triu(torch.ones(1, 1, seq_length, seq_length), diagonal=1).bool()
        causal_mask = ~causal_mask  # 反转，下三角为 True
        causal_mask = causal_mask.to(device=input_ids.device)
        # 转换为 float mask: True -> 0.0, False -> -inf
        causal_mask_float = torch.zeros_like(causal_mask, dtype=torch.float32)
        causal_mask_float = causal_mask_float.masked_fill(~causal_mask, float('-inf'))
        causal_mask = causal_mask_float
    print(f"\nCorrect 4D causal mask (from AttentionMaskConverter):")
    print(f"Shape: {causal_mask.shape}")
    print(causal_mask[0, 0, :, :])

    # 方法2: 手动创建 causal mask
    print("\n\nManual causal mask (upper triangular):")
    manual_causal = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    manual_causal = ~manual_causal  # 反转，下三角为 True
    print(manual_causal)

    # ========== 对比两种 mask ==========
    print("\n" + "=" * 60)
    print("4. 对比分析")
    print("=" * 60)
    print("deepseek.py 创建的 mask 是一个广播后的 padding mask，不是 causal mask!")
    print("这会导致模型可以看到未来的 token，造成输出乱码。")

    # ========== 测试 decode 阶段的处理 ==========
    print("\n" + "=" * 60)
    print("5. Decode 阶段的 attention_mask 处理")
    print("=" * 60)

    # 模拟第一个 token 生成后的状态
    past_key_values_length = seq_length
    new_seq_len = 1  # 新生成一个 token

    # 原始 attention_mask 是 [batch, 1, seq_len, seq_len]
    print(f"Original attention_mask shape: {attention_mask_4d.shape}")

    # deepseek.py 中的处理
    if attention_mask_4d.dim() == 4 and attention_mask_4d.shape[-1] == attention_mask_4d.shape[-2]:
        seq_len = attention_mask_4d.shape[-1]
        attention_mask_truncated = attention_mask_4d[..., :1, :]
        print(f"After [:, :, :1, :]: {attention_mask_truncated.shape}")

    # 创建新的 attention_mask
    new_attention_mask = torch.ones(
        (attention_mask_4d.shape[0], 1, 1, seq_len + 1),
        dtype=torch.bool,
    )
    new_attention_mask[..., :seq_len] = attention_mask_truncated[..., :seq_len]
    print(f"New attention_mask shape: {new_attention_mask.shape}")
    print(f"New attention_mask: {new_attention_mask}")

    print("\n" + "=" * 60)
    print("6. 问题总结")
    print("=" * 60)
    print("""
发现的问题:
1. attention_mask 不是 causal mask，模型可以看到未来 token
2. decode 阶段的 mask 处理逻辑有问题，从 [..., :1, :] 截断后再扩展可能不正确
3. 正确的做法应该是在每一层使用 position_ids 和 past_key_values 来处理位置信息
   同时使用正确的 causal mask

修复建议:
1. 使用 AttentionMaskConverter._create_4d_causal_attention_mask 创建正确的 mask
2. 或者使用 transformers 的 prepare_4d_causal_attention_mask 函数
3. 确保 decode 阶段 mask 正确更新
""")


def test_with_real_model():
    """使用真实模型测试不同 mask 的效果"""
    print("\n" + "=" * 60)
    print("使用真实模型测试 (简化版)")
    print("=" * 60)

    model_path = "/mnt/g/Models/DeepSeek-v2-lite-chat"

    # 加载模型和 tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    config = transformers.AutoConfig.from_pretrained(model_path)
    if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
        rope_scaling = dict(config.rope_scaling)
        if 'factor' in rope_scaling:
            rope_scaling['factor'] = float(rope_scaling['factor'])
        if 'beta_fast' in rope_scaling:
            rope_scaling['beta_fast'] = float(rope_scaling['beta_fast'])
        if 'beta_slow' in rope_scaling:
            rope_scaling['beta_slow'] = float(rope_scaling['beta_slow'])
        config.rope_scaling = rope_scaling

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 准备输入
    text = "Please tell me a joke."
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print(f"Input: {text}")
    print(f"Input IDs: {inputs['input_ids']}")
    print(f"Attention mask: {inputs['attention_mask']}")

    # 标准生成
    print("\n--- 使用 transformers 标准 generate (应该正常) ---")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")

    # 手动前向传播测试 mask
    print("\n--- 测试不同 mask 对 logits 的影响 ---")
    with torch.no_grad():
        # 正确的方式 (使用默认 attention_mask)
        outputs_correct = model(**inputs)
        logits_correct = outputs_correct.logits

    print(f"Logits shape: {logits_correct.shape}")
    print(f"Last token top 5 tokens (correct): {torch.topk(logits_correct[0, -1], 5)}")


if __name__ == "__main__":
    debug_attention_mask()

    # 如果需要测试真实模型，取消下面的注释
    # print("\n注意: 测试真实模型需要 GPU 和模型文件")
    # test_with_real_model()
