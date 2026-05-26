import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2RotaryEmbedding
from transformers.masking_utils import create_causal_mask


# def _patch_dynamic_cache():
#     """为旧版 DynamicCache 补上 DeepSeek 模型所需的 get_usable_length 方法。
#     DeepSeek 调用: get_usable_length(new_seq_len, layer_idx)
#     DynamicCache: get_seq_length(layer_idx=0)
#     """
#     import transformers.cache_utils
#     dc = transformers.cache_utils.DynamicCache
#     if not hasattr(dc, 'get_usable_length'):
#         def get_usable_length(self, new_seq_len=None, layer_idx=0):
#             return self.get_seq_length(layer_idx)
#         dc.get_usable_length = get_usable_length


# _patch_dynamic_cache()


class FiddlerDeepSeek:

    def __init__(self, args):
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")

        # 先加载 config 并修复 rope_scaling 参数类型，再加载模型
        config = transformers.AutoConfig.from_pretrained(args.model)
        if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
            rope_scaling = dict(config.rope_scaling)
            if 'factor' in rope_scaling:
                rope_scaling['factor'] = float(rope_scaling['factor'])
            if 'beta_fast' in rope_scaling:
                rope_scaling['beta_fast'] = float(rope_scaling['beta_fast'])
            if 'beta_slow' in rope_scaling:
                rope_scaling['beta_slow'] = float(rope_scaling['beta_slow'])
            config.rope_scaling = rope_scaling

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            torch_dtype=self.dtype,
        )

        self.batch_size = args.batch_size
        self.lm_head = self.model.lm_head
        self.model = self.model.model

        first_layer_mlp = self.model.layers[1].mlp
        self.expert_placeholder = copy.deepcopy(
            first_layer_mlp.experts[0]
        ).to(self.dev)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.rotary_emb = DeepseekV2RotaryEmbedding(config=self.model.config, device=self.dev)
        self.config = self.model.config  # 保存 config 用于创建 causal mask

        self.past_key_value = transformers.cache_utils.DynamicCache()
        self.past_key_values_length = 0

        self.cpu_offload = args.cpu_offload
        self.beam_width = args.beam_width
        self.n_layer = len(self.model.layers)
        self.n_expert = self.model.config.n_routed_experts
        self.n_shared_experts = 2

        self.latency_cpu = 15
        self.latency_gpu = 40 # 将专家权重从 CPU 拷贝到 GPU 的固定延迟

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        load_model_tick = time.time()
        self.bring_non_expert_to_gpu()
        print("non-expert modules loaded to GPU, time:", time.time() - load_model_tick)

        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)
        max_seq_len = getattr(args, 'input_token', None) or 128
        max_seq_len += getattr(args, 'n_token', 20)
        n_expert_on_gpu = self.calc_n_expert_on_gpu(max_seq_len=max_seq_len)
        print(
            f"Number of experts on GPU: {n_expert_on_gpu}/{(self.n_layer - 1) * self.n_expert}"
        )

        self.set_expert_loc(n_expert_on_gpu)

        load_model_tick = time.time()
        self.bring_expert_to_gpu()
        print("experts loaded to GPU, time:", time.time() - load_model_tick)
        print("Model is ready.")

    def bring_non_expert_to_gpu(self):
        self.lm_head.to(self.dev)
        self.model.embed_tokens.to(self.dev)
        self.model.norm.to(self.dev)
        self.model.layers[0].to(self.dev)
        for i in range(len(self.model.layers)):
            if i != 0:
                self.model.layers[i].self_attn.to(self.dev)
                self.model.layers[i].input_layernorm.to(self.dev)
                self.model.layers[i].mlp.gate.to(self.dev)
                self.model.layers[i].post_attention_layernorm.to(self.dev)
        for i in range(1, self.n_layer):
            self.model.layers[i].mlp.shared_experts.to(self.dev)

    def set_expert_loc(self, n_expert_on_gpu, popular_experts=None):
        if popular_experts is None:
            hot_experts_file = './hot/deep.txt'
            if os.path.exists(hot_experts_file):
                try:
                    with open(hot_experts_file, 'r') as f:
                        popular_experts = [
                            tuple(map(int, line.strip().split(',')))
                            for line in f if line.strip()
                        ]
                    print(f"Loaded hot experts from {hot_experts_file}")
                except Exception as e:
                    print(f"Error loading hot experts: {e}")
            if popular_experts is None:
                popular_experts = []
                for layer in range(1, self.n_layer):
                    for expert in range(self.n_expert):
                        popular_experts.append((layer, expert))
        n_expert_on_gpu = min(n_expert_on_gpu, len(popular_experts))
        for i in range(n_expert_on_gpu):
            i_layer, i_expert = popular_experts[i]
            self.expert_loc[i_layer, i_expert] = 1

    def bring_expert_to_gpu(self):
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                if self.is_expert_in_gpu(i, j):
                    self.model.layers[i].mlp.experts[j].to(self.dev)

    def is_expert_in_gpu(self, i_layer, i_expert):
        return self.expert_loc[i_layer, i_expert] == 1

    def calc_n_expert_on_gpu(self, max_seq_len=160):
        fine_expert = self.model.layers[1].mlp.experts[0]
        n_param = sum(p.numel() for p in fine_expert.parameters())
        bytes_per_param = 2 if self.dtype == torch.bfloat16 else 4
        expert_mem_bytes = n_param * bytes_per_param
        expert_mem_mb = expert_mem_bytes / 1024 / 1024
        print(f"Single expert params: {n_param}, memory: {expert_mem_mb:.2f} MB")

        cfg = self.model.config
        bs = self.batch_size
        seq = max_seq_len
        n_layer = self.n_layer

        kv_cache_bytes = 2 * cfg.kv_lora_rank * bytes_per_param * n_layer * bs * seq

        attn_score_bytes = bs * cfg.num_attention_heads * seq * seq * bytes_per_param

        activation_bytes = bs * seq * cfg.hidden_size * bytes_per_param * 6 * n_layer

        causal_mask_bytes = bs * seq * (seq + seq) * bytes_per_param

        runtime_overhead = kv_cache_bytes + attn_score_bytes + activation_bytes + causal_mask_bytes
        safety_factor = 1.2
        runtime_overhead = int(runtime_overhead * safety_factor)

        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        allocated = torch.cuda.memory_allocated(self.dev)
        available = total_mem - allocated - runtime_overhead

        print(f"Total GPU: {total_mem / 1024 / 1024:.2f} MB")
        print(f"Already allocated: {allocated / 1024 / 1024:.2f} MB")
        print(f"  - KV cache est: {kv_cache_bytes / 1024 / 1024:.2f} MB")
        print(f"  - Attn scores est: {attn_score_bytes / 1024 / 1024:.2f} MB")
        print(f"  - Activations est: {activation_bytes / 1024 / 1024:.2f} MB")
        print(f"  - Causal mask est: {causal_mask_bytes / 1024 / 1024:.2f} MB")
        print(f"Runtime overhead (x{safety_factor}): {runtime_overhead / 1024 / 1024:.2f} MB")
        print(f"Available for experts: {available / 1024 / 1024:.2f} MB")

        return max(0, int(available // expert_mem_bytes))

    def initial_beam_tensor(self, input_tensor):
        # input_tensor shape: [batch, beam_width] 或 [batch, seq_len, beam_width]
        # 如果是 3D，取最后一个位置（预测下一个 token）；如果是 2D，直接使用
        if input_tensor.dim() == 3:
            input_tensor = input_tensor[:, -1, :]  # [batch, beam_width]
        assert input_tensor.shape[-1] == self.beam_width
        # 取第一个候选（概率最高的）
        return input_tensor[:, 0].view(-1, 1)

    def tokenize(self, text, input_token=None):
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            raise ValueError("text should be str or list of str")

        if len(text) < self.batch_size:
            text = text + [text[-1]] * (self.batch_size - len(text))
        elif len(text) > self.batch_size:
            text = text[:self.batch_size]

        encodings = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=input_token,
            return_tensors="pt"
        )
        input_ids = encodings.input_ids.to(self.dev)
        attention_mask = encodings.attention_mask.to(self.dev)

        seq_length = input_ids.shape[1]
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=self.dev
        ).unsqueeze(0).expand(input_ids.shape[0], -1)

        # 返回 2D attention_mask，让 mixtral_forward 中创建 causal mask
        return input_ids, position_ids, attention_mask

    def generate(self, text=None, output_token=20, input_token=None):
        torch.set_num_threads(16)
        self.past_key_value = transformers.cache_utils.DynamicCache()
        self.past_key_values_length = 0

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        if text is None:
            text = ["default input"] * self.batch_size
        elif isinstance(text, str):
            text = [text] * self.batch_size

        prompt = f"<|User|>: {text[0]} \n<|Assistant|>:"
        input_ids, position_ids, attention_mask = self.tokenize(text, input_token)

        if input_token is not None:
            input_ids = torch.stack([
                ids[:input_token] if len(ids) > input_token else ids
                for ids in input_ids
            ])
            position_ids = torch.stack([
                pos[:input_token] if len(pos) > input_token else pos
                for pos in position_ids
            ])
            attention_mask = attention_mask[:, :input_token] if input_token is not None else attention_mask

        tick = time.time()
        is_decode = False
        prefill_time, decode_time = 0, 0
        decode_strings = ["" for _ in range(input_ids.shape[0])]
        search_start = False
        probs = torch.full((input_ids.shape[0], 1), 1.0)

        for i_token in range(output_token):
            if is_decode:
                for i in range(input_ids.shape[0]):
                    decode_strings[i] += " " + self.tokenizer.decode(input_ids[i, :])

                # Decode 阶段：更新 attention_mask
                # 此时 input_ids 是 [batch, 1]，表示生成一个新 token
                # attention_mask 需要更新为 [batch, past_key_values_length + 1]
                past_seq_len = self.past_key_values_length
                attention_mask = torch.ones(
                    input_ids.shape[0], past_seq_len + 1,
                    dtype=torch.long, device=self.dev
                )
            else:
                # Prefill 阶段：attention_mask 保持不变
                pass

            new_position_ids = torch.arange(
                self.past_key_values_length,
                self.past_key_values_length + input_ids.shape[1],
                dtype=torch.long,
                device=self.dev
            ).unsqueeze(0).expand(input_ids.shape[0], -1)

            # 构建 cache_position
            cache_position = torch.arange(
                self.past_key_values_length,
                self.past_key_values_length + input_ids.shape[1],
                dtype=torch.long,
                device=self.dev
            )

            logits = self.mixtral_forward(input_ids, new_position_ids, attention_mask, cache_position, is_prefill=not is_decode)

            logits = logits.to("cpu")
            logits = F.softmax(logits, dim=-1)

            self.past_key_values_length += logits.shape[1]

            if search_start:
                new_probs, output = torch.topk(logits, 1, dim=-1)
                new_probs = new_probs[:, -1].flatten().view(-1, 1)
            else:
                new_probs, output = torch.topk(logits, self.beam_width, dim=-1)
                new_probs = self.initial_beam_tensor(new_probs)
                output = self.initial_beam_tensor(output)
                search_start = True

            probs = probs * new_probs

            input_ids = output[:, -1].flatten().view(-1, 1).to(self.dev)

            position_ids = (
                torch.arange(
                    self.past_key_values_length,
                    self.past_key_values_length + 1,
                    dtype=torch.long,
                    device=self.dev,
                )
                .unsqueeze(0)
                .view(-1, 1)
            )

            if not is_decode:
                prefill_time += time.time() - tick
                tick = time.time()
            is_decode = True

        decode_time = time.time() - tick
        probs = probs.view(-1, self.beam_width)
        max_ids = torch.argmax(probs, dim=-1)

        print("--------------------")
        print(f"Input: {text}")
        print(f"Output: {decode_strings[max_ids[0]]}")

        return (
            prefill_time,
            decode_time,
            self.cnt_expert_hit / max(self.cnt_expert_all, 1),
        )

    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids, attention_mask, cache_position, is_prefill=False):
        hidden_dim = self.model.config.hidden_size
        force_gpu = is_prefill

        inps = self.model.embed_tokens(input_ids)

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # 计算 position_embeddings (cos, sin tuple)
        position_embeddings = self.rotary_emb(inps, position_ids)

        # 使用 transformers 官方的 create_causal_mask 创建正确的 causal mask
        # 替换原来错误的 padding mask 扩展逻辑
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inps,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=self.past_key_value,
            position_ids=position_ids,
        )

        for i_layer, layer in enumerate(self.model.layers):
            if i_layer == 0:
                # 第一层：完整的 attention + dense FFN（非 MoE）
                original_inps_shape = inps.shape
                inps_residual = inps
                inps = layer.input_layernorm(inps)
                inps = inps.view(batch_size, seq_len, hidden_dim)

                attn_output = layer.self_attn(
                    hidden_states=inps,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=self.past_key_value,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

                if isinstance(attn_output, tuple):
                    if len(attn_output) == 2:
                        inps, present_key_value = attn_output
                    else:
                        inps, _, present_key_value = attn_output
                else:
                    present_key_value = None

                inps = inps_residual + inps
                inps_residual = inps
                inps = layer.post_attention_layernorm(inps)
                inps = inps.view(batch_size, seq_len, hidden_dim)
                # 第一层使用 dense MLP（非 MoE）
                inps = layer.mlp(inps)
                inps = inps_residual + inps
                continue

            original_inps_shape = inps.shape
            inps_residual = inps
            inps = layer.input_layernorm(inps)

            inps = inps.view(batch_size, seq_len, hidden_dim)

            attn_output = layer.self_attn(
                hidden_states=inps,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=self.past_key_value,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            if isinstance(attn_output, tuple):
                if len(attn_output) == 2:
                    inps, present_key_value = attn_output
                else:
                    inps, _, present_key_value = attn_output
            else:
                present_key_value = None

            inps = inps_residual + inps
            inps_residual = inps
            inps = layer.post_attention_layernorm(inps)
            inps = inps.view(batch_size, seq_len, hidden_dim)

            selected_experts, routing_weights  = layer.mlp.gate(inps)

            shared_output = layer.mlp.shared_experts(inps)

            inps_flat = inps.view(-1, hidden_dim)
            inps_after_experts = torch.zeros_like(inps_flat, device=self.dev)
            experts = layer.mlp.experts

            if self.cpu_offload == 0 or force_gpu:
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=self.n_expert
                ).permute(2, 1, 0)

                for i_expert in range(len(experts)):
                    is_cuda = self.is_expert_in_gpu(i_layer, i_expert)
                    idx, top_2 = torch.where(expert_mask[i_expert])

                    if top_2.shape[0] == 0:
                        continue

                    top_2_list = top_2.tolist()
                    idx_list = idx.tolist()

                    current_state = inps_flat[None, top_2_list].reshape(-1, hidden_dim)

                    if not is_cuda:
                        self.expert_placeholder.load_state_dict(
                            experts[i_expert].state_dict()
                        )
                        current_state = self.expert_placeholder(current_state)
                    else:
                        current_state = experts[i_expert](current_state)

                    # DeepSeek expert.forward() 不接受 routing_weights，需外部乘
                    current_state = current_state * routing_weights[top_2_list, idx_list, None]

                    inps_after_experts.index_add_(
                        0, top_2, current_state.to(inps_after_experts.dtype)
                    )

                    if not is_cuda:
                        experts[i_expert] = experts[i_expert].to("cpu")
            else:
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=self.n_expert
                ).permute(2, 1, 0)

                # 只收集有 token 分配的专家，避免 2^40 枚举
                idxs, top_2s = {}, {}
                active_experts = []
                for i_expert in range(len(experts)):
                    idx, top_2 = torch.where(expert_mask[i_expert])
                    if top_2.shape[0] > 0:
                        idxs[i_expert] = idx
                        top_2s[i_expert] = top_2
                        active_experts.append(i_expert)

                # 构建 active 专家的代价表
                # NOTE:但是在prefill阶段，可能激活过多专家导致2^n_active枚举不可行
                n_active = len(active_experts)
                # print(f"Layer {i_layer}: active_experts={active_experts}")
                cost_cpu = np.zeros(n_active, dtype=float)
                cost_gpu = np.zeros(n_active, dtype=float)
                for bit, i_expert in enumerate(active_experts):
                    token_count = top_2s[i_expert].shape[0]
                    cost_cpu[bit] = token_count * self.latency_cpu
                    cost_gpu[bit] = self.latency_gpu
                    if self.is_expert_in_gpu(i_layer, i_expert):
                        cost_gpu[bit] = 0
                        self.cnt_expert_hit += token_count
                    self.cnt_expert_all += token_count

                # 穷举 2^n_active 种配置（通常 n_active <= 10，2^10=1024 可接受）
                best_config = -1
                best_cost = float("inf")
                for config in range(1 << n_active):
                    sum_cost = 0
                    for bit in range(n_active):
                        if (config >> bit) & 1:
                            sum_cost += cost_cpu[bit]
                        else:
                            sum_cost += cost_gpu[bit]
                    if sum_cost < best_cost:
                        best_cost = sum_cost
                        best_config = config

                # 解码配置 → CPU/GPU 专家列表
                cpu_experts = []
                gpu_experts = []
                for bit, i_expert in enumerate(active_experts):
                    if (best_config >> bit) & 1:
                        cpu_experts.append(i_expert)
                    else:
                        gpu_experts.append(i_expert)
                # print(f"Layer {i_layer}: best_cost={best_cost:.2f}, cpu_experts={cpu_experts}, gpu_experts={gpu_experts}")

                for i_expert in gpu_experts:
                    top_2_list = top_2s[i_expert].tolist()
                    idx_list = idxs[i_expert].tolist()
                    current_state = inps_flat[None, top_2_list].reshape(-1, hidden_dim)
                    if self.is_expert_in_gpu(i_layer, i_expert):
                        current_state = experts[i_expert](current_state)
                    else:
                        self.expert_placeholder.load_state_dict(
                            experts[i_expert].state_dict()
                        )
                        current_state = self.expert_placeholder(current_state)
                    current_state = current_state * routing_weights[top_2_list, idx_list, None]
                    inps_after_experts.index_add_(
                        0,
                        top_2s[i_expert].to(self.dev, non_blocking=True),
                        current_state.to(inps_after_experts.dtype),
                    )

                for i_expert in cpu_experts:
                    top_2_list = top_2s[i_expert].tolist()
                    idx_list = idxs[i_expert].tolist()
                    current_state = inps_flat[None, top_2_list].reshape(-1, hidden_dim)
                    current_state = self.run_expert_at_cpu(
                        i_layer,
                        i_expert,
                        current_state.to("cpu"),
                    )
                    current_state = current_state * routing_weights[top_2_list, idx_list, None].to("cpu")
                    inps_after_experts.index_add_(
                        0,
                        top_2s[i_expert].to(self.dev, non_blocking=True),
                        current_state.to(self.dev).to(inps_after_experts.dtype),
                    )

            total_expert_output = shared_output.view(-1, hidden_dim) + inps_after_experts
            inps = inps_residual + total_expert_output.reshape(batch_size, seq_len, hidden_dim)

        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        self.present_key_value = present_key_value
        return lm_logis

    def run_expert_at_cpu(self, i_layer, i_expert, inps):
        return self.model.layers[i_layer].mlp.experts[i_expert](inps)
