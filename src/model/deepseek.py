import copy
import json
import os
import time
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2RotaryEmbedding
from transformers.masking_utils import create_causal_mask

from expert_scheduling import (
    ExpertSchedulingStrategy,
    GPUOnlyStrategy,
    FiddlerStrategy,
    PrefetchHybridStrategy,
    _build_latency_lookup,
)
from placeholder_manager import ExpertPlaceholderManager
from eviction_strategy import LRUEvictionStrategy
from expert_predictor import ExpertPredictor, GatePredictor
from expert_executor import ExpertExecutionManager
from expert_latency import ExpertLatencyModel
from expert_types import ExpertDemand, ExpertKey, ExpertLayerContext, ExpertSchedule, build_assignments

class FiddlerDeepSeek:

    def __init__(self, args):
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")

        # 先加载 config 并修复 rope_scaling 参数类型，再加载模型
        config = transformers.AutoConfig.from_pretrained(args.model)

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            torch_dtype=self.dtype,
        )

        self.batch_size = args.batch_size
        self.lm_head = self.model.lm_head
        self.model = self.model.model

        first_layer_mlp = self.model.layers[1].mlp
        template_expert = first_layer_mlp.experts[0]
        self.placeholder_manager = ExpertPlaceholderManager(
            template_expert=template_expert,
            device=self.dev,
            num_placeholders=12,
            eviction_strategy=LRUEvictionStrategy(),
        )
        self.expert_placeholder = self.placeholder_manager._placeholders[0]

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

        ### 加载基准数据，设置专家调度策略的 CPU/GPU 延迟参数
        self.latency_cpu = 0.142
        self.latencu_copy = 1.67
        self.latencu_gpu = 0.093
        self.latency_cpu_table = {1: 0.142}
        self.latency_gpu_table = {1: 0.093}
        benchmark_data = self._load_benchmark_data(args.model)
        if benchmark_data is not None:
            try:
                self.latency_cpu = benchmark_data["expert_cpu"][0]["avg_time_ms"]
                self.latencu_copy = benchmark_data["expert_weight_copy"]["avg_ms"]
                self.latencu_gpu = benchmark_data["expert_gpu"][0]["avg_time_ms"]
                self.latency_cpu_table = _build_latency_lookup(benchmark_data["expert_cpu"])
                self.latency_gpu_table = _build_latency_lookup(benchmark_data["expert_gpu"])
                print(f"Loaded benchmark latency_cpu={self.latency_cpu:.4f}ms, latencu_copy={self.latencu_copy:.4f}ms, latencu_gpu={self.latencu_gpu:.4f}ms")
                print(f"Latency CPU table: {self.latency_cpu_table}")
                print(f"Latency GPU table: {self.latency_gpu_table}")
            except (KeyError, TypeError) as e:
                print(f"Warning: Malformed benchmark data, using defaults: {e}")
        else:
            print(f"Benchmark file not found, using default latency_cpu={self.latency_cpu}ms, latencu_copy={self.latencu_copy}ms")

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        # 初始化策略
        if args.cpu_offload == 0:
            self.expert_strategy = GPUOnlyStrategy(self.dev, self.is_expert_in_gpu)
        else:
            self.expert_strategy = PrefetchHybridStrategy(
                self.dev, self.is_expert_in_gpu,
                t_io=self.latencu_copy,
                latency_cpu_table=self.latency_cpu_table,
                latency_gpu_table=self.latency_gpu_table,
            )
            self.hybrid_strategy = self.expert_strategy
        self.gpu_only_strategy = GPUOnlyStrategy(self.dev, self.is_expert_in_gpu)

        self.expert_predictor = GatePredictor()
        self.predicted_next_demands = []
        self.latency_model = ExpertLatencyModel(
            t_io=self.latencu_copy,
            latency_cpu_table=self.latency_cpu_table,
            latency_gpu_table=self.latency_gpu_table,
        )

        load_model_tick = time.time()
        self.bring_non_expert_to_gpu()
        print("non-expert modules loaded to GPU, time:", time.time() - load_model_tick)

        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)
        n_expert_on_gpu = self.calc_n_expert_on_gpu()
        print(
            f"Number of experts on GPU: {n_expert_on_gpu}/{(self.n_layer - 1) * self.n_expert}"
        )

        self.set_expert_loc(n_expert_on_gpu)

        load_model_tick = time.time()
        self.bring_expert_to_gpu()
        self.expert_executor = ExpertExecutionManager(
            device=self.dev,
            placeholder_manager=self.placeholder_manager,
            model=self.model,
            is_expert_in_gpu=self.is_expert_in_gpu,
        )
        print("experts loaded to GPU, time:", time.time() - load_model_tick)
        print("Model is ready.")

    def _load_benchmark_data(self, model_path):
        model_name = os.path.basename(model_path.rstrip('/')).lower()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, f"micro_{model_name}.json")
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"Warning: Failed to load benchmark file {filepath}: {e}")
            return None



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
        n_expert_on_gpu = min(n_expert_on_gpu, len( popular_experts))
        for i in range(n_expert_on_gpu):
            i_layer, i_expert = popular_experts[i]
            self.expert_loc[i_layer, i_expert] = 1

    def bring_expert_to_gpu(self):
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                if self.is_expert_in_gpu(i, j):
                    self.model.layers[i].mlp.experts[j].to(self.dev)
                    self.placeholder_manager.mark_static_gpu_resident(i, j)
                else:
                    self.placeholder_manager.mark_cpu_resident(i, j)

    def is_expert_in_gpu(self, i_layer, i_expert):
        return self.expert_loc[i_layer, i_expert] == 1 or self.placeholder_manager.is_on_gpu(i_layer, i_expert)

    def calc_n_expert_on_gpu(self):
        fine_expert = self.model.layers[1].mlp.experts[0]
        # numel:单个参数张量里的元素总数，即参数的维度乘积
        n_param = sum(p.numel() for p in fine_expert.parameters())
        bytes_per_param = 2 if self.dtype == torch.bfloat16 else 4
        expert_mem_mb = n_param * bytes_per_param / 1024 / 1024
        print(f"Number of parameters in a single expert: {n_param}, memory: {expert_mem_mb:.2f} MB")

        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        # 80% of total memory for safety margin
        #torch.cuda.memory_allocated：PyTorch 官方提供的显存统计 API，专门统计已使用的显存
        free_mem = total_mem * 0.80 - torch.cuda.memory_allocated(self.dev) 
        print(f"Total GPU memory: {total_mem / 1024 / 1024:.2f} MB, Free GPU memory: {free_mem / 1024 / 1024:.2f} MB")
        return int(free_mem // (n_param * 2))

    def initial_beam_tensor(self, input_tensor):
        # input_tensor shape: [batch, seq_len, beam_width] 或 [batch, beam_width]
        # Prefill 阶段，我们需要展开为 [batch*beam_width] 的形式
        if input_tensor.dim() == 3:
            input_tensor = input_tensor[:, -1, :]  # [batch, beam_width]
        assert input_tensor.shape[-1] == self.beam_width
        # 展开为 [batch*beam_width] 形状，后续再 view 回来
        return input_tensor.flatten()  # [batch*beam_width]

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
                new_probs = new_probs[:, -1].flatten()  # [batch]
                probs = probs * new_probs.repeat_interleave(self.beam_width)  # 广播到 [batch*beam_width]
                # print(new_probs.shape)
            else:
                new_probs, output = torch.topk(logits, self.beam_width, dim=-1)
                new_probs = self.initial_beam_tensor(new_probs)  # [batch*beam_width]
                output = self.initial_beam_tensor(output)      # [batch*beam_width]
                probs = new_probs
                # print(new_probs.shape, output.shape)
                search_start = True

            if search_start:
                # Decode阶段：input_ids需要保持单个候选的第一个元素
                input_ids = output.view(-1, 1).to(self.dev)
            else:
                # Prefill阶段：选择第一个 beam 作为起始
                input_ids = output.view(-1, self.beam_width)[:, 0].to(self.dev)


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

            selected_experts, routing_weights = layer.mlp.gate(inps)

            # 与当前层专家执行并行：预测下一层活跃专家
            predict_future = None
            if i_layer + 1 < self.n_layer:
                with ThreadPoolExecutor(max_workers=1) as pred_executor:
                    predict_future = pred_executor.submit(
                        self.expert_predictor.predict, inps, self.model, i_layer, 1
                    )

            # 收集预测结果（在专家执行完成后应已就绪）
            if predict_future is not None:
                pred_result = predict_future.result()
                self.predicted_next_demands = pred_result or []
            else:
                self.predicted_next_demands = []

            shared_output = layer.mlp.shared_experts(inps)

            inps_flat = inps.view(-1, hidden_dim)
            inps_after_experts = torch.zeros_like(inps_flat, device=self.dev)
            experts = layer.mlp.experts
        
            # 选择策略
            if force_gpu:
                strategy = self.gpu_only_strategy
            else:
                strategy = self.expert_strategy

            # 策略决策和预处理（PrefetchHybridStrategy 返回 4-tuple）
            result_tuple = strategy.decide_and_prepare(
                i_layer, experts, selected_experts, routing_weights, self.n_expert,
                future_demands=self.predicted_next_demands,
                placement=self.placeholder_manager.snapshot(),
                is_prefill=is_prefill,
            )
            if len(result_tuple) == 4:
                cpu_experts, gpu_experts, prefetch_experts, expert_assignments = result_tuple
            else:
                cpu_experts, gpu_experts, expert_assignments = result_tuple
                prefetch_experts = []
            print(f"Layer {i_layer}: GPU experts: {gpu_experts}, CPU experts: {cpu_experts}, Prefetch: {prefetch_experts}")

            # 更新统计（如果是 Hybrid 策略）
            if hasattr(self, 'hybrid_strategy') and not force_gpu:
                self.cnt_expert_hit = self.hybrid_strategy.cnt_expert_hit
                self.cnt_expert_all = self.hybrid_strategy.cnt_expert_all

            schedule = ExpertSchedule(
                cpu=[ExpertDemand(ExpertKey(i_layer, eid), expert_assignments[eid][0].shape[0]) for eid in cpu_experts],
                gpu=[ExpertDemand(ExpertKey(i_layer, eid), expert_assignments[eid][0].shape[0]) for eid in gpu_experts],
                preload=[ExpertDemand(ExpertKey(i_layer + 1, eid), 1, source="predicted") for eid in prefetch_experts],
            )
            context = ExpertLayerContext(
                layer=i_layer,
                experts=experts,
                inps_flat=inps_flat,
                hidden_dim=hidden_dim,
                assignments=build_assignments(expert_assignments),
            )
            inps_after_experts = self.expert_executor.execute(schedule, context)
            
            total_expert_output = shared_output.view(-1, hidden_dim) + inps_after_experts
            inps = inps_residual + total_expert_output.reshape(batch_size, seq_len, hidden_dim)

        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        self.present_key_value = present_key_value
        return lm_logis

    def _prefetch_next_layer_experts(self, i_layer, prefetch_experts):
        """预取下一层专家权重到 GPU placeholder"""
        next_layer = i_layer + 1
        if next_layer >= self.n_layer:
            return
        next_experts = self.model.layers[next_layer].mlp.experts
        tick = time.time()
        loaded = 0
        for expert_id in prefetch_experts:
            if self.is_expert_in_gpu(next_layer, expert_id):
                continue
            placeholder = self.placeholder_manager.acquire_placeholder(next_layer, expert_id)
            if placeholder is None:
                break
            self.placeholder_manager.load_weights(placeholder, next_experts[expert_id])
            loaded += 1
        elapsed = time.time() - tick
        if loaded > 0:
            print(f"  Prefetch layer {next_layer}: {loaded} experts loaded in {elapsed*1000:.2f}ms")

    def run_expert_at_cpu(self, i_layer, i_expert, inps):
        return self.model.layers[i_layer].mlp.experts[i_expert](inps)
    
    def _execute_gpu_experts(self, i_layer, experts, gpu_experts, expert_assignments, inps_flat, hidden_dim):
        """执行 GPU 专家，返回结果张量（在 GPU 上）"""
        result = torch.zeros_like(inps_flat, device=self.dev)
        for i_expert in gpu_experts:
            top_2, routing_weight_subset = expert_assignments[i_expert]
            current_state = inps_flat[None, top_2.tolist()].reshape(-1, hidden_dim)
            
            if self.is_expert_in_gpu(i_layer, i_expert):
                current_state = experts[i_expert](current_state)
            else:
                placeholder = self.placeholder_manager.acquire_placeholder(i_layer, i_expert)
                self.placeholder_manager.load_weights(placeholder, experts[i_expert])
                current_state = placeholder(current_state)
                self.placeholder_manager.release_by_layer(i_layer)
            
            current_state = current_state * routing_weight_subset
            result.index_add_(0, top_2.to(self.dev, non_blocking=True), current_state.to(result.dtype))
        return result
    
    def _execute_cpu_experts(self, i_layer, experts, cpu_experts, expert_assignments, inps_flat, hidden_dim):
        """执行 CPU 专家，返回结果张量（在 CPU 上，之后会传回 GPU）"""
        result = torch.zeros_like(inps_flat, device='cpu')
        for i_expert in cpu_experts:
            top_2, routing_weight_subset = expert_assignments[i_expert]
            current_state = inps_flat[None, top_2.tolist()].reshape(-1, hidden_dim)
            current_state = self.run_expert_at_cpu(i_layer, i_expert, current_state.to("cpu"))
            current_state = current_state * routing_weight_subset.to("cpu")
            result.index_add_(0, top_2.to('cpu', non_blocking=True), current_state.to(result.dtype))
        return result
