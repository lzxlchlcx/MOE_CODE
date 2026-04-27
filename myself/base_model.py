"""
PDScope FiddlerBaseModel: Unified MoE Inference Base Class

Extracts common logic from all four model implementations and adds:
- PDScope AdaptSched integration (Phase 2/3)
- Multi-stream concurrent transfer (Phase 4)
- Unified placeholder management
- Template method pattern for model-specific differences
"""

import copy
import threading
import time
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import transformers

from config import ModelConfig, load_cpu_time_table_from_file
from logger import BenchmarkLogger
from scheduler import AdaptSchedScheduler


if not hasattr(transformers.cache_utils.DynamicCache, "get_usable_length"):

    def _get_usable_length(self, kv_len, layer_idx=0):
        seq_len = self.get_seq_length(layer_idx)
        return seq_len if seq_len > 0 else kv_len

    transformers.cache_utils.DynamicCache.get_usable_length = _get_usable_length


class FiddlerBaseModel:
    """
    Abstract base class for MoE GPU-CPU hybrid inference.

    Subclasses must implement:
        _load_model()           -> the raw HF model
        _get_expert(layer, eid) -> expert module
        _get_gate(layer)        -> gate module
        _compute_gate(gate, inps) -> (selected_experts, routing_weights)
        _compute_attention(layer, inps, mask, pos_ids, past_kv) -> attn_output
        _expand_mask(attention_mask, seq_len) -> 4D mask
        _get_shared_experts_output(layer, inps) -> tensor or None
        _get_expert_layer_idx_for_placeholder(layer_idx) -> which layer to deep-copy from
    """

    def __init__(self, config: ModelConfig):
        if isinstance(config, ModelConfig):
            self.config = config
        else:
            self.config = ModelConfig.from_args(config)

        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")
        self.hot_experts = {}

        # Logger
        self.logger = BenchmarkLogger(self.config)

        # Scheduler (PDScope AdaptSched)
        self.scheduler = AdaptSchedScheduler(self.config)
        if not self.config.cpu_time_table and os.path.exists(
            self.config.cpu_time_table_file
        ):
            cpu_table = load_cpu_time_table_from_file(self.config.cpu_time_table_file)
            self.config.cpu_time_table = cpu_table
            self.scheduler.set_cpu_time_table(cpu_table)

        # Model loading (subclass implements)
        self.model = self._load_model()

        # Batch / cache
        self.batch_size = self.config.batch_size
        self.cache = self.config.cache
        self.cpu_offload = self.config.cpu_offload
        self.beam_width = self.config.beam_width

        # Extract components
        self.lm_head = (
            self.model.lm_head if hasattr(self.model, "lm_head") else self.model
        )
        if hasattr(self.model, "model"):
            self.lm_head = self.model.lm_head
            self.model = self.model.model

        # MoE config
        self.n_layer = self._get_n_layers()
        self.n_expert = self._get_n_experts()
        self.n_shared_experts = self._get_n_shared_experts()
        self.top_k = self.config.top_k
        self.hidden_dim = self._get_hidden_size()

        # Placeholders (unified management)
        self._init_placeholders()

        # Multi-stream transfer (Phase 4)
        param_count = len(self.config.expert_param_names)
        self._transfer_streams = [torch.cuda.Stream() for _ in range(param_count)]
        self._prefetch_stream = torch.cuda.Stream()
        self._prefetch_lock = threading.Lock()

        # Prefetch state
        self.prefetch_list = {}
        self.prefetching_list = {}
        self._prefetch_thread = None
        self.is_decode = False
        self.prefil_pre = False

        # Tokenizer
        self.tokenizer = self._create_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # KV cache
        self.past_key_value = transformers.cache_utils.DynamicCache()
        self.past_key_values_length = 0

        # Statistics
        self._init_stats()

        # GPU resource allocation
        self.bring_non_expert_to_gpu()
        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)
        self.expert_loc_now = np.zeros((self.n_layer, self.n_expert), dtype=int)
        n_expert_on_gpu = self.calc_n_expert_on_gpu()
        print(
            f"Number of experts on GPU: {n_expert_on_gpu}/{self._get_total_experts()}"
        )
        self.set_expert_loc(n_expert_on_gpu)

        # Load experts to GPU
        tick = time.time()
        self.bring_expert_to_gpu()
        print(f"Experts moved total time: {(time.time() - tick) * 1000:.2f}ms")
        print("Model is ready.")

    # ================================================================
    # Abstract / overridable methods (subclass must implement)
    # ================================================================

    def _load_model(self):
        raise NotImplementedError

    def _create_tokenizer(self):
        raise NotImplementedError

    def _get_expert(self, layer, expert_id):
        raise NotImplementedError

    def _get_gate(self, layer):
        raise NotImplementedError

    def _compute_gate(self, gate, inps):
        """Returns (selected_experts, routing_weights)"""
        raise NotImplementedError

    def _compute_attention(self, layer, inps, attention_mask, position_ids, past_kv):
        raise NotImplementedError

    def _expand_mask(self, attention_mask, seq_len):
        raise NotImplementedError

    def _get_shared_experts_output(self, layer, inps):
        return None

    def _is_layer_shared_only(self, layer_idx):
        return False

    def _get_n_layers(self):
        return len(self.model.layers)

    def _get_n_experts(self):
        return self.config.n_routed_experts

    def _get_n_shared_experts(self):
        return self.config.n_shared_experts

    def _get_hidden_size(self):
        return self.config.hidden_size

    def _get_total_experts(self):
        start = 1 if self._is_layer_shared_only(0) else 0
        return (self.n_layer - start) * self.n_expert

    def _get_placeholder_source_layer(self):
        for i in range(self.n_layer):
            if not self._is_layer_shared_only(i):
                return i
        return 0

    # ================================================================
    # Placeholder management (unified)
    # ================================================================

    def _init_placeholders(self):
        src_layer = self._get_placeholder_source_layer()
        num = self.config.num_placeholders
        self._placeholders = []
        self._placeholder_in_use = [False] * num
        self._placeholder_to_expert = [None] * num
        self.expert_to_placeholder = {}

        for i in range(num):
            expert = self._get_expert(self.model.layers[src_layer], i % self.n_expert)
            ph = copy.deepcopy(expert).to(self.dev)
            self._placeholders.append(ph)

    def _get_available_placeholder(self):
        for i, in_use in enumerate(self._placeholder_in_use):
            if not in_use:
                self._placeholder_in_use[i] = True
                return i, self._placeholders[i]
        return -1, None

    def _release_placeholder(self, index):
        if 0 <= index < len(self._placeholder_in_use):
            self._placeholder_in_use[index] = False
            self._placeholder_to_expert[index] = None

    def release_placeholders_for_layer(self, current_layer_idx):
        for i in range(len(self._placeholder_to_expert)):
            stored = self._placeholder_to_expert[i]
            if stored is not None:
                stored_layer, stored_expert = stored
                if stored_layer < current_layer_idx or (
                    stored_layer == self.n_layer - 1 and current_layer_idx <= 1
                ):
                    self._release_placeholder(i)
                    if stored in self.expert_to_placeholder:
                        del self.expert_to_placeholder[stored]

    def _find_placeholder_for_expert(self, layer_idx, expert_id):
        for i, stored in enumerate(self._placeholder_to_expert):
            if stored == (layer_idx, expert_id):
                return i
        return -1

    # ================================================================
    # Expert transfer (with multi-stream support - Phase 4)
    # ================================================================

    def _pin_expert_weights(self, layer_idx, expert_id):
        expert = self._get_expert(self.model.layers[layer_idx], expert_id)
        if next(expert.parameters()).is_cuda:
            return False
        for name in self.config.expert_param_names:
            w = getattr(expert, name)
            w.weight.data = w.weight.data.pin_memory()
        return True

    def _copy_expert_to_placeholder_multistream(
        self, layer_idx, expert_id, placeholder
    ):
        """Multi-stream concurrent transfer: each param matrix on separate CUDA stream."""
        param_names = self.config.expert_param_names
        expert = self._get_expert(self.model.layers[layer_idx], expert_id)

        if len(self._transfer_streams) >= len(param_names):
            for stream, name in zip(self._transfer_streams, param_names):
                with torch.cuda.stream(stream):
                    dst = getattr(placeholder, name).weight.data
                    src = getattr(expert, name).weight.data
                    dst.copy_(src, non_blocking=True)
            for stream in self._transfer_streams[: len(param_names)]:
                torch.cuda.synchronize(stream)
        else:
            for name in param_names:
                dst = getattr(placeholder, name).weight.data
                src = getattr(expert, name).weight.data
                dst.copy_(src, non_blocking=True)
            torch.cuda.synchronize()

    def _copy_expert_to_placeholder_singlestream(
        self, layer_idx, expert_id, placeholder
    ):
        """Single-stream transfer (legacy fallback)."""
        expert = self._get_expert(self.model.layers[layer_idx], expert_id)
        for name in self.config.expert_param_names:
            dst = getattr(placeholder, name).weight.data
            src = getattr(expert, name).weight.data
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()

    def async_ondemand(self, layer_idx, expert_id, placeholder):
        if not self._pin_expert_weights(layer_idx, expert_id):
            return
        self._copy_expert_to_placeholder_multistream(layer_idx, expert_id, placeholder)

    def async_load_expert(self, layer_idx, expert_id):
        expert = self._get_expert(self.model.layers[layer_idx], expert_id)
        if next(expert.parameters()).is_cuda:
            return -1

        self._pin_expert_weights(layer_idx, expert_id)

        idx, placeholder = self._get_available_placeholder()
        if placeholder is None:
            return -1

        self._placeholder_to_expert[idx] = (layer_idx, expert_id)
        self.expert_to_placeholder[(layer_idx, expert_id)] = (idx, placeholder)

        # Launch all param transfers on _prefetch_stream, using child streams
        # via record_wait to avoid nesting (nested with torch.cuda.stream
        # overwrites the outer stream, breaking subsequent synchronize).
        expert = self._get_expert(self.model.layers[layer_idx], expert_id)
        for stream, name in zip(self._transfer_streams, self.config.expert_param_names):
            with torch.cuda.stream(stream):
                dst = getattr(placeholder, name).weight.data
                src = getattr(expert, name).weight.data
                dst.copy_(src, non_blocking=True)
            # Enqueue a wait on prefetch_stream so that synchronize(prefetch_stream)
            # will block until this child stream completes.
            self._prefetch_stream.wait_stream(stream)

        return idx

    # ================================================================
    # GPU resource management
    # ================================================================

    def bring_non_expert_to_gpu(self):
        if hasattr(self, "lm_head") and self.lm_head is not None:
            self.lm_head.to(self.dev)
        self.model.embed_tokens.to(self.dev)
        self.model.norm.to(self.dev)

        if self._is_layer_shared_only(0):
            self.model.layers[0].to(self.dev)

        for i in range(len(self.model.layers)):
            if not self._is_layer_shared_only(i):
                layer = self.model.layers[i]
                layer.self_attn.to(self.dev)
                layer.input_layernorm.to(self.dev)
                layer.post_attention_layernorm.to(self.dev)
                gate = self._get_gate(layer)
                if hasattr(gate, "to"):
                    gate.to(self.dev)

        if self.n_shared_experts > 0:
            shared_layer_start = 1 if self._is_layer_shared_only(0) else 0
            for i in range(shared_layer_start, self.n_layer):
                try:
                    self.model.layers[i].mlp.shared_experts.to(self.dev)
                except Exception:
                    pass

    def bring_expert_to_gpu(self):
        expert_count = 0
        try:
            for i in range(self.n_layer):
                for j in range(self.n_expert):
                    if self.is_expert_in_gpu(i, j):
                        expert = self._get_expert(self.model.layers[i], j)
                        expert.to(self.dev)
                        expert_count += 1
            model_name = self.config.model_name
            with open("test.txt", "a") as f:
                f.write(
                    f"Model: {model_name}, batch_size: {self.batch_size}, loaded experts: {expert_count}\n"
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                model_name = self.config.model_name
                with open("test.txt", "a") as f:
                    f.write(
                        f"Model: {model_name}, batch_size: {self.batch_size}, OOM at: {expert_count}\n"
                    )
                raise
            else:
                raise

    def is_expert_in_gpu(self, i_layer, i_expert):
        return self.expert_loc[i_layer, i_expert] == 1

    def is_expert_in_gpu_now(self, i_layer, i_expert):
        expert = self._get_expert(self.model.layers[i_layer], i_expert)
        return next(expert.parameters()).is_cuda

    def calc_n_expert_on_gpu(self):
        src_layer = self._get_placeholder_source_layer()
        fine_expert = self._get_expert(self.model.layers[src_layer], 0)
        n_param = sum(p.numel() for p in fine_expert.parameters())

        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        used_mem = torch.cuda.memory_allocated(self.dev)
        free_mem = total_mem * 0.95 - used_mem

        expert_mem_bytes = n_param * 2
        calculated_n = int(free_mem / expert_mem_bytes)

        if self.batch_size >= 16:
            buffer_factor = 0.4
        elif self.batch_size >= 8:
            buffer_factor = 0.5
        else:
            buffer_factor = 0.7

        n_expert = int(calculated_n * buffer_factor)
        n_expert = max(n_expert, 1)
        max_experts = self._get_total_experts()
        n_expert = min(n_expert, max_experts)

        print(
            f"[calc_n_expert_on_gpu] GPU free: {free_mem / (1024**3):.2f}GB, "
            f"expert size: {expert_mem_bytes / (1024**2):.2f}MB, "
            f"result: {n_expert}/{max_experts}"
        )
        return n_expert

    def set_expert_loc(self, n_expert_on_gpu, popular_experts=None):
        if popular_experts is None:
            hot_experts_file = self.config.hot_expert_file
            if os.path.exists(hot_experts_file):
                try:
                    with open(hot_experts_file, "r") as f:
                        popular_experts = [
                            tuple(map(int, line.strip().split(",")))
                            for line in f
                            if line.strip()
                        ]
                except Exception as e:
                    print(f"Error loading hot experts: {e}")

            if popular_experts is None:
                popular_experts = []
                start = 1 if self._is_layer_shared_only(0) else 0
                for layer in range(start, self.n_layer):
                    for expert in range(min(40, self.n_expert)):
                        popular_experts.append((layer, expert))

        n_expert_on_gpu = min(n_expert_on_gpu, len(popular_experts))
        for i in range(n_expert_on_gpu):
            i_layer, i_expert = popular_experts[i]
            if i_layer < self.n_layer and i_expert < self.n_expert:
                self.expert_loc[i_layer, i_expert] = 1

    # ================================================================
    # Statistics
    # ================================================================

    def _init_stats(self):
        self.expert_selection_stats = []
        self.expert_time_stats = []
        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        layer_range = (
            range(1, self.n_layer)
            if self._is_layer_shared_only(0)
            else range(self.n_layer)
        )
        self.expert_selection_history = {i: [] for i in layer_range}
        self.hit_stats = {i: {"hits": 0, "total": 0} for i in layer_range}
        self.expert_weight_accumulator = {
            i: torch.zeros(self.top_k, device=self.dev) for i in layer_range
        }
        self.cpu_expert_time_per_layer = {i: 0.0 for i in layer_range}
        self.current_iter_expert_stats = {
            i: {"expert_ids": [], "token_counts": []} for i in layer_range
        }
        self.last_iter_expert_stats = {
            i: {"expert_ids": [], "token_counts": []} for i in layer_range
        }
        self.layer_time_stats = []
        self.layer_time_accumulator = {i: 0.0 for i in layer_range}

    # ================================================================
    # Tokenize
    # ================================================================

    def tokenize(self, text, input_token):
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            raise ValueError("text should be str or list of str")

        if len(text) < self.batch_size:
            text = text + [text[-1]] * (self.batch_size - len(text))
        elif len(text) > self.batch_size:
            text = text[: self.batch_size]

        encodings = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=input_token,
            return_tensors="pt",
        )
        input_ids = encodings.input_ids.to(self.dev)
        attention_mask = encodings.attention_mask.bool().to(self.dev)

        seq_length = input_ids.shape[1]
        position_ids = (
            torch.arange(seq_length, dtype=torch.long, device=self.dev)
            .unsqueeze(0)
            .expand(input_ids.shape[0], -1)
        )

        attention_mask = self._expand_mask(attention_mask, seq_length)
        return input_ids, position_ids, attention_mask

    # ================================================================
    # Generate loop
    # ================================================================

    def initial_beam_tensor(self, input_tensor):
        assert input_tensor.shape[-1] == self.beam_width
        input_tensor = input_tensor[:, -1]
        row_idx = torch.tensor(
            [
                i * self.beam_width
                for i in range(input_tensor.shape[0] // self.beam_width)
            ]
        )
        return input_tensor[row_idx].view(-1, 1)

    def generate(self, text=None, output_token=20, input_token=None):
        torch.set_num_threads(self.config.cpu_threads)

        self.past_key_value = transformers.cache_utils.DynamicCache()
        self.past_key_values_length = 0
        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0
        self.expert_selection_stats = []
        self.expert_time_stats = []
        self._init_stats()

        if text is None:
            text = ["default input"] * self.batch_size
        elif isinstance(text, str):
            text = [text] * self.batch_size

        input_ids, position_ids, attention_mask = self.tokenize(text, input_token)

        if input_token is not None:
            input_ids = torch.stack(
                [
                    ids[:input_token] if len(ids) > input_token else ids
                    for ids in input_ids
                ]
            )
            position_ids = torch.stack(
                [
                    pos[:input_token] if len(pos) > input_token else pos
                    for pos in position_ids
                ]
            )
            attention_mask = attention_mask[..., :input_token]

        tick = time.time()
        self.is_decode = False
        prefill_time, decode_time = 0, 0
        decode_strings = ["" for _ in range(input_ids.shape[0])]
        search_start = False
        probs = torch.full((input_ids.shape[0], 1), 1.0)
        self.token_decode_times = []
        self.perf_stats = {
            "token_embedding": [],
            "self_attention": [],
            "moe_gating": [],
            "expert_compute": [],
            "expert_compute-cpu": [],
        }

        for i_token in range(output_token):
            token_start_time = time.time()

            if self.is_decode:
                for i in range(input_ids.shape[0]):
                    decode_strings[i] += " " + self.tokenizer.decode(
                        input_ids[i, :].tolist()
                    )

            if self.is_decode:
                if (
                    attention_mask.dim() == 4
                    and attention_mask.shape[-1] == attention_mask.shape[-2]
                ):
                    attention_mask = attention_mask[..., :1, :]
                seq_len = attention_mask.shape[-1]
                new_attention_mask = torch.ones(
                    (attention_mask.shape[0], 1, 1, seq_len + 1),
                    dtype=torch.bool,
                    device=self.dev,
                )
                new_attention_mask[..., :seq_len] = attention_mask[..., :seq_len]
                attention_mask = new_attention_mask

            new_position_ids = (
                torch.arange(
                    self.past_key_values_length,
                    self.past_key_values_length + input_ids.shape[1],
                    dtype=torch.long,
                    device=self.dev,
                )
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1)
            )
            logits = self.mixtral_forward(input_ids, new_position_ids, attention_mask)
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

            token_time = time.time() - token_start_time
            self.token_decode_times.append(token_time)
            self.logger.log_token_decode(i_token, token_time)

            if not self.is_decode:
                prefill_time += time.time() - tick
                tick = time.time()
            self.is_decode = True

        decode_time = time.time() - tick
        probs = probs.view(-1, self.beam_width)
        max_ids = torch.argmax(probs, dim=-1)

        hit_rate = (
            self.cnt_expert_hit / self.cnt_expert_all
            if self.cnt_expert_all > 0
            else 0.0
        )
        return (
            prefill_time,
            decode_time,
            hit_rate,
            {
                "perf_stats": self.perf_stats,
                "expert_selection": self.expert_selection_stats,
                "expert_time": self.expert_time_stats,
                "layer_time": self.layer_time_stats,
                "outputs": decode_strings,
                "expert_hot_stats": self.get_expert_stats(),
                "layer_time_avg": {
                    i: self.layer_time_accumulator[i]
                    / max(
                        1, len([x for x in self.layer_time_stats if x["layer_id"] == i])
                    )
                    for i in self.layer_time_accumulator
                },
            },
        )

    # ================================================================
    # Forward pass (core inference with PDScope scheduling)
    # ================================================================

    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids, attention_mask):
        hidden_dim = self.hidden_dim
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        tick = time.time()
        inps = self.model.embed_tokens(input_ids)
        self.perf_stats["token_embedding"].append(time.time() - tick)

        if self.is_decode:
            layer_times = {i: 0.0 for i in range(self.n_layer)}

        present_key_value = None

        for i_layer, layer in enumerate(self.model.layers):
            layer_tick = time.time()

            # Shared-only layer (e.g., layer 0 in deepseek/moon)
            if self._is_layer_shared_only(i_layer):
                inps = layer.mlp(inps)
                continue

            # Release placeholders from previous layers
            self.release_placeholders_for_layer(i_layer)

            # Wait for prefetch
            if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
                self._prefetch_thread.join()
            torch.cuda.synchronize(self._prefetch_stream)

            # Residual + LayerNorm
            inps_residual = inps
            self.cpu_expert_time_per_layer[i_layer] = 0
            inps = layer.input_layernorm(inps)
            inps = inps.view(batch_size, seq_len, hidden_dim)

            # Self-Attention
            tick = time.time()
            attn_output = self._compute_attention(
                layer, inps, attention_mask, position_ids, self.past_key_value
            )
            torch.cuda.synchronize()
            self.perf_stats["self_attention"].append(time.time() - tick)

            if isinstance(attn_output, tuple):
                if len(attn_output) == 2:
                    inps, present_key_value = attn_output
                else:
                    inps, _, present_key_value = attn_output
            else:
                inps = attn_output
                present_key_value = None

            inps = inps_residual + inps
            inps_residual = inps
            inps = layer.post_attention_layernorm(inps)
            inps = inps.view(batch_size, seq_len, hidden_dim)

            # Gate / routing
            tick = time.time()
            gate = self._get_gate(layer)
            selected_experts, routing_weights = self._compute_gate(gate, inps)
            torch.cuda.synchronize()
            self.perf_stats["moe_gating"].append(time.time() - tick)

            self.expert_selection_stats.append(
                {"layer_id": i_layer, "expert_ids": selected_experts.tolist()}
            )

            # Shared experts
            shared_output = self._get_shared_experts_output(layer, inps)
            if shared_output is None:
                shared_output = torch.zeros_like(inps)

            # Expert token count statistics
            expert_token_counts = {}
            for expert_id in selected_experts.unique():
                mask = (selected_experts == expert_id).any(dim=1)
                expert_token_counts[expert_id.item()] = mask.sum().item()

            sorted_experts = sorted(
                expert_token_counts.items(), key=lambda x: x[1], reverse=True
            )
            self.current_iter_expert_stats[i_layer] = {
                "expert_ids": [e[0] for e in sorted_experts],
                "token_counts": [e[1] for e in sorted_experts],
            }

            # Filter: remove experts already on GPU
            filtered_experts = []
            for eid, tc in sorted_experts:
                already_gpu = False
                if self.is_expert_in_gpu_now(i_layer, eid):
                    already_gpu = True
                elif (
                    i_layer in self.prefetch_list and eid in self.prefetch_list[i_layer]
                ):
                    already_gpu = True
                elif (
                    i_layer in self.prefetching_list
                    and eid in self.prefetching_list[i_layer]
                ):
                    already_gpu = True
                else:
                    ph_idx = self._find_placeholder_for_expert(i_layer, eid)
                    if ph_idx >= 0:
                        already_gpu = True
                if not already_gpu:
                    filtered_experts.append((eid, tc))

            # Predict next layer experts (Gate-based prediction)
            next_layer_predicted = None
            if i_layer < self.n_layer - 1 and not self._is_layer_shared_only(
                i_layer + 1
            ):
                next_layer = self.model.layers[i_layer + 1]
                with torch.no_grad():
                    next_gate = self._get_gate(next_layer)
                    next_selected, _ = self._compute_gate(next_gate, inps)
                next_counts = {}
                for batch_idx in range(batch_size * seq_len):
                    for expert in next_selected[batch_idx]:
                        eid = expert.item()
                        next_counts[eid] = next_counts.get(eid, 0) + 1
                next_layer_predicted = sorted(
                    next_counts.items(), key=lambda x: x[1], reverse=True
                )

            # PDScope scheduling
            n_gpu_current = sum(
                1
                for eid, _ in sorted_experts
                if self.is_expert_in_gpu_now(i_layer, eid)
            )
            n_gpu_next = 0
            if next_layer_predicted:
                for eid, _ in next_layer_predicted:
                    if i_layer + 1 < self.n_layer and self.is_expert_in_gpu_now(
                        i_layer + 1, eid
                    ):
                        n_gpu_next += 1

            schedule_result = self.scheduler.schedule(
                layer_idx=i_layer,
                sorted_experts=filtered_experts,
                is_decode=self.is_decode,
                next_layer_predicted=next_layer_predicted,
                n_experts_on_gpu_current=n_gpu_current,
                n_experts_on_gpu_next=n_gpu_next,
            )
            ondemand_experts = schedule_result["ondemand_experts"]

            # Expert processing
            if self.cpu_offload == 0:
                inps_after_experts = self._process_all_gpu(
                    i_layer,
                    selected_experts,
                    routing_weights,
                    inps,
                    batch_size,
                    seq_len,
                    hidden_dim,
                )
            else:
                inps_after_experts = self._process_offload(
                    i_layer,
                    selected_experts,
                    routing_weights,
                    inps,
                    ondemand_experts,
                    batch_size,
                    seq_len,
                    hidden_dim,
                )

            total_expert_output = shared_output + inps_after_experts
            inps = inps_residual + total_expert_output.reshape(
                batch_size, seq_len, hidden_dim
            )

            if self.is_decode:
                layer_time = time.time() - layer_tick
                self.layer_time_stats.append(
                    {
                        "layer_id": i_layer,
                        "time": layer_time,
                        "token_step": self.past_key_values_length,
                    }
                )
                self.layer_time_accumulator[i_layer] += layer_time

            # Async prefetch for next layer
            prefetch_experts = schedule_result.get("prefetch_experts", [])
            if (
                self.config.prefetch_enabled
                and self.is_decode
                and prefetch_experts
                and i_layer < self.n_layer - 1
            ):
                self._start_prefetch(i_layer + 1, prefetch_experts)

            # Cleanup
            if i_layer in self.prefetch_list:
                del self.prefetch_list[i_layer]
            if i_layer in self.prefetching_list:
                del self.prefetching_list[i_layer]

        self.get_hot_expert()
        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)
        self.present_key_value = present_key_value
        return lm_logis

    # ================================================================
    # Expert processing (GPU-CPU hybrid)
    # ================================================================

    def _process_all_gpu(
        self,
        i_layer,
        selected_experts,
        routing_weights,
        inps,
        batch_size,
        seq_len,
        hidden_dim,
    ):
        inps_after = torch.zeros_like(inps, device=self.dev)
        for i_expert in selected_experts.unique():
            mask = (selected_experts == i_expert).any(dim=1)
            if not mask.any():
                continue
            expert_input = inps[mask.view(batch_size, seq_len)].view(-1, hidden_dim)
            output = self.run_expert_at_gpu(i_layer, i_expert.item(), expert_input)
            flat_mask = mask.view(-1)
            weights = routing_weights[flat_mask].gather(
                1,
                (selected_experts[flat_mask] == i_expert)
                .long()
                .argmax(dim=1, keepdim=True),
            )
            output = output * weights
            inps_after.view(-1, hidden_dim).index_add_(
                0, mask.nonzero().squeeze(1), output.to(inps_after.dtype)
            )
        return inps_after

    def _process_offload(
        self,
        i_layer,
        selected_experts,
        routing_weights,
        inps,
        ondemand_experts,
        batch_size,
        seq_len,
        hidden_dim,
    ):
        inps_after_experts = torch.zeros_like(inps, device=self.dev)
        selected_expert_ids = selected_experts.unique().tolist()

        experts_in_gpu = []
        experts_in_placeholder = []
        experts_ondemand = []
        cpu_experts = []

        for i_expert in selected_expert_ids:
            self.cnt_expert_all += 1
            if self.expert_loc[i_layer, i_expert] == 1 and self.is_expert_in_gpu_now(
                i_layer, i_expert
            ):
                self.cnt_expert_hit += 1
                self.logger.log_expert_hit(True)
            else:
                self.logger.log_expert_hit(False)

            if self.is_expert_in_gpu_now(i_layer, i_expert):
                experts_in_gpu.append(i_expert)
                self.logger.log_expert_class("gpu")
                continue

            ph_idx = self._find_placeholder_for_expert(i_layer, i_expert)
            if ph_idx >= 0:
                experts_in_placeholder.append((i_expert, ph_idx))
                self.logger.log_expert_class("prefetch")
                continue

            if i_expert in ondemand_experts:
                experts_ondemand.append(i_expert)
                self.logger.log_expert_class("ondemand")
            elif (
                i_layer in self.prefetch_list
                and i_expert in self.prefetch_list[i_layer]
            ) or (
                i_layer in self.prefetching_list
                and i_expert in self.prefetching_list[i_layer]
            ):
                experts_in_placeholder.append((i_expert, -1))
                self.logger.log_expert_class("loading")
            else:
                cpu_experts.append(i_expert)
                self.logger.log_expert_class("cpu")

        gpu_results = []
        cpu_results = []
        gpu_time = 0.0
        cpu_time = 0.0

        def process_gpu_experts():
            nonlocal gpu_time
            start = time.time()
            results = []

            for i_expert in experts_in_gpu:
                mask = (selected_experts == i_expert).any(dim=1)
                if not mask.any():
                    continue
                expert_input = inps[mask.view(batch_size, seq_len)].view(-1, hidden_dim)
                tick = time.time()
                output = self.run_expert_at_gpu(i_layer, i_expert, expert_input)
                self.perf_stats["expert_compute"].append(time.time() - tick)
                flat_mask = mask.view(-1)
                weights = routing_weights[flat_mask].gather(
                    1,
                    (selected_experts[flat_mask] == i_expert)
                    .long()
                    .argmax(dim=1, keepdim=True),
                )
                results.append((mask.nonzero().squeeze(1), output * weights))

            for i_expert, ph_idx in experts_in_placeholder:
                mask = (selected_experts == i_expert).any(dim=1)
                if not mask.any():
                    continue
                expert_input = inps[mask.view(batch_size, seq_len)].view(-1, hidden_dim)
                tick = time.time()

                if ph_idx >= 0 and ph_idx < len(self._placeholders):
                    torch.cuda.synchronize(self._prefetch_stream)
                    placeholder = self._placeholders[ph_idx]
                    output = placeholder(expert_input)
                else:
                    idx, placeholder = self._get_available_placeholder()
                    if placeholder is not None:
                        self._placeholder_to_expert[idx] = (i_layer, i_expert)
                        self.async_ondemand(i_layer, i_expert, placeholder)
                        output = placeholder(expert_input)
                    else:
                        expert_input_cpu = expert_input.to("cpu")
                        output = self.run_expert_at_cpu(
                            i_layer, i_expert, expert_input_cpu
                        )
                        output = output.to(self.dev)

                torch.cuda.synchronize()
                self.perf_stats["expert_compute"].append(time.time() - tick)
                flat_mask = mask.view(-1)
                weights = routing_weights[flat_mask].gather(
                    1,
                    (selected_experts[flat_mask] == i_expert)
                    .long()
                    .argmax(dim=1, keepdim=True),
                )
                results.append((mask.nonzero().squeeze(1), output * weights))

            for i_expert in experts_ondemand:
                mask = (selected_experts == i_expert).any(dim=1)
                if not mask.any():
                    continue
                expert_input = inps[mask.view(batch_size, seq_len)].view(-1, hidden_dim)
                tick = time.time()

                idx, placeholder = self._get_available_placeholder()
                if placeholder is not None:
                    self._placeholder_to_expert[idx] = (i_layer, i_expert)
                    self.async_ondemand(i_layer, i_expert, placeholder)
                    output = placeholder(expert_input)
                else:
                    expert_input_cpu = expert_input.to("cpu")
                    output = self.run_expert_at_cpu(i_layer, i_expert, expert_input_cpu)
                    output = output.to(self.dev)

                self.perf_stats["expert_compute"].append(time.time() - tick)
                flat_mask = mask.view(-1)
                weights = routing_weights[flat_mask].gather(
                    1,
                    (selected_experts[flat_mask] == i_expert)
                    .long()
                    .argmax(dim=1, keepdim=True),
                )
                results.append((mask.nonzero().squeeze(1), output * weights))

            gpu_results.extend(results)
            gpu_time = time.time() - start

        def process_cpu_experts():
            nonlocal cpu_time
            start = time.time()
            for i_expert in cpu_experts:
                mask = (selected_experts == i_expert).any(dim=1)
                if not mask.any():
                    continue
                expert_input = (
                    inps[mask.view(batch_size, seq_len)].view(-1, hidden_dim).to("cpu")
                )
                tick = time.time()
                output = self.run_expert_at_cpu(i_layer, i_expert, expert_input)
                self.perf_stats["expert_compute-cpu"].append(time.time() - tick)
                flat_mask = mask.view(-1)
                weights = (
                    routing_weights[flat_mask]
                    .gather(
                        1,
                        (selected_experts[flat_mask] == i_expert)
                        .long()
                        .argmax(dim=1, keepdim=True),
                    )
                    .to("cpu")
                )
                output = output * weights
                cpu_results.append((mask.nonzero().squeeze(1), output))
            cpu_time = time.time() - start

        parallel_start = time.time()
        gpu_thread = threading.Thread(target=process_gpu_experts)
        cpu_thread = threading.Thread(target=process_cpu_experts)
        gpu_thread.start()
        cpu_thread.start()
        gpu_thread.join()
        cpu_thread.join()
        parallel_time = time.time() - parallel_start

        self.logger.log_layer_stats(i_layer, gpu_time, cpu_time, parallel_time)

        # Unified reduction in main thread (no race condition)
        for mask_index, expert_output in gpu_results:
            expert_output = expert_output.view(-1, hidden_dim)
            inps_after_experts.view(-1, hidden_dim).index_add_(
                0, mask_index, expert_output.to(inps_after_experts.dtype)
            )

        for mask_index, expert_output in cpu_results:
            expert_output = expert_output.view(-1, hidden_dim)
            inps_after_experts.view(-1, hidden_dim).index_add_(
                0,
                mask_index.to(self.dev),
                expert_output.to(self.dev).to(inps_after_experts.dtype),
            )

        return inps_after_experts

    # ================================================================
    # Prefetch thread
    # ================================================================

    def _start_prefetch(self, next_layer_idx, expert_ids):
        if next_layer_idx not in self.prefetch_list:
            self.prefetch_list[next_layer_idx] = []
        if next_layer_idx not in self.prefetching_list:
            self.prefetching_list[next_layer_idx] = []

        tasks = []
        for eid in expert_ids:
            if not self.is_expert_in_gpu_now(next_layer_idx, eid):
                if (
                    eid not in self.prefetch_list[next_layer_idx]
                    and eid not in self.prefetching_list[next_layer_idx]
                ):
                    tasks.append(eid)

        if not tasks:
            return

        self.prefetching_list[next_layer_idx].extend(tasks)

        def _do_prefetch(tasks, layer_idx):
            for eid in tasks:
                try:
                    idx = self.async_load_expert(layer_idx, eid)
                    if idx >= 0:
                        with self._prefetch_lock:
                            if layer_idx in self.prefetch_list:
                                self.prefetch_list[layer_idx].append(eid)
                            if (
                                layer_idx in self.prefetching_list
                                and eid in self.prefetching_list[layer_idx]
                            ):
                                self.prefetching_list[layer_idx].remove(eid)
                except RuntimeError:
                    with self._prefetch_lock:
                        if (
                            layer_idx in self.prefetching_list
                            and eid in self.prefetching_list[layer_idx]
                        ):
                            self.prefetching_list[layer_idx].remove(eid)
                    break

        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            self._prefetch_thread.join()
        self._prefetch_thread = threading.Thread(
            target=_do_prefetch, args=(tasks, next_layer_idx), daemon=True
        )
        self._prefetch_thread.start()

    # ================================================================
    # Expert execution
    # ================================================================

    def run_expert_at_gpu(self, i_layer, i_expert, inps):
        start_time = time.time()
        expert = self._get_expert(self.model.layers[i_layer], i_expert)
        result = expert(inps)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        token_count = inps.shape[0]
        self.current_iter_expert_stats[i_layer]["expert_ids"].append(i_expert)
        self.current_iter_expert_stats[i_layer]["token_counts"].append(token_count)
        self.expert_time_stats.append(
            {
                "layer_id": i_layer,
                "expert_id": i_expert,
                "time": elapsed,
                "device": "gpu",
                "token_count": token_count,
                "status": "veryhot"
                if token_count > 2
                else ("hot" if token_count > 1 else "normal"),
            }
        )
        return result

    def run_expert_at_cpu(self, i_layer, i_expert, inps):
        start_time = time.time()
        expert = self._get_expert(self.model.layers[i_layer], i_expert)
        result = expert(inps)
        elapsed = time.time() - start_time

        if self.is_decode:
            self.cpu_expert_time_per_layer[i_layer] += elapsed

        token_count = inps.shape[0]
        self.current_iter_expert_stats[i_layer]["expert_ids"].append(i_expert)
        self.current_iter_expert_stats[i_layer]["token_counts"].append(token_count)
        self.expert_time_stats.append(
            {
                "layer_id": i_layer,
                "expert_id": i_expert,
                "time": elapsed,
                "device": "cpu",
                "token_count": token_count,
                "status": "veryhot"
                if token_count > 2
                else ("hot" if token_count > 1 else "normal"),
            }
        )
        return result

    # ================================================================
    # Hot expert tracking
    # ================================================================

    def get_hot_expert(self):
        if not self.is_decode:
            return {}

        hot_experts = {}
        layer_range = (
            range(1, self.n_layer)
            if self._is_layer_shared_only(0)
            else range(self.n_layer)
        )
        for layer_id in layer_range:
            eids = self.current_iter_expert_stats[layer_id]["expert_ids"]
            tcs = self.current_iter_expert_stats[layer_id]["token_counts"]
            if eids:
                sorted_pairs = sorted(zip(eids, tcs), key=lambda x: x[1], reverse=True)
                hot_experts[layer_id] = [e[0] for e in sorted_pairs]
            self.last_iter_expert_stats[layer_id] = {
                "expert_ids": list(eids),
                "token_counts": list(tcs),
            }
            self.current_iter_expert_stats[layer_id]["expert_ids"].clear()
            self.current_iter_expert_stats[layer_id]["token_counts"].clear()

        self.hot_experts = hot_experts
        return hot_experts

    def get_expert_stats(self):
        stats = {
            "hot_experts": {
                i: {"count": 0, "hot": 0, "veryhot": 0}
                for i in self.layer_time_accumulator
            },
            "hot_counts": {2: 0, 3: 0, 4: 0, 5: 0},
            "token_distribution": {},
        }
        for record in self.expert_time_stats:
            layer = record["layer_id"]
            if layer in stats["hot_experts"]:
                stats["hot_experts"][layer]["count"] += 1
                if record["status"] == "hot":
                    stats["hot_experts"][layer]["hot"] += 1
                elif record["status"] == "veryhot":
                    stats["hot_experts"][layer]["veryhot"] += 1

        token_counts = [r["token_count"] for r in self.expert_time_stats]
        for count in set(token_counts):
            stats["token_distribution"][count] = token_counts.count(count)
        return stats
