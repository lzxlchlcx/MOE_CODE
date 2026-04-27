"""Moonlight MoE model (PDScope refactored)"""

import torch
import transformers
from base_model import FiddlerBaseModel
from config import ModelConfig


class FiddlerMoon(FiddlerBaseModel):
    def _load_model(self):
        return transformers.AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=self.dtype,
            use_cache=True,
            trust_remote_code=True,
        )

    def _create_tokenizer(self):
        tok = transformers.AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )
        tok.pad_token = tok.eos_token
        return tok

    def _get_expert(self, layer, expert_id):
        return layer.mlp.experts[expert_id]

    def _get_gate(self, layer):
        return layer.mlp.gate

    def _compute_gate(self, gate, inps):
        selected_experts, routing_weights = gate(inps)
        return selected_experts, routing_weights

    def _compute_attention(self, layer, inps, attention_mask, position_ids, past_kv):
        return layer.self_attn(
            hidden_states=inps,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_kv,
            use_cache=True,
        )

    def _expand_mask(self, attention_mask, seq_len):
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(-1)
            attention_mask = attention_mask.expand(-1, -1, -1, seq_len)
        return attention_mask

    def _get_shared_experts_output(self, layer, inps):
        try:
            return layer.mlp.shared_experts(inps)
        except Exception:
            return None

    def _is_layer_shared_only(self, layer_idx):
        return layer_idx == 0

    def _get_n_experts(self):
        return self.model.config.n_routed_experts

    def _get_n_shared_experts(self):
        return 2
