"""Qwen3 MoE model (PDScope refactored)"""

import torch
import torch.nn.functional as F
import transformers
from base_model import FiddlerBaseModel
from config import ModelConfig


class FiddlerQwen(FiddlerBaseModel):
    def _load_model(self):
        return transformers.AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=self.dtype,
            use_cache=True,
            trust_remote_code=True,
        )

    def _create_tokenizer(self):
        tok = transformers.AutoTokenizer.from_pretrained(self.config.model_path)
        tok.pad_token = tok.eos_token
        return tok

    def _get_expert(self, layer, expert_id):
        return layer.mlp.experts[expert_id]

    def _get_gate(self, layer):
        return layer.mlp.gate

    def _compute_gate(self, gate, inps):
        router_logits = gate(inps)
        routing_weights = F.softmax(router_logits, dim=1)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        return selected_experts, routing_weights

    def _compute_attention(self, layer, inps, attention_mask, position_ids, past_kv):
        position_embeddings = self.model.rotary_emb(inps, position_ids)
        return layer.self_attn(
            hidden_states=inps,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_kv,
            use_cache=True,
        )

    def _expand_mask(self, attention_mask, seq_len):
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(-1, 32, -1, -1)
        return attention_mask

    def _get_shared_experts_output(self, layer, inps):
        return None

    def _is_layer_shared_only(self, layer_idx):
        return False

    def _get_n_experts(self):
        return len(self.model.layers[0].mlp.experts)

    def _get_n_shared_experts(self):
        return 0
