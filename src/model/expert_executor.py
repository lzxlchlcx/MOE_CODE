import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

import torch

from expert_types import ExpertLayerContext, ExpertSchedule


class ExpertExecutionManager:
    def __init__(self, device, placeholder_manager, model, is_expert_in_gpu: Callable[[int, int], bool]):
        self.device = device
        self.placeholder_manager = placeholder_manager
        self.model = model
        self.is_expert_in_gpu = is_expert_in_gpu
        self.preload_request_count = 0
        self.preload_success_count = 0
        self.preload_skip_count = 0
        self.preload_hit_count = 0

    def execute(self, schedule: ExpertSchedule, context: ExpertLayerContext) -> torch.Tensor:
        gpu_result = None
        cpu_result = None
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            if schedule.gpu_expert_ids:
                futures[executor.submit(self.execute_gpu_experts, context, schedule.gpu_expert_ids)] = "gpu"
            if schedule.cpu_expert_ids:
                futures[executor.submit(self.execute_cpu_experts, context, schedule.cpu_expert_ids)] = "cpu"
            if schedule.preload:
                futures[executor.submit(self.preload_experts, schedule.preload)] = "preload"

            results = {}
            for future in list(futures.keys()):
                results[futures[future]] = future.result()
            gpu_result = results.get("gpu")
            cpu_result = results.get("cpu")

        result = torch.zeros_like(context.inps_flat, device=self.device)
        if gpu_result is not None:
            result += gpu_result
        if cpu_result is not None:
            result += cpu_result.to(self.device, non_blocking=True)
        return result

    def execute_gpu_experts(self, context: ExpertLayerContext, expert_ids: List[int]) -> torch.Tensor:
        result = torch.zeros_like(context.inps_flat, device=self.device)
        for expert_id in expert_ids:
            assignment = context.assignments[expert_id]
            current_state = context.inps_flat[None, assignment.token_indices.tolist()].reshape(-1, context.hidden_dim)
            placeholder = self.placeholder_manager.get_placeholder_for_expert(context.layer, expert_id)
            if self.is_expert_in_gpu(context.layer, expert_id):
                current_state = context.experts[expert_id](current_state)
            elif placeholder is not None:
                self.preload_hit_count += 1
                current_state = placeholder(current_state)
            else:
                placeholder = self.placeholder_manager.acquire_placeholder(context.layer, expert_id)
                if placeholder is None:
                    raise RuntimeError(f"No placeholder available for expert ({context.layer}, {expert_id})")
                self.placeholder_manager.load_weights(placeholder, context.experts[expert_id])
                current_state = placeholder(current_state)
                self.placeholder_manager.release_by_layer(context.layer)

            current_state = current_state * assignment.routing_weights
            result.index_add_(
                0,
                assignment.token_indices.to(self.device, non_blocking=True),
                current_state.to(result.dtype),
            )
        return result

    def execute_cpu_experts(self, context: ExpertLayerContext, expert_ids: List[int]) -> torch.Tensor:
        result = torch.zeros_like(context.inps_flat, device="cpu")
        for expert_id in expert_ids:
            assignment = context.assignments[expert_id]
            current_state = context.inps_flat[None, assignment.token_indices.tolist()].reshape(-1, context.hidden_dim)
            current_state = self.model.layers[context.layer].mlp.experts[expert_id](current_state.to("cpu"))
            current_state = current_state * assignment.routing_weights.to("cpu")
            result.index_add_(
                0,
                assignment.token_indices.to("cpu", non_blocking=True),
                current_state.to(result.dtype),
            )
        return result

    def preload_experts(self, demands) -> None:
        by_layer: Dict[int, List[int]] = {}
        for demand in demands:
            by_layer.setdefault(demand.key.layer, []).append(demand.key.expert_id)

        for layer, expert_ids in by_layer.items():
            if layer >= len(self.model.layers):
                continue
            next_experts = self.model.layers[layer].mlp.experts
            tick = time.time()
            loaded = 0
            for expert_id in expert_ids:
                self.preload_request_count += 1
                if self.is_expert_in_gpu(layer, expert_id) or self.placeholder_manager.is_on_gpu(layer, expert_id):
                    self.preload_skip_count += 1
                    continue
                if self.placeholder_manager.is_loading(layer, expert_id):
                    self.preload_skip_count += 1
                    continue
                self.placeholder_manager.mark_loading(layer, expert_id)
                try:
                    placeholder = self.placeholder_manager.acquire_free_placeholder(layer, expert_id)
                    if placeholder is None:
                        self.preload_skip_count += 1
                        continue
                    self.placeholder_manager.load_weights(placeholder, next_experts[expert_id])
                    loaded += 1
                    self.preload_success_count += 1
                finally:
                    self.placeholder_manager.unmark_loading(layer, expert_id)
            elapsed = time.time() - tick
            if loaded > 0:
                print(f"  Preload layer {layer}: {loaded} experts loaded in {elapsed*1000:.2f}ms")
