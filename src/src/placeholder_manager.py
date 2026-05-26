import copy
import threading
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn

from eviction_strategy import EvictionStrategy


class ExpertPlaceholderManager:
    """专家占位符管理器，管理多个 GPU 占位符的分配、权重加载、释放和淘汰"""

    def __init__(
        self,
        template_expert: nn.Module,
        device: torch.device,
        num_placeholders: int = 1,
        eviction_strategy: Optional[EvictionStrategy] = None,
    ):
        self._device = device
        self._eviction_strategy = eviction_strategy

        self._placeholders: List[nn.Module] = []
        for i in range(num_placeholders):
            ph = copy.deepcopy(template_expert).to(device)
            self._placeholders.append(ph)

        self._available: set = set(range(num_placeholders))
        self._occupied: Dict[int, Tuple[int, int]] = {}
        self._reverse_map: Dict[Tuple[int, int], int] = {}
        self._lock = threading.Lock()

    @property
    def num_placeholders(self) -> int:
        return len(self._placeholders)

    def acquire_placeholder(
        self, layer_id: int, expert_id: int
    ) -> Optional[nn.Module]:
        with self._lock:
            if (layer_id, expert_id) in self._reverse_map:
                pid = self._reverse_map[(layer_id, expert_id)]
                if self._eviction_strategy is not None:
                    self._eviction_strategy.on_access(pid)
                return self._placeholders[pid]

            pid = self._find_free_placeholder()
            if pid is None:
                if self._eviction_strategy is None:
                    return None
                pid = self._evict_one()
                if pid is None:
                    return None

            self._assign(pid, layer_id, expert_id)
            return self._placeholders[pid]

    def load_weights(self, placeholder: nn.Module, expert: nn.Module):
        with self._lock:
            placeholder.load_state_dict(expert.state_dict())
            pid = self._placeholder_to_id(placeholder)
            if pid is not None and self._eviction_strategy is not None:
                self._eviction_strategy.on_access(pid)

    def release_placeholder(self, placeholder: nn.Module):
        with self._lock:
            pid = self._placeholder_to_id(placeholder)
            if pid is not None:
                self._release(pid)

    def release_by_layer(self, layer_id: int):
        with self._lock:
            to_release = [
                pid for pid, (lid, _) in self._occupied.items() if lid == layer_id
            ]
            for pid in to_release:
                self._release(pid)

    def is_available(self, placeholder: nn.Module) -> bool:
        pid = self._placeholder_to_id(placeholder)
        return pid is not None and pid in self._available

    def get_expert_for_placeholder(
        self, placeholder: nn.Module
    ) -> Optional[Tuple[int, int]]:
        pid = self._placeholder_to_id(placeholder)
        return self._occupied.get(pid)

    def get_placeholder_for_expert(
        self, layer_id: int, expert_id: int
    ) -> Optional[nn.Module]:
        with self._lock:
            pid = self._reverse_map.get((layer_id, expert_id))
            if pid is not None:
                return self._placeholders[pid]
            return None

    def _find_free_placeholder(self) -> Optional[int]:
        if self._available:
            return next(iter(self._available))
        return None

    def _evict_one(self) -> Optional[int]:
        occupied_ids = list(self._occupied.keys())
        if not occupied_ids:
            return None
        victim_id = self._eviction_strategy.select_victim(occupied_ids)
        if victim_id is not None:
            self._release(victim_id)
        return victim_id

    def _assign(self, pid: int, layer_id: int, expert_id: int):
        self._available.discard(pid)
        self._occupied[pid] = (layer_id, expert_id)
        self._reverse_map[(layer_id, expert_id)] = pid
        if self._eviction_strategy is not None:
            self._eviction_strategy.on_acquire(pid, layer_id, expert_id)

    def _release(self, pid: int):
        expert_key = self._occupied.pop(pid, None)
        if expert_key is not None:
            self._reverse_map.pop(expert_key, None)
        self._available.add(pid)
        if self._eviction_strategy is not None:
            self._eviction_strategy.on_release(pid)

    def _placeholder_to_id(self, placeholder: nn.Module) -> Optional[int]:
        for i, ph in enumerate(self._placeholders):
            if ph is placeholder:
                return i
        return None
