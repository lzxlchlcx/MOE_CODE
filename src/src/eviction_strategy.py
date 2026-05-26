import time
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple


class EvictionStrategy(ABC):
    """占位符淘汰策略抽象基类"""

    @abstractmethod
    def on_acquire(self, placeholder_id: int, layer_id: int, expert_id: int):
        raise NotImplementedError

    @abstractmethod
    def on_access(self, placeholder_id: int):
        raise NotImplementedError

    @abstractmethod
    def on_release(self, placeholder_id: int):
        raise NotImplementedError

    @abstractmethod
    def select_victim(self, occupied_ids: List[int]) -> Optional[int]:
        raise NotImplementedError


class LRUEvictionStrategy(EvictionStrategy):
    """LRU（最近最少使用）淘汰策略"""

    def __init__(self):
        self._access_time: dict = {}

    def on_acquire(self, placeholder_id: int, layer_id: int, expert_id: int):
        self._access_time[placeholder_id] = time.monotonic()

    def on_access(self, placeholder_id: int):
        self._access_time[placeholder_id] = time.monotonic()

    def on_release(self, placeholder_id: int):
        self._access_time.pop(placeholder_id, None)

    def select_victim(self, occupied_ids: List[int]) -> Optional[int]:
        if not occupied_ids:
            return None
        return min(occupied_ids, key=lambda pid: self._access_time.get(pid, 0))


class FIFOEvictionStrategy(EvictionStrategy):
    """FIFO（先进先出）淘汰策略"""

    def __init__(self):
        self._acquire_order: dict = {}
        self._counter = 0

    def on_acquire(self, placeholder_id: int, layer_id: int, expert_id: int):
        self._acquire_order[placeholder_id] = self._counter
        self._counter += 1

    def on_access(self, placeholder_id: int):
        pass

    def on_release(self, placeholder_id: int):
        self._acquire_order.pop(placeholder_id, None)

    def select_victim(self, occupied_ids: List[int]) -> Optional[int]:
        if not occupied_ids:
            return None
        return min(occupied_ids, key=lambda pid: self._acquire_order.get(pid, float('inf')))
