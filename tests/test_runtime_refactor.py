import os
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "src", "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from expert_latency import ExpertLatencyModel
from expert_scheduling import PDScopeScheduler
from expert_types import ExpertDemand, ExpertKey, ExpertLayerRequest, PlacementSnapshot
from placeholder_manager import ExpertPlaceholderManager


def test_scheduler_returns_disjoint_current_experts():
    latency = ExpertLatencyModel(
        t_io=1.0,
        latency_cpu_table={1: 0.5, 4: 3.0},
        latency_gpu_table={1: 0.1, 4: 0.2},
    )
    request = ExpertLayerRequest(
        layer=1,
        phase="decode",
        current=[
            ExpertDemand(ExpertKey(1, 0), 1, source="current"),
            ExpertDemand(ExpertKey(1, 1), 1, source="current"),
        ],
        future=[ExpertDemand(ExpertKey(2, 3), 1, source="predicted")],
        assignments={},
    )
    placement = PlacementSnapshot(gpu_resident={(1, 0)}, free_placeholders=1)
    schedule = PDScopeScheduler().schedule(request, placement, latency)
    cpu = {(d.key.layer, d.key.expert_id) for d in schedule.cpu}
    gpu = {(d.key.layer, d.key.expert_id) for d in schedule.gpu}
    assert cpu.isdisjoint(gpu)
    assert cpu | gpu == {(1, 0), (1, 1)}


def test_placeholder_snapshot_treats_placeholder_as_gpu():
    class TinyExpert:
        def to(self, device):
            return self

    manager = ExpertPlaceholderManager(TinyExpert(), device="cpu", num_placeholders=1)
    placeholder = manager.acquire_placeholder(1, 7)
    assert placeholder is not None
    snapshot = manager.snapshot()
    assert snapshot.is_on_gpu(1, 7)
    assert manager.is_on_gpu(1, 7)
