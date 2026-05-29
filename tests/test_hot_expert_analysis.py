import json
import os
import sys
from collections import Counter


ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT, "src")
SCRIPTS_DIR = os.path.join(SRC_DIR, "scripts")
for path in (SRC_DIR, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import analyze_hot_experts
from analyze_hot_experts import load_sharegpt_prompts, sample_prompts, write_hot_outputs


def test_sharegpt_sampling_is_limited_and_reproducible(tmp_path):
    dataset = tmp_path / "sharegpt.json"
    dataset.write_text(
        json.dumps(
            [
                {"conversations": [{"from": "human", "value": "prompt 1"}]},
                {"conversations": [{"from": "gpt", "value": "skip"}, {"from": "human", "value": "prompt 2"}]},
                {"conversations": [{"from": "human", "value": "prompt 3"}]},
            ]
        ),
        encoding="utf-8",
    )

    prompts = load_sharegpt_prompts(str(dataset))
    first = sample_prompts(prompts, 2, seed=7)
    second = sample_prompts(prompts, 2, seed=7)
    oversized = sample_prompts(prompts, 99, seed=7)

    assert first == second
    assert len(first) == 2
    assert sorted(oversized) == sorted(prompts)


def test_write_hot_outputs_sorts_and_writes_stats(tmp_path):
    counts = Counter({(2, 7): 3, (1, 4): 3, (1, 2): 5})

    txt_path, stats_path = write_hot_outputs(
        counts=counts,
        output_dir=str(tmp_path / "hot"),
        output_name="deep.txt",
        dataset_path="sharegpt.json",
        seed=11,
        requested_samples=10,
        actual_samples=3,
        n_token=4,
    )

    assert os.path.exists(txt_path)
    assert os.path.exists(stats_path)
    assert open(txt_path, encoding="utf-8").read().splitlines() == [
        "1,2",
        "1,4",
        "2,7",
    ]

    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)
    assert stats["seed"] == 11
    assert stats["requested_samples"] == 10
    assert stats["actual_samples"] == 3
    assert stats["n_token"] == 4
    assert stats["total_routes"] == 11
    assert stats["experts"][0] == {"layer": 1, "expert": 2, "count": 5}


def test_hot_output_is_set_expert_loc_compatible(tmp_path):
    txt_path, _ = write_hot_outputs(
        counts=Counter({(3, 9): 1, (2, 8): 2}),
        output_dir=str(tmp_path / "hot"),
        output_name="deep.txt",
        dataset_path="sharegpt.json",
        seed=0,
        requested_samples=1,
        actual_samples=1,
        n_token=1,
    )

    parsed = [
        tuple(map(int, line.strip().split(",")))
        for line in open(txt_path, encoding="utf-8")
        if line.strip()
    ]
    assert parsed == [(2, 8), (3, 9)]


def test_analysis_entrypoint_generates_hot_files(tmp_path, monkeypatch):
    dataset = tmp_path / "sharegpt.json"
    dataset.write_text(
        json.dumps([{"conversations": [{"from": "human", "value": "prompt"}]}]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "hot"

    class FakeModel:
        def __init__(self, args):
            self.hot_expert_counts = Counter()

        def reset_hot_expert_stats(self):
            self.hot_expert_counts.clear()

        def generate(self, batch, output_token):
            self.hot_expert_counts.update({(1, 2): 4, (2, 3): 1})

    monkeypatch.setattr(analyze_hot_experts, "FiddlerDeepSeek", FakeModel)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_hot_experts.py",
            "--model",
            "fake-model",
            "--dataset",
            str(dataset),
            "--num-samples",
            "1",
            "--seed",
            "0",
            "--n-token",
            "2",
            "--output-dir",
            str(output_dir),
        ],
    )

    analyze_hot_experts.main()

    assert (output_dir / "deep.txt").read_text(encoding="utf-8").splitlines() == ["1,2", "2,3"]
    assert (output_dir / "deep_stats.json").exists()
