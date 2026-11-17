import json
import pytest
from experiments.acquisition.metadata import MetadataPublisher, BackendHooks, read_export
from experiments.acquisition.probe import calibrate, empirical_threshold
from defenses.fixed_budget import FixedBudget, eligible
from defenses.dynamic_timing import DynamicTiming


def test_publisher_exports_real_transitions_and_excludes_cleanup(tmp_path):
    path = tmp_path/"export.jsonl"
    with MetadataPublisher(path, "r") as publisher:
        hooks = BackendHooks(publisher, "quest")
        publisher.snapshot(0, [], time_ms=0)
        hooks.completed("LOAD", 0, [1], phase="decode", time_ms=1)
        hooks.completed("HIT", 0, [1], phase="decode", time_ms=2)
        hooks.completed("EVICT", 0, [1], phase="decode", time_ms=3)
        hooks.completed("EVICT", 0, [999], phase="decode", cleanup=True, time_ms=4)
        publisher.snapshot(0, time_ms=5)
    obs = read_export(path, "r")
    assert [e["op"] for e in obs["events"]] == ["LOAD", "HIT", "EVICT"]
    assert obs["snapshots"][-1]["resident"] == []


def test_retrieval_excludes_query_copies_and_rejects_decode_load(tmp_path):
    with MetadataPublisher(tmp_path/"e.jsonl", "r") as p:
        h = BackendHooks(p, "retrievalattention")
        h.completed("LOAD", 0, [1], phase="decode", transfer_kind="query")
        with pytest.raises(ValueError): h.completed("LOAD", 0, [1], phase="decode")
        with pytest.raises(ValueError): p.event("HIT", 0, 99)


def test_fixed_budget_uses_real_callbacks_and_attention_only_required():
    defense = FixedBudget(seed=42)
    plan = defense.plan([1, 2], range(3, 100))
    copied, published = [], []
    def transfer(block):
        copied.append(block)
        return f"tensor-{block}"
    attention = defense.execute(plan, transfer, published.append)
    assert len(copied) == 65 and copied == published
    assert attention == ["tensor-1", "tensor-2"]
    assert plan.within_budget
    assert set(plan.padding).isdisjoint(plan.required)
    over = defense.plan(range(66), [])
    assert not over.within_budget and len(over.transfer_order) == 66
    with pytest.raises(ValueError): defense.plan([1], [])
    assert eligible("retrievalattention", "prefill")
    assert not eligible("retrievalattention", "decode")


def test_timing_preserves_call_and_bounds():
    timing = DynamicTiming(seed=4)
    delays = []
    for _ in range(10):
        assert timing.execute(lambda x: x+1, 4, sleeper=delays.append) == 5
    assert all(0 <= d <= .005 for d in delays)
    assert len(set(delays)) > 1


def test_calibration_no_test_reference_and_empirical_fpr():
    values = list(range(1, 101))
    upper, lower = empirical_threshold(values), empirical_threshold(values, upper=False)
    assert sum(v > upper for v in values) <= 5
    assert sum(v < lower for v in values) <= 5
    baseline = dict(provenance="measured", h2d=[dict(latency_us=150)],
                    compute=[dict(latency_us=v) for v in values], page=[])
    c = calibrate([baseline]*3)
    assert c["baseline_h2d_us"] == 150 and c["page_threshold_us"] is None
    with pytest.raises(ValueError): calibrate([baseline])
