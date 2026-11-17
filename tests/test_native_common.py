"""Transport/HTTP integration fixtures; these tests do not load a language model."""
import json
from pathlib import Path
import threading
import pytest

from backends.common import NativeExport
from backends.server import make_server
from experiments.acquisition.consumer import collect_exports
from experiments.collection import collect, native_describe, run_schedule
from attacks.metadata import reconstruct_metadata


def test_native_exports_keep_three_views_and_separate_reference(tmp_path):
    with NativeExport(tmp_path / "exports", "opaque", sample_ms=1000, capacity_bytes=65536) as exports:
        publisher = exports.publisher
        publisher.snapshot(0, [], kind="initialization")
        publisher.event("LOAD", 0, 7)
        exports._sample()
        publisher.event("HIT", 0, 7)
        publisher.event("EVICT", 0, 7)
        receipt = exports.receipt()
    observation = collect_exports(receipt["metadata_exports"], receipt["metadata_shm_exports"], request_id="opaque")
    assert [row["op"] for row in observation["source_views"]["logs"]["events"]] == ["LOAD", "HIT", "EVICT"]
    for source in ("telemetry", "shm"):
        view = observation["source_views"][source]
        assert not view["events"]
        assert any(row["resident"] == [7] for row in view["snapshots"])
        assert view["snapshots"][-1]["resident"] == []
    reference = json.loads(Path(receipt["reference_path"]).read_text())
    assert [row["op"] for row in reference["events"]] == ["LOAD", "HIT", "EVICT"]
    assert all("prompt" not in row and "tensor" not in row for row in reference["events"])


def test_native_prefill_first_load_has_sampled_empty_baseline(tmp_path):
    with NativeExport(tmp_path / "exports", "prefill", sample_ms=1000, capacity_bytes=65536) as exports:
        exports.publisher.event("LOAD", 0, 7)
        exports._sample()
        receipt = exports.receipt()
    observation = collect_exports(receipt["metadata_exports"], receipt["metadata_shm_exports"], request_id="prefill")
    for source in ("telemetry", "shm"):
        events = reconstruct_metadata(observation["source_views"][source])
        assert [(event["op"], event["block_id"]) for event in events] == [("LOAD", 7)]


class FixtureRuntime:
    """Controlled request handler used only for localhost transport tests."""
    config = {"system": "quest"}
    concurrent_requests = True

    def __init__(self, barrier=None):
        self.barrier = barrier

    def describe_request(self, prompt, max_new_tokens=None):
        return dict(capacity_slots=13, token_count=1000, block_tokens=16,
                    residency_fraction=.2, generation_tokens=max_new_tokens or 2)

    def generate(self, prompt, request_id, export_dir, max_new_tokens=None):
        if self.barrier:
            self.barrier.wait(timeout=5)
        with NativeExport(export_dir, request_id, sample_ms=1000, capacity_bytes=65536) as exports:
            exports.publisher.snapshot(0, [], kind="initialization")
            exports.publisher.event("LOAD", 0, 3)
            exports._sample()
            exports.publisher.event("HIT", 0, 3)
            result = dict(self.describe_request(prompt, max_new_tokens), **exports.receipt(),
                          fixture=True, prompt=prompt, generated_text="fixture text", system="quest")
        return result


class FailingFixtureRuntime(FixtureRuntime):
    def generate(self, prompt, request_id, export_dir, max_new_tokens=None):
        with NativeExport(export_dir, request_id, sample_ms=1000, capacity_bytes=65536) as exports:
            exports.publisher.event("LOAD", 0, 3)
            raise RuntimeError("Controlled fixture failure after one completed operation")


def start_server(runtime, path):
    server = make_server(runtime, "127.0.0.1", 0, path)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    return server, worker, f"http://127.0.0.1:{server.server_port}/generate"


def test_native_http_routes_concurrent_requests_to_separate_exports(tmp_path):
    server, worker, endpoint = start_server(FixtureRuntime(threading.Barrier(4)), tmp_path / "service")
    try:
        description = native_describe(endpoint, {"prompt": "fixture"})
        assert description["capacity_slots"] == 13
        schedule = run_schedule(endpoint,
            [{"sample_id": f"fixture-{index}", "payload": {"prompt": "fixture"}} for index in range(4)],
            4, [0, 0, 0, 0], collect_receipts=True)
        assert all(row["status"] == "completed" for row in schedule)
        paths = [row["native_receipt"]["reference_path"] for row in schedule]
        assert len(set(paths)) == 4
        assert all(Path(path).exists() for path in paths)
        assert all("prompt" not in row["native_receipt"] and "generated_text" not in row["native_receipt"] for row in schedule)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_native_collection_uses_returned_exports_and_actual_capacity(tmp_path):
    from datasets.preprocessing.annotate import load_taxonomy, label_record
    from datasets.preprocessing.grounding import validate_grounded_record, validate_grounding_receipt
    _, rules, _ = load_taxonomy()
    source = dict(domain="medical", text="What is diabetes mellitus?", source_id="fixture-source")
    labels, _ = label_record(source, rules)
    source["labels"] = dict(domain="medical", **labels)
    annotation = validate_grounded_record(source)
    server, worker, endpoint = start_server(FixtureRuntime(), tmp_path / "service")
    config_path = Path(__file__).resolve().parents[1] / "configs" / "quest.yaml"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(config=str(config_path), channel="meta", endpoint=endpoint,
        native=True, target_id="target", concurrency=1, arrival_offsets_ms=[0],
        requests=[dict(sample_id="target", source_id="fixture-source", domain="medical", labels=source["labels"],
                       annotation=annotation, prompt_sha256=annotation["prompt_sha256"],
                       payload={"prompt": source["text"]})])), encoding="utf-8")
    try:
        row = collect(manifest, tmp_path / "collected")
        assert row["acquisition_status"] == "completed"
        assert row["token_count"] == 1000
        assert row["defense"] == "none"
        assert validate_grounding_receipt(row) == annotation
        assert "text" not in row and "prompt" not in row
        observation = json.loads((tmp_path / "collected" / row["observation_path"]).read_text())
        assert [event["op"] for event in observation["events"]] == ["LOAD", "HIT"]
        assert all(observation["source_views"][source]["snapshots"] for source in ("telemetry", "shm"))
        assert (tmp_path / "collected" / row["reference_path"]).exists()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


@pytest.mark.parametrize("runtime_failure", [False, True])
def test_failed_acquisition_preserves_completed_reference_operations(tmp_path, monkeypatch, runtime_failure):
    if not runtime_failure:
        def fail_observer(*args, **kwargs):
            raise ValueError("Controlled observer failure")
        monkeypatch.setattr("experiments.acquisition.consumer.collect_exports", fail_observer)
    runtime = FailingFixtureRuntime() if runtime_failure else FixtureRuntime()
    server, worker, endpoint = start_server(runtime, tmp_path / "service")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(
        config=str(Path(__file__).resolve().parents[1] / "configs" / "quest.yaml"),
        channel="meta", endpoint=endpoint, native=True, target_id="target",
        concurrency=1, arrival_offsets_ms=[0],
        requests=[dict(sample_id="target", source_id="fixture-source", domain="medical",
                       payload={"prompt": "fixture"})])), encoding="utf-8")
    try:
        row = collect(manifest, tmp_path / "collected")
        assert row["acquisition_status"] == "failed"
        assert row["request_status"] == ("failed" if runtime_failure else "completed")
        reference = json.loads((tmp_path / "collected" / row["reference_path"]).read_text())
        assert reference["events"][0]["op"] == "LOAD"
        assert row["native_receipt_path"] == "native-receipt.json"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
