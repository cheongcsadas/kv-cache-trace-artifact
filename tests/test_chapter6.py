"""Small explicit fixtures exercise real forests and paired defense evaluation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from experiments.chapter6.cli import execute
from experiments.chapter6.evaluate import load_captures, latency, _validate_defense_execution, LOAD_BATCH_UNITS
from experiments.chapter6.plotting import figure7
from experiments.chapter6.specification import SYSTEMS, DEFENSES

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("medical", "financial", "legal", "code")


def fixture_manifest(tmp_path):
    captures = []
    for system in SYSTEMS:
        for defense in DEFENSES:
            directory = tmp_path/system/defense
            directory.mkdir(parents=True)
            config = yaml.safe_load((ROOT/"configs"/f"{system}.yaml").read_text())
            config.setdefault("experiment", {})["defense"] = defense
            config["inference"]["generation_tokens"] = 128
            config["cache"]["selected_token_budget"] = 512
            config_path = directory/"config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            collection = dict(system=system, setting="templated_short", block_tokens=16,
                              residency_fraction=.2, defense=defense)
            capture = dict(system=system, defense=defense, config=str(config_path),
                           setting="templated_short", channel="meta", collection=collection)
            for split, count in (("train", 2), ("test", 1)):
                rows = []
                for domain_index, domain in enumerate(DOMAINS):
                    for i in range(count):
                        original_id = f"{split}-{domain}-{i}"
                        sample = f"{system}-{defense}-{original_id}"
                        seconds = {"none": 4., "fixed_budget": 5., "dynamic_timing": 4.5}[defense]
                        rows.append(dict(sample_id=sample, original_sample_id=original_id,
                            source_id="source-"+original_id, domain=domain, system=system,
                            token_count=1000, prompt_sha256=hashlib.sha256(original_id.encode()).hexdigest(),
                            defense=defense, provenance="synthetic", request_status="completed",
                            acquisition_status="completed", collection=collection, template_id="shared-v1",
                            observation=dict(request_id=sample, start_ms=0, end_ms=30, snapshots=[],
                                events=[dict(op=op, layer=0, block_id=domain_index*10, time_ms=stamp)
                                        for op, stamp in (("LOAD", 1), ("HIT", 10), ("EVICT", 20))]),
                            native_receipt=dict(system=system, defense=defense, request_id=sample, token_count=1000,
                                execution_start_monotonic_ms=1000., execution_end_monotonic_ms=1000.+seconds*1000)))
                path = directory/f"{split}.jsonl"
                path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
                capture[split] = str(path)
            captures.append(capture)
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(dict(captures=captures)), encoding="utf-8")
    return path


def fixture_execution_counters(row, config):
    """Explicit contract fixture, never a captured experimental measurement."""
    defense, system = row["defense"], row["system"]
    fixed = defense == "fixed_budget"
    required, padding = (80, 55) if fixed else (10, 0)
    result = dict(row["native_receipt"], block_tokens=16, residency_fraction=.2,
        capacity_slots=13, generation_tokens=config["inference"]["generation_tokens"],
        selected_token_budget=config["cache"]["selected_token_budget"],
        defense_parameters=config["defenses"].get(defense, {}),
        defense_load_batch_unit=LOAD_BATCH_UNITS[system], defense_load_batches=2,
        defense_delay_calls=2 if defense == "dynamic_timing" else 0,
        defense_batch_transfer_counts=[65, 70] if fixed else [5, 5],
        defense_overflow_batches=int(fixed))
    if system == "quest":
        result.update(defense_required_loads=required, defense_padding_loads=padding)
    elif system == "infinigen":
        result.update(required_load_count=required, defense_padding_loads=padding)
    else:
        result["transfer_counts"] = dict(required=required, padding=padding)
    return result


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("defense", DEFENSES)
def test_native_defense_validation_binds_actual_parameters_and_batch_counts(tmp_path, system, defense):
    config = yaml.safe_load((ROOT/"configs"/f"{system}.yaml").read_text())
    config["experiment"] = dict(defense=defense)
    config["inference"]["generation_tokens"], config["cache"]["selected_token_budget"] = 128, 512
    row = dict(sample_id="sample", token_count=1000, system=system, defense=defense,
               native_receipt=dict(request_id="sample", token_count=1000, system=system, defense=defense,
                   execution_start_monotonic_ms=1000., execution_end_monotonic_ms=6000.))
    row["native_receipt"] = fixture_execution_counters(row, config)
    result = _validate_defense_execution(row, tmp_path/"data.jsonl", config)
    assert result == dict(eligible_batches=2,
        required_loads=80 if defense == "fixed_budget" else 10,
        padding_loads=55 if defense == "fixed_budget" else 0,
        delay_calls=2 if defense == "dynamic_timing" else 0)


@pytest.mark.parametrize("defense,field,value", [
    ("fixed_budget", "defense_parameters", dict(load_pages=64, block_tokens=16)),
    ("dynamic_timing", "defense_parameters", dict(max_delay_ms=4)),
    ("fixed_budget", "defense_load_batch_unit", "per_page"),
    ("fixed_budget", "defense_batch_transfer_counts", [32, 103]),
    ("fixed_budget", "defense_padding_loads", 0),
    ("fixed_budget", "defense_overflow_batches", 0),
    ("dynamic_timing", "defense_delay_calls", 10),
    ("fixed_budget", "defense_load_batches", 0),
    ("none", "block_tokens", 8),
    ("none", "residency_fraction", .4),
    ("none", "capacity_slots", 32),
    ("none", "generation_tokens", 64),
    ("none", "selected_token_budget", 1024),
    ("none", "request_id", "wrong-request"),
    ("none", "token_count", 3000),
])
def test_declared_defense_is_insufficient_without_matching_executed_counters(tmp_path, defense, field, value):
    config = yaml.safe_load((ROOT/"configs"/"quest.yaml").read_text())
    config["experiment"] = dict(defense=defense)
    config["inference"]["generation_tokens"], config["cache"]["selected_token_budget"] = 128, 512
    row = dict(sample_id="sample", token_count=1000, system="quest", defense=defense,
               native_receipt=dict(request_id="sample", token_count=1000, system="quest", defense=defense,
                   execution_start_monotonic_ms=1000., execution_end_monotonic_ms=6000.))
    row["native_receipt"] = fixture_execution_counters(row, config)
    row["native_receipt"][field] = value
    with pytest.raises(ValueError):
        _validate_defense_execution(row, tmp_path/"data.jsonl", config)


def test_measured_capture_path_requires_receipts_and_retains_failed_attempts(tmp_path):
    # Construct controlled receipt objects to exercise the measured-data contract.
    # These records are confined to this test and are not experimental results.
    path = fixture_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    for capture in manifest["captures"]:
        config = yaml.safe_load(Path(capture["config"]).read_text())
        for split in ("train", "test"):
            data = Path(capture[split])
            rows = [json.loads(line) for line in data.read_text().splitlines()]
            for row in rows:
                row["provenance"] = "measured"
                row["native_receipt"] = fixture_execution_counters(row, config)
            data.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    captures = load_captures(manifest, tmp_path, accuracy=True, pilot=True)
    assert captures["quest", "fixed_budget"]["execution_audit"]["test"]["validated_completed_attempts"] == 4
    data = Path(manifest["captures"][1]["test"])
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    rows[0].pop("native_receipt")
    data.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="execution receipt"):
        load_captures(manifest, tmp_path, accuracy=True, pilot=True)
    rows[0]["request_status"] = "failed"
    data.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    captures = load_captures(manifest, tmp_path, accuracy=True, pilot=True)
    result = captures["quest", "fixed_budget"]
    assert len(result["rows"]["test"]) == 4
    assert result["execution_audit"]["test"]["validated_completed_attempts"] == 3


def test_table8_executes_all_15_real_forest_training_and_prediction_paths(tmp_path):
    manifest = fixture_manifest(tmp_path)
    output = tmp_path/"accuracy"
    result = execute("e13_defense_accuracy", manifest, output, pilot=True, allow_synthetic=True)
    assert len(result["runs"]) == 15
    assert all(row["test_count"] == 4 and row["train_count"] == 8 for row in result["runs"])
    for row in result["runs"]:
        assert (output/row["id"]/"model.joblib").is_file()
        assert len((output/row["id"]/"predictions.jsonl").read_text().splitlines()) == 4
    for system in SYSTEMS:
        plain = json.loads((output/f"{system}-none-none"/"metrics.json").read_text())
        for defense in DEFENSES[1:]:
            aware = json.loads((output/f"{system}-{defense}-aware"/"metrics.json").read_text())
            oblivious = json.loads((output/f"{system}-{defense}-oblivious"/"metrics.json").read_text())
            assert oblivious["input_sha256"]["train"] == plain["input_sha256"]["train"]
            assert aware["input_sha256"]["train"] != plain["input_sha256"]["train"]
            assert oblivious["input_sha256"]["test"] == aware["input_sha256"]["test"]
    assert len((output/"table8.csv").read_text().splitlines()) == 4


def test_latency_uses_paired_native_execution_and_computes_overhead(tmp_path):
    manifest = fixture_manifest(tmp_path)
    result = execute("e14_defense_latency", manifest, tmp_path/"latency", pilot=True, allow_synthetic=True)
    assert len(result["runs"]) == 9
    for row in result["runs"]:
        assert row["paired_completed_prompts"] == 4
        assert row["mean_seconds"] == {"none": 4., "fixed_budget": 5., "dynamic_timing": 4.5}[row["defense"]]
        assert row["overhead_percent"] == {"none": 0., "fixed_budget": 25., "dynamic_timing": 12.5}[row["defense"]]
        assert row["timing_scope"] == "native_execution"


def test_latency_retains_failure_count_and_uses_common_completed_triplets(tmp_path):
    path = fixture_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    capture = next(c for c in manifest["captures"] if c["system"] == "quest" and c["defense"] == "fixed_budget")
    records = [json.loads(line) for line in Path(capture["test"]).read_text().splitlines()]
    records[0]["request_status"] = "failed"
    Path(capture["test"]).write_text("".join(json.dumps(r)+"\n" for r in records), encoding="utf-8")
    rows = latency(manifest, tmp_path, pilot=True, allow_synthetic=True)
    for row in (r for r in rows if r["system"] == "quest"):
        assert row["prompts"] == 4 and row["paired_completed_prompts"] == 3
        assert row["unpaired_prompts"] == 1
        assert row["failed_prompts"] == int(row["defense"] == "fixed_budget")


@pytest.mark.parametrize("field,value", [("request_id", "another-request"), ("token_count", 3000)])
def test_latency_rejects_receipt_from_a_different_request(tmp_path, field, value):
    path = fixture_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    data = Path(manifest["captures"][0]["test"])
    records = [json.loads(line) for line in data.read_text().splitlines()]
    records[0]["native_receipt"][field] = value
    data.write_text("".join(json.dumps(row)+"\n" for row in records), encoding="utf-8")
    with pytest.raises(ValueError, match="Latency receipt"):
        latency(manifest, tmp_path, pilot=True, allow_synthetic=True)


def test_plot_axis_error_keeps_computed_latency_results(tmp_path):
    pytest.importorskip("matplotlib")
    path = fixture_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    data = Path(manifest["captures"][0]["test"])
    records = [json.loads(line) for line in data.read_text().splitlines()]
    for row in records:
        row["native_receipt"]["execution_end_monotonic_ms"] = 10000.
    data.write_text("".join(json.dumps(row)+"\n" for row in records), encoding="utf-8")
    output = tmp_path/"latency"
    with pytest.raises(ValueError, match="--auto-y"):
        execute("e14_defense_latency", path, output, pilot=True, allow_synthetic=True, plot=True)
    assert json.loads((output/"summary.json").read_text())["runs"][0]["mean_seconds"] == 9.


@pytest.mark.parametrize("mutation", ["prompt", "defense", "generation", "encoding", "missing"])
def test_mismatched_defense_comparisons_are_rejected(tmp_path, mutation):
    path = fixture_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    capture = manifest["captures"][1]
    if mutation in {"generation", "encoding"}:
        config = yaml.safe_load(Path(capture["config"]).read_text())
        if mutation == "generation":
            config["inference"]["generation_tokens"] += 1
        else:
            config["implementation_choices"]["metadata_spatial_bins"] += 1
        Path(capture["config"]).write_text(yaml.safe_dump(config), encoding="utf-8")
    elif mutation == "missing":
        manifest["captures"].pop()
    else:
        records = [json.loads(line) for line in Path(capture["test"]).read_text().splitlines()]
        records[0]["prompt_sha256" if mutation == "prompt" else "defense"] = "0"*64 if mutation == "prompt" else "none"
        Path(capture["test"]).write_text("".join(json.dumps(row)+"\n" for row in records), encoding="utf-8")
    with pytest.raises(ValueError):
        load_captures(manifest, tmp_path, accuracy=True, pilot=True, allow_synthetic=True)


def test_figure7_renders_measured_bars_and_rejects_silent_clipping(tmp_path):
    pytest.importorskip("matplotlib")
    path = fixture_manifest(tmp_path)
    rows = latency(json.loads(path.read_text()), tmp_path, pilot=True, allow_synthetic=True)
    assert figure7(rows, tmp_path/"plot") == ["figure7.png", "figure7.pdf"]
    assert (tmp_path/"plot"/"figure7.png").stat().st_size > 1000
    changed = deepcopy(rows)
    changed[0]["mean_seconds"] = 8.2
    with pytest.raises(ValueError, match="--auto-y"):
        figure7(changed, tmp_path/"clipped")
    figure7(changed, tmp_path/"expanded", auto_y=True)


def test_existing_latency_summary_can_be_plotted_without_recapture(tmp_path):
    pytest.importorskip("matplotlib")
    from experiments.e14_defense_latency.plot import main
    path = fixture_manifest(tmp_path)
    result = execute("e14_defense_latency", path, tmp_path/"latency", pilot=True, allow_synthetic=True)
    assert len(result["runs"]) == 9
    main(["--input", str(tmp_path/"latency"/"summary.json"), "--output", str(tmp_path/"replot"), "--auto-y"])
    assert (tmp_path/"replot"/"figure7.png").is_file()


def test_native_plan_constructs_nine_real_defense_conditions_before_collection(tmp_path, monkeypatch):
    from experiments.chapter6 import live
    path = fixture_manifest(tmp_path)
    captures = json.loads(path.read_text())["captures"]
    project = dict(backends={}, datasets={}, generation_tokens=128, selected_token_budget=512)
    for system in SYSTEMS:
        baseline = next(c for c in captures if c["system"] == system and c["defense"] == "none")
        project["backends"][system] = dict(config=baseline["config"], source_dir="external", model_path="models")
        project["datasets"][system] = {"templated_short": {}}
        for split in ("train", "test"):
            original = [json.loads(line) for line in Path(baseline[split]).read_text().splitlines()]
            for row in original:
                row["prompt"] = "Original controlled fixture " + row["original_sample_id"]
                row.pop("prompt_sha256")
            prompts = tmp_path/f"{system}-{split}-prompts.jsonl"
            prompts.write_text("".join(json.dumps(row)+"\n" for row in original), encoding="utf-8")
            project["datasets"][system]["templated_short"][split] = str(prompts)
    project_path = tmp_path/"project.json"
    project_path.write_text(json.dumps(project), encoding="utf-8")
    seen = []
    def capture(project, base, condition, paths, config, directory, **kwargs):
        defense = config["experiment"]["defense"]
        seen.append((condition["system"], defense, config["defenses"]))
        assert condition["defense"] == defense
        return dict(config=str(directory/"config.yaml"), condition=dict(defense=defense),
                    **{split: str(value) for split, value in paths.items()})
    monkeypatch.setattr(live, "_collect_condition", capture)
    output = live.prepare_native("e13_defense_accuracy", project_path, tmp_path/"native", pilot=True)
    assert len(json.loads(output.read_text())["captures"]) == 9
    assert {(s, d) for s, d, _ in seen} == {(s, d) for s in SYSTEMS for d in DEFENSES}
    assert all(v["fixed_budget"] == dict(load_pages=65, block_tokens=16) and v["dynamic_timing"] == dict(max_delay_ms=5)
               for _, _, v in seen)
