"""Source evidence and measured observations remain bound before model fitting."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from datasets.preprocessing.annotate import load_taxonomy, label_record
from datasets.preprocessing.grounding import validate_grounded_record
from experiments.chapter5.live import _prompts
from experiments.collection import collect

ROOT = Path(__file__).resolve().parents[1]
runner = importlib.import_module("experiments.run")


def source_record(identity):
    _, rules, _ = load_taxonomy()
    row = dict(sample_id=identity, source_id="source-"+identity, domain="medical", text="What is asthma?")
    labels, _ = label_record(row, rules)
    row["labels"] = dict(domain="medical", **labels)
    row["prompt"] = row["text"]
    row["annotation"] = validate_grounded_record(row)
    row["prompt_sha256"] = row["annotation"]["prompt_sha256"]
    return row


def dump_rows(path, rows):
    path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")


def test_native_prompt_rejects_conflicting_nested_domain(tmp_path):
    row = source_record("source")
    row["labels"]["domain"] = "code"
    path = tmp_path/"input.jsonl"
    dump_rows(path, [row])
    with pytest.raises(ValueError, match="domains disagree"):
        _prompts(path, ground_semantic=True)


def test_payload_substitution_fails_before_native_service_call(tmp_path, monkeypatch):
    row = source_record("target")
    row["payload"] = dict(prompt="Unrelated replacement content")
    manifest = dict(config=str(ROOT/"configs"/"quest.yaml"), channel="meta", native=True,
                    endpoint="http://127.0.0.1:1/generate", target_id="target", requests=[row])
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid evidence must be rejected before contacting the model")
    monkeypatch.setattr("experiments.collection.native_describe", unexpected)
    with pytest.raises(ValueError, match="grounded prompt"):
        collect(path, tmp_path/"capture")


@pytest.mark.parametrize("attack", ["MSA", "SSA"])
@pytest.mark.parametrize("mutation", ["missing_evidence", "changed_labels", "swapped_trace"])
def test_measured_semantic_training_rejects_unbound_data_before_fit(tmp_path, monkeypatch, attack, mutation):
    paths = {}
    for split in ("train", "test"):
        row = source_record(split)
        row["provenance"] = "measured"
        obs = dict(request_id=split, start_ms=0, end_ms=30, capacity_slots=1,
                   events=[dict(op="LOAD", layer=0, block_id=1, time_ms=1)])
        payload = json.dumps(obs, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        row.update(observation=obs, observation_sha256=hashlib.sha256(payload).hexdigest())
        if split == "train":
            if mutation == "missing_evidence":
                row.pop("annotation")
            elif mutation == "changed_labels":
                row["labels"]["keyword"] = "diabetes_mellitus"
            else:
                row["observation"]["request_id"] = "another-request"
        paths[split] = tmp_path/f"{split}.jsonl"
        dump_rows(paths[split], [row])
    calibration = tmp_path/"calibration.json"
    from attacks.hardware import Calibration
    calibration.write_text(json.dumps(Calibration(150, None, None).to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="grounding|Grounding|trace bound"):
        runner.run(ROOT/"configs"/"quest.yaml", paths["train"], paths["test"], tmp_path/"model",
                   attack, calibration if attack == "SSA" else None)
    assert not (tmp_path/"model").exists()
