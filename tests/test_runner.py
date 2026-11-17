"""Integration regressions for honest train/test and defense evaluation."""

import copy
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from attacks.encoding import EncodedTrace
from attacks.hardware import LocalizationError
from experiments.config import load_config
from experiments.matrix import matrix

runner = importlib.import_module("experiments.run")
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quest.yaml"


def row(name, **kwargs):
    value = dict(sample_id=name, source_id="source-" + name, provenance="measured",
                 domain="medical", observation={"name": name,"request_id":name,"capacity_slots":13}, defense="none")
    value.update(kwargs)
    return value


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def save_inputs(tmp_path, train, test):
    train_file, test_file = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    write_rows(train_file, train)
    write_rows(test_file, test)
    return train_file, test_file


def encoded(observation, **kwargs):
    if observation.get("fail"):
        raise LocalizationError("No target activity")
    return EncodedTrace(np.array([[1.0]], dtype=np.float32), np.array([True]),
                        ("test_feature",), "hardware", 0, 10)


class FixedMedicalModel:
    def __init__(self, **kwargs):
        pass

    def fit(self, features, labels):
        self.fit_count = len(features)
        return self

    def predict(self, features):
        return ["medical"] * len(features)


def test_source_leakage_blocked_even_when_sample_ids_differ():
    with pytest.raises(ValueError, match="source-record leakage"):
        runner.validate_split([row("train", source_id="shared")], [row("test", source_id="shared")])


def test_sample_overlap_and_duplicate_ids_blocked():
    with pytest.raises(ValueError, match="sample overlap"):
        runner.validate_split([row("same", source_id="one")], [row("same", source_id="two")])
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        runner.validate_split([row("train"), row("train")], [row("test")])


def test_cross_template_test_requires_held_out_templates():
    with pytest.raises(ValueError, match="held out"):
        runner.validate_split([row("train", template_id="t1")], [row("test", template_id="t1")], True)
    runner.validate_split([row("train", template_id="t1")], [row("test", template_id="t2")], True)


@pytest.mark.parametrize("train_defense,test_defense,mode", [
    ("none", "fixed_budget", "aware"),
    ("none", "none", "aware"),
    ("fixed_budget", "fixed_budget", "oblivious"),
    ("none", "none", "oblivious"),
    ("fixed_budget", "fixed_budget", "none"),
    ("none", "unknown", "oblivious"),
    ("none", "none", "invalid-mode"),
])
def test_invalid_defense_modes_rejected_before_encoding(tmp_path, train_defense, test_defense, mode):
    train_file, test_file = save_inputs(tmp_path, [row("train", defense=train_defense)],
                                       [row("test", defense=test_defense)])
    with patch.object(runner, "encode_metadata", side_effect=AssertionError("Encoded invalid data")):
        with pytest.raises(ValueError):
            runner.run(CONFIG, train_file, test_file, tmp_path / "out", "MAA", defense_mode=mode)


@pytest.mark.parametrize("train_defense,test_defense,mode", [
    ("none", "none", "none"),
    ("fixed_budget", "fixed_budget", "aware"),
    ("dynamic_timing", "dynamic_timing", "aware"),
    ("none", "fixed_budget", "oblivious"),
    ("none", "dynamic_timing", "oblivious"),
])
def test_valid_defense_conditions_complete(tmp_path, train_defense, test_defense, mode):
    train_file, test_file = save_inputs(tmp_path, [row("train", defense=train_defense)],
                                       [row("test", defense=test_defense)])
    with patch.object(runner, "MAA", FixedMedicalModel), patch.object(runner, "encode_metadata", encoded):
        report = runner.run(CONFIG, train_file, test_file, tmp_path / "out", "MAA", defense_mode=mode)
    assert report["metrics"]["ai_asr_percent"] == 100
    assert report["defense_mode"] == mode
    assert (tmp_path / "out" / "predictions.jsonl").is_file()


def test_localization_failures_are_retained_in_end_to_end_denominator(tmp_path):
    train_file, test_file = save_inputs(
        tmp_path, [row("train"), row("train-failure", observation={"fail": True,"capacity_slots":13})],
        [row("test"), row("test-failure", observation={"fail": True,"capacity_slots":13})],
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"baseline_h2d_us": 150, "compute_threshold_us": 10,
                                       "page_threshold_us": 20}), encoding="utf-8")
    with patch.object(runner, "SAA", FixedMedicalModel), patch.object(runner, "encode_hardware", encoded):
        report = runner.run(CONFIG, train_file, test_file, tmp_path / "out", "SAA", calibration)
    assert report["metrics"]["n_test"] == 2
    assert report["metrics"]["ai_asr_percent"] == 50
    assert report["metrics"]["failures"] == 1
    assert report["train_localization_failures"] == ["train-failure"]
    predictions = [json.loads(x) for x in (tmp_path / "out" / "predictions.jsonl").read_text().splitlines()]
    assert predictions[1]["prediction"] is None


@pytest.mark.parametrize("provenance,allow_synthetic", [("synthetic", False), ("measured", True)])
def test_synthetic_requires_opt_in_and_cannot_mix_with_measured(tmp_path, provenance, allow_synthetic):
    train_file, test_file = save_inputs(tmp_path, [row("train", provenance="synthetic")],
                                       [row("test", provenance=provenance)])
    with pytest.raises(ValueError, match="Synthetic|cannot be mixed"):
        runner.run(CONFIG, train_file, test_file, tmp_path / "out", "MAA", allow_synthetic=allow_synthetic)


def test_mismatched_semantic_domain_rejected_even_if_observation_would_fail(tmp_path):
    labels = dict(domain="legal", intent="review", attribute="employment", keyword="contract")
    train_file, test_file = save_inputs(tmp_path, [row("train", labels=labels)], [row("test", labels=labels)])
    with pytest.raises(ValueError, match="disagrees"):
        runner.run(CONFIG, train_file, test_file, tmp_path / "out", "MSA")


def test_matrix_uses_config_and_profiles_both_lengths():
    config = copy.deepcopy(load_config(CONFIG))
    config["cache"]["block_tokens"] = 32
    config["cache"]["residency_fraction"] = 0.4
    profiling = list(matrix(config, "profiling"))
    assert {r["setting"] for r in profiling} == {"templated_short", "templated_long"}
    assert len(profiling) == 4 * 4 * 2
    assert all(r["block_tokens"] == 32 and r["residency_fraction"] == 0.4 for r in profiling)
    residency = [r for r in matrix(config, "sensitivity") if r["factor"] == "residency_fraction"]
    assert all(r["context_tokens"] == 1000 and r["setting"] == "templated_short" for r in residency)
