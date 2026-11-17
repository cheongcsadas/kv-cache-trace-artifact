"""Paper parameter contracts, quotas and source/label isolation."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from experiments.paper_protocol import (
    DOMAINS, PAPER_FIXED, SWEEP_LEVELS, validate_calibration,
    validate_dataset_protocol, validate_fixed_selected_budget, validate_paper_config, validate_hardware_observation,
)

ROOT = Path(__file__).resolve().parents[1]


def config(system="quest"):
    return yaml.safe_load((ROOT/"configs"/f"{system}.yaml").read_text(encoding="utf-8"))


def rows(partition, count=2, semantic=False):
    result = []
    for domain in DOMAINS:
        for i in range(count):
            row = dict(sample_id=f"{partition}-{domain}-{i}", source_id=f"source-{partition}-{domain}-{i}",
                       domain=domain, provenance="measured", template_id=f"template-{partition}")
            if semantic:
                row["labels"] = dict(domain=domain, intent=f"{domain}-i{i%3}",
                                     attribute=f"{domain}-a{i%60}", keyword=f"{domain}-k{i%60}")
            result.append(row)
    return result


@pytest.mark.parametrize("system", ["quest", "infinigen", "retrievalattention"])
def test_default_system_configs_match_paper(system):
    result = validate_paper_config(config(system))
    assert result["system"] == system
    assert result["observation_horizon_ms"] == 5120


@pytest.mark.parametrize("path, value", [
    ("classifier.n_estimators", 99), ("classifier.max_depth", 11),
    ("observation.window_ms", 11), ("observation.max_windows", 256),
    ("observation.h2d_bytes", 1000), ("observation.probe_interval_ms", 2),
    ("observation.reference_repetitions", 2), ("observation.h2d_threshold_us", 199),
    ("observation.consecutive_samples", 2), ("observation.compute_shape", [128, 128]),
    ("observation.compute_dtype", "float32"), ("observation.probe_pages", 32),
    ("observation.probe_page_bytes", 2_000_000), ("observation.idle_gap_ms", 21),
    ("observation.empirical_fpr", .06), ("observation.submission_error_max_ms", 2),
    ("data.train_fraction", .7), ("data.test_per_domain", 500),
    ("data.intent_classes_per_domain", 4), ("data.attribute_classes_per_domain", 59),
    ("data.keyword_classes_per_domain", 61), ("data.short_mean_tokens", 1024),
    ("data.long_mean_tokens", 3584), ("defenses.fixed_budget.load_pages", 64),
    ("defenses.dynamic_timing.max_delay_ms", 10), ("evaluation.reference_merge_gap_ms", 49),
    ("evaluation.localization_failures_in_denominator", False),
    ("cache.initial_state", "warm"),
])
def test_changed_paper_constants_are_rejected(path, value):
    c = config()
    current = c
    keys = path.split(".")
    for key in keys[:-1]: current = current[key]
    current[keys[-1]] = value
    with pytest.raises(ValueError, match="Paper parameter"):
        validate_paper_config(c)


def test_bool_not_accepted_as_numeric_and_nan_rejected():
    c = config()
    c["observation"]["probe_interval_ms"] = True
    with pytest.raises(ValueError): validate_paper_config(c)
    c = config()
    c["classifier"]["max_depth"] = float("nan")
    with pytest.raises(ValueError): validate_paper_config(c)


@pytest.mark.parametrize("factor, path, level", [
    ("block_tokens", ("cache", "block_tokens"), 8),
    ("residency_fraction", ("cache", "residency_fraction"), .6),
    ("training_per_domain", ("data", "train_per_domain"), 3200),
])
def test_sweeps_require_explicit_permission(factor, path, level):
    c = config()
    c[path[0]][path[1]] = level
    with pytest.raises(ValueError): validate_paper_config(c)
    assert factor in validate_paper_config(c, factor)["allowed_sweep"]
    c[path[0]][path[1]] = 17
    with pytest.raises(ValueError): validate_paper_config(c, factor)


def test_context_and_async_concurrency_paper_levels():
    c = config()
    c["experiment"] = dict(context_tokens=4000, concurrent_workloads=16)
    with pytest.raises(ValueError): validate_paper_config(c)
    validate_paper_config(c, ["context_tokens", "concurrent_workloads"])
    c["experiment"]["concurrent_workloads"] = 2
    with pytest.raises(ValueError): validate_paper_config(c, ["context_tokens", "concurrent_workloads"])


def test_wrong_model_even_with_family_cannot_pass():
    c = config("infinigen")
    c["model"]["family"] = "Mistral-7B"
    c["model"]["id"] = "meta-llama/Llama-2-7b-hf"
    with pytest.raises(ValueError): validate_paper_config(c)


def test_run_specific_values_are_validated_without_inventing_them():
    c = config()
    assert c["cache"]["capacity_slots"] is None
    validate_paper_config(c)
    c["cache"]["capacity_slots"] = 13
    c["cache"]["selected_token_budget"] = 128
    c["evaluation"]["metadata_time_tolerance_ms"] = .5
    validate_paper_config(c)
    c["cache"]["capacity_slots"] = 0
    with pytest.raises(ValueError): validate_paper_config(c)


def test_selected_token_budget_held_fixed():
    a, b = config(), config()
    with pytest.raises(ValueError): validate_fixed_selected_budget([a, b])
    a["cache"]["selected_token_budget"] = b["cache"]["selected_token_budget"] = 128
    b["cache"]["block_tokens"] = 32
    assert validate_fixed_selected_budget([a, b]) == 128
    b["cache"]["selected_token_budget"] = 256
    with pytest.raises(ValueError): validate_fixed_selected_budget([a, b])


def test_baseline_is_measured_not_forced_to_150():
    c = config()
    calibration = dict(baseline_h2d_us=45.6, compute_threshold_us=30, page_threshold_us=80)
    validate_calibration(calibration, c)
    calibration["h2d_threshold_us"] = 100
    with pytest.raises(ValueError): validate_calibration(calibration, c)


def test_submission_error_is_bounded_by_one_ms():
    c = config()
    calibration = dict(baseline_h2d_us=45.6, compute_threshold_us=30, page_threshold_us=80,
                       submission_error_ms=.5)
    validate_calibration(calibration, c)
    calibration["submission_error_ms"] = 1.01
    with pytest.raises(ValueError): validate_calibration(calibration, c)


def receipt(c):
    keys = ("h2d_bytes", "probe_interval_ms", "compute_shape", "compute_dtype", "probe_pages", "probe_page_bytes")
    return {k: deepcopy(c["observation"][k]) for k in keys}


def test_imported_observation_without_receipt_still_validates_capacity():
    c = config()
    validate_hardware_observation(dict(capacity_slots=13), c)
    c["cache"]["capacity_slots"] = 13
    validate_hardware_observation(dict(capacity_slots=13), c)
    with pytest.raises(ValueError, match="disagrees"):
        validate_hardware_observation(dict(capacity_slots=14), c)
    for value in (None, 0, -1, True, 13.5):
        with pytest.raises(ValueError, match="positive integer"):
            validate_hardware_observation(dict(capacity_slots=value), c)


@pytest.mark.parametrize("field,value", [
    ("h2d_bytes", 2048), ("probe_interval_ms", 2), ("compute_shape", [128,128]),
    ("compute_dtype", "float32"), ("probe_pages", 32), ("probe_page_bytes", 1048576),
])
def test_measurement_receipt_cannot_disagree_with_run_config(field,value):
    c = config()
    parameters = receipt(c)
    validate_hardware_observation(dict(capacity_slots=13,acquisition_parameters=parameters), c)
    parameters[field] = value
    with pytest.raises(ValueError, match="Measured probe parameter"):
        validate_hardware_observation(dict(capacity_slots=13,acquisition_parameters=parameters), c)


def test_complete_receipt_required_but_stride_remains_configurable():
    c = config()
    parameters = receipt(c)
    parameters["page_stride_bytes"] = 8192
    validate_hardware_observation(dict(capacity_slots=13,acquisition_parameters=parameters), c)
    del parameters["compute_dtype"]
    with pytest.raises(ValueError, match="lacks"):
        validate_hardware_observation(dict(capacity_slots=13,acquisition_parameters=parameters), c)


def test_dataset_quotas_and_failed_attempts_stay_in_counts():
    train, test = rows("train"), rows("test")
    test[0]["observation"] = {"h2d": []}
    result = validate_dataset_protocol(train, test, train_per_domain=2, test_per_domain=2)
    assert result["test_total"] == 8
    with pytest.raises(ValueError):
        validate_dataset_protocol(train, test[:-1], train_per_domain=2, test_per_domain=2)


def test_paper_default_3000_1000_and_total_test_override():
    with pytest.raises(ValueError, match="3000"):
        validate_dataset_protocol(rows("train"), rows("test"))
    result = validate_dataset_protocol(rows("train"), rows("test"), train_per_domain=2,
                                       test_per_domain=None, test_total=8)
    assert result["test_total"] == 8
    with pytest.raises(ValueError):
        validate_dataset_protocol(rows("train"), rows("test"), test_total=1000)


@pytest.mark.parametrize("key", ["sample_id", "source_id"])
def test_overlap_is_rejected_even_with_count_checks_disabled(key):
    train, test = rows("train"), rows("test")
    test[0][key] = train[0][key]
    with pytest.raises(ValueError, match="overlap"):
        validate_dataset_protocol(train, test, train_per_domain=None, test_per_domain=None)


def test_measured_provenance_and_template_holdout():
    train, test = rows("train"), rows("test")
    validate_dataset_protocol(train, test, train_per_domain=2, test_per_domain=2, cross_template=True)
    test[0]["template_id"] = train[0]["template_id"]
    with pytest.raises(ValueError, match="template"):
        validate_dataset_protocol(train, test, train_per_domain=2, test_per_domain=2, cross_template=True)
    test[0]["provenance"] = "synthetic"
    with pytest.raises(ValueError, match="measured"):
        validate_dataset_protocol(train, test, train_per_domain=2, test_per_domain=2)


def test_semantic_3_60_60_and_closed_set():
    train, test = rows("train", 60, semantic=True), rows("test", 60, semantic=True)
    result = validate_dataset_protocol(train, test, train_per_domain=60, test_per_domain=60, semantic=True)
    assert result["semantic_vocabulary_sizes"]["medical"] == {"intent": 3, "attribute": 60, "keyword": 60}
    test[0]["labels"]["keyword"] = "unseen keyword"
    with pytest.raises(ValueError, match="outside the training"):
        validate_dataset_protocol(train, test, train_per_domain=60, test_per_domain=60, semantic=True)


def test_explicit_vocabulary_uses_actual_training_classes():
    train, test = rows("train", 3, semantic=True), rows("test", 3, semantic=True)
    vocab = {d: {slot: [f"{d}-{prefix}{i}" for i in range(n)]
                 for slot, prefix, n in [("intent", "i", 3), ("attribute", "a", 60), ("keyword", "k", 60)]}
             for d in DOMAINS}
    with pytest.raises(ValueError,match="cover the declared"):
        validate_dataset_protocol(train,test,train_per_domain=3,test_per_domain=3,
                                  semantic=True,label_vocabularies=vocab)
    for domain in vocab:
        for slot in vocab[domain]: vocab[domain][slot]=vocab[domain][slot][:3]
    result=validate_dataset_protocol(train, test, train_per_domain=3, test_per_domain=3,
                                     semantic=True, label_vocabularies=vocab,
                                     semantic_class_counts=dict(intent=3,attribute=3,keyword=3))
    assert result["semantic_vocabulary_sizes"]["medical"] == dict(intent=3,attribute=3,keyword=3)
    vocab["medical"]["intent"].append("extra")
    with pytest.raises(ValueError, match="3"):
        validate_dataset_protocol(train, test, train_per_domain=3, test_per_domain=3,
                                  semantic=True, label_vocabularies=vocab,
                                  semantic_class_counts=dict(intent=3,attribute=3,keyword=3))
