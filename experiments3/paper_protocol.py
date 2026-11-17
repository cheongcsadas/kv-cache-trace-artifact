"""Executable parameter and dataset contracts for Sections 5.1--5.6.

The paper constants are distinct from run-specific backend settings and the
configurable representation choices. Sweep permissions must be explicit.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
from datasets.preprocessing.labels import validate_label_vocabulary
from datasets.preprocessing.sources import source_ids


DOMAINS = ("medical", "financial", "legal", "code")
MODEL_FAMILIES = {
    "quest": "Llama-2-7B",
    "infinigen": "Mistral-7B",
    "retrievalattention": "Qwen2.5-7B-Instruct",
}
SWEEP_LEVELS = {
    "residency_fraction": (0.2, 0.4, 0.6),
    "block_tokens": (8, 16, 32),
    "context_tokens": (1000, 2000, 3000, 4000),
    "concurrent_workloads": (1, 4, 8, 16),
    "training_per_domain": (400, 800, 1600, 3200),
}
SEMANTIC_CLASSES = {"intent": 3, "attribute": 60, "keyword": 60}
PAPER_FIXED = {
    "data.domains": list(DOMAINS),
    "data.test_per_domain": 1000,
    "data.train_fraction": 0.75,
    "data.short_mean_tokens": 1000,
    "data.long_mean_tokens": 3000,
    "data.settings": ["templated_short", "templated_long", "unstructured_short", "unstructured_long", "cross_template"],
    "data.intent_classes_per_domain": 3,
    "data.attribute_classes_per_domain": 60,
    "data.keyword_classes_per_domain": 60,
    "observation.window_ms": 10,
    "observation.max_windows": 512,
    "observation.h2d_bytes": 1024,
    "observation.probe_interval_ms": 1,
    "observation.reference_repetitions": 3,
    "observation.h2d_threshold_us": 200,
    "observation.consecutive_samples": 3,
    "observation.compute_shape": [256, 256],
    "observation.compute_dtype": "float16",
    "observation.probe_pages": 64,
    "observation.probe_page_bytes": 2097152,
    "observation.idle_gap_ms": 20,
    "observation.empirical_fpr": 0.05,
    "observation.submission_error_max_ms": 1,
    "classifier.n_estimators": 100,
    "classifier.max_depth": 10,
    "defenses.fixed_budget.load_pages": 65,
    "defenses.fixed_budget.block_tokens": 16,
    "defenses.dynamic_timing.max_delay_ms": 5,
    "evaluation.reference_merge_gap_ms": 50,
    "evaluation.localization_failures_in_denominator": True,
    "cache.initial_state": "empty",
}
PAPER_ENVIRONMENT = {
    "os": "Linux", "gpu": "NVIDIA GeForce RTX 4090", "gpu_memory_gb": 24,
    "cpu": "Intel Core i9-14900K", "cpu_cores": 24,
    "driver": "595.71.05", "cuda_toolkit": "12.4.1",
}


def _get(config, path):
    current = config
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"Required paper parameter is absent: {path}")
        current = current[key]
    return current


def _equal(actual, expected):
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)):
        return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and actual == expected)
    if isinstance(expected, list):
        return (isinstance(actual, (list, tuple)) and len(actual) == len(expected)
                and all(_equal(a, e) for a, e in zip(actual, expected)))
    return actual == expected


def _require(config, path, expected):
    actual = _get(config, path)
    if not _equal(actual, expected):
        raise ValueError(f"Paper parameter {path} must be {expected!r}; received {actual!r}")


def _sweeps(allowed_sweep):
    allowed = set() if allowed_sweep is None else (
        {allowed_sweep} if isinstance(allowed_sweep, str) else set(allowed_sweep))
    if allowed - set(SWEEP_LEVELS):
        raise ValueError(f"Unknown paper sweep: {sorted(allowed-set(SWEEP_LEVELS))}")
    return allowed


def validate_paper_config(config: Mapping, allowed_sweep=None) -> dict:
    """Validate paper defaults; allow only explicitly selected published sweeps.

    ``allowed_sweep`` is a name or iterable of names from ``SWEEP_LEVELS``.
    A context/concurrency setting can be supplied in ``experiment`` as
    ``context_tokens``/``concurrent_workloads``. Model checkpoint revisions,
    dtype, generation length, capacity and seed remain run-specific settings.
    Return a compact validated parameter mapping for the caller's run manifest.
    """
    if not isinstance(config, Mapping):
        raise ValueError("Configuration must be a mapping")
    allowed = _sweeps(allowed_sweep)
    system = config.get("system")
    if system not in MODEL_FAMILIES:
        raise ValueError("Unknown system/model pairing")
    model = _get(config, "model")
    family = model.get("family")
    if family is not None and family != MODEL_FAMILIES[system]:
        raise ValueError("Model family disagrees with the paper system/model pairing")
    identifier = str(model.get("id", ""))
    if not identifier:
        raise ValueError("Model identifier is required")
    normalized = identifier.lower().replace("_", "-")
    expected_name = MODEL_FAMILIES[system].lower()
    if expected_name not in normalized:
        raise ValueError("Model identifier disagrees with the paper system/model pairing")
    for path, expected in PAPER_FIXED.items():
        _require(config, path, expected)
    label_vocabulary=config.get("data",{}).get("label_vocabulary")
    if label_vocabulary is not None:
        label_vocabulary=validate_label_vocabulary(label_vocabulary)
    for factor, default, path in (
        ("block_tokens", 16, "cache.block_tokens"),
        ("residency_fraction", .2, "cache.residency_fraction"),
        ("training_per_domain", 3000, "data.train_per_domain"),
    ):
        value = _get(config, path)
        if factor in allowed:
            values = SWEEP_LEVELS[factor] + ((default,) if default not in SWEEP_LEVELS[factor] else ())
            if not any(_equal(value, level) for level in values):
                raise ValueError(f"{path} must be a published sweep level: {values}")
        else:
            _require(config, path, default)
    for factor, levels in SWEEP_LEVELS.items():
        _require(config, f"sweeps.{factor}", list(levels))
    experiment = config.get("experiment", {})
    for factor, default in (("context_tokens", 1000), ("concurrent_workloads", 1)):
        if factor in experiment:
            value = experiment[factor]
            values = SWEEP_LEVELS[factor] if factor in allowed else (default,)
            if not any(_equal(value, level) for level in values):
                raise ValueError(f"experiment.{factor} must be one of {values}")
    slots = _get(config, "cache.capacity_slots")
    if slots is not None and (isinstance(slots, bool) or not isinstance(slots, int) or slots < 1):
        raise ValueError("cache.capacity_slots must be a positive integer when set")
    selected = _get(config, "cache.selected_token_budget")
    if selected is not None and (isinstance(selected, bool) or not isinstance(selected, int) or selected < 1):
        raise ValueError("cache.selected_token_budget must be a positive integer when set")
    tolerance = _get(config, "evaluation.metadata_time_tolerance_ms")
    if tolerance is not None and (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                                  or not math.isfinite(tolerance) or tolerance < 0):
        raise ValueError("metadata timing tolerance must be finite and nonnegative")
    return dict(system=system, model_family=MODEL_FAMILIES[system], allowed_sweep=sorted(allowed),
                block_tokens=config["cache"]["block_tokens"],
                residency_fraction=config["cache"]["residency_fraction"],
                train_per_domain=config["data"]["train_per_domain"],
                test_per_domain=config["data"]["test_per_domain"],
                observation_horizon_ms=5120,
                semantic_vocabulary_sizes={} if label_vocabulary is None else
                    {domain:{slot:len(labels) for slot,labels in slots.items()}
                     for domain,slots in label_vocabulary.items()})


def validate_calibration(calibration: Mapping, config: Mapping) -> None:
    """Freeze the paper detector settings while retaining measured thresholds."""
    if hasattr(calibration, "to_dict"):
        calibration = calibration.to_dict()
    if not isinstance(calibration, Mapping):
        raise ValueError("Calibration must be a mapping or expose to_dict()")
    mapping = {
        "h2d_threshold_us": "h2d_threshold_us", "consecutive_samples": "consecutive_samples",
        "sample_interval_ms": "probe_interval_ms", "idle_gap_ms": "idle_gap_ms",
    }
    observation = config["observation"]
    for calibration_key, config_key in mapping.items():
        value = calibration.get(calibration_key, observation[config_key])
        if not _equal(value, observation[config_key]):
            raise ValueError(f"Calibration {calibration_key} disagrees with configuration")
    error = calibration.get("submission_error_ms", observation["submission_error_max_ms"])
    if (isinstance(error, bool) or not isinstance(error, (int, float)) or not math.isfinite(error)
            or not 0 <= error <= observation["submission_error_max_ms"]):
        raise ValueError("Submission timing uncertainty must be between 0 and 1 ms")
    for name in ("baseline_h2d_us", "compute_threshold_us", "page_threshold_us"):
        value = calibration.get(name)
        if value is None and name != "baseline_h2d_us":
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"Calibration {name} must be a positive finite measured latency")


def validate_hardware_observation(observation: Mapping, config: Mapping) -> None:
    """Match measured probe settings and KV capacity to the selected run.

    The acquisition receipt is optional for imported observations. When present,
    all six probe dimensions/cadence fields must match the run configuration.
    Page stride and kernel launch geometry remain representation choices.
    """
    if not isinstance(observation, Mapping):
        raise ValueError("Hardware observation must be a mapping")
    slots = observation.get("capacity_slots")
    if isinstance(slots, bool) or not isinstance(slots, int) or slots < 1:
        raise ValueError("Hardware observation capacity_slots must be a positive integer")
    configured = config["cache"].get("capacity_slots")
    if configured is not None and slots != configured:
        raise ValueError("Observed capacity_slots disagrees with configured GPU KV-pool capacity")
    if "acquisition_parameters" not in observation:
        return
    receipt = observation["acquisition_parameters"]
    if not isinstance(receipt, Mapping):
        raise ValueError("Hardware acquisition_parameters must be a mapping")
    for field in ("h2d_bytes", "probe_interval_ms", "compute_shape", "compute_dtype",
                  "probe_pages", "probe_page_bytes"):
        if field not in receipt:
            raise ValueError(f"Acquisition receipt lacks required probe parameter {field}")
        if not _equal(receipt[field], config["observation"][field]):
            raise ValueError(f"Measured probe parameter {field} disagrees with run configuration")


def _count_argument(value, name):
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError(f"{name} must be a positive integer or None")


def validate_dataset_protocol(
    train: Sequence[Mapping], test: Sequence[Mapping], *, train_per_domain=3000,
    test_per_domain=1000, test_total=None, semantic=False, cross_template=False,
    domains=DOMAINS, require_measured=True, label_vocabularies=None,
    semantic_class_counts=None,
) -> dict:
    """Validate attempted records before any localization or label filtering.

    Quotas count all attempts, including failed observations. ``test_total`` is
    for experiments declaring a total rather than a per-domain test budget; set
    ``test_per_domain=None`` with it. None quotas allow an explicitly selected
    subset, while source/sample isolation is always enforced. Semantic vocabularies
    can be supplied as ``{domain: {slot: [labels]}}``; otherwise the train/test
    label union supplies the vocabulary. Each test label must occur in training.
    """
    train, test = list(train), list(test)
    if not train or not test:
        raise ValueError("Training and test partitions must be nonempty")
    domains = tuple(domains)
    if not domains or len(set(domains)) != len(domains) or set(domains)-set(DOMAINS):
        raise ValueError("Expected distinct paper domain labels")
    for value, name in ((train_per_domain, "train_per_domain"), (test_per_domain, "test_per_domain"),
                        (test_total, "test_total")):
        _count_argument(value, name)
    if test_total is not None and test_per_domain is not None:
        raise ValueError("Choose total or per-domain test count, not both")
    counts = {}
    ids, sources = [], []
    for partition, rows, quota in (("train", train, train_per_domain), ("test", test, test_per_domain)):
        partition_ids, partition_sources = set(), set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("Dataset rows must be mappings")
            for key in ("sample_id", "source_id"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Nonempty {key} required for source isolation")
            if row["sample_id"] in partition_ids:
                raise ValueError(f"Duplicate sample_id in {partition}")
            partition_ids.add(row["sample_id"])
            partition_sources.update(source_ids(row))
            if row.get("domain") not in domains:
                raise ValueError("Unexpected domain in experiment dataset")
            if require_measured and row.get("provenance") != "measured":
                raise ValueError("Paper experiment requires measured observations")
        ids.append(partition_ids)
        sources.append(partition_sources)
        tally = Counter(row["domain"] for row in rows)
        if set(tally) != set(domains):
            raise ValueError(f"Every selected domain must be represented in {partition}")
        if quota is not None and any(tally[d] != quota for d in domains):
            raise ValueError(f"{partition} requires exactly {quota} attempts per domain; received {dict(tally)}")
        counts[partition] = dict(tally)
    if ids[0] & ids[1]:
        raise ValueError("Train/test sample IDs overlap")
    if sources[0] & sources[1]:
        raise ValueError("Train/test source records overlap")
    if test_total is not None and len(test) != test_total:
        raise ValueError(f"Test set requires exactly {test_total} attempts")
    if cross_template:
        template_sets = []
        for rows in (train, test):
            if any(not isinstance(r.get("template_id"), str) or not r["template_id"] for r in rows):
                raise ValueError("Cross-template evaluation requires template IDs")
            template_sets.append({r["template_id"] for r in rows})
        if template_sets[0] & template_sets[1]:
            raise ValueError("Cross-template test templates overlap training")
    cardinalities = {}
    if semantic:
        if label_vocabularies is not None:
            label_vocabularies=validate_label_vocabulary(label_vocabularies, class_counts=semantic_class_counts)
        expected_counts = dict(SEMANTIC_CLASSES if semantic_class_counts is None else semantic_class_counts)
        if set(expected_counts) != set(SEMANTIC_CLASSES):
            raise ValueError("Semantic class counts require intent, attribute and keyword")
        for value in expected_counts.values():
            _count_argument(value, "semantic class count")
            if value is None:
                raise ValueError("Semantic class counts must be positive integers")
        for domain in domains:
            train_domain = [r for r in train if r["domain"] == domain]
            test_domain = [r for r in test if r["domain"] == domain]
            for row in train_domain + test_domain:
                labels = row.get("labels")
                if not isinstance(labels, Mapping) or labels.get("domain") != domain:
                    raise ValueError("Semantic domain label must agree with row domain")
                if any(not isinstance(labels.get(s), str) or not labels[s].strip() for s in SEMANTIC_CLASSES):
                    raise ValueError("Each semantic slot must be one nonempty string label")
            cardinalities[domain] = {}
            for slot, expected in expected_counts.items():
                if label_vocabularies is not None and semantic_class_counts is None:
                    expected=len(label_vocabularies[domain][slot])
                training_labels = {r["labels"][slot] for r in train_domain}
                testing_labels = {r["labels"][slot] for r in test_domain}
                if testing_labels-training_labels:
                    raise ValueError(f"Test {domain}/{slot} labels are outside the training label set")
                if label_vocabularies is None:
                    vocabulary = training_labels | testing_labels
                else:
                    raw = label_vocabularies.get(domain, {}).get(slot)
                    if not isinstance(raw, (list, tuple)) or any(not isinstance(v, str) or not v for v in raw):
                        raise ValueError("Semantic vocabulary must list nonempty labels per domain and slot")
                    vocabulary = set(raw)
                    if len(vocabulary) != len(raw):
                        raise ValueError("Duplicate label in semantic vocabulary")
                    if (training_labels | testing_labels)-vocabulary:
                        raise ValueError("Dataset label is outside the declared semantic vocabulary")
                    if vocabulary-training_labels:
                        raise ValueError("Training data must cover the declared semantic vocabulary")
                if len(vocabulary) != expected:
                    raise ValueError(f"{domain}/{slot} requires {expected} labels; received {len(vocabulary)}")
                cardinalities[domain][slot] = len(vocabulary)
    return dict(counts=counts, train_total=len(train), test_total=len(test),
                source_disjoint=True, semantic_vocabulary_sizes=cardinalities)


def validate_fixed_selected_budget(configs: Sequence[Mapping]) -> int:
    """Enforce the fixed selected-token budget in the block-size sweep (Table 6)."""
    budgets = [config.get("cache", {}).get("selected_token_budget") for config in configs]
    if not budgets or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in budgets):
        raise ValueError("Block-size sweep requires a positive selected-token budget in every condition")
    if len(set(budgets)) != 1:
        raise ValueError("Selected-token budget must be held fixed when block size changes")
    return budgets[0]
