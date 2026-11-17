"""Metrics from Section 5.1.3, computed from local saved records.

All ``*_percent`` results use the 0--100 scale. ``accuracy`` and ``macro_f1``
use 0--1. Missing predictions remain in denominators. Matching is deliberately
chronological rather than an assignment that optimizes the reported F1 score.
This module only evaluates supplied records; it does not collect observations.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any


SLOTS = ("domain", "intent", "attribute", "keyword")
OPERATIONS = {"HIT", "LOAD", "EVICT"}


def _paired(truth: Sequence, predictions: Sequence) -> tuple[list, list]:
    truth, predictions = list(truth), list(predictions)
    if not truth:
        raise ValueError("Evaluation needs at least one reference record")
    if len(truth) != len(predictions):
        raise ValueError("Every reference record needs a prediction or an explicit None")
    return truth, predictions


def _label(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Labels must be nonempty strings; use None for a failed prediction")
    return value


def attribute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str | None],
    labels: Sequence[str] | None = None,
) -> dict:
    """Domain accuracy (AI-ASR), macro-F1, and a confusion table.

``None`` denotes a failed attempt. The final confusion column represents these
failures; they count as false negatives for the true class. Supply all four
domain labels explicitly when comparing a subset with the four-domain setting.
Unlisted classes are rejected instead of silently dropped from the table.
"""
    truth, predictions = _paired(y_true, y_pred)
    truth = [_label(v) for v in truth]
    predictions = [None if v is None else _label(v) for v in predictions]
    if labels is None:
        classes = sorted(set(truth) | {p for p in predictions if p is not None})
    else:
        classes = [_label(v) for v in labels]
        if not classes or len(set(classes)) != len(classes):
            raise ValueError("labels must be a nonempty sequence of distinct classes")
    if set(truth) - set(classes) or {p for p in predictions if p is not None} - set(classes):
        raise ValueError("A reference or predicted class is absent from labels")
    index = {label: i for i, label in enumerate(classes)}
    matrix = [[0] * (len(classes) + 1) for _ in classes]
    for target, predicted in zip(truth, predictions):
        matrix[index[target]][len(classes) if predicted is None else index[predicted]] += 1
    per_domain = {}
    f1_scores = []
    for i, label in enumerate(classes):
        support = sum(matrix[i])
        tp = matrix[i][i]
        fp = sum(row[i] for j, row in enumerate(matrix) if j != i)
        fn = support - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        f1_scores.append(f1)
        per_domain[label] = {
            "n_test": support,
            "correct": tp,
            "failures": matrix[i][-1],
            "ai_asr_percent": 100 * tp / support if support else None,
            "f1": f1,
        }
    correct = sum(matrix[i][i] for i in range(len(classes)))
    return {
        "n_test": len(truth),
        "correct": correct,
        "failures": predictions.count(None),
        "accuracy": correct / len(truth),
        "ai_asr_percent": 100 * correct / len(truth),
        "macro_f1": fmean(f1_scores),
        "per_domain": per_domain,
        "confusion_labels": classes,
        "confusion_prediction_labels": classes + [None],
        "confusion_matrix": matrix,
    }


def _semantic_record(record: Mapping | None, *, reference: bool) -> dict:
    if record is None and not reference:
        return dict.fromkeys(SLOTS)
    if not isinstance(record, Mapping):
        raise ValueError("Semantic records must be mappings with domain, intent, attribute, keyword")
    result = {}
    for slot in SLOTS:
        value = record.get(slot)
        if value is None and not reference:
            result[slot] = None
        else:
            result[slot] = _label(value)
    return result


def semantic_metrics(
    truth: Sequence[Mapping[str, str]],
    predictions: Sequence[Mapping[str, str | None] | None],
) -> dict:
    """SR-ASR is mean slot accuracy, not joint semantic success.

Keywords are complete single labels (including multiword labels), never token
sets. Missing slots score zero. A wholly failed attempt must be ``None`` and
scores zero for all slots. Slot label IDs should be consistent across domains.
"""
    targets, predictions = _paired(truth, predictions)
    failed = sum(p is None for p in predictions)
    targets = [_semantic_record(t, reference=True) for t in targets]
    predicted = [_semantic_record(p, reference=False) for p in predictions]
    correct = {
        slot: sum(t[slot] == p[slot] for t, p in zip(targets, predicted))
        for slot in SLOTS
    }
    joint = sum(all(t[s] == p[s] for s in SLOTS) for t, p in zip(targets, predicted))
    n = len(targets)
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, target in enumerate(targets):
        grouped[target["domain"]].append(i)
    per_domain = {}
    for domain, indices in sorted(grouped.items()):
        counts = {
            slot: sum(targets[i][slot] == predicted[i][slot] for i in indices)
            for slot in SLOTS
        }
        per_domain[domain] = {
            "n_test": len(indices),
            "failures": sum(predictions[i] is None for i in indices),
            "slot_accuracy_percent": {s: 100 * counts[s] / len(indices) for s in SLOTS},
            "sr_asr_percent": 100 * sum(counts.values()) / (4 * len(indices)),
            "joint_accuracy_percent": 100 * sum(
                all(targets[i][s] == predicted[i][s] for s in SLOTS) for i in indices
            ) / len(indices),
        }
    return {
        "n_test": n,
        "failures": failed,
        "slot_correct": correct,
        "slot_accuracy_percent": {s: 100 * correct[s] / n for s in SLOTS},
        "sr_asr_percent": 100 * sum(correct.values()) / (4 * n),
        "joint_correct": joint,
        "joint_accuracy_percent": 100 * joint / n,
        "per_domain": per_domain,
        "domain_confusion": attribute_metrics(
            [t["domain"] for t in targets], [p["domain"] for p in predicted]
        ),
    }


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _time(record: Mapping) -> float:
    if "time_ms" in record:
        return _number(record["time_ms"], "time_ms")
    if "timestamp_ms" in record:
        return _number(record["timestamp_ms"], "timestamp_ms")
    raise ValueError("Event requires time_ms; interval-derived evidence is not an exact event")


def _operation(record: Mapping) -> str:
    op = record.get("op", record.get("operation"))
    if op not in OPERATIONS:
        raise ValueError("Event/window op must be HIT, LOAD, or EVICT")
    return op


def _scope_match(a: Mapping, b: Mapping) -> bool:
    # Mixed attributed/unattributed records must not match by accident.
    return a.get("request_id") == b.get("request_id")


def _match_summary(n_predicted: int, n_reference: int, pairs: list[tuple[int, int]]) -> dict:
    tp = len(pairs)
    fp, fn = n_predicted - tp, n_reference - tp
    denominator = 2 * tp + fp + fn
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "n_predicted": n_predicted, "n_reference": n_reference,
        "precision_percent": 100 * tp / n_predicted if n_predicted else 0.0,
        "recall_percent": 100 * tp / n_reference if n_reference else 0.0,
        "trr_percent": 100 * 2 * tp / denominator if denominator else 0.0,
        "evaluable": bool(n_predicted or n_reference),
        "matched_indices": [{"predicted": p, "reference": r} for p, r in pairs],
    }


def metadata_trr(
    predicted: Sequence[Mapping], reference: Sequence[Mapping], *, tolerance_ms: float
) -> dict:
    """One-to-one metadata event matching with an explicit timing tolerance.

The paper does not specify the metadata tolerance numerically. Callers must
provide one chosen before evaluation. Events require ``layer``, ``block_id``,
``op``, and ``time_ms``. Request IDs must agree when present. Snapshot intervals
must not be relabeled as exact events to use this metric.
"""
    tolerance = _number(tolerance_ms, "tolerance_ms")
    if tolerance < 0:
        raise ValueError("tolerance_ms cannot be negative")
    predicted, reference = list(predicted), list(reference)
    for row in predicted + reference:
        if not isinstance(row, Mapping) or "layer" not in row or "block_id" not in row:
            raise ValueError("Metadata event requires layer and block_id")
        _time(row)
        _operation(row)
    p_order = sorted(range(len(predicted)), key=lambda i: (_time(predicted[i]), i))
    r_order = sorted(range(len(reference)), key=lambda i: (_time(reference[i]), i))
    matched = set()
    pairs = []
    for pi in p_order:
        p = predicted[pi]
        for ri in r_order:
            r = reference[ri]
            if ri in matched:
                continue
            if (
                _scope_match(p, r)
                and _operation(p) == _operation(r)
                and p["layer"] == r["layer"]
                and p["block_id"] == r["block_id"]
                and abs(_time(p) - _time(r)) <= tolerance
            ):
                matched.add(ri)
                pairs.append((pi, ri))
                break
    result = _match_summary(len(predicted), len(reference), pairs)
    result.update({"channel": "metadata", "tolerance_ms": tolerance})
    return result


def merge_reference_windows(events: Sequence[Mapping], *, merge_gap_ms: float = 50.0) -> list[dict]:
    """Merge chronologically adjacent same-type reference events within requests.

An intervening different operation prevents a merge. The gap comparison is
strictly ``< merge_gap_ms``; exactly 50 ms does not merge under paper settings.
Layer/block IDs do not constrain anonymous hardware windows. Single events
produce point windows and are retained rather than dropped.
"""
    gap = _number(merge_gap_ms, "merge_gap_ms")
    if gap <= 0:
        raise ValueError("merge_gap_ms must be positive")
    grouped: dict[Any, list[tuple[int, Mapping]]] = defaultdict(list)
    for i, event in enumerate(events):
        _operation(event)
        _time(event)
        grouped[event.get("request_id")].append((i, event))
    windows = []
    for request_id, rows in grouped.items():
        rows.sort(key=lambda item: (_time(item[1]), item[0]))
        current = None
        for original, event in rows:
            time, op = _time(event), _operation(event)
            if current is not None and current["op"] == op and time - current["end_ms"] < gap:
                current["end_ms"] = time
                current["reference_indices"].append(original)
            else:
                current = {
                    "op": op, "start_ms": time, "end_ms": time,
                    "reference_indices": [original],
                }
                if request_id is not None:
                    current["request_id"] = request_id
                windows.append(current)
    return sorted(windows, key=lambda w: (w["start_ms"], w["reference_indices"][0]))


def _interval(window: Mapping) -> tuple[float, float]:
    start = _number(window.get("start_ms"), "start_ms")
    end = _number(window.get("end_ms"), "end_ms")
    if end < start:
        raise ValueError("Window end_ms precedes start_ms")
    _operation(window)
    return start, end


def hardware_trr(
    predicted_windows: Sequence[Mapping], reference_events: Sequence[Mapping], *,
    merge_gap_ms: float = 50.0,
) -> dict:
    """Anonymous operation-window F1 using chronological one-to-one matching.

Reference events are merged by :func:`merge_reference_windows`. Intervals use
closed endpoints for matching, so a singleton reference event can match a
predicted window containing its time. A boundary-only intersection also counts;
this endpoint convention is an implementation choice not specified in the paper.
"""
    predicted = list(predicted_windows)
    reference = merge_reference_windows(reference_events, merge_gap_ms=merge_gap_ms)
    for window in predicted:
        _interval(window)
    order = sorted(range(len(predicted)), key=lambda i: (_interval(predicted[i])[0], i))
    matched = set()
    pairs = []
    for pi in order:
        p = predicted[pi]
        p_start, p_end = _interval(p)
        for ri, r in enumerate(reference):
            if ri in matched:
                continue
            if (
                _scope_match(p, r) and _operation(p) == r["op"]
                and max(p_start, r["start_ms"]) <= min(p_end, r["end_ms"])
            ):
                matched.add(ri)
                pairs.append((pi, ri))
                break
    result = _match_summary(len(predicted), len(reference), pairs)
    result.update({
        "channel": "hardware", "merge_gap_ms": merge_gap_ms,
        "reference_windows": reference, "overlap_convention": "closed_intervals",
    })
    return result


def macro_average(results: Sequence[Mapping], keys: Sequence[str]) -> dict:
    """Equal-weight mean across independently evaluated system configurations.

Every metric must be defined for every configuration; no failed or empty
configuration is silently excluded. This is distinct from pooling prompts.
"""
    results = list(results)
    if not results:
        raise ValueError("At least one configuration is required")
    if any(r.get("evaluable") is False for r in results):
        raise ValueError("Cannot macro-average an empty trace evaluation")
    return {
        key: fmean(_number(row.get(key), key) for row in results)
        for key in keys
    }


def _latencies(values: Mapping[str, float] | Sequence[Mapping]) -> dict[str, float]:
    if isinstance(values, Mapping):
        items = list(values.items())
    else:
        items = [(r["prompt_id"], r["latency_ms"]) for r in values]
    result = {}
    for key, value in items:
        key = _label(key)
        if key in result:
            raise ValueError("Duplicate latency ID; use a unique prompt/run ID per measurement")
        latency = _number(value, "latency_ms")
        if latency <= 0:
            raise ValueError("Latency must be strictly positive")
        result[key] = latency
    return result


def latency_overhead(
    baseline: Mapping[str, float] | Sequence[Mapping],
    defended: Mapping[str, float] | Sequence[Mapping],
) -> dict:
    """Latency overhead on exactly paired prompt/run IDs, using measured values.

Mappings contain ID -> milliseconds. Rows contain ``prompt_id`` and
``latency_ms``. Report both ratio-of-means and mean paired relative overhead;
they answer different questions and are not interchangeable.
"""
    base, defense = _latencies(baseline), _latencies(defended)
    if not base or set(base) != set(defense):
        raise ValueError("Baseline and defended measurements need the same nonempty ID set")
    mean_base, mean_defense = fmean(base.values()), fmean(defense.values())
    paired = {key: 100 * (defense[key] / base[key] - 1) for key in sorted(base)}
    return {
        "n_pairs": len(base),
        "baseline_mean_ms": mean_base,
        "defended_mean_ms": mean_defense,
        "overhead_percent": 100 * (mean_defense / mean_base - 1),
        "mean_paired_overhead_percent": fmean(paired.values()),
        "paired_overhead_percent": paired,
    }
