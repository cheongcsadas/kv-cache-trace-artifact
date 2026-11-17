"""Shared measured-data contracts and result writers for Chapter 5."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

DOMAINS = ("medical", "financial", "legal", "code")
SYSTEMS = ("quest", "infinigen", "retrievalattention")


def resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else Path(base)/path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def observation(row, data_path):
    if "observation" in row:
        return row["observation"]
    return read_json(resolve(Path(data_path).parent, row["observation_path"]))


def materialize_paths(rows, data_path):
    """Preserve references when writing a selected auxiliary subset elsewhere."""
    result = []
    for row in rows:
        row = dict(row)
        for field in ("observation_path", "reference_path", "schedule_path", "native_receipt_path"):
            if row.get(field):
                row[field] = str(resolve(Path(data_path).parent, row[field]).resolve())
        result.append(row)
    return result


def write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")


def require_provenance(rows, allow_synthetic=False):
    if not rows:
        raise ValueError("An experiment needs nonempty observations")
    provenance = {r.get("provenance") for r in rows}
    allowed = {"measured", "synthetic"} if allow_synthetic else {"measured"}
    if len(provenance) != 1 or not provenance <= allowed:
        raise ValueError("One explicit provenance is required; measured is the default")
    return next(iter(provenance))


def summarize_counts(tp, fp, fn):
    return dict(tp=tp, fp=fp, fn=fn,
                precision_percent=100*tp/(tp+fp) if tp+fp else 0.0,
                recall_percent=100*tp/(tp+fn) if tp+fn else 0.0,
                trr_percent=200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)


def nested_subset(rows, per_domain, seed, semantic=False):
    """Deterministic nested subsets, retaining the training pool's semantic labels."""
    if isinstance(per_domain, bool) or not isinstance(per_domain, int) or per_domain <= 0:
        raise ValueError("Training count must be a positive integer")
    result = []
    for domain in DOMAINS:
        pool = [r for r in rows if r["domain"] == domain]
        if len(pool) < per_domain:
            raise ValueError(f"Insufficient expanded auxiliary pool for {domain}: {len(pool)} < {per_domain}")
        def rank(row):
            return hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).digest()
        pool.sort(key=rank)
        chosen = []
        if semantic:
            uncovered = {(slot, r["labels"][slot]) for r in pool for slot in ("intent", "attribute", "keyword")}
            while uncovered:
                best = max(pool, key=lambda r: sum((s, r["labels"][s]) in uncovered for s in ("intent", "attribute", "keyword")))
                chosen.append(best)
                pool.remove(best)
                uncovered -= {(s, best["labels"][s]) for s in ("intent", "attribute", "keyword")}
            if len(chosen) > per_domain:
                raise ValueError("Training size cannot cover the semantic vocabulary")
        result.extend(chosen + pool[:per_domain-len(chosen)])
    return result


def fingerprint(rows):
    payload = sorted(rows,key=lambda r:r["sample_id"])
    return hashlib.sha256(json.dumps(payload, sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def write_summary(output, experiment, records, protocol):
    result = dict(experiment=experiment, protocol=protocol, runs=records)
    write_json(output/"summary.json", result)
    flat = []
    for record in records:
        row = {}
        def visit(prefix, value):
            if isinstance(value, dict):
                for key, item in value.items(): visit(f"{prefix}.{key}" if prefix else key, item)
            elif value is None or isinstance(value, (str, float, int, bool)):
                row[prefix] = value
        visit("", record)
        flat.append(row)
    fields = sorted(set().union(*(row.keys() for row in flat)))
    with (output/"summary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    return result


def validate_collection(run, config, rows):
    """Check a recorded collection condition against the supplied run config."""
    condition = run.get("collection", {})
    if not condition:
        raise ValueError("Each run needs a recorded collection condition")
    for key in ("block_tokens", "residency_fraction"):
        if condition.get(key) != config["cache"][key]:
            raise ValueError(f"Collected {key} does not match the run configuration")
    if condition.get("system") != config["system"]:
        raise ValueError("Collected system differs from run configuration")
    if condition.get("setting") != run.get("setting"):
        raise ValueError("Collected prompt setting differs from run setting")
    for row in rows:
        if "collection" in row and row["collection"] != condition:
            raise ValueError("Mixed collection conditions in one dataset")
        if row.get("system", config["system"]) != config["system"]:
            raise ValueError("Observation system differs from collection system")
    return condition


def finite_positive(value, field):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return float(value)
