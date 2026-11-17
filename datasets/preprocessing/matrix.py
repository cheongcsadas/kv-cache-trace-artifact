"""Validate measured, source-separated inputs across the paper's prompt matrix."""
from collections import defaultdict
from collections.abc import Mapping
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path

from .labels import CLASS_LIMITS, DOMAINS, validate_label_vocabulary
from .sources import source_ids

SYSTEMS = ("quest", "infinigen", "retrievalattention")
TRANSFORMERS_VERSIONS = {"quest": "4.45.2", "infinigen": "4.57.1", "retrievalattention": "4.47.1"}
SETTINGS = {"templated_short": 1000, "templated_long": 3000,
            "unstructured_short": 1000, "unstructured_long": 3000,
            "cross_template": 1000}
MODEL_IDENTITIES = {
    "quest": ("meta-llama/Llama-2-7b-hf", "llama", "01c7f73d771dfac7d292323805ebc428287df4f9"),
    "infinigen": ("mistralai/Mistral-7B-v0.1", "mistral", "27d67f1b5f57dc0953326b2601d68371d40ea8da"),
    "retrievalattention": ("Qwen/Qwen2.5-7B-Instruct", "qwen2", "a09a35458c702b33eeacc393d103063234e8bc28"),
}
# SHA256 of tokenizer/configuration assets at the official immutable revisions
# above. These identify the selected 7B checkpoints independently of local names.
IDENTITY_HASHES = {
    "quest": {
        "config.json": "9242e7db1bc2a17873e66084c3b1c6ed10883076e156b338fd6a7775748e2e3c",
        "tokenizer.json": "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e",
        "tokenizer_config.json": "f514e7c3008881b6ba7e6a0cdb44c71ce47dc335920dac143ae7bc788197e53a",
    },
    "infinigen": {
        "config.json": "cf25cdf4719f181d1d1d371973285d9afe9afde0d0c6a6fd48de857555ce1e0d",
        "tokenizer.json": "11c08db21487c885d8c792180f0be237f6a261b89a46f128a6a80a3aa4bd1720",
        "tokenizer_config.json": "ddb008229511e51607002ffe28925001c4a9ca4177dc4de3a655d085cc610b99",
    },
    "retrievalattention": {
        "config.json": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
        "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    },
}
ASSET_NAMES = ("config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
               "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _serializable(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(item) for item in value]
    if hasattr(value, "content"):
        return {key: getattr(value, key) for key in
                ("content", "single_word", "lstrip", "rstrip", "normalized", "special")
                if hasattr(value, key)}
    raise ValueError(f"Unsupported tokenizer metadata value: {type(value).__name__}")


def _assets(tokenizer):
    name = getattr(tokenizer, "name_or_path", "")
    paths = {}
    if isinstance(name, str) and name and Path(name).is_dir():
        paths = {key: Path(name) / key for key in ASSET_NAMES if (Path(name) / key).is_file()}
    elif isinstance(name, str) and "/" in name:
        # Resolve already-cached HF assets without network access or model loads.
        from transformers.utils.hub import cached_file
        options = getattr(tokenizer, "init_kwargs", {})
        for key in ASSET_NAMES:
            found = cached_file(name, key, revision=options.get("_commit_hash", options.get("revision")),
                                local_files_only=True, _raise_exceptions_for_missing_entries=False,
                                _raise_exceptions_for_connection_errors=False)
            if found:
                paths[key] = Path(found)
    result, config = {}, None
    for key, path in paths.items():
        if path.stat().st_size > 50_000_000:
            raise ValueError(f"Unexpectedly large tokenizer asset: {key}")
        data = path.read_bytes()
        result[key] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if key == "config.json":
            config = json.loads(data)
    return result, config


def describe_tokenizer(tokenizer, *, system=None):
    """Fingerprint actual tokenizer behavior and small local assets, never weights.

    Lightweight test tokenizers may have no assets; their unverified identity is
    explicit and they cannot certify a complete real-data matrix.
    """
    if system is not None and system not in SYSTEMS:
        raise ValueError("Unknown tokenizer system")
    assets, config = _assets(tokenizer)
    config_type = config.get("model_type") if config else None
    if system and config is not None and config_type != MODEL_IDENTITIES[system][1]:
        raise ValueError(f"{system}: tokenizer model identity mismatch ({config_type})")
    matches = [key for key, expected in IDENTITY_HASHES.items()
               if all(assets.get(name, {}).get("sha256") == sha for name, sha in expected.items())]
    matched = matches[0] if len(matches) == 1 else None
    if system and matched is not None and matched != system:
        raise ValueError(f"{system}: tokenizer assets belong to {matched}")
    # Reject an explicitly named wrong checkpoint even if its architecture agrees.
    declared = config.get("_name_or_path") if config else None
    if (system and isinstance(declared, str) and "/" in declared and not Path(declared).is_dir()
            and declared != MODEL_IDENTITIES[system][0]):
        raise ValueError(f"{system}: tokenizer model identity mismatch ({declared})")
    versions = {}
    for package in ("transformers", "tokenizers"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        "schema_version": 1, "system": system or matched,
        "class": type(tokenizer).__module__ + "." + type(tokenizer).__name__,
        "versions": versions, "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "vocab_size": getattr(tokenizer, "vocab_size", None), "model_type": config_type,
        "assets": assets, "identity_verified": matched is not None,
        "model_id": MODEL_IDENTITIES[matched][0] if matched else None,
        "revision": MODEL_IDENTITIES[matched][2] if matched else None,
        "special_tokens": _serializable(getattr(tokenizer, "special_tokens_map_extended", {})),
        "special_token_ids": {key: getattr(tokenizer, key, None) for key in
                              ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")},
        "configuration": {key: _serializable(getattr(tokenizer, key, None)) for key in
                          ("add_bos_token", "add_eos_token", "padding_side", "truncation_side",
                           "clean_up_tokenization_spaces", "model_max_length")},
        "encoding": {"add_special_tokens": True, "apply_chat_template": False},
    }
    backend = getattr(tokenizer, "backend_tokenizer", None)
    result["backend_sha256"] = (hashlib.sha256(backend.to_str().encode()).hexdigest()
                                if backend is not None else None)
    result["fingerprint"] = _digest(result)
    return result


def _positive_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _labels(row):
    nested = row.get("labels", {})
    if not isinstance(nested, Mapping):
        raise ValueError("Row labels must be an object")
    result = {}
    for key in ("domain", *CLASS_LIMITS):
        if key in row and key in nested and row[key] != nested[key]:
            raise ValueError(f"Row and nested labels disagree: {key}")
        if key in row or key in nested:
            result[key] = row.get(key, nested.get(key))
    return result


def _row_sources(row):
    values = source_ids(row)
    spans = row.get("source_spans")
    if spans is not None:
        if not isinstance(spans, list) or not spans:
            raise ValueError("Source spans must be a nonempty list")
        span_sources = {span.get("source_id") for span in spans if isinstance(span, Mapping)}
        if len([span for span in spans if isinstance(span, Mapping)]) != len(spans) or span_sources != values:
            raise ValueError("Source spans disagree with primary/supplementary source IDs")
    return values


def summarize_condition(train, test, profiling=None, *, system, setting, vocabulary=None,
                        train_per_domain=3000, test_per_domain=1000, profiling_per_domain=3200,
                        tolerance=.1, domains=DOMAINS):
    """Validate each domain independently using prepared rows' measured counts."""
    if system not in SYSTEMS:
        raise ValueError("Unknown matrix system")
    target = SETTINGS.get(setting)
    if target is None and setting in ("context_1000", "context_2000", "context_3000", "context_4000"):
        target = int(setting.split("_")[1])
    if target is None:
        raise ValueError("Unknown matrix setting")
    domains = tuple(domains)
    if not domains or len(domains) != len(set(domains)) or set(domains) - set(DOMAINS):
        raise ValueError("Domains must be a nonempty unique subset of the four domains")
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or
            not math.isfinite(tolerance) or not 0 <= tolerance < 1):
        raise ValueError("Invalid token mean tolerance")
    quotas = {"train": train_per_domain, "test": test_per_domain, "profiling": profiling_per_domain}
    if any(not _positive_count(value) for value in quotas.values()):
        raise ValueError("Per-domain quotas must be positive integers")
    if profiling is not None and profiling_per_domain < train_per_domain:
        raise ValueError("Profiling must cover the training quota")
    if vocabulary is not None:
        vocabulary = validate_label_vocabulary(vocabulary)
    partitions = {"train": list(train), "test": list(test)}
    if profiling is not None:
        partitions["profiling"] = list(profiling)
    by_domain, identifiers, all_sources, row_signatures = {}, {}, {}, {}
    if any(not isinstance(row, Mapping) for rows in partitions.values() for row in rows):
        raise ValueError("Condition rows must be objects")
    semantic = vocabulary is not None or any(any(slot in _labels(row) for slot in CLASS_LIMITS)
                                           for rows in partitions.values() for row in rows)
    for partition, rows in partitions.items():
        grouped, seen, sources, signatures = defaultdict(list), set(), set(), {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("Condition rows must be objects")
            identity = row.get("id", row.get("sample_id"))
            if not isinstance(identity, str) or not identity or identity in seen:
                raise ValueError(f"{partition}: missing or duplicate record ID")
            seen.add(identity)
            labels = _labels(row)
            domain = labels.get("domain")
            if domain not in domains:
                raise ValueError(f"{partition}: unknown or unexpected domain {domain}")
            count = row.get("token_count")
            if not _positive_count(count) or row.get("context_tokens", count) != count:
                raise ValueError("Measured token_count must be a consistent positive integer")
            if row.get("setting", setting) != setting or row.get("target_tokens", target) != target:
                raise ValueError("Row setting/token target does not match its condition")
            if semantic and any(not isinstance(labels.get(slot), str) or not labels[slot].strip()
                                for slot in CLASS_LIMITS):
                raise ValueError("Semantic rows require all three nonempty labels")
            provenance = _row_sources(row)
            sources.update(provenance)
            grouped[domain].append((row, labels, provenance))
            signatures[identity] = _digest({"labels": labels, "tokens": count,
                                           "sources": sorted(provenance),
                                           "prompt_sha256": row.get("prompt_sha256")})
        if set(grouped) != set(domains):
            raise ValueError(f"{partition}: missing domains {sorted(set(domains)-set(grouped))}")
        by_domain[partition], identifiers[partition] = grouped, seen
        all_sources[partition], row_signatures[partition] = sources, signatures
    if all_sources["train"] & all_sources["test"] or identifiers["train"] & identifiers["test"]:
        raise ValueError("Training/test primary or supplementary sources overlap")
    if profiling is not None:
        if all_sources["profiling"] & all_sources["test"] or identifiers["profiling"] & identifiers["test"]:
            raise ValueError("Profiling/test primary or supplementary sources overlap")
        if not identifiers["train"] <= identifiers["profiling"]:
            raise ValueError("Training rows must be contained in the profiling pool")
        if any(row_signatures["train"][key] != row_signatures["profiling"][key] for key in identifiers["train"]):
            raise ValueError("Training/profiling copies disagree")
    result = {"schema_version": 1, "system": system, "setting": setting, "target_tokens": target,
              "target_basis": "artifact_choice" if setting == "cross_template" else "paper",
              "tolerance": tolerance, "semantic": semantic, "source_disjoint": True, "domains": {}}
    for domain in domains:
        stats, training_labels = {}, None
        for partition in partitions:
            rows = by_domain[partition][domain]
            if len(rows) != quotas[partition]:
                raise ValueError(f"{domain}/{partition}: expected {quotas[partition]} records, found {len(rows)}")
            counts = [row["token_count"] for row, _, _ in rows]
            mean = sum(counts) / len(counts)
            if abs(mean - target) > target * tolerance:
                raise ValueError(f"{domain}/{partition}: token mean {mean:.3f} misses target {target}")
            classes = {slot: sorted({labels[slot] for _, labels, _ in rows}) for slot in CLASS_LIMITS} if semantic else {}
            if semantic:
                if partition == "train":
                    training_labels = classes
                for slot, limit in CLASS_LIMITS.items():
                    if partition != "test" and len(classes[slot]) != limit:
                        raise ValueError(f"{domain}/{partition}/{slot}: expected {limit} training classes, found {len(classes[slot])}")
                    expected = vocabulary[domain][slot] if vocabulary is not None else training_labels[slot]
                    if set(classes[slot]) - set(expected) or (partition != "test" and set(classes[slot]) != set(expected)):
                        raise ValueError(f"{domain}/{partition}/{slot}: labels violate the fixed training vocabulary")
            sources = set().union(*(provenance for _, _, provenance in rows))
            stats[partition] = {"count": len(rows), "token_mean": mean, "token_min": min(counts),
                                "token_max": max(counts), "class_counts": {slot: len(values) for slot, values in classes.items()},
                                "classes": classes, "source_count": len(sources),
                                "source_ids_sha256": _digest(sorted(sources))}
        result["domains"][domain] = stats
    return result


def validate_complete_matrix(summary):
    """Require 3 systems x 5 settings x 4 domains with real tokenizer assets."""
    if not isinstance(summary, Mapping) or set(summary.get("tokenizers", {})) != set(SYSTEMS):
        raise ValueError("A complete matrix requires all three tokenizer descriptions")
    descriptions = summary["tokenizers"]
    for system, entry in descriptions.items():
        if not isinstance(entry, Mapping):
            raise ValueError("Invalid tokenizer description")
        expected_id, expected_type, revision = MODEL_IDENTITIES[system]
        if (entry.get("system") != system or entry.get("model_id") != expected_id or
                entry.get("model_type") != expected_type or entry.get("revision") != revision or
                entry.get("identity_verified") is not True or not entry.get("backend_sha256") or
                not all(entry.get("assets", {}).get(name, {}).get("sha256") == sha
                        for name, sha in IDENTITY_HASHES[system].items())):
            raise ValueError(f"{system}: a complete matrix requires verified official tokenizer assets")
        if entry.get("versions", {}).get("transformers") != TRANSFORMERS_VERSIONS[system]:
            raise ValueError(f"{system}: matrix tokenizer must use its native Transformers version")
        payload = {key: value for key, value in entry.items() if key != "fingerprint"}
        if entry.get("fingerprint") != _digest(payload):
            raise ValueError("Tokenizer fingerprint mismatch")
    conditions = summary.get("conditions")
    if not isinstance(conditions, list):
        raise ValueError("Matrix conditions must be a list")
    expected = {(system, setting) for system in SYSTEMS for setting in SETTINGS}
    seen, common_vocabulary = set(), {}
    for condition in conditions:
        if not isinstance(condition, Mapping):
            raise ValueError("Invalid matrix condition")
        pair = condition.get("system"), condition.get("setting")
        if pair not in expected or pair in seen:
            raise ValueError("Unexpected or duplicate matrix condition")
        seen.add(pair)
        if set(condition.get("domains", {})) != set(DOMAINS):
            raise ValueError("Every matrix condition requires all four domains")
        if condition.get("source_disjoint") is not True or condition.get("semantic") is not True:
            raise ValueError("Complete semantic matrix requires source separation and semantic labels")
        target = SETTINGS[pair[1]]
        if condition.get("target_tokens") != target:
            raise ValueError("Matrix target does not match setting")
        expected_basis = "artifact_choice" if pair[1] == "cross_template" else "paper"
        if condition.get("target_basis") != expected_basis:
            raise ValueError("Matrix token target provenance does not match setting")
        tolerance = condition.get("tolerance")
        if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or not 0 <= tolerance <= .1:
            raise ValueError("Complete matrix tolerance must be at most 10 percent")
        if condition.get("tokenizer_fingerprint") != descriptions[pair[0]]["fingerprint"]:
            raise ValueError("Condition is not bound to its tokenizer fingerprint")
        for domain, stats in condition["domains"].items():
            for partition, quota in (("train", 3000), ("test", 1000), ("profiling", 3200)):
                item = stats.get(partition)
                if not isinstance(item, Mapping) or item.get("count") != quota:
                    raise ValueError(f"{domain}/{partition}: wrong complete-matrix quota")
                mean = item.get("token_mean")
                if not isinstance(mean, (int, float)) or not math.isfinite(mean) or abs(mean-target) > target*tolerance:
                    raise ValueError(f"{domain}/{partition}: invalid per-domain token mean")
                for slot, limit in CLASS_LIMITS.items():
                    labels = item.get("classes", {}).get(slot, [])
                    if (not isinstance(labels, list) or len(labels) != len(set(labels)) or
                            item.get("class_counts", {}).get(slot) != len(labels) or
                            (partition != "test" and len(labels) != limit) or not labels):
                        raise ValueError(f"{domain}/{partition}/{slot}: invalid class coverage")
                    if partition != "train" and set(labels) - set(stats["train"]["classes"][slot]):
                        raise ValueError("Matrix semantic labels violate training closed set")
            classes = {slot: set(stats["train"]["classes"][slot]) for slot in CLASS_LIMITS}
            if domain in common_vocabulary and classes != common_vocabulary[domain]:
                raise ValueError("Every matrix condition must use the same fixed domain vocabulary")
            common_vocabulary[domain] = classes
    if seen != expected:
        raise ValueError(f"Incomplete matrix: missing {sorted(expected-seen)}")
    return {"systems": 3, "settings": 5, "domains": 4, "conditions": 60}
