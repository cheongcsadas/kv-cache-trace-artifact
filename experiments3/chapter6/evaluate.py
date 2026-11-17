"""Compute defense accuracy and paired native inference latency from collected data."""
from collections import Counter
import json
import math
from pathlib import Path
from statistics import fmean, median

from datasets.preprocessing.sources import source_ids
from experiments.config import load_config
from experiments.paper_protocol import validate_paper_config, validate_dataset_protocol, DOMAINS
from experiments.chapter5.common import read_json, read_rows, resolve, require_provenance, validate_collection
from experiments.run import run as train_attack
from .specification import SYSTEMS, DEFENSES


LOAD_BATCH_UNITS = {
    "quest": "per_layer_decode_step",
    "infinigen": "per_kv_head_layer_decode_step",
    "retrievalattention": "per_layer_prefill_persistent_context",
}


def _native_receipt(row, dataset_path):
    receipt = row.get("native_receipt")
    if receipt is None:
        if not row.get("native_receipt_path"):
            raise ValueError("Completed native observation requires its execution receipt")
        receipt = read_json(resolve(Path(dataset_path).parent, row["native_receipt_path"]))
    if not isinstance(receipt, dict):
        raise ValueError("Native execution receipt must be an object")
    return receipt


def _validate_defense_execution(row, dataset_path, config):
    """Require completed measured attempts to attest executed loading operations.

    Failed attempts remain in evaluation denominators even when no execution
    receipt could be produced. Explicit synthetic fixtures are handled by the
    caller, and are never counted as validated native executions.
    """
    if any(row.get(field, "completed") != "completed" for field in ("request_status", "acquisition_status")):
        return None
    receipt = _native_receipt(row, dataset_path)
    if receipt.get("execution_status", "completed") != "completed":
        raise ValueError("Completed observation has a failed native execution receipt")
    _latency(row, dataset_path, receipt=receipt)  # Bind request identity and actual interval.
    system = config["system"]
    defense = config.get("experiment", {}).get("defense") or "none"
    for field in ("block_tokens", "residency_fraction"):
        if receipt.get(field) != config["cache"][field]:
            raise ValueError(f"Executed defense {field} differs from the paper configuration")
    capacity = math.ceil(math.ceil(row["token_count"] / config["cache"]["block_tokens"])
                         * config["cache"]["residency_fraction"])
    if receipt.get("capacity_slots") != capacity:
        raise ValueError("Executed defense residency capacity disagrees with the actual prompt length")
    for group, field in (("inference", "generation_tokens"), ("cache", "selected_token_budget")):
        declared = config[group].get(field)
        if (isinstance(declared, bool) or not isinstance(declared, int) or declared < 1
                or receipt.get(field) != declared):
            raise ValueError(f"Executed defense {field} must match the declared run setting")
    if receipt.get("defense_parameters") != config["defenses"].get(defense, {}):
        raise ValueError("Executed defense parameters differ from the configured 65-page/5-ms policy")
    if receipt.get("defense_load_batch_unit") != LOAD_BATCH_UNITS[system]:
        raise ValueError("Executed defense uses a different eligible loading-operation unit")

    def counter(value, name):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Native defense counter {name} must be a nonnegative integer")
        return value

    batches = counter(receipt.get("defense_load_batches"), "load_batches")
    delays = counter(receipt.get("defense_delay_calls"), "delay_calls")
    transferred = receipt.get("defense_batch_transfer_counts")
    if not isinstance(transferred, list) or len(transferred) != batches or not batches:
        raise ValueError("Native execution must record each actual eligible loading batch")
    if any(counter(value, "batch transfer") == 0 for value in transferred):
        raise ValueError("An eligible loading batch must contain actual transfers")
    if system == "quest":
        required, padding = receipt.get("defense_required_loads"), receipt.get("defense_padding_loads")
    elif system == "infinigen":
        required, padding = receipt.get("required_load_count"), receipt.get("defense_padding_loads")
    else:
        counts = receipt.get("transfer_counts", {})
        required, padding = counts.get("required"), counts.get("padding")
    required, padding = counter(required, "required loads"), counter(padding, "padding loads")
    if not required or sum(transferred) != required + padding:
        raise ValueError("Actual required/padding transfers disagree with native loading batches")
    if delays != (batches if defense == "dynamic_timing" else 0):
        raise ValueError("Timing defense must delay once per eligible loading batch")
    if defense == "fixed_budget":
        budget = config["defenses"]["fixed_budget"]["load_pages"]
        if any(value < budget for value in transferred):
            raise ValueError("Fixed-budget native batches must contain at least 65 actual transfers")
        if receipt.get("defense_overflow_batches", 0) != sum(value > budget for value in transferred):
            raise ValueError("Fixed-budget overflow count disagrees with actual transfer batches")
    elif padding:
        raise ValueError("Padding transfers require the fixed-budget defense")
    return dict(eligible_batches=batches, required_loads=required, padding_loads=padding, delay_calls=delays)


def _identity(row):
    value = row.get("original_sample_id")
    if not isinstance(value, str) or not value:
        raise ValueError("Cross-defense pairing requires original_sample_id from the prepared prompt")
    digest = row.get("prompt_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Cross-defense pairing requires the rendered prompt_sha256")
    tokens = row.get("token_count")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
        raise ValueError("Cross-defense pairing requires actual tokenizer token_count")
    return value, (tuple(sorted(source_ids(row))), row["domain"], digest, tokens)


def _paired_signature(rows):
    result = {}
    for row in rows:
        identity, signature = _identity(row)
        if identity in result:
            raise ValueError("Duplicate original prompt in one defense condition")
        result[identity] = signature
    return result


def load_captures(manifest, base, *, accuracy, pilot=False, allow_synthetic=False):
    if allow_synthetic and not pilot:
        raise ValueError("Synthetic fixtures require explicit pilot mode")
    captures = manifest.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ValueError("Chapter 6 manifest requires nonempty captures")
    result, signatures, settings = {}, {}, {}
    for capture in captures:
        system, defense = capture.get("system"), capture.get("defense")
        key = system, defense
        if system not in SYSTEMS or defense not in DEFENSES or key in result:
            raise ValueError("Defense captures require distinct system/defense conditions")
        if capture.get("setting") != "templated_short" or capture.get("channel") != "meta":
            raise ValueError("Section 6 evaluates short templated metadata observations")
        config_path = resolve(base, capture["config"]).resolve()
        config = load_config(config_path)
        validate_paper_config(config)
        if config["system"] != system or (config.get("experiment", {}).get("defense") or "none") != defense:
            raise ValueError("Native configuration and defense condition disagree")
        # Apart from the defense setting, compare the execution-sensitive inputs.
        same_settings = {name: config[name] for name in (
            "model", "backend", "inference", "cache", "classifier", "observation", "implementation_choices")}
        if settings.setdefault(system, same_settings) != same_settings:
            raise ValueError("Defense comparisons require identical model/cache/generation/probe/classifier settings")
        paths = {split: resolve(base, capture[split]).resolve() for split in ("train", "test") if split in capture}
        if "test" not in paths or accuracy and "train" not in paths:
            raise ValueError("Accuracy needs train/test captures; latency needs held-out test captures")
        rows = {split: read_rows(path) for split, path in paths.items()}
        execution_audit = {}
        for split, values in rows.items():
            require_provenance(values, allow_synthetic)
            sample_ids = [row.get("sample_id") for row in values]
            if any(not isinstance(value, str) or not value for value in sample_ids) or len(set(sample_ids)) != len(values):
                raise ValueError("Each defense split requires distinct nonempty sample_id values")
            if any((row.get("defense") or "none") != defense for row in values):
                raise ValueError("Observation defense labels disagree with the executed condition")
            validate_collection(capture, config, values)
            if capture["collection"].get("defense") != defense:
                raise ValueError("Collection receipt must record its actual defense")
            signature = _paired_signature(values)
            if signatures.setdefault((system, split), signature) != signature:
                raise ValueError("Defended and undefended conditions must use exactly the same rendered prompts")
            verified = [_validate_defense_execution(row, paths[split], config)
                        for row in values if row.get("provenance") == "measured"]
            completed = [record for record in verified if record is not None]
            execution_audit[split] = dict(validated_completed_attempts=len(completed),
                **{name: sum(record[name] for record in completed)
                   for name in ("eligible_batches", "required_loads", "padding_loads", "delay_calls")})
        if "train" in rows:
            validate_dataset_protocol(rows["train"], rows["test"],
                train_per_domain=None if pilot else 3000, test_per_domain=None if pilot else 1000,
                require_measured=not allow_synthetic)
        else:
            tally = Counter(row["domain"] for row in rows["test"])
            if set(tally) != set(DOMAINS) or not pilot and any(tally[d] != 1000 for d in DOMAINS):
                raise ValueError("Defense latency uses 1,000 held-out prompts per domain")
        result[key] = dict(capture=capture, config_path=config_path, config=config, paths=paths,
                           rows=rows, execution_audit=execution_audit)
    if set(result) != {(system, defense) for system in SYSTEMS for defense in DEFENSES}:
        raise ValueError("Chapter 6 requires three systems and all three loading conditions")
    return result


def accuracy(manifest, base, output, *, pilot=False, allow_synthetic=False):
    captures = load_captures(manifest, base, accuracy=True, pilot=pilot, allow_synthetic=allow_synthetic)
    records = []
    for system in SYSTEMS:
        for defense in DEFENSES:
            testing = captures[system, defense]
            for mode in (("none",) if defense == "none" else ("aware", "oblivious")):
                training = captures[system, "none" if mode == "oblivious" else defense]
                # Source disjointness is checked again across the actual selected
                # cross-condition train/test files by the attack runner.
                run_id = f"{system}-{defense}-{mode}"
                report = train_attack(testing["config_path"], training["paths"]["train"],
                    testing["paths"]["test"], Path(output)/run_id, "MAA",
                    defense_mode=mode, allow_synthetic=allow_synthetic, paper_protocol=not pilot)
                records.append(dict(id=run_id, system=system, defense=defense, attacker_setting=mode,
                    attack="MAA", channel="meta", setting="templated_short", metrics=report["metrics"],
                    train_count=report["train_count"], test_count=report["test_count"],
                    training_failures=len(report["train_localization_failures"]),
                    test_failures=len(report["test_localization_failures"]),
                    native_execution_validation=dict(train=training["execution_audit"]["train"],
                                                     test=testing["execution_audit"]["test"])))
    baselines = {row["system"]: row["metrics"]["ai_asr_percent"] for row in records if row["defense"] == "none"}
    for row in records:
        row["reduction_percentage_points"] = baselines[row["system"]] - row["metrics"]["ai_asr_percent"]
    return records


def _latency(row, dataset_path, *, receipt=None):
    if any(row.get(field, "completed") != "completed" for field in ("request_status", "acquisition_status")):
        return None
    if receipt is None:
        receipt = _native_receipt(row, dataset_path)
    if receipt.get("execution_status", "completed") != "completed":
        return None
    if receipt.get("request_id") != row.get("sample_id"):
        raise ValueError("Latency receipt request_id differs from observation sample_id")
    if receipt.get("token_count") != row.get("token_count"):
        raise ValueError("Latency receipt token_count differs from the paired prompt")
    if receipt.get("system") != row.get("system"):
        raise ValueError("Latency receipt system differs from observation")
    if (receipt.get("defense") or "none") != (row.get("defense") or "none"):
        raise ValueError("Latency receipt defense differs from observation")
    start, end = (receipt.get(field) for field in ("execution_start_monotonic_ms", "execution_end_monotonic_ms"))
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in (start, end)) or end <= start:
        raise ValueError("Native inference interval must have positive finite duration")
    return (end-start)/1000


def latency(manifest, base, *, pilot=False, allow_synthetic=False):
    captures = load_captures(manifest, base, accuracy=False, pilot=pilot, allow_synthetic=allow_synthetic)
    result = []
    for system in SYSTEMS:
        timings = {}
        for defense in DEFENSES:
            actual = captures[system, defense]
            timings[defense] = {row["original_sample_id"]: _latency(row, actual["paths"]["test"])
                                for row in actual["rows"]["test"]}
        ids = list(timings["none"])
        paired = [identity for identity in ids if all(timings[d][identity] is not None for d in DEFENSES)]
        baseline = fmean(timings["none"][identity] for identity in paired) if paired else None
        for defense in DEFENSES:
            values = [timings[defense][identity] for identity in paired]
            mean = fmean(values) if values else None
            result.append(dict(id=f"{system}-{defense}", system=system, defense=defense,
                setting="templated_short", prompts=len(ids), paired_completed_prompts=len(paired),
                failed_prompts=sum(value is None for value in timings[defense].values()),
                unpaired_prompts=len(ids)-len(paired), timing_scope="native_execution",
                mean_seconds=mean, median_seconds=median(values) if values else None,
                baseline_mean_seconds=baseline,
                native_execution_validation=captures[system, defense]["execution_audit"]["test"],
                overhead_percent=100*(mean/baseline-1) if baseline is not None else None))
    return result
