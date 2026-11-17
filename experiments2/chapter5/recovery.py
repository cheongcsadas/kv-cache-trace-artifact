"""Actual metadata correlation and anonymous hardware activity-window recovery."""
from collections import defaultdict
from copy import deepcopy
from statistics import fmean
import math
from attacks.metadata import reconstruct_metadata
from attacks.hardware import Calibration, LocalizationError, hardware_activity_windows
from evaluation import hardware_trr, merge_reference_windows
from experiments.config import load_config
from experiments.paper_protocol import validate_paper_config, validate_calibration, validate_hardware_observation
from .common import read_json, read_rows, resolve, require_provenance, summarize_counts


def metadata_evidence_score(predicted, reference, tolerance_ms):
    """Chronological 1:1 matching of exact events and interval-censored evidence.

    Explicit evidence uses the declared absolute tolerance. A snapshot transition
    matches a same-block/type reference inside its observed sampling interval;
    no midpoint is invented. The interval convention is (start, end].
    """
    if not math.isfinite(tolerance_ms) or tolerance_ms < 0:
        raise ValueError("Metadata tolerance must be nonnegative")
    for event in predicted:
        bounds=(event["interval_start_ms"],event["interval_end_ms"]) if event.get("evidence") == "snapshot" else (event["time_ms"],)
        if any(not isinstance(t,(int,float)) or isinstance(t,bool) or not math.isfinite(t) for t in bounds):
            raise ValueError("Metadata evidence requires finite timestamps")
        if len(bounds) == 2 and bounds[1] <= bounds[0]: raise ValueError("Invalid snapshot interval")
    if any(not isinstance(r["time_ms"],(int,float)) or not math.isfinite(r["time_ms"]) for r in reference):
        raise ValueError("Reference events require finite timestamps")
    def endpoint(e):
        return e["interval_end_ms"] if e.get("evidence") == "snapshot" else e["time_ms"]
    predictions = sorted(predicted, key=endpoint)
    refs = sorted(reference, key=lambda e: e["time_ms"])
    matched = set()
    for event in predictions:
        for index, ref in enumerate(refs):
            if index in matched: continue
            if any(event.get(k) != ref.get(k) for k in ("op", "layer", "block_id", "request_id")): continue
            if event.get("evidence") == "snapshot":
                timing = event["interval_start_ms"] < ref["time_ms"] <= event["interval_end_ms"]
            else:
                timing = abs(event["time_ms"]-ref["time_ms"]) <= tolerance_ms
            if timing:
                matched.add(index)
                break
    return summarize_counts(len(matched), len(predictions)-len(matched), len(refs)-len(matched))


def metadata_sources(obs, source):
    if source in obs.get("source_views", {}):
        return deepcopy(obs["source_views"][source])
    result = deepcopy(obs)
    if source == "combined": return result
    result["events"] = [r for r in result.get("events", []) if r.get("source", "logs") == source]
    result["snapshots"] = [r for r in result.get("snapshots", []) if r.get("source", "telemetry") == source]
    return result


def execute_recovery(manifest, base, hardware, pilot=False, allow_synthetic=False):
    """Consume paired observation/reference paths while keeping references out of encoding."""
    runs = manifest["runs"]
    records = []
    identities = set()
    auxiliary_sources = set()
    sources_by_system = defaultdict(set)
    systems_seen = set()
    for run in runs:
        system = run["system"]
        if system not in {"quest", "infinigen", "retrievalattention"}: raise ValueError("Unknown system")
        systems_seen.add(system)
        config=load_config(resolve(base,run["config"]))
        validate_paper_config(config)
        if config["system"] != system: raise ValueError("Recovery run and config systems differ")
        pairs_path = resolve(base, run["pairs"])
        pairs = read_rows(pairs_path)
        require_provenance(pairs, allow_synthetic)
        calibration = Calibration(**read_json(resolve(base, run["calibration"]))) if hardware else None
        if hardware: validate_calibration(calibration.to_dict(),config)
        if not hardware and "tolerance_ms" not in run:
            raise ValueError("Metadata recovery requires a declared tolerance_ms")
        accum = defaultdict(lambda: [0, 0, 0])
        failures = 0
        acquisition_failures = 0
        reference_failures = 0
        for pair in pairs:
            identity = (system, pair["sample_id"])
            if identity in identities: raise ValueError("Duplicate auxiliary prompt in recovery experiment")
            identities.add(identity)
            if not isinstance(pair.get("source_id"),str) or not pair["source_id"]:
                raise ValueError("Auxiliary prompts require source_id")
            if pair["source_id"] in sources_by_system[system]:
                raise ValueError("Each system must evaluate each auxiliary source record exactly once")
            sources_by_system[system].add(pair["source_id"])
            auxiliary_sources.add(pair["source_id"])
            failed=any(pair.get(k,"completed") != "completed" for k in ("request_status","acquisition_status"))
            obs = None if failed else read_json(resolve(pairs_path.parent, pair["observation_path"]))
            acquisition_failures += int(failed)
            reference_path = pair.get("reference_path")
            reference_path = resolve(pairs_path.parent,reference_path) if reference_path else None
            if reference_path is None or not reference_path.is_file():
                # An absent evaluator trace is unknown ground truth, not an
                # empty event list. Keep the attempt and invalidate full TRR.
                reference_failures += 1
                continue
            reference = read_json(reference_path)
            if isinstance(reference, dict): reference = reference["events"]
            if hardware:
                if not failed: validate_hardware_observation(obs,config)
                # Reference request identity is evaluator-side only; each pair is
                # independently matched so strip it instead of giving it to probes.
                reference = [{k:v for k,v in r.items() if k != "request_id"} for r in reference]
                try:
                    if failed: raise LocalizationError("Acquisition failed")
                    predicted = hardware_activity_windows(obs, calibration)
                except LocalizationError:
                    predicted = []
                    failures += 1
                refs = merge_reference_windows(reference, merge_gap_ms=50)
                for op in ("HIT", "LOAD", "EVICT"):
                    # Preserve original run adjacency before filtering by type.
                    ref_windows = [r for r in refs if r["op"] == op]
                    pred_windows = [r for r in predicted if r["op"] == op]
                    score = match_windows(pred_windows, ref_windows)
                    for i,key in enumerate(("tp", "fp", "fn")): accum[op][i] += score[key]
            else:
                for source in ("logs", "telemetry", "shm", "combined"):
                    predicted = [] if failed else reconstruct_metadata(metadata_sources(obs, source))
                    score = metadata_evidence_score(predicted, reference, float(run["tolerance_ms"]))
                    for i,key in enumerate(("tp", "fp", "fn")): accum[source][i] += score[key]
        categories=("HIT","LOAD","EVICT") if hardware else ("logs","telemetry","shm","combined")
        scores={key:summarize_counts(*accum[key]) for key in categories}
        if reference_failures:
            for score in scores.values():
                for metric in ("precision_percent","recall_percent","trr_percent"):
                    score[metric]=None
        records.append(dict(system=system, prompts=len(pairs), localization_failures=failures,
                            acquisition_failures=acquisition_failures,
                            reference_failures=reference_failures,
                            scorable_prompts=len(pairs)-reference_failures, scores=scores))
    if not pilot:
        if len(auxiliary_sources) != 1000: raise ValueError("Section 5.2 uses a pool of 1,000 unique auxiliary source records")
        if systems_seen != {"quest", "infinigen", "retrievalattention"}:
            raise ValueError("Recovery evaluation requires all three system configurations")
        if any(sources != auxiliary_sources for sources in sources_by_system.values()):
            raise ValueError("All three systems must evaluate the same 1,000 auxiliary source records")
    # Pool within each system first; macro-average systems, not arbitrarily sized runs.
    pooled = defaultdict(lambda: defaultdict(lambda: [0,0,0]))
    for record in records:
        for category, score in record["scores"].items():
            for i,key in enumerate(("tp","fp","fn")): pooled[record["system"]][category][i] += score[key]
    categories = sorted(set().union(*(p.keys() for p in pooled.values())))
    macro = {}
    incomplete_systems={record["system"] for record in records if record["reference_failures"]}
    for category in categories:
        scores = [summarize_counts(*pooled[system][category]) for system in sorted(pooled)]
        macro[category] = {key:None if incomplete_systems else fmean(s[key] for s in scores)
                           for key in ("precision_percent","recall_percent","trr_percent")}
    records.append(dict(system="macro_average", unique_auxiliary_sources=len(auxiliary_sources),
                        prompt_configuration_observations=len(identities),
                        reference_failures=sum(record["reference_failures"] for record in records), scores=macro))
    return records


def match_windows(predicted, reference):
    used = set()
    for pred in sorted(predicted, key=lambda r:r["start_ms"]):
        for index, ref in enumerate(sorted(reference,key=lambda r:r["start_ms"])):
            if index not in used and pred["op"] == ref["op"] and max(pred["start_ms"],ref["start_ms"]) <= min(pred["end_ms"],ref["end_ms"]):
                used.add(index)
                break
    return summarize_counts(len(used), len(predicted)-len(used), len(reference)-len(used))
