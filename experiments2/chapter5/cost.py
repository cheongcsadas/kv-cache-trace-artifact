"""Aggregate measured offline observation-collection times for Table 7."""
from collections import Counter
from statistics import fmean, median
from .common import DOMAINS, SYSTEMS, read_rows, resolve, require_provenance, finite_positive


def execute_cost(manifest, base, pilot=False, allow_synthetic=False):
    results=[]
    for run in manifest["runs"]:
        if run["system"] not in SYSTEMS or run["channel"] not in {"meta","hw"}:
            raise ValueError("Invalid profiling system/channel")
        if run.get("timing_scope") != "observation_collection":
            raise ValueError("Table 7 requires observation_collection timings excluding setup/calibration/training")
        rows=read_rows(resolve(base,run["timings"]))
        require_provenance(rows,allow_synthetic)
        ids=[r["sample_id"] for r in rows]
        if len(ids) != len(set(ids)): raise ValueError("Duplicate profiling prompt")
        counts=Counter(r["domain"] for r in rows)
        if set(counts) != set(DOMAINS): raise ValueError("Profiling cost requires all four domains")
        if not pilot and any(counts[d] != 3000 for d in DOMAINS):
            raise ValueError("Table 7 uses 3,000 auxiliary prompts/domain, 12,000 total")
        milliseconds=[]
        for row in rows:
            if row.get("system",run["system"]) != run["system"] or row.get("channel",run["channel"]) != run["channel"]:
                raise ValueError("Timing records have mixed configurations")
            if any(row.get(field,"completed") != "completed" for field in ("status","request_status","acquisition_status")):
                raise ValueError("Profiling collection timings must cover completed acquisitions; report retries separately")
            if row.get("timing_scope","observation_collection") != "observation_collection":
                raise ValueError("Timing row is not an observation-collection measurement")
            value=row.get("collection_ms")
            milliseconds.append(finite_positive(value,"collection_ms"))
        seconds=[v/1000 for v in milliseconds]
        results.append(dict(id=run["id"],system=run["system"],channel=run["channel"],prompts=len(rows),
            per_domain_counts=dict(counts),mean_seconds_per_prompt=fmean(seconds),
            median_seconds_per_prompt=median(seconds),total_observation_hours=sum(seconds)/3600))
    return results
