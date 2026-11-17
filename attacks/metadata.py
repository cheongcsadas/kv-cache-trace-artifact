"""Correlate exported management records without turning snapshots into exact logs."""
from collections import defaultdict
import hashlib
import math
import numpy as np
from .encoding import allocate, time_ms, window_index

OPS = ("HIT", "LOAD", "EVICT")
EXCLUDED = {"allocation", "allocate", "initialization", "cleanup", "request_completion", "completion",
            "query_transfer", "result_transfer", "cpu_attention"}


def _excluded(row):
    return row.get("kind") in EXCLUDED or row.get("cause") in EXCLUDED


def reconstruct_metadata(observation: dict) -> list[dict]:
    """Return explicit events and interval-censored net transitions (paper Eq. 4).

    Snapshots contain CPU-backed, previously computed blocks only. Allocation and
    completion removals must be marked by the publisher, including excluded_blocks
    on either adjacent snapshot. No HIT is inferred from unchanged residency.
    Explicit events take precedence over overlapping snapshot evidence.
    """
    request_id = observation["request_id"]
    events = []
    for row in observation.get("events", []):
        if row.get("request_id", request_id) != request_id:
            continue
        if _excluded(row):
            continue
        if row["op"] not in OPS:
            raise ValueError(f"unsupported management operation: {row['op']}")
        events.append({"request_id": request_id, "layer": int(row["layer"]),
                       "block_id": row["block_id"], "op": row["op"],
                       "time_ms": time_ms(row), "evidence": "explicit",
                       "step": row.get("step")})
    # A publisher may export the same event on several transports. event_id, when
    # provided, must be deduplicated by the acquisition reader before this call.
    explicit = list(events)
    groups = defaultdict(list)
    for row in observation.get("snapshots", []):
        if row.get("request_id", request_id) == request_id:
            groups[(int(row["layer"]), row.get("source", "default"))].append(row)
    inferred = []
    for (layer, _source), snapshots in groups.items():
        snapshots = sorted(snapshots, key=time_ms)
        for old, new in zip(snapshots, snapshots[1:]):
            a, b = time_ms(old), time_ms(new)
            if b <= a:
                raise ValueError("snapshot times must strictly increase per source/layer")
            # An initialization snapshot establishes the starting resident set;
            # following changes can be valid management. Never infer the reset
            # into initialization, or compare through cleanup/allocation epochs.
            old_is_initial = old.get("kind") == "initialization" or old.get("cause") == "initialization"
            if (_excluded(old) and not old_is_initial) or _excluded(new):
                continue
            excluded = set(old.get("excluded_blocks", [])) | set(new.get("excluded_blocks", []))
            before = set(old["resident_blocks"] if "resident_blocks" in old else old["resident"]) - excluded
            after = set(new["resident_blocks"] if "resident_blocks" in new else new["resident"]) - excluded
            for op, blocks in (("LOAD", after - before), ("EVICT", before - after)):
                for block in sorted(blocks, key=str):
                    duplicate = any(e["layer"] == layer and e["block_id"] == block
                                    and e["op"] == op and a < e["time_ms"] <= b
                                    for e in explicit)
                    duplicate |= any(e["layer"] == layer and e["block_id"] == block
                                     and e["op"] == op and max(a, e["interval_start_ms"])
                                     < min(b, e["interval_end_ms"]) for e in inferred)
                    if not duplicate:
                        inferred.append({"request_id": request_id, "layer": layer,
                                         "block_id": block, "op": op, "time_ms": None,
                                         "evidence": "snapshot", "interval_start_ms": a,
                                         "interval_end_ms": b})
    events.extend(inferred)
    return sorted(events, key=lambda e: e["time_ms"] if e["time_ms"] is not None else e["interval_end_ms"])


def _bucket(layer, block, count):
    # Unlike Python hash(), BLAKE2 has stable assignments across processes.
    digest = hashlib.blake2b(f"{layer}:{block}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % count


def encode_metadata(observation: dict, window_ms=10.0, max_windows=512,
                    spatial_bins=32, retained_ops=OPS, rebase_retained=False):
    """Flattenable block-associated encoding; bin choices are artifact defaults.

    Interval evidence is distributed by overlap, not assigned a fabricated event
    timestamp. Exact feature definitions and hashing are in attacks/README.md.
    """
    if spatial_bins < 1:
        raise ValueError("spatial_bins must be positive")
    if not set(retained_ops) <= set(OPS):
        raise ValueError("unknown retained metadata operation")
    events = [e for e in reconstruct_metadata(observation) if e["op"] in retained_ops]
    if "start_ms" not in observation or "end_ms" not in observation:
        raise ValueError("metadata observation requires start_ms and end_ms")
    start, end = float(observation["start_ms"]), float(observation["end_ms"])
    if rebase_retained:
        clipped = []
        for event in events:
            if event["time_ms"] is not None:
                if start <= event["time_ms"] <= end:
                    clipped.append(event)
            else:
                a, b = max(start, event["interval_start_ms"]), min(end, event["interval_end_ms"])
                if b > a:
                    clipped.append(dict(event, interval_start_ms=a, interval_end_ms=b))
        events = clipped
        # Removing an event must not leave its timestamp behind in the duration,
        # origin, padding mask or offsets of the remaining event groups.
        if events:
            start = min(e["time_ms"] if e["time_ms"] is not None else e["interval_start_ms"] for e in events)
            end = max(e["time_ms"] if e["time_ms"] is not None else e["interval_end_ms"] for e in events)
            end = max(end, start + window_ms)
        else:
            start, end = 0., float(window_ms)
    names = tuple(["explicit_hit", "explicit_load", "explicit_evict", "net_load", "net_evict",
                   "event_count", "unique_blocks", "unique_layers", "interevent_mean_ms",
                   "interevent_max_ms", "event_offset_mean_ms", "event_offset_std_ms",
                   "relative_window", "offset_ms"] +
                  [f"{op.lower()}_block_bin_{i}" for op in OPS for i in range(spatial_bins)])
    result = allocate(start, end, float(window_ms), int(max_windows), names, "meta")
    exact_by_window = defaultdict(list)
    n = int(result.mask.sum())
    for event in events:
        op = OPS.index(event["op"])
        spatial = 14 + op * spatial_bins + _bucket(event["layer"], event["block_id"], spatial_bins)
        if event["evidence"] == "explicit":
            t = event["time_ms"]
            if not start <= t <= end:
                continue
            j = window_index(t, start, window_ms)
            if j >= n:
                continue
            result.values[j, op] += 1
            result.values[j, 5] += 1
            result.values[j, spatial] += 1
            exact_by_window[j].append(event)
        else:
            a, b = event["interval_start_ms"], event["interval_end_ms"]
            for j in range(n):
                left, right = start + j * window_ms, min(end, start + (j + 1) * window_ms)
                overlap = max(0.0, min(b, right) - max(a, left))
                if overlap:
                    weight = overlap / (b - a)
                    result.values[j, 3 if event["op"] == "LOAD" else 4] += weight
                    result.values[j, spatial] += weight
    for j in range(n):
        ev = exact_by_window[j]
        if ev:
            times = np.array([e["time_ms"] for e in ev])
            gaps = np.diff(times)
            result.values[j, 6:12] = [len({(e["layer"], str(e["block_id"])) for e in ev}),
                len({e["layer"] for e in ev}), float(gaps.mean()) if gaps.size else 0,
                float(gaps.max()) if gaps.size else 0, float((times-start).mean()), float(times.std())]
        result.values[j, 12:14] = [j / max(1, max_windows - 1), j * window_ms]
    if rebase_retained and not events:
        result.values.fill(0)
        result.mask.fill(False)
    return result
