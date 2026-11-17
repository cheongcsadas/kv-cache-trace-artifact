"""Anonymous hardware observations, calibrated without victim reference events."""
from dataclasses import dataclass, asdict
import math
import numpy as np
from .encoding import allocate, time_ms, window_index


class LocalizationError(ValueError):
    """A failed attack attempt; evaluation must retain it in the denominator."""


@dataclass(frozen=True)
class Calibration:
    baseline_h2d_us: float
    compute_threshold_us: float | None
    page_threshold_us: float | None
    h2d_threshold_us: float = 200.0
    consecutive_samples: int = 3
    sample_interval_ms: float = 1.0
    idle_gap_ms: float = 20.0
    submission_error_ms: float = 1.0
    max_sample_gap_factor: float = 1.5

    def __post_init__(self):
        numbers = [self.baseline_h2d_us, self.h2d_threshold_us, self.sample_interval_ms,
                   self.idle_gap_ms, self.max_sample_gap_factor]
        numbers += [x for x in (self.compute_threshold_us, self.page_threshold_us) if x is not None]
        if not all(math.isfinite(x) and x > 0 for x in numbers):
            raise ValueError("calibration latencies and intervals must be finite and positive")
        if (isinstance(self.consecutive_samples, bool) or not isinstance(self.consecutive_samples, int)
                or self.consecutive_samples < 1 or not math.isfinite(self.submission_error_ms) or self.submission_error_ms < 0):
            raise ValueError("invalid persistence or submission-time tolerance")

    @classmethod
    def fit(cls, baseline_repetitions, compute_idle_us=None, page_idle_us=None, empirical_fpr=.05, **kwargs):
        """Freeze thresholds on disjoint auxiliary idle measurements.

        The H2D reference is the median of three repetition medians. Strict > and
        < tests combined with conservative empirical quantiles bound each idle
        channel's empirical trigger fraction by 5% (ties do not trigger).
        """
        if len(baseline_repetitions) != 3:
            raise ValueError("exactly three baseline repetitions are required")
        if not math.isfinite(empirical_fpr) or not 0 <= empirical_fpr < 1:
            raise ValueError("empirical_fpr must lie in [0, 1)")
        def vector(values):
            v = np.asarray(values, dtype=float)
            if v.ndim != 1 or not v.size or not np.all(np.isfinite(v)) or np.any(v <= 0):
                raise ValueError("calibration requires nonempty positive finite vectors")
            return v
        reference = float(np.median([np.median(vector(x)) for x in baseline_repetitions]))
        def threshold(values, upper):
            values = np.sort(vector(values))
            allowed = math.floor(len(values)*empirical_fpr)
            return float(values[len(values)-allowed-1] if upper else values[allowed])
        compute = None if compute_idle_us is None else threshold(compute_idle_us, True)
        page = None if page_idle_us is None else threshold(page_idle_us, False)
        return cls(reference, compute, page, **kwargs)

    def to_dict(self):
        return asdict(self)


FEATURE_NAMES = (
    "h2d_samples", "h2d_mean_us", "h2d_std_us", "h2d_min_us", "h2d_max_us",
    "h2d_median_us", "h2d_p90_us", "excess_mean_us", "excess_max_us", "above_threshold_samples",
    "transfer_batches", "transfer_duration_sum_ms", "transfer_duration_mean_ms", "transfer_duration_max_ms",
    "interbatch_gap_mean_ms", "interbatch_gap_max_ms", "compute_samples", "compute_mean_us",
    "compute_above_threshold_samples", "candidate_hit_fraction", "page_invocations", "page_mean_us",
    "page_std_us", "page_response_fraction", "page_response_transitions", "pressure_increment_batches",
    "pressure_cumulative_batches", "relative_window", "offset_ms", "transfer_active_fraction",
    "compute_active_fraction", "page_positive_responses",
)

_FORBIDDEN = {"block_id", "block_ids", "resident_blocks", "resident", "events", "snapshots", "reference",
              "reference_events", "victim_events", "victim_trace", "ground_truth", "prompt", "prompt_text",
              "input_ids", "prompt_tokens", "kv_tensor", "kv_values", "generated_response"}


def _check_isolation(value):
    if isinstance(value, dict):
        bad = _FORBIDDEN.intersection(value)
        if bad:
            raise ValueError(f"hardware observation contains forbidden fields: {sorted(bad)}")
        for child in value.values():
            _check_isolation(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _check_isolation(child)


def _samples(rows):
    pairs = sorted((time_ms(row), float(row["latency_us"])) for row in rows)
    if any(not math.isfinite(v) or v < 0 for _, v in pairs):
        raise ValueError("probe latencies must be finite and nonnegative")
    if any(a[0] == b[0] for a, b in zip(pairs, pairs[1:])):
        raise ValueError("probe timestamps must be unique within each channel")
    return np.array(pairs, dtype=float).reshape((-1, 2))


def _segments(samples, threshold, min_samples, interval, max_gap):
    segments, first, last, count = [], None, None, 0
    def finish():
        if count >= min_samples:
            segments.append({"start_ms": float(first), "end_ms": float(last + interval), "samples": count})
    for t, latency in samples:
        if latency > threshold:
            if first is None or t - last > max_gap:
                finish()
                first, count = t, 0
            last, count = t, count + 1
        else:
            finish()
            first, last, count = None, None, 0
    finish()
    return segments


def localize_hardware(observation: dict, calibration: Calibration) -> dict:
    """Find the first sustained H2D segment and a subsequent 20 ms idle gap.

    A capture ending before the idle gap is a failure, not an inferred completion.
    Ambiguity must be flagged using attacker-side scheduling/timing information.
    A candidate interval is not claimed to be the complete request lifetime.
    """
    _check_isolation(observation)
    if observation.get("ambiguous", False):
        raise LocalizationError("ambiguous request association")
    samples = _samples(observation.get("h2d", []))
    if not len(samples):
        raise LocalizationError("no H2D samples")
    submission = float(observation["submission_ms"])
    if not math.isfinite(submission):
        raise ValueError("submission timestamp must be finite")
    samples = samples[samples[:, 0] >= submission - calibration.submission_error_ms]
    gap = calibration.sample_interval_ms * calibration.max_sample_gap_factor
    segments = _segments(samples, calibration.h2d_threshold_us,
                         calibration.consecutive_samples, calibration.sample_interval_ms, gap)
    if not segments:
        raise LocalizationError("no sustained transfer activity after submission")
    start = segments[0]["start_ms"]
    idle_start, previous = None, None
    for t, latency in samples[samples[:, 0] >= segments[0]["end_ms"]]:
        if latency <= calibration.h2d_threshold_us:
            if idle_start is None or (previous is not None and t - previous > gap):
                idle_start = t
            if t - idle_start >= calibration.idle_gap_ms:
                end = float(idle_start)
                return {"start_ms": start, "end_ms": end,
                        "transfer_segments": [s for s in segments if s["start_ms"] < end]}
        else:
            idle_start = None
        previous = t
    raise LocalizationError("capture has no verified idle ending")


def encode_hardware(observation: dict, calibration: Calibration, window_ms=10.0,
                    max_windows=512, channels=("h2d", "compute", "page", "pressure")):
    """Encode 32 named features per window; hardware input never contains block IDs.

    Ablations zero omitted signal features. Localization always uses the H2D
    anchor required by the paper; it is unchanged across ablations.
    """
    allowed = {"h2d", "compute", "page", "pressure"}
    if not set(channels) <= allowed:
        raise ValueError("unknown hardware feature channel")
    bounds = localize_hardware(observation, calibration)
    start, end = bounds["start_ms"], bounds["end_ms"]
    capacity = observation["capacity_slots"]
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("capacity_slots must be a positive integer")
    result = allocate(start, end, float(window_ms), int(max_windows), FEATURE_NAMES, "hw")
    data = {name: _samples(observation.get(name, [])) for name in ("h2d", "compute", "page")}
    if len(data["compute"]) and "compute" in channels and calibration.compute_threshold_us is None:
        raise ValueError("computation threshold must be calibrated before inference")
    if len(data["page"]) and "page" in channels and calibration.page_threshold_us is None:
        raise ValueError("page threshold must be calibrated before inference")
    for name, samples in data.items():
        data[name] = samples[(samples[:, 0] >= start) & (samples[:, 0] <= end)]
    # A segment assigned by onset may span the truncation boundary. Its duration
    # must still describe only the retained observation horizon, rather than
    # reveal transfer activity from a later portion of the capture.
    retained_end = min(end, start + window_ms * max_windows)
    segments = [dict(s, end_ms=min(s["end_ms"], retained_end))
                for s in bounds["transfer_segments"] if s["start_ms"] <= retained_end]
    cumulative, previous_pressure = 0, 0
    n = int(result.mask.sum())
    for j in range(n):
        a, b = start + j * window_ms, min(end, start + (j + 1) * window_ms)
        def window_samples(name):
            s = data[name]
            return s[[window_index(t, start, window_ms) == j for t in s[:, 0]]]
        h2d, compute, page = [window_samples(name) for name in ("h2d", "compute", "page")]
        local_segments = [s for s in segments if window_index(s["start_ms"], start, window_ms) == j]
        cumulative += len(local_segments)
        pressure = max(0, cumulative - capacity)
        x = result.values[j]
        if "h2d" in channels:
            if len(h2d):
                v = h2d[:, 1]
                excess = v - calibration.baseline_h2d_us
                x[:10] = [len(v), v.mean(), v.std(), v.min(), v.max(), np.median(v), np.quantile(v, .9),
                          excess.mean(), excess.max(), np.sum(v > calibration.h2d_threshold_us)]
            x[10] = len(local_segments)
            durations = [s["end_ms"] - s["start_ms"] for s in local_segments]
            gaps = []
            for s in local_segments:
                idx = segments.index(s)
                if idx:
                    gaps.append(max(0., s["start_ms"] - segments[idx - 1]["end_ms"]))
            if durations:
                x[11:14] = [sum(durations), np.mean(durations), max(durations)]
            if gaps:
                x[14:16] = [np.mean(gaps), max(gaps)]
            overlap = sum(max(0., min(b, s["end_ms"]) - max(a, s["start_ms"])) for s in segments)
            x[29] = min(1., overlap / (b - a))
        if "compute" in channels and len(compute):
            active = compute[:, 1] > calibration.compute_threshold_us
            transfers = np.array([any(s["start_ms"] <= t < s["end_ms"] for s in segments) for t in compute[:, 0]])
            x[16:20] = [len(compute), compute[:, 1].mean(), active.sum(), np.mean(active & ~transfers)]
            x[30] = active.mean()
        if "page" in channels and len(page):
            response = page[:, 1] < calibration.page_threshold_us  # Eq. (13): lower, not higher.
            x[20:25] = [len(page), page[:, 1].mean(), page[:, 1].std(), response.mean(),
                        np.count_nonzero(np.diff(response.astype(int)))]
            x[31] = response.sum()
        if "pressure" in channels:
            x[25:27] = [pressure - previous_pressure, pressure]
        previous_pressure = pressure
        x[27:29] = [j / max(1, max_windows - 1), j * window_ms]
    return result


def hardware_activity_windows(observation, calibration, window_ms=10.0, max_windows=512):
    """Produce anonymous operation-related windows for external TRR scoring.

    LOAD denotes detected transfer batches; HIT denotes computation in statistical
    windows without detected transfers; EVICT denotes positive replacement-pressure
    increments. None of these intervals exposes logical KV-block identities.
    """
    bounds = localize_hardware(observation, calibration)
    encoded = encode_hardware(observation, calibration, window_ms, max_windows)
    end = min(bounds["end_ms"], bounds["start_ms"] + window_ms*max_windows)
    result = [{"op": "LOAD", "start_ms": s["start_ms"], "end_ms": min(s["end_ms"], end)}
              for s in bounds["transfer_segments"] if s["start_ms"] <= end]
    names = {name: i for i, name in enumerate(encoded.feature_names)}
    for j in range(int(encoded.mask.sum())):
        a, b = encoded.start_ms+j*window_ms, min(end, encoded.start_ms+(j+1)*window_ms)
        for op, field in (("HIT", "candidate_hit_fraction"), ("EVICT", "pressure_increment_batches")):
            if encoded.values[j, names[field]] > 0:
                result.append(dict(op=op, start_ms=a, end_ms=b))
    return sorted(result, key=lambda r: (r["start_ms"], r["end_ms"], r["op"]))


def observation_from_probe_rows(rows, submission_ms, capacity_slots, ambiguous=False):
    """Convert acquisition JSONL records to independent timestamped probe series."""
    result = {"submission_ms": submission_ms, "capacity_slots": capacity_slots,
              "ambiguous": ambiguous, "h2d": [], "compute": [], "page": []}
    for row in rows:
        _check_isolation(row)
        for name in ("h2d", "compute", "page"):
            if row.get(f"{name}_us") is not None:
                t = row.get(f"{name}_timestamp_ms", row.get("time_ms", row.get("timestamp_ms")))
                if t is None:
                    raise ValueError("probe sample has no timestamp")
                result[name].append({"time_ms": t, "latency_us": row[f"{name}_us"]})
    return result
