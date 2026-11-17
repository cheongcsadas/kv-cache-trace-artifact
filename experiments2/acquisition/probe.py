"""Standalone CUDA-context acquisition for a locally controlled experiment.

The page probe owns separate CUDA VMM allocation handles. Independent workers
sample H2D, computation, and page activity against absolute monotonic deadlines.
"""
import argparse
import json
import math
import os
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class ProbeSettings:
    h2d_bytes: int = 1024
    probe_interval_ms: float = 1.0
    compute_shape: tuple = (256, 256)
    compute_dtype: str = "float16"
    probe_pages: int = 64
    probe_page_bytes: int = 2097152
    page_stride_bytes: int = 4096

    def __post_init__(self):
        for value in (self.h2d_bytes, self.probe_pages, self.probe_page_bytes, self.page_stride_bytes, *self.compute_shape):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Probe dimensions must be positive integers")
        if len(self.compute_shape) != 2 or self.compute_dtype != "float16":
            raise ValueError("Compute probe requires an FP16 two-dimensional matrix shape")
        if not math.isfinite(self.probe_interval_ms) or self.probe_interval_ms <= 0:
            raise ValueError("Probe interval must be positive and finite")
        if self.probe_page_bytes % self.page_stride_bytes:
            raise ValueError("Page scan stride must divide the page size")

    @classmethod
    def from_config(cls, values=None):
        values = values or {}
        return cls(**{name: tuple(values[name]) if name == "compute_shape" else values[name]
                      for name in cls.__dataclass_fields__ if name in values})


def completion_deadline(maximum_deadline_s, completion, *, now_s, idle_gap_ms, sample_interval_ms):
    """Bound capture lifetime while retaining an independently observed idle tail.

    Completion is a collection-control signal only. The detector still derives
    its own start/end interval from measured H2D samples. Two additional sampling
    intervals allow the first idle sample to fall after the completion signal.
    """
    values = (maximum_deadline_s, now_s, idle_gap_ms, sample_interval_ms)
    if (any(isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) for value in values)
            or idle_gap_ms < 0 or sample_interval_ms <= 0):
        raise ValueError("Capture deadline and tail parameters must be finite")
    if not isinstance(completion, dict) or completion.get("clock") != "perf_counter":
        raise ValueError("Capture completion and observer must use perf_counter")
    finished = completion.get("completion_monotonic_ms")
    if (isinstance(finished, bool) or not isinstance(finished, (int, float))
            or not math.isfinite(finished)):
        raise ValueError("Capture completion timestamp must be finite")
    tail_s = (idle_gap_ms + 2 * sample_interval_ms) / 1000
    return min(maximum_deadline_s, max(now_s, finished / 1000) + tail_s)


def capture(duration_s, capacity_slots, submission_monotonic_ms=None, device="cuda:0",
            page_probe=True, ready_file=None, anchor_file=None, observation_config=None, stop_file=None):
    if (not math.isfinite(duration_s) or duration_s <= 0 or isinstance(capacity_slots, bool)
            or not isinstance(capacity_slots, int) or capacity_slots < 1):
        raise ValueError("Positive duration and capacity required")
    settings = ProbeSettings.from_config(observation_config)
    stop_path = Path(stop_file) if stop_file else None
    if stop_path is not None and stop_path.exists():
        raise FileExistsError(stop_path)
    idle_gap_ms = (observation_config or {}).get("idle_gap_ms", 20.0)
    if (isinstance(idle_gap_ms, bool) or not isinstance(idle_gap_ms, (int, float))
            or not math.isfinite(idle_gap_ms) or idle_gap_ms <= 0):
        raise ValueError("Idle-gap duration must be positive and finite")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for measured capture; no synthetic fallback")
    torch.cuda.set_device(device)
    host = torch.zeros(settings.h2d_bytes, dtype=torch.uint8, pin_memory=True)
    dest = torch.empty(settings.h2d_bytes, dtype=torch.uint8, device=device)
    m, k = settings.compute_shape
    a = torch.ones((m, k), dtype=torch.float16, device=device)
    b = torch.ones((k, m), dtype=torch.float16, device=device)
    c = torch.empty((m, m), dtype=torch.float16, device=device)
    streams = {key: torch.cuda.Stream() for key in ("h2d", "compute", "page")}
    pairs = {key: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for key in streams}
    preparation_done = torch.cuda.Event() if page_probe else None
    if page_probe:
        from torch.utils.cpp_extension import load
        extension = load(name="kvtrail_page_scan_vmm", sources=[str(Path(__file__).with_name("page_probe.cu"))],
                         extra_cuda_cflags=["-O2"], extra_ldflags=["cuda.lib" if os.name == "nt" else "-lcuda"],
                         verbose=False)
        pages, granularity = extension.allocate_pages(settings.probe_pages, settings.probe_page_bytes,
                                                      torch.cuda.current_device())
        auxiliary, _ = extension.allocate_pages(settings.probe_pages, settings.probe_page_bytes,
                                               torch.cuda.current_device())
        checksum = torch.zeros(1, dtype=torch.int64, device=device)
    # Compile/allocate/warm up before publishing the measurement epoch.
    for _ in range(3):
        dest.copy_(host, non_blocking=True)
        torch.mm(a, b, out=c)
        if page_probe: extension.scan(pages, checksum, settings.page_stride_bytes)
    torch.cuda.synchronize()
    epoch_ms = time.perf_counter() * 1000
    result = dict(h2d=[], compute=[], page=[], provenance="measured", capacity_slots=capacity_slots,
                  submission_ms=0.0 if submission_monotonic_ms is None else submission_monotonic_ms-epoch_ms,
                  epoch_monotonic_ms=epoch_ms, clock="perf_counter", ambiguous=False,
                  sample_time_anchor="cpu_submission_after_page_preparation",
                  latency_clock="cuda_events",
                  acquisition_parameters=asdict(settings),
                  page_probe_implementation="cuda-vmm-separate-pages-stride4096" if page_probe else "disabled")
    if page_probe:
        result["page_allocation"] = dict(handles=settings.probe_pages, bytes_per_handle=settings.probe_page_bytes,
                                          minimum_granularity_bytes=granularity, auxiliary_handles=settings.probe_pages)
    deadline = time.perf_counter() + duration_s
    errors, stop = [], threading.Event()
    gate = threading.Event()
    channels = [channel for channel in ("h2d", "compute", "page") if channel != "page" or page_probe]
    startup = threading.Barrier(len(channels) + 1)
    first_samples = {channel: threading.Event() for channel in channels}
    cadence = settings.probe_interval_ms / 1000
    result["scheduling"] = {}

    def observe(channel):
        try:
          torch.cuda.set_device(device)
          stream = streams[channel]
          start, end = pairs[channel]
          startup.wait(timeout=30)
          gate.wait()
          next_tick, missed = epoch_ms / 1000, 0
          while not stop.is_set() and time.perf_counter() < deadline:
            now = time.perf_counter()
            if now < next_tick:
                stop.wait(next_tick - now)
            if stop.is_set() or time.perf_counter() >= deadline:
                break
            if channel == "page":
                with torch.cuda.stream(stream):
                    checksum.zero_()
                    extension.scan(auxiliary, checksum, settings.page_stride_bytes)
                    preparation_done.record(stream)
                # Preparation is excluded from both the timed invocation and its
                # CPU submission timestamp, even when GPU work queues are busy.
                preparation_done.synchronize()
                if stop.is_set() or time.perf_counter() >= deadline:
                    break
            stamp = time.perf_counter()*1000-epoch_ms
            with torch.cuda.stream(stream):
                start.record(stream)
                if channel == "h2d": dest.copy_(host, non_blocking=True)
                elif channel == "compute": torch.mm(a, b, out=c)
                else: extension.scan(pages, checksum, settings.page_stride_bytes)
                end.record(stream)
            end.synchronize()
            result[channel].append(dict(time_ms=stamp, latency_us=start.elapsed_time(end)*1000))
            first_samples[channel].set()
            next_tick += cadence
            now = time.perf_counter()
            if next_tick < now:
                skipped = math.floor((now - next_tick) / cadence) + 1
                missed += skipped
                next_tick += skipped * cadence
          result["scheduling"][channel] = dict(missed_deadlines=missed, samples=len(result[channel]))
        except BaseException as exc:
            errors.append(exc)
            stop.set()

    workers = [threading.Thread(target=observe, args=(channel,), name=f"probe-{channel}")
               for channel in channels]
    for worker in workers:
        worker.start()
    try:
        startup.wait(timeout=30)
        epoch_ms = time.perf_counter() * 1000
        deadline = epoch_ms / 1000 + duration_s
        result["epoch_monotonic_ms"] = epoch_ms
        result["submission_ms"] = 0.0 if submission_monotonic_ms is None else submission_monotonic_ms - epoch_ms
        gate.set()
        for channel in channels:
            if not first_samples[channel].wait(timeout=min(30, duration_s)):
                raise RuntimeError(f"No completed {channel} sample before observer readiness")
        if ready_file:
            ready = Path(ready_file)
            ready.parent.mkdir(parents=True, exist_ok=True)
            if ready.exists():
                raise FileExistsError(ready)
            pending = ready.with_suffix(ready.suffix + ".pending")
            with pending.open("x", encoding="utf-8") as stream:
                json.dump(dict(epoch_monotonic_ms=epoch_ms, clock="perf_counter"), stream)
            pending.replace(ready)
    except BaseException as exc:
        errors.append(exc)
        stop.set()
    finally:
        gate.set()
    # A completed request schedule may shorten the maximum capture, but no
    # completion timestamp is exported to the observation or detector. With no
    # control file (including idle calibration), the entire duration is sampled.
    completion_seen = False
    try:
        while any(worker.is_alive() for worker in workers):
            if stop_path is not None and not completion_seen and stop_path.exists():
                completion = json.loads(stop_path.read_text(encoding="utf-8"))
                deadline = completion_deadline(deadline, completion, now_s=time.perf_counter(),
                    idle_gap_ms=idle_gap_ms, sample_interval_ms=settings.probe_interval_ms)
                completion_seen = True
            # A bounded join also yields when an acquisition worker fails.
            for worker in workers:
                worker.join(timeout=.005)
    except BaseException as exc:
        errors.append(exc)
        stop.set()
    finally:
        for worker in workers:
            worker.join()
    if errors:
        raise RuntimeError("CUDA probe acquisition failed") from errors[0]
    result["end_ms"] = time.perf_counter()*1000-epoch_ms
    if anchor_file:
        anchor = json.loads(Path(anchor_file).read_text(encoding="utf-8"))
        if anchor.get("clock", "perf_counter") != "perf_counter":
            raise ValueError("Request anchor and observer must use the same perf_counter clock")
        result["submission_ms"] = float(anchor["submission_monotonic_ms"])-epoch_ms
        result["ambiguous"] = bool(anchor.get("ambiguous", False))
    return result


def empirical_threshold(values, upper=True, fpr=.05):
    """Choose a strict comparison threshold with <=fpr baseline positives."""
    if not values or not 0 <= fpr < 1:
        raise ValueError("Calibration samples required")
    values = sorted(float(v) for v in values)
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Invalid calibration latency")
    allowed = math.floor(len(values)*fpr)
    return values[len(values)-allowed-1] if upper else values[allowed]


def calibrate(repetitions, observation_config=None):
    from attacks.hardware import Calibration
    profile = observation_config or {}
    if len(repetitions) != 3:
        raise ValueError("Appendix D requires three independent baseline repetitions")
    for r in repetitions:
        if r.get("provenance") != "measured":
            raise ValueError("Calibration requires measured baselines")
    if profile.get("reference_repetitions", 3) != 3:
        raise ValueError("The calibration profile requires three baseline repetitions")
    settings = ProbeSettings.from_config(profile)
    for r in repetitions:
        if "acquisition_parameters" in r and ProbeSettings.from_config(r["acquisition_parameters"]) != settings:
            raise ValueError("Calibration capture parameters do not match the configured probes")
    h2d = [[p["latency_us"] for p in r["h2d"]] for r in repetitions]
    compute = [p["latency_us"] for r in repetitions for p in r["compute"]]
    page = [p["latency_us"] for r in repetitions for p in r["page"]]
    return Calibration.fit(h2d, compute or None, page or None,
        empirical_fpr=profile.get("empirical_fpr", .05),
        h2d_threshold_us=profile.get("h2d_threshold_us", 200.0),
        consecutive_samples=profile.get("consecutive_samples", 3),
        sample_interval_ms=settings.probe_interval_ms,
        idle_gap_ms=profile.get("idle_gap_ms", 20.0),
        submission_error_ms=profile.get("submission_error_max_ms", 1.0)).to_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cap = commands.add_parser("capture")
    cap.add_argument("--output", required=True)
    cap.add_argument("--seconds", type=float, default=8)
    cap.add_argument("--capacity-slots", type=int, required=True)
    cap.add_argument("--submission-monotonic-ms", type=float)
    cap.add_argument("--device", default="cuda:0")
    cap.add_argument("--no-page-probe", action="store_true")
    cap.add_argument("--ready-file", help="Created after compilation and warmup; client waits for this file")
    cap.add_argument("--anchor-file", help="Client-written submission_monotonic_ms on the same host")
    cap.add_argument("--stop-file", help="Optional schedule completion signal; retains an idle sampling tail")
    cap.add_argument("--config", help="System YAML controlling probe sizes, cadence and detector parameters")
    cap.add_argument("--sweep", choices=["block_tokens", "residency_fraction", "context_tokens", "training_per_domain", "concurrent_workloads"])
    cal = commands.add_parser("calibrate")
    cal.add_argument("--baselines", nargs=3, required=True)
    cal.add_argument("--output", required=True)
    cal.add_argument("--config", help="System YAML matching the baseline capture configuration")
    cal.add_argument("--sweep", choices=["block_tokens", "residency_fraction", "context_tokens", "training_per_domain", "concurrent_workloads"])
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists(): raise FileExistsError(path)
    if args.config:
        from experiments.config import load_config
        from experiments.paper_protocol import validate_paper_config
        configuration = load_config(args.config)
        validate_paper_config(configuration, allowed_sweep=args.sweep)
        configured_slots = configuration["cache"]["capacity_slots"]
        if args.command == "capture" and configured_slots is not None and args.capacity_slots != configured_slots:
            raise ValueError("Probe capacity_slots must match the configured native GPU KV pool")
        profile = configuration["observation"]
    else: profile = None
    if args.command == "capture":
        result = capture(args.seconds, args.capacity_slots, args.submission_monotonic_ms,
                         args.device, not args.no_page_probe, args.ready_file, args.anchor_file, profile,
                         args.stop_file)
    else:
        result = calibrate([json.loads(Path(p).read_text(encoding="utf-8")) for p in args.baselines], profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream: json.dump(result, stream, indent=2)


if __name__ == "__main__": main()
