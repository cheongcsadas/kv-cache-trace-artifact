"""Collect local inference timings and observations with asynchronous request arrivals.

The local inference service runs the selected native backend. Payloads stay in
the HTTP request path; GPU observations contain only probe measurements.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import http.client
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse


def _endpoint(value):
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Collection endpoint must be the inference service on the same host")
    if parsed.username or parsed.password: raise ValueError("Do not embed credentials in the endpoint")
    return value


def run_schedule(endpoint, requests, concurrency, arrival_offsets_ms, *, timeout_s=120,
                 target_id=None, anchor_path=None, ambiguous=False, collect_receipts=False):
    """Send real inference requests at declared, unsynchronized arrival offsets.

    requests: [{sample_id, payload}]. Returns completed/error attempts in input
    order. Payloads and response contents are excluded from timing records.
    """
    endpoint = _endpoint(endpoint)
    if concurrency not in (1, 4, 8, 16): raise ValueError("Paper concurrency levels are 1,4,8,16")
    if not requests or len(requests) != len(arrival_offsets_ms):
        raise ValueError("Each request needs an arrival offset")
    ids = [r["sample_id"] for r in requests]
    if len(ids) != len(set(ids)) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("Unique nonempty sample IDs required")
    if any(not isinstance(r.get("payload"), dict) for r in requests):
        raise ValueError("Each request needs its native HTTP payload")
    offsets = [float(x) for x in arrival_offsets_ms]
    if any(not math.isfinite(x) or x < 0 for x in offsets): raise ValueError("Invalid arrival offset")
    if not math.isfinite(timeout_s) or timeout_s <= 0: raise ValueError("Invalid timeout")
    if anchor_path and ids.count(target_id) != 1: raise ValueError("Exactly one target ID is required")
    if anchor_path and Path(anchor_path).exists(): raise FileExistsError(anchor_path)
    if anchor_path: Path(anchor_path).parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(endpoint)
    start = time.perf_counter()
    active = 0
    lock = threading.Lock()

    def send(index):
        nonlocal active
        delay = start+offsets[index]/1000-time.perf_counter()
        if delay > 0: time.sleep(delay)
        row = requests[index]
        # Serialization precedes submission time so it is not mistaken for network latency.
        body = json.dumps(row["payload"]).encode("utf-8")
        stamp = time.perf_counter()
        active_now, was_active = 0, False
        connection = None
        send_end = None
        anchor_worker, anchor_errors = None, []
        def publish_anchor(uncertainty):
            try:
                anchor = Path(anchor_path)
                pending = anchor.with_suffix(anchor.suffix+".pending")
                with pending.open("x", encoding="utf-8") as stream:
                    json.dump(dict(submission_monotonic_ms=stamp*1000, clock="perf_counter",
                        submission_uncertainty_ms=uncertainty,
                        ambiguous=ambiguous or uncertainty is None or uncertainty > 1.0), stream)
                pending.replace(anchor)
            except Exception as exc:
                anchor_errors.append(type(exc).__name__)
        status, error, http_status = "completed", None, None
        native_receipt = None
        try:
            connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
            connection = connection_class(parsed.hostname, parsed.port, timeout=timeout_s)
            # Connection establishment and JSON serialization precede the actual
            # submission anchor. No anchor-file I/O occurs before the socket send.
            connection.connect()
            path = parsed.path or "/"
            if parsed.query: path += "?"+parsed.query
            connection.putrequest("POST", path)
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(len(body)))
            connection.putheader("X-KvTrail-Request-ID", row["sample_id"])
            with lock:
                active += 1
                active_now, was_active = active, True
            stamp = time.perf_counter()
            connection.endheaders(body)
            send_end = time.perf_counter()
            if anchor_path and row["sample_id"] == target_id:
                anchor_worker = threading.Thread(target=publish_anchor, args=((send_end-stamp)*1000,))
                anchor_worker.start()
            response = connection.getresponse()
            http_status = response.status
            if collect_receipts:
                payload = response.read(8 * 1024 * 1024 + 1)
                if len(payload) > 8 * 1024 * 1024:
                    raise ValueError("Native receipt exceeds size limit")
                native_receipt = json.loads(payload)
                if not isinstance(native_receipt, dict):
                    raise ValueError("Native response must be a receipt object")
            else:
                while response.read(65536): continue
            if not 200 <= http_status < 300:
                status, error = "failed", f"HTTP{http_status}"
        except Exception as exc:
            status, error = "failed", type(exc).__name__
        finally:
            finished = time.perf_counter()
            if connection is not None: connection.close()
            with lock:
                if was_active: active -= 1
        uncertainty = (send_end-stamp)*1000 if send_end is not None else None
        if anchor_worker:
            anchor_worker.join()
        elif anchor_path and row["sample_id"] == target_id:
            publish_anchor(uncertainty)
        if anchor_errors:
            status, error = "failed", "AnchorExport"+anchor_errors[0]
        record = dict(sample_id=row["sample_id"], planned_offset_ms=offsets[index],
            submission_ms=(stamp-start)*1000, completion_ms=(finished-start)*1000,
            submission_monotonic_ms=stamp*1000, completion_monotonic_ms=finished*1000,
            latency_ms=(finished-stamp)*1000, concurrency=concurrency,
            active_at_submission=active_now, status=status, error=error, http_status=http_status,
            submission_uncertainty_ms=uncertainty,
            clock="perf_counter", provenance="measured")
        if collect_receipts:
            record["native_receipt"] = native_receipt
        return index, record

    result = [None]*len(requests)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        # Sort submission order to avoid a future-arrival request occupying all workers.
        futures = [pool.submit(send, index) for index in sorted(range(len(requests)), key=lambda i: offsets[i])]
        for future in as_completed(futures):
            index, result[index] = future.result()
    return result


def native_describe(endpoint, payload):
    """Tokenize on the loaded native backend before starting the observer."""
    parsed = urlparse(_endpoint(endpoint))
    connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_class(parsed.hostname, parsed.port, timeout=120)
    try:
        connection.request("POST", "/describe", json.dumps(payload), {"Content-Type": "application/json"})
        response = connection.getresponse()
        result = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(f"Native describe failed: {result}")
        return result
    finally:
        connection.close()


def _validate_native_receipt(receipt, description, config, request_id):
    if not isinstance(receipt, dict):
        raise ValueError("Inference did not return a native receipt")
    if receipt.get("request_id") != request_id:
        raise ValueError("Native runtime returned a different request identity")
    if receipt.get("system") != config["system"]:
        raise ValueError("Native runtime returned a different system")
    for field in ("capacity_slots", "token_count"):
        if receipt.get(field) != description.get(field):
            raise ValueError(f"Actual backend {field} differs from preflight allocation")
    for field in ("block_tokens", "residency_fraction"):
        if receipt.get(field) != config["cache"][field]:
            raise ValueError(f"Actual backend {field} differs from requested setting")
    if not receipt.get("metadata_exports") or not receipt.get("reference_path"):
        raise ValueError("Native operation exports were not returned")
    expected_defense = config.get("experiment", {}).get("defense") or "none"
    if (receipt.get("defense") or "none") != expected_defense:
        raise ValueError("Actual backend defense differs from the requested condition")


def collect(manifest, output):
    """Run a target attempt, concurrent workload schedule, and selected acquisition."""
    from .config import load_config
    from .paper_protocol import validate_paper_config
    from .acquisition.consumer import collect_exports
    manifest_path = Path(manifest).resolve()
    spec = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    config_path = (base/spec["config"]).resolve()
    config = load_config(config_path)
    validate_paper_config(config, allowed_sweep=spec.get("sweep"))
    channel = spec["channel"]
    if channel not in {"meta", "hw"}: raise ValueError("channel must be meta or hw")
    _endpoint(spec["endpoint"])
    target_id = spec["target_id"]
    target = [row for row in spec["requests"] if row["sample_id"] == target_id]
    if len(target) != 1: raise ValueError("Exactly one target request is required")
    target = target[0]
    if "annotation" in target:
        from datasets.preprocessing.grounding import validate_grounding_receipt
        annotation = validate_grounding_receipt(target)
        prompt = target.get("payload", {}).get("prompt")
        if not isinstance(prompt, str) or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != target["prompt_sha256"]:
            raise ValueError("Collected payload differs from the grounded prompt")
        if any(prompt[item["start"]:item["end"]] != item["text"]
               for item in annotation["prompt_evidence"].values()):
            raise ValueError("Collected payload differs from its semantic evidence")
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if spec.get("restart_command"):
        command = spec["restart_command"]
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            raise ValueError("restart_command must be an argument array")
        subprocess.run(command, check=True, cwd=base, timeout=spec.get("restart_timeout_s", 120),
                       stdout=subprocess.DEVNULL)
    native = bool(spec.get("native"))
    description = native_describe(spec["endpoint"], target["payload"]) if native else None
    if native and spec.get("concurrency", 1) > 1 and not description.get("concurrent_requests"):
        raise ValueError("The selected native runtime cannot execute concurrent requests")
    process = None
    ready, anchor, stop_file = out/"ready.json", out/"anchor.json", out/"capture-stop.json"
    setup_start = time.perf_counter()
    try:
        if channel == "hw":
            capacity = description["capacity_slots"] if native else config["cache"]["capacity_slots"]
            if not isinstance(capacity, int) or capacity < 1: raise ValueError("Set actual capacity_slots")
            seconds = float(spec.get("capture_seconds", 8))
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("capture_seconds must be positive and finite")
            start_timeout = float(spec.get("probe_start_timeout_s", 180))
            if not math.isfinite(start_timeout) or start_timeout <= 0:
                raise ValueError("probe_start_timeout_s must be positive and finite")
            command = [spec.get("probe_python", sys.executable), "-m", "experiments.acquisition.probe", "capture", "--config", str(config_path),
                       "--seconds", str(seconds), "--capacity-slots", str(capacity),
                       "--ready-file", str(ready), "--anchor-file", str(anchor),
                       "--stop-file", str(stop_file),
                       "--output", str(out/"observation.json")]
            command.extend(["--device", spec.get("device", "cuda:0")])
            if spec.get("sweep"):
                command.extend(["--sweep", spec["sweep"]])
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1])
            deadline = time.perf_counter()+start_timeout
            while not ready.exists():
                if process.poll() is not None: raise RuntimeError("Probe exited before ready")
                if time.perf_counter() >= deadline: raise TimeoutError("Probe warmup timed out")
                time.sleep(.01)
        acquisition_start = time.perf_counter()
        setup_ms = (acquisition_start-setup_start)*1000
        schedule = run_schedule(spec["endpoint"], spec["requests"], spec.get("concurrency", 1),
            spec["arrival_offsets_ms"], timeout_s=spec.get("request_timeout_s", 120),
            target_id=target_id, anchor_path=anchor if channel == "hw" else None,
            ambiguous=spec.get("ambiguous", False), collect_receipts=native)
        if process is not None:
            pending_stop = stop_file.with_suffix(".pending")
            with pending_stop.open("x", encoding="utf-8") as stream:
                json.dump(dict(clock="perf_counter", completion_monotonic_ms=
                               max(row["completion_monotonic_ms"] for row in schedule)), stream)
            pending_stop.replace(stop_file)
        selected = next(r for r in schedule if r["sample_id"] == target_id)
        receipt = selected.get("native_receipt") if native else None
        acquisition_status, failure_reason = "completed", None
        try:
            if native:
                if selected["status"] != "completed":
                    raise RuntimeError(f"Native request failed: {receipt}")
                _validate_native_receipt(receipt, description, config, target_id)
            if process:
                process.wait(timeout=float(spec.get("capture_seconds", 8))+30)
                if process.returncode: raise RuntimeError("Probe process failed")
            else:
                exports = receipt if native else spec
                observation = collect_exports([base/name for name in exports.get("metadata_exports", [])],
                    [base/name for name in exports.get("metadata_shm_exports", [])], request_id=target_id)
                observation["provenance"] = "measured"
                (out/"observation.json").write_text(json.dumps(observation), encoding="utf-8")
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            acquisition_status, failure_reason = "failed", str(exc)
        # Keep truth separate from acquisition success. A failed observer or a
        # generation error after some operations must not erase those events
        # from the recovery denominator.
        if native and isinstance(receipt, dict):
            (out/"native-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            if receipt.get("reference_path"):
                try:
                    reference = json.loads(Path(receipt["reference_path"]).read_text(encoding="utf-8"))
                    if channel == "hw":
                        epoch_path = out/"observation.json" if (out/"observation.json").exists() else ready
                        observed = json.loads(epoch_path.read_text(encoding="utf-8"))
                        shift = receipt["epoch_monotonic_ms"] - observed["epoch_monotonic_ms"]
                        for event in reference["events"]:
                            event["time_ms"] += shift
                    (out/"reference.json").write_text(json.dumps(reference), encoding="utf-8")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    acquisition_status = "failed"
                    failure_reason = (failure_reason + "; " if failure_reason else "") + "Reference export: " + str(exc)
        elapsed = (time.perf_counter()-acquisition_start)*1000
        row = {key: target[key] for key in ("sample_id", "source_id", "support_source_ids", "source_spans",
                   "prompt_sha256", "domain", "labels", "template_id", "token_count", "annotation") if key in target}
        row.update(provenance="measured", system=config["system"],
                   defense=config.get("experiment", {}).get("defense") or "none",
                   collection_ms=elapsed, latency_ms=selected["latency_ms"], request_status=selected["status"],
                   acquisition_status=acquisition_status, failure_reason=failure_reason, setup_ms=setup_ms,
                   concurrency=spec.get("concurrency", 1), schedule_path="schedule.jsonl")
        if (out/"observation.json").exists():
            row["observation_path"] = "observation.json"
            row["observation_sha256"] = hashlib.sha256((out/"observation.json").read_bytes()).hexdigest()
        if (out/"reference.json").exists(): row["reference_path"] = "reference.json"
        if native:
            row["token_count"] = description["token_count"]
            row["native_receipt_path"] = "native-receipt.json" if (out/"native-receipt.json").exists() else None
        (out/"attempt.jsonl").write_text(json.dumps(row)+"\n", encoding="utf-8")
        (out/"schedule.jsonl").write_text("".join(json.dumps(r)+"\n" for r in schedule), encoding="utf-8")
        return row
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = collect(args.manifest, args.output)
    print(json.dumps(dict(sample_id=result["sample_id"], status=result["request_status"])))


if __name__ == "__main__": main()
