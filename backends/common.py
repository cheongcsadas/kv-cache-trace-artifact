"""Actual-operation exports shared by the native runtimes.

The reference stream stays on the evaluator side. Telemetry samples the resident
state independently of the log transport; shared-memory readers receive only
the sampled allowlisted state, never tensor storage or prompt contents.
"""
from collections import defaultdict
import json
from pathlib import Path
import threading
import time

from experiments.acquisition.metadata import MetadataPublisher
from experiments.acquisition.consumer import MmapMetadataWriter


class NativeExport:
    def __init__(self, export_dir, request_id, *, sample_ms=10.0,
                 capacity_bytes=64 * 1024 * 1024):
        if sample_ms <= 0:
            raise ValueError("Telemetry sampling period must be positive")
        self.directory = Path(export_dir).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.request_id = str(request_id)
        self.sample_ms = float(sample_ms)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._resident = defaultdict(set)
        self._sequence = 0
        self._first_reference = True
        self._closed = False
        self._error = None
        self._telemetry = (self.directory / "telemetry.jsonl").open("x", encoding="utf-8")
        self._reference = (self.directory / "reference.json").open("x", encoding="utf-8")
        self._reference.write('{"events":[')
        self._mmap = MmapMetadataWriter(self.directory / "residency.mmap", capacity_bytes)
        self.publisher = MetadataPublisher(self.directory / "events.jsonl", request_id, sinks=(self,))
        self.epoch_monotonic_ms = self.publisher.epoch * 1000
        self._thread = threading.Thread(target=self._sample_loop, name="kv-telemetry", daemon=True)
        self._thread.start()

    def write(self, record):
        """Receive a completed native operation from MetadataPublisher."""
        with self._lock:
            if self._error:
                raise RuntimeError("Metadata sampling failed") from self._error
            layer = record["layer"]
            if record["record_type"] == "event":
                if layer not in self._resident:
                    # The first LOAD starts an empty request-local bank. Preserve
                    # that baseline before the first sampling tick, including
                    # backends whose prefill emits events without a snapshot.
                    self._snapshot(layer, 0.0, "initialization")
                if not self._first_reference:
                    self._reference.write(",")
                self._reference.write(json.dumps(record, allow_nan=False))
                self._reference.flush()
                self._first_reference = False
                if record["op"] == "LOAD":
                    self._resident[layer].add(record["block_id"])
                elif record["op"] == "EVICT":
                    self._resident[layer].discard(record["block_id"])
            else:
                self._resident[layer] = set(record["resident"])
                # Preserve initialization/cleanup causes so snapshot differencing
                # cannot misclassify allocation or teardown as LOAD/EVICT.
                self._snapshot(layer, record["time_ms"], record.get("kind", "management"),
                               record.get("excluded_blocks", ()))

    def _snapshot(self, layer, stamp, kind="management", excluded=()):
        base = dict(record_type="snapshot", request_id=self.request_id, layer=layer,
                    time_ms=stamp, resident=sorted(self._resident[layer], key=str),
                    kind=kind, excluded_blocks=list(excluded))
        for source in ("telemetry", "shm"):
            row = dict(base, source=source,
                       event_id=f"{self.request_id}:{source}:{self._sequence}")
            if source == "telemetry":
                self._telemetry.write(json.dumps(row, allow_nan=False) + "\n")
                self._telemetry.flush()
            else:
                self._mmap.write(row)
        self._sequence += 1

    def _sample(self):
        with self._lock:
            stamp = (time.perf_counter() - self.publisher.epoch) * 1000
            for layer in sorted(self._resident):
                self._snapshot(layer, stamp)

    def _sample_loop(self):
        try:
            while not self._stop.wait(self.sample_ms / 1000):
                self._sample()
        except Exception as exc:
            self._error = exc
            self._stop.set()

    def receipt(self):
        return dict(metadata_exports=[str(self.directory / "events.jsonl"),
                                      str(self.directory / "telemetry.jsonl")],
                    metadata_shm_exports=[str(self.directory / "residency.mmap")],
                    reference_path=str(self.directory / "reference.json"),
                    epoch_monotonic_ms=self.epoch_monotonic_ms,
                    telemetry_sample_ms=self.sample_ms)

    def close(self):
        if self._closed:
            return
        self._stop.set()
        self._thread.join()
        try:
            if not self._error:
                self._sample()
        finally:
            self.publisher.close()
            self._reference.write("]}\n")
            self._reference.close()
            self._telemetry.close()
            self._mmap.close()
            self._closed = True
        if self._error:
            raise RuntimeError("Metadata sampling failed") from self._error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as closing_error:
            closing_error.native_export_receipt = self.receipt()
            raise
        if exc is not None:
            # An unsuccessful generation still has evaluator-side ground truth
            # for the operations completed before it failed.
            exc.native_export_receipt = self.receipt()
