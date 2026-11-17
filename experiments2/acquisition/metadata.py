"""Trusted native-operation export. This module never chooses KV blocks or evicts them."""
import json
import math
import time
from collections import defaultdict
from pathlib import Path


class MetadataPublisher:
    """Publish an allowlisted view after actual native operations complete.

    The caller supplies relative monotonic timestamps in ms, or the publisher uses
    its own epoch. Snapshot blocks must already exist and have CPU backing copies.
    A consumer receives JSONL events, never tensors, prompt text or labels.
    """
    def __init__(self, destination, request_id, clock=time.perf_counter, sinks=()):
        self.path = Path(destination)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8")
        self.request_id = str(request_id)
        self.clock, self.epoch = clock, clock()
        self.resident = defaultdict(set)
        self.last_time = -1.0
        self.sinks = tuple(sinks)
        self.sequence = 0

    def _write(self, record, time_ms=None):
        stamp = (self.clock() - self.epoch) * 1000 if time_ms is None else float(time_ms)
        if not math.isfinite(stamp) or stamp < 0 or stamp < self.last_time:
            raise ValueError("Publisher timestamps must be finite, nonnegative and monotonic")
        record.update(request_id=self.request_id, time_ms=stamp,
                      event_id=f"{self.request_id}:{self.sequence}")
        self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()
        for sink in self.sinks:
            sink.write(dict(record))
        self.last_time = stamp
        self.sequence += 1

    @staticmethod
    def _index(value, field):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
        return value

    @staticmethod
    def _block(value):
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("Block IDs must be strings or nonnegative integers")
        if isinstance(value, int) and value < 0 or isinstance(value, str) and not value.strip():
            raise ValueError("Invalid logical block ID")
        return value

    def event(self, op, layer, block_id, step=0, time_ms=None):
        if op not in {"HIT", "LOAD", "EVICT"}:
            raise ValueError("Not a management event")
        layer, step = self._index(layer, "layer"), self._index(step, "step")
        block_id = self._block(block_id)
        resident = self.resident[layer]
        if op == "HIT" and block_id not in resident:
            raise ValueError("HIT requires known GPU residency")
        if op == "EVICT" and block_id not in resident:
            raise ValueError("EVICT requires known GPU residency")
        if op == "LOAD" and block_id in resident:
            raise ValueError("LOAD cannot denote resident reuse")
        self._write(dict(record_type="event", source="logs", kind="management",
                         op=op, layer=layer, block_id=block_id, step=step), time_ms)
        if op == "LOAD": resident.add(block_id)
        if op == "EVICT": resident.remove(block_id)

    def snapshot(self, layer, resident_blocks=None, source="telemetry", time_ms=None,
                 kind="management", excluded_blocks=()):
        if source not in {"telemetry", "shm"}:
            raise ValueError("Snapshot source must be telemetry or shm")
        if kind not in {"management", "initialization", "cleanup", "allocation"}:
            raise ValueError("Invalid snapshot kind")
        layer = self._index(layer, "layer")
        blocks = (set(self._block(block) for block in resident_blocks)
                  if resident_blocks is not None else set(self.resident[layer]))
        excluded = set(self._block(block) for block in excluded_blocks)
        self._write(dict(record_type="snapshot", source=source, layer=layer,
                         resident=sorted(blocks, key=str), kind=kind,
                         excluded_blocks=sorted(excluded, key=str)), time_ms)
        self.resident[layer] = blocks

    def close(self):
        self.stream.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


class BackendHooks:
    """Map observed native operations to Table 1 without introducing new policies.

    Install callbacks in the actual backend version after successful transfer/reuse/
    release. RetrievalAttention CPU retrieval and query/result transfers are ignored.
    Cleanup and new-token allocation are never reported as management eviction/load.
    """
    def __init__(self, publisher, system):
        if system not in {"quest", "infinigen", "retrievalattention"}:
            raise ValueError("Unknown backend")
        self.publisher, self.system = publisher, system

    def completed(self, operation, layer, block_ids, *, phase, step=0,
                  transfer_kind="kv", cpu_backed=True, cleanup=False, time_ms=None,
                  kind="management"):
        if kind not in {"management", "initialization", "allocation", "cleanup",
                        "cpu_attention", "query_transfer", "result_transfer"}:
            raise ValueError("Unknown operation kind")
        if cleanup or not cpu_backed or transfer_kind != "kv" or kind != "management":
            return
        if phase not in {"prefill", "decode"}:
            raise ValueError("Invalid phase")
        if self.system == "retrievalattention" and operation == "LOAD" and phase != "prefill":
            raise ValueError("Table 1 maps RetrievalAttention KV LOAD to fixed-context prefill")
        for block in block_ids:
            self.publisher.event(operation, layer, block, step, time_ms)


def read_export(path, request_id):
    events, snapshots = [], []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["request_id"] != str(request_id):
                continue
            if row.get("record_type") not in {"event", "snapshot"}:
                raise ValueError("Unknown exported record type")
            (events if row["record_type"] == "event" else snapshots).append(row)
    times = [r["time_ms"] for r in events + snapshots]
    return dict(request_id=str(request_id), events=events, snapshots=snapshots,
                start_ms=min(times, default=0), end_ms=max(times, default=0))
