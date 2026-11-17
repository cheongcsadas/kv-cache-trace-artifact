"""Read designated metadata exports through logs, telemetry files and shared mmap.

The mmap channel is an append-only single-writer stream. Readers open it read-only;
the committed offset is advanced only after the full record has been flushed.
"""
import argparse
import json
import math
import mmap
from pathlib import Path
import struct
import time

MAGIC = b"KVTRAIL1"
HEADER_SIZE = 64
HEADER = struct.Struct("<8sQQQ")  # magic, committed offset, sequence, closed
LENGTH = struct.Struct("<Q")
ALLOWED = {"record_type", "source", "kind", "request_id", "time_ms", "event_id", "layer",
           "block_id", "step", "op", "resident", "resident_blocks", "excluded_blocks", "cause"}


def export_record(record):
    if not isinstance(record, dict) or set(record)-ALLOWED:
        raise ValueError("Metadata transport accepts only the designated exported fields")
    if record.get("record_type") not in {"event", "snapshot"}:
        raise ValueError("Unknown metadata record type")
    if not isinstance(record.get("request_id"), str) or not record["request_id"]:
        raise ValueError("Metadata record requires an opaque request_id")
    stamp = float(record["time_ms"])
    if not math.isfinite(stamp) or stamp < 0:
        raise ValueError("Invalid metadata timestamp")
    return dict(record)


class MmapMetadataWriter:
    """Trusted publisher transport; fixed capacity prevents unbounded allocation."""
    def __init__(self, path, capacity_bytes=16*1024*1024):
        if not isinstance(capacity_bytes, int) or capacity_bytes <= HEADER_SIZE:
            raise ValueError("Invalid shared export capacity")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x+b")
        self.stream.truncate(capacity_bytes)
        self.buffer = mmap.mmap(self.stream.fileno(), capacity_bytes, access=mmap.ACCESS_WRITE)
        self.offset, self.sequence, self.closed = HEADER_SIZE, 0, False
        self.buffer[:HEADER.size] = HEADER.pack(MAGIC, self.offset, self.sequence, 0)
        self.buffer.flush()

    def write(self, record):
        if self.closed: raise ValueError("Shared export is closed")
        payload = json.dumps(export_record(record), ensure_ascii=False, allow_nan=False).encode("utf-8")
        end = self.offset+LENGTH.size+len(payload)
        if end > len(self.buffer): raise BufferError("Shared export capacity exceeded")
        self.buffer[self.offset:end] = LENGTH.pack(len(payload))+payload
        self.buffer.flush()
        self.offset, self.sequence = end, self.sequence+1
        self.buffer[:HEADER.size] = HEADER.pack(MAGIC, self.offset, self.sequence, 0)
        self.buffer.flush()

    def close(self):
        if self.closed: return
        self.buffer[:HEADER.size] = HEADER.pack(MAGIC, self.offset, self.sequence, 1)
        self.buffer.flush()
        self.buffer.close()
        self.stream.close()
        self.closed = True

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


class MmapMetadataReader:
    def __init__(self, path):
        self.stream = Path(path).open("rb")
        self.buffer = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
        self.offset, self.closed = HEADER_SIZE, False
        if len(self.buffer) < HEADER_SIZE or self.buffer[:8] != MAGIC:
            self.close()
            raise ValueError("Invalid metadata mmap header")

    def read_available(self):
        magic, committed, _sequence, closed = HEADER.unpack(self.buffer[:HEADER.size])
        if magic != MAGIC or committed < self.offset or committed > len(self.buffer):
            raise ValueError("Corrupt shared export cursor")
        rows = []
        while self.offset < committed:
            if self.offset+LENGTH.size > committed: raise ValueError("Incomplete shared record length")
            size, = LENGTH.unpack(self.buffer[self.offset:self.offset+LENGTH.size])
            end = self.offset+LENGTH.size+size
            if end > committed: raise ValueError("Incomplete committed shared record")
            payload = self.buffer[self.offset+LENGTH.size:end]
            rows.append(export_record(json.loads(payload)))
            self.offset = end
        self.closed = bool(closed)
        return rows

    def close(self):
        self.buffer.close()
        self.stream.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


class JsonlMetadataReader:
    """Tail complete newline-terminated export records without consuming partial writes."""
    def __init__(self, path):
        self.stream = Path(path).open("rb")

    def read_available(self):
        rows = []
        while True:
            position = self.stream.tell()
            line = self.stream.readline()
            if not line: break
            if not line.endswith(b"\n"):
                self.stream.seek(position)
                break
            if line.strip(): rows.append(export_record(json.loads(line)))
        return rows

    def close(self): self.stream.close()


def collect_exports(log_paths=(), shm_paths=(), *, request_id, seconds=0,
                    poll_interval_ms=1.0, clock=time.perf_counter, sleeper=time.sleep):
    """Collect only the requested association; seconds=0 reads currently committed records."""
    if not math.isfinite(seconds) or seconds < 0 or not math.isfinite(poll_interval_ms) or poll_interval_ms <= 0:
        raise ValueError("Invalid collection duration or polling interval")
    readers, events, snapshots, seen = [], [], [], set()
    views = {name: dict(events=[], snapshots=[]) for name in ("logs", "telemetry", "shm")}
    view_seen = {name: set() for name in views}
    try:
        readers.extend(JsonlMetadataReader(path) for path in log_paths)
        readers.extend(MmapMetadataReader(path) for path in shm_paths)
        if not readers: raise ValueError("At least one metadata export is required")
        deadline = clock()+seconds
        while True:
            for reader in readers:
                for row in reader.read_available():
                    if row["request_id"] != request_id: continue
                    transport = "shm" if isinstance(reader, MmapMetadataReader) else row.get("source", "logs")
                    if transport not in views: raise ValueError("Unknown exported metadata source")
                    group = "events" if row["record_type"] == "event" else "snapshots"
                    view_row = dict(row, transport=transport)
                    if transport == "shm":
                        view_row["source"] = "shm"
                    key = (row["request_id"], row.get("event_id"))
                    if row.get("event_id") is None or key not in view_seen[transport]:
                        views[transport][group].append(view_row)
                        view_seen[transport].add(key)
                    if "event_id" in row:
                        key = (row["request_id"], row["event_id"])
                        if key in seen: continue
                        seen.add(key)
                    (events if row["record_type"] == "event" else snapshots).append(row)
            remaining = deadline-clock()
            if remaining <= 0: break
            sleeper(min(poll_interval_ms/1000, remaining))
    finally:
        for reader in readers: reader.close()
    times = [r["time_ms"] for r in events+snapshots]
    if not times or max(times) <= min(times):
        raise ValueError("Need at least two distinct exported timestamps")
    for view in views.values():
        view.update(request_id=request_id, start_ms=min(times), end_ms=max(times))
    return dict(request_id=request_id, events=sorted(events, key=lambda x:x["time_ms"]),
                snapshots=sorted(snapshots, key=lambda x:x["time_ms"]), start_ms=min(times), end_ms=max(times),
                source_views=views)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs", nargs="*", default=[])
    p.add_argument("--shm", nargs="*", default=[])
    p.add_argument("--request-id", required=True)
    p.add_argument("--seconds", type=float, default=0)
    p.add_argument("--poll-interval-ms", type=float, default=1)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists(): raise FileExistsError(output)
    observation = collect_exports(a.logs, a.shm, request_id=a.request_id, seconds=a.seconds,
                                  poll_interval_ms=a.poll_interval_ms)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream: json.dump(observation, stream, indent=2)


if __name__ == "__main__": main()
