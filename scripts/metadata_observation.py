"""Join exported logs/telemetry/SHM JSONL records for one request."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.acquisition.metadata import read_export


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--request-id", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = dict(request_id=a.request_id, events=[], snapshots=[])
    seen = set()
    for path in a.inputs:
        obs = read_export(path, a.request_id)
        for channel in ("events", "snapshots"):
            for row in obs[channel]:
                # Deduplicate only explicit publisher IDs, never identical-time
                # events that may represent different native operations.
                key = row.get("event_id")
                if key is not None:
                    if key in seen: continue
                    seen.add(key)
                result[channel].append(row)
    times = [r["time_ms"] for r in result["events"]+result["snapshots"]]
    if not times or max(times) <= min(times): raise ValueError("Need a nonempty observation interval")
    result.update(start_ms=min(times), end_ms=max(times))
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream: json.dump(result, stream, indent=2)


if __name__ == "__main__": main()
