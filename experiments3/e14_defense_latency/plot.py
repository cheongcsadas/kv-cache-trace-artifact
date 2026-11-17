"""Render Figure 7 from an existing paired-latency summary."""
import argparse
import json
from pathlib import Path

from experiments.chapter6.plotting import figure7


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--auto-y", action="store_true")
    args = parser.parse_args(argv)
    summary = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if summary.get("experiment") != "e14_defense_latency":
        parser.error("Figure 7 requires e14_defense_latency results")
    for name in figure7(summary["runs"], args.output, auto_y=args.auto_y):
        print(name)


if __name__ == "__main__":
    main()
