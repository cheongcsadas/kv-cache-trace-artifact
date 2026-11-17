"""Command-line entry point for local, auditable corpus preparation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from .core import DOMAINS, TEMPLATES, normalize, read_jsonl, render_records, split_records, write_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    norm = commands.add_parser("normalize", help="Normalize a local corpus; download nothing")
    norm.add_argument("--dataset", required=True, choices=sorted(DOMAINS))
    norm.add_argument("--source", required=True, type=Path)
    norm.add_argument("--annotations", type=Path, help="JSON map from normalized ID to semantic slots")
    norm.add_argument("--output", required=True, type=Path)
    split = commands.add_parser("split", help="Partition underlying sources without overlap")
    split.add_argument("--input", nargs="+", required=True, type=Path)
    split.add_argument("--output-dir", required=True, type=Path)
    split.add_argument("--seed", default=42, type=int)
    split.add_argument("--train-fraction", default=0.75, type=float)
    split.add_argument("--train-per-domain", type=int)
    split.add_argument("--test-per-domain", type=int)
    render = commands.add_parser("render", help="Render a shared template or preserve free text")
    render.add_argument("--input", required=True, type=Path)
    render.add_argument("--output", required=True, type=Path)
    render.add_argument("--template-id", choices=[*TEMPLATES, "unstructured"], default="shared-v1")
    render.add_argument("--local-tokenizer", type=Path, help="Existing tokenizer directory; offline only")
    render.add_argument("--min-tokens", type=int)
    render.add_argument("--max-tokens", type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "normalize":
            annotations = None
            if args.annotations:
                annotations = json.loads(args.annotations.read_text(encoding="utf-8-sig"))
                if not isinstance(annotations, dict):
                    raise ValueError("Annotation file must be a JSON object keyed by record ID")
            rows = normalize(args.dataset, args.source, annotations)
            write_jsonl(args.output, rows)
            print(json.dumps({"records": len(rows), "domain": DOMAINS[args.dataset],
                              "annotated_records": sum(all(k in r for k in
                                                          ("intent", "attribute", "keyword"))
                                                       for r in rows)}))
        elif args.command == "split":
            rows = [row for path in args.input for row in read_jsonl(path)]
            partitions = split_records(rows, seed=args.seed, train_fraction=args.train_fraction,
                                       train_per_domain=args.train_per_domain,
                                       test_per_domain=args.test_per_domain)
            # Refuse existing destinations before writing either partition.
            args.output_dir.mkdir(parents=True, exist_ok=False)
            report = {
                "seed": args.seed, "train_fraction_requested": args.train_fraction,
                "exact_quotas": args.train_per_domain is not None,
                "input_records": len(rows),
                "omitted_records": len(rows) - sum(map(len, partitions.values())),
                "source_overlap": 0,
                "partitions": {},
            }
            for name, partition in partitions.items():
                write_jsonl(args.output_dir / (name + ".jsonl"), partition)
                report["partitions"][name] = {
                    "records": len(partition),
                    "source_groups": len({row["source_id"] for row in partition}),
                    "domains": dict(sorted(Counter(row["domain"] for row in partition).items())),
                }
            (args.output_dir / "split_report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report))
        else:
            tokenizer = None
            if args.local_tokenizer:
                if not args.local_tokenizer.is_dir():
                    raise ValueError("--local-tokenizer must be an existing local directory")
                try:
                    from transformers import AutoTokenizer
                except ImportError as exc:
                    raise ValueError("Token counting requires the optional transformers dependency") from exc
                tokenizer = AutoTokenizer.from_pretrained(str(args.local_tokenizer),
                                                          local_files_only=True,
                                                          trust_remote_code=False)
            rows = render_records(read_jsonl(args.input), template_id=args.template_id,
                                  tokenizer=tokenizer, min_tokens=args.min_tokens,
                                  max_tokens=args.max_tokens)
            write_jsonl(args.output, rows)
            counts = [row["token_count"] for row in rows if "token_count" in row]
            print(json.dumps({"records": len(rows), "template_id": args.template_id,
                              "token_count_mean": sum(counts) / len(counts) if counts else None,
                              "token_count_min": min(counts) if counts else None,
                              "token_count_max": max(counts) if counts else None}))
    except (OSError, ValueError) as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
