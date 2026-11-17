"""Verify every published label-evidence entry against normalized corpus text."""
import argparse
import hashlib
import json
from pathlib import Path
import yaml

from .annotate import load_taxonomy
from .grounding import validate_grounded_record


def verify(inputs, evidence_path=None):
    root=Path(__file__).resolve().parents[2]
    index=json.loads(Path(evidence_path or Path(__file__).with_name("label_evidence.json")).read_text(encoding="utf-8"))
    taxonomy,_,digest=load_taxonomy()
    catalog=yaml.safe_load((root/"configs/datasets.yaml").read_text(encoding="utf-8"))["datasets"]
    if index.get("schema_version") != 1 or index.get("taxonomy_sha256") != digest:
        raise ValueError("Evidence index must match the fixed label taxonomy")
    if set(index.get("domains",{})) != set(taxonomy["domains"]):
        raise ValueError("Evidence index requires all four domains")
    expected={}
    for domain,group in index["domains"].items():
        spec=catalog.get(group.get("dataset"),{})
        if any(group.get(field) != spec.get(field) for field in ("repository","revision")) or spec.get("domain") != domain:
            raise ValueError("Evidence index corpus revision differs from the pinned catalog")
        records=group.get("records",[])
        if len(records) != 60 or any({entry.get(slot) for entry in records} !=
                {entry["label"] for entry in taxonomy["domains"][domain][slot]} for slot in ("attribute","keyword")):
            raise ValueError("Evidence index requires all 60 distinct keyword/property pairs")
        for entry in records:
            key=entry["record_id"]
            if key in expected: raise ValueError("Evidence records must have distinct normalized IDs")
            expected[key]=(domain,entry)
    found=set()
    for path in inputs:
        with Path(path).open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():continue
                row=json.loads(line)
                if row.get("id") not in expected:continue
                domain,entry=expected[row["id"]]
                if row.get("domain") != domain or row.get("source_id") != entry["source_id"]:
                    raise ValueError("Evidence record source identity differs from normalized input")
                if hashlib.sha256(row["text"].encode()).hexdigest() != entry["source_text_sha256"]:
                    raise ValueError("Evidence record source text differs from its digest")
                checked=dict(row,**{slot:entry[slot] for slot in ("intent","attribute","keyword")},
                             annotation=dict(evidence=entry["evidence"],taxonomy_sha256=digest))
                validate_grounded_record(checked)
                found.add(row["id"])
    if found != set(expected):
        raise ValueError(f"Normalized inputs are missing {len(set(expected)-found)} evidence source records")
    return {domain:60 for domain in index["domains"]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",nargs="+",required=True)
    parser.add_argument("--evidence")
    args=parser.parse_args(argv)
    print(json.dumps(verify(args.input,args.evidence),sort_keys=True))


if __name__ == "__main__":main()
