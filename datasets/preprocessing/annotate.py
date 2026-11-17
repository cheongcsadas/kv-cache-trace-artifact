"""Derive auditable single-label semantic targets from normalized source text.

Rules operate on the original material only. Labels and matched evidence are
sidecar metadata: the text sent to a victim model is never rewritten here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from collections.abc import Mapping

import yaml

from .core import DOMAINS, SLOTS, read_jsonl, validate_records, write_jsonl


DEFAULT_TAXONOMY = Path(__file__).resolve().parents[2] / "configs" / "semantic_labels.yaml"
DOMAIN_NAMES = tuple(DOMAINS.values())


def _strings(values, field):
    if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f"{field} must be a list of nonempty strings")
    return values


def _alias_expression(alias):
    # Match whole terms/identifiers, with whitespace flexibility inside phrases.
    # In particular, 'int' cannot match 'print' and 'tax' cannot match 'syntax'.
    return r"(?<!\w)" + r"\s+".join(re.escape(part) for part in alias.split()) + r"(?!\w)"


def load_taxonomy(path=DEFAULT_TAXONOMY):
    source = Path(path).read_text(encoding="utf-8")
    taxonomy = yaml.safe_load(source)
    if (not isinstance(taxonomy, Mapping) or taxonomy.get("schema_version") != 1
            or set(taxonomy.get("domains", {})) != set(DOMAIN_NAMES)):
        raise ValueError("Taxonomy requires schema_version 1 and all four domains")
    compiled = {}
    for domain in DOMAIN_NAMES:
        spec = taxonomy["domains"][domain]
        groups = {}
        for name, limit in (("intent", 3), ("attribute", 60), ("keyword", 60)):
            entries = spec.get(name)
            if not isinstance(entries, list) or len(entries) != limit:
                raise ValueError(f"{domain}.{name} requires exactly {limit} classes")
            if name == "intent" and len(entries) != 3:
                raise ValueError(f"{domain} must define three intent rules")
            labels = [entry.get("label") for entry in entries if isinstance(entry, Mapping)]
            if (len(labels) != len(entries) or any(not isinstance(v, str) or not v.strip() for v in labels)
                    or len(set(labels)) != len(labels)):
                raise ValueError(f"{domain}.{name} labels must be unique nonempty strings")
            groups[name] = []
            aliases_seen = {}
            for index, entry in enumerate(entries):
                if not isinstance(entry.get("description"), str) or not entry["description"].strip():
                    raise ValueError(f"{domain}.{name}.{entry['label']} needs a class description")
                aliases = _strings(entry.get("aliases", []), "aliases")
                patterns = _strings(entry.get("patterns", []), "patterns")
                excluded_requests = _strings(entry.get("excluded_request_aliases", []), "excluded_request_aliases")
                if excluded_requests and name != "keyword":
                    raise ValueError("Request disambiguation belongs to keyword rules")
                if name != "attribute" and not (aliases or patterns):
                    raise ValueError(f"{domain}.{name}.{entry['label']} needs a text rule")
                if name == "keyword" and patterns:
                    raise ValueError("Keywords use explicit aliases, not arbitrary regular expressions")
                for alias in aliases:
                    normalized = " ".join(alias.casefold().split())
                    previous = aliases_seen.setdefault(normalized, entry["label"])
                    if previous != entry["label"]:
                        raise ValueError(f"Alias {alias!r} maps to multiple {domain}.{name} labels")
                priority = entry.get("priority", 0)
                if isinstance(priority, bool) or not isinstance(priority, int):
                    raise ValueError("Rule priority must be an integer")
                expressions = [_alias_expression(alias) for alias in aliases] + patterns
                groups[name].append(dict(entry, index=index, priority=priority,
                    compiled=[re.compile(pattern, re.IGNORECASE) for pattern in expressions],
                    excluded_requests=[re.compile(_alias_expression(alias), re.IGNORECASE)
                                       for alias in excluded_requests]))
        attribute_labels = {entry["label"] for entry in groups["attribute"]}
        if any(entry.get("attribute") not in attribute_labels for entry in groups["keyword"]):
            raise ValueError(f"Every {domain} keyword must name its attribute category")
        if {entry["attribute"] for entry in groups["keyword"]} != attribute_labels:
            raise ValueError(f"Every {domain} attribute must have a grounded keyword rule")
        for attribute in groups["attribute"]:
            linked={entry["label"] for entry in groups["keyword"] if entry["attribute"] == attribute["label"]}
            if set(attribute.get("evidence_keywords",())) != linked:
                raise ValueError(f"Attribute evidence rule disagrees with keyword mapping: {domain}/{attribute['label']}")
        compiled[domain] = groups
    return taxonomy, compiled, hashlib.sha256(source.encode("utf-8")).hexdigest()


def _focus(text, domain, request=None):
    """Use the question/docstring position established by core.normalize()."""
    if request is not None:
        if not isinstance(request, str) or not request.strip() or request not in text:
            raise ValueError("Core request must be an actual substring of its source record")
        start = text.rfind(request)
        return start, start + len(request)
    paragraphs = list(re.finditer(r"\S[\s\S]*?(?=\n[ \t]*\n|\Z)", text))
    selected = paragraphs[0] if domain == "medical" else paragraphs[-1]
    return selected.start(), selected.end()


def _matches(entries, text, start, end):
    candidates = []
    for entry in entries:
        for expression in entry["compiled"]:
            for match in expression.finditer(text, start, end):
                if match.end() == match.start():
                    raise ValueError("Annotation rules must not match empty text")
                score = (-entry["priority"], -(match.end()-match.start()), match.start(), entry["index"])
                candidates.append((score, entry, match))
    if not candidates:
        return None
    _, entry, match = min(candidates, key=lambda candidate: candidate[0])
    return entry, dict(start=match.start(), end=match.end(), text=match.group())


def _keyword_allowed(entry, text, start, end):
    """Reject explicit alternate meanings in the actual source request.

    A background comparison to another condition cannot override this exclusion
    when keyword matching falls back from the request to the surrounding source.
    """
    return not any(expression.search(text, start, end) for expression in entry.get("excluded_requests", ()))


def label_record(record, compiled):
    """Resolve one intent and one grounded keyword/category pair.

    Intent is matched only in the source request/docstring. Keyword search first
    uses that region, then the original context when the request is elliptical.
    Priority, longest alias, earliest occurrence, and configuration order provide
    stable tie breaking. A record without complete evidence receives no targets.
    """
    domain, text = record["domain"], record["text"]
    rules = compiled[domain]
    start, end = _focus(text, domain, record.get("request"))
    intent = _matches(rules["intent"], text, start, end)
    keyword_rules = [entry for entry in rules["keyword"] if _keyword_allowed(entry, text, start, end)]
    keyword = _matches(keyword_rules, text, start, end)
    if keyword is None:
        keyword = _matches(keyword_rules, text, 0, len(text))
    missing = ([] if intent else ["intent"]) + ([] if keyword else ["attribute", "keyword"])
    if missing:
        return None, dict(status="excluded_semantic", missing_slots=missing, evidence={})
    intent_rule, intent_evidence = intent
    keyword_rule, keyword_evidence = keyword
    labels = dict(intent=intent_rule["label"], attribute=keyword_rule["attribute"], keyword=keyword_rule["label"])
    evidence = dict(intent=intent_evidence, keyword=keyword_evidence,
                    attribute=dict(keyword_evidence, category_rule=keyword_rule["attribute"]))
    return labels, dict(status="annotated", evidence=evidence)


def _vocabulary(candidates):
    return {domain: {slot: sorted({labels[slot] for row, labels, _ in candidates
                                  if row["domain"] == domain and labels is not None})
                     for slot in SLOTS} for domain in DOMAIN_NAMES}


def annotate(records, output, *, taxonomy_path=DEFAULT_TAXONOMY, fit_records=None):
    """Write grounded annotations and the complete fixed paper-size vocabulary.

    ``fit_records`` audits coverage on training sources before writing output.
    Coverage never changes the declared classes. A fixed taxonomy is used, with
    no learned rules, artificial samples or label-frequency-based assignment.
    """
    records = [dict(row) for row in records]
    validate_records(records)
    taxonomy, compiled, digest = load_taxonomy(taxonomy_path)
    candidates = [(row, *label_record(row, compiled)) for row in records]
    vocabulary={domain:{slot:[entry["label"] for entry in taxonomy["domains"][domain][slot]]
                        for slot in SLOTS} for domain in DOMAIN_NAMES}
    if fit_records is not None:
        fit_records = [dict(row) for row in fit_records]
        validate_records(fit_records)
        covered = _vocabulary([(row, *label_record(row, compiled)) for row in fit_records])
        missing={domain:{slot:sorted(set(vocabulary[domain][slot])-set(covered[domain][slot]))
                         for slot in SLOTS} for domain in DOMAIN_NAMES}
        if any(labels for slots in missing.values() for labels in slots.values()):
            raise ValueError("Training-source coverage must include every declared class: "+json.dumps(missing))
    # These counts describe observations, separately from the fixed vocabulary.
    observed = _vocabulary(candidates)
    annotations, annotated = {}, []
    counts = {domain: Counter() for domain in DOMAIN_NAMES}
    for row, labels, detail in candidates:
        clean = {key: value for key, value in row.items() if key not in (*SLOTS, "labels", "annotation")}
        clean["annotation"] = dict(detail, method="rule_derived", taxonomy_sha256=digest)
        counts[row["domain"]]["records"] += 1
        counts[row["domain"]][detail["status"]] += 1
        if labels is not None:
            annotations[row["id"]] = labels
            clean.update(labels)
        annotated.append(clean)
    summary = dict(method="rule_derived", taxonomy_sha256=digest,
        vocabulary_basis="fixed_taxonomy", training_coverage_checked=fit_records is not None,
        records=len(records), annotated=len(annotations), excluded_semantic=len(records)-len(annotations),
        domain_counts={domain: dict(counts[domain]) for domain in DOMAIN_NAMES},
        class_counts={domain: {slot: len(vocabulary[domain][slot]) for slot in SLOTS} for domain in DOMAIN_NAMES},
        observed_class_counts={domain: {slot: len(observed[domain][slot]) for slot in SLOTS} for domain in DOMAIN_NAMES})
    if fit_records is not None:
        summary["fit_record_ids_sha256"] = hashlib.sha256("\n".join(sorted(r["id"] for r in fit_records)).encode()).hexdigest()
        summary["fit_records"] = len(fit_records)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("annotations.json", annotations), ("label_vocabulary.json", vocabulary),
                        ("annotation_summary.json", summary), ("label_definitions.json", taxonomy)):
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    write_jsonl(output/"annotated.jsonl", annotated)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Normalized corpus JSONL files")
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--fit-input", nargs="+", help="Audit all 3/60/60 declared classes against training-only JSONL files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    records = [row for path in args.input for row in read_jsonl(path)]
    fitted = None if args.fit_input is None else [row for path in args.fit_input for row in read_jsonl(path)]
    print(json.dumps(annotate(records, args.output, taxonomy_path=args.taxonomy, fit_records=fitted), ensure_ascii=False))


if __name__ == "__main__":
    main()
