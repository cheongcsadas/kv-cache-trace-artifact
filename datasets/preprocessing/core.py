"""Deterministic local data preparation without inferred semantic annotations.

The publication does not specify the exact prompt templates, annotation vocabulary,
or source-record selection. The templates in this module are explicit artifact
choices, rather than recovered experimental inputs. No model or corpus is fetched.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping
import xml.etree.ElementTree as ET
from .sources import source_ids


DOMAINS = {
    "medquad": "medical", "finqa": "financial", "cuad": "legal", "codexglue": "code",
}
SLOTS = ("intent", "attribute", "keyword")
BACKGROUND = (
    "Use the supplied material to address the request. Keep the response focused "
    "on the information provided and state any missing information."
)
TEMPLATES = {
    "shared-v1": BACKGROUND + "\n\nMaterial and request:\n{text}\n\nResponse:\n",
    "shared-v2": BACKGROUND + "\n\n<material>\n{text}\n</material>\n\nAnswer:\n",
    "shared-v3": "Task instructions:\n" + BACKGROUND + "\n\nInput:\n{text}\n\nResult:\n",
}


def _digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:24]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(filter(None, (_text(item) for item in value)))
    return str(value).strip()


def _record(dataset: str, source: str, key: str, text: str, request: str | None = None) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError(f"Empty text in {dataset} record {key!r}")
    row = {
        "id": dataset + ":" + _digest(source, key, text),
        "source_id": dataset + ":" + _digest(source),
        "domain": DOMAINS[dataset],
        "text": text,
    }
    if request: row["request"]=request
    return row


def _files(source: Path, suffix: str) -> list[Path]:
    files = sorted(source.rglob("*" + suffix)) if source.is_dir() else [source]
    if not files:
        raise ValueError(f"No {suffix} files found in input")
    if any(not path.is_file() for path in files):
        raise ValueError("Input is not a file or corpus directory")
    return files


def _medquad(source: Path) -> Iterable[dict[str, Any]]:
    for path in _files(source, ".xml"):
        # Public MedQuAD has one document with QAPair elements per XML file.
        # URL/title are used only to group records, never as public path metadata.
        xml = path.read_text(encoding="utf-8-sig")
        if "<!DOCTYPE" in xml.upper() or "<!ENTITY" in xml.upper():
            raise ValueError("XML declarations with external entities are unsupported")
        root = ET.fromstring(xml)
        source_key = root.attrib.get("url") or root.findtext("URL")
        if not source_key:
            # Content fallback remains stable if the input is moved or renamed.
            source_key = _digest(xml)
        pairs = root.findall(".//QAPair") or root.findall("./qaPairs/pair")
        if not pairs:
            # The pinned public corpus includes documents with an empty
            # QAPairs container, alongside the nonempty QA documents.
            if root.tag == "Document" and root.find("QAPairs") is not None:
                continue
            raise ValueError("MedQuAD XML document has no QAPair records")
        for index, pair in enumerate(pairs):
            question = (pair.findtext("Question") or pair.findtext("question") or "").strip()
            answer = (pair.findtext("Answer") or pair.findtext("answer") or "").strip()
            if not question:
                raise ValueError("MedQuAD QAPair is missing Question")
            text = "\n\n".join(filter(None, [question, answer]))
            yield _record("medquad", source_key, str(index), text,question)


def _finqa(source: Path) -> Iterable[dict[str, Any]]:
    for path in _files(source, ".json"):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("data", data.get("records"))
        if not isinstance(data, list):
            raise ValueError("FinQA input must contain a JSON array of records")
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError("FinQA records must be objects")
            qa = item.get("qa", {})
            question = _text(qa.get("question") if isinstance(qa, dict) else None)
            if not question:
                question = _text(item.get("question"))
            if not question:
                raise ValueError("FinQA record is missing a question")
            table = "\n".join(" | ".join(_text(cell) for cell in row)
                              for row in item.get("table", []))
            context = "\n\n".join(filter(None, [
                _text(item.get("pre_text")), table, _text(item.get("post_text")),
            ]))
            # Official record IDs are stable across downloads and file placement.
            source_key = _text(item.get("id")) or _digest(context, question)
            yield _record("finqa", source_key, "question", "\n\n".join(
                filter(None, [context, question])),question)


def _cuad(source: Path) -> Iterable[dict[str, Any]]:
    for path in _files(source, ".json"):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        docs = data.get("data") if isinstance(data, dict) else None
        if not isinstance(docs, list):
            raise ValueError("CUAD input must be a SQuAD-style object with data")
        for doc in docs:
            paragraphs = doc.get("paragraphs", [])
            context_key = "\n".join(_text(p.get("context")) for p in paragraphs)
            # A contract, including all of its questions, is one split group.
            source_key = _text(doc.get("title")) or _digest(context_key)
            for pidx, paragraph in enumerate(paragraphs):
                context = _text(paragraph.get("context"))
                for qidx, question in enumerate(paragraph.get("qas", [])):
                    text = _text(question.get("question"))
                    if not context or not text:
                        raise ValueError("CUAD question or contract context is missing")
                    key = _text(question.get("id")) or f"{pidx}:{qidx}"
                    yield _record("cuad", source_key, key, context + "\n\n" + text,text)


def _codexglue(source: Path) -> Iterable[dict[str, Any]]:
    for path in _files(source, ".jsonl"):
        for index, item in enumerate(read_jsonl(path)):
            # Supported public task exports include code-to-text and defect
            # detection. Single-label task targets are not semantic slot labels.
            code = _text(item.get("code") or item.get("func"))
            prose = _text(item.get("docstring") or item.get("text") or item.get("nl"))
            if not code and not prose:
                raise ValueError("CodeXGLUE row needs code/func, docstring/text, or nl")
            source_key = (_text(item.get("url")) or _text(item.get("source_id"))
                          or _digest(code or prose))
            key = _text(item.get("id", item.get("idx"))) or "record"
            request = prose or code
            if "nl" in item and not (item.get("docstring") or item.get("text")):
                # CONCODE's nl combines the natural-language request with class
                # fields/methods. Keep the complete source and stable IDs, while
                # intent matching and curation anchor only the actual request.
                separator = re.search(r"(?<!\w)concode_(?:field|elem)_sep(?!\w)", prose)
                request = prose[:separator.start()].strip() if separator else prose
                if not request:
                    raise ValueError("CONCODE nl requires a natural-language request before class-context separators")
            yield _record("codexglue", source_key, key,
                          "\n\n".join(filter(None, [code, prose])),request)


def normalize(dataset: str, source: str | Path,
              annotations: Mapping[str, Mapping[str, Any]] | None = None
              ) -> list[dict[str, Any]]:
    """Read a local corpus and optionally join annotations by normalized ``id``.

    Annotation maps must contain only ``intent``, ``attribute`` and ``keyword``
    scalar labels. This deliberately does not turn dataset-specific QA types,
    answer strings, or classifier targets into unverified semantic annotations.
    Normalize once to obtain stable IDs, annotate those IDs, and normalize again
    with the annotation map. Unknown IDs and unsupported fields are rejected.
    """
    readers = {"medquad": _medquad, "finqa": _finqa, "cuad": _cuad,
               "codexglue": _codexglue}
    if dataset not in readers:
        raise ValueError(f"Unknown dataset: {dataset}")
    unique = {}
    for row in readers[dataset](Path(source)):
        previous = unique.setdefault(row["id"], row)
        if previous != row:
            raise ValueError("Conflicting corpus records share a normalized ID")
    rows = list(unique.values())
    validate_records(rows)
    annotations = annotations or {}
    known = {row["id"] for row in rows}
    if set(annotations) - known:
        raise ValueError("Annotation map contains IDs absent from normalized records")
    for row in rows:
        supplied = annotations.get(row["id"], {})
        if not isinstance(supplied, Mapping) or set(supplied) - set(SLOTS):
            raise ValueError("Annotations may contain only intent, attribute, keyword")
        for name, value in supplied.items():
            if not isinstance(value,str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty single-label string")
            row[name] = value
    return rows


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {number} must be an object")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write a new output; never silently overwrite a prepared split or corpus."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def validate_records(rows: Iterable[Mapping[str, Any]]) -> None:
    ids: set[str] = set()
    source_domains: dict[str, str] = {}
    count = 0
    for row in rows:
        count += 1
        for field in ("id", "source_id", "domain", "text"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing or empty string field: {field}")
        if row["id"] in ids:
            raise ValueError("Duplicate normalized record ID")
        ids.add(row["id"])
        if row["domain"] not in DOMAINS.values():
            raise ValueError("Unknown domain label")
        previous = source_domains.setdefault(row["source_id"], row["domain"])
        if previous != row["domain"]:
            raise ValueError("One source group occurs in multiple domains")
    if not count:
        raise ValueError("No records supplied")


def _covering_groups(groups: list[tuple[str, list[dict[str, Any]]]], minimum: int,
                     maximum: int) -> set[str]:
    """Find groups with a total in [minimum, maximum] using bounded subset sum.

    Rows inside selected source groups may be subsampled to meet exact quotas;
    unselected rows from a selected group are never moved to the opposite split.
    """
    if minimum == 0:
        return set()
    if maximum < minimum or not groups:
        raise ValueError("Insufficient records for source-disjoint quotas")
    cap = min(maximum, minimum + max(len(rows) for _, rows in groups) - 1)
    mask = (1 << (cap + 1)) - 1
    reachable = 1
    parents: dict[int, tuple[int, int]] = {}
    for index, (_, rows) in enumerate(groups):
        size = len(rows)
        newly = ((reachable << size) & mask) & ~reachable
        bits = newly
        while bits:
            bit = bits & -bits
            total = bit.bit_length() - 1
            parents[total] = (total - size, index)
            bits ^= bit
        reachable |= newly
        candidates = reachable >> minimum
        if candidates:
            total = minimum + (candidates & -candidates).bit_length() - 1
            selected = set()
            while total:
                total, group_index = parents[total]
                selected.add(groups[group_index][0])
            return selected
    raise ValueError("Source groups cannot satisfy the requested disjoint quotas")


def split_records(records: Iterable[Mapping[str, Any]], *, seed: int = 42,
                  train_fraction: float = 0.75,
                  train_per_domain: int | None = None,
                  test_per_domain: int | None = None,
                  required_train_sources=None) -> dict[str, list[dict[str, Any]]]:
    """Split by underlying source within each domain using a fixed RNG seed.

    With explicit quotas, each domain yields exactly the requested number of
    training and testing rows. Surplus rows are omitted. Without quotas, all rows
    are retained and 75/25 is approximate at the indivisible source-group level.
    A group cannot appear in both partitions. Seed 42 is an artifact default;
    the paper does not report a split seed.
    """
    rows = [dict(row) for row in records]
    validate_records(rows)
    required_train_sources = set(required_train_sources or ())
    if required_train_sources - {row["source_id"] for row in rows}:
        raise ValueError("Required training sources are absent from the source pool")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    if (train_per_domain is None) != (test_per_domain is None):
        raise ValueError("Supply both train_per_domain and test_per_domain")
    quotas = train_per_domain is not None
    if quotas and required_train_sources:
        raise ValueError("Reserve semantic training sources before selecting exact row quotas")
    if quotas:
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1
               for x in (train_per_domain, test_per_domain)):
            raise ValueError("Per-domain quotas must be positive integers")
        if {row["domain"] for row in rows} != set(DOMAINS.values()):
            raise ValueError("Exact evaluation quotas require all four domains")
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in sorted(rows, key=lambda x: x["id"]):
        grouped[row["domain"]][row["source_id"]].append(row)
    result: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for domain in sorted(grouped):
        rng = random.Random(f"{seed}:{domain}")
        groups = sorted(grouped[domain].items())
        rng.shuffle(groups)
        total = sum(len(values) for _, values in groups)
        if len(groups) < 2:
            raise ValueError(f"Domain {domain} needs at least two source groups")
        if quotas:
            assert train_per_domain is not None and test_per_domain is not None
            selected = _covering_groups(groups, train_per_domain, total - test_per_domain)
        else:
            target = round(total * train_fraction)
            # Grouped splitting cannot generally hit an exact row ratio. Choose
            # a nonempty prefix closest to the target, retaining all records.
            mandatory = {key for key, _ in groups if key in required_train_sources}
            remaining = [(key, values) for key, values in groups if key not in mandatory]
            if not remaining:
                raise ValueError("Training coverage leaves no source-disjoint test group")
            running = sum(len(values) for key, values in groups if key in mandatory)
            candidates = [(abs(running-target), 0)] if mandatory else []
            for index, (_, values) in enumerate(remaining[:-1], 1):
                running += len(values)
                candidates.append((abs(running - target), index))
            chosen_count = min(candidates)[1]
            selected = mandatory | {key for key, _ in remaining[:chosen_count]}
        train = [dict(row, split="train") for key, values in groups
                 if key in selected for row in values]
        test = [dict(row, split="test") for key, values in groups
                if key not in selected for row in values]
        rng.shuffle(train)
        rng.shuffle(test)
        if quotas:
            train, test = train[:train_per_domain], test[:test_per_domain]
        result["train"].extend(train)
        result["test"].extend(test)
    train_sources = {row["source_id"] for row in result["train"]}
    if train_sources & {row["source_id"] for row in result["test"]}:
        raise AssertionError("Source-disjoint split invariant failed")
    return result


def render_records(records: Iterable[Mapping[str, Any]], *, template_id: str = "shared-v1",
                   tokenizer: Any = None, min_tokens: int | None = None,
                   max_tokens: int | None = None) -> list[dict[str, Any]]:
    """Render a common shell and optionally measure true token counts.

    ``unstructured`` uses the shared free-wording renderer while preserving
    source evidence. A tokenizer must offer
    ``encode(text, add_special_tokens=True)``. Bounds reject unmatched records;
    no truncation, synthetic padding, or silent lexical-label destruction occurs.
    These counts describe the rendered input including tokenizer special tokens,
    not a backend chat wrapper or generation marker.
    """
    from .rendering import render_prompt
    rows = [dict(row) for row in records]
    validate_records(rows)
    if template_id not in TEMPLATES and template_id != "unstructured":
        raise ValueError("Unknown template ID")
    if any(value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                 or value < 1) for value in (min_tokens, max_tokens)):
        raise ValueError("Token bounds must be positive integers")
    if min_tokens is not None and max_tokens is not None and min_tokens > max_tokens:
        raise ValueError("min_tokens exceeds max_tokens")
    if tokenizer is None and (min_tokens is not None or max_tokens is not None):
        raise ValueError("Token-length constraints require a tokenizer")
    rendered = []
    for row in rows:
        prompt = render_prompt(row["text"], template_id, record=row)
        output = dict(row, prompt=prompt, template_id=template_id)
        if tokenizer is not None:
            count = len(tokenizer.encode(prompt, add_special_tokens=True))
            if min_tokens is not None and count < min_tokens:
                raise ValueError(f"Record {row['id']} is shorter than min_tokens")
            if max_tokens is not None and count > max_tokens:
                raise ValueError(f"Record {row['id']} is longer than max_tokens")
            output["token_count"] = count
            output["token_count_scope"] = "rendered_prompt_with_tokenizer_special_tokens"
        rendered.append(output)
    return rendered


def assert_template_disjoint(train: Iterable[Mapping[str, Any]],
                             test: Iterable[Mapping[str, Any]]) -> None:
    """Validate both source and template holdout for cross-template evaluation."""
    train, test = list(train), list(test)
    validate_records(train)
    validate_records(test)
    if any(not row.get("template_id") for row in train + test):
        raise ValueError("Cross-template records require template_id")
    if set().union(*(source_ids(row) for row in train)) & set().union(*(source_ids(row) for row in test)):
        raise ValueError("Source records overlap between train and test")
    if {row["template_id"] for row in train} & {row["template_id"] for row in test}:
        raise ValueError("Test templates are present in training data")
