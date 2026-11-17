"""Bind semantic targets to original source spans and the actual rendered prompt."""
from functools import lru_cache
import hashlib
import json
from collections.abc import Mapping

from .annotate import load_taxonomy, _focus, _matches, _keyword_allowed
from .core import SLOTS


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _rules():
    return load_taxonomy()


def _targets(row):
    domain = row.get("domain")
    if domain not in _rules()[1]:
        raise ValueError("Grounded targets require a paper domain")
    nested = row.get("labels", {})
    if not isinstance(nested, Mapping):
        raise ValueError("Semantic labels must be a mapping")
    labels = dict(domain=domain)
    if "domain" in nested and nested["domain"] != domain:
        raise ValueError("Grounded domain differs from semantic labels")
    for slot in SLOTS:
        label = row.get(slot, nested.get(slot))
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Grounded targets require all semantic labels")
        if slot in nested and nested[slot] != label:
            raise ValueError("Top-level and nested semantic labels disagree")
        labels[slot] = label
    return labels


def _span(text, evidence):
    if not isinstance(evidence, Mapping):
        raise ValueError("Semantic evidence must identify an exact original source span")
    start, end = evidence.get("start"), evidence.get("end")
    if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int)
            or not isinstance(end, int) or not 0 <= start < end <= len(text)
            or text[start:end] != evidence.get("text")):
        raise ValueError("Semantic evidence must identify an exact original source span")
    return start, end


def _receipt_digest(annotation):
    return _digest(json.dumps({key: value for key, value in annotation.items() if key != "receipt_sha256"},
                             sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def validate_grounded_record(row, *, prompt=None):
    """Verify raw-source labels and preserve their evidence in the rendered input.

    Return an annotation receipt for the caller to store in ``row['annotation']``.
    Explicit annotation maps use the same class-specific text rules as automatic
    annotations; a vocabulary name alone never supplies evidence.
    """
    labels = _targets(row)
    domain = labels["domain"]
    _, compiled, taxonomy_digest = _rules()
    rules = compiled[domain]
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Grounding requires the original normalized source text")
    source_id = row.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("Grounding requires the original source_id")
    prompt = row.get("prompt", text) if prompt is None else prompt
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Grounding requires a nonempty rendered prompt")
    annotation = row.get("annotation", {})
    if not isinstance(annotation, Mapping):
        raise ValueError("Semantic annotation must be a mapping")
    for field, expected in (("taxonomy_sha256", taxonomy_digest), ("source_id", source_id),
                            ("source_text_sha256", _digest(text)), ("labels", labels)):
        if field in annotation and annotation[field] != expected:
            raise ValueError(f"Grounding annotation {field} differs from source or targets")
    selected = {}
    for slot in SLOTS:
        selected[slot] = next((entry for entry in rules[slot] if entry["label"] == labels[slot]), None)
        if selected[slot] is None:
            raise ValueError(f"Unknown grounded {domain}/{slot} class: {labels[slot]}")
    if selected["keyword"]["attribute"] != labels["attribute"]:
        raise ValueError("Semantic attribute does not match the grounded keyword's property")
    start, end = _focus(text, domain, row.get("request"))
    if not _keyword_allowed(selected["keyword"], text, start, end):
        raise ValueError("Source request explicitly excludes the declared keyword meaning")
    evidence = {}
    declared = annotation.get("evidence", {})
    if not isinstance(declared, Mapping):
        raise ValueError("Semantic evidence must be a mapping")
    for slot in ("intent", "keyword"):
        lower, upper = (start, end) if slot == "intent" else (0, len(text))
        if slot in declared:
            span_start, span_end = _span(text, declared[slot])
            if not lower <= span_start < span_end <= upper or not any(
                    (match := expression.match(text, span_start)) is not None and match.end() == span_end
                    for expression in selected[slot]["compiled"]):
                raise ValueError(f"Source evidence does not justify the declared {slot} label")
            evidence[slot] = dict(start=span_start, end=span_end, text=text[span_start:span_end])
        else:
            match = _matches([selected[slot]], text, start, end)
            if match is None and slot == "keyword":
                match = _matches([selected[slot]], text, 0, len(text))
            if match is None:
                raise ValueError(f"Source text has no evidence for declared {domain}/{slot} label")
            evidence[slot] = match[1]
    evidence["attribute"] = dict(evidence["keyword"], category_rule=labels["attribute"])
    if "attribute" in declared:
        _span(text, declared["attribute"])
        if any(declared["attribute"].get(field) != evidence["attribute"][field]
               for field in ("start", "end", "text", "category_rule")):
            raise ValueError("Attribute evidence must follow the actual grounded keyword")
    prompt_evidence = {}
    for slot, item in evidence.items():
        rules_for_slot = selected["keyword" if slot == "attribute" else slot]["compiled"]
        matches = [match for expression in rules_for_slot for match in expression.finditer(prompt)
                   if match.group() == item["text"]]
        if not matches:
            raise ValueError(f"Rendered prompt removed original {slot} evidence")
        position = min(match.start() for match in matches)
        prompt_evidence[slot] = dict(start=position, end=position+len(item["text"]), text=item["text"])
    result = dict(status="annotated", method="source_text_grounded", schema_version=1,
                  labels=labels, source_id=source_id, source_text_sha256=_digest(text),
                  source_length=len(text), prompt_sha256=_digest(prompt), prompt_length=len(prompt),
                  taxonomy_sha256=taxonomy_digest, evidence=evidence, prompt_evidence=prompt_evidence)
    if selected["keyword"].get("excluded_requests"):
        request = text[start:end]
        position = prompt.find(request)
        if position < 0:
            raise ValueError("Rendered prompt removed original request disambiguation evidence")
        result["request_evidence"] = dict(start=start, end=end, text=request)
        result["prompt_request_evidence"] = dict(start=position, end=position+len(request), text=request)
        if "request_evidence" in annotation and annotation["request_evidence"] != result["request_evidence"]:
            raise ValueError("Grounding request evidence differs from the original source")
    result["receipt_sha256"] = _receipt_digest(result)
    return result


def _receipt_span(value, length):
    if not isinstance(value, Mapping):
        raise ValueError("Grounding receipt has invalid evidence span")
    start, end, text = value.get("start"), value.get("end"), value.get("text")
    if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int)
            or not 0 <= start < end <= length or not isinstance(text, str) or len(text) != end-start):
        raise ValueError("Grounding receipt has invalid evidence span")


def validate_grounding_receipt(row):
    """Check labels and prompt binding after source text is removed from collection.

    This checks consistency of the receipt produced above; the source-to-text
    verification is performed before native collection, while raw text is present.
    """
    labels = _targets(row)
    annotation = row.get("annotation")
    if not isinstance(annotation, Mapping) or annotation.get("schema_version") != 1:
        raise ValueError("Semantic observations require a source-grounding receipt")
    for field, expected in (("labels", labels), ("source_id", row.get("source_id")),
                            ("prompt_sha256", row.get("prompt_sha256")),
                            ("taxonomy_sha256", _rules()[2])):
        if annotation.get(field) != expected or expected is None:
            raise ValueError(f"Grounding receipt {field} differs from observation")
    for field in ("prompt_sha256", "source_text_sha256", "taxonomy_sha256"):
        value = annotation.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Grounding receipt requires valid SHA-256 digests")
    for field, length_field in (("evidence", "source_length"), ("prompt_evidence", "prompt_length")):
        length = annotation.get(length_field)
        values = annotation.get(field)
        if isinstance(length, bool) or not isinstance(length, int) or length < 1 or not isinstance(values, Mapping) or set(values) != set(SLOTS):
            raise ValueError("Grounding receipt requires complete source and prompt spans")
        for value in values.values():
            _receipt_span(value, length)
    rules = _rules()[1][labels["domain"]]
    selected = {slot: next((entry for entry in rules[slot] if entry["label"] == labels[slot]), None)
                for slot in SLOTS}
    if any(value is None for value in selected.values()):
        raise ValueError("Grounding receipt labels are absent from the fixed taxonomy")
    evidence, prompt_evidence = annotation["evidence"], annotation["prompt_evidence"]
    if (selected["keyword"]["attribute"] != labels["attribute"] or
            evidence["attribute"].get("category_rule") != labels["attribute"] or
            any(evidence["attribute"][field] != evidence["keyword"][field] for field in ("start", "end", "text"))):
        raise ValueError("Grounding receipt attribute does not follow the keyword property")
    for slot in ("intent", "keyword"):
        if not any(expression.fullmatch(evidence[slot]["text"]) for expression in selected[slot]["compiled"]):
            raise ValueError("Grounding receipt evidence does not justify its semantic labels")
    if any(evidence[slot]["text"] != prompt_evidence[slot]["text"] for slot in SLOTS):
        raise ValueError("Grounding receipt prompt evidence differs from its original source")
    if selected["keyword"].get("excluded_requests"):
        source_request = annotation.get("request_evidence")
        prompt_request = annotation.get("prompt_request_evidence")
        _receipt_span(source_request, annotation["source_length"])
        _receipt_span(prompt_request, annotation["prompt_length"])
        if (source_request["text"] != prompt_request["text"] or
                not source_request["start"] <= evidence["intent"]["start"] < evidence["intent"]["end"] <= source_request["end"]):
            raise ValueError("Grounding receipt request evidence is inconsistent")
        if not _keyword_allowed(selected["keyword"], source_request["text"], 0, len(source_request["text"])):
            raise ValueError("Grounding receipt request explicitly excludes the keyword meaning")
    if annotation.get("receipt_sha256") != _receipt_digest(annotation):
        raise ValueError("Grounding receipt content differs from its digest")
    return annotation
