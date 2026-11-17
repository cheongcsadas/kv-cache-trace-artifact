"""Render the artifact's prompt wording without changing source provenance.

The shared templates are the repository's explicit template choices. The free
wording below is also an artifact construction, not an original paper template.
Its connective sentences are generated presentation text: source_spans continue
to refer only to the original corpus text, never to these sentences. Token counts
must be computed on the complete returned prompt, with tokenizer special tokens.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping

from .core import TEMPLATES


# Use the same domain-independent pool for every corpus. Vary both placement and
# wording without adding named fields, annotation labels, or domain-specific hints.
_FREE_WORDING = (
    ("Please help me with the following material and the request it contains.\n\n",
     "\n\nKeep the answer focused on the supplied information."),
    ("", "\n\nCould you address the request using this material? If information is missing, say so."),
    ("I would like an answer based on this material.\n\n",
     "\n\nPlease explain only what the information supports."),
    ("Here is the information I have to work with.\n\n",
     "\n\nPlease use it to respond to the request and mention any missing information."),
    ("Please consider this material when answering.\n\n", ""),
    ("", "\n\nUsing what I have provided, please give a focused response to the request."),
)


def _source_content(text: str, record: Mapping | None) -> str:
    """Express CONCODE's context separators as prose, preserving code and request.

    Separators are serialization controls in the natural-language/class-context
    part of CONCODE. The original request and complete source-code prefix, when
    present, are protected. Java literals and comments in retained source excerpts
    are protected too, so a literal containing a separator name is not rewritten.
    """
    if not record or record.get("domain") != "code":
        return text
    protected = []
    request = record.get("request")
    original = record.get("text")
    if isinstance(request, str) and request:
        protected.append(request)
        if isinstance(original, str) and request in original:
            code = original[:original.rfind(request)].rstrip()
            if code:
                protected.append(code)
    literals = r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])\'|//[^\n]*|/\*[\s\S]*?\*/'
    guards = "|".join(re.escape(value) for value in sorted(protected, key=len, reverse=True))
    separator = r"(?<!\w)concode_(?P<kind>field|elem)_sep(?!\w)"
    expression = re.compile((guards + "|" if guards else "") + literals + "|" + separator)

    def replace(match):
        kind = match.group("kind")
        if kind is None:
            return match.group()
        return "; the surrounding class also provides " if kind == "field" else ", "

    return expression.sub(replace, text)


def render_prompt(text: str, template_id: str, record: Mapping | None = None) -> str:
    """Return exactly the prompt to tokenize and send to the native backend.

    Unstructured wording is deterministic for a source record and independent of
    its semantic labels. Passing the same record during curation and final prompt
    preparation therefore produces identical connective text and token counts.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Prompt source text must be a nonempty string")
    if record is not None and not isinstance(record, Mapping):
        raise ValueError("Prompt record must be a mapping")
    if template_id != "unstructured":
        if template_id not in TEMPLATES:
            raise ValueError(f"Unknown prompt template: {template_id}")
        return TEMPLATES[template_id].format(text=text)
    identity = next((record.get(key) for key in ("id", "sample_id", "source_id")
                     if isinstance(record.get(key), str) and record[key]), text) if record else text
    index = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big") % len(_FREE_WORDING)
    prefix, suffix = _FREE_WORDING[index]
    return prefix + _source_content(text, record) + suffix
