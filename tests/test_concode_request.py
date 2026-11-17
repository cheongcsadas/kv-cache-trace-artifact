"""CONCODE request/context separation preserves normalized source identity."""
import json

import pytest

from datasets.preprocessing.annotate import _focus
from datasets.preprocessing.core import normalize, _record, _digest


@pytest.mark.parametrize("separator", ["concode_field_sep", "concode_elem_sep"])
@pytest.mark.parametrize("code", ["boolean function ( ) { return isParsed ; }", ""])
def test_request_is_only_natural_language_before_first_structural_separator(tmp_path, separator, code):
    prose = f"check if details are parsed . {separator} Container parent concode_elem_sep boolean isParsed"
    path = tmp_path/"concode.jsonl"
    path.write_text(json.dumps(dict(code=code, nl=prose))+"\n", encoding="utf-8")
    row, = normalize("codexglue", path)
    # The pre-change identity and text depend on the complete original source.
    previous = _record("codexglue", _digest(code or prose), "record",
                       "\n\n".join(filter(None, [code, prose])), prose or code)
    assert {key: row[key] for key in ("id", "source_id", "text", "domain")} == {
        key: previous[key] for key in ("id", "source_id", "text", "domain")}
    assert row["request"] == "check if details are parsed ."
    start, end = _focus(row["text"], row["domain"], row["request"])
    assert row["text"][start:end] == row["request"]
    assert end-start < len(prose)


@pytest.mark.parametrize("prose", ["", "   ", "concode_field_sep String field", "concode_elem_sep void method"])
def test_empty_concode_natural_language_is_not_replaced_with_code(tmp_path, prose):
    path = tmp_path/"empty.jsonl"
    path.write_text(json.dumps(dict(code="return 1;", nl=prose))+"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="natural-language request"):
        normalize("codexglue", path)


def test_plain_docstring_and_separator_like_identifiers_are_unchanged(tmp_path):
    path = tmp_path/"plain.jsonl"
    rows = [dict(id="nl", code="return x;", nl="Return my_concode_field_sep_value."),
            dict(id="doc", code="return y;", docstring="Return the string concode_field_sep unchanged.")]
    path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    records = normalize("codexglue", path)
    assert records[0]["request"] == rows[0]["nl"]
    assert records[1]["request"] == rows[1]["docstring"]
