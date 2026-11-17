"""Shared rendering counts the exact prompt sent to each native tokenizer."""
from copy import deepcopy

import pytest

from datasets.preprocessing.core import TEMPLATES
from datasets.preprocessing.curation import SourceCurator
from datasets.preprocessing.rendering import render_prompt


@pytest.mark.parametrize("template_id", sorted(TEMPLATES))
def test_shared_templates_retain_the_same_domain_independent_shell(template_id):
    text = "Original request and evidence."
    assert render_prompt(text, template_id, {"id": "one"}) == TEMPLATES[template_id].format(text=text)


def test_free_wording_is_varied_stable_and_independent_of_target_labels():
    text = "Original request and evidence."
    outputs = set()
    for index in range(50):
        row = {"id": f"record-{index}", "domain": "medical", "intent": "first"}
        prompt = render_prompt(text, "unstructured", row)
        assert prompt == render_prompt(text, "unstructured", dict(row, intent="other", keyword="different"))
        assert text in prompt
        assert "Material and request:" not in prompt and "<material>" not in prompt
        outputs.add(prompt)
    assert len(outputs) == 6


def test_concode_context_becomes_prose_without_rewriting_request_code_or_provenance():
    code = 'String function() { return "concode_field_sep"; } // concode_elem_sep'
    request = "return the stored value ."
    text = code + "\n\n" + request + " concode_field_sep String value concode_elem_sep int size"
    row = dict(id="concode:one", source_id="source:one", domain="code", text=text,
               request=request, source_spans=[dict(start=0, end=len(text), sha256="original")])
    before = deepcopy(row)
    prompt = render_prompt(text, "unstructured", row)
    assert code in prompt and request in prompt
    assert "String value" in prompt and "int size" in prompt
    assert "concode_field_sep" not in prompt.replace(code, "")
    assert "concode_elem_sep" not in prompt.replace(code, "")
    assert row == before


def test_concode_literal_and_comment_excerpts_remain_exact():
    code = 'String function() { return "concode_field_sep"; } /* concode_elem_sep */'
    request = "return the size ."
    original = code + "\n\n" + request + " concode_field_sep int size"
    # A curator may retain a source excerpt rather than the entire code prefix.
    excerpt = 'return "concode_field_sep"; } /* concode_elem_sep */'
    row = dict(id="concode:two", domain="code", request=request, text=original)
    prompt = render_prompt(excerpt + "\n\n" + request + " concode_field_sep int size", "unstructured", row)
    assert excerpt in prompt and request in prompt
    assert "concode_field_sep int size" not in prompt


def test_original_request_containing_control_names_is_preserved():
    request = "explain the identifier concode_field_sep ."
    row = dict(id="concode:three", domain="code", request=request,
               text=request + " concode_field_sep int size")
    prompt = render_prompt(row["text"], "unstructured", row)
    assert request in prompt
    assert prompt.count("concode_field_sep") == 1


def test_curator_count_includes_all_free_wording_and_special_tokens():
    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is True
            return [0] * (len(text) + 2)

    row = dict(id="source:four", source_id="source:four", domain="medical", text="A real request.")
    curator = SourceCurator([row], Tokenizer())
    expected = len(render_prompt(row["text"], "unstructured", row)) + 2
    assert curator.count(row["text"], "unstructured", row) == expected
    assert expected > len(row["text"]) + 2


@pytest.mark.parametrize("text,template,record", [("", "unstructured", None),
                                                 ("text", "unknown", None),
                                                 ("text", "unstructured", [])])
def test_invalid_render_inputs_are_rejected(text, template, record):
    with pytest.raises(ValueError):
        render_prompt(text, template, record)
