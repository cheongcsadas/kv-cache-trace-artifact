"""Source evidence, prompt preservation, and source-disjoint coverage contracts."""
from copy import deepcopy
import hashlib
import pytest

from datasets.preprocessing.annotate import label_record, load_taxonomy
from datasets.preprocessing.core import normalize, split_records, DOMAINS
from datasets.preprocessing.grounding import validate_grounded_record, validate_grounding_receipt
from datasets.preprocessing.prepare import measured_candidates, semantic_training_sources


def source():
    text = "What is asthma?\n\nA source paragraph about the condition."
    row = dict(id="record", source_id="source", domain="medical", text=text, request="What is asthma?")
    labels, detail = label_record(row, load_taxonomy()[1])
    return dict(row, **labels, annotation=detail)


def test_verified_receipt_binds_source_labels_and_actual_prompt():
    row = source()
    row["prompt"] = "Input:\n" + row["text"]
    row["annotation"] = validate_grounded_record(row)
    row["prompt_sha256"] = hashlib.sha256(row["prompt"].encode()).hexdigest()
    receipt = validate_grounding_receipt(row)
    for slot in ("intent", "attribute", "keyword"):
        evidence = receipt["prompt_evidence"][slot]
        assert row["prompt"][evidence["start"]:evidence["end"]] == evidence["text"]
    row.pop("text")
    row.pop("prompt")
    assert validate_grounding_receipt(row) == receipt
    row["keyword"] = "diabetes_mellitus"
    with pytest.raises(ValueError, match="labels"):
        validate_grounding_receipt(row)


def test_rehashed_receipt_cannot_relabel_existing_text_evidence():
    from datasets.preprocessing.grounding import _receipt_digest
    row=source()
    row["annotation"]=validate_grounded_record(row)
    row["prompt_sha256"]=row["annotation"]["prompt_sha256"]
    row.update(keyword="diabetes_mellitus",attribute="glucose_homeostasis")
    row["annotation"]["labels"].update(keyword=row["keyword"],attribute=row["attribute"])
    row["annotation"]["evidence"]["attribute"]["category_rule"]=row["attribute"]
    row["annotation"]["receipt_sha256"]=_receipt_digest(row["annotation"])
    with pytest.raises(ValueError, match="does not justify"):
        validate_grounding_receipt(row)


@pytest.mark.parametrize("mutation", ["span", "label", "substring", "variant", "source"])
def test_configured_labels_cannot_replace_original_text_evidence(mutation):
    row = source()
    if mutation == "span": row["annotation"]["evidence"]["keyword"]["start"] += 1
    elif mutation == "label": row["keyword"] = "diabetes_mellitus"
    elif mutation == "substring": row["prompt"] = "What is notasthma?"
    elif mutation == "variant": row["prompt"] = "Explain this unrelated subject."
    else:
        row["annotation"] = validate_grounded_record(row)
        row["text"] += " changed"
    with pytest.raises(ValueError): validate_grounded_record(row)


def test_measured_prompt_variant_cannot_drop_semantic_evidence():
    class Tokenizer:
        def encode(self, text, add_special_tokens=True): return [0] * 1000
    row = source()
    with pytest.raises(ValueError, match="removed original"):
        measured_candidates([row], Tokenizer(), "templated_short", 1000, "shared-v1", .1,
                            variants={row["id"]: {"templated_short": "Unrelated material."}}, semantic=True)


def test_short_genuine_class_is_retained_for_mean_based_semantic_selection():
    class Tokenizer:
        def encode(self,text,add_special_tokens=True):return list(range(len(text.split())+1))
    row=source()
    candidates=measured_candidates([row],Tokenizer(),"templated_short",1000,"shared-v1",.1,semantic=True)
    assert len(candidates) == 1 and candidates[0]["length_band_eligible"] is False
    validate_grounding_receipt(candidates[0])


def test_exact_semantic_quota_checks_mean_without_deleting_rare_short_classes():
    from datasets.preprocessing.prepare import select_quota
    rows=[]
    for domain in DOMAINS.values():
        for index in range(60):
            rows.append(dict(id=f"{domain}-{index}",domain=domain,intent=f"i{index%3}",attribute=f"a{index}",
                             keyword=f"k{index}",token_count=100 if index==0 else 1000,target_tokens=1000,
                             length_tolerance=.1,length_band_eligible=index!=0))
    selected=select_quota(rows,60,42,semantic=True)
    assert len(selected) == 240 and sum(row["token_count"]==100 for row in selected) == 4
    for row in rows:row["token_count"]=100
    with pytest.raises(ValueError,match="tokenizer mean"):
        select_quota(rows,60,42,semantic=True)


def test_multiline_explicit_request_remains_the_annotation_focus():
    row = dict(id="code", source_id="code", domain="code", text="Return the HashMap.\n\nAdditional words.",
               request="Return the HashMap.\n\nAdditional words.")
    labels, _ = label_record(row, load_taxonomy()[1])
    assert labels["keyword"] == "hash_map"


def test_source_split_keeps_only_available_rare_class_in_training():
    rows=[]
    for domain in DOMAINS.values():
        for index in range(12):
            rows.append(dict(id=f"{domain}-{index}", source_id=f"{domain}-{index}", domain=domain, text="fixture",
                             intent="i", attribute="rare" if index == 0 else "common", keyword="k"))
    protected=semantic_training_sources(rows)
    split=split_records(rows, required_train_sources=protected)
    assert protected <= {row["source_id"] for row in split["train"]}
    assert {row["source_id"] for row in split["train"]}.isdisjoint(row["source_id"] for row in split["test"])
    assert all(any(row["domain"] == domain and row["attribute"] == "rare" for row in split["train"])
               for domain in DOMAINS.values())


def test_medquad_accepts_pinned_empty_and_lowercase_document_variants(tmp_path):
    (tmp_path/"empty.xml").write_text('<Document url="empty"><QAPairs/></Document>')
    (tmp_path/"lower.xml").write_text('<doc url="lower"><qaPairs><pair><question>What is asthma?</question><answer>Source answer.</answer></pair></qaPairs></doc>')
    rows=normalize("medquad", tmp_path)
    assert len(rows) == 1 and rows[0]["request"] == "What is asthma?"


def test_exact_duplicate_source_rows_are_not_counted_twice(tmp_path):
    import json
    path=tmp_path/"code.jsonl"
    row=dict(code="String value;",nl="Return the String.")
    path.write_text((json.dumps(row)+"\n")*2)
    assert len(normalize("codexglue", path)) == 1


def test_concode_official_natural_language_only_test_inputs_normalize(tmp_path):
    import json
    from datasets.acquire import normalize_download
    path=tmp_path/"test.json"
    path.write_text(json.dumps(dict(nl="Return the String.",code=""))+"\n")
    output=tmp_path/"normalized.jsonl"
    assert normalize_download("codexglue",tmp_path,dict(inputs=["test.json"],domain="code"),output) == 1
    assert json.loads(output.read_text())["text"] == "Return the String."


def test_published_evidence_verifier_rejects_tampered_source_text(tmp_path):
    import json
    from pathlib import Path
    from datasets.preprocessing.verify_evidence import verify
    index=json.loads((Path(__file__).resolve().parents[1]/"datasets/preprocessing/label_evidence.json").read_text(encoding="utf-8"))
    domain,group=next(iter(index["domains"].items()))
    entry=group["records"][0]
    path=tmp_path/"tampered.jsonl"
    path.write_text(json.dumps(dict(id=entry["record_id"],source_id=entry["source_id"],domain=domain,
                                   text="Modified source text that does not match the published digest."))+"\n")
    with pytest.raises(ValueError,match="source text differs"):
        verify([path])
