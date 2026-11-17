import copy
import json

import pytest
import yaml

from datasets.preprocessing.annotate import DEFAULT_TAXONOMY, annotate, label_record, load_taxonomy, main
from datasets.preprocessing.core import read_jsonl


def row(domain, text, name="one"):
    return dict(id=f"{domain}-{name}", source_id=f"source-{domain}-{name}", domain=domain, text=text)


EXAMPLES = [
    ("medical", "What is asthma?\n\nAirway inflammation is discussed here.",
     "condition_information", "reversible_airway_obstruction", "asthma"),
    ("medical", "How is diabetes mellitus treated?\n\nThe answer discusses management.",
     "treatment_prevention", "glucose_homeostasis", "diabetes_mellitus"),
    ("medical", "What is the prognosis for lung cancer?\n\nOutcome information.",
     "prognosis_inheritance", "pulmonary_malignancy", "lung_cancer"),
    ("financial", "Revenue was 15. Net income was 3.\n\nWhat was total revenue?",
     "amount_calculation", "recognized_business_income", "revenue"),
    ("financial", "Revenue was 15. Net income was 3.\n\nWhat percentage was net income?",
     "ratio_percentage", "residual_period_profit", "net_income"),
    ("financial", "Revenue was 15. Net income was 3.\n\nWhat was the change in net income?",
     "change_comparison", "residual_period_profit", "net_income"),
    ("legal", "The agreement contains many terms.\n\nHighlight the parts related to Governing Law.",
     "identify_contract_terms", "applicable_legal_regime", "governing_law"),
    ("legal", "The agreement contains many terms.\n\nHighlight the parts related to License Grant.",
     "assess_rights_restrictions", "licensed_use_authorization", "license_grant"),
    ("legal", "The agreement contains many terms.\n\nHighlight the parts related to Termination for Convenience.",
     "assess_obligations_remedies", "elective_exit_right", "termination_convenience"),
    ("code", "public Object getValue() { return items; }\n\nReturn the HashMap value.",
     "retrieve_inspect", "hashed_key_value_storage", "hash_map"),
    ("code", "public void append(Object value) { }\n\nAdd a String to the buffer.",
     "modify_transform", "immutable_text_handling", "string"),
    ("code", "public boolean isEmpty() { return false; }\n\nCheck whether the array is empty.",
     "validate_control", "indexed_sequence_handling", "array"),
]


@pytest.mark.parametrize("domain,text,intent,attribute,keyword", EXAMPLES)
def test_three_grounded_intents_and_specific_subjects_per_domain(domain, text, intent, attribute, keyword):
    _, compiled, _ = load_taxonomy()
    labels, detail = label_record(row(domain, text), compiled)
    assert labels == dict(intent=intent, attribute=attribute, keyword=keyword)
    assert detail["status"] == "annotated"
    for evidence in detail["evidence"].values():
        assert text[evidence["start"]:evidence["end"]] == evidence["text"]


def test_taxonomy_bounds_meanings_and_expected_coverage():
    taxonomy, _, _ = load_taxonomy()
    assert {domain: len(spec["keyword"]) for domain, spec in taxonomy["domains"].items()} == {
        "medical": 60, "financial": 60, "legal": 60, "code": 60}
    for spec in taxonomy["domains"].values():
        assert len(spec["intent"]) == 3
        assert len(spec["attribute"]) == 60
        assert all(entry["description"] for group in ("intent", "attribute", "keyword") for entry in spec[group])


def test_boundaries_prevent_substrings_and_stale_labels_are_removed(tmp_path):
    records = [row("code", "Return the sprintcounter.", "miss"),
               row("medical", "What is asthma?", "hit")]
    records[0].update(intent="stale", attribute="stale", keyword="stale", labels={"intent": "stale"})
    original = copy.deepcopy(records)
    summary = annotate(records, tmp_path/"annotations")
    annotated = read_jsonl(tmp_path/"annotations/annotated.jsonl")
    assert summary["records"] == 2 and summary["annotated"] == 1 and summary["excluded_semantic"] == 1
    assert annotated[0]["domain"] == "code" and annotated[0]["text"] == "Return the sprintcounter."
    assert not {"intent", "attribute", "keyword", "labels"} & set(annotated[0])
    assert annotated[0]["annotation"]["missing_slots"] == ["attribute", "keyword"]
    assert records == original


def test_observed_vocabulary_and_sidecars_use_only_complete_records(tmp_path):
    records = [row(domain, text, str(i)) for i, (domain, text, *_) in enumerate(EXAMPLES)]
    records.append(row("medical", "What is an unnamed condition?", "excluded"))
    output = tmp_path/"labeled"
    summary = annotate(records, output)
    annotations = json.loads((output/"annotations.json").read_text())
    vocabulary = json.loads((output/"label_vocabulary.json").read_text())
    annotated = read_jsonl(output/"annotated.jsonl")
    assert len(annotations) == 12 and len(annotated) == 13
    assert all(set(value) == {"intent", "attribute", "keyword"} for value in annotations.values())
    assert all(len(vocabulary[domain]["intent"]) == 3 for domain in vocabulary)
    assert summary["class_counts"]["medical"] == dict(intent=3, attribute=60, keyword=60)
    assert summary["observed_class_counts"]["medical"] == dict(intent=3, attribute=3, keyword=3)
    assert all(a["text"] == r["text"] and a["source_id"] == r["source_id"] for a, r in zip(annotated, records))
    assert (output/"label_definitions.json").is_file()
    with pytest.raises(FileExistsError):
        annotate(records, output)


def test_training_coverage_cannot_shrink_the_fixed_vocabulary(tmp_path):
    training = [row("medical", "What is asthma?", "train")]
    testing = [row("medical", "What is asthma?", "test-known")]
    with pytest.raises(ValueError, match="Training-source coverage"):
        annotate(testing, tmp_path/"out", fit_records=training)
    assert not (tmp_path/"out").exists()


def test_request_focus_ignores_context_intent_and_prefers_specific_alias():
    _, compiled, _ = load_taxonomy()
    record = row("financial", "Revenue increased and expenses decreased.\n\nWhat was the operating income?")
    labels, detail = label_record(record, compiled)
    assert labels["intent"] == "amount_calculation"
    assert labels["keyword"] == "operating_income"
    assert detail["evidence"]["keyword"]["text"] == "operating income"
    record = row("medical", "What is high blood pressure?\n\nTreatment and prognosis may vary.")
    labels, _ = label_record(record, compiled)
    assert labels["intent"] == "condition_information" and labels["keyword"] == "hypertension"


def test_code_keyword_prefers_specific_api_and_breaks_equal_length_ties_by_text():
    _, compiled, _ = load_taxonomy()
    record = row("code", "Return a String from the PreparedStatement.")
    labels, detail = label_record(record, compiled)
    assert labels["keyword"] == "prepared_statement"
    assert detail["evidence"]["keyword"]["text"] == "PreparedStatement"
    record = row("code", "Return a queue before a stack.")
    labels, _ = label_record(record, compiled)
    assert labels["keyword"] == "queue"
    record = row("code", "Return a stack before a queue.")
    labels, _ = label_record(record, compiled)
    assert labels["keyword"] == "stack"


def test_incomplete_record_never_creates_an_annotation(tmp_path):
    records = [row("medical", "What is asthma?", "complete"),
               row("medical", "Diabetes appears here without a recognized question.", "partial")]
    annotate(records, tmp_path/"out")
    vocabulary = json.loads((tmp_path/"out/label_vocabulary.json").read_text())
    assert len(vocabulary["medical"]["keyword"]) == 60
    annotations = json.loads((tmp_path/"out/annotations.json").read_text())
    assert set(annotations) == {"medical-complete"}


def test_config_rejects_ambiguous_keyword_aliases(tmp_path):
    taxonomy = yaml.safe_load(DEFAULT_TAXONOMY.read_text(encoding="utf-8"))
    taxonomy["domains"]["medical"]["keyword"][1]["aliases"].append("diabetes mellitus")
    path = tmp_path/"bad.yaml"
    path.write_text(yaml.safe_dump(taxonomy), encoding="utf-8")
    with pytest.raises(ValueError, match="maps to multiple"):
        load_taxonomy(path)


def test_cli_accepts_normalized_jsonl_and_writes_sidecars(tmp_path, capsys):
    path = tmp_path/"normalized.jsonl"
    path.write_text(json.dumps(row("medical", "What is asthma?"))+"\n", encoding="utf-8")
    main(["--input", str(path), "--output", str(tmp_path/"out")])
    assert json.loads(capsys.readouterr().out)["annotated"] == 1
    assert json.loads((tmp_path/"out/annotations.json").read_text())["medical-one"]["keyword"] == "asthma"


def test_every_declared_attribute_and_keyword_has_executable_grounding(tmp_path):
    # Exhaustive short-text rule fixtures are separate from corpus acquisition.
    taxonomy, compiled, _ = load_taxonomy()
    records=[]
    shells={"medical":"What is {}?", "financial":"What is {}?",
            "legal":"Highlight the parts related to {}.", "code":"Return {}."}
    for domain,spec in taxonomy["domains"].items():
        seen_attributes=set()
        for entry in spec["keyword"]:
            text=shells[domain].format(entry["aliases"][0])
            source=row(domain,text,entry["label"])
            labels,detail=label_record(source,compiled)
            assert labels is not None, (domain,entry["label"])
            assert labels["keyword"] == entry["label"], (domain,entry["label"],labels)
            assert labels["attribute"] == entry["attribute"]
            assert labels["attribute"] != labels["keyword"]
            seen_attributes.add(labels["attribute"])
            for evidence in detail["evidence"].values():
                assert source["text"][evidence["start"]:evidence["end"]] == evidence["text"]
            records.append(source)
        assert len(seen_attributes) == 60
    records += [row(domain,text,"intent-example-"+str(i))
                for i,(domain,text,*_) in enumerate(EXAMPLES)]
    summary=annotate(records,tmp_path/"out",fit_records=records)
    assert summary["training_coverage_checked"] is True
    assert all(counts == dict(intent=3,attribute=60,keyword=60)
               for counts in summary["observed_class_counts"].values())
