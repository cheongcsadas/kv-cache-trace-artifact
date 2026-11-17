"""Prevent shared word fragments from assigning unrelated medical properties."""
import hashlib

import pytest

from datasets.preprocessing.annotate import label_record, load_taxonomy
from datasets.preprocessing.grounding import (validate_grounded_record, validate_grounding_receipt,
                                              _receipt_digest)


@pytest.mark.parametrize("question,keyword,attribute", [
    ("How to prevent Diabetes insipidus - nephrogenic ?", "diabetes_mellitus", "glucose_homeostasis"),
    ("What is intracranial hypertension?", "hypertension", "elevated_arterial_pressure"),
    ("What is portal hypertension?", "hypertension", "elevated_arterial_pressure"),
    ("What is lupus pernio?", "lupus", "systemic_autoimmune_involvement"),
    ("What is discoid lupus erythematosus?", "lupus", "systemic_autoimmune_involvement"),
    ("What is stomach flu?", "influenza", "seasonal_respiratory_viral_infection"),
    ("What is the non-epileptic form?", "epilepsy", "recurrent_seizure_disorder"),
])
def test_ambiguous_shared_words_are_not_sufficient_evidence(question, keyword, attribute):
    row = dict(id="fixture", source_id="fixture-source", domain="medical", text=question, request=question)
    labels, detail = label_record(row, load_taxonomy()[1])
    assert labels is None and detail["status"] == "excluded_semantic"
    row.update(intent="treatment_prevention" if "prevent" in question else "condition_information",
               keyword=keyword, attribute=attribute)
    with pytest.raises(ValueError, match="no evidence|explicitly excludes"):
        validate_grounded_record(row)


@pytest.mark.parametrize("term,keyword", [
    ("diabetes mellitus", "diabetes_mellitus"), ("type 1 diabetes", "diabetes_mellitus"),
    ("type 2 diabetes", "diabetes_mellitus"), ("high blood pressure", "hypertension"),
    ("arterial hypertension", "hypertension"), ("systemic lupus erythematosus", "lupus"),
    ("influenza", "influenza"), ("epilepsy", "epilepsy"),
])
def test_specific_medical_terms_remain_grounded_through_collection_receipt(term, keyword):
    text = f"What is {term}?"
    row = dict(id="fixture", source_id="fixture-source", domain="medical", text=text, request=text)
    labels, _ = label_record(row, load_taxonomy()[1])
    assert labels["keyword"] == keyword
    row.update(labels)
    row["annotation"] = validate_grounded_record(row)
    row["prompt_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    validate_grounding_receipt(row)


def test_old_bare_diabetes_evidence_cannot_be_rehashed_under_new_taxonomy():
    text = "What is diabetes mellitus?"
    row = dict(id="fixture", source_id="fixture-source", domain="medical", text=text, request=text)
    row.update(label_record(row, load_taxonomy()[1])[0])
    receipt = validate_grounded_record(row)
    for field in ("evidence", "prompt_evidence"):
        for slot in ("keyword", "attribute"):
            receipt[field][slot]["text"] = "diabetes"
            receipt[field][slot]["end"] = receipt[field][slot]["start"] + len("diabetes")
    receipt["receipt_sha256"] = _receipt_digest(receipt)
    row.update(annotation=receipt, prompt_sha256=receipt["prompt_sha256"])
    with pytest.raises(ValueError, match="does not justify"):
        validate_grounding_receipt(row)


@pytest.mark.parametrize("question,keyword,attribute", [
    ("What is (are) Acquired hemophilia ?", "hemophilia", "impaired_coagulation_factor_function"),
    ("What is (are) Acquired hemophilia A ?", "hemophilia", "impaired_coagulation_factor_function"),
    ("What is (are) Cryptogenic organizing pneumonia ?", "pneumonia", "pulmonary_inflammation"),
])
def test_medical_properties_do_not_assert_unsupported_inheritance_or_infection(question, keyword, attribute):
    row = dict(id="fixture", source_id="fixture-source", domain="medical", text=question, request=question)
    labels, _ = label_record(row, load_taxonomy()[1])
    assert labels["keyword"] == keyword and labels["attribute"] == attribute
    row.update(labels)
    assert validate_grounded_record(row)["labels"]["attribute"] == attribute


def test_insipidus_question_cannot_fall_back_to_mellitus_in_a_contrast_sentence():
    question = "What is (are) nephrogenic diabetes insipidus ?"
    text = question + "\n\nDiabetes mellitus is a different disorder."
    row = dict(id="fixture", source_id="fixture-source", domain="medical", text=text, request=question)
    labels, _ = label_record(row, load_taxonomy()[1])
    assert labels is None
    row.update(intent="condition_information", keyword="diabetes_mellitus", attribute="glucose_homeostasis")
    with pytest.raises(ValueError, match="explicitly excludes"):
        validate_grounded_record(row)


def test_mellitus_question_preserves_disambiguation_in_metadata_only_receipt():
    question = "What is diabetes mellitus?"
    text = question + "\n\nThis differs from diabetes insipidus."
    row = dict(id="fixture", source_id="fixture-source", domain="medical", text=text, request=question)
    row.update(label_record(row, load_taxonomy()[1])[0])
    receipt = validate_grounded_record(row)
    assert receipt["request_evidence"]["text"] == question
    row.update(annotation=receipt, prompt_sha256=receipt["prompt_sha256"])
    validate_grounding_receipt(row)
    row.pop("text")
    for field in ("request_evidence", "prompt_request_evidence"):
        receipt[field].update(text="What is diabetes insipidus?", start=0, end=len("What is diabetes insipidus?"))
    receipt["receipt_sha256"] = _receipt_digest(receipt)
    with pytest.raises(ValueError, match="explicitly excludes"):
        validate_grounding_receipt(row)


def test_semantic_variant_cannot_discard_the_disambiguating_request():
    question = "What is diabetes mellitus?"
    row = dict(id="fixture", source_id="fixture-source", domain="medical", text=question, request=question)
    row.update(label_record(row, load_taxonomy()[1])[0])
    with pytest.raises(ValueError, match="request disambiguation"):
        validate_grounded_record(row, prompt="What is the condition called diabetes mellitus?")
