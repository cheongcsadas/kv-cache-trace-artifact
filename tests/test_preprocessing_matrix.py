"""Synthetic small rows exercise the matrix validator's rejection boundaries."""
from copy import deepcopy
import hashlib
import json

import pytest

from datasets.preprocessing import matrix


def rows(partition, count, *, target=1000, domains=matrix.DOMAINS):
    return [dict(id=f"{domain}-{partition}-{i}", source_id=f"{domain}-{partition}-{i}",
                 domain=domain, intent=f"i{i % 3}", attribute=f"a{i % 60}", keyword=f"k{i % 60}",
                 token_count=target, context_tokens=target, target_tokens=target)
            for domain in domains for i in range(count)]


def condition(*, target=1000, domains=matrix.DOMAINS):
    train, test = rows("train", 60, target=target, domains=domains), rows("test", 5, target=target, domains=domains)
    profiling = deepcopy(train) + rows("extra", 3, target=target, domains=domains)
    return train, test, profiling


def summarize(train, test, profiling=None, **kwargs):
    return matrix.summarize_condition(train, test, profiling, system="quest",
        setting=kwargs.pop("setting", "templated_short"), train_per_domain=60, test_per_domain=5,
        profiling_per_domain=63, **kwargs)


def test_per_domain_measured_counts_and_semantic_coverage():
    result = summarize(*condition())
    assert set(result["domains"]) == set(matrix.DOMAINS)
    assert result["source_disjoint"]
    for domain in result["domains"].values():
        assert domain["train"]["count"] == 60
        assert domain["train"]["class_counts"] == {"intent": 3, "attribute": 60, "keyword": 60}
        assert domain["test"]["token_mean"] == 1000
        assert domain["profiling"]["count"] == 63


def test_missing_domain_is_rejected_even_when_other_domains_satisfy_quota():
    train, test, _ = condition()
    with pytest.raises(ValueError, match="missing domains"):
        summarize([row for row in train if row["domain"] != "medical"], test)


def test_overall_mean_cannot_hide_two_bad_domains():
    train, test, _ = condition()
    for row in train:
        if row["domain"] in ("medical", "financial"):
            row["token_count"] = row["context_tokens"] = 800 if row["domain"] == "medical" else 1200
    assert sum(row["token_count"] for row in train) / len(train) == 1000
    with pytest.raises(ValueError, match="medical/train: token mean"):
        summarize(train, test)


def test_missing_training_class_and_closed_set_test_are_rejected():
    train, test, _ = condition()
    train[59]["attribute"] = "a0"
    with pytest.raises(ValueError, match="attribute: expected 60"):
        summarize(train, test)
    train, test, _ = condition()
    test[0]["keyword"] = "unknown"
    with pytest.raises(ValueError, match="fixed training vocabulary"):
        summarize(train, test)


@pytest.mark.parametrize("partition", ["train", "profiling"])
def test_supplementary_test_source_leak_is_rejected(partition):
    train, test, profiling = condition()
    selected = train if partition == "train" else profiling
    selected[0]["support_source_ids"] = [test[-1]["source_id"]]
    with pytest.raises(ValueError, match="supplementary sources overlap"):
        summarize(train, test, profiling)


def test_omitted_support_source_cannot_hide_in_spans():
    train, test, _ = condition()
    train[0]["source_spans"] = [{"source_id": train[0]["source_id"]}, {"source_id": test[0]["source_id"]}]
    with pytest.raises(ValueError, match="Source spans disagree"):
        summarize(train, test)


def test_duplicate_record_ids_cannot_fill_quota():
    train, test, _ = condition()
    train[1]["id"] = train[0]["id"]
    with pytest.raises(ValueError, match="duplicate record ID"):
        summarize(train, test)


def test_profiling_must_contain_unchanged_training_rows():
    train, test, profiling = condition()
    profiling[0]["prompt_sha256"] = "changed"
    with pytest.raises(ValueError, match="copies disagree"):
        summarize(train, test, profiling)


def test_nested_labels_must_agree_and_all_slots_are_required():
    train, test, _ = condition()
    train[0]["labels"] = {"domain": "code"}
    with pytest.raises(ValueError, match="labels disagree"):
        summarize(train, test)
    train[0].pop("labels")
    train[0].pop("keyword")
    with pytest.raises(ValueError, match="all three nonempty labels"):
        summarize(train, test)


def test_cross_template_default_is_explicit_artifact_choice():
    result = summarize(*condition(), setting="cross_template")
    assert result["target_tokens"] == 1000 and result["target_basis"] == "artifact_choice"
    result = summarize(*condition(target=3000), setting="unstructured_long")
    assert result["target_tokens"] == 3000 and result["target_basis"] == "paper"


def test_single_domain_audit_does_not_certify_complete_matrix():
    result = summarize(*condition(domains=("medical",)), domains=("medical",))
    assert set(result["domains"]) == {"medical"}
    with pytest.raises(ValueError, match="three tokenizer descriptions"):
        matrix.validate_complete_matrix({"conditions": [result]})


class SmallTokenizer:
    is_fast = False
    special_tokens_map_extended = {"bos_token": "<s>"}
    bos_token_id = 1
    add_bos_token = True


def test_small_tokenizer_metadata_is_explicitly_unverified():
    result = matrix.describe_tokenizer(SmallTokenizer(), system="quest")
    assert result["assets"] == {} and not result["identity_verified"]
    assert result["special_tokens"] == {"bos_token": "<s>"}
    assert result["special_token_ids"]["bos_token_id"] == 1
    assert result["fingerprint"]


def test_misnamed_model_directory_does_not_override_asset_identity(tmp_path):
    directory = tmp_path / "llama-2-7b-hf"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"model_type": "mistral"}))
    tokenizer = SmallTokenizer()
    tokenizer.name_or_path = str(directory)
    with pytest.raises(ValueError, match="model identity mismatch"):
        matrix.describe_tokenizer(tokenizer, system="quest")


def test_wrong_mistral_checkpoint_name_is_rejected(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mistral",
        "_name_or_path": "mistralai/Mistral-7B-Instruct-v0.3"}))
    tokenizer = SmallTokenizer()
    tokenizer.name_or_path = str(tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        matrix.describe_tokenizer(tokenizer, system="infinigen")


def test_hash_only_small_assets_and_capture_actual_configuration(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "tokenizer.json").write_text('{}')
    (tmp_path / "model.safetensors").write_bytes(b"never included")
    tokenizer = SmallTokenizer()
    tokenizer.name_or_path = str(tmp_path)
    first = matrix.describe_tokenizer(tokenizer, system="quest")
    assert set(first["assets"]) == {"config.json", "tokenizer.json"}
    assert first["assets"]["tokenizer.json"]["sha256"] == hashlib.sha256(b'{}').hexdigest()
    assert not first["identity_verified"]
    tokenizer.add_bos_token = False
    second = matrix.describe_tokenizer(tokenizer, system="quest")
    assert first["fingerprint"] != second["fingerprint"]


def _matrix():
    # Structural validation fixture, not a downloaded model/corpus assertion.
    descriptions = {}
    for system, (identity, model_type, revision) in matrix.MODEL_IDENTITIES.items():
        item = {"system": system, "model_id": identity, "model_type": model_type, "revision": revision,
                "identity_verified": True, "backend_sha256": "synthetic-backend",
                "versions": {"transformers": matrix.TRANSFORMERS_VERSIONS[system]},
                "assets": {name: {"sha256": sha} for name, sha in matrix.IDENTITY_HASHES[system].items()}}
        item["fingerprint"] = matrix._digest(item)
        descriptions[system] = item
    conditions = []
    for system in matrix.SYSTEMS:
        for setting, target in matrix.SETTINGS.items():
            item = summarize(*condition(target=target), setting=setting)
            item["system"] = system
            item["tokenizer_fingerprint"] = descriptions[system]["fingerprint"]
            for domain in item["domains"].values():
                for split, count in (("train", 3000), ("test", 1000), ("profiling", 3200)):
                    domain[split]["count"] = count
            conditions.append(item)
    return {"conditions": conditions, "tokenizers": descriptions}


def test_complete_matrix_requires_exactly_sixty_domain_conditions():
    summary = _matrix()
    assert matrix.validate_complete_matrix(summary) == {"systems": 3, "settings": 5, "domains": 4, "conditions": 60}
    summary["conditions"].pop()
    with pytest.raises(ValueError, match="Incomplete matrix"):
        matrix.validate_complete_matrix(summary)


@pytest.mark.parametrize("mutation,match", [
    (lambda s: s["conditions"].append(deepcopy(s["conditions"][0])), "duplicate"),
    (lambda s: s["conditions"][0]["domains"].pop("code"), "four domains"),
    (lambda s: s["conditions"][0]["domains"]["legal"]["train"].update(token_mean=800), "token mean"),
    (lambda s: s["conditions"][0]["domains"]["legal"]["train"].update(count=2999), "quota"),
    (lambda s: s["conditions"][0].update(tokenizer_fingerprint="wrong"), "fingerprint"),
    (lambda s: s["tokenizers"]["quest"].update(identity_verified=False), "official tokenizer"),
    (lambda s: s["tokenizers"]["quest"]["assets"]["tokenizer.json"].update(sha256="wrong"), "official tokenizer"),
    (lambda s: s["tokenizers"]["quest"]["versions"].update(transformers="4.57.1"), "native Transformers"),
])
def test_incomplete_or_inconsistent_matrix_is_rejected(mutation, match):
    summary = _matrix()
    mutation(summary)
    with pytest.raises(ValueError, match=match):
        matrix.validate_complete_matrix(summary)


def test_complete_matrix_requires_one_fixed_vocabulary_across_conditions():
    summary = _matrix()
    domain = summary["conditions"][1]["domains"]["medical"]
    for partition in ("train", "profiling"):
        domain[partition]["classes"]["attribute"].remove("a59")
        domain[partition]["classes"]["attribute"].append("alternate")
    with pytest.raises(ValueError, match="same fixed domain vocabulary"):
        matrix.validate_complete_matrix(summary)
