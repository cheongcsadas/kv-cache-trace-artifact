import json
from pathlib import Path
import zipfile
import pytest
from datasets.acquire import extract_zip, normalize_download, safe_relative
from datasets.preprocessing.prepare import prepare, measured_candidates, select_quota, join_annotations, fit_common_vocabulary
from datasets.preprocessing.core import DOMAINS, read_jsonl


@pytest.mark.parametrize("name", ["../escape", "x/../../escape", "C:/escape", "x\\..\\escape", "/absolute", "x:stream"])
def test_corpus_archive_path_rejection(name):
    with pytest.raises(ValueError):
        safe_relative(name)


def test_archive_extraction_rejects_escape(tmp_path):
    archive = tmp_path/"bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError):
        extract_zip(archive, tmp_path/"unpacked")
    assert not (tmp_path/"outside.txt").exists()


def test_concode_jsonl_suffix_is_normalized(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    (source/"train.json").write_text(json.dumps({"nl":"return zero", "code":"int f(){return 0;}"})+"\n")
    destination = tmp_path/"normalized.jsonl"
    count = normalize_download("codexglue", source,
        dict(domain="code", inputs=["train.json"]), destination)
    assert count == 1
    row = read_jsonl(destination)[0]
    assert row["domain"] == "code" and "return zero" in row["text"] and "int f" in row["text"]


class WordTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [0] * (len(text.split()) + int(add_special_tokens))


def test_live_bridge_real_token_counts_source_separation_and_quotas(tmp_path):
    rows = [dict(id=f"{d}-{i}", source_id=f"{d}-source-{i}", domain=d,
                 text="source "*1000) for d in DOMAINS.values() for i in range(24)]
    result = prepare(rows, {"quest": WordTokenizer()}, tmp_path/"prepared",
        train_per_domain=3, test_per_domain=2, profiling_per_domain=4, auxiliary_per_domain=1,
        settings=["templated_short", "cross_template"], context_targets=(1000,))
    pair = result["datasets"]["quest"]["cross_template"]
    train, test = read_jsonl(pair["train"]), read_jsonl(pair["test"])
    auxiliary = read_jsonl(result["auxiliary"]["quest"])
    assert len(train) == 12 and len(test) == 8 and len(auxiliary) == 4
    assert len(read_jsonl(result["profiling"]["quest"]["cross_template"]["train"])) == 16
    assert {r["source_id"] for r in train}.isdisjoint(r["source_id"] for r in test+auxiliary)
    assert {r["template_id"] for r in train} == {"shared-v1"}
    assert {r["template_id"] for r in test} == {"shared-v2"}
    assert all(row["token_count"] == len(WordTokenizer().encode(row["prompt"])) for row in train+test)


def test_length_selection_never_pads_short_sources():
    rows = [dict(id="small", source_id="source", domain="medical", text="short text")]
    assert measured_candidates(rows, WordTokenizer(), "templated_long", 3000, "shared-v1", .1) == []


def test_auxiliary_pool_uses_unique_sources_shared_by_model_tokenizers(tmp_path):
    rows = [dict(id=f"{d}-{i}-{j}", source_id=f"{d}-source-{i}", domain=d,
                 text="source "*1000) for d in DOMAINS.values() for i in range(24) for j in range(3)]
    result = prepare(rows, {"quest": WordTokenizer(), "infinigen": WordTokenizer()}, tmp_path/"prepared",
        train_per_domain=3, test_per_domain=2, profiling_per_domain=4, auxiliary_per_domain=2,
        settings=["templated_short"], context_targets=())
    auxiliary = [read_jsonl(result["auxiliary"][system]) for system in ("quest", "infinigen")]
    assert all(len({row["source_id"] for row in values}) == 8 for values in auxiliary)
    assert {r["id"] for r in auxiliary[0]} == {r["id"] for r in auxiliary[1]}
    for system in ("quest", "infinigen"):
        pair = result["datasets"][system]["templated_short"]
        chosen = read_jsonl(pair["train"]) + read_jsonl(pair["test"])
        assert {r["source_id"] for r in chosen}.isdisjoint(r["source_id"] for r in auxiliary[0])


def test_semantic_labels_are_required(tmp_path):
    rows = [dict(id=f"{d}-{i}", source_id=f"{d}-{i}", domain=d, text="material")
            for d in DOMAINS.values() for i in range(4)]
    with pytest.raises(ValueError, match="annotation"):
        prepare(rows, {"quest": WordTokenizer()}, tmp_path/"output", semantic=True)
    assert not (tmp_path/"output").exists()


def test_full_data_bridge_supplies_each_system_and_every_prompt_condition(tmp_path):
    rows=[dict(id=f"{d}-{i}",source_id=f"{d}-{i}",domain=d,text="source "*1000)
          for d in DOMAINS.values() for i in range(24)]
    variants={r["id"]:{setting:"source "*length for setting,length in
        (("templated_long",3000),("unstructured_long",3000),("context_2000",2000),
         ("context_3000",3000),("context_4000",4000))} for r in rows}
    result=prepare(rows,{s:WordTokenizer() for s in ("quest","infinigen","retrievalattention")},
        tmp_path/"prepared",variants=variants,train_per_domain=3,test_per_domain=2,
        profiling_per_domain=4,auxiliary_per_domain=1)
    for system in result["datasets"]:
        assert len(result["datasets"][system]) == 5
        assert set(result["contexts"][system]) == {"1000","2000","3000","4000"}
        assert len(read_jsonl(result["transfer_test"][system])) == 8
        for section in ("datasets","contexts","profiling"):
            for pair in result[section][system].values():
                train,test=read_jsonl(pair["train"]),read_jsonl(pair["test"])
                assert {r["source_id"] for r in train}.isdisjoint(r["source_id"] for r in test)
                assert all(r["token_count"] == len(WordTokenizer().encode(r["prompt"])) for r in train+test)


def full_vocabulary():
    return {d:{slot:[f"{slot}-{i}" for i in range(n)]
               for slot,n in (("intent",3),("attribute",60),("keyword",60))} for d in DOMAINS.values()}


def test_explicit_semantic_vocabulary_controls_prepared_training_labels(tmp_path):
    from datasets.preprocessing.annotate import load_taxonomy
    from datasets.preprocessing.grounding import validate_grounding_receipt
    taxonomy, _, _ = load_taxonomy()
    rows, labels = [], {}
    vocabulary = {domain: {slot: [entry["label"] for entry in spec[slot]]
                          for slot in ("intent", "attribute", "keyword")}
                  for domain, spec in taxonomy["domains"].items()}
    # Artificial protocol fixtures use real text rules, not arbitrary labels.
    for domain, spec in taxonomy["domains"].items():
        for i, keyword in enumerate(spec["keyword"]):
            intent = spec["intent"][i % 3]
            request = intent["aliases"][0] + " " + keyword["aliases"][0]
            for j in range(8):
                identity = f"{domain}-{i}-{j}"
                rows.append(dict(id=identity, source_id=identity, domain=domain,
                                 text="source "*1000+"\n\n"+request, request=request))
                labels[identity] = dict(intent=intent["label"], attribute=keyword["attribute"], keyword=keyword["label"])
    result=prepare(rows,{"quest":WordTokenizer()},tmp_path/"prepared",annotations=labels,
        label_vocabulary=vocabulary,semantic=True,fit_vocabulary=True,settings=["templated_short"],context_targets=(),
        train_per_domain=60,test_per_domain=20,profiling_per_domain=64,auxiliary_per_domain=1)
    assert result["label_vocabulary"] == vocabulary
    for domain in DOMAINS.values():
        selected=[r for r in read_jsonl(result["datasets"]["quest"]["templated_short"]["train"]) if r["domain"] == domain]
        for row in selected:
            validate_grounding_receipt(row)
        for slot in ("intent","attribute","keyword"):
            assert {r["labels"][slot] for r in selected} == set(vocabulary[domain][slot])


def test_reduced_semantic_vocabulary_is_rejected_before_data_output(tmp_path):
    rows=[dict(id=f"{d}-{i}",source_id=f"{d}-{i}",domain=d,text="source")
          for d in DOMAINS.values() for i in range(8)]
    vocab=full_vocabulary()
    vocab["medical"]["attribute"].pop()
    with pytest.raises(ValueError,match="exactly 60"):
        prepare(rows,{"quest":WordTokenizer()},tmp_path/"out",label_vocabulary=vocab)
    assert not (tmp_path/"out").exists()


def test_semantic_annotation_shapes_and_label_types_are_checked():
    rows=[dict(id="record",domain="medical")]
    with pytest.raises(ValueError,match="supply intent"):
        join_annotations(rows,{"record":dict(intent="consult")},require_complete=True)
    with pytest.raises(ValueError,match="string"):
        join_annotations(rows,{"record":dict(keyword=["one","two"])})


def test_fitted_vocabulary_audits_coverage_without_reducing_classes():
    vocab=full_vocabulary()
    short={domain:{(f"intent-{i%3}",f"attribute-{i}",f"keyword-{i}") for i in range(60)} for domain in DOMAINS.values()}
    assert fit_common_vocabulary(vocab,[short,short]) == vocab
    long={domain:set(triples) for domain,triples in short.items()}
    long["medical"].remove(("intent-2","attribute-59","keyword-59"))
    with pytest.raises(ValueError,match="must cover all 60"):
        fit_common_vocabulary(vocab,[short,long])


def test_test_quota_can_use_closed_subset_of_declared_semantic_classes():
    vocabulary=full_vocabulary()
    candidates=[dict(id=f"{domain}-{i}",domain=domain,intent="intent-0",attribute="attribute-0",keyword="keyword-0")
                for domain in DOMAINS.values() for i in range(3)]
    assert len(select_quota(candidates,2,42,True,vocabulary,require_coverage=False)) == 8
    with pytest.raises(ValueError,match="cover the declared"):
        select_quota(candidates,2,42,True,vocabulary)
