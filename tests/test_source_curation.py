"""Small artificial source corpora exercise genuine-span curation and isolation."""
import hashlib
import json
import pytest
from datasets.preprocessing.curation import SourceCurator
from datasets.preprocessing.prepare import prepare
from datasets.preprocessing.core import DOMAINS,read_jsonl
from datasets.preprocessing.sources import source_ids
from experiments.run import validate_split
from datasets.preprocessing.rendering import render_prompt


class WordTokenizer:
    def encode(self,text,add_special_tokens=True):
        return [0]*(len(text.split())+int(add_special_tokens))


def _row(identity,domain="medical",n=100):
    request=f"Explain source {identity}."
    text=" ".join(f"{identity}word{i}" for i in range(n))+"\n\n"+request
    return dict(id=identity,source_id=identity,domain=domain,text=text,request=request)


def test_short_source_uses_distinct_real_partition_material():
    rows=[_row(str(i)) for i in range(20)]
    curated=SourceCurator(rows,WordTokenizer()).curate(rows[0],1000,"shared-v1")
    assert curated is not None
    assert rows[0]["request"] in curated["text"]
    assert len(curated["support_source_ids"]) == len(set(curated["support_source_ids"]))
    originals={row["id"]:row for row in rows}
    parts=[]
    for span in curated["source_spans"]:
        original=originals[span["record_id"]]["text"][span["start"]:span["end"]]
        parts.append(original)
        assert original in curated["text"]
        assert hashlib.sha256(original.encode()).hexdigest() == span["sha256"]
    assert "\n\n".join(parts) == curated["text"]


def test_curation_does_not_repeat_duplicate_corpus_text_to_fill_budget():
    row=_row("primary")
    copies=[dict(row,id=str(i),source_id=str(i)) for i in range(30)]
    assert SourceCurator(copies,WordTokenizer()).curate(copies[0],1000,"shared-v1") is None


def test_long_semantic_source_preserves_request_and_every_grounded_span():
    row=_row("long",n=4000)
    labels={"intent":"explanation","attribute":"condition","keyword":"sensitive term"}
    evidence={}
    for slot,term in zip(labels,("longword20","longword2000","longword3500")):
        start=row["text"].index(term)
        evidence[slot]=dict(start=start,end=start+len(term),text=term)
    row.update(labels,annotation=dict(evidence=evidence))
    support=[dict(_row(str(i)),**labels) for i in range(20)]
    curator=SourceCurator([row]+support,WordTokenizer(),semantic=True)
    result=curator.curate(row,1000,"shared-v1")
    assert result is not None and row["request"] in result["text"]
    assert all(item["text"] in result["text"] for item in evidence.values())
    assert all(label not in result["text"] for label in labels.values())


def test_long_single_source_class_uses_available_context_without_supplements():
    row=_row("rare",n=4000)
    labels=dict(intent="explanation",attribute="condition",keyword="topic")
    evidence={}
    for slot,term in zip(labels,("rareword20","rareword2000","rareword3500")):
        start=row["text"].index(term)
        evidence[slot]=dict(start=start,end=start+len(term),text=term)
    row.update(labels,annotation=dict(evidence=evidence))
    curator=SourceCurator([row],WordTokenizer(),semantic=True)
    result=curator.curate(row,1000,"shared-v1")
    assert result is not None and not result["support_source_ids"]
    assert 900 <= curator.count(result["text"],"shared-v1") <= 1100


def test_support_material_cannot_change_domain_or_semantic_target():
    row=_row("medical-primary")
    cross_domain=[_row(f"code-{i}","code") for i in range(40)]
    assert SourceCurator([row]+cross_domain,WordTokenizer()).curate(row,1000,"shared-v1") is None
    row.update(intent="explain",attribute="source",keyword="primary")
    conflicts=[dict(_row(f"other-{i}"),intent="repair",attribute="source",keyword="other") for i in range(40)]
    assert SourceCurator([row]+conflicts,WordTokenizer(),semantic=True).curate(row,1000,"shared-v1") is None


def test_semantic_source_evidence_is_checked_before_short_source_expansion():
    row=_row("source")
    row.update(intent="explain",attribute="source",keyword="source",
        annotation=dict(evidence={"keyword":dict(start=0,end=3,text="bad")}))
    with pytest.raises(ValueError,match="exact original"):
        SourceCurator([row],WordTokenizer(),semantic=True).curate(row,1000,"shared-v1")


def test_supplementary_sources_participate_in_attack_train_test_isolation():
    train=[dict(sample_id="train",source_id="one",support_source_ids=["shared"],provenance="synthetic")]
    test=[dict(sample_id="test",source_id="two",support_source_ids=["shared"],provenance="synthetic")]
    with pytest.raises(ValueError,match="source-record leakage"):
        validate_split(train,test)


def test_preparation_curates_all_lengths_without_cross_partition_sources(tmp_path):
    rows=[_row(f"{domain}-{i}",domain,n=400) for domain in DOMAINS.values() for i in range(48)]
    result=prepare(rows,{"quest":WordTokenizer()},tmp_path/"prepared",curate_lengths=True,
        train_per_domain=3,test_per_domain=2,profiling_per_domain=4,auxiliary_per_domain=1)
    for section in ("datasets","contexts","profiling"):
        for pair in result[section]["quest"].values():
            train,test=read_jsonl(pair["train"]),read_jsonl(pair["test"])
            assert set().union(*(source_ids(row) for row in train)).isdisjoint(
                set().union(*(source_ids(row) for row in test)))
            assert all(.9*row["target_tokens"] <= row["token_count"] <= 1.1*row["target_tokens"] for row in train+test)


@pytest.mark.parametrize("template", ["shared-v1", "shared-v2", "unstructured"])
def test_cached_curation_preserves_exact_text_spans_and_actual_counts(template):
    class Uncached(SourceCurator):
        def count(self,text,template_id,record=None):
            return len(self.tokenizer.encode(render_prompt(text,template_id,record=record),add_special_tokens=True))
    rows=[_row(str(i)) for i in range(48)]+[_row("long",n=5000)]
    cached=SourceCurator(rows,WordTokenizer())
    uncached=Uncached(rows,WordTokenizer())
    for row in (rows[0],rows[-1]):
        for target in (1000,3000):
            actual=cached.curate(row,target,template)
            expected=uncached.curate(row,target,template)
            assert json.dumps(actual,sort_keys=True).encode() == json.dumps(expected,sort_keys=True).encode()
            assert actual is not None
            rendered=render_prompt(actual["text"],template,record=row)
            assert cached.count(actual["text"],template,row) == len(WordTokenizer().encode(rendered))


def test_cache_keys_include_actual_record_specific_rendering(monkeypatch):
    calls=[]
    def rendered(text,template_id,record=None):
        calls.append(record["id"])
        return text+" "+record["id"]
    monkeypatch.setattr("datasets.preprocessing.curation.render_prompt",rendered)
    class Counted(WordTokenizer):
        encoded=0
        def encode(self,text,**kwargs):
            self.encoded+=1
            return super().encode(text,**kwargs)
    tokenizer=Counted()
    curator=SourceCurator([_row("one")],tokenizer)
    assert curator.count("text","unstructured",{"id":"one"}) == 3
    assert curator.count("text","unstructured",{"id":"two words"}) == 4
    assert curator.count("text","unstructured",{"id":"one"}) == 3
    assert tokenizer.encoded == 2
    assert calls == ["one","two words","one"]


def test_exact_cache_respects_entry_and_text_limits():
    from datasets.preprocessing.curation import _TextCache
    cache=_TextCache(max_entries=2,max_chars=5)
    assert cache.get("ab",lambda:1) == 1
    assert cache.get("cd",lambda:2) == 2
    assert cache.get("ab",lambda:pytest.fail("repeat should use exact cached value")) == 1
    assert cache.get("ef",lambda:3) == 3
    assert list(cache.values) == ["ab","ef"] and cache.chars == 4
    assert cache.get("too large",lambda:4) == 4
    assert "too large" not in cache.values
    assert cache.get("xxx",lambda:5) == 5
    assert cache.chars <= 5 and len(cache.values) <= 2


def test_curator_passes_primary_record_to_every_rendered_token_count(monkeypatch):
    seen=[]
    def rendered(text,template_id,record=None):
        seen.append(None if record is None else record["id"])
        return text
    monkeypatch.setattr("datasets.preprocessing.curation.render_prompt",rendered)
    rows=[_row(str(i)) for i in range(48)]
    assert SourceCurator(rows,WordTokenizer()).curate(rows[0],1000,"unstructured") is not None
    assert seen and set(seen) == {rows[0]["id"]}


def test_concurrent_cache_misses_keep_exact_text_accounting():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from datasets.preprocessing.curation import _TextCache
    cache=_TextCache(max_entries=3,max_chars=12)
    barrier=Barrier(2)
    def compute():
        barrier.wait(timeout=5)
        return 17
    with ThreadPoolExecutor(max_workers=2) as executor:
        results=list(executor.map(lambda _:cache.get("same text",compute),range(2)))
    assert results == [17,17]
    assert cache.chars == len("same text") and len(cache.values) == 1
    assert cache.get("second",lambda:18) == 18
    assert cache.chars == len("second")
