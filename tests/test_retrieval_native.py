"""Numerical contracts for the native index and Qwen cache integration."""
import json
import os
from pathlib import Path
from types import MethodType
import numpy as np
import pytest
from backends.retrievalattention.budget import budget


def test_generation_context_limit_counts_only_tokens_consumed_by_decoder():
    from types import SimpleNamespace
    from backends.retrievalattention.runtime import Runtime
    runtime = Runtime.__new__(Runtime)
    runtime.tokenizer = SimpleNamespace(encode=lambda prompt, **kwargs: list(range(64)))
    runtime.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=65))
    runtime.config = dict(cache=dict(block_tokens=16, residency_fraction=.2, selected_token_budget=64),
                          inference=dict(generation_tokens=2))
    runtime.layers, runtime.model_id = 2, "tiny-qwen"
    assert runtime.describe_request("context")["generation_tokens"] == 2
    with pytest.raises(ValueError, match="position limit"):
        runtime.describe_request("context", 3)


@pytest.mark.parametrize("block,fraction,expected", [(16, .2, 13), (8, .2, 26), (32, .2, 7), (16, .4, 26), (16, .6, 39)])
def test_pool_capacity_changes_physical_page_count(block, fraction, expected):
    layout = budget(1024, block, fraction)
    assert layout["capacity_slots_per_layer"] == expected
    assert len(layout["persistent_blocks"]) == expected
    assert len(set(layout["persistent_blocks"])) == expected


def test_native_faiss_grouped_query_index():
    pytest.importorskip("faiss", exc_type=ImportError)
    from backends.retrievalattention.index import FlatIndex
    rng = np.random.default_rng(9)
    keys, queries = rng.normal(size=(2, 19, 8)), rng.normal(size=(4, 1, 8))
    index = FlatIndex(2, 8)
    index.paraadd(keys, 1)
    distance, ids = index.search(queries, 2, 5, 1)
    for head in range(4):
        scores = keys[head//2] @ queries[head, 0]
        expected = np.argsort(-scores)[:5]
        np.testing.assert_array_equal(ids[head, 0], expected)
        np.testing.assert_allclose(distance[head, 0], scores[expected], atol=1e-5)


def _native(monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.47.1":
        pytest.skip("Qwen adapter pins transformers 4.47.1")
    source = os.environ.get("KVTRAIL_RETRIEVAL_SOURCE")
    if not source:
        pytest.skip("Set KVTRAIL_RETRIEVAL_SOURCE to the pinned MInference checkout")
    from backends.retrievalattention.upstream import load_native
    # Isolate the native index ABI from attention arithmetic. The independent
    # test above exercises FAISS itself when that compiled dependency is usable.
    class IndexOracle:
        def __init__(self, head_num, dim):
            self.keys = None

        @property
        def ntotal(self):
            return self.keys.shape[1]

        def paraadd(self, keys, core):
            self.keys = np.array(keys) if self.keys is None else np.concatenate([self.keys, keys], axis=1)

        def get_index_type(self):
            return "Flat"

        def search(self, queries, groups, top_k, core):
            scores = np.stack([query @ self.keys[h//groups].T for h, query in enumerate(queries)])
            ids = np.argsort(-scores, axis=-1)[..., :top_k]
            return np.take_along_axis(scores, ids, axis=-1), ids
    monkeypatch.setattr("backends.retrievalattention.index.FlatIndex", IndexOracle)
    # A CPU numerical oracle with FlashAttention's tensor/LSE return convention.
    # Production load_native imports the actual installed CUDA FlashAttention.
    def cpu_attention(q, k, v, **kwargs):
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        groups = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        scores = q.float() @ k.float().transpose(-1, -2) / q.shape[-1]**.5
        output = (scores.softmax(-1) @ v.float()).to(q.dtype)
        return output.transpose(1, 2), scores.logsumexp(-1), None
    return load_native(source, flash_kernel=cpu_attention)


def test_layerwise_prefill_and_native_decode_equal_dense_attention(tmp_path, monkeypatch):
    native = _native(monkeypatch)
    import torch
    from backends.retrievalattention.cache import RetrievalCache
    from experiments.acquisition.metadata import MetadataPublisher
    torch.manual_seed(4)
    config = dict(cache=dict(block_tokens=16, residency_fraction=.2, selected_token_budget=1000))
    q, k, v = torch.randn(1, 4, 65, 8), torch.randn(1, 2, 65, 8), torch.randn(1, 2, 65, 8)
    path = tmp_path / "events.jsonl"
    with MetadataPublisher(path, "request") as publisher:
        cache = RetrievalCache(native, config, 65, 3, 1, publisher)
        actual = cache.prefill(q, k, v, 0)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), is_causal=True)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        assert cache.pools[0][0].shape[-2] == 16
        for _ in range(3):
            dq, dk, dv = torch.randn(1, 4, 1, 8), torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8)
            actual = cache.decode(dq, dk, dv, 0)
            k, v = torch.cat([k, dk], 2), torch.cat([v, dv], 2)
            expected = torch.nn.functional.scaled_dot_product_attention(dq, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1))
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert sum(row["op"] == "LOAD" for row in rows) == 1
    assert sum(row["op"] == "EVICT" for row in rows) == 0
    assert sum(row["op"] == "HIT" for row in rows) == 3
    assert all(row["step"] == 0 for row in rows if row["op"] == "LOAD")


def test_tiny_qwen_installed_attention_matches_dense_model(tmp_path, monkeypatch):
    native = _native(monkeypatch)
    import copy
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from backends.retrievalattention.cache import RetrievalCache
    from backends.retrievalattention.runtime import qwen_attention
    from experiments.acquisition.metadata import MetadataPublisher
    torch.manual_seed(8)
    qconfig = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attn_implementation="eager")
    dense = Qwen2ForCausalLM(qconfig).eval()
    model = copy.deepcopy(dense)
    for layer in model.model.layers:
        layer.self_attn.forward = MethodType(qwen_attention, layer.self_attn)
    config = dict(cache=dict(block_tokens=16, residency_fraction=.2, selected_token_budget=1000))
    tokens = torch.randint(0, 64, (1, 64))
    with MetadataPublisher(tmp_path/"model.jsonl", "qwen") as publisher, torch.inference_mode():
        cache = RetrievalCache(native, config, 64, 2, 2, publisher)
        a, b = model(input_ids=tokens, past_key_values=cache, use_cache=True), dense(input_ids=tokens, use_cache=True)
        torch.testing.assert_close(a.logits, b.logits, atol=3e-6, rtol=3e-5)
        token = b.logits[:, -1:].argmax(-1)
        a = model(input_ids=token, past_key_values=cache, use_cache=True)
        b = dense(input_ids=token, past_key_values=b.past_key_values, use_cache=True)
        torch.testing.assert_close(a.logits, b.logits, atol=3e-6, rtol=3e-5)


def test_fixed_budget_transfers_real_padding_without_changing_attention(tmp_path, monkeypatch):
    native = _native(monkeypatch)
    import torch
    from backends.retrievalattention.cache import RetrievalCache
    from experiments.acquisition.metadata import MetadataPublisher
    torch.manual_seed(20)
    config = dict(cache=dict(block_tokens=16, residency_fraction=.2, selected_token_budget=1000),
                  experiment=dict(defense="fixed_budget"),
                  defenses=dict(fixed_budget=dict(load_pages=65)))
    q, k, v = torch.randn(1, 4, 65, 8), torch.randn(1, 2, 65, 8), torch.randn(1, 2, 65, 8)
    path = tmp_path/"defense.jsonl"
    with MetadataPublisher(path, "defended") as publisher:
        cache = RetrievalCache(native, config, 65, 2, 1, publisher)
        allocate, transfer = cache._allocate_padding, cache._transfer_page
        observed = []
        def allocate_distinct(layer, key, value):
            allocate(layer, key, value)
            for page in range(65):
                cache.padding_cpu[layer][0][:, :, page*16:(page+1)*16].fill_(page+1)
                cache.padding_cpu[layer][1][:, :, page*16:(page+1)*16].fill_(page+2)
        def observe_copy(layer, page, offsets):
            result = transfer(layer, page, offsets)
            if page not in offsets:
                offset = (page-cache.layout["context_blocks"])*16
                for target, source in zip(cache.padding_scratch[layer], cache.padding_cpu[layer]):
                    torch.testing.assert_close(target, source[:, :, offset:offset+16])
                observed.append(page)
            return result
        cache._allocate_padding, cache._transfer_page = allocate_distinct, observe_copy
        actual = cache.prefill(q, k, v, 0)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), is_causal=True)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        assert len(observed) == 64
        assert cache.transfer_counts["required"] == 1
        assert cache.transfer_counts["padding"] == len(observed)
        receipt = cache.allocation_receipt()
        assert receipt["padding_scratch_bytes"] == 2*1*2*16*8*4
        before = dict(cache.transfer_counts)
        cache.decode(torch.randn(1,4,1,8), torch.randn(1,2,1,8), torch.randn(1,2,1,8), 0)
        assert cache.transfer_counts == before
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert sum(r["op"] == "LOAD" for r in records) == 65
    assert all("padding" not in r for r in records)


@pytest.mark.parametrize("tokens,required", [(1024, 13), (4096, 52)])
def test_paper_context_residency_enters_one_real_fixed_budget_batch(tmp_path, monkeypatch, tokens, required):
    native = _native(monkeypatch)
    import torch
    from backends.retrievalattention.cache import RetrievalCache
    from experiments.acquisition.metadata import MetadataPublisher
    config = dict(cache=dict(block_tokens=16, residency_fraction=.2, selected_token_budget=512),
                  experiment=dict(defense="fixed_budget"),
                  defenses=dict(fixed_budget=dict(load_pages=65, block_tokens=16)))
    key = torch.arange(tokens*4, dtype=torch.float32).reshape(1,1,tokens,4)
    value = key+100
    with MetadataPublisher(tmp_path/"paper-context.jsonl", "context") as publisher:
        cache = RetrievalCache(native, config, tokens, 2, 1, publisher)
        cache.cpu_context[0] = (key.clone(), value.clone())
        cache.pools[0] = (torch.empty(1,1,required*16,4), torch.empty(1,1,required*16,4))
        cache._allocate_padding(0,key,value)
        blocks = cache.layout["persistent_blocks"]
        actual_key, actual_value = cache._stage(0, blocks)
        positions = torch.tensor([index for block in blocks for index in range(block*16,(block+1)*16)])
        torch.testing.assert_close(actual_key,key.index_select(2,positions))
        torch.testing.assert_close(actual_value,value.index_select(2,positions))
        assert cache.slots == required and cache.transfer_counts["required"] == required
        assert cache.transfer_counts["padding"] == 65-required
        assert cache.load_batches == 1 and cache.batch_transfer_counts == [65]


@pytest.mark.parametrize("tokens,slots", [(64, 1), (160, 2)])
def test_dynamic_delay_runs_once_before_the_prefill_load_batch(tmp_path, monkeypatch, tokens, slots):
    native = _native(monkeypatch)
    import torch
    from backends.retrievalattention.cache import RetrievalCache
    from experiments.acquisition.metadata import MetadataPublisher
    config = dict(cache=dict(block_tokens=16, residency_fraction=.2, selected_token_budget=1000),
                  experiment=dict(defense="dynamic_timing"),
                  defenses=dict(dynamic_timing=dict(max_delay_ms=5)))
    events = []
    with MetadataPublisher(tmp_path/"delay.jsonl", "delayed") as publisher:
        cache = RetrievalCache(native, config, tokens, 2, 1, publisher)
        execute, transfer = cache.timing.execute, cache._transfer_page
        def delayed(load, *args):
            return execute(load, *args, sleeper=lambda seconds: events.append(("delay", seconds)))
        def copied(*args):
            events.append(("copy", None))
            return transfer(*args)
        cache.timing.execute, cache._transfer_page = delayed, copied
        cache.prefill(torch.randn(1,4,tokens,8), torch.randn(1,2,tokens,8), torch.randn(1,2,tokens,8), 0)
        assert [event for event, _ in events] == ["delay"] + ["copy"]*slots
        assert cache.batch_transfer_counts == [slots]
        assert all(0 <= seconds <= .005 for event, seconds in events if event == "delay")
        count = len(events)
        cache.decode(torch.randn(1,4,1,8), torch.randn(1,2,1,8), torch.randn(1,2,1,8), 0)
        assert len(events) == count
