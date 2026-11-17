"""Numerical checks for physical slot bounds and the Mistral predictor port."""
import os
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")
from backends.infinigen.cache import LayerPool
from backends.infinigen.predictor import rotary, Predictor
from backends.infinigen.upstream import load_primitives
from experiments.acquisition.metadata import MetadataPublisher, BackendHooks


def test_bounded_pool_matches_selected_attention_and_exports(tmp_path):
    torch.manual_seed(3)
    key, value = torch.randn(2, 64, 8), torch.randn(2, 64, 8)
    query = torch.randn(4, 8)
    indices = torch.tensor([[0, 16, 32, 48], [1, 17, 33, 49], [2, 18, 34, 50], [3, 19, 35, 51]])
    new_key, new_value = torch.randn(2, 8), torch.randn(2, 8)
    path = tmp_path / "operations.jsonl"
    with MetadataPublisher(path, "numerical") as publisher:
        pool = LayerPool(layer=0, kv_heads=2, head_dim=8, context_tokens=64,
                         maximum_tokens=66, block_tokens=16, residency_fraction=.2,
                         dtype=torch.float32, device="cpu", hooks=BackendHooks(publisher, "infinigen"))
        pool.append(key, value)
        assert pool.loads == 0
        pool.prefetch(indices, groups=2, step=0)
        output = pool.attend(query, indices, 0, groups=2, current_key=new_key, current_value=new_value)
        expected = []
        for head in range(4):
            selected_k = torch.cat((key[head//2, indices[head]], new_key[head//2, None]))
            selected_v = torch.cat((value[head//2, indices[head]], new_value[head//2, None]))
            weights = torch.softmax(selected_k @ query[head] / 8**.5, dim=-1)
            expected.append(weights @ selected_v)
        torch.testing.assert_close(output, torch.stack(expected), atol=2e-6, rtol=2e-6)
        assert pool.slots.shape == (2, 1, 2, 16, 8)
        assert pool.capacity_slots == 2
        assert all(len(table) <= 1 for table in pool.resident)
        assert pool.loads > 0 and pool.evictions > 0 and pool.hits > 0
        assert pool.transfer_bytes == pool.loads * 2 * 16 * 8 * 4


def test_append_updates_resident_tail_without_management_load(tmp_path):
    with MetadataPublisher(tmp_path / "ops.jsonl", "tail") as publisher:
        pool = LayerPool(layer=0, kv_heads=1, head_dim=4, context_tokens=17,
                         maximum_tokens=20, block_tokens=16, residency_fraction=.2,
                         dtype=torch.float32, device="cpu", hooks=BackendHooks(publisher, "infinigen"))
        pool.append(torch.zeros(1, 17, 4), torch.zeros(1, 17, 4))
        pool.get(0, 1, 0)
        loads = pool.loads
        pool.append(torch.ones(1, 1, 4), 2*torch.ones(1, 1, 4))
        kv, count = pool.get(0, 1, 1)
        assert count == 2 and pool.loads == loads
        torch.testing.assert_close(kv[0, 1], torch.ones(4))
        torch.testing.assert_close(kv[1, 1], 2*torch.ones(4))


def test_rotary_preserves_dot_products_for_same_position():
    query, key = torch.randn(4, 5, 8), torch.randn(4, 5, 8)
    positions = torch.arange(5)
    qrot, krot = rotary(query, positions, 10000), rotary(key, positions, 10000)
    torch.testing.assert_close((qrot*krot).sum(-1), (query*key).sum(-1))


def test_generation_context_limit_counts_only_tokens_consumed_by_decoder():
    from types import SimpleNamespace
    from backends.infinigen.runtime import Runtime
    runtime = Runtime.__new__(Runtime)
    runtime.tokenizer = lambda prompt: {"input_ids": list(range(64))}
    runtime.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=65))
    runtime.default_generation, runtime.block, runtime.fraction = 2, 16, .2
    runtime.kv_heads, runtime.layers, runtime.defense, runtime.selected_budget = 2, [None], "none", 64
    assert runtime.describe_request("context")["generation_tokens"] == 2
    with pytest.raises(ValueError, match="position limit"):
        runtime.describe_request("context", 3)


def test_pinned_upstream_prediction_port():
    source = os.environ.get("KVTRAIL_INFINIGEN_SOURCE")
    if not source:
        pytest.skip("Set KVTRAIL_INFINIGEN_SOURCE to the pinned source checkout")
    primitives = load_primitives(source)
    torch.manual_seed(4)
    q, k = torch.randn(4, 20, 8), torch.randn(2, 20, 8)
    predictor = Predictor(primitives, q, k, 24, .5, 5, 6)
    query = torch.randn(4, 1, 8)
    selected = predictor.select(query)
    reduced = predictor._reduce(query) / 8**.5
    scores = torch.bmm(reduced, predictor.partial[:20].permute(1, 2, 0))
    threshold = scores.max(-1, keepdim=True).values-5
    count = min(int((scores > threshold).sum(-1).float().mean()), 6)
    expected = scores[:, 0].topk(count, dim=-1).indices
    torch.testing.assert_close(selected, expected)
    predictor.append(torch.randn(2, 1, 8))
    selected = predictor.select(query, earliest=12)
    assert selected.min() >= 12 and selected.max() < 21


@pytest.mark.parametrize("defense", ["none", "fixed_budget", "dynamic_timing"])
def test_mistral_gqa_prefill_and_full_selection_decode_match_dense(defense):
    source = os.environ.get("KVTRAIL_INFINIGEN_SOURCE")
    if not source:
        pytest.skip("Set KVTRAIL_INFINIGEN_SOURCE to the pinned source checkout")
    transformers = pytest.importorskip("transformers")
    from backends.infinigen.runtime import Runtime
    torch.manual_seed(8)
    config = transformers.MistralConfig(vocab_size=64, hidden_size=32,
              intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
              num_key_value_heads=2, max_position_embeddings=128, sliding_window=None)
    config._attn_implementation = "eager"
    model = transformers.MistralForCausalLM(config).eval()
    runtime = Runtime.__new__(Runtime)
    runtime.t, runtime.model = torch, model
    runtime.device = torch.device("cpu")
    runtime.layers = model.model.layers
    runtime.heads, runtime.kv_heads, runtime.dim, runtime.groups = 4, 2, 8, 2
    runtime.theta, runtime.window = 10000., None
    runtime.ratio, runtime.alpha, runtime.selected_budget = .5, 1e6, 128
    runtime.primitives = load_primitives(source)
    pools = [LayerPool(layer=i, kv_heads=2, head_dim=8, context_tokens=20,
                      maximum_tokens=24, block_tokens=16, residency_fraction=.2,
                      dtype=torch.float32, device="cpu", hooks=None, defense=defense,
                      sleeper=lambda seconds: None) for i in range(2)]
    tokens = torch.randint(0, 64, (20,))
    with torch.inference_mode():
        logits, predictors = runtime._prefill(tokens, pools)
        expected = model(tokens[None], use_cache=False).logits[0, -1:]
        torch.testing.assert_close(logits, expected, atol=1e-6, rtol=1e-5)
        token = logits.argmax(-1)
        decoded = runtime._decode(token, 20, pools, predictors, 0)
        expected = model(torch.cat((tokens, token))[None], use_cache=False).logits[0, -1:]
        torch.testing.assert_close(decoded, expected, atol=1e-6, rtol=1e-5)


def test_fixed_budget_pads_one_native_batch_and_preserves_attention(tmp_path):
    import json
    torch.manual_seed(9)
    key,value = torch.randn(1,64,4),torch.randn(1,64,4)
    query = torch.randn(1,4)
    indices = torch.tensor([[0,17,34,51]])
    path = tmp_path/"defended.jsonl"
    with MetadataPublisher(path,"defended") as publisher:
        pool=LayerPool(layer=0,kv_heads=1,head_dim=4,context_tokens=64,maximum_tokens=68,
                       block_tokens=16,residency_fraction=.2,dtype=torch.float32,
                       device="cpu",hooks=BackendHooks(publisher,"infinigen"),defense="fixed_budget")
        pool.padding_host.fill_(123.)
        pool.append(key,value)
        output=pool.attend(query,indices,0,groups=1)
        weights=torch.softmax(key[0,indices[0]] @ query[0] / 2.,dim=-1)
        expected=weights @ value[0,indices[0]]
        torch.testing.assert_close(output[0],expected,atol=2e-6,rtol=2e-6)
        assert pool.loads == 4 and pool.padding_loads == 61
        assert pool.batch_transfer_counts == [65]
        assert pool.load_batches == 1 and pool.overflow_batches == 0
        assert torch.all(pool.scratch == 123.)
        assert pool.padding_transfer_bytes == 61*2*16*4*4
        assert len(pool.resident[0]) == 1
    events=[json.loads(line) for line in path.read_text().splitlines()]
    assert sum(row["op"] == "LOAD" for row in events) == 65
    assert all(set(row) == set(events[0]) for row in events)
    assert all("padding" not in str(row["block_id"]) for row in events)


def test_fixed_budget_hit_only_batch_has_no_padding_and_overflow_keeps_all_required():
    pool=LayerPool(layer=0,kv_heads=1,head_dim=4,context_tokens=1120,maximum_tokens=1120,
                   block_tokens=16,residency_fraction=.2,dtype=torch.float32,device="cpu",
                   hooks=None,defense="fixed_budget")
    pool.append(torch.ones(1,1120,4),2*torch.ones(1,1120,4))
    consumed=[]
    pool._batch(0,range(70),0,lambda block,kv,count:consumed.append(block))
    assert sorted(consumed) == list(range(70))
    assert pool.loads == 70 and pool.padding_loads == 0
    assert pool.overflow_batches == 1 and pool.batch_transfer_counts == [70]
    resident=list(pool.resident[0])
    pool._batch(0,resident,1)
    assert pool.loads == 70 and pool.padding_loads == 0 and pool.load_batches == 1


def test_dynamic_timing_delays_real_transfers_but_not_hits():
    delays=[]
    pool=LayerPool(layer=0,kv_heads=1,head_dim=4,context_tokens=64,maximum_tokens=64,
                   block_tokens=16,residency_fraction=.2,dtype=torch.float32,device="cpu",
                   hooks=None,defense="dynamic_timing",sleeper=delays.append)
    pool.append(torch.ones(1,64,4),2*torch.ones(1,64,4))
    pool._batch(0,[0,1],0)
    kv,count=pool.get(0,1,1)
    assert pool.loads == 2 and pool.hits == 1
    assert len(delays) == pool.delay_calls == 2
    assert all(0 <= delay <= .005 for delay in delays)
    assert count == 16 and torch.all(kv[0] == 1) and torch.all(kv[1] == 2)


@pytest.mark.parametrize("defense", ["none", "fixed_budget", "dynamic_timing"])
def test_decode_step_budget_spans_prefetch_and_all_gqa_demand_heads(defense):
    torch.manual_seed(91)
    key, value, query = torch.randn(2,64,8), torch.randn(2,64,8), torch.randn(4,8)
    indices = torch.tensor([[0,16,32,48], [1,17,33,49], [2,18,34,50], [3,19,35,51]])
    delays = []
    pool = LayerPool(layer=0, kv_heads=2, head_dim=8, context_tokens=64, maximum_tokens=66,
        block_tokens=16, residency_fraction=.2, dtype=torch.float32, device="cpu", hooks=None,
        defense=defense, sleeper=delays.append)
    pool.append(key, value)
    for step in (0, 1):
        pool.prefetch(indices, 2, step)
        actual = pool.attend(query, indices, step, 2)
        expected = torch.stack([torch.softmax(key[h//2, indices[h]] @ query[h]/8**.5, -1)
                                @ value[h//2, indices[h]] for h in range(4)])
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        assert pool.load_batches == (step+1)*2
        assert pool.capacity_slots == 2 and all(len(table) <= 1 for table in pool.resident)
    assert sum(pool.batch_transfer_counts) == pool.loads+pool.padding_loads
    if defense == "fixed_budget":
        assert pool.batch_transfer_counts == [65]*4
        assert pool.padding_loads == 260-pool.loads
    if defense == "dynamic_timing":
        assert len(delays) == pool.delay_calls == 4
        assert all(0 <= delay <= .005 for delay in delays)


@pytest.mark.parametrize("defense", ["fixed_budget", "dynamic_timing"])
def test_prefetched_hit_only_step_does_not_start_another_defense_operation(defense):
    delays = []
    pool = LayerPool(layer=0, kv_heads=1, head_dim=4, context_tokens=64, maximum_tokens=64,
        block_tokens=16, residency_fraction=.2, dtype=torch.float32, device="cpu", hooks=None,
        defense=defense, sleeper=delays.append)
    pool.append(torch.ones(1,64,4), torch.ones(1,64,4))
    indices = torch.tensor([[0,1], [2,3]])
    for step in (0,1):
        pool.prefetch(indices, 2, step)
        pool.attend(torch.ones(2,4), indices, step, 2)
        assert pool.load_batches == 1
    if defense == "fixed_budget":
        assert pool.batch_transfer_counts == [65] and pool.padding_loads == 64
    else:
        assert pool.batch_transfer_counts == [1] and len(delays) == 1


def test_whole_decode_step_overflow_keeps_all_required_pages_without_padding():
    pool = LayerPool(layer=0, kv_heads=1, head_dim=4, context_tokens=1120, maximum_tokens=1120,
        block_tokens=16, residency_fraction=.2, dtype=torch.float32, device="cpu", hooks=None,
        defense="fixed_budget")
    pool.append(torch.ones(1,1120,4), 2*torch.ones(1,1120,4))
    indices = torch.arange(0,1120,16)[None]
    pool.prefetch(indices, 1, 0)
    actual = pool.attend(torch.ones(1,4), indices, 0, 1)
    torch.testing.assert_close(actual, 2*torch.ones(1,4))
    assert pool.loads == 70 and pool.padding_loads == 0
    assert pool.batch_transfer_counts == [70] and pool.overflow_batches == 1

