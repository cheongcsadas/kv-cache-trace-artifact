import math

import pytest

from backends.quest.residency import LogicalPool, capacity_for_prompt, merge_attention_chunk
from backends.quest.runtime import SOURCE_HASHES, verify_source, RequestLocalController


def test_paper_residency_changes_physical_capacity():
    assert capacity_for_prompt(1000, 16, .2) == 13
    assert capacity_for_prompt(4000, 16, .2) == 50
    assert capacity_for_prompt(1000, 8, .2) == 25
    assert capacity_for_prompt(1000, 32, .2) == 7
    assert capacity_for_prompt(1000, 16, .6) == 38
    with pytest.raises(ValueError):
        capacity_for_prompt(1, 16, .2)


def test_streamed_attention_preserves_per_head_quest_selection():
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(182)
    q = torch.randn(4, 8, generator=generator)
    keys = torch.randn(37, 4, 8, generator=generator)
    values = torch.randn(37, 4, 8, generator=generator)
    allowed = torch.rand(4, 37, generator=generator) > .6
    # Explicitly exercise chunks in which a head has no eligible token.
    allowed[0, :12] = False
    allowed[:, -1] = True
    scores = torch.einsum("hd,thd->ht", q, keys) / math.sqrt(8)
    scores.masked_fill_(~allowed, -torch.inf)
    expected = torch.einsum("ht,thd->hd", scores.softmax(-1), values)
    state = None
    for left, right in ((0, 5), (5, 12), (12, 25), (25, 37)):
        state = merge_attention_chunk(q, keys[left:right], values[left:right], allowed[:, left:right], state)
    _, denominator, numerator = state
    torch.testing.assert_close(numerator / denominator[:, None], expected, rtol=1e-5, atol=1e-6)


def test_logical_pool_does_not_keep_full_gpu_tensor():
    class Original:
        capacity, num_layers, block_len = 8, 2, 16
    pool = LogicalPool(Original(), 3)
    assert pool.alloc_block() == 3
    pool.free_block(2)
    assert pool.alloc_block() == 2
    with pytest.raises(RuntimeError, match="released"):
        _ = pool.buf


def test_modified_native_source_is_rejected(tmp_path):
    path = tmp_path / next(iter(SOURCE_HASHES))
    path.parent.mkdir(parents=True)
    path.write_text("modified upstream source", encoding="utf-8")
    with pytest.raises(ValueError, match="source mismatch"):
        verify_source(tmp_path)


def test_controller_context_isolated_between_concurrent_requests():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    model = RequestLocalController()
    model.iController = "main"
    barrier = threading.Barrier(4)
    def run(value):
        assert model.iController is None
        model.iController = value
        barrier.wait(timeout=5)
        return model.iController
    with ThreadPoolExecutor(max_workers=4) as workers:
        assert list(workers.map(run, range(4))) == list(range(4))
    assert model.iController == "main"


def cpu_pool(torch, full, publisher, defense):
    from collections import OrderedDict
    from types import SimpleNamespace
    from backends.quest.residency import ResidentKV
    from experiments.acquisition.metadata import BackendHooks
    class CpuOperations:
        # Exercise real tensor copies and cache arithmetic on small CPU tensors.
        cuda = SimpleNamespace(current_stream=lambda device: SimpleNamespace(synchronize=lambda: None))
        def __getattr__(self, key):
            return getattr(torch, key)
    pool = ResidentKV.__new__(ResidentKV)
    pool.torch = CpuOperations()
    pool.device, pool.capacity, pool.tail_slot, pool.tail_page = "cpu", 3, 2, full.shape[1]-1
    pool.block_tokens, pool.step = 16, 0
    pool.maps = [OrderedDict()]
    pool.cpu = full
    pool.gpu = torch.empty(1, 3, 2, 16, 2, 4)
    pool.gpu[:, 2].copy_(full[:, pool.tail_page])
    pool.timing = pool.fixed_budget = None
    pool.loads = pool.padding_loads = pool.delay_calls = 0
    pool.load_batches = pool.overflow_batches = 0
    pool.batch_transfer_counts = []
    delays = []
    if defense == "fixed_budget":
        from defenses.fixed_budget import FixedBudget
        pool.fixed_budget = FixedBudget(65, 16, seed=21)
        pool.padding_start = full.shape[1]
        pool.padding_cpu = torch.zeros(65, 2, 16, 2, 4)
        pool.padding_gpu = torch.empty(1, 2, 16, 2, 4)
        pool.padding_resident = [None]
    if defense == "dynamic_timing":
        from defenses.dynamic_timing import DynamicTiming
        pool.timing = DynamicTiming(5, seed=21)
        execute = pool.timing.execute
        pool.timing.execute = lambda load, *args: execute(load, *args, sleeper=delays.append)
    pool.controller = SimpleNamespace(kv_cache=SimpleNamespace(last_page_len=3))
    pool.hooks = BackendHooks(publisher, "quest")
    return pool, delays


def dense_selected_attention(torch, full, selected, query):
    expected = []
    for head in range(2):
        historical = full[0, selected[head].long()]
        keys = torch.cat([historical[:, 0, :, head].flatten(0, 1), full[0, -1, 0, :3, head]])
        values = torch.cat([historical[:, 1, :, head].flatten(0, 1), full[0, -1, 1, :3, head]])
        scores = keys @ query[0, head] / 2
        expected.append(scores.softmax(0) @ values)
    return torch.stack(expected).unsqueeze(0)


@pytest.mark.parametrize("defense", ["none", "fixed_budget", "dynamic_timing"])
def test_physical_slot_reuse_preserves_sparse_attention(tmp_path, defense):
    torch = pytest.importorskip("torch")
    from experiments.acquisition.metadata import MetadataPublisher
    generator = torch.Generator().manual_seed(711)
    full = torch.randn(1, 6, 2, 16, 2, 4, generator=generator)
    with MetadataPublisher(tmp_path / "events.jsonl", "request") as publisher:
        pool, delays = cpu_pool(torch, full, publisher, defense)
        selected = torch.tensor([[0, 1, 3], [1, 2, 4]], dtype=torch.int32)
        query = torch.randn(1, 2, 4, generator=generator)
        expected = dense_selected_attention(torch, full, selected, query)
        actual = pool.attention(query, 0, selected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert len(pool.maps[0]) == 2
        assert pool.gpu.shape[1] == 3
        assert pool.loads == 5 and pool.load_batches == 1
        assert pool.batch_transfer_counts == [65 if defense == "fixed_budget" else 5]
        assert len(delays) == pool.delay_calls == int(defense == "dynamic_timing")
        assert all(0 <= delay <= .005 for delay in delays)
        # A second layer/step selecting only resident operands does no H2D work,
        # so it gets neither another padding budget nor another timing delay.
        resident = torch.tensor([list(pool.maps[0])]*2, dtype=torch.int32)
        actual = pool.attention(query, 0, resident)
        torch.testing.assert_close(actual, dense_selected_attention(torch, full, resident, query), rtol=1e-5, atol=1e-6)
        assert pool.load_batches == 1 and pool.loads == 5
        assert len(delays) == pool.delay_calls == int(defense == "dynamic_timing")
    import json
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert sum(row["op"] == "LOAD" for row in events) == (65 if defense == "fixed_budget" else 5)
    assert sum(row["op"] == "EVICT" for row in events) == (62 if defense == "fixed_budget" else 3)


def test_one_fixed_budget_for_selection_larger_than_both_pool_and_budget(tmp_path):
    torch = pytest.importorskip("torch")
    from experiments.acquisition.metadata import MetadataPublisher
    generator = torch.Generator().manual_seed(111)
    full = torch.randn(1, 71, 2, 16, 2, 4, generator=generator)
    with MetadataPublisher(tmp_path/"overflow.jsonl", "overflow") as publisher:
        pool, _ = cpu_pool(torch, full, publisher, "fixed_budget")
        selected = torch.tensor([list(range(70))]*2, dtype=torch.int32)
        query = torch.randn(1, 2, 4, generator=generator)
        actual = pool.attention(query, 0, selected)
        torch.testing.assert_close(actual, dense_selected_attention(torch, full, selected, query), rtol=1e-5, atol=1e-6)
        assert pool.loads == 70 and pool.padding_loads == 0
        assert pool.batch_transfer_counts == [70]
        assert pool.overflow_batches == pool.load_batches == 1
        assert len(pool.maps[0]) == 2 and pool.gpu.shape[1] == 3


def test_fixed_budget_consumes_resident_operands_before_shuffled_misses(tmp_path):
    torch = pytest.importorskip("torch")
    from experiments.acquisition.metadata import MetadataPublisher
    generator = torch.Generator().manual_seed(333)
    full = torch.randn(1, 6, 2, 16, 2, 4, generator=generator)
    with MetadataPublisher(tmp_path/"resident.jsonl", "resident") as publisher:
        pool, _ = cpu_pool(torch, full, publisher, "fixed_budget")
        pool.maps[0][4] = 0
        pool.gpu[0, 0].copy_(full[0, 4])
        publisher.snapshot(0, [4], kind="initialization")
        selected = torch.tensor([[0, 1, 3], [1, 2, 4]], dtype=torch.int32)
        query = torch.randn(1, 2, 4, generator=generator)
        actual = pool.attention(query, 0, selected)
        torch.testing.assert_close(actual, dense_selected_attention(torch, full, selected, query), rtol=1e-5, atol=1e-6)
        assert pool.loads == 4 and pool.padding_loads == 61
        assert pool.batch_transfer_counts == [65]
