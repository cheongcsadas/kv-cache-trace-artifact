"""Physical page residency and exact streamed attention for Quest's selected pages.

The upstream CUDA estimator and top-k selector still choose pages independently
for each attention head. Streaming prevents the union of those choices from
silently increasing the configured GPU cache budget.
"""
from collections import OrderedDict
import math


def capacity_for_prompt(tokens, block_tokens, residency_fraction):
    if tokens < 1 or block_tokens not in (8, 16, 32):
        raise ValueError("Positive prompt length and a paper block size are required")
    if not 0 < residency_fraction <= 1:
        raise ValueError("residency_fraction must be in (0, 1]")
    capacity = math.ceil(math.ceil(tokens / block_tokens) * residency_fraction)
    if capacity < 2:
        raise ValueError("The GPU pool needs a historical slot and a mutable-tail slot")
    return capacity


class LogicalPool:
    """Keep upstream logical allocation while releasing its full GPU KV tensor."""
    def __init__(self, source, allocated):
        self.capacity = source.capacity
        self.num_layers = source.num_layers
        self.block_len = source.block_len
        self._free = set(range(allocated, self.capacity))

    def alloc_block(self):
        if not self._free:
            raise RuntimeError("Quest request exceeded its allocated context")
        index = min(self._free)
        self._free.remove(index)
        return index

    def free_block(self, index):
        if index in self._free:
            raise RuntimeError("Double release of a logical KV page")
        self._free.add(index)

    @property
    def buf(self):
        raise RuntimeError("Full GPU KV storage was released; use the resident pool")


def merge_attention_chunk(q, keys, values, allowed, state=None):
    """FP32 online softmax, equivalent to attention over the concatenated chunks.

    q: [heads, dim]; keys/values: [tokens, heads, dim]; allowed: [heads,tokens].
    The state contains only an output accumulator and two scalars per head.
    """
    import torch
    scores = torch.einsum("hd,thd->ht", q.float(), keys.float()) / math.sqrt(q.shape[-1])
    scores.masked_fill_(~allowed, -torch.inf)
    chunk_max = scores.amax(dim=-1)
    if state is None:
        previous_max = torch.full_like(chunk_max, -torch.inf)
        denominator = torch.zeros_like(chunk_max)
        numerator = torch.zeros_like(q, dtype=torch.float32)
    else:
        previous_max, denominator, numerator = state
    maximum = torch.maximum(previous_max, chunk_max)
    # A head may have no selected page in a particular chunk.
    safe_maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    old_scale = torch.exp(previous_max - safe_maximum)
    weights = torch.exp(scores - safe_maximum[:, None])
    denominator = denominator * old_scale + weights.sum(-1)
    numerator = numerator * old_scale[:, None] + torch.einsum("ht,thd->hd", weights, values.float())
    return maximum, denominator, numerator


class ResidentKV:
    def __init__(self, controller, capacity, hooks, max_seq_len, *, timing=None, fixed_budget=None):
        import torch
        self.torch = torch
        self.controller = controller
        self.capacity, self.tail_slot = capacity, capacity - 1
        self.hooks, self.timing = hooks, timing
        self.fixed_budget = fixed_budget
        self.loads = self.padding_loads = self.delay_calls = 0
        self.load_batches = self.overflow_batches = 0
        self.batch_transfer_counts = []
        self.block_tokens = controller.page_size
        self.device = controller.device
        self.step = -1
        source = controller.kv_cache.pool
        source_ids = list(controller.kv_cache.indicies)
        full = source.buf
        layers, _, _, block, heads, dim = full.shape
        max_pages = math.ceil(max_seq_len / block)
        self.cpu = torch.empty((layers, max_pages, 2, block, heads, dim),
                               dtype=full.dtype, device="cpu", pin_memory=True)
        self.gpu = torch.empty((layers, capacity, 2, block, heads, dim),
                               dtype=full.dtype, device=self.device)
        self.padding_cpu = self.padding_gpu = None
        self.padding_resident = [None] * layers
        if fixed_budget is not None:
            self.padding_start = max_pages
            self.padding_cpu = torch.zeros((fixed_budget.budget, 2, block, heads, dim),
                                            dtype=full.dtype, device="cpu", pin_memory=True)
            self.padding_gpu = torch.empty((layers, 2, block, heads, dim),
                                           dtype=full.dtype, device=self.device)
        # CPU backing is established by real D2H copies before any management event.
        for logical, physical in enumerate(source_ids):
            self.cpu[:, logical].copy_(full[:, physical], non_blocking=True)
        self.tail_page = len(source_ids) - 1
        self.gpu[:, self.tail_slot].copy_(full[:, source_ids[-1]], non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()
        self.maps = [OrderedDict() for _ in range(layers)]
        controller.kv_cache._pool = LogicalPool(source, len(source_ids))
        controller.kv_cache._indicies = list(range(len(source_ids)))
        self.last_sequence_length = controller.kv_cache.seqlen
        self.cpu_backed_pages = len(source_ids)

    def begin_decode(self):
        length = self.controller.kv_cache.seqlen
        if length == self.last_sequence_length:
            return
        if length != self.last_sequence_length + 1:
            raise RuntimeError("Quest supports one new token per decode step")
        next_tail = (length - 1) // self.block_tokens
        if next_tail != self.tail_page:
            self.cpu[:, self.tail_page].copy_(self.gpu[:, self.tail_slot], non_blocking=True)
            self.torch.cuda.current_stream(self.device).synchronize()
            self.cpu_backed_pages = self.tail_page + 1
            self.tail_page = next_tail
            self.gpu[:, self.tail_slot].zero_()
        self.last_sequence_length = length
        self.step += 1

    def _event(self, operation, layer, page):
        self.hooks.completed(operation, layer, [page], phase="decode", step=self.step)

    def load_chunk(self, layer, pages):
        """Copy a physical chunk; defense budgets belong to the whole decode batch."""
        if len(pages) > self.tail_slot or len(set(pages)) != len(pages) or any(p >= self.tail_page or p < 0 for p in pages):
            raise ValueError("Historical page chunk exceeds the physical pool")
        mapping = self.maps[layer]
        protected = set(pages)
        reused = [page for page in pages if page in mapping]
        missing = [page for page in pages if page not in mapping]
        def transfer_required(page):
            free = sorted(set(range(self.tail_slot)) - set(mapping.values()))
            if free:
                slot = free[0]
            else:
                victim = next(p for p in mapping if p not in protected)
                # Prior attention reads finish before a slot is released/reused.
                self.torch.cuda.current_stream(self.device).synchronize()
                slot = mapping.pop(victim)
                self._event("EVICT", layer, victim)
            self.gpu[layer, slot].copy_(self.cpu[layer, page], non_blocking=True)
            self.torch.cuda.current_stream(self.device).synchronize()
            mapping[page] = slot
            self.loads += 1
            return slot
        for page in missing:
            transfer_required(page)
            self._event("LOAD", layer, page)
        for page in pages:
            mapping.move_to_end(page)
        return [mapping[page] for page in pages], reused

    def attention(self, query, layer, selected):
        """Consume the real upstream Quest top-k output, retaining per-head selection."""
        torch = self.torch
        rows = selected.detach().to("cpu").tolist()
        if len(rows) != query.shape[-2]:
            raise ValueError("Quest top-k head count does not match the query")
        sets = [set(int(page) for page in row) for row in rows]
        pages = sorted(set().union(*sets))
        if any(page < 0 or page >= self.tail_page for page in pages):
            raise ValueError("Quest selector returned a non-historical page")
        q = query.squeeze(0)
        state = None
        before = self.loads + self.padding_loads
        delayed = False

        def consume(chunk):
            nonlocal state, delayed
            missing = any(page not in self.maps[layer] for page in chunk)
            if missing and self.timing is not None and not delayed:
                # One independent delay before the first physical copy in this
                # layer/decode-step loading operation, regardless of chunk count.
                delayed = True
                self.delay_calls += 1
                slots, reused = self.timing.execute(self.load_chunk, layer, chunk)
            else:
                slots, reused = self.load_chunk(layer, chunk)
            # The temporary attention workspace is at most one resident-sized chunk.
            kv = self.gpu[layer, slots]
            keys = kv[:, 0].flatten(0, 1)
            values = kv[:, 1].flatten(0, 1)
            allowed = torch.tensor([[p in selected_set for p in chunk] for selected_set in sets],
                                   dtype=torch.bool, device=q.device).repeat_interleave(self.block_tokens, dim=1)
            state = merge_attention_chunk(q, keys, values, allowed, state)
            torch.cuda.current_stream(self.device).synchronize()
            for page in reused:
                self._event("HIT", layer, page)

        if self.fixed_budget is not None:
            resident = [page for page in pages if page in self.maps[layer]]
            missing = [page for page in pages if page not in self.maps[layer]]
            # Consume existing operands before shuffled misses can evict them.
            # This keeps the selected set unchanged without growing the pool.
            if resident:
                consume(resident)
            if missing:
                candidates = [self.padding_start + index for index in range(self.fixed_budget.budget)
                              if self.padding_start + index != self.padding_resident[layer]]
                plan = self.fixed_budget.plan(missing, candidates)
                self.overflow_batches += int(not plan.within_budget)
                required = set(missing)

                def transfer(page):
                    if page in required:
                        # Consume immediately: the next required page may reuse
                        # this same physical slot when selection exceeds capacity.
                        consume([page])
                    else:
                        previous = self.padding_resident[layer]
                        if previous is not None:
                            torch.cuda.current_stream(self.device).synchronize()
                            self._event("EVICT", layer, previous)
                        self.padding_gpu[layer].copy_(self.padding_cpu[page-self.padding_start], non_blocking=True)
                        torch.cuda.current_stream(self.device).synchronize()
                        self.padding_resident[layer] = page
                        self.padding_loads += 1
                        self._event("LOAD", layer, page)

                # Each path publishes after its real copy, with identical schema.
                self.fixed_budget.execute(plan, transfer, lambda page: None)
        else:
            for start in range(0, len(pages), self.tail_slot):
                consume(pages[start:start + self.tail_slot])
        transferred = self.loads + self.padding_loads - before
        if transferred:
            self.load_batches += 1
            self.batch_transfer_counts.append(transferred)
        # Every head attends to the live tail, which is new-token allocation rather
        # than CPU-backed management traffic and therefore has no synthetic HIT.
        tail_length = self.controller.kv_cache.last_page_len
        tail = self.gpu[layer, self.tail_slot, :, :tail_length]
        allowed = torch.ones((q.shape[0], tail_length), dtype=torch.bool, device=q.device)
        _, denominator, numerator = merge_attention_chunk(q, tail[0], tail[1], allowed, state)
        return (numerator / denominator[:, None]).to(query.dtype).unsqueeze(0)


class Overlay:
    def __init__(self, original, **values):
        self.original, self.values = original, values

    def __getattr__(self, key):
        if key in self.values:
            return self.values[key]
        return getattr(self.original, key)


def on_native_default_stream(function):
    """Order upstream kernels that explicitly launch on CUDA stream 0.

    Only these native kernel boundaries use the default stream. Weight matrix
    operations and streamed attention remain on each request's own CUDA stream.
    Event waits avoid device-wide synchronization and preserve tensor lifetimes.
    """
    from functools import wraps
    @wraps(function)
    def call(*args, **kwargs):
        import torch
        current = torch.cuda.current_stream()
        default = torch.cuda.default_stream()
        if current == default:
            return function(*args, **kwargs)
        def record(value, stream):
            if isinstance(value, torch.Tensor) and value.is_cuda:
                value.record_stream(stream)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    record(item, stream)
            elif isinstance(value, dict):
                for item in value.values():
                    record(item, stream)
        default.wait_stream(current)
        record(args, default)
        record(kwargs, default)
        with torch.cuda.stream(default):
            result = function(*args, **kwargs)
        current.wait_stream(default)
        record(result, current)
        return result
    return call


def install_native_bindings():
    """Bind physical residency directly at upstream append and attention call sites."""
    import quest.utils
    import quest.models.llama as llama
    from quest.utils.controller import InferenceController
    if getattr(quest.utils, "_kvtrail_bindings", False):
        return
    original_append = on_native_default_stream(quest.utils.append_kv)
    original_attention = on_native_default_stream(quest.utils.decode_sparse_attn)
    for name in ("apply_rope_in_place", "prefill_forward", "decode_estimate", "decode_topk", "rms_norm_forward"):
        setattr(quest.utils, name, on_native_default_stream(getattr(quest.utils, name)))
    llama.rms_norm_forward = quest.utils.rms_norm_forward
    InferenceController.begin_forward = on_native_default_stream(InferenceController.begin_forward)
    InferenceController.end_forward = on_native_default_stream(InferenceController.end_forward)

    def append(k, v, iController, layer_idx):
        controller = iController
        pool = getattr(controller, "_kvtrail_pool", None)
        if pool is None:
            return original_append(k, v, controller, layer_idx)
        pool.begin_decode()
        indices = controller.kv_indices_with_last.clone()
        indices[-1] = pool.tail_slot
        cache = Overlay(controller.kv_cache, buf_layer=lambda layer: pool.gpu[layer])
        bound = Overlay(controller, kv_cache=cache, kv_indices_with_last=indices,
                        kv_last_page_idx=pool.tail_slot)
        return original_append(k, v, bound, layer_idx)

    def attention(q, iController, layer_idx, topk_indices, rope_scale=None, rope_theta=None):
        controller = iController
        pool = getattr(controller, "_kvtrail_pool", None)
        if pool is None:
            return original_attention(q, controller, layer_idx, topk_indices, rope_scale, rope_theta)
        return pool.attention(q, layer_idx, topk_indices)

    quest.utils.append_kv = append
    quest.utils.decode_sparse_attn = attention
    quest.utils._kvtrail_bindings = True
