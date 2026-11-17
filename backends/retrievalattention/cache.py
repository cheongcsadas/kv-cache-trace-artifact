"""CPU-backed layerwise prefill and native RetrievalAttention decode cache."""
import torch
from transformers.cache_utils import Cache
from experiments.acquisition.metadata import BackendHooks
from .budget import budget


class RetrievalCache(Cache):
    def __init__(self, native, config, prompt_tokens, generation_tokens, layers, publisher):
        super().__init__()
        self.native, self.config = native, config
        self.block = int(config["cache"]["block_tokens"])
        self.layout = budget(prompt_tokens, self.block, config["cache"]["residency_fraction"])
        self.slots = self.layout["capacity_slots_per_layer"]
        self.prompt_tokens, self.layers, self.seen = prompt_tokens, layers, 0
        self.hooks = BackendHooks(publisher, "retrievalattention")
        self.vector = native.VectorDB_KV_Cache(prompt_tokens + generation_tokens, generation_tokens)
        self.vector.cache_size = self.layout["persistent_tokens"]
        self.vector.start_size = self.layout["sink_slots"] * self.block
        self.vector.recent_size = self.vector.cache_size - self.vector.start_size
        self.core = int(config.get("backend", {}).get("cpu_threads", 1))
        if self.core < 1:
            raise ValueError("backend.cpu_threads must be positive")
        self.top_k = config["cache"].get("selected_token_budget")
        if not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("Set a positive cache.selected_token_budget for CPU retrieval")
        self.pools, self.cpu_context = {}, {}
        self.defense = config.get("experiment", {}).get("defense")
        if self.defense not in (None, "none", "fixed_budget", "dynamic_timing"):
            raise ValueError("Unknown native defense condition")
        self.fixed = self.timing = None
        if self.defense == "fixed_budget":
            from defenses.fixed_budget import FixedBudget
            self.fixed = FixedBudget(config["defenses"]["fixed_budget"]["load_pages"], self.block)
        elif self.defense == "dynamic_timing":
            from defenses.dynamic_timing import DynamicTiming
            self.timing = DynamicTiming(config["defenses"]["dynamic_timing"]["max_delay_ms"])
        self.padding_cpu, self.padding_scratch = {}, {}
        self.transfer_counts = dict(required=0, padding=0, required_bytes=0, padding_bytes=0)
        self.load_batches = self.delay_calls = 0
        self.batch_transfer_counts = []

    def get_seq_length(self, layer_idx=0):
        return self.seen

    def get_max_cache_shape(self):
        return None

    def update(self, *args, **kwargs):
        raise RuntimeError("RetrievalCache is driven by its installed Qwen attention path")

    @staticmethod
    def _wait(tensor):
        if tensor.is_cuda:
            torch.cuda.current_stream(tensor.device).synchronize()

    def _allocate_padding(self, layer, key, value):
        if self.fixed is None:
            return
        cpu_shape = (*key.shape[:2], self.fixed.budget*self.block, key.shape[-1])
        scratch_shape = (*key.shape[:2], self.block, key.shape[-1])
        self.padding_cpu[layer] = (
            torch.zeros(cpu_shape, dtype=key.dtype, device="cpu", pin_memory=key.is_cuda),
            torch.zeros(cpu_shape, dtype=value.dtype, device="cpu", pin_memory=value.is_cuda))
        self.padding_scratch[layer] = (
            torch.empty(scratch_shape, dtype=key.dtype, device=key.device),
            torch.empty(scratch_shape, dtype=value.dtype, device=value.device))

    def _transfer_page(self, layer, block_id, offsets):
        """Complete a real CPU-to-device KV copy before publishing its LOAD."""
        if block_id in offsets:
            key_cpu, value_cpu = self.cpu_context[layer]
            key_pool, value_pool = self.pools[layer]
            offset, count = offsets[block_id]
            start = block_id*self.block
            key_target, value_target = key_pool[:, :, offset:offset+count], value_pool[:, :, offset:offset+count]
            key_source, value_source = key_cpu[:, :, start:start+count], value_cpu[:, :, start:start+count]
            category = "required"
        else:
            key_cpu, value_cpu = self.padding_cpu[layer]
            start = (block_id-self.layout["context_blocks"])*self.block
            key_source, value_source = key_cpu[:, :, start:start+self.block], value_cpu[:, :, start:start+self.block]
            key_target, value_target = self.padding_scratch[layer]
            category = "padding"
        key_target.copy_(key_source, non_blocking=True)
        value_target.copy_(value_source, non_blocking=True)
        self._wait(key_target)
        self.transfer_counts[category] += 1
        self.transfer_counts[category+"_bytes"] += (
            key_source.numel()*key_source.element_size() + value_source.numel()*value_source.element_size())
        return block_id

    def _stage(self, layer, blocks):
        """Fill bounded slots; optional padding copies use separate physical scratch."""
        key_cpu, value_cpu = self.cpu_context[layer]
        key_pool, value_pool = self.pools[layer]
        offsets, used = {}, 0
        for block_id in blocks:
            count = min(self.block, self.prompt_tokens-block_id*self.block)
            offsets[block_id] = (used, count)
            used += count
        def transfer(block_id):
            return self._transfer_page(layer, block_id, offsets)
        def publish(block_id):
            self.hooks.completed("LOAD", layer, [block_id], phase="prefill")
            if block_id not in offsets:
                # The scratch mapping is released before its next real overwrite.
                # The public event schema carries no padding discriminator.
                self.hooks.completed("EVICT", layer, [block_id], phase="prefill")
        def load_batch():
            if self.fixed is not None:
                first = self.layout["context_blocks"]
                plan = self.fixed.plan(blocks, range(first, first+self.fixed.budget))
                self.fixed.execute(plan, transfer, publish)
                self.batch_transfer_counts.append(len(plan.transfer_order))
            else:
                for block_id in blocks:
                    transfer(block_id)
                    publish(block_id)
                self.batch_transfer_counts.append(len(blocks))
        if blocks:
            self.load_batches += 1
            if self.timing is not None:
                self.delay_calls += 1
                self.timing.execute(load_batch)
            else:
                load_batch()
        return key_pool[:, :, :used], value_pool[:, :, :used]

    def allocation_receipt(self):
        nbytes = lambda tensors: sum(t.numel()*t.element_size() for pair in tensors.values() for t in pair)
        return dict(defense=self.defense, transfer_counts=dict(self.transfer_counts),
                    defense_parameters=self.config.get("defenses", {}).get(self.defense, {}),
                    defense_load_batch_unit="per_layer_prefill_persistent_context",
                    defense_load_batches=self.load_batches, defense_delay_calls=self.delay_calls,
                    defense_overflow_batches=sum(n > self.fixed.budget for n in self.batch_transfer_counts) if self.fixed is not None else 0,
                    defense_batch_transfer_counts=list(self.batch_transfer_counts),
                    padding_scratch_slots_per_layer=int(self.fixed is not None),
                    padding_scratch_bytes=nbytes(self.padding_scratch),
                    padding_cpu_bytes=nbytes(self.padding_cpu),
                    context_pool_bytes=nbytes(self.pools))

    def prefill(self, query, key, value, layer):
        if query.shape[0] != 1 or query.shape[-2] != self.prompt_tokens or key.shape[-2] != self.prompt_tokens:
            raise ValueError("RetrievalAttention uses one complete unpadded request per cache")
        if layer == 0:
            self.seen = self.prompt_tokens
        shape = (*key.shape[:2], self.slots*self.block, key.shape[-1])
        self.pools[layer] = (torch.empty(shape, dtype=key.dtype, device=key.device),
                             torch.empty(shape, dtype=value.dtype, device=value.device))
        self._allocate_padding(layer, key, value)
        key_cpu = torch.empty(key.shape, dtype=key.dtype, device="cpu", pin_memory=key.is_cuda)
        value_cpu = torch.empty(value.shape, dtype=value.dtype, device="cpu", pin_memory=value.is_cuda)
        key_cpu.copy_(key, non_blocking=key.is_cuda)
        value_cpu.copy_(value, non_blocking=value.is_cuda)
        self._wait(key)
        self.cpu_context[layer] = (key_cpu, value_cpu)
        groups = query.shape[1] // key.shape[1]
        # Prefill projections are already on the execution device. Compute causal
        # attention from those inputs before retaining the bounded KV cache.
        # Reloading every context page from CPU solely to recompute prefill would
        # introduce artificial management traffic absent from this loading path.
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1), is_causal=True)
        self._wait(output)
        persistent = self.layout["persistent_blocks"]
        gpu_k, gpu_v = self._stage(layer, persistent)
        self.vector.gpu_key_cache.append(gpu_k)
        self.vector.gpu_value_cache.append(gpu_v)
        # New decode tokens are allocations, separately accounted from fixed-context
        # residency. They are never mislabeled as CPU-to-GPU KV LOAD events.
        temporary_shape = (*key.shape[:2], self.vector.temp_cache, key.shape[-1])
        self.vector.temp_gpu_key_cache.append(torch.empty(temporary_shape, dtype=key.dtype, device=key.device))
        self.vector.temp_gpu_value_cache.append(torch.empty(temporary_shape, dtype=value.dtype, device=value.device))
        persistent_tokens = {i for b in persistent for i in range(b*self.block, min((b+1)*self.block, self.prompt_tokens))}
        cpu_tokens = torch.tensor([i for i in range(self.prompt_tokens) if i not in persistent_tokens], dtype=torch.long)
        cpu_k = key_cpu.index_select(2, cpu_tokens).float().numpy()
        cpu_v = value_cpu.index_select(2, cpu_tokens)
        self.vector.key_vec_cache_update(cpu_k, layer, self.core)
        self.vector.val_vec_cache_update(cpu_v, layer, self.layers-1)
        # Only the native index and CPU values are retained after construction.
        del self.cpu_context[layer]
        return output.to(query.dtype)

    def decode(self, query, key, value, layer):
        if query.shape[-2] != 1:
            raise ValueError("Autoregressive decode requires one token per step")
        if layer == 0:
            self.seen += 1
        key_index, gpu_key = self.vector.sync_key_update(key, layer, self.layers-1, self.core, False)
        cpu_value, gpu_value = self.vector.sync_value_update(value, layer, self.layers-1, False)
        effective_k = min(self.top_k, key_index.ntotal)
        output = self.native.retr_attn(query, (key_index, gpu_key), (cpu_value, gpu_value),
            dict(layer_idx=layer, cpu_threads=self.core, num_key_value_groups=query.shape[1]//key.shape[1],
                 attn_forward_config=dict(top_k=effective_k, from_layer=0)))
        self._wait(output)
        self.hooks.completed("HIT", layer, self.layout["persistent_blocks"],
                             phase="decode", step=self.seen-self.prompt_tokens)
        return output
