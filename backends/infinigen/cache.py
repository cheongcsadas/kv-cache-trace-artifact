"""Bounded physical KV slots with pinned host backing and completed-operation exports."""
from collections import OrderedDict
import math


class LayerPool:
    """One request/layer: CPU backing, fixed GPU slots, and a separate partial-key cache.

    A physical slot contains K and V for ``block_tokens`` positions of one KV
    head. Native prediction selects tokens per query head; GQA maps query heads
    onto their shared KV head before the selected tokens access these slots.
    """
    def __init__(self, *, layer, kv_heads, head_dim, context_tokens, maximum_tokens,
                 block_tokens, residency_fraction, dtype, device, hooks, torch_module=None,
                 defense="none", defense_config=None, sleeper=None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        t = self.torch
        if block_tokens not in (8, 16, 32) or not 0 < residency_fraction <= 1:
            raise ValueError("Invalid physical block size or residency fraction")
        self.layer, self.kv_heads, self.head_dim = layer, kv_heads, head_dim
        self.block_tokens, self.maximum_tokens = block_tokens, maximum_tokens
        self.slots_per_head = max(1, math.ceil(math.ceil(context_tokens / block_tokens) * residency_fraction))
        self.device = t.device(device)
        self.hooks = hooks
        self.length = 0
        self.cuda = self.device.type == "cuda"
        self.host = t.empty((2, kv_heads, maximum_tokens, head_dim), dtype=dtype,
                            device="cpu", pin_memory=self.cuda)
        self.slots = t.empty((kv_heads, self.slots_per_head, 2, block_tokens, head_dim),
                             dtype=dtype, device=self.device)
        self.resident = [OrderedDict() for _ in range(kv_heads)]
        self.valid = [dict() for _ in range(kv_heads)]
        self.loads = self.hits = self.evictions = 0
        self.transfer_bytes = 0
        if defense not in {"none", "fixed_budget", "dynamic_timing"}:
            raise ValueError("Unknown InfiniGen loading defense")
        self.defense, self.sleeper = defense, sleeper
        self.padding_loads = self.padding_evictions = self.padding_transfer_bytes = 0
        self.overflow_batches = self.load_batches = self.delay_calls = 0
        self.batch_transfer_counts = []
        self.padding_host = self.scratch = None
        self.fixed = self.timing = None
        self._step = None
        self._operations = {}
        self._planned_batches = {}
        settings = defense_config or {}
        if defense == "fixed_budget":
            from defenses.fixed_budget import FixedBudget
            fixed = settings.get("fixed_budget", {})
            if block_tokens != 16 or fixed.get("block_tokens",16) != block_tokens:
                raise ValueError("Fixed-budget defense uses physical 16-token blocks")
            self.fixed = FixedBudget(budget=fixed.get("load_pages",65), block_tokens=block_tokens)
            # Reserved CPU-backed logical pages, in the same ID namespace and
            # tensor layout as ordinary pages. An extra candidate avoids reusing
            # the scratch-resident page as a purported LOAD in the next batch.
            self.padding_start = math.ceil(maximum_tokens/block_tokens)
            self.padding_host = t.zeros((kv_heads,self.fixed.budget+1,2,block_tokens,head_dim),
                                         dtype=dtype,device="cpu",pin_memory=self.cuda)
            self.scratch = t.empty((kv_heads,2,block_tokens,head_dim),dtype=dtype,device=self.device)
            self.padding_resident = [None]*kv_heads
        elif defense == "dynamic_timing":
            from defenses.dynamic_timing import DynamicTiming
            self.timing = DynamicTiming(settings.get("dynamic_timing",{}).get("max_delay_ms",5.0))

    @property
    def capacity_slots(self):
        return self.kv_heads * self.slots_per_head

    def _complete(self):
        if self.cuda:
            self.torch.cuda.current_stream(self.device).synchronize()

    def _event(self, op, head, block, step):
        if self.hooks is not None:
            self.hooks.completed(op, self.layer, [f"{head}:{block}"], phase="decode", step=step)

    def append(self, key, value):
        """Persist newly computed K/V; allocation is not a management LOAD."""
        count = key.shape[-2]
        if key.shape != value.shape or key.shape[0] != self.kv_heads:
            raise ValueError("Expected (kv_heads, tokens, head_dim) K/V")
        end = self.length + count
        if end > self.maximum_tokens:
            raise ValueError("KV capacity exceeded")
        self.host[0, :, self.length:end].copy_(key, non_blocking=self.cuda)
        self.host[1, :, self.length:end].copy_(value, non_blocking=self.cuda)
        self._complete()
        # Update an already resident partial last block using fresh GPU K/V.
        for head in range(self.kv_heads):
            for pos in range(self.length, end):
                block, offset = divmod(pos, self.block_tokens)
                slot = self.resident[head].get(block)
                if slot is not None:
                    self.slots[head, slot, 0, offset].copy_(key[head, pos-self.length])
                    self.slots[head, slot, 1, offset].copy_(value[head, pos-self.length])
                    self.valid[head][block] = offset + 1
        self.length = end

    def _get(self, head, block, step, *, reuse=True):
        """Fetch an existing CPU-backed block, or reuse its actual GPU slot."""
        start = block * self.block_tokens
        if head not in range(self.kv_heads) or start >= self.length or block < 0:
            raise ValueError("Invalid KV block access")
        table = self.resident[head]
        if block in table:
            slot = table.pop(block)
            table[block] = slot
            if reuse:
                self._complete()
                self._event("HIT", head, block, step)
                self.hits += 1
            return self.slots[head, slot], self.valid[head][block]
        if len(table) == self.slots_per_head:
            old, slot = table.popitem(last=False)
            self._complete()
            self.valid[head].pop(old)
            self._event("EVICT", head, old, step)
            self.evictions += 1
        else:
            slot = next(i for i in range(self.slots_per_head) if i not in table.values())
        count = min(self.block_tokens, self.length-start)
        def copy():
            self.slots[head, slot, :, :count].copy_(self.host[:, head, start:start+count], non_blocking=self.cuda)
        operation = self._operations.get(head) if self._step == step else None
        if self.timing is not None and (operation is None or not operation["delayed"]):
            self.delay_calls += 1
            if operation is not None:
                operation["delayed"] = True
            options = {} if self.sleeper is None else {"sleeper":self.sleeper}
            self.timing.execute(copy,**options)
        else:
            copy()
        self._complete()
        table[block] = slot
        self.valid[head][block] = count
        self._event("LOAD", head, block, step)
        self.loads += 1
        self.transfer_bytes += 2 * count * self.head_dim * self.host.element_size()
        self._record_transfer(head, step)
        return self.slots[head, slot], count

    def _record_transfer(self, head, step):
        operation = self._operations.get(head) if self._step == step else None
        if operation is None:
            return
        if operation["index"] is None:
            operation["index"] = len(self.batch_transfer_counts)
            self.batch_transfer_counts.append(0)
            self.load_batches += 1
            if self.fixed is not None and operation["required"] > self.fixed.budget:
                self.overflow_batches += 1
        self.batch_transfer_counts[operation["index"]] += 1

    def _padding_transfer(self, head, block, step):
        previous = self.padding_resident[head]
        if previous == block:
            raise ValueError("Padding LOAD must target a nonresident CPU-backed page")
        if previous is not None:
            self._complete()
            self._event("EVICT",head,previous,step)
            self.padding_evictions += 1
        self.scratch[head].copy_(self.padding_host[head,block-self.padding_start],non_blocking=self.cuda)
        self._complete()
        self.padding_resident[head] = block
        self._event("LOAD",head,block,step)
        self.padding_loads += 1
        self.padding_transfer_bytes += self.scratch[head].numel()*self.scratch.element_size()
        self._record_transfer(head, step)

    def _plan_step(self, indices, groups, step, *, prefetch):
        """Plan one loading operation per KV head across rehearsal and demand.

        Simulating the exact LRU order accounts for repeated physical loads when
        selected tokens exceed the bank. Padding is distributed across that whole
        operation; query-head sub-batches never receive a fresh 65-page budget.
        Required transfers can be consumed immediately, keeping the GPU bound.
        """
        self._step, self._operations, self._planned_batches = step, {}, {}
        for head in range(self.kv_heads):
            self._operations[head] = dict(index=None, delayed=False, required=0)
            if self.fixed is None:
                continue
            selected = indices[head*groups:(head+1)*groups].tolist()
            sequences = []
            if prefetch:
                ranked = list(dict.fromkeys(int(token)//self.block_tokens for row in selected for token in row))
                sequences.append(ranked[:self.slots_per_head])
            sequences.extend(sorted({int(token)//self.block_tokens for token in row}) for row in selected)
            resident = OrderedDict(self.resident[head])
            batches = []
            for blocks in sequences:
                hits = [block for block in blocks if block in resident]
                missing = [block for block in blocks if block not in resident]
                self.fixed.rng.shuffle(missing)
                for block in hits + missing:
                    if block in resident:
                        resident.move_to_end(block)
                    else:
                        if len(resident) == self.slots_per_head:
                            resident.popitem(last=False)
                        resident[block] = None
                batches.append(dict(blocks=blocks, hits=hits, missing=missing, order=[]))
            required = sum(len(batch["missing"]) for batch in batches)
            self._operations[head]["required"] = required
            if required:
                candidates = [self.padding_start+i for i in range(self.fixed.budget+1)
                              if self.padding_start+i != self.padding_resident[head]]
                padding = self.fixed.rng.sample(candidates, max(0, self.fixed.budget-required))
                # Interleave real pages and padding over the complete step while
                # respecting the model's prefetch-before-query dependency.
                slots = [None]*required + padding
                self.fixed.rng.shuffle(slots)
                planned = [(batch, block) for batch in batches for block in batch["missing"]]
                position = 0
                for item in slots:
                    if item is None:
                        batch, block = planned[position]
                        batch["order"].append(block)
                        position += 1
                    else:
                        batch = planned[min(position, required-1)][0]
                        batch["order"].append(item)
            self._planned_batches[head] = batches

    def _batch(self, head, blocks, step, consume=None, *, reuse=True):
        """One native per-head prefetch or per-query-head demand batch.

        Required blocks are consumed immediately after their real transfer, so
        shuffling does not require retaining more KV than the physical bank.
        Scratch padding never becomes an attention operand. A hit-only batch
        performs zero padding copies, and >K required loads are never dropped.
        """
        blocks = list(blocks)
        if head not in range(self.kv_heads) or any(
                isinstance(block,bool) or not isinstance(block,int) or block < 0
                or block*self.block_tokens >= self.length for block in blocks):
            raise ValueError("Native load batch references an invalid CPU-backed block")
        if len(set(blocks)) != len(blocks):
            raise ValueError("A native load batch must name distinct blocks")
        def access(block):
            resident = block in self.resident[head]
            kv, count = self._get(head, block, step, reuse=False)
            if consume is not None:
                consume(block, kv, count)
            if resident and reuse:
                # Attribute HIT to the completed use, not to the earlier lookup.
                self._complete()
                self._event("HIT", head, block, step)
                self.hits += 1
        if self.fixed is not None and self._step == step and head in self._planned_batches:
            plans = self._planned_batches[head]
            if not plans:
                raise ValueError("Native step has already consumed its planned loading batches")
            plan = plans.pop(0)
            if plan["blocks"] != blocks:
                raise ValueError("Native selected blocks differ from the planned decode step")
            for block in plan["hits"]:
                if block not in self.resident[head]:
                    raise RuntimeError("Prefetch/demand residency diverged from its loading plan")
                access(block)
            required = set(plan["missing"])
            for block in plan["order"]:
                if block in required:
                    if block in self.resident[head]:
                        raise RuntimeError("Planned CPU-backed page is already resident")
                    access(block)
                else:
                    self._padding_transfer(head, block, step)
            return
        if self.fixed is None:
            before = self.loads
            for block in blocks:
                access(block)
            transferred = self.loads-before
            if transferred and not (self._step == step and head in self._operations):
                self.load_batches += 1
                self.batch_transfer_counts.append(transferred)
            return
        required = [block for block in blocks if block not in self.resident[head]]
        # Consume hits before shuffled misses can evict them. This changes only
        # evaluation order; online softmax retains the exact selected token set.
        for block in blocks:
            if block not in required:
                access(block)
        if not required:
            return
        candidates = [self.padding_start+i for i in range(self.fixed.budget+1)
                      if self.padding_start+i != self.padding_resident[head]]
        plan = self.fixed.plan(required,candidates)
        self.load_batches += 1
        self.overflow_batches += int(not plan.within_budget)
        self.batch_transfer_counts.append(len(plan.transfer_order))
        required_set = set(required)
        def transfer(block):
            if block in required_set:
                access(block)
            else:
                self._padding_transfer(head,block,step)
        # Both transfer paths publish only after completed tensor copies. The
        # generic executor's publish callback is empty to avoid duplicate LOADs.
        self.fixed.execute(plan,transfer,lambda block: None)

    def get(self, head, block, step, *, reuse=True):
        """Single-block demand access, also used by diagnostics and unit checks."""
        result = []
        self._batch(head,[block],step,lambda block,kv,count:result.append((kv,count)),reuse=reuse)
        return result[0]

    def prefetch(self, indices, groups, step):
        """Prefetch native selected blocks without exceeding physical slot count."""
        self._plan_step(indices, groups, step, prefetch=True)
        # Stable first occurrence preserves native token ranking. Later blocks
        # are loaded on demand by the bounded attention loop.
        for head in range(self.kv_heads):
            blocks = []
            for token in indices[head*groups:(head+1)*groups].reshape(-1).tolist():
                block = int(token) // self.block_tokens
                if block not in blocks:
                    blocks.append(block)
            self._batch(head,blocks[:self.slots_per_head],step,reuse=False)

    def attend(self, query, indices, step, groups, current_key=None, current_value=None):
        """Exact softmax over the predictor's selected token set in bounded space.

        Online log-sum-exp combines blocks without materializing a full selected
        KV tensor on GPU. Consequently the resident KV pool remains fixed even
        when the selected tokens span more blocks than the configured budget.
        """
        t = self.torch
        if self._step != step:
            self._plan_step(indices, groups, step, prefetch=False)
        output = t.empty_like(query)
        for qhead in range(query.shape[0]):
            selected = sorted(set(int(x) for x in indices[qhead].tolist()))
            if not selected and current_key is None:
                raise ValueError("Attention cannot have an empty token set")
            by_block = {}
            for token in selected:
                by_block.setdefault(token // self.block_tokens, []).append(token % self.block_tokens)
            maximum = t.tensor(float("-inf"), device=self.device)
            denominator = t.zeros((), dtype=t.float32, device=self.device)
            numerator = t.zeros(self.head_dim, dtype=t.float32, device=self.device)
            if current_key is not None:
                maximum = (current_key[qhead // groups].float() @ query[qhead].float()) / math.sqrt(self.head_dim)
                denominator = t.ones((), dtype=t.float32, device=self.device)
                numerator = current_value[qhead // groups].float().clone()
            def consume(block,kv,valid):
                nonlocal maximum,denominator,numerator
                offsets = by_block[block]
                if max(offsets) >= valid:
                    raise ValueError("Prediction referenced unwritten KV positions")
                offsets = t.tensor(offsets, dtype=t.long, device=self.device)
                keys = kv[0].index_select(0, offsets).float()
                values = kv[1].index_select(0, offsets).float()
                logits = (keys @ query[qhead].float()) / math.sqrt(self.head_dim)
                next_max = t.maximum(maximum, logits.max())
                factor = t.exp(maximum-next_max)
                weights = t.exp(logits-next_max)
                numerator = numerator * factor + weights @ values
                denominator = denominator * factor + weights.sum()
                maximum = next_max
            self._batch(qhead // groups,by_block,step,consume)
            output[qhead] = (numerator / denominator).to(query.dtype)
        if self.fixed is not None:
            if any(self._planned_batches.values()):
                raise RuntimeError("Decode left planned KV loading sub-batches unconsumed")
            for operation in self._operations.values():
                if operation["required"]:
                    expected = max(self.fixed.budget, operation["required"])
                    if self.batch_transfer_counts[operation["index"]] != expected:
                        raise RuntimeError("Actual decode transfers differ from its fixed loading budget")
        return output

