"""Real Mistral generation with native InfiniGen prediction and CPU/GPU KV movement."""
from pathlib import Path
import math
import time

from .upstream import REVISION, load_primitives
from .cache import LayerPool
from .predictor import Predictor, rotary


class Runtime:
    concurrent_requests = True

    def __init__(self, config, source_dir, model_path=None, device="cuda:0"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if config.get("system") != "infinigen":
            raise ValueError("InfiniGen runtime requires the infinigen system configuration")
        self.defense = config.get("experiment", {}).get("defense", "none")
        if self.defense not in {"none","fixed_budget","dynamic_timing"}:
            raise ValueError("Unknown InfiniGen loading defense")
        self.config, self.t = config, torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Measured InfiniGen generation requires CUDA")
        self.primitives = load_primitives(source_dir)
        cache = config["cache"]
        self.block = int(cache["block_tokens"])
        if self.defense == "fixed_budget" and self.block != 16:
            raise ValueError("Fixed-budget defense requires 16-token physical blocks")
        self.fraction = float(cache["residency_fraction"])
        selected = cache.get("selected_token_budget")
        if isinstance(selected, bool) or not isinstance(selected, int) or selected < 1:
            raise ValueError("Set cache.selected_token_budget to the measured selection budget")
        self.selected_budget = selected
        backend = config.get("backend", {})
        if backend.get("revision") not in (None, REVISION):
            raise ValueError(f"InfiniGen primitives require revision {REVISION}")
        self.ratio = float(backend.get("partial_weight_ratio", 0.2))
        self.alpha = float(backend.get("alpha", 5.0))
        identifier = model_path or config["model"]["id"]
        kwargs = {"revision": config["model"].get("revision"), "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(identifier, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(identifier, torch_dtype=torch.float16,
                         low_cpu_mem_usage=True, attn_implementation="eager", **kwargs).to(self.device).eval()
        mc = self.model.config
        if mc.model_type != "mistral":
            raise ValueError("This port requires an actual Mistral model checkpoint")
        if getattr(mc, "rope_scaling", None):
            raise ValueError("Use the paper's standard-RoPE Mistral checkpoint")
        self.layers = self.model.model.layers
        self.heads, self.kv_heads = mc.num_attention_heads, mc.num_key_value_heads
        self.dim = mc.hidden_size // self.heads
        self.groups = self.heads // self.kv_heads
        self.theta = float(mc.rope_theta)
        self.window = getattr(mc, "sliding_window", None)
        self.default_generation = config.get("inference", {}).get("generation_tokens")
        inference = config.get("inference", {})
        if inference.get("dtype") not in (None, "float16", "fp16"):
            raise ValueError("This runtime uses FP16 model weights and KV tensors")
        sampling = inference.get("sampling")
        if isinstance(sampling, dict):
            valid_sampling = sampling == {"do_sample": False}
        else:
            valid_sampling = sampling in (None, False, "greedy")
        if not valid_sampling:
            raise ValueError("Configure greedy decoding for this measured runtime")

    def describe_request(self, prompt, max_new_tokens=None):
        count = self.default_generation if max_new_tokens is None else max_new_tokens
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Set a positive inference.generation_tokens or max_new_tokens")
        n = len(self.tokenizer(prompt)["input_ids"])
        if n+count-1 > self.model.config.max_position_embeddings:
            raise ValueError("Request exceeds the checkpoint position limit")
        slots = max(1, math.ceil(math.ceil(n/self.block)*self.fraction))
        return dict(token_count=n, generation_tokens=count, block_tokens=self.block,
                    residency_fraction=self.fraction, capacity_slots=slots,
                    capacity_unit="per_layer_per_kv_head_block",
                    capacity_slots_per_layer=slots*self.kv_heads,
                    physical_capacity_slots=slots*self.kv_heads*len(self.layers),
                    defense=self.defense,
                    selected_token_budget=self.selected_budget)

    def _qkv(self, layer, hidden, positions):
        q = layer.self_attn.q_proj(hidden).reshape(-1, self.heads, self.dim).transpose(0, 1)
        k = layer.self_attn.k_proj(hidden).reshape(-1, self.kv_heads, self.dim).transpose(0, 1)
        v = layer.self_attn.v_proj(hidden).reshape(-1, self.kv_heads, self.dim).transpose(0, 1)
        return rotary(q, positions, self.theta), rotary(k, positions, self.theta), v

    def _prefill(self, tokens, pools):
        t = self.t
        hidden = self.model.model.embed_tokens(tokens)
        positions = t.arange(tokens.shape[0], device=self.device)
        predictors = []
        for i, layer in enumerate(self.layers):
            norm = layer.input_layernorm(hidden)
            query, key, value = self._qkv(layer, norm, positions)
            expanded_key = key.repeat_interleave(self.groups, dim=0)
            expanded_value = value.repeat_interleave(self.groups, dim=0)
            mask = None
            if self.window and tokens.shape[0] > self.window:
                distance = positions[:, None]-positions[None, :]
                mask = (distance >= 0) & (distance < self.window)
            attended = t.nn.functional.scaled_dot_product_attention(query, expanded_key,
                        expanded_value, attn_mask=mask, is_causal=mask is None)
            hidden = hidden + layer.self_attn.o_proj(attended.transpose(0, 1).reshape(-1, self.heads*self.dim))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            pools[i].append(key, value)
            predictors.append(Predictor(self.primitives, query, key, pools[i].maximum_tokens,
                                         self.ratio, self.alpha, self.selected_budget))
        return self.model.lm_head(self.model.model.norm(hidden[-1:])), predictors

    def _decode(self, token, position, pools, predictors, step):
        t = self.t
        hidden = self.model.model.embed_tokens(token)
        positions = t.tensor([position], dtype=t.long, device=self.device)
        selected = None
        for i, layer in enumerate(self.layers):
            norm = layer.input_layernorm(hidden)
            # Rehearse the subsequent layer from this layer's normalized input.
            # It uses only its partial key cache and Q projection; actual K/V
            # transfers occur into next layer's bounded slots before attention.
            next_selected = None
            earliest = max(0, position-self.window+1) if self.window else 0
            if i+1 < len(self.layers):
                next_attention = self.layers[i+1].self_attn
                rehearsal = next_attention.q_proj(norm).reshape(1, self.heads, self.dim).transpose(0, 1)
                rehearsal = rotary(rehearsal, positions, self.theta)
                next_selected = predictors[i+1].select(rehearsal, earliest=earliest)
                pools[i+1].prefetch(next_selected, self.groups, step)
            query, key, value = self._qkv(layer, norm, positions)
            if selected is None:
                # The first layer has no predecessor for rehearsal.
                selected = predictors[i].select(query, earliest=earliest)
                pools[i].prefetch(selected, self.groups, step)
            attended = pools[i].attend(query[:, 0], selected, step, self.groups,
                                      current_key=key[:, 0], current_value=value[:, 0])
            hidden = hidden + layer.self_attn.o_proj(attended.reshape(1, -1))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            pools[i].append(key, value)
            predictors[i].append(key)
            selected = next_selected
        return self.model.lm_head(self.model.model.norm(hidden))

    def generate(self, prompt, request_id, export_dir, max_new_tokens=None):
        from experiments.acquisition.metadata import BackendHooks
        from backends.common import NativeExport
        t = self.t
        count = self.default_generation if max_new_tokens is None else max_new_tokens
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Set a positive inference.generation_tokens or max_new_tokens")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty prompt is required")
        self.describe_request(prompt, count)
        path = Path(export_dir)
        path.mkdir(parents=True, exist_ok=True)
        tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0].to(self.device)
        n = int(tokens.shape[0])
        maximum = n + count
        stream = t.cuda.Stream(device=self.device)
        stream.wait_stream(t.cuda.current_stream(self.device))
        started = time.perf_counter()
        with NativeExport(path, request_id) as exports:
            publisher = exports.publisher
            hooks = BackendHooks(publisher, "infinigen")
            with t.inference_mode(), t.cuda.device(self.device), t.cuda.stream(stream):
                pools = [LayerPool(layer=i, kv_heads=self.kv_heads, head_dim=self.dim,
                                   context_tokens=n, maximum_tokens=maximum, block_tokens=self.block,
                                   residency_fraction=self.fraction, dtype=t.float16,
                                   device=self.device, hooks=hooks, defense=self.defense,
                                   defense_config=self.config["defenses"]) for i in range(len(self.layers))]
                capacity = pools[0].slots_per_head
                configured_capacity = self.config["cache"].get("capacity_slots")
                if configured_capacity is not None and configured_capacity != capacity:
                    raise ValueError(f"Configured capacity_slots {configured_capacity} differs from allocated {capacity}")
                for pool in pools:
                    publisher.snapshot(pool.layer, [], kind="initialization")
                logits, predictors = self._prefill(tokens, pools)
                generated = []
                for step in range(count):
                    token = logits.argmax(dim=-1)
                    generated.append(int(token.item()))
                    # Fixed generation length keeps profiling conditions aligned.
                    if step+1 < count:
                        logits = self._decode(token, n+step, pools, predictors, step)
                stream.synchronize()
                receipt = dict(system="infinigen", implementation="mistral-gqa-predictive-offload-port-v1",
                    backend_revision=REVISION, token_count=n, generation_tokens=len(generated),
                    capacity_slots=capacity, capacity_slots_per_layer=pools[0].capacity_slots,
                    physical_capacity_slots=sum(p.capacity_slots for p in pools),
                    capacity_unit="per_layer_per_kv_head_block",
                    capacity_slots_per_head=pools[0].slots_per_head, block_tokens=self.block,
                    residency_fraction=self.fraction, selected_token_budget=self.selected_budget,
                    partial_weight_ratio=self.ratio, alpha=self.alpha,
                    kv_pool_bytes=sum(p.slots.numel()*p.slots.element_size() for p in pools),
                    host_kv_bytes=sum(p.host.numel()*p.host.element_size() for p in pools),
                    partial_key_bytes=sum(p.partial.numel()*p.partial.element_size() for p in predictors),
                    defense=self.defense,
                    defense_load_batch_unit="per_kv_head_layer_decode_step",
                    defense_parameters=self.config["defenses"].get(self.defense,{}),
                    defense_padding_gpu_bytes=sum(0 if p.scratch is None else p.scratch.numel()*p.scratch.element_size() for p in pools),
                    defense_padding_cpu_bytes=sum(0 if p.padding_host is None else p.padding_host.numel()*p.padding_host.element_size() for p in pools),
                    defense_padding_loads=sum(p.padding_loads for p in pools),
                    defense_overflow_batches=sum(p.overflow_batches for p in pools),
                    defense_load_batches=sum(p.load_batches for p in pools),
                    defense_batch_transfer_counts=[count for p in pools for count in p.batch_transfer_counts],
                    defense_delay_calls=sum(p.delay_calls for p in pools),
                    load_count=sum(p.loads+p.padding_loads for p in pools),
                    required_load_count=sum(p.loads for p in pools), hit_count=sum(p.hits for p in pools),
                    eviction_count=sum(p.evictions+p.padding_evictions for p in pools),
                    h2d_kv_bytes=sum(p.transfer_bytes+p.padding_transfer_bytes for p in pools),
                    elapsed_ms=(time.perf_counter()-started)*1000,
                    text=self.tokenizer.decode(generated, skip_special_tokens=True))
            receipt.update(exports.receipt())
        return receipt

