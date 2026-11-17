"""Qwen2.5 model loading, bounded KV prefill, and native CPU retrieval decoding."""
from pathlib import Path
import time
from types import MethodType
from .budget import budget
from .upstream import REVISION, load_native, source_text


def qwen_attention(self, hidden_states, attention_mask=None, position_ids=None,
                   past_key_value=None, output_attentions=False, use_cache=False,
                   cache_position=None, position_embeddings=None, **kwargs):
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    from .cache import RetrievalCache
    if not isinstance(past_key_value, RetrievalCache) or not use_cache:
        raise ValueError("The installed Qwen attention requires its request-local RetrievalCache")
    batch, length, _ = hidden_states.shape
    q = self.q_proj(hidden_states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    cos, sin = self.rotary_emb(v, position_ids) if position_embeddings is None else position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if self.layer_idx not in past_key_value.pools:
        attended = past_key_value.prefill(q, k, v, self.layer_idx)
    else:
        attended = past_key_value.decode(q, k, v, self.layer_idx)
    output = attended.transpose(1, 2).contiguous().reshape(batch, length, self.hidden_size)
    return self.o_proj(output), None, past_key_value


class Runtime:
    concurrent_requests = True

    def __init__(self, config, source_dir, model_path=None, device="cuda:0"):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if transformers.__version__ != "4.47.1":
            raise RuntimeError("This Qwen attention adapter requires transformers==4.47.1")
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("Measured RetrievalAttention inference requires a CUDA GPU")
        if config.get("system") != "retrievalattention":
            raise ValueError("Wrong system configuration")
        if config.get("backend", {}).get("revision") not in (None, REVISION):
            raise ValueError("Configured revision disagrees with the pinned native source")
        defense = config.get("experiment", {}).get("defense")
        if defense not in (None, "none", "fixed_budget", "dynamic_timing"):
            raise ValueError("Unknown native defense condition")
        if defense == "fixed_budget" and config["cache"]["block_tokens"] != 16:
            raise ValueError("The fixed-budget defense requires 16-token logical pages")
        threads = config.get("backend", {}).get("cpu_threads", 1)
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("backend.cpu_threads must be a positive integer")
        source_text(source_dir)
        self.native = load_native(source_dir)
        self.config, self.device = config, torch.device(device)
        self.model_id = str(model_path or config["model"]["id"])
        dtype = config["inference"].get("dtype")
        if dtype not in {"float16", "bfloat16"}:
            raise ValueError("inference.dtype must explicitly select float16 or bfloat16")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id,
            revision=config["model"].get("revision"), trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id,
            revision=config["model"].get("revision"), torch_dtype=getattr(torch, dtype),
            attn_implementation="eager", trust_remote_code=False).to(self.device).eval()
        if self.model.config.model_type != "qwen2":
            raise ValueError("RetrievalAttention runtime expects Qwen2.5 weights")
        if getattr(self.model.config, "use_sliding_window", False):
            raise ValueError("This fixed-context experiment uses full-context Qwen attention")
        self.layers = len(self.model.model.layers)
        for layer in self.model.model.layers:
            layer.self_attn.forward = MethodType(qwen_attention, layer.self_attn)

    def _length(self, override):
        value = override if override is not None else self.config["inference"].get("generation_tokens")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Supply a positive generation length")
        return value

    def describe_request(self, prompt, max_new_tokens=None):
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        length = self._length(max_new_tokens)
        if len(ids) + length - 1 > self.model.config.max_position_embeddings:
            raise ValueError("Request exceeds the model position limit")
        layout = budget(len(ids), self.config["cache"]["block_tokens"],
                        self.config["cache"]["residency_fraction"])
        capacity = layout["capacity_slots_per_layer"]
        padding_slots = int(self.config.get("experiment", {}).get("defense") == "fixed_budget")
        specified = self.config["cache"].get("capacity_slots")
        if specified is not None and specified != capacity:
            raise ValueError("Configured capacity_slots disagrees with actual resident allocation")
        return dict(system="retrievalattention", model=self.model_id, backend_revision=REVISION,
            backend_repository="https://github.com/microsoft/MInference", index_backend="faiss.IndexFlatIP",
            cpu_threads=self.config.get("backend", {}).get("cpu_threads", 1),
            token_count=len(ids), generation_tokens=length, capacity_slots=capacity,
            capacity_unit="logical_pages_per_layer", physical_capacity_slots=(capacity+padding_slots)*self.layers,
            physical_context_capacity_slots=capacity*self.layers,
            padding_scratch_slots_per_layer=padding_slots,
            block_tokens=self.config["cache"]["block_tokens"],
            residency_fraction=self.config["cache"]["residency_fraction"],
            resident_context_tokens_per_layer=layout["persistent_tokens"],
            selected_token_budget=self.config["cache"]["selected_token_budget"],
            new_token_allocation_tokens_per_layer=length, layer_count=self.layers,
            context_tokens=len(ids), empty_pool=True)

    def generate(self, prompt, request_id, export_dir, max_new_tokens=None):
        import torch
        from backends.common import NativeExport
        from .cache import RetrievalCache
        receipt = self.describe_request(prompt, max_new_tokens)
        tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        sampling = self.config["inference"].get("sampling")
        if not isinstance(sampling, dict) or sampling.get("do_sample") is not False:
            raise ValueError("Set inference.sampling: {do_sample: false} for this deterministic runner")
        stream = torch.cuda.Stream(device=self.device)
        started = time.perf_counter()
        generated = []
        with NativeExport(export_dir, request_id) as export:
            cache = RetrievalCache(self.native, self.config, tokens.shape[-1],
                                   receipt["generation_tokens"], self.layers, export.publisher)
            with torch.inference_mode(), torch.cuda.stream(stream):
                current = tokens.to(self.device)
                for _ in range(receipt["generation_tokens"]):
                    result = self.model(input_ids=current, past_key_values=cache,
                                        use_cache=True, num_logits_to_keep=1)
                    current = result.logits[:, -1:].argmax(dim=-1)
                    generated.append(int(current.item()))
                    # Fixed output length keeps the experimental work budget fixed.
                stream.synchronize()
                receipt.update(cache.allocation_receipt())
                for layer in range(self.layers):
                    export.publisher.snapshot(layer, [], kind="cleanup")
                del cache
            receipt.update(export.receipt())
        receipt.update(request_id=str(request_id), provenance="measured",
                       generated_tokens=len(generated), elapsed_s=time.perf_counter()-started,
                       output_text=self.tokenizer.decode(generated, skip_special_tokens=True))
        return receipt
