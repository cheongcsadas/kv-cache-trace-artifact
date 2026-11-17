"""Load the pinned upstream Quest Llama model and execute real cached inference."""
from __future__ import annotations

import hashlib
from contextvars import ContextVar
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from experiments.acquisition.metadata import BackendHooks
from .residency import ResidentKV, capacity_for_prompt, install_native_bindings


SOURCE_REVISION = "01c1623bf9395009520874e989e29f683203b357"
SOURCE_HASHES = {
    "quest/models/QuestAttention.py": "d8c10e62920c42bae9e5c708f291d2849ad27527df17b02f5a78f2cb14eb278a",
    "quest/models/llama.py": "f4ea4b299b1555ac9032bc99d0d2f811be911d386137f015505fcaf7e00eb838",
    "quest/utils/__init__.py": "69fd75256a3985ffe23921e89aa498af17b3fd2225318b7951afd7b0a15c55a6",
    "quest/utils/controller.py": "b2d43ec89086400380e20ae30866616e08f3b69402a58beb84f2ebfaff379543",
    "quest/utils/kv_cache.py": "26d6e0a61da0700f84fb99a0ec8194e6b1bb853b93ec49854add4bb8d8bf23e2",
}


def verify_source(source_dir):
    source = Path(source_dir).resolve()
    for name, expected in SOURCE_HASHES.items():
        path = source / name
        data = path.read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Quest source mismatch: {name}; use upstream {SOURCE_REVISION}")
    if (source / ".git").exists():
        result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True)
        if result.stdout.strip() != SOURCE_REVISION:
            raise ValueError("Quest checkout revision differs from the pinned integration")
    return source


class RequestLocalController:
    """Keep upstream request state isolated while sharing read-only model weights."""
    @property
    def iController(self):
        return self.__dict__["_kvtrail_controller_context"].get()

    @iController.setter
    def iController(self, value):
        if "_kvtrail_controller_context" not in self.__dict__:
            self.__dict__["_kvtrail_controller_context"] = ContextVar("quest_controller", default=None)
        self.__dict__["_kvtrail_controller_context"].set(value)


def install_request_local_model():
    import quest.models.llama as native
    if not issubclass(native.LlamaModel, RequestLocalController):
        class RequestLocalLlamaModel(RequestLocalController, native.LlamaModel):
            """Upstream Llama computation with a context-local KV controller."""
        native.LlamaModel = RequestLocalLlamaModel


class Runtime:
    system = "quest"
    supports_concurrent_requests = True
    concurrent_requests = True

    def __init__(self, config, source_dir, model_path=None, device="cuda:0"):
        import torch
        if not torch.cuda.is_available() or not str(device).startswith("cuda"):
            raise RuntimeError("Quest native inference requires CUDA")
        if config.get("system") != "quest":
            raise ValueError("Quest Runtime requires a Quest configuration")
        source = verify_source(source_dir)
        configured_revision = config.get("backend", {}).get("revision")
        if configured_revision and configured_revision != SOURCE_REVISION:
            raise ValueError("Configured Quest revision differs from the supported source")
        sys.path.insert(0, str(source))
        import quest
        if Path(quest.__file__).resolve().parent != source / "quest":
            raise RuntimeError("A different Quest installation was imported in this process")
        install_request_local_model()
        from quest.models.llama import LlamaForCausalLM
        from transformers import AutoTokenizer
        self.config, self.torch = config, torch
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.block_tokens = int(config["cache"]["block_tokens"])
        self.residency_fraction = float(config["cache"]["residency_fraction"])
        selected = config["cache"].get("selected_token_budget")
        if not isinstance(selected, int) or isinstance(selected, bool) or selected < self.block_tokens * 2:
            raise ValueError("Set selected_token_budget to at least two logical blocks")
        if selected % self.block_tokens:
            raise ValueError("Quest selected_token_budget must be divisible by block_tokens")
        self.selected_token_budget = selected
        self.defense = config.get("experiment", {}).get("defense")
        if self.defense not in (None, "none", "fixed_budget", "dynamic_timing"):
            raise ValueError("Unknown native defense condition")
        if self.defense == "fixed_budget" and self.block_tokens != 16:
            raise ValueError("The fixed-budget defense uses 16-token pages")
        dtype = config.get("inference", {}).get("dtype")
        if dtype != "float16":
            raise ValueError("The pinned Quest CUDA kernels require inference.dtype=float16")
        sampling = config.get("inference", {}).get("sampling")
        if sampling not in ("greedy", {"do_sample": False}):
            raise ValueError("Set inference.sampling=greedy for deterministic token generation")
        identifier = str(model_path or config["model"]["id"])
        options = dict(trust_remote_code=False)
        if model_path is None and config["model"].get("revision"):
            options["revision"] = config["model"]["revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(identifier, **options)
        self.model = LlamaForCausalLM.from_pretrained(identifier, torch_dtype=torch.float16,
                                                    low_cpu_mem_usage=True, **options).to(self.device).eval()
        if self.model.config.num_attention_heads != self.model.config.num_key_value_heads:
            raise ValueError("The pinned Quest Llama integration requires Llama-2 MHA")
        install_native_bindings()
        self.source_dir = source

    def describe_request(self, prompt, max_new_tokens=None):
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        count = max_new_tokens if max_new_tokens is not None else self.config["inference"].get("generation_tokens")
        if not isinstance(count, int) or isinstance(count, bool) or count < 2:
            raise ValueError("Set generation_tokens >=2 so inference includes decode KV management")
        max_seq_len = len(ids) + count - 1
        if len(ids) <= self.block_tokens:
            raise ValueError("Quest prefill must span more than one logical page")
        if max_seq_len > self.model.config.max_position_embeddings:
            raise ValueError("Prompt plus generation exceeds the model context limit")
        capacity = capacity_for_prompt(len(ids), self.block_tokens, self.residency_fraction)
        configured = self.config["cache"].get("capacity_slots")
        if configured is not None and configured != capacity:
            raise ValueError("capacity_slots disagrees with the actual prompt residency budget")
        return dict(token_count=len(ids), generation_tokens=count, capacity_slots=capacity,
                    historical_capacity_slots=capacity - 1, block_tokens=self.block_tokens,
                    capacity_unit="per_layer_block",
                    physical_capacity_slots=capacity * self.model.config.num_hidden_layers,
                    residency_fraction=self.residency_fraction, max_seq_len=max_seq_len,
                    selected_token_budget=self.selected_token_budget,
                    supports_concurrent_requests=self.supports_concurrent_requests)

    def generate(self, prompt, request_id, export_dir, max_new_tokens=None):
        # Read-only model parameters are shared; cache, selector buffers, export
        # streams, and CUDA streams belong to one request context.
        with self.torch.cuda.device(self.device):
            stream = self.torch.cuda.Stream(device=self.device)
            with self.torch.cuda.stream(stream):
                try:
                    return self._generate(prompt, request_id, export_dir, max_new_tokens)
                finally:
                    stream.synchronize()
                    self.model.model.iController = None

    def _generate(self, prompt, request_id, export_dir, max_new_tokens):
        from backends.common import NativeExport
        torch = self.torch
        description = self.describe_request(prompt, max_new_tokens)
        suffix = hashlib.sha256(str(request_id).encode()).hexdigest()[:24]
        destination = Path(export_dir).resolve() / suffix
        destination.mkdir(parents=True, exist_ok=True)
        receipt_path = destination / "receipt.json"
        if receipt_path.exists():
            raise FileExistsError("Request export paths already exist")
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        self.model.model.iController = None
        self.model.quest_init(page_size=self.block_tokens, max_seq_len=description["max_seq_len"],
                              token_budget=self.selected_token_budget, dtype=torch.float16, device=self.device)
        # Shared export capacity scales with the maximum possible page visits.
        layers = self.model.config.num_hidden_layers
        max_pages = math.ceil(description["max_seq_len"] / self.block_tokens)
        stream_bytes = max(1 << 20, layers * max_pages * description["generation_tokens"] * 3 * 512)
        start = time.perf_counter()
        generated = []
        with NativeExport(destination, request_id, capacity_bytes=stream_bytes) as exports:
            publisher = exports.publisher
            hooks = BackendHooks(publisher, "quest")
            for layer in range(layers):
                publisher.snapshot(layer, [], source="telemetry", kind="initialization")
            with torch.inference_mode():
                logits = self.model(input_ids=inputs, use_cache=False).logits[:, -1]
                controller = self.model.model.iController
                timing = fixed_budget = None
                if self.defense == "dynamic_timing":
                    from defenses.dynamic_timing import DynamicTiming
                    timing = DynamicTiming(self.config["defenses"]["dynamic_timing"]["max_delay_ms"])
                if self.defense == "fixed_budget":
                    from defenses.fixed_budget import FixedBudget
                    fixed_budget = FixedBudget(self.config["defenses"]["fixed_budget"]["load_pages"], self.block_tokens)
                pool = ResidentKV(controller, description["capacity_slots"], hooks,
                                  description["max_seq_len"], timing=timing, fixed_budget=fixed_budget)
                controller._kvtrail_pool = pool
                for step in range(description["generation_tokens"]):
                    token = logits.argmax(dim=-1, keepdim=True)
                    generated.append(int(token.item()))
                    if step + 1 < description["generation_tokens"]:
                        logits = self.model(input_ids=token, use_cache=False).logits[:, -1]
                torch.cuda.current_stream(self.device).synchronize()
            for layer in range(layers):
                publisher.snapshot(layer, source="telemetry")
            export_receipt = exports.receipt()
        end = time.perf_counter()
        receipt = dict(description, **export_receipt, request_id=str(request_id), system="quest", provenance="measured",
                       backend_revision=SOURCE_REVISION, integration="native-quest-selector-streamed-offload",
                       elapsed_ms=(end-start)*1000,
                       model_id=self.config["model"]["id"],
                       model_revision=self.config["model"].get("revision"), initial_state="empty",
                       defense=self.defense, padding_scratch_slots_per_layer=int(pool.padding_gpu is not None),
                       defense_load_batch_unit="per_layer_decode_step",
                       defense_parameters=self.config["defenses"].get(self.defense, {}),
                       defense_load_batches=pool.load_batches,
                       defense_required_loads=pool.loads, defense_padding_loads=pool.padding_loads,
                       defense_overflow_batches=pool.overflow_batches,
                       defense_delay_calls=pool.delay_calls,
                       defense_batch_transfer_counts=pool.batch_transfer_counts,
                       padding_gpu_bytes=0 if pool.padding_gpu is None else pool.padding_gpu.numel()*pool.padding_gpu.element_size(),
                       gpu_kv_pool_bytes=pool.gpu.numel()*pool.gpu.element_size(),
                       cpu_kv_backing_bytes=pool.cpu.numel()*pool.cpu.element_size())
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return receipt
