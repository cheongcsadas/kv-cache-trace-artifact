# RetrievalAttention / Qwen2.5

The runtime loads Qwen2.5-7B-Instruct weights and installs its attention path on
every Qwen layer. Prefill uses CPU-backed KV pages and a physically allocated
GPU pool. Decode calls the vector-cache and `retr_attn` routines from
[Microsoft MInference](https://github.com/microsoft/MInference/blob/29ef1974fcb5a1440ccf92d7d7996900230fddcf/minference/modules/retr_attn.py).
The source commit and file digest are checked before model loading.

Use a separate Linux CUDA environment with the dependencies in `requirements.txt`.
Install the CUDA build of PyTorch first and build FlashAttention with
`pip install flash-attn==2.7.3 --no-build-isolation`. Pass the MInference checkout
as `source_dir`; the runtime imports the two pinned definitions directly, so
unrelated MInference sparsity kernels are not installed or executed.

The native module's `papyfaiss.FlatIndex` API is supplied by per-KV-head native
FAISS `IndexFlatIP` indexes. This selects the upstream flat-index retrieval path,
and the receipt records `faiss.IndexFlatIP`. The upstream decoding function runs
CPU vector search and partial attention, runs GPU attention on persistent and
new-token KV, and combines the two outputs using their log normalizers. Two
numerical expressions are stabilized and one head-index assignment is narrowed.
This integration uses the MInference RetrievalAttention implementation; it does
not use the RetroInfer implementation currently in microsoft/RetrievalAttention.

`cache.block_tokens` sets the physical slot width. Per-layer slot count is
`ceil(ceil(input_tokens / block_tokens) * residency_fraction)`, capped to leave a
CPU retrieval partition. Prefill attention uses the Q/K/V projections already on
the execution device with causal scaled-dot-product attention. After establishing
CPU backing, one layerwise H2D batch fills the persistent GPU slots; each completed
copy emits LOAD. Persistent sink/recent pages preserve the native arrangement with a scaled
1:4 block allocation. CPU index construction excludes these resident pages.
Decode preserves the fixed-context pages, uses the upstream temporary cache for
newly generated tokens, and records resident HITs. New-token allocations and
query/result transfers are excluded from management LOAD records.

Configure `inference.dtype`, `inference.generation_tokens`,
`inference.sampling: {do_sample: false}`, and a positive
`cache.selected_token_budget`. These are explicit run settings. The selected
token budget is held fixed across block-size sweeps; retrieval clips it only to
the available CPU partition size. Each request gets its own cache, CUDA stream,
CPU index, and export files while model weights are shared.

The runtime interface is
`Runtime(config, source_dir, model_path=None, device="cuda:0")`,
`describe_request(prompt, max_new_tokens=None)` and
`generate(prompt, request_id, export_dir, max_new_tokens=None)`.
Receipts include actual logical/physical capacity, native provenance and paths
to management logs, sampled telemetry, mmap snapshots and evaluator references.

Set `experiment.defense: fixed_budget` to pad each eligible prefill load batch
to 65 real transfers at 16-token granularity. The policy shuffles required and
padding transfers. Padding pages have pinned CPU backing and one separately
allocated GPU scratch slot per layer; the scratch copies never enter attention
or the CPU retrieval index. Their completed LOAD/EVICT records use the same
metadata schema as context pages. Receipts report scratch allocation and actual
transfer bytes separately from the context-residency budget.

Set `experiment.defense: dynamic_timing` to execute an independent Uniform(0,5 ms)
host delay before each layer's persistent-context loading batch. Decode keeps the native CPU
retrieval path; it adds no KV LOAD events or defense delays to query/result
transfers. `experiment.defense: none` selects the normal Chapter 5 condition.
