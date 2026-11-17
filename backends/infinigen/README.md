# InfiniGen with Mistral-7B

`runtime.Runtime` loads Mistral checkpoint weights and runs prefill and token-by-token
decoding with InfiniGen prediction, a pinned host KV store, and bounded device KV
slots. Generation uses the actual embedding, RMSNorm, projections, MLP, and language
model head from the checkpoint. It does not invoke an ordinary full-cache generation
path in place of predictive offloading.

The InfiniGen source is pinned to
[`3754f52d131c30ab7a310e3c4cf356e8e3326ac3`](https://github.com/snu-comparch/InfiniGen/tree/3754f52d131c30ab7a310e3c4cf356e8e3326ac3).
The runtime loads the original column-selection and speculation routines, verifies
their Git blob fingerprints, and adapts the Llama RoPE rehearsal to Mistral grouped
query attention. The run receipt identifies this integration as
`mistral-gqa-predictive-offload-port-v1`. The upstream speed backend targets OPT;
this Mistral port is a separately identified integration.

Each layer keeps full K/V on pinned CPU memory and a reduced partial-key cache on
GPU. During decoding the normalized input of layer `i` rehearses the query for
layer `i+1`. Native threshold/top-k speculation chooses historical tokens, which
map to physical blocks. The next layer's slots are prefetched, and selected tokens
are evaluated with online log-sum-exp attention. The current token uses freshly
computed K/V directly. Orthogonal skewing follows RoPE, preserving Q/K dot products;
only the prediction uses reduced columns. GQA K/V is shared across query heads.

`cache.block_tokens` controls physical slot length. `cache.residency_fraction`
controls the allocated slots in every layer/head bank:
`ceil(ceil(context_tokens / block_tokens) * residency_fraction)`.
The bank size remains fixed throughout one request. A cache miss copies real K/V
from pinned host backing; replacement uses LRU within that bank. A selected set
larger than the bank is streamed through the same slots. Attention workspaces and
the reduced partial-key cache are accounted separately from resident full K/V.

`capacity_slots` in the receipt is one bank's block capacity; `capacity_unit` is
`per_layer_per_kv_head_block`. `physical_capacity_slots` reports the allocation over
all layers and KV heads. Exported IDs `head:block` preserve head-specific selection.
Only completed copies, reuse, and releases produce LOAD, HIT, and EVICT. New-token
allocation and request cleanup do not produce management events.

Supply `cache.selected_token_budget`, `inference.generation_tokens`, and the model
checkpoint path or revision in the run configuration. The predictor's `alpha=5`
and `partial_weight_ratio=0.2` are configurable InfiniGen port settings; they are
recorded in the receipt, separately from the KvTrail paper's fixed parameters.
The current decoding policy is FP16 and greedy with a fixed generation length.
The base checkpoint is `mistralai/Mistral-7B-v0.1`; an instruction-tuned checkpoint
is a different experimental condition.

The common local server constructs:

```python
from backends.infinigen.runtime import Runtime

runtime = Runtime(config, source_dir="external/InfiniGen", model_path="models/Mistral-7B-v0.1")
description = runtime.describe_request(prompt, max_new_tokens=128)
receipt = runtime.generate(prompt, "request-001", "runs/request-001", max_new_tokens=128)
```

Every request has independent CPU/GPU pools, prediction state and a CUDA stream;
model weights are read-only and reused. The common export writer produces logs,
telemetry snapshots, a mmap transport, and an evaluator-only reference file.

## Loading defenses

Set `experiment.defense` to `fixed_budget` or `dynamic_timing` to execute the
configured defense in the actual KV transfer path. One eligible operation is a
KV head's complete loading work in one layer/decode step, spanning its early
prefetch and all associated GQA query-head demand accesses. A simulated LRU plan
accounts for repeated physical loads when selected tokens exceed the bank, and
distributes padding across the operation to reach `defenses.fixed_budget.load_pages`
(65 by default). Required transfers are shuffled within the dependency-ordered
prefetch and demand sub-batches and interleaved with padding. Required blocks are consumed after
their completed transfers; padding copies land in a separate scratch slot and
are excluded from attention. Hits are consumed before shuffled misses may replace
their slots. The online softmax combines the same selected token set.

Hit-only batches generate no padding. If a required set exceeds 65 blocks, all
required transfers execute, padding is zero, and the receipt counts the overflow
batch. Reserved padding pages use the same logical-ID format and event schema as
ordinary pages. Their initialization is separate from LOAD; their actual H2D
copies and scratch releases produce completed LOAD/EVICT records.

Timing defense draws an independent Uniform(0,5 ms) host delay immediately before
the first actual missing-block copy in the whole operation; resident hits receive no delay. The run receipt
records the batch unit, required/padding transfer counts, bytes, scratch allocation,
overflow count and delay count. These are evaluator-side accounting fields, not
observer-visible padding labels.
