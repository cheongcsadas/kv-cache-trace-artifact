# Quest + Llama-2-7B

The runtime loads `quest.models.llama.LlamaForCausalLM` from upstream commit
`01c1623bf9395009520874e989e29f683203b357`. Five native source files are checked
against pinned SHA-256 digests before import. Build upstream `quest._kernels`
from that checkout and install this directory's requirements in its own environment.
Run `python scripts/build_backend.py --source-dir external/Quest --cuda-architectures 89`
inside that environment to compile and install the operators, using the pinned RAFT,
FlashInfer and pybind submodules. The build requires gcc-11/g++-11, CMake and Ninja.

`Runtime(config, source_dir, model_path=None, device="cuda:0")` loads model weights
once. `describe_request` tokenizes the actual input and calculates its pool budget;
`generate` performs prefill, autoregressive decoding, and metadata export.
Set `cache.selected_token_budget`, `inference.generation_tokens`,
`inference.dtype: float16`, and `inference.sampling: {do_sample: false}` before running.
The selected-token budget must be divisible by the configured block size and
remain constant when comparing the 8/16/32-token block conditions.

The native Quest CUDA estimator and top-k selector operate on the model's actual
query/key statistics. Prefill establishes pinned CPU KV backing. The full prefill
GPU tensor is then released, and decode uses a physically allocated pool of
`ceil(residency_fraction * ceil(prompt_tokens / block_tokens))` pages per layer.
One slot holds the mutable tail. Historical pages occupy the remaining slots.
The receipt gives both this per-layer budget and the total physical slot count.

Selected pages are streamed through the bounded pool when the union of per-head
top-k selections exceeds capacity. An online FP32 softmax combines these chunks
while preserving each head's native selection. This is the artifact's offloading
extension to Quest; the selector itself comes from the pinned upstream kernels.
LOAD is exported after a real pinned-memory H2D transfer; HIT after resident data
has been consumed; EVICT after prior attention reads finish and a slot is released.
Prefill initialization and mutable-tail allocation are excluded from those events.

Requests share read-only model weights and use ContextVar-local controllers,
KV pools, kernel workspaces, and CUDA streams. Upstream kernels that launch on
CUDA stream 0 have explicit event handoffs at their call boundaries. Those native
kernel calls share stream 0; request weight operations and streamed attention use
the request streams. Overlapping requests therefore increase live KV memory but
do not allocate another copy of the model weights.

For `experiment.defense: dynamic_timing`, each layer/decode-step loading operation
executes one configured uniform delay before its first actual H2D copy. For
`fixed_budget`, the whole operation shares one 65-page real/padding transfer budget;
internal streaming chunks do not restart that budget. Existing resident operands
are consumed before shuffled misses can replace them. Each required page is
consumed immediately after loading, preserving the GPU slot bound and selected
attention inputs. Operations exceeding 65 required pages retain all transfers
and record the overflow. Hit-only operations add neither padding nor delays.
Padding has pinned CPU
backing and a separate GPU scratch page per layer, is excluded from attention,
and uses the same exported event schema. Scratch bytes are recorded separately
from the configured inference KV pool.

The runtime exports actual logs, independently sampled telemetry/shared-memory
residency views, and a separate evaluator reference stream through `NativeExport`.
