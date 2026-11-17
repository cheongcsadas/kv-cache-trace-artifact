# Experiment configuration

## Fixed parameters

The three system/model families, two observation channels, four classifiers, source-disjoint 75/25 split,
10 ms windows, 512-window mask/truncation, RF parameters, predicted-domain semantic routing,
single-label keyword scoring, failure denominators, 50 ms reference merging, and defense parameters
follow Sections 4-6 and Appendices C-E. Native metadata hooks follow Table 1.

## Representation and calibration

| Item | Configuration |
|---|---|
| Random seed | 42, configurable |
| Feature vector | 32 hardware features; 110 metadata features with 32 bins per operation |
| Metadata spatial hashing | BLAKE2 of layer/block |
| Snapshot transitions | interval-overlap weights |
| Hardware gap tolerance | 1.5× target sample interval |
| Calibration quantile | conservative empirical lower/upper quantiles for ≤5% baseline positives |
| Page scan | 64 independent 2 MiB CUDA VMM allocation handles; stride 4096; 64 blocks ×128 CUDA threads |
| Auxiliary preparation | scan 64 separate equal-size auxiliary allocations before each page invocation |
| H2D scheduling | independent worker, 1 ms absolute deadlines; actual timestamps and missed deadlines retained |
| Input templates | shared-v1/v2/v3, with separate train/test template IDs for cross-template runs |

The hardware page scan is a generic cross-page measurement. One CUDA-event measurement covers the
complete invocation across all pages. Its event starts after auxiliary preparation on the same stream.
Use the hardware feature names in attacks/hardware.py when defining an experiment's representation.
The configurable feature layout and template text are repository implementation choices.

## GPU preparation

1. Create a separate Linux/CUDA environment for each selected backend and install its build dependencies.
2. Run `scripts/setup_native.py` with the `--model-revision` shown in [the integration commands](INTEGRATION.md) to fetch pinned sources and the checkpoint matching the prepared tokenizer assets.
3. Explicitly choose `cache.selected_token_budget` and `inference.generation_tokens`; the native
   runtime calculates actual slot capacity from token count, block size and residency fraction.
4. Use `backends.server` or a Chapter 5 `--native-manifest` entry point. Native runtimes export
   completed transfers/reuse/release through `NativeExport` and preserve separate reference records.
5. Prepare source-disjoint train/test records and per-domain 3/60/60 semantic labels.
6. Calibrate three auxiliary baseline repetitions, then freeze thresholds before collecting test observations.
7. Each native request creates an empty KV pool while reusing model weights; the observer runs separately.
8. Use a common host monotonic clock for the submission anchor and observer. Store ambiguity decisions
   with each attempt. Include unsuccessful localization attempts in the evaluation input.

## Result files

The runner records configuration, calibration, package versions, input digests, predictions and localization
failures with each experiment. `synthetic` denotes pipeline fixtures; `measured` denotes collected observations.
Keep the raw observations and acquisition receipts alongside measured result files.

The classifier receives channel-specific observations only. Reference events stay in the evaluation path.
For metadata TRR, choose and record `tolerance_ms` before evaluation. For hardware TRR, use same-type
reference windows with the configured 50 ms merge criterion. Preserve interval evidence as intervals.

Run a fresh training step for each feature ablation. Defense-aware runs use defended train and test data;
defense-oblivious runs use undefended training data and defended test data. Measure latency using matched
prompt IDs, including actual transfer completion and actual inserted delays.
