# Run Chapter 5 from native inference

Each experiment directory's existing entry point accepts `--native-manifest`.
It starts the selected backend, loads the model once per physical condition,
submits the rendered prompts, collects operation exports or independent hardware
measurements, and passes those observations into the experiment's evaluator.

```bash
python -m experiments.e03_attribute_inference --native-manifest native-project.json --output runs/e03
python -m experiments.e09_cache_sensitivity --native-manifest native-project.json --output runs/e09
```

`--capture-output` selects the observation directory; the default is the output
path plus `-native`. `--manifest` continues to evaluate existing observations.
`--pilot` permits smaller datasets while retaining the same method settings.
Native collection never enables synthetic observations.

Keep the domain and semantic preparation outputs separate, then join both into
one native project. The project builder accepts the domain `project_data.json`
through `--data` and the semantic file through `--semantic-data`:

```bash
python scripts/build_native_project.py \
  --data prepared/domain/project_data.json \
  --semantic-data prepared/semantic/project_data.json \
  --config quest=local/quest.yaml --config infinigen=local/infinigen.yaml \
  --config retrievalattention=local/retrievalattention.yaml \
  --python quest=.venv-quest/bin/python --python infinigen=.venv-infinigen/bin/python \
  --python retrievalattention=.venv-retrieval/bin/python \
  --metadata-time-tolerance-ms 1 --output native-project.json
```

MAA/SAA consume `datasets` and `profiling` with domain targets only. MSA/SSA and
the within-domain intent experiment consume `semantic_datasets`; the semantic
profiling-size conditions use `semantic_profiling`. Semantic lists and annotation
rules are prepared using `datasets/preprocessing/README.md`. Existing projects
with a single fully labeled pool continue to use `datasets` and `profiling`.

The project JSON uses paths relative to itself. Its main structure is:

```json
{
  "backends": {
    "quest": {"config":"configs/quest.yaml","source_dir":"external/Quest","model_path":"models/Llama-2-7b-hf","python":".venv-quest/bin/python","device":"cuda:0"},
    "infinigen": {"config":"configs/infinigen.yaml","source_dir":"external/InfiniGen","model_path":"models/Mistral-7B-v0.1","python":".venv-infinigen/bin/python","device":"cuda:0"},
    "retrievalattention": {"config":"configs/retrievalattention.yaml","source_dir":"external/MInference","model_path":"models/Qwen2.5-7B-Instruct","python":".venv-retrieval/bin/python","device":"cuda:0"}
  },
  "generation_tokens":128,
  "selected_token_budget":512,
  "metadata_time_tolerance_ms":1,
  "capture_seconds":8,
  "baseline_seconds":8,
  "arrival_gap_ms":1,
  "datasets": {
    "quest": {"templated_short":{"train":"prepared/quest/ts-train.jsonl","test":"prepared/quest/ts-test.jsonl"}}
  },
  "profiling":{"quest":{"templated_short":{"train":"prepared/quest/ts-profiling-train.jsonl","test":"prepared/quest/ts-test.jsonl"}}},
  "semantic_datasets":{"quest":{"templated_short":{"train":"prepared/semantic/quest/ts-train.jsonl","test":"prepared/semantic/quest/ts-test.jsonl"}}},
  "semantic_profiling":{"quest":{"templated_short":{"train":"prepared/semantic/quest/ts-profiling-train.jsonl","test":"prepared/semantic/quest/ts-test.jsonl"}}},
  "auxiliary":{"quest":"prepared/quest/auxiliary.jsonl"},
  "contexts":{"quest":{"1000":{"train":"prepared/quest/1k-train.jsonl","test":"prepared/quest/1k-test.jsonl"}}},
  "transfer_test":{"quest":"prepared/quest/transfer-test-1000.jsonl"},
  "intent_pairs":{"medical":{"positive":"clinical_diagnosis","negative":"clinical_explanation","channel":"meta"}}
}
```

Fill the system/setting entries required by the selected experiment. The displayed
generation length, selection budget, matching tolerance and arrival spacing are
example run choices, not additional paper constants. They are explicit so each
collected condition records what was executed. `generation_tokens` and
`selected_token_budget` may also be set separately in each backend entry.
An optional `probe_python` selects a shared CUDA probe environment; otherwise the
backend Python interpreter is used. Calibration runs the loaded model idle and
launches three independent probe processes, then freezes the measured thresholds.

Each rendered JSONL row contains a unique `sample_id` (or `id`), `source_id`,
`domain`, and `prompt` (or `text`). Semantic experiments also require curated
`labels` with `domain`, `intent`, `attribute`, and `keyword`. Cross-template rows
retain `template_id`, with disjoint templates and sources between train and test.
Labels are evaluator inputs and never enter the hardware observation.
Curated supplementary text retains `support_source_ids` and `source_spans` through
collection; all primary and supplementary sources participate in split isolation.
An explicit `label_vocabulary` carries the actual semantic classes into training.

| Experiment | Native inputs and execution |
|---|---|
| e01 | `auxiliary` for all three systems, sharing 1,000 source records; logs, telemetry, mmap and reference exports come from actual operations |
| e02 | Same auxiliary pool; independent hardware observer and evaluator-only reference events |
| e03 | All systems and five prompt settings; metadata and hardware collections feed MAA/SAA |
| e04 | Quest `templated_short` with curated slots; MSA/SSA use separate channel captures |
| e05 | Main attribute and semantic grids with failed attempts retained |
| e06 | Quest short inputs and `intent_pairs` for all four domains; each pair declares its channel |
| e07 | Quest short training inputs plus `transfer_test.quest` with 1,000 test prompts total; all three feature variants reuse the same hardware captures/calibration |
| e08 | Full/remove-HIT/remove-LOAD/remove-EVICT/remove-all variants reuse identical measured records |
| e09 | Actual server cache configurations change residency/block size; `contexts[system]["1000"/"2000"/"3000"/"4000"]` supplies the actual tokenized prompt conditions |
| e10 | 1/4/8/16 request schedules; backgrounds use training prompts, distinct arrival offsets, independent request caches and backend CUDA streams |
| e11 | 3,000 short training prompts per domain; collect time excludes backend startup and baseline calibration |
| e12 | `profiling[system][setting]` for domain attacks and `semantic_profiling[system][setting]` for semantic attacks supply short/long training files with 3,200 prompts per domain and fixed test files; nested 400/800/1600/3200 subsets reuse the expanded captures |

The bridge checks source isolation, prompt availability, quotas and semantic labels
before launching models. Token counts come from the loaded backend tokenizer and
the execution receipt; changing a context label does not rewrite or truncate text.
Prepare each context file with that same tokenizer using the preprocessing commands.
Before paper-mode collection, the backend tokenizer checks each partition's actual
mean length against the short/long/context condition. The declared operational
tolerance `context_mean_tolerance_fraction` defaults to 0.1; pilot mode allows
small inputs. This tolerance is a collection check, not a reported paper constant.
The block-size sweep keeps the selected-token budget fixed.

e10 validates overlap of backend execution intervals rather than HTTP connection
lifetimes. A failed background request cannot establish the requested concurrency.
Hardware acquisition keeps its observer separate from the inference process.
The capture cache distinguishes system, channel, physical configuration and input
content digests; classifier/feature choices share data without re-running inference.
