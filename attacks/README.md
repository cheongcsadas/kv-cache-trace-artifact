# Observation and inference API

The four attack packages separate observation channels and inference targets:

| Package | Channel | Output |
| --- | --- | --- |
| `MAA` | exported management metadata | domain |
| `MSA` | exported management metadata | domain, intent, attribute, keyword |
| `SAA` | anonymous attacker-side probe measurements | domain |
| `SSA` | anonymous attacker-side probe measurements | domain, intent, attribute, keyword |

`from attacks.MAA import MAA` constructs a classifier with `fit(X, y)` and
`predict(X)`. Semantic variants accept a list of dictionaries with `domain`,
`intent`, `attribute`, and `keyword` strings. Each semantic model trains its own
domain classifier. Predicted domains select downstream classifiers; inference
does not accept true domains. Keywords are single labels, not multilabel sets.
The default forest has 100 estimators and maximum depth 10 (Appendix E).
Seed 0 and one training worker are reproducibility choices, not paper parameters.

Both encoders return `EncodedTrace`: `values`, `mask`, `feature_names`, `channel`,
`start_ms`, `end_ms`, and `truncated`. `flatten()` appends the binary mask to the
flattened window features. Windows are 10 ms, with at most 512 windows. Padded
windows are zero; observed idle windows retain a true mask. Windows are
right-closed `(a,b]`, with the initial onset assigned to the first window.

## Metadata

```python
from attacks import encode_metadata
observation = {
    "request_id": "r1", "start_ms": 0., "end_ms": 20.,
    "events": [{"time_ms": 5., "layer": 0, "block_id": 3, "op": "LOAD"}],
    "snapshots": [
        {"time_ms": 0., "layer": 0, "resident_blocks": []},
        {"time_ms": 20., "layer": 0, "resident_blocks": [3]},
    ],
}
x = encode_metadata(observation).flatten()
```

All times are milliseconds on a shared monotonic clock. `timestamp_ms` is an
alias of `time_ms`; `resident` is an alias of `resident_blocks`. The publisher
must export only previously computed, CPU-backed residency, and identify
allocation, initialization, cleanup, query/result transfer, and CPU-only
attention records using `kind` or `cause`. `excluded_blocks` removes known
allocation/cleanup effects from snapshot differences. Wrong-request records are
ignored. Different layers and source streams are differenced separately.

`reconstruct_metadata()` retains exact event times and interval-censored
snapshot evidence. Snapshots cannot reveal HIT, invisible load/evict cycles, or
event order inside an interval. Explicit operations take precedence over
overlapping interval evidence for the same layer, block and operation. Duplicate
transport records should be deduplicated by stable event ID at acquisition.
Export explicit HIT records at native GPU-resident reuse sites.

The metadata representation uses 14 scalar features and 32 stable BLAKE2 block-identity bins per operation
(110 features per window). Block hashes include layer IDs; collisions are
possible. Snapshot evidence contributes fractional overlap weight to windows,
and never becomes an exact timestamp. These are configurable representation choices.

## Hardware

```python
from attacks import Calibration, encode_hardware
calibration = Calibration.fit(
    baseline_repetitions=[[149., 151.], [148., 150.], [151., 153.]],
    compute_idle_us=[10., 11., 12., 13.], page_idle_us=[90., 95., 100., 105.],
)
# measurement contains submission_ms, capacity_slots, and independent h2d,
# compute, page lists of {time_ms, latency_us}; no management or prompt records.
# x = encode_hardware(measurement, calibration).flatten()
```

Calibration must use disjoint auxiliary runs. Baseline H2D latency is the median
of three repetition medians. Computation uses a conservative empirical 95th
percentile and a strict greater-than test. Page response uses a conservative
5th percentile and a strict less-than test, matching Eq. (13). The paper gives
the false-positive target, but not this quantile estimator. Thresholds are frozen
before test inference. A missing optional threshold is permitted only when its
probe is absent or its features are excluded.

The H2D detector requires three consecutive samples strictly above 200 us. The
first sustained segment after the submission anchor supplies the onset. A
verified 20 ms below-threshold interval supplies the candidate end. An
insufficient recording, absence of a sustained segment, or an explicitly marked
ambiguous association raises `LocalizationError`. Every such attempt must remain
in end-to-end evaluation denominators. This localization is not a claim that all
decoding has finished. For irregular recordings, a gap larger than 1.5 times the
nominal 1 ms probe interval breaks a run; this missing-sample rule is an artifact
choice absent from the paper. Generic overlapping workloads cannot be identified
reliably from these signals alone; acquisition should set `ambiguous=true` when
request association is known to be ambiguous from attacker-visible scheduling.

`hardware.FEATURE_NAMES` enumerates all 32 features, covering H2D latency and
excess statistics, sample counts, segment counts and duration, inter-segment gaps,
computation activity, anonymous page invocation responses, and replacement
pressure. Segment statistics are assigned by onset. Every page latency is a
whole 64-page kernel invocation, never an individual page or victim KV block.
The exact 32-dimensional list is an artifact choice because Appendix E gives
feature families and an approximate dimension, not the original feature order.

Pressure follows `max(0, cumulative_detected_batches - capacity_slots)`; its
increment is expressed in detected batches. It does not count evicted blocks.
Probe records containing block identities, reference events, prompt data or KV
contents are rejected. `channels=("h2d",)` runs the transfer-only feature
ablation; all variants still use H2D for request localization.

Neither channel reconstructs verbatim prompts. Models output closed-set labels;
no accuracy from synthetic fixtures is a measured reproduction of the paper.

## Retrained ablations

MSA/SSA training retains the fixed domain-specific vocabulary through observation
localization. Both the command-line runner and `fit_observations` reject a filter
that removes the final example of any training class. Each fitted semantic forest
exposes its actual `classes_`; `training_support` records the examples passed to
each head, and `fitted_sample_ids` identifies the successful training observations.

Measured semantic rows carry an `annotation` receipt binding their source labels
and evidence to `prompt_sha256`, together with `observation_sha256` binding the
collected trace. File observations use the SHA-256 of the exact file bytes. Inline
observations use UTF-8 JSON with sorted keys, compact separators, unescaped Unicode
and no non-finite numbers. These audit fields are separate from the encoder input.

`encode_ablation(observation, channel, ablation, calibration=None, **kwargs)` is
the shared train/test/predict entry point. `channel` is `meta` or `hw`; keyword
arguments are passed to the corresponding encoder. Supported settings are:

| Setting | Metadata representation | Hardware representation |
| --- | --- | --- |
| `full` | complete representation | all four feature families |
| `remove_hit` | discard HIT before aggregation | exclude computation features |
| `remove_load` | discard explicit and snapshot LOAD | exclude H2D, pressure and candidate-HIT fraction |
| `remove_evict` | discard explicit and snapshot EVICT | exclude pressure |
| `remove_all` | constant zero feature vector and mask | constant zero feature vector and mask |
| `h2d_only` | unavailable | H2D latency and transfer-batch features |
| `page_only` | unavailable | anonymous page-probe features (paper's TLB-only) |
| `fusion` | unavailable | H2D plus page-probe features |

Each setting must retrain its own forest(s). Metadata removal happens before
event totals, unique counts, timing statistics and spatial hashes. Its origin and
duration are recomputed from retained evidence, preventing removed timestamps
from leaking through offsets or padding. If no evidence remains, the encoded
vector and mask are zero. Snapshot intervals stay interval-censored.

Hardware LOAD removal also removes its derived pressure and candidate-HIT
features. The three probe-source comparisons exclude computation and pressure
features. They share the H2D-localized observation interval and associated
mask/offsets: `page_only` therefore means page-only inference features conditional
on transfer-based localization. It does not demonstrate independent page-only
request localization. Localization failures remain failures in every setting.

The event-group definitions retain page features in individual event removals and remove them in the strict
`remove_all` control. This control also removes time and mask features so the
representation cannot retain request duration. Record these feature-group definitions
with each experiment's measured inputs and configuration.
