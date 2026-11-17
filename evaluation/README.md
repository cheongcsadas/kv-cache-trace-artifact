# Evaluation contracts

`attribute_metrics(truth, predictions)` computes AI-ASR, per-domain scores, Macro-F1 and a confusion matrix.
Supply `None` for each failed localization. A missing row is not an acceptable representation of a failed attempt.

`semantic_metrics(truth, predictions)` accepts mappings with `domain`, `intent`, `attribute`, `keyword`.
SR-ASR is the equally weighted mean of four slot accuracies. Joint accuracy is a separate metric.
Keywords with spaces are whole single labels. Use unambiguous cross-domain label strings.

`metadata_trr(predicted, reference, tolerance_ms=...)` matches type, layer, block, request when present, and timing.
Snapshot-derived intervals remain intervals in the encoder; declare how they are scored instead of inventing event timestamps.
`hardware_trr(predicted_windows, reference_events, merge_gap_ms=50)` matches type and overlap only.
Reference event lists must be ordered on a consistent clock; consecutive same-type events separated by <50 ms merge.
An intervening different operation splits the run. The endpoint convention is closed, preserving singleton event windows.
Matching is one-to-one in chronological order. Do not compare hardware-window TRR numerically as if it were block-event TRR.

`latency_overhead(baseline, defended)` requires identical prompt IDs with positive measured latency values.
`macro_average(results, keys)` averages valid per-system metrics and does not silently replace an unmeasured system with zero.
See tests/test_metrics.py for executable examples and exact accepted schemas.
