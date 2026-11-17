"""Regression checks for metric definitions and denominator/matching errors."""

import unittest

from evaluation.metrics import (
    attribute_metrics, hardware_trr, latency_overhead, macro_average,
    merge_reference_windows, metadata_trr, semantic_metrics,
)


def event(time, op="LOAD", block=1, layer=0, **extra):
    return {"time_ms": time, "op": op, "block_id": block, "layer": layer, **extra}


class PredictionMetricsTest(unittest.TestCase):
    def test_localization_failures_stay_in_denominator_and_confusion(self):
        result = attribute_metrics(
            ["medical", "medical", "legal", "legal"],
            ["medical", None, "medical", "legal"],
        )
        self.assertEqual(result["ai_asr_percent"], 50)
        self.assertEqual(result["failures"], 1)
        self.assertEqual(sum(map(sum, result["confusion_matrix"])), 4)
        self.assertEqual(result["per_domain"]["medical"]["failures"], 1)

    def test_no_silent_dropping_of_labels_or_records(self):
        with self.assertRaises(ValueError):
            attribute_metrics(["medical"], [])
        with self.assertRaises(ValueError):
            attribute_metrics(["medical"], ["legal"], labels=["medical"])
        with self.assertRaises(ValueError):
            attribute_metrics([], [])

    def test_semantic_slot_mean_is_not_joint_accuracy(self):
        truth = [{"domain": "medical", "intent": "consult", "attribute": "lung",
                  "keyword": "lung cancer"}] * 2
        predictions = [dict(truth[0], keyword="cancer"), None]
        result = semantic_metrics(truth, predictions)
        self.assertEqual(result["sr_asr_percent"], 37.5)
        self.assertEqual(result["joint_accuracy_percent"], 0)
        self.assertEqual(result["slot_accuracy_percent"]["domain"], 50)
        self.assertEqual(result["slot_accuracy_percent"]["keyword"], 0)
        self.assertEqual(result["failures"], 1)

    def test_missing_slots_do_not_match_missing_ground_truth(self):
        truth = [{"domain": "legal", "intent": "review", "attribute": "employment",
                  "keyword": "severance"}]
        result = semantic_metrics(truth, [{"domain": "legal"}])
        self.assertEqual(result["sr_asr_percent"], 25)
        with self.assertRaises(ValueError):
            semantic_metrics([{"domain": "legal"}], [None])

    def test_macro_average_does_not_pool_unequal_configuration_sizes(self):
        result = macro_average([
            attribute_metrics(["medical"] * 4, ["medical"] * 4),
            attribute_metrics(["medical"], [None]),
        ], ["ai_asr_percent"])
        self.assertEqual(result["ai_asr_percent"], 50)


class TraceMetricsTest(unittest.TestCase):
    def test_metadata_one_to_one_and_identity_scope(self):
        reference = [event(10), event(20, layer=1), event(30, request_id="a")]
        predicted = [event(10), event(10), event(20), event(30, request_id="b")]
        result = metadata_trr(predicted, reference, tolerance_ms=0)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 3, 2))
        self.assertAlmostEqual(result["trr_percent"], 200 / 7)

    def test_metadata_tolerance_and_chronological_conflicts(self):
        reference = [event(1), event(0)]
        predicted = [event(1), event(0.5)]
        result = metadata_trr(predicted, reference, tolerance_ms=1)
        self.assertEqual(result["matched_indices"], [
            {"predicted": 1, "reference": 1}, {"predicted": 0, "reference": 0},
        ])
        with self.assertRaises(ValueError):
            metadata_trr([{"start_ms": 0, "end_ms": 10, "op": "LOAD",
                           "layer": 0, "block_id": 1}], reference, tolerance_ms=1)

    def test_reference_merging_is_strict_and_does_not_cross_other_operations(self):
        windows = merge_reference_windows([
            event(0), event(49), event(99), event(100, "HIT"), event(101),
        ])
        self.assertEqual([(w["op"], w["start_ms"], w["end_ms"]) for w in windows], [
            ("LOAD", 0, 49), ("LOAD", 99, 99), ("HIT", 100, 100), ("LOAD", 101, 101),
        ])

    def test_reference_requests_never_merge(self):
        windows = merge_reference_windows([
            event(1, request_id="a"), event(2, request_id="b"), event(3, request_id="a"),
        ])
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["request_id"], "a")
        self.assertEqual(windows[0]["end_ms"], 3)

    def test_hardware_ignores_block_identity_but_never_reuses_reference(self):
        reference = [event(0, block=12), event(20, block=19), event(21, "EVICT")]
        predicted = [
            {"op": "LOAD", "start_ms": 0, "end_ms": 15},
            {"op": "LOAD", "start_ms": 10, "end_ms": 22},
            {"op": "EVICT", "start_ms": 21, "end_ms": 21},
        ]
        result = hardware_trr(predicted, reference)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 1, 0))
        self.assertEqual(result["trr_percent"], 80)

    def test_invalid_and_empty_trace_evaluations(self):
        with self.assertRaises(ValueError):
            hardware_trr([{"op": "LOAD", "start_ms": 10, "end_ms": 1}], [])
        with self.assertRaises(ValueError):
            metadata_trr([event(float("nan"))], [], tolerance_ms=1)
        result = metadata_trr([], [], tolerance_ms=1)
        self.assertFalse(result["evaluable"])
        self.assertEqual(result["trr_percent"], 0)
        with self.assertRaises(ValueError):
            macro_average([result], ["trr_percent"])


class LatencyMetricsTest(unittest.TestCase):
    def test_paired_overhead_and_ratio_of_means(self):
        result = latency_overhead({"p1": 10, "p2": 100}, {"p2": 110, "p1": 20})
        self.assertAlmostEqual(result["overhead_percent"], 100 * 20 / 110)
        self.assertAlmostEqual(result["mean_paired_overhead_percent"], 55)
        self.assertEqual(result["n_pairs"], 2)

    def test_unpaired_or_duplicate_measurements_are_rejected(self):
        with self.assertRaises(ValueError):
            latency_overhead({"p1": 10}, {"p2": 10})
        with self.assertRaises(ValueError):
            latency_overhead({"p1": 0}, {"p1": 10})
        with self.assertRaises(ValueError):
            latency_overhead([{"prompt_id": "p1", "latency_ms": 10}] * 2, {"p1": 10})


if __name__ == "__main__":
    unittest.main()
