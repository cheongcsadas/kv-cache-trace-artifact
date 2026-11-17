"""Observation boundaries, censored evidence, and semantic routing regressions."""
import importlib.util
import unittest
import numpy as np
from attacks import Calibration, LocalizationError, encode_hardware, encode_metadata, reconstruct_metadata
from attacks.hardware import localize_hardware, observation_from_probe_rows, FEATURE_NAMES
from attacks.MAA import MAA
from attacks.MSA import MSA
from attacks.SAA import SAA
from attacks.SSA import SSA
from attacks.ablation import encode_ablation


def hardware_fixture():
    active = {5, 6, 7, 16, 17, 18, 45, 46, 47}
    return {"submission_ms": 0., "capacity_slots": 1,
            "h2d": [{"time_ms": i, "latency_us": 250 if i in active else 150} for i in range(80)],
            "compute": [{"time_ms": 8, "latency_us": 50}, {"time_ms": 16, "latency_us": 50}],
            "page": [{"time_ms": 8, "latency_us": 90}, {"time_ms": 9, "latency_us": 110}]}


class MetadataTests(unittest.TestCase):
    def observation(self):
        return {"request_id": "r", "start_ms": 0, "end_ms": 20,
                "snapshots": [{"time_ms": 0, "layer": 1, "resident_blocks": [1]},
                              {"time_ms": 20, "layer": 1, "resident_blocks": [2]}]}

    def test_snapshot_has_intervals_without_invented_hits(self):
        events = reconstruct_metadata(self.observation())
        self.assertEqual({e["op"] for e in events}, {"LOAD", "EVICT"})
        self.assertTrue(all(e["time_ms"] is None for e in events))
        self.assertTrue(all(e["interval_start_ms"] == 0 and e["interval_end_ms"] == 20 for e in events))

    def test_interval_evidence_is_fractional_not_exact_events(self):
        encoded = encode_metadata(self.observation())
        np.testing.assert_allclose(encoded.values[:2, 3:5], .5)
        self.assertEqual(float(encoded.values[:, :3].sum()), 0)
        self.assertEqual(encoded.mask.sum(), 2)
        self.assertEqual(encoded.flatten().shape, (512 * 111,))

    def test_explicit_deduplicates_snapshot_but_keeps_repeated_hits(self):
        observation = self.observation()
        observation["events"] = [
            {"time_ms": 5, "layer": 1, "block_id": 2, "op": "LOAD"},
            {"time_ms": 6, "layer": 1, "block_id": 2, "op": "HIT"},
            {"time_ms": 7, "layer": 1, "block_id": 2, "op": "HIT"}]
        events = reconstruct_metadata(observation)
        self.assertEqual(sum(e["op"] == "LOAD" for e in events), 1)
        self.assertEqual(sum(e["op"] == "HIT" for e in events), 2)

    def test_cleanup_and_allocation_are_not_management_events(self):
        observation = self.observation()
        observation["snapshots"][1]["excluded_blocks"] = [1, 2]
        observation["events"] = [{"time_ms": 10, "layer": 1, "block_id": 1,
                                  "op": "EVICT", "cause": "cleanup"}]
        self.assertEqual(reconstruct_metadata(observation), [])

    def test_request_layer_and_time_isolation(self):
        observation = self.observation()
        observation["events"] = [{"time_ms": 10, "request_id": "other", "layer": 1,
                                  "block_id": 2, "op": "LOAD"}]
        self.assertEqual(len(reconstruct_metadata(observation)), 2)
        observation["snapshots"][1]["layer"] = 2
        self.assertEqual(reconstruct_metadata(observation), [])

    def test_initialization_snapshot_establishes_baseline(self):
        observation = self.observation()
        observation["snapshots"][0]["kind"] = "initialization"
        events = reconstruct_metadata(observation)
        self.assertEqual({e["op"] for e in events}, {"LOAD", "EVICT"})

    def test_right_closed_windows_and_truncation(self):
        observation = {"request_id": "r", "start_ms": 0, "end_ms": 6000,
                       "events": [{"time_ms": 10, "layer": 1, "block_id": 0, "op": "HIT"},
                                  {"time_ms": 5900, "layer": 1, "block_id": 0, "op": "HIT"}]}
        result = encode_metadata(observation)
        self.assertTrue(result.truncated)
        self.assertEqual(result.values[0, 0], 1)
        self.assertEqual(result.values[:, 0].sum(), 1)


class HardwareTests(unittest.TestCase):
    def setUp(self):
        self.calibration = Calibration(150, 40, 100)

    def test_localization_ends_at_first_verified_idle_interval(self):
        bounds = localize_hardware(hardware_fixture(), self.calibration)
        self.assertEqual((bounds["start_ms"], bounds["end_ms"]), (5, 19))
        self.assertEqual(len(bounds["transfer_segments"]), 2)

    def test_missing_idle_tail_is_failure(self):
        observation = hardware_fixture()
        observation["h2d"] = observation["h2d"][:25]
        with self.assertRaises(LocalizationError):
            encode_hardware(observation, self.calibration)

    def test_isolated_spikes_do_not_start_attack(self):
        observation = hardware_fixture()
        observation["h2d"] = [{"time_ms": i, "latency_us": 250 if i % 5 == 0 else 150} for i in range(80)]
        with self.assertRaises(LocalizationError):
            localize_hardware(observation, self.calibration)

    def test_nonconsecutive_missing_samples_do_not_form_batch(self):
        observation = {"submission_ms": 0, "capacity_slots": 1,
                       "h2d": [{"time_ms": i, "latency_us": 250} for i in (0, 4, 8)]}
        with self.assertRaises(LocalizationError):
            localize_hardware(observation, self.calibration)

    def test_strict_threshold_and_page_encoding_direction(self):
        observation = hardware_fixture()
        result = encode_hardware(observation, self.calibration)
        self.assertEqual(result.values.shape, (512, 32))
        self.assertEqual(len(FEATURE_NAMES), 32)
        self.assertEqual(result.mask.sum(), 2)
        self.assertEqual(result.values[0, 23], .5)
        self.assertEqual(result.values[0, 31], 1)
        self.assertEqual(result.values[0, 19], 1)
        self.assertEqual(result.values[1, 19], 0)
        self.assertEqual(result.values[:, 25].sum(), 1)
        self.assertEqual(result.flatten().shape, (512 * 33,))

    def test_h2d_only_ablation_retains_localization(self):
        result = encode_hardware(hardware_fixture(), self.calibration, channels=("h2d",))
        self.assertGreater(result.values[:, 10].sum(), 0)
        self.assertEqual(result.values[:, 16:27].sum(), 0)
        self.assertEqual(result.values[:, 30:32].sum(), 0)

    def test_truncated_transfer_duration_does_not_include_future_capture(self):
        def observation(active_until):
            return {"submission_ms": 0, "capacity_slots": 1,
                    "h2d": [{"time_ms": i, "latency_us": 250 if i < active_until else 150}
                            for i in range(active_until + 30)], "compute": [], "page": []}
        short_tail = encode_hardware(observation(15), self.calibration, max_windows=1)
        long_tail = encode_hardware(observation(80), self.calibration, max_windows=1)
        self.assertTrue(short_tail.truncated and long_tail.truncated)
        np.testing.assert_array_equal(short_tail.flatten(), long_tail.flatten())
        self.assertEqual(short_tail.values[0, 11], 10)
        self.assertEqual(short_tail.values[0, 12], 10)
        self.assertEqual(short_tail.values[0, 13], 10)

    def test_metadata_is_rejected_on_hardware_path(self):
        observation = hardware_fixture()
        observation["reference"] = []
        with self.assertRaises(ValueError):
            encode_hardware(observation, self.calibration)

    def test_ambiguity_is_a_failure(self):
        observation = hardware_fixture()
        observation["ambiguous"] = True
        with self.assertRaises(LocalizationError):
            encode_hardware(observation, self.calibration)

    def test_empirical_false_positive_calibration(self):
        idle = np.arange(1, 101)
        c = Calibration.fit([[149, 151], [148, 150], [151, 153]], idle, idle)
        self.assertEqual(c.baseline_h2d_us, 150)
        self.assertLessEqual(np.mean(idle > c.compute_threshold_us), .05)
        self.assertLessEqual(np.mean(idle < c.page_threshold_us), .05)

    def test_probe_conversion_uses_each_probe_clock(self):
        o = observation_from_probe_rows([{"time_ms": 0, "h2d_timestamp_ms": 1, "h2d_us": 100,
                                         "compute_timestamp_ms": 2, "compute_us": 10}], 0, 1)
        self.assertEqual(o["h2d"][0]["time_ms"], 1)
        self.assertEqual(o["compute"][0]["time_ms"], 2)


class AblationTests(unittest.TestCase):
    def test_removed_hit_cannot_survive_in_aggregate_histogram_or_duration(self):
        common = [{"time_ms": 10, "layer": 1, "block_id": 1, "op": "LOAD"},
                  {"time_ms": 20, "layer": 1, "block_id": 2, "op": "LOAD"}]
        a = {"request_id": "r", "start_ms": 0, "end_ms": 100,
             "events": common + [{"time_ms": 100, "layer": 9, "block_id": 999, "op": "HIT"}]}
        b = {"request_id": "r", "start_ms": -200, "end_ms": 500,
             "events": common + [{"time_ms": t, "layer": 9, "block_id": t, "op": "HIT"} for t in (-200, 3, 500)]}
        ea = encode_ablation(a, "meta", "remove_hit")
        eb = encode_ablation(b, "meta", "remove_hit")
        np.testing.assert_array_equal(ea.flatten(), eb.flatten())
        self.assertEqual((ea.start_ms, ea.end_ms), (10, 20))
        self.assertEqual(ea.values[:, 0].sum(), 0)

    def test_removed_snapshot_load_cannot_survive_in_spatial_features(self):
        def observation(block):
            return {"request_id": "r", "start_ms": 0, "end_ms": 20,
                    "snapshots": [{"time_ms": 0, "layer": 0, "resident": [1]},
                                  {"time_ms": 20, "layer": 0, "resident": [block]}]}
        a = encode_ablation(observation(2), "meta", "remove_load")
        b = encode_ablation(observation(50), "meta", "remove_load")
        np.testing.assert_array_equal(a.flatten(), b.flatten())
        self.assertEqual(a.values[:, 3].sum(), 0)
        self.assertGreater(a.values[:, 4].sum(), 0)

    def test_remove_all_redacts_timing_and_mask(self):
        obs = {"request_id": "r", "start_ms": 0, "end_ms": 500,
               "events": [{"time_ms": 10, "layer": 1, "block_id": 1, "op": "LOAD"}]}
        self.assertFalse(encode_ablation(obs, "meta", "remove_all").flatten().any())
        self.assertFalse(encode_ablation(hardware_fixture(), "hw", "remove_all", Calibration(150, 40, 100)).flatten().any())

    def test_hardware_load_removal_removes_derived_dependencies(self):
        result = encode_ablation(hardware_fixture(), "hw", "remove_load", Calibration(150, 40, 100))
        self.assertEqual(result.values[:, :16].sum(), 0)
        self.assertEqual(result.values[:, 19].sum(), 0)
        self.assertEqual(result.values[:, 25:27].sum(), 0)
        self.assertEqual(result.values[:, 29].sum(), 0)
        self.assertGreater(result.values[:, 16].sum(), 0)
        self.assertGreater(result.values[:, 20].sum(), 0)

    def test_probe_source_comparisons_do_not_include_extra_channels(self):
        c = Calibration(150, 40, 100)
        h2d = encode_ablation(hardware_fixture(), "hw", "h2d_only", c)
        page = encode_ablation(hardware_fixture(), "hw", "page_only", c)
        fusion = encode_ablation(hardware_fixture(), "hw", "fusion", c)
        self.assertEqual(h2d.values[:, 16:27].sum(), 0)
        self.assertEqual(page.values[:, :20].sum(), 0)
        self.assertEqual(fusion.values[:, 16:20].sum(), 0)
        self.assertEqual(fusion.values[:, 25:27].sum(), 0)
        np.testing.assert_array_equal(fusion.values[:, :16], h2d.values[:, :16])
        np.testing.assert_array_equal(fusion.values[:, 20:25], page.values[:, 20:25])

    def test_ablated_failed_localization_still_fails(self):
        obs = hardware_fixture()
        obs["ambiguous"] = True
        with self.assertRaises(LocalizationError):
            encode_ablation(obs, "hw", "remove_all", Calibration(150, 40, 100))

    def test_metadata_rejects_probe_ablation(self):
        with self.assertRaises(ValueError):
            encode_ablation({}, "meta", "h2d_only")


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
class ModelTests(unittest.TestCase):
    def test_paper_forest_parameters_and_separate_domain_model(self):
        X = np.array([[0], [0], [1], [1]], dtype=float)
        y = ["medical", "medical", "code", "code"]
        labels = [{"domain": d, "intent": d + "_intent", "attribute": d + "_attribute",
                   "keyword": d + "_keyword"} for d in y]
        attribute = MAA(seed=2).fit(X, y)
        semantic = MSA(seed=2).fit(X, labels)
        self.assertEqual(attribute.classifier.n_estimators, 100)
        self.assertEqual(attribute.classifier.max_depth, 10)
        self.assertIsNot(attribute.classifier, semantic.domain_classifier)
        self.assertEqual(SAA.channel, "hw")
        self.assertEqual(SSA.channel, "hw")

    def test_downstream_routing_uses_prediction_not_true_domain(self):
        X = np.array([[0], [0], [1], [1]], dtype=float)
        labels = [{"domain": "medical" if i < 2 else "code", "intent": str(i // 2),
                   "attribute": "a" + str(i // 2), "keyword": "k" + str(i // 2)} for i in range(4)]
        attack = SSA().fit(X, labels)
        class AlwaysCode:
            def predict(self, X):
                return np.array(["code"] * len(X))
        attack.domain_classifier = AlwaysCode()
        predicted = attack.predict(X[:1])
        self.assertEqual(predicted, [{"domain": "code", "intent": "1", "attribute": "a1", "keyword": "k1"}])

    def test_multi_label_keyword_is_rejected(self):
        with self.assertRaises(ValueError):
            MSA().fit([[0]], [{"domain": "medical", "intent": "x", "attribute": "a", "keyword": ["a", "b"]}])


if __name__ == "__main__":
    unittest.main()
