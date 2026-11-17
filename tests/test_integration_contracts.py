import json
from pathlib import Path
import tempfile
import unittest

from defenses.dynamic_timing import DynamicTiming
from defenses.fixed_budget import FixedBudget, LoadPlan, eligible
from experiments.acquisition.metadata import BackendHooks, MetadataPublisher, read_export


class IntegrationContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "export.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def test_invalid_snapshot_does_not_change_residency(self):
        with MetadataPublisher(self.path, "request") as publisher:
            publisher.snapshot(0, [1], time_ms=10)
            with self.assertRaises(ValueError):
                publisher.snapshot(0, [2], time_ms=9)
            publisher.event("HIT", 0, 1, time_ms=11)
            self.assertEqual({1}, publisher.resident[0])
        self.assertEqual(1, len(read_export(self.path, "request")["events"]))

    def test_nonfinite_negative_and_fractional_fields_rejected(self):
        with MetadataPublisher(self.path, "request") as publisher:
            for value in (float("nan"), float("inf"), -0.1):
                with self.assertRaises(ValueError):
                    publisher.snapshot(0, [], time_ms=value)
            for layer in (1.5, -1, True):
                with self.assertRaises(ValueError):
                    publisher.event("LOAD", layer, 1, time_ms=1)
            with self.assertRaises(ValueError):
                publisher.event("LOAD", 0, None, time_ms=1)
        self.assertEqual("", self.path.read_text(encoding="utf-8"))

    def test_nonmanagement_operations_are_excluded(self):
        with MetadataPublisher(self.path, "request") as publisher:
            hook = BackendHooks(publisher, "retrievalattention")
            for kind in ("initialization", "allocation", "cleanup", "cpu_attention",
                         "query_transfer", "result_transfer"):
                hook.completed("LOAD", 0, [0], phase="decode", kind=kind)
            self.assertEqual({}, dict(publisher.resident))
            hook.completed("LOAD", 0, [0], phase="prefill", time_ms=1)
        self.assertEqual(1, len(read_export(self.path, "request")["events"]))

    def test_snapshot_excluded_blocks_are_preserved(self):
        with MetadataPublisher(self.path, "request") as publisher:
            publisher.snapshot(0, [1, 2], excluded_blocks=[2], time_ms=0)
        observation = read_export(self.path, "request")
        self.assertEqual([2], observation["snapshots"][0]["excluded_blocks"])

    def test_invalid_fixed_budget_plan_cannot_publish(self):
        defense = FixedBudget(budget=2)
        forged = LoadPlan((1,), (1,), (), True)
        calls = []
        with self.assertRaises(ValueError):
            defense.execute(forged, calls.append, calls.append)
        self.assertEqual([], calls)
        for budget in (True, 0, 1.5):
            with self.assertRaises(ValueError):
                FixedBudget(budget=budget)

    def test_failed_transfer_is_not_published(self):
        defense = FixedBudget(budget=1)
        plan = defense.plan([1], [])
        published = []
        def transfer(block):
            raise RuntimeError("transfer failed")
        with self.assertRaises(RuntimeError):
            defense.execute(plan, transfer, published.append)
        self.assertEqual([], published)

    def test_delay_is_before_load_and_finite(self):
        for value in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                DynamicTiming(value)
        calls = []
        result = DynamicTiming(seed=3).execute(lambda: calls.append("load") or 7,
                                              sleeper=lambda delay: calls.append(delay))
        self.assertEqual(7, result)
        self.assertTrue(0 <= calls[0] <= .005)
        self.assertEqual("load", calls[1])
        with self.assertRaises(ValueError):
            eligible("quest", "unknown")

    def test_reader_rejects_unknown_record_type(self):
        self.path.write_text(json.dumps({"request_id": "request", "record_type": "other"}),
                             encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "record type"):
            read_export(self.path, "request")


if __name__ == "__main__":
    unittest.main()
