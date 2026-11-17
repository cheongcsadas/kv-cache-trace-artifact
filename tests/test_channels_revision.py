import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from attacks.MAA import MAA
from attacks.MSA import MSA
from attacks.SAA import SAA
from attacks.SSA import SSA
from attacks.hardware import Calibration, hardware_activity_windows
from experiments.acquisition.probe import ProbeSettings, calibrate
from experiments.acquisition.metadata import MetadataPublisher
from experiments.acquisition.consumer import MmapMetadataWriter, MmapMetadataReader, JsonlMetadataReader, collect_exports


def metadata(domain=0):
    return dict(request_id="r", start_ms=0, end_ms=30, snapshots=[], events=[
        dict(op="LOAD", layer=0, block_id=domain*20+i, time_ms=float(i+1)) for i in range(4)])


def hardware(domain=0, fail=False):
    return dict(submission_ms=0, capacity_slots=1, ambiguous=fail,
        h2d=[dict(time_ms=float(i), latency_us=230+domain*25 if i<4 or 10<=i<14 else 150) for i in range(60)],
        compute=[dict(time_ms=float(i), latency_us=80 if i>=4 else 20) for i in range(60)], page=[])


class ChannelRevisionTests(unittest.TestCase):
    def test_probe_profile_matches_paper_and_config_overrides_apply(self):
        default = ProbeSettings()
        self.assertEqual((default.h2d_bytes, default.compute_shape, default.probe_pages, default.probe_page_bytes),
                         (1024, (256,256), 64, 2097152))
        altered = ProbeSettings.from_config(dict(h2d_bytes=2048, probe_interval_ms=2, compute_shape=[128,128]))
        self.assertEqual(altered.h2d_bytes, 2048)
        self.assertEqual(altered.probe_interval_ms, 2)
        self.assertEqual(altered.compute_shape, (128,128))
        with self.assertRaises(ValueError): ProbeSettings(h2d_bytes=True)

    def test_calibration_uses_profile_and_rejects_mixed_capture_profiles(self):
        baseline = dict(provenance="measured", h2d=[dict(latency_us=150)],
            compute=[dict(latency_us=float(i)) for i in range(1,101)], page=[])
        config = dict(h2d_threshold_us=240, consecutive_samples=4, probe_interval_ms=2, idle_gap_ms=25)
        result = calibrate([baseline]*3, config)
        self.assertEqual(result["h2d_threshold_us"], 240)
        self.assertEqual(result["sample_interval_ms"], 2)
        self.assertEqual(result["consecutive_samples"], 4)
        self.assertEqual(result["idle_gap_ms"], 25)
        baseline["acquisition_parameters"] = dict(h2d_bytes=2048)
        with self.assertRaises(ValueError): calibrate([baseline]*3)

    def test_empirical_false_positive_bound_and_baseline_reference(self):
        cal = Calibration.fit([[100,200], [120,180], [130,170]], range(1,101), range(1,101))
        self.assertEqual(cal.baseline_h2d_us, 150)
        self.assertLessEqual(sum(v>cal.compute_threshold_us for v in range(1,101)), 5)
        self.assertLessEqual(sum(v<cal.page_threshold_us for v in range(1,101)), 5)
        with self.assertRaises(ValueError): Calibration(150,40,50,consecutive_samples=2.5)

    def test_hardware_windows_are_anonymous_and_pressure_is_increment(self):
        windows = hardware_activity_windows(hardware(), Calibration(150,40,None))
        self.assertEqual(len([w for w in windows if w["op"]=="LOAD"]), 2)
        self.assertTrue(any(w["op"]=="HIT" for w in windows))
        self.assertTrue(any(w["op"]=="EVICT" for w in windows))
        self.assertTrue(all(set(w)=={"op","start_ms","end_ms"} for w in windows))
        clipped = hardware_activity_windows(hardware(), Calibration(150,40,None), max_windows=1)
        self.assertTrue(all(w["end_ms"]<=10 for w in clipped))

    def test_mmap_export_actual_roundtrip_and_deduplicated_log_fusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with MmapMetadataWriter(root/"stream.mmap") as sink:
                with MetadataPublisher(root/"logs.jsonl", "r", sinks=[sink]) as publisher:
                    publisher.snapshot(0, [], source="shm", time_ms=0)
                    publisher.event("LOAD", 0, 2, time_ms=1)
                    publisher.event("HIT", 0, 2, time_ms=2)
                    publisher.snapshot(0, source="shm", time_ms=3)
                observation = collect_exports([root/"logs.jsonl"], [root/"stream.mmap"], request_id="r")
                self.assertEqual(len(observation["events"]), 2)
                self.assertEqual(len(observation["snapshots"]), 2)
                self.assertEqual(len(observation["source_views"]["logs"]["events"]), 2)
                self.assertEqual(len(observation["source_views"]["shm"]["events"]), 2)
                self.assertTrue(all(r["transport"]=="shm" for r in observation["source_views"]["shm"]["events"]))
                with MmapMetadataReader(root/"stream.mmap") as reader:
                    self.assertEqual(len(reader.read_available()), 4)
                    self.assertEqual(reader.read_available(), [])
            with MmapMetadataReader(root/"stream.mmap") as reader:
                reader.read_available()
                self.assertTrue(reader.closed)

    def test_mmap_capacity_and_export_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            with MmapMetadataWriter(Path(tmp)/"small.mmap", capacity_bytes=100) as sink:
                row = dict(record_type="event", request_id="r", time_ms=1, op="LOAD", layer=0, block_id=0)
                with self.assertRaises(BufferError): sink.write(row)
                with self.assertRaises(ValueError): sink.write(dict(row, prompt="private"))

    def test_log_tail_waits_for_full_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"log.jsonl"
            row = dict(record_type="event",request_id="r",time_ms=1,op="LOAD",layer=0,block_id=0)
            path.write_text(json.dumps(row), encoding="utf-8")
            reader = JsonlMetadataReader(path)
            self.assertEqual(reader.read_available(), [])
            with path.open("a",encoding="utf-8") as stream: stream.write("\n")
            self.assertEqual(len(reader.read_available()), 1)
            reader.close()

    def test_all_four_variants_train_and_predict_observations(self):
        domains = ["medical","financial","legal","code"]
        semantic = [dict(domain=d,intent=d+"-i",attribute=d+"-a",keyword=d+"-k") for d in domains]
        for cls, channel, labels in [(MAA,"meta",domains),(MSA,"meta",semantic),(SAA,"hw",domains),(SSA,"hw",semantic)]:
            obs = [metadata(i) if channel=="meta" else hardware(i) for i in range(4)]
            model = cls(n_estimators=100, max_depth=10, seed=42)
            model.fit_observations(obs, labels, calibration=Calibration(150,40,None) if channel=="hw" else None,
                                   max_windows=8)
            predictions = model.predict_observations(obs)
            self.assertEqual(len(predictions),4)
            self.assertTrue(all(p is not None for p in predictions))
            forest = model.domain_classifier if channel and cls in (MSA,SSA) else model.classifier
            self.assertEqual(len(forest.estimators_),100)
            self.assertEqual(forest.max_depth,10)
            if channel=="hw": self.assertEqual(model.predict_observations([hardware(fail=True)]),[None])


if __name__ == "__main__": unittest.main()
