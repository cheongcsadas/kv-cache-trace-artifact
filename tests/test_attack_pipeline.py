"""Small observation-to-checkpoint regressions using the actual four attacks."""
import importlib
import json
from pathlib import Path

import joblib
import pytest

from attacks.hardware import Calibration, encode_hardware
from attacks.metadata import encode_metadata


ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("medical", "financial", "legal", "code")
run = importlib.import_module("experiments.run").run


def observation(channel, domain_index):
    if channel == "meta":
        return dict(request_id="target", start_ms=0, end_ms=30, events=[
            dict(op="LOAD", layer=domain_index, block_id=domain_index*10+i, time_ms=i+1)
            for i in range(domain_index+1)])
    return dict(submission_ms=0, capacity_slots=1,
                h2d=[dict(time_ms=i, latency_us=230+domain_index*40 if i < 4 else 150)
                     for i in range(40)],
                compute=[dict(time_ms=4, latency_us=50+domain_index)],
                page=[dict(time_ms=4, latency_us=80+domain_index)])


@pytest.mark.parametrize("attack,channel", [
    ("MAA", "meta"), ("MSA", "meta"), ("SAA", "hw"), ("SSA", "hw"),
])
def test_actual_encoder_forests_checkpoint_and_failed_attempts(tmp_path, attack, channel):
    paths = {}
    for split in ("train", "test"):
        rows = []
        for index, domain in enumerate(DOMAINS):
            sample_id = f"{split}-{domain}"
            path = tmp_path / f"{sample_id}.json"
            path.write_text(json.dumps(observation(channel, index)), encoding="utf-8")
            rows.append(dict(sample_id=sample_id, source_id=sample_id, domain=domain,
                             system="quest", provenance="synthetic", defense="none",
                             observation_path=path.name, request_status="completed",
                             acquisition_status="completed",
                             labels=dict(domain=domain, intent=domain+"-i",
                                         attribute=domain+"-a", keyword=domain+"-k")))
        if split == "test":
            # Acquisition can fail before an observation file exists. The row
            # still carries its labels and must survive in every denominator.
            rows.append(dict(rows[0], sample_id="test-failure", source_id="failed-source",
                             acquisition_status="failed", observation_path="absent.json"))
        paths[split] = tmp_path / f"{split}.jsonl"
        paths[split].write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    calibration = Calibration(150, 40, 100)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration.to_dict()), encoding="utf-8")
    output = tmp_path / "run"
    report = run(ROOT/"configs/quest.yaml", paths["train"], paths["test"], output,
                 attack, calibration_path if channel == "hw" else None, allow_synthetic=True)
    predictions = [json.loads(line)["prediction"]
                   for line in (output/"predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(predictions) == 5 and predictions[-1] is None
    assert report["metrics"]["n_test"] == 5
    assert report["metrics"]["failures"] == 1
    bundle = joblib.load(output/"model.joblib")
    model = bundle["model"]
    semantic = attack in {"MSA", "SSA"}
    forests = ([model.domain_classifier] + [forest for heads in model.heads.values() for forest in heads.values()]
               if semantic else [model.classifier])
    assert len(forests) == (13 if semantic else 1)
    assert all(len(forest.estimators_) == 100 and forest.max_depth == 10 for forest in forests)
    encoded = (encode_metadata(observation(channel, 0)) if channel == "meta" else
               encode_hardware(observation(channel, 0), calibration))
    assert len(encoded.flatten()) == 512 * (111 if channel == "meta" else 33)
    assert model.predict(encoded.flatten()[None, :])[0] == predictions[0]
    if semantic:
        assert all(set(prediction) == {"domain", "intent", "attribute", "keyword"}
                   for prediction in predictions[:-1])
