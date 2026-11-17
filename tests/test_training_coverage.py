"""Real encoders and fitted forests retain every required semantic target."""
import importlib
import json
from pathlib import Path

import joblib
import pytest
import yaml

from attacks.MSA import MSA
from attacks.SSA import SSA
from attacks.hardware import Calibration
from datasets.preprocessing.annotate import load_taxonomy

ROOT = Path(__file__).resolve().parents[1]
run = importlib.import_module("experiments.run").run


def metadata():
    return dict(request_id="fixture", start_ms=0, end_ms=30,
                events=[dict(op="LOAD", layer=0, block_id=0, time_ms=1)])


def hardware(*, failed=False):
    return dict(submission_ms=0, capacity_slots=1,
                h2d=[dict(time_ms=i, latency_us=250 if i < 4 and not failed else 150) for i in range(40)],
                compute=[], page=[])


@pytest.mark.parametrize("model_type", [MSA, SSA])
def test_direct_observation_api_cannot_erase_a_semantic_class(model_type):
    labels = [dict(domain="medical", intent="explain", attribute="a", keyword="k"),
              dict(domain="medical", intent="explain", attribute="rare", keyword="rare")]
    observations = [metadata(), dict(acquisition_status="failed")] if model_type is MSA else [hardware(), hardware(failed=True)]
    with pytest.raises(ValueError, match="Successfully localized training"):
        model_type().fit_observations(observations, labels,
            calibration=Calibration(150, None, None) if model_type is SSA else None)


@pytest.mark.parametrize("attack", ["MSA", "SSA"])
def test_all_configured_classes_exist_in_real_fitted_heads(tmp_path, attack):
    # Numerical fixtures exercise data-to-RF wiring, not corpus availability.
    taxonomy, _, _ = load_taxonomy()
    vocabulary = {domain: {slot: [entry["label"] for entry in spec[slot]]
                           for slot in ("intent", "attribute", "keyword")}
                  for domain, spec in taxonomy["domains"].items()}
    config = yaml.safe_load((ROOT/"configs"/"quest.yaml").read_text(encoding="utf-8"))
    config["data"]["label_vocabulary"] = vocabulary
    config["implementation_choices"]["metadata_spatial_bins"] = 2
    (tmp_path/"config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    paths = {}
    for split, repeats in (("train", 2), ("test", 1)):
        rows = []
        for domain, spec in taxonomy["domains"].items():
            for index, keyword in enumerate(spec["keyword"]):
                for repeat in range(repeats):
                    identity = f"{split}-{domain}-{index}-{repeat}"
                    rows.append(dict(sample_id=identity, source_id=identity, domain=domain,
                        provenance="synthetic", labels=dict(domain=domain,
                            intent=spec["intent"][index%3]["label"], attribute=keyword["attribute"], keyword=keyword["label"]),
                        observation=metadata() if attack == "MSA" else hardware()))
        paths[split] = tmp_path/f"{split}.jsonl"
        paths[split].write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    calibration = tmp_path/"calibration.json"
    calibration.write_text(json.dumps(Calibration(150, None, None).to_dict()), encoding="utf-8")
    report = run(tmp_path/"config.yaml", paths["train"], paths["test"], tmp_path/"trained", attack,
                 calibration if attack == "SSA" else None, allow_synthetic=True)
    model = joblib.load(tmp_path/"trained"/"model.joblib")["model"]
    assert report["fitted_train_count"] == len(report["fitted_sample_ids"]) == 480
    assert set(model.domain_classifier.classes_) == set(vocabulary)
    for domain, slots in vocabulary.items():
        assert report["semantic_vocabulary_sizes"][domain] == dict(intent=3, attribute=60, keyword=60)
        for slot, declared in slots.items():
            assert set(model.heads[domain][slot].classes_) == set(declared)
            support = report["training_support"]["by_domain"][domain][slot]
            assert set(support) == set(declared) and min(support.values()) >= 2
            assert sum(support.values()) == 120
