"""Train and evaluate one channel using source-disjoint exported observations."""
import argparse
import hashlib
import json
from pathlib import Path
import importlib.metadata
import joblib
import numpy as np
from attacks.MAA import MAA
from attacks.MSA import MSA
from attacks.SAA import SAA
from attacks.SSA import SSA
from attacks.metadata import encode_metadata
from attacks.hardware import Calibration, LocalizationError, encode_hardware
from attacks.ablation import encode_ablation, ABLATIONS
from attacks.models import require_training_coverage
from evaluation import attribute_metrics, semantic_metrics
from .config import load_config
from .paper_protocol import validate_paper_config, validate_calibration, validate_dataset_protocol, validate_hardware_observation
from datasets.preprocessing.sources import source_ids


def load_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows: raise ValueError("Empty observation dataset")
    return rows


def validate_split(train, test, cross_template=False):
    for rows in (train, test):
        if not rows: raise ValueError("Training and test datasets must both be nonempty")
        if any(not isinstance(r.get("sample_id"), str) or not r["sample_id"] for r in rows):
            raise ValueError("Nonempty string sample_id is required")
        ids = [r["sample_id"] for r in rows]
        if len(set(ids)) != len(ids): raise ValueError("Duplicate sample_id")
        for r in rows:
            if not isinstance(r.get("source_id"), str) or not r["source_id"]:
                raise ValueError("Nonempty string source_id is required for split isolation")
            if r.get("provenance") not in {"measured", "synthetic"}:
                raise ValueError("Explicit measured/synthetic provenance is required")
    if set().union(*(source_ids(row) for row in train)) & set().union(*(source_ids(row) for row in test)):
        raise ValueError("Training/test source-record leakage")
    if {r["sample_id"] for r in train} & {r["sample_id"] for r in test}:
        raise ValueError("Training/test sample overlap")
    if cross_template:
        if any(not r.get("template_id") for r in train+test):
            raise ValueError("Cross-template evaluation requires template IDs")
        if {r["template_id"] for r in train} & {r["template_id"] for r in test}:
            raise ValueError("Cross-template test templates must be held out")


def run(config_path, train_path, test_path, output, attack, calibration_path=None,
        allow_synthetic=False, cross_template=False, defense_mode="none", ablation="full",
        allowed_sweep=None, paper_protocol=False):
    if attack not in {"MAA", "MSA", "SAA", "SSA"}: raise ValueError("Unknown attack")
    if defense_mode not in {"none", "aware", "oblivious"}: raise ValueError("Unknown defense mode")
    if ablation not in ABLATIONS:
        raise ValueError("Unknown ablation")
    if attack in {"MAA", "MSA"} and ablation in {"h2d_only", "page_only", "fusion"}:
        raise ValueError("Probe-source comparisons require the hardware channel")
    config = load_config(config_path)
    validate_paper_config(config, allowed_sweep=allowed_sweep)
    if Path(output).exists(): raise FileExistsError(output)
    train, test = load_rows(train_path), load_rows(test_path)
    validate_split(train, test, cross_template)
    if paper_protocol:
        validate_dataset_protocol(train, test, train_per_domain=config["data"]["train_per_domain"],
                                  test_per_domain=config["data"]["test_per_domain"],
                                  semantic=attack in {"MSA", "SSA"}, cross_template=cross_template,
                                  label_vocabularies=config["data"].get("label_vocabulary"),
                                  require_measured=not allow_synthetic)
    elif attack in {"MSA","SSA"} and config["data"].get("label_vocabulary") is not None:
        validate_dataset_protocol(train,test,train_per_domain=None,test_per_domain=None,
            semantic=True,cross_template=cross_template,require_measured=not allow_synthetic,
            label_vocabularies=config["data"]["label_vocabulary"])
    for row in train + test:
        if row.get("system", config["system"]) != config["system"]:
            raise ValueError("Observation system does not match configuration")
        if row.get("domain") not in config["data"]["domains"]:
            raise ValueError("Observation requires a configured domain label")
        if attack in {"MSA", "SSA"}:
            labels = row.get("labels", {})
            if any(not isinstance(labels.get(s), str) or not labels[s]
                   for s in ("domain", "intent", "attribute", "keyword")):
                raise ValueError("Semantic evaluation needs all four reference labels, including failed attempts")
            if labels["domain"] != row["domain"]:
                raise ValueError("Semantic domain label disagrees with row domain")
            if row.get("provenance") == "measured" or "annotation" in row:
                from datasets.preprocessing.grounding import validate_grounding_receipt
                validate_grounding_receipt(row)
    provenances = {r["provenance"] for r in train+test}
    if len(provenances) != 1: raise ValueError("Measured and synthetic observations cannot be mixed")
    if "synthetic" in provenances and not allow_synthetic:
        raise ValueError("Synthetic observations require --allow-synthetic")
    train_defenses = {r.get("defense", "none") for r in train}
    test_defenses = {r.get("defense", "none") for r in test}
    if len(train_defenses) != 1 or len(test_defenses) != 1:
        raise ValueError("One defense condition per training/test set required")
    if (train_defenses | test_defenses) - {"none", "fixed_budget", "dynamic_timing"}:
        raise ValueError("Unknown observation defense condition")
    if defense_mode == "aware" and (train_defenses != test_defenses or train_defenses == {"none"}):
        raise ValueError("Aware evaluation requires matching defended train and test observations")
    if defense_mode == "oblivious" and (train_defenses != {"none"} or test_defenses == {"none"}):
        raise ValueError("Oblivious evaluation requires undefended train and defended test")
    if defense_mode == "none" and (train_defenses != {"none"} or test_defenses != {"none"}):
        raise ValueError("Select aware/oblivious for defended observations")
    hardware = attack in {"SAA", "SSA"}
    semantic = attack in {"MSA", "SSA"}
    calibration = None
    if hardware:
        if calibration_path is None: raise ValueError("Hardware channel requires frozen calibration")
        calibration = Calibration(**json.loads(Path(calibration_path).read_text(encoding="utf-8")))
        validate_calibration(calibration.to_dict(), config)
    params = dict(window_ms=config["observation"]["window_ms"], max_windows=config["observation"]["max_windows"])
    def encode(row, base):
        if row.get("acquisition_status", "completed") != "completed" or row.get("request_status", "completed") != "completed":
            raise LocalizationError(row.get("failure_reason", "request or observation acquisition failed"))
        obs = row.get("observation")
        if obs is None:
            contents = (Path(base).parent/row["observation_path"]).read_bytes()
            obs = json.loads(contents.decode("utf-8"))
        else:
            contents = json.dumps(obs, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        if "observation_sha256" in row or semantic and row.get("provenance") == "measured":
            if hashlib.sha256(contents).hexdigest() != row.get("observation_sha256"):
                raise ValueError("Observation content differs from the trace bound to its labels")
        if row.get("provenance") == "measured" and not hardware and obs.get("request_id") != row["sample_id"]:
            raise ValueError("Metadata observation belongs to a different request")
        receipt = row.get("native_receipt")
        if receipt is None and row.get("native_receipt_path"):
            receipt = json.loads((Path(base).parent/row["native_receipt_path"]).read_text(encoding="utf-8"))
        if receipt is not None and receipt.get("request_id") != row["sample_id"]:
            raise ValueError("Native receipt belongs to a different labeled request")
        if hardware: validate_hardware_observation(obs, config)
        if ablation != "full":
            options = {} if hardware else dict(spatial_bins=config["implementation_choices"]["metadata_spatial_bins"])
            return encode_ablation(obs, "hw" if hardware else "meta", ablation, calibration, **params, **options)
        if hardware: return encode_hardware(obs, calibration=calibration, **params)
        return encode_metadata(obs, spatial_bins=config["implementation_choices"]["metadata_spatial_bins"], **params)
    def target(row): return row["labels"] if semantic else row["domain"]
    features, labels, failures, fitted_sample_ids = [], [], [], []
    names = None
    for row in train:
        try:
            encoded = encode(row, train_path)
        except LocalizationError:
            failures.append(row["sample_id"])
            continue
        names = encoded.feature_names
        features.append(encoded.flatten())
        labels.append(target(row))
        fitted_sample_ids.append(row["sample_id"])
    if not features: raise ValueError("No successfully localized training observations")
    declared_vocabulary=config["data"].get("label_vocabulary") if semantic else None
    require_training_coverage([target(row) for row in train], labels,
                              semantic=semantic, vocabulary=declared_vocabulary)
    model = {"MAA": MAA, "MSA": MSA, "SAA": SAA, "SSA": SSA}[attack](
        seed=config["implementation_choices"]["seed"], **config["classifier"])
    model.fit(np.stack(features), labels)
    predictions = []
    failures_test = []
    for row in test:
        try:
            encoded = encode(row, test_path)
            predictions.append(model.predict(encoded.flatten()[None, :])[0])
        except LocalizationError as exc:
            predictions.append(None)
            failures_test.append(dict(sample_id=row["sample_id"], reason=str(exc)))
    truth = [target(row) for row in test]
    metrics = semantic_metrics(truth, predictions) if semantic else attribute_metrics(truth, predictions)
    report = dict(attack=attack, system=config["system"], provenance=next(iter(provenances)),
                  defense_mode=defense_mode, ablation=ablation, allowed_sweep=allowed_sweep,
                  metrics=metrics, train_count=len(train), test_count=len(test),
                  fitted_train_count=len(fitted_sample_ids), fitted_sample_ids=fitted_sample_ids,
                  train_localization_failures=failures, test_localization_failures=failures_test,
                  features=list(names), config=config,
                  calibration=None if calibration is None else calibration.to_dict(),
                  input_sha256={"train": hashlib.sha256(Path(train_path).read_bytes()).hexdigest(),
                                "test": hashlib.sha256(Path(test_path).read_bytes()).hexdigest()},
                  versions={p: importlib.metadata.version(p) for p in ("numpy", "scikit-learn", "joblib")})
    if semantic:
        report["training_support"] = model.training_support_
        report["semantic_vocabulary_sizes"] = {domain: {slot: len(heads[slot].classes_)
            for slot in ("intent", "attribute", "keyword")} for domain, heads in model.heads.items()}
        fitted_ids = set(fitted_sample_ids)
        report["training_grounding"] = {row["sample_id"]: row["annotation"]["receipt_sha256"]
            for row in train if "annotation" in row and row["sample_id"] in fitted_ids}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out/"metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out/"predictions.jsonl").write_text("".join(json.dumps(dict(sample_id=r["sample_id"], prediction=p),
                                    ensure_ascii=False)+"\n" for r,p in zip(test,predictions)), encoding="utf-8")
    joblib.dump(dict(model=model, config=config, calibration=calibration, attack=attack, ablation=ablation), out/"model.joblib")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--attack", required=True, choices=["MAA", "MSA", "SAA", "SSA"])
    p.add_argument("--calibration")
    p.add_argument("--allow-synthetic", action="store_true")
    p.add_argument("--cross-template", action="store_true")
    p.add_argument("--defense-mode", choices=["none", "aware", "oblivious"], default="none")
    p.add_argument("--ablation", choices=ABLATIONS, default="full")
    p.add_argument("--sweep", choices=["block_tokens", "residency_fraction", "context_tokens", "training_per_domain", "concurrent_workloads"])
    p.add_argument("--paper-protocol", action="store_true", help="Require the full paper dataset counts and semantic vocabulary")
    a = p.parse_args()
    report = run(a.config, a.train, a.test, a.output, a.attack, a.calibration,
                 a.allow_synthetic, a.cross_template, a.defense_mode, a.ablation, a.sweep, a.paper_protocol)
    print(json.dumps(dict(provenance=report["provenance"], metrics=report["metrics"]), indent=2))


if __name__ == "__main__": main()
