"""Execute Chapter 5 inference comparisons, retraining each requested condition."""
from collections import Counter, defaultdict
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from attacks.ablation import encode_ablation
from attacks.hardware import Calibration, LocalizationError
from evaluation import attribute_metrics
from experiments.config import load_config
from experiments.run import run as run_inference, validate_split
from experiments.paper_protocol import validate_paper_config, validate_dataset_protocol, validate_calibration, validate_hardware_observation
from .common import (DOMAINS, read_json, read_rows, resolve, observation, require_provenance,
                     materialize_paths, nested_subset, write_rows, write_json, fingerprint,
                     validate_collection, finite_positive)


def validate_condition(experiment, run, config):
    setting = run.get("setting")
    if setting not in config["data"]["settings"]: raise ValueError("Invalid prompt setting")
    attack = run["attack"]
    allowed = ["MAA","SAA"] if experiment == "e03_attribute_inference" else ["MAA","MSA","SAA","SSA"]
    if experiment == "e04_semantic_recovery": allowed = ["MSA","SSA"]
    if experiment == "e07_transfer_contribution": allowed = ["SAA"]
    if experiment == "e09_cache_sensitivity": allowed = ["MAA"]
    if attack not in allowed: raise ValueError("Attack is not part of this experiment")
    if experiment in {"e04_semantic_recovery", "e07_transfer_contribution"} and config["system"] != "quest":
        raise ValueError("This experiment uses Quest-Llama")
    if experiment == "e05_end_to_end" and attack in {"MSA","SSA"} and (config["system"] != "quest" or setting != "templated_short"):
        raise ValueError("Main-study semantic end-to-end evaluation uses Quest with short templates")
    if experiment in {"e04_semantic_recovery","e07_transfer_contribution","e08_event_ablation","e10_concurrency"} and setting != "templated_short":
        raise ValueError("This experiment uses short templated prompts")
    ablation = run.get("ablation", "full")
    if experiment == "e07_transfer_contribution" and ablation not in {"h2d_only","page_only","fusion"}:
        raise ValueError("Transfer comparison requires h2d_only, page_only or fusion")
    if experiment == "e08_event_ablation" and ablation not in {"full","remove_hit","remove_load","remove_evict","remove_all"}:
        raise ValueError("Invalid event ablation")
    if experiment not in {"e07_transfer_contribution","e08_event_ablation"} and ablation != "full":
        raise ValueError("This experiment uses the full representation")
    if experiment == "e12_profiling_scale":
        if setting not in {"templated_short","templated_long"}: raise ValueError("Profiling scale uses TS and TL")
        if run.get("train_per_domain") not in [400,800,1600,3200]: raise ValueError("Invalid profiling size")
    if experiment == "e10_concurrency" and run.get("concurrent_workloads") not in [1,4,8,16]:
        raise ValueError("Concurrency must be 1, 4, 8 or 16")
    allowed_sweep = "training_per_domain" if experiment == "e12_profiling_scale" else (
        "concurrent_workloads" if experiment == "e10_concurrency" else None)
    if experiment == "e09_cache_sensitivity":
        factor, value = run.get("factor"), run.get("value")
        values = {"residency_fraction":[.2,.4,.6], "block_tokens":[8,16,32], "context_tokens":[1000,2000,3000,4000]}
        if factor not in values or value not in values[factor]: raise ValueError("Invalid sensitivity setting")
        if setting not in {"templated_short", "templated_long"} or (factor != "context_tokens" and setting != "templated_short"):
            raise ValueError("Cache sensitivity uses templated inputs and short inputs for budget/block sweeps")
        if factor in config["cache"] and config["cache"][factor] != value:
            raise ValueError("Sensitivity condition is not reflected in the configuration")
        if run["collection"].get(factor) != value: raise ValueError("Collection receipt does not match sensitivity value")
        allowed_sweep = factor
    validate_paper_config(config, allowed_sweep=allowed_sweep)
    return allowed_sweep


def validate_schedule(path, concurrency, target_ids):
    rows = read_rows(path)
    if not rows: raise ValueError("Concurrency requires a recorded asynchronous schedule")
    target_ids = set(target_ids)
    by_id = {}
    native = any("native_receipt" in row for row in rows)
    absolute=["submission_monotonic_ms" in row and "completion_monotonic_ms" in row for row in rows]
    if any(absolute) and not all(absolute): raise ValueError("Schedule rows must use a consistent clock")
    use_absolute=all(absolute)
    normalized=[]
    failed_targets=[]
    for row in rows:
        if row["sample_id"] in by_id: raise ValueError("Duplicate scheduled request ID")
        by_id[row["sample_id"]] = row
        if native:
            receipt = row.get("native_receipt")
            if row.get("status") != "completed":
                if row["sample_id"] in target_ids:
                    failed_targets.append(row["sample_id"])
                    continue
                raise ValueError("Failed native background requests cannot establish model concurrency")
            if not isinstance(receipt,dict) or not all(k in receipt for k in
                    ("execution_start_monotonic_ms","execution_end_monotonic_ms")):
                raise ValueError("Native concurrency requires measured model execution intervals")
        start_key,end_key=("submission_monotonic_ms","completion_monotonic_ms") if use_absolute else ("submission_ms","completion_ms")
        if native:
            start = float(row["native_receipt"]["execution_start_monotonic_ms"])
            end = float(row["native_receipt"]["execution_end_monotonic_ms"])
        else:
            start, end = float(row[start_key]), float(row[end_key])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("Invalid measured request interval")
        normalized.append((start,end,row))
    if not target_ids <= set(by_id): raise ValueError("Every target attempt must have a schedule record")
    events=sorted((stamp,delta) for start,end,_ in normalized for stamp,delta in ((start,1),(end,-1)))
    active = peak = 0
    full_segments=[]
    previous=None
    for stamp,delta in events:
        if previous is not None and stamp > previous and active == concurrency:
            full_segments.append((previous,stamp))
        active += delta
        peak = max(peak,active)
        previous=stamp
    if peak > concurrency: raise ValueError("Measured schedule exceeds declared concurrency")
    if concurrency > 1:
        # A global maximum can come from a different capture session. Require
        # the requested overlap while each successful target actually executes.
        full_ends=[end for _,end in full_segments]
        for target_start,target_end,target in normalized:
            if target["sample_id"] not in target_ids: continue
            index=bisect_right(full_ends,target_start)
            if index == len(full_segments) or full_segments[index][0] >= target_end:
                raise ValueError("A target execution never attains requested concurrency")
        intervals=sorted((start,end,float(row["planned_offset_ms"])) for start,end,row in normalized)
        component_offsets=set()
        component_end=-math.inf
        for start,end,offset in intervals:
            if start >= component_end: component_offsets=set()
            if not math.isfinite(offset) or offset < 0 or offset in component_offsets:
                raise ValueError("Asynchronous workloads require distinct finite planned offsets within each overlapping session")
            component_offsets.add(offset)
            component_end=max(component_end,end)
        if any(r.get("concurrency",concurrency) != concurrency for r in rows):
            raise ValueError("Schedule concurrency disagrees with the experiment condition")
    return dict(requests=len(rows), peak_active=peak,
                failed_target_requests=len(failed_targets),
                clock="native_execution_monotonic" if native else ("monotonic_absolute" if use_absolute else "common_epoch_relative"),
                total_observed_ms=max((end for _,end,_ in normalized),default=0)-min((start for start,_,_ in normalized),default=0))


def execute_inference(experiment, manifest, base, output, pilot=False, allow_synthetic=False):
    records, fingerprints, sensitivity_budgets, pool_fingerprints = [], {}, {}, {}
    for item in manifest["runs"]:
        run = dict(item)
        run_id = run["id"]
        config_path = resolve(base, run["config"])
        config = load_config(config_path)
        allowed_sweep = validate_condition(experiment, run, config)
        train_path, test_path = resolve(base,run["train"]), resolve(base,run["test"])
        train, test = read_rows(train_path), read_rows(test_path)
        require_provenance(train+test, allow_synthetic)
        validate_split(train,test,run["setting"] == "cross_template")
        condition = validate_collection(run,config,train+test)
        if experiment == "e09_cache_sensitivity" and run["factor"] == "block_tokens":
            budget = condition.get("selected_token_budget")
            if not isinstance(budget,int) or isinstance(budget,bool) or budget <= 0:
                raise ValueError("Block-size sweep needs a positive selected-token budget")
            if config["cache"].get("selected_token_budget") != budget:
                raise ValueError("Selected-token budget must agree between configuration and collection")
            previous = sensitivity_budgets.setdefault(config["system"],budget)
            if previous != budget: raise ValueError("Selected-token budget must remain fixed across block sizes")
        semantic = run["attack"] in {"MSA","SSA"}
        count = run.get("train_per_domain", 3000) if experiment == "e12_profiling_scale" else 3000
        if experiment == "e12_profiling_scale":
            channel = "meta" if run["attack"] in {"MAA", "MSA"} else "hw"
            pool_key=(config["system"],run["setting"],"semantic" if semantic else "domain",channel)
            pool_value=(fingerprint(train),config["implementation_choices"]["seed"])
            if pool_fingerprints.setdefault(pool_key,pool_value) != pool_value:
                raise ValueError("Profiling sizes must use the same expanded auxiliary pool and seed")
            train = nested_subset(train,count,config["implementation_choices"]["seed"],semantic)
            selected_path = output/(run_id+"-selected-train.jsonl")
            write_rows(selected_path, materialize_paths(train,train_path))
            train_path = selected_path
            key = (config["system"],run["setting"],"semantic" if semantic else "domain",channel)
            current = fingerprint(test)
            if fingerprints.setdefault(key,current) != current: raise ValueError("Profiling sizes must share a fixed test set")
        validate_dataset_protocol(train,test,train_per_domain=None if pilot else count,
            test_per_domain=None if pilot or experiment == "e07_transfer_contribution" else 1000,
            test_total=None if pilot or experiment != "e07_transfer_contribution" else 1000,
            semantic=semantic and (not pilot or config["data"].get("label_vocabulary") is not None),
            cross_template=run["setting"] == "cross_template",
            label_vocabularies=config["data"].get("label_vocabulary"),
            require_measured=not allow_synthetic)
        group = (config["system"], run["setting"], run["attack"])
        if experiment in {"e07_transfer_contribution","e08_event_ablation"}:
            pair = (fingerprint(train),fingerprint(test))
            if fingerprints.setdefault(group,pair) != pair: raise ValueError("Ablations must use identical train/test records")
        schedule = None
        if experiment == "e10_concurrency":
            if condition.get("concurrent_workloads") != run["concurrent_workloads"]:
                raise ValueError("Collection concurrency differs from evaluated concurrency")
            schedule = validate_schedule(resolve(base,run["schedule"]),run["concurrent_workloads"],
                                         [r["sample_id"] for r in train+test])
        calibration = resolve(base,run["calibration"]) if run.get("calibration") else None
        if calibration and experiment in {"e07_transfer_contribution", "e08_event_ablation"}:
            calibration_key=("calibration",)+group
            calibrated_hash=hashlib.sha256(calibration.read_bytes()).hexdigest()
            if fingerprints.setdefault(calibration_key,calibrated_hash) != calibrated_hash:
                raise ValueError("Feature comparisons must use the same frozen calibration")
        result = run_inference(config_path,train_path,test_path,output/run_id,run["attack"],
            calibration_path=calibration,allow_synthetic=allow_synthetic,
            cross_template=run["setting"] == "cross_template",ablation=run.get("ablation","full"),
            allowed_sweep=allowed_sweep)
        record = dict(id=run_id,system=config["system"],attack=run["attack"],setting=run["setting"],
                      ablation=run.get("ablation","full"),metrics=result["metrics"],
                      train_count=len(train),test_count=len(test),collection=condition)
        if semantic: record["semantic_vocabulary_sizes"]=result["semantic_vocabulary_sizes"]
        for key in ("factor","value","concurrent_workloads","train_per_domain"):
            if key in run: record[key] = run[key]
        if schedule: record["schedule"] = schedule
        if experiment == "e05_end_to_end":
            record["localization"] = dict(attempts=len(test),failed=len(result["test_localization_failures"]),
                success_percent=100*(len(test)-len(result["test_localization_failures"]))/len(test),
                retained_horizon_ms=config["observation"]["window_ms"]*config["observation"]["max_windows"])
        if experiment == "e04_semantic_recovery":
            predictions=read_rows(output/run_id/"predictions.jsonl")
            descriptions=[]
            for row in predictions:
                p=row["prediction"]
                text=None if p is None else (
                    f"The inferred request concerns {p['domain']}. Intent: {p['intent']}; "
                    f"sensitive attribute: {p['attribute']}; keyword: {p['keyword']}.")
                descriptions.append(dict(sample_id=row["sample_id"],predicted_labels=p,description=text))
            write_rows(output/run_id/"semantic_descriptions.jsonl",descriptions)
        records.append(record)
    return records


def execute_intent(manifest, base, output, pilot=False, allow_synthetic=False):
    records=[]
    for run in manifest["runs"]:
        config=load_config(resolve(base,run["config"]))
        validate_paper_config(config)
        if config["system"] != "quest": raise ValueError("Within-domain intent experiment uses Quest-Llama")
        domain=run["domain"]
        if domain not in DOMAINS: raise ValueError("Unknown domain")
        positive,negative=run["positive_label"],run["negative_label"]
        if not isinstance(positive,str) or not isinstance(negative,str) or not positive or not negative or positive == negative:
            raise ValueError("Choose two distinct intent labels")
        paths=[resolve(base,run[s]) for s in ("train","test")]
        subsets=[]
        for path in paths:
            rows=read_rows(path)
            require_provenance(rows,allow_synthetic)
            if any(r["domain"] != domain for r in rows): raise ValueError("A within-domain run must hold corpus/domain fixed")
            subsets.append([r for r in rows if r["labels"]["intent"] in {positive,negative}])
        train,test=subsets
        validate_split(train,test)
        if {r["labels"]["intent"] for r in train} != {positive,negative}: raise ValueError("Both training intent classes are required")
        if {r["labels"]["intent"] for r in test} != {positive,negative}: raise ValueError("Both test intent classes are required")
        validate_collection(run,config,train+test)
        channel=run["channel"]
        if channel not in {"meta","hw"}: raise ValueError("Channel must be meta or hw")
        cal=Calibration(**read_json(resolve(base,run["calibration"]))) if channel == "hw" else None
        if cal is not None: validate_calibration(cal.to_dict(),config)
        def encode(row,path):
            options=dict(window_ms=10,max_windows=512)
            if channel == "meta": options["spatial_bins"]=config["implementation_choices"]["metadata_spatial_bins"]
            obs = observation(row,path)
            if channel == "hw": validate_hardware_observation(obs,config)
            return encode_ablation(obs,channel,calibration=cal,**options).flatten()
        X,y=[],[]
        for row in train:
            if row.get("request_status", "completed") != "completed" or row.get("acquisition_status", "completed") != "completed": continue
            try: vector=encode(row,paths[0])
            except LocalizationError: continue
            X.append(vector); y.append(row["labels"]["intent"])
        if set(y) != {positive,negative}: raise ValueError("Localized training data must cover both intent classes")
        model=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=config["implementation_choices"]["seed"],n_jobs=1)
        model.fit(np.stack(X),y)
        predictions=[]
        for row in test:
            if row.get("request_status", "completed") != "completed" or row.get("acquisition_status", "completed") != "completed":
                predictions.append(None)
                continue
            try: predictions.append(str(model.predict(encode(row,paths[1])[None,:])[0]))
            except LocalizationError: predictions.append(None)
        truth=[r["labels"]["intent"] for r in test]
        metrics=attribute_metrics(truth,predictions,labels=[negative,positive])
        positive_indices=[i for i,t in enumerate(truth) if t == positive]
        metrics["positive_class_accuracy_percent"]=100*sum(predictions[i] == positive for i in positive_indices)/len(positive_indices)
        directory=output/run["id"];directory.mkdir()
        joblib.dump(dict(model=model,channel=channel,calibration=cal,positive_label=positive,negative_label=negative),directory/"model.joblib")
        write_rows(directory/"predictions.jsonl",[dict(sample_id=r["sample_id"],prediction=p) for r,p in zip(test,predictions)])
        records.append(dict(id=run["id"],system="quest",domain=domain,channel=channel,metrics=metrics))
    return records
