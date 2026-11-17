"""Collect real native-backend executions before dispatching Chapter 5 evaluation."""
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import yaml
from collections import defaultdict

from .common import read_json, read_rows, write_json, write_rows, resolve
from .specification import canonical_spec


def conditions(experiment, project):
    """Enumerate the actual paper conditions; feature ablations share captures."""
    spec = canonical_spec(experiment)
    if experiment in {"e01_metadata_recovery", "e02_hardware_recovery"}:
        return [dict(system=s, channel="hw" if experiment.startswith("e02") else "meta",
                     setting="auxiliary") for s in spec["systems"]]
    if experiment == "e11_profiling_cost":
        return [dict(system=s, channel=c, setting="templated_short")
                for s,c in itertools.product(spec["systems"],spec["channels"])]
    if experiment == "e06_within_domain_intent":
        pairs = project.get("intent_pairs", {})
        result = []
        for domain in spec["domains"]:
            pair = pairs.get(domain, {})
            if not pair.get("positive") or not pair.get("negative") or pair["positive"] == pair["negative"]:
                raise ValueError(f"Declare two distinct intent_pairs labels for {domain}")
            channel = pair.get("channel", "meta")
            if channel not in {"meta", "hw"}:
                raise ValueError("Intent channel must be meta or hw")
            result.append(dict(system="quest", setting="templated_short", channel=channel,
                               domain=domain, positive_label=pair["positive"], negative_label=pair["negative"]))
        return result
    if experiment == "e05_end_to_end":
        result = [dict(system=s, attack=a, setting=t) for group in spec["condition_groups"]
                  for s,a,t in itertools.product(group["systems"],group["attacks"],group["settings"])]
    elif experiment == "e09_cache_sensitivity":
        result = [dict(system=s, attack="MAA", factor=f, value=v,
                       setting="templated_long" if f == "context_tokens" and v >= 3000 else "templated_short")
                  for s in spec["systems"] for f,values in spec["factors"].items() for v in values]
    else:
        result = [dict(system=s, attack=a, setting=t) for s,a,t in
                  itertools.product(spec["systems"],spec["attacks"],spec["settings"])]
        factor = {"e07_transfer_contribution": "ablations", "e08_event_ablation": "ablations",
                  "e10_concurrency": "concurrent_workloads", "e12_profiling_scale": "train_per_domain"}.get(experiment)
        if factor:
            name = "ablation" if factor == "ablations" else factor
            result = [dict(row, **{name:value}) for row in result for value in spec[factor]]
    return [dict(row, channel="meta" if row["attack"] in {"MAA", "MSA"} else "hw") for row in result]


def _paths(project, condition, experiment):
    system, setting = condition["system"], condition["setting"]
    semantic=condition.get("attack") in {"MSA","SSA"} or experiment == "e06_within_domain_intent"
    datasets=project.get("semantic_datasets",project.get("datasets",{})) if semantic else project.get("datasets",{})
    if setting == "auxiliary":
        value = project.get("auxiliary", {}).get(system)
        if not value:
            raise ValueError(f"Declare auxiliary.{system} prompt JSONL")
        return dict(pairs=value)
    if condition.get("factor") == "context_tokens":
        value = project.get("contexts", {}).get(system, {}).get(str(condition["value"]))
        if not value:
            raise ValueError(f"Declare actual context-{condition['value']} prompts for {system}")
    elif experiment == "e12_profiling_scale":
        profiling=project.get("semantic_profiling",project.get("profiling",{})) if semantic else project.get("profiling",{})
        value = profiling.get(system, {}).get(setting)
        if value is None:
            value = datasets.get(system, {}).get(setting)
    else:
        value = datasets.get(system, {}).get(setting)
    if not isinstance(value, dict) or not all(value.get(k) for k in ("train", "test")):
        raise ValueError(f"Declare datasets.{system}.{setting}.train and .test")
    if experiment == "e11_profiling_cost":
        return dict(timings=value["train"])
    if experiment == "e07_transfer_contribution" and project.get("transfer_test", {}).get(system):
        return dict(train=value["train"],test=project["transfer_test"][system])
    return {key:value[key] for key in ("train", "test")}


def _prompts(path, *, ground_semantic=False):
    result, ids = [], set()
    for row in read_rows(path):
        row = dict(row)
        prompt = row.get("prompt", row.get("text"))
        identity = row.get("sample_id", row.get("id"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Rendered prompt/text is required in {path}")
        prompt_digest=hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if row.get("prompt_sha256",prompt_digest) != prompt_digest:
            raise ValueError("Declared prompt_sha256 differs from the actual rendered input")
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError(f"Unique sample_id/id strings are required in {path}")
        if not isinstance(row.get("source_id"), str) or not row["source_id"]:
            raise ValueError("Native dataset rows must preserve source_id")
        if row.get("domain") not in {"medical", "financial", "legal", "code"}:
            raise ValueError("Native dataset rows need a paper domain label")
        labels = dict(row.get("labels", {}))
        if "domain" in labels and labels["domain"] != row["domain"]:
            raise ValueError("Top-level and nested semantic domains disagree")
        labels["domain"] = row["domain"]
        for slot in ("intent", "attribute", "keyword"):
            if slot in row:
                if slot in labels and labels[slot] != row[slot]:
                    raise ValueError("Top-level and nested semantic labels disagree")
                labels[slot] = row[slot]
        row.update(sample_id=identity, prompt=prompt, labels=labels, prompt_sha256=prompt_digest)
        complete_labels = all(isinstance(labels.get(slot), str) and labels[slot]
                              for slot in ("intent", "attribute", "keyword"))
        if ground_semantic and not complete_labels:
            raise ValueError("Supply curated semantic labels before native collection")
        if ground_semantic or complete_labels and ("text" in row or "annotation" in row):
            from datasets.preprocessing.grounding import validate_grounded_record
            row["annotation"] = validate_grounded_record(row, prompt=prompt)
        ids.add(identity)
        result.append(row)
    if not result:
        raise ValueError(f"Empty prompt dataset: {path}")
    return result


def _configuration(project, backend, base, condition):
    from experiments.config import load_config
    config = deepcopy(load_config(resolve(base, backend["config"])))
    if config["system"] != condition["system"]:
        raise ValueError("Native backend and experiment system differ")
    if "label_vocabulary" in project:
        config["data"]["label_vocabulary"]=deepcopy(project["label_vocabulary"])
    for group, key in (("inference", "generation_tokens"), ("cache", "selected_token_budget")):
        value = backend.get(key, project.get(key, config[group].get(key)))
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Declare a positive {key}; the paper does not fix this run setting")
        config[group][key] = value
    config["cache"]["capacity_slots"] = None  # Actual per-request allocation comes from describe().
    factor = condition.get("factor")
    if factor in {"block_tokens", "residency_fraction"}:
        config["cache"][factor] = condition["value"]
    if config["inference"]["generation_tokens"] < 2:
        raise ValueError("Native Chapter 5 generation_tokens must be >=2 to exercise decode KV management")
    if condition["system"] == "quest":
        block=config["cache"]["block_tokens"]
        selected=config["cache"]["selected_token_budget"]
        if selected < 2*block or selected % block:
            raise ValueError("Quest selected_token_budget must cover two blocks and be divisible by block_tokens")
    if factor == "context_tokens":
        config.setdefault("experiment", {})[factor] = condition["value"]
    concurrency = condition.get("concurrent_workloads", 1)
    if "concurrent_workloads" in condition:
        config.setdefault("experiment", {})["concurrent_workloads"] = concurrency
    from experiments.paper_protocol import validate_paper_config
    validate_paper_config(config, allowed_sweep=_sweep(condition))
    return config


def _sweep(condition):
    return condition.get("factor") or ("concurrent_workloads" if "concurrent_workloads" in condition else None)


def _capture_key(condition, paths, config):
    payload = dict(system=condition["system"], channel=condition["channel"], setting=condition["setting"],
                   paths={k:str(Path(v).resolve()) for k,v in paths.items()},
                   input_digests={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in paths.items()},
                   cache=config["cache"], inference=config["inference"], model=config["model"], backend=config["backend"],
                   defense=config.get("experiment",{}).get("defense") or "none",
                   defense_parameters=config["defenses"],
                   factor=condition.get("factor"), value=condition.get("value"),
                   concurrent_workloads=condition.get("concurrent_workloads",1))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def _preflight_inputs(paths, condition, experiment, pilot, label_vocabulary=None):
    from collections import Counter
    from experiments.paper_protocol import DOMAINS, validate_dataset_protocol
    semantic = condition.get("attack") in {"MSA", "SSA"} or experiment == "e06_within_domain_intent"
    inputs = {key:_prompts(path, ground_semantic=semantic) for key,path in paths.items()}
    if "train" in inputs:
        validate_dataset_protocol(inputs["train"], inputs["test"],
            train_per_domain=None if pilot or experiment == "e06_within_domain_intent" else (3200 if experiment == "e12_profiling_scale" else 3000),
            test_per_domain=None if pilot or experiment in {"e06_within_domain_intent","e07_transfer_contribution"} else 1000,
            test_total=1000 if experiment == "e07_transfer_contribution" and not pilot else None,
            semantic=condition.get("attack") in {"MSA","SSA"} and (not pilot or label_vocabulary is not None),
            label_vocabularies=label_vocabulary,
            cross_template=condition["setting"] == "cross_template", require_measured=False)
        if condition.get("attack") in {"MSA","SSA"} or experiment == "e06_within_domain_intent":
            slots = ("intent",) if experiment == "e06_within_domain_intent" else ("intent","attribute","keyword")
            for row in inputs["train"]+inputs["test"]:
                if any(not isinstance(row["labels"].get(s),str) or not row["labels"][s] for s in slots):
                    raise ValueError("Supply curated semantic labels before native collection")
        if experiment == "e06_within_domain_intent":
            required={condition["positive_label"],condition["negative_label"]}
            for rows in inputs.values():
                labels={r["labels"]["intent"] for r in rows if r["domain"] == condition["domain"]}
                if not required <= labels:
                    raise ValueError("Both declared intent labels must occur in each partition")
    if "pairs" in inputs:
        sources = [r["source_id"] for r in inputs["pairs"]]
        if len(sources) != len(set(sources)):
            raise ValueError("Recovery requires one prompt per unique auxiliary source record")
        if not pilot and len(sources) != 1000:
            raise ValueError("Recovery requires 1,000 unique auxiliary source records")
    if "timings" in inputs and not pilot:
        counts = Counter(r["domain"] for r in inputs["timings"])
        if set(counts) != set(DOMAINS) or any(counts[d] != 3000 for d in DOMAINS):
            raise ValueError("Profiling cost requires 3,000 auxiliary prompts per domain")
    return {r["source_id"] for r in inputs.get("pairs",[])}


def _calibrate(config_path, config, service, prompt, directory, backend, project, sweep):
    """Three independent idle probe processes; thresholds freeze before collection."""
    from experiments.acquisition.probe import calibrate
    description = service.describe(prompt, config["inference"]["generation_tokens"])
    baselines = []
    for index in range(3):
        path = directory/f"baseline-{index}.json"
        command = [str(project.get("probe_python", backend.get("python", sys.executable))),
                   "-m", "experiments.acquisition.probe", "capture", "--config", str(config_path),
                   "--seconds", str(project.get("baseline_seconds", 8)),
                   "--device", backend.get("device", "cuda:0"),
                   "--capacity-slots", str(description["capacity_slots"]), "--output", str(path)]
        if sweep:
            command += ["--sweep", sweep]
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[2],
                       timeout=float(project.get("probe_start_timeout_s", 180))+float(project.get("baseline_seconds",8))+30)
        baselines.append(read_json(path))
    result = directory/"calibration.json"
    write_json(result, calibrate(baselines, config["observation"]))
    return result


def _context_target(project, condition, inputs):
    """Resolve the declared input length, separating cross-template artifact choices."""
    setting = condition["setting"]
    if condition.get("factor") == "context_tokens":
        target, origin = condition["value"], "paper_context_sweep"
    elif setting.endswith("_long"):
        target, origin = 3000, "paper_long_context"
    elif setting.endswith("_short"):
        target, origin = 1000, "paper_short_context"
    elif setting == "cross_template":
        if "target_tokens" in condition:
            target, origin = condition["target_tokens"], "declared_condition"
        elif "cross_template_target_tokens" in project:
            target, origin = project["cross_template_target_tokens"], "declared_project"
        else:
            declared = [row["target_tokens"] for rows in inputs.values() for row in rows
                        if "target_tokens" in row]
            target, origin = (declared[0], "prepared_rows") if declared else (1000, "artifact_default")
    else:
        return None, None
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError("The context target must be a positive integer token count")
    for rows in inputs.values():
        for row in rows:
            if "target_tokens" in row and (isinstance(row["target_tokens"], bool)
                    or not isinstance(row["target_tokens"], int) or row["target_tokens"] != target):
                raise ValueError("Prepared prompt target_tokens differs from the declared context condition")
    return target, origin


def _validate_context_lengths(project, condition, inputs, config, service):
    """Measure every rendered input with the runtime tokenizer, per split and domain."""
    target, origin = _context_target(project, condition, inputs)
    if target is None:
        return None
    tolerance = project.get("context_mean_tolerance_fraction", .1)
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance) or not 0 <= tolerance < 1):
        raise ValueError("context_mean_tolerance_fraction must lie in [0,1)")
    summaries = {}
    for split, rows in inputs.items():
        counts = defaultdict(list)
        for row in rows:
            domain = row.get("domain")
            if domain not in {"medical", "financial", "legal", "code"}:
                raise ValueError("Tokenizer context validation requires a paper domain")
            measured = service.describe(row["prompt"], config["inference"]["generation_tokens"])["token_count"]
            if isinstance(measured, bool) or not isinstance(measured, int) or measured < 1:
                raise ValueError("Native tokenizer must return a positive integer token_count")
            if "token_count" in row and (isinstance(row["token_count"], bool)
                    or not isinstance(row["token_count"], int) or row["token_count"] != measured):
                raise ValueError(f"Prepared {split}/{domain} token_count differs from the native rendered prompt")
            counts[domain].append(measured)
        if not counts:
            raise ValueError(f"Empty {split} prompt dataset")
        summaries[split] = {}
        for domain, values in counts.items():
            mean = sum(values) / len(values)
            if abs(mean-target) > tolerance*target:
                raise ValueError(f"Actual {split}/{domain} tokenizer mean {mean:.1f} "
                                 f"does not match context condition {target}")
            summaries[split][domain] = dict(count=len(values), mean_tokens=mean,
                                          minimum_tokens=min(values), maximum_tokens=max(values))
    return dict(target_tokens=target, target_source=origin, relative_tolerance=tolerance,
                tokenizer=config["model"], condition=condition["setting"], splits=summaries,
                token_count_scope="rendered_prompt_with_tokenizer_special_tokens")


def _collect_condition(project, base, condition, paths, config, directory, *, pilot=False):
    from backends.service import NativeService
    from experiments.collection import collect
    backend = project["backends"][condition["system"]]
    directory.mkdir(parents=True, exist_ok=False)
    config_path = directory/"config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    semantic = condition.get("attack") in {"MSA", "SSA"} or "positive_label" in condition
    inputs = {split:_prompts(path, ground_semantic=semantic) for split,path in paths.items()}
    if "train" in inputs and "test" in inputs:
        from experiments.paper_protocol import validate_dataset_protocol
        validate_dataset_protocol(inputs["train"],inputs["test"], train_per_domain=None,
            test_per_domain=None,cross_template=condition["setting"] == "cross_template",require_measured=False)
    training = inputs.get("train", next(iter(inputs.values())))
    sweep = _sweep(condition)
    result = dict(config=str(config_path), rows={}, condition=dict(system=condition["system"],
        setting=condition["setting"], block_tokens=config["cache"]["block_tokens"],
        residency_fraction=config["cache"]["residency_fraction"],
        selected_token_budget=config["cache"]["selected_token_budget"],
        defense=config.get("experiment",{}).get("defense") or "none",
        concurrent_workloads=condition.get("concurrent_workloads",1)))
    if condition.get("factor"):
        result["condition"][condition["factor"]] = condition["value"]
    schedules = []
    with NativeService(config_path, resolve(base,backend["source_dir"]),
                       model_path=str(resolve(base,backend["model_path"])) if backend.get("model_path") else None,
                       python=backend.get("python",sys.executable), device=backend.get("device","cuda:0"),
                       output=directory/"service", sweep=sweep,
                       start_timeout_s=float(project.get("backend_start_timeout_s",1800))) as service:
        if not pilot:
            context = _validate_context_lengths(project, condition, inputs, config, service)
            if context is not None:
                write_json(directory/"tokenizer-context.json", context)
                result["condition"]["target_tokens"] = context["target_tokens"]
        if condition["channel"] == "hw":
            result["calibration"] = str(_calibrate(config_path,config,service,training[0]["prompt"],
                                                  directory,backend,project,sweep))
        for split, rows in inputs.items():
            measured = []
            for index, original in enumerate(rows):
                identity = f"{directory.name}-{split}-{index:06d}"
                target = {key:original[key] for key in ("source_id","domain","labels","template_id",
                    "support_source_ids","source_spans","prompt_sha256","annotation") if key in original}
                target.update(sample_id=identity, payload=dict(prompt=original["prompt"],
                         max_new_tokens=config["inference"]["generation_tokens"]))
                requests = [target]
                concurrency = condition.get("concurrent_workloads",1)
                for background in range(concurrency-1):
                    row = training[(index+background+1) % len(training)]
                    requests.append(dict(sample_id=f"{identity}-bg-{background}",
                        payload=dict(prompt=row["prompt"], max_new_tokens=config["inference"]["generation_tokens"])))
                gap = float(project.get("arrival_gap_ms", 1.0))
                if not math.isfinite(gap) or gap <= 0:
                    raise ValueError("arrival_gap_ms must be positive for asynchronous workloads")
                spec = dict(native=True, config=str(config_path), endpoint=service.endpoint,
                    channel=condition["channel"], target_id=identity, requests=requests, concurrency=concurrency,
                    arrival_offsets_ms=[gap*i for i in range(concurrency)],
                    capture_seconds=project.get("capture_seconds",8),
                    request_timeout_s=project.get("request_timeout_s",120),
                    probe_start_timeout_s=project.get("probe_start_timeout_s",180),
                    probe_python=project.get("probe_python",backend.get("python",sys.executable)),
                    device=backend.get("device","cuda:0"))
                if sweep:
                    spec["sweep"] = sweep
                spec_path = directory/f"{identity}-collection.json"
                write_json(spec_path,spec)
                attempt_dir = directory/identity
                record = collect(spec_path,attempt_dir)
                for key in ("observation_path","reference_path","schedule_path","native_receipt_path"):
                    if record.get(key):
                        record[key] = str((attempt_dir/record[key]).resolve())
                record["collection"] = result["condition"]
                record["original_sample_id"] = original["sample_id"]
                measured.append(record)
                schedules.extend(read_rows(attempt_dir/"schedule.jsonl"))
            output = directory/f"{split}.jsonl"
            write_rows(output,measured)
            result[split] = str(output)
            result["rows"][split] = measured
    schedule_path = directory/"schedules.jsonl"
    write_rows(schedule_path,schedules)
    result["schedule"] = str(schedule_path)
    return result


def prepare_native(experiment, project_path, capture_output, *, pilot=False):
    """Run native capture once per physical condition and return an evaluation manifest."""
    project_path = Path(project_path).resolve()
    project, base = read_json(project_path), project_path.parent
    for backend in project.get("backends",{}).values():
        executable = backend.get("python")
        if executable and ("/" in executable or "\\" in executable):
            backend["python"] = str(resolve(base,executable).resolve())
    executable = project.get("probe_python")
    if executable and ("/" in executable or "\\" in executable):
        project["probe_python"] = str(resolve(base,executable).resolve())
    output = Path(capture_output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if experiment == "e01_metadata_recovery":
        tolerance = project.get("metadata_time_tolerance_ms")
        if not isinstance(tolerance,(int,float)) or isinstance(tolerance,bool) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Declare metadata_time_tolerance_ms before acquisition")
    plans, auxiliary_sources = [], set()
    for condition in conditions(experiment,project):
        backend = project.get("backends",{}).get(condition["system"])
        if not isinstance(backend,dict) or not backend.get("config") or not backend.get("source_dir"):
            raise ValueError(f"Declare config/source_dir for {condition['system']}")
        paths = {k:resolve(base,v) for k,v in _paths(project,condition,experiment).items()}
        config = _configuration(project,backend,base,condition)
        auxiliary_sources.update(_preflight_inputs(paths,condition,experiment,pilot,config["data"].get("label_vocabulary")))
        key = _capture_key(condition,paths,config)
        plans.append((condition,paths,config,key))
    if auxiliary_sources and not pilot and len(auxiliary_sources) != 1000:
        raise ValueError("All three systems must share the same 1,000 auxiliary source records")
    output.mkdir(parents=True,exist_ok=False)
    runs, cached = [], {}
    for index,(condition,paths,config,key) in enumerate(plans):
        if key not in cached:
            cached[key] = _collect_condition(project,base,condition,paths,config,output/key,pilot=pilot)
        captured = cached[key]
        run = dict(condition,id=f"{index:03d}-{condition['system']}",config=captured["config"],
                   collection=captured["condition"])
        for name in ("train","test","pairs","timings","calibration","schedule"):
            if name in captured:
                run[name] = captured[name]
        if experiment == "e01_metadata_recovery":
            run["tolerance_ms"] = tolerance
        if experiment == "e11_profiling_cost":
            run["timing_scope"] = "observation_collection"
        if experiment == "e06_within_domain_intent":
            for split in ("train","test"):
                path = output/f"{run['id']}-{split}.jsonl"
                write_rows(path,[r for r in captured["rows"][split] if r["domain"] == condition["domain"]])
                run[split] = str(path)
        if experiment == "e12_profiling_scale":
            revised = deepcopy(config)
            revised["data"]["train_per_domain"] = condition["train_per_domain"]
            path = output/f"{run['id']}-config.yaml"
            path.write_text(yaml.safe_dump(revised,sort_keys=False),encoding="utf-8")
            run["config"] = str(path)
        runs.append(run)
    manifest = output/"manifest.json"
    write_json(manifest,dict(experiment=experiment,runs=runs))
    return manifest
