"""Native launch planning, capture reuse and execution-concurrency invariants."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
import yaml

from experiments.chapter5.live import conditions, _configuration, _capture_key, _preflight_inputs, _paths, prepare_native
from experiments.chapter5.specification import canonical_spec
from experiments.chapter5.cli import check_complete
from experiments.chapter5.inference import validate_schedule

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = sorted(path.name for path in (ROOT/"experiments").glob("e[0-9][0-9]_*")
                     if int(path.name[1:3]) <= 12)


def _project(tmp_path):
    from datasets.preprocessing.annotate import load_taxonomy, label_record
    taxonomy, compiled, _ = load_taxonomy()
    train, test = [], []
    domains = ["medical","financial","legal","code"]
    for split, rows in (("train",train),("test",test)):
        for domain in domains:
            for i in range(2):
                text = taxonomy["domains"][domain]["intent"][i]["aliases"][0]+" "+taxonomy["domains"][domain]["keyword"][0]["aliases"][0]
                row = dict(sample_id=f"{split}-{domain}-{i}", source_id=f"source-{split}-{domain}-{i}",
                           domain=domain, text=text, prompt=text)
                labels, _ = label_record(row, compiled)
                assert labels is not None
                row["labels"] = dict(domain=domain, **labels)
                rows.append(row)
        (tmp_path/f"{split}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
    systems=["quest","infinigen","retrievalattention"]
    return dict(backends={s:dict(config=str(ROOT/"configs"/f"{s}.yaml"),source_dir="external") for s in systems},
        datasets={s:{setting:dict(train="train.jsonl",test="test.jsonl") for setting in
                  ("templated_short","templated_long","unstructured_short","unstructured_long","cross_template")} for s in systems},
        generation_tokens=4, selected_token_budget=64,
        intent_pairs={d:dict(positive=taxonomy["domains"][d]["intent"][1]["label"],
                            negative=taxonomy["domains"][d]["intent"][0]["label"],channel="meta") for d in domains})


@pytest.mark.parametrize("experiment",EXPERIMENTS)
def test_native_condition_grid_covers_each_experiment(experiment,tmp_path):
    project=_project(tmp_path)
    rows=conditions(experiment,project)
    check_complete(experiment,dict(runs=rows),canonical_spec(experiment),tmp_path)


def test_cache_sweep_changes_executed_manager_configuration(tmp_path):
    project=_project(tmp_path)
    for condition in conditions("e09_cache_sensitivity",project):
        config=_configuration(project,project["backends"][condition["system"]],tmp_path,condition)
        group="experiment" if condition["factor"] == "context_tokens" else "cache"
        assert config[group][condition["factor"]] == condition["value"]
        assert config["cache"]["selected_token_budget"] == 64
        assert config["inference"]["generation_tokens"] == 4


def test_capture_reuse_ignores_classifier_but_not_prompt_contents(tmp_path):
    project=_project(tmp_path)
    condition=conditions("e08_event_ablation",project)[0]
    config=_configuration(project,project["backends"]["quest"],tmp_path,condition)
    paths=dict(train=tmp_path/"train.jsonl",test=tmp_path/"test.jsonl")
    original=_capture_key(condition,paths,config)
    changed=dict(condition,ablation="remove_hit",attack="MSA",train_per_domain=400)
    assert _capture_key(changed,paths,config) == original
    with paths["train"].open("a",encoding="utf-8") as stream:
        stream.write("\n")
    assert _capture_key(condition,paths,config) != original


def test_capture_key_separates_actual_defenses_and_their_parameters(tmp_path):
    project = _project(tmp_path)
    condition = conditions("e08_event_ablation", project)[0]
    config = _configuration(project, project["backends"]["quest"], tmp_path, condition)
    paths = dict(train=tmp_path/"train.jsonl", test=tmp_path/"test.jsonl")
    keys = []
    for defense in ("none", "fixed_budget", "dynamic_timing"):
        config.setdefault("experiment", {})["defense"] = defense
        keys.append(_capture_key(condition, paths, config))
    assert len(set(keys)) == 3
    config["defenses"]["dynamic_timing"]["max_delay_ms"] = 4
    assert _capture_key(condition, paths, config) not in keys


def test_rendered_prompt_digest_is_computed_and_checked(tmp_path):
    import hashlib
    from experiments.chapter5.live import _prompts
    _project(tmp_path)
    path = tmp_path/"train.jsonl"
    rows = _prompts(path)
    assert rows[0]["prompt_sha256"] == hashlib.sha256(rows[0]["prompt"].encode()).hexdigest()
    rows[0]["prompt"] += " changed"
    path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="prompt_sha256"):
        _prompts(path)


def test_native_ablation_prepares_six_physical_collections(tmp_path,monkeypatch):
    from experiments.chapter5 import live
    project=_project(tmp_path)
    path=tmp_path/"project.json"
    path.write_text(json.dumps(project),encoding="utf-8")
    calls=[]
    def capture(project,base,condition,paths,config,directory,**kwargs):
        calls.append((condition,config))
        return dict(config=str(directory/"config.yaml"),train=str(directory/"train.jsonl"),
                    test=str(directory/"test.jsonl"),calibration=str(directory/"calibration.json"),
                    condition=dict(system=condition["system"],setting=condition["setting"],
                        block_tokens=config["cache"]["block_tokens"],residency_fraction=config["cache"]["residency_fraction"]))
    monkeypatch.setattr(live,"_collect_condition",capture)
    manifest=prepare_native("e08_event_ablation",path,tmp_path/"captures",pilot=True)
    result=json.loads(manifest.read_text(encoding="utf-8"))
    assert len(calls) == 6
    assert len(result["runs"]) == 60
    for system in project["backends"]:
        paths={r["train"] for r in result["runs"] if r["system"] == system and r["channel"] == "hw"}
        assert len(paths) == 1


def test_missing_semantic_labels_fail_before_any_model_launch(tmp_path,monkeypatch):
    from experiments.chapter5 import live
    project=_project(tmp_path)
    rows=[json.loads(line) for line in (tmp_path/"train.jsonl").read_text().splitlines()]
    rows[0]["labels"].pop("intent")
    (tmp_path/"train.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
    path=tmp_path/"project.json"
    path.write_text(json.dumps(project),encoding="utf-8")
    monkeypatch.setattr(live,"_collect_condition",lambda *args:pytest.fail("launched before validation"))
    with pytest.raises(ValueError,match="curated semantic"):
        prepare_native("e04_semantic_recovery",path,tmp_path/"captures",pilot=True)


def _schedule(tmp_path,parallel=True,failed=False):
    rows=[]
    for i in range(4):
        rows.append(dict(sample_id=str(i),status="failed" if failed and i == 3 else "completed",
            planned_offset_ms=i,concurrency=4,submission_monotonic_ms=i,completion_monotonic_ms=100+i,
            native_receipt=dict(execution_start_monotonic_ms=i if parallel else 20*i,
                                execution_end_monotonic_ms=50+i if parallel else 20*i+10)))
    path=tmp_path/"schedule.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
    return path


def test_http_overlap_does_not_substitute_for_native_execution_concurrency(tmp_path):
    with pytest.raises(ValueError,match="never attains"):
        validate_schedule(_schedule(tmp_path,parallel=False),4,["0"])
    result=validate_schedule(_schedule(tmp_path),4,["0"])
    assert result["clock"] == "native_execution_monotonic"
    assert result["peak_active"] == 4


def test_failed_background_requests_do_not_establish_native_concurrency(tmp_path):
    with pytest.raises(ValueError,match="Failed native"):
        validate_schedule(_schedule(tmp_path,failed=True),4,["0"])


def test_failed_native_target_is_retained_without_invalidating_backgrounds(tmp_path):
    result=validate_schedule(_schedule(tmp_path,failed=True),4,["3"])
    assert result["failed_target_requests"] == 1
    assert result["requests"] == 4


def test_concurrency_peak_must_overlap_each_target(tmp_path):
    path=_schedule(tmp_path)
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows.append(dict(sample_id="isolated-target",status="completed",planned_offset_ms=0,concurrency=4,
        submission_monotonic_ms=100,completion_monotonic_ms=110,
        native_receipt=dict(execution_start_monotonic_ms=100,execution_end_monotonic_ms=110)))
    path.write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")
    with pytest.raises(ValueError,match="target execution never attains"):
        validate_schedule(path,4,["0","isolated-target"])


def test_duplicate_auxiliary_sources_fail_before_capture(tmp_path):
    project=_project(tmp_path)
    rows=[json.loads(line) for line in (tmp_path/"train.jsonl").read_text().splitlines()]
    rows[1]["source_id"]=rows[0]["source_id"]
    path=tmp_path/"auxiliary.jsonl"
    path.write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")
    with pytest.raises(ValueError,match="one prompt per unique"):
        _preflight_inputs(dict(pairs=path),conditions("e01_metadata_recovery",project)[0],
                          "e01_metadata_recovery",True)


def test_invalid_paper_parameter_fails_before_model_launch(tmp_path,monkeypatch):
    from experiments.chapter5 import live
    project=_project(tmp_path)
    config=yaml.safe_load((ROOT/"configs/quest.yaml").read_text())
    config["observation"]["window_ms"]=20
    (tmp_path/"invalid.yaml").write_text(yaml.safe_dump(config),encoding="utf-8")
    project["backends"]["quest"]["config"]="invalid.yaml"
    path=tmp_path/"project.json"
    path.write_text(json.dumps(project),encoding="utf-8")
    monkeypatch.setattr(live,"_collect_condition",lambda *args:pytest.fail("launched before validation"))
    with pytest.raises(ValueError,match="window_ms"):
        prepare_native("e03_attribute_inference",path,tmp_path/"captures",pilot=True)


def test_domain_and_semantic_tasks_use_their_own_prepared_pools(tmp_path):
    project=_project(tmp_path)
    semantic=dict(train="semantic-train.jsonl",test="semantic-test.jsonl")
    profile=dict(train="semantic-profile.jsonl",test="semantic-test.jsonl")
    project["semantic_datasets"]={"quest":{"templated_short":semantic}}
    project["semantic_profiling"]={"quest":{"templated_short":profile}}
    base=dict(system="quest",setting="templated_short")
    assert _paths(project,dict(base,attack="MAA"),"e08_event_ablation")["train"] == "train.jsonl"
    assert _paths(project,dict(base,attack="MSA"),"e08_event_ablation") == semantic
    assert _paths(project,dict(base,attack="SSA"),"e12_profiling_scale") == profile
    assert _paths(project,base,"e06_within_domain_intent") == semantic


def test_native_execution_parameters_fail_before_model_loading(tmp_path):
    project=_project(tmp_path)
    condition=dict(system="quest",setting="templated_short",channel="meta",factor="block_tokens",value=32)
    project["selected_token_budget"]=32
    with pytest.raises(ValueError,match="two blocks"):
        _configuration(project,project["backends"]["quest"],tmp_path,condition)
    project["selected_token_budget"]=64
    project["generation_tokens"]=1
    with pytest.raises(ValueError,match=">=2"):
        _configuration(project,project["backends"]["quest"],tmp_path,condition)
