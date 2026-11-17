import copy
import json
from pathlib import Path
import pytest
import yaml
from experiments.chapter5.cli import execute, expected_grid
from experiments.chapter5.common import nested_subset, write_rows, fingerprint
from experiments.chapter5.recovery import metadata_evidence_score, metadata_sources, match_windows
from experiments.chapter5.inference import validate_schedule, validate_condition
from experiments.config import load_config
from experiments.chapter5.specification import validate_spec, canonical_spec

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/quest.yaml"
DOMAINS=("medical","financial","legal","code")


def fixture_rows(split, n=2):
    rows=[]
    for d,domain in enumerate(DOMAINS):
        for i in range(n):
            sid=f"{split}-{domain}-{i}"
            rows.append(dict(sample_id=sid,source_id=sid,provenance="synthetic",domain=domain,
                labels=dict(domain=domain,intent=f"intent-{i%2}",attribute=f"attribute-{i%2}",keyword=f"keyword-{i%2}"),
                observation=dict(request_id=sid,start_ms=0,end_ms=30,snapshots=[],events=[
                    dict(request_id=sid,time_ms=float(j+1),op="LOAD",layer=0,block_id=d*10+j) for j in range(2+d)])))
    return rows


def write_json(path,value):
    path.write_text(json.dumps(value),encoding="utf-8")


def classification_manifest(tmp_path, attack="MAA"):
    train,test=fixture_rows("train"),fixture_rows("test")
    write_rows(tmp_path/"train.jsonl",train);write_rows(tmp_path/"test.jsonl",test)
    run=dict(id="quest_run",config=str(CONFIG),train="train.jsonl",test="test.jsonl",attack=attack,
        setting="templated_short",collection=dict(system="quest",block_tokens=16,residency_fraction=.2,setting="templated_short",context_tokens=1000))
    return run


def test_e03_executes_actual_rf_and_evaluation(tmp_path):
    run=classification_manifest(tmp_path)
    write_json(tmp_path/"manifest.json",dict(runs=[run]))
    result=execute("e03_attribute_inference",tmp_path/"manifest.json",tmp_path/"result",pilot=True,allow_synthetic=True)
    assert result["runs"][0]["metrics"]["n_test"] == 8
    assert (tmp_path/"result/quest_run/model.joblib").exists()
    assert (tmp_path/"result/summary.csv").exists()


def test_e04_outputs_descriptions_from_predictions(tmp_path):
    run=classification_manifest(tmp_path,"MSA")
    write_json(tmp_path/"manifest.json",dict(runs=[run]))
    result=execute("e04_semantic_recovery",tmp_path/"manifest.json",tmp_path/"result",pilot=True,allow_synthetic=True)
    descriptions=[json.loads(l) for l in (tmp_path/"result/quest_run/semantic_descriptions.jsonl").read_text().splitlines()]
    assert len(descriptions) == 8
    assert descriptions[0]["predicted_labels"]["keyword"] in descriptions[0]["description"]
    assert "sr_asr_percent" in result["runs"][0]["metrics"]


def test_explicit_vocabulary_reaches_real_semantic_training_and_results(tmp_path):
    run=classification_manifest(tmp_path,"MSA")
    training=fixture_rows("full-train",60)
    for index,row in enumerate(training):
        row["labels"].update(intent=f"intent-{index%3}",attribute=f"attribute-{index%60}",keyword=f"keyword-{index%60}")
    (tmp_path/"train.jsonl").write_text("".join(json.dumps(row)+"\n" for row in training),encoding="utf-8")
    config=load_config(CONFIG)
    config["implementation_choices"]["metadata_spatial_bins"]=2
    config["data"]["label_vocabulary"]={domain:{slot:[f"{slot}-{i}" for i in range(count)]
        for slot,count in (("intent",3),("attribute",60),("keyword",60))} for domain in DOMAINS}
    (tmp_path/"configured.yaml").write_text(yaml.safe_dump(config),encoding="utf-8")
    run["config"]="configured.yaml"
    write_json(tmp_path/"manifest.json",dict(runs=[run]))
    result=execute("e04_semantic_recovery",tmp_path/"manifest.json",tmp_path/"result",pilot=True,allow_synthetic=True)
    assert result["runs"][0]["semantic_vocabulary_sizes"]["medical"] == dict(intent=3,attribute=60,keyword=60)
    training[0]["acquisition_status"]="failed"
    (tmp_path/"train.jsonl").write_text("".join(json.dumps(row)+"\n" for row in training),encoding="utf-8")
    with pytest.raises(ValueError,match="Successfully localized training"):
        execute("e04_semantic_recovery",tmp_path/"manifest.json",tmp_path/"failed-training",pilot=True,allow_synthetic=True)


def test_e05_failed_request_without_observation_is_scored(tmp_path):
    run=classification_manifest(tmp_path)
    test=fixture_rows("test")
    test[0].pop("observation");test[0]["request_status"]="failed"
    (tmp_path/"test.jsonl").unlink()
    write_rows(tmp_path/"test.jsonl",test)
    write_json(tmp_path/"manifest.json",dict(runs=[run]))
    result=execute("e05_end_to_end",tmp_path/"manifest.json",tmp_path/"result",pilot=True,allow_synthetic=True)
    assert result["runs"][0]["metrics"]["n_test"] == 8
    assert result["runs"][0]["metrics"]["failures"] == 1
    assert result["runs"][0]["localization"]["failed"] == 1


@pytest.mark.parametrize("attack,experiment", [("MAA", "e03_attribute_inference"), ("MSA", "e04_semantic_recovery")])
def test_localization_cannot_remove_a_training_class_without_explicit_vocabulary(tmp_path, attack, experiment):
    run = classification_manifest(tmp_path, attack)
    training = fixture_rows("train")
    if attack == "MAA":
        for row in training:
            if row["domain"] == "medical":
                row["acquisition_status"] = "failed"
    else:
        training[0]["acquisition_status"] = "failed"
    (tmp_path/"train.jsonl").write_text("".join(json.dumps(row)+"\n" for row in training), encoding="utf-8")
    write_json(tmp_path/"manifest.json", dict(runs=[run]))
    with pytest.raises(ValueError, match="Successfully localized training"):
        execute(experiment, tmp_path/"manifest.json", tmp_path/"result", pilot=True, allow_synthetic=True)


def test_synthetic_requires_explicit_pilot(tmp_path):
    run=classification_manifest(tmp_path)
    write_json(tmp_path/"manifest.json",dict(runs=[run]))
    with pytest.raises(ValueError,match="pilot"):
        execute("e03_attribute_inference",tmp_path/"manifest.json",tmp_path/"result",allow_synthetic=True)


def test_trace_interval_matching_never_uses_midpoint():
    pred=[dict(request_id="r",layer=0,block_id=1,op="LOAD",evidence="snapshot",interval_start_ms=0,interval_end_ms=100)]
    ref=[dict(request_id="r",layer=0,block_id=1,op="LOAD",time_ms=99)]
    assert metadata_evidence_score(pred,ref,0)["tp"] == 1
    assert metadata_evidence_score(pred*2,ref,0)["fp"] == 1
    ref[0]["time_ms"]=0
    assert metadata_evidence_score(pred,ref,0)["tp"] == 0
    with pytest.raises(ValueError): metadata_evidence_score(pred,ref,float("nan"))


def test_metadata_source_views_precede_source_filtering():
    view=dict(request_id="r",events=[dict(op="LOAD",source="logs")],snapshots=[])
    obs=dict(request_id="r",events=[],snapshots=[],source_views={"shm":view})
    assert metadata_sources(obs,"shm") == view
    assert match_windows([dict(op="LOAD",start_ms=0,end_ms=3)]*2,[dict(op="LOAD",start_ms=1,end_ms=2)])["fp"] == 1


def test_e01_correlates_real_input_snapshots(tmp_path):
    obs=dict(request_id="r",start_ms=0,end_ms=10,events=[],snapshots=[
        dict(layer=0,time_ms=0,resident=[],source="telemetry"),dict(layer=0,time_ms=10,resident=[1],source="telemetry")])
    refs=[dict(request_id="r",layer=0,block_id=1,op="LOAD",time_ms=9)]
    write_json(tmp_path/"obs.json",obs);write_json(tmp_path/"ref.json",refs)
    write_rows(tmp_path/"pairs.jsonl",[dict(sample_id="p",source_id="s",provenance="synthetic",observation_path="obs.json",reference_path="ref.json")])
    write_json(tmp_path/"manifest.json",dict(runs=[dict(id="q",system="quest",config=str(CONFIG),pairs="pairs.jsonl",tolerance_ms=0)]))
    result=execute("e01_metadata_recovery",tmp_path/"manifest.json",tmp_path/"result",pilot=True,allow_synthetic=True)
    assert result["runs"][0]["scores"]["telemetry"]["trr_percent"] == 100
    assert result["runs"][-1]["unique_auxiliary_sources"] == 1


def test_e02_derives_windows_from_probe_measurements(tmp_path):
    obs=dict(submission_ms=0,capacity_slots=1,h2d=[dict(time_ms=float(i),latency_us=250 if i < 5 else 150) for i in range(35)],compute=[],page=[])
    refs=[dict(time_ms=2,op="LOAD",layer=0,block_id=1)]
    write_json(tmp_path/"obs.json",obs);write_json(tmp_path/"ref.json",refs)
    write_json(tmp_path/"cal.json",dict(baseline_h2d_us=150,compute_threshold_us=None,page_threshold_us=None))
    write_rows(tmp_path/"pairs.jsonl",[dict(sample_id="p",source_id="s",provenance="synthetic",observation_path="obs.json",reference_path="ref.json")])
    write_json(tmp_path/"manifest.json",dict(runs=[dict(id="q",system="quest",config=str(CONFIG),pairs="pairs.jsonl",calibration="cal.json")]))
    result=execute("e02_hardware_recovery",tmp_path/"manifest.json",tmp_path/"result",pilot=True,allow_synthetic=True)
    assert result["runs"][0]["scores"]["LOAD"]["tp"] == 1


def test_recovery_missing_ground_truth_retains_attempt_without_fabricating_score(tmp_path):
    write_rows(tmp_path/"pairs.jsonl",[dict(sample_id="p",source_id="s",provenance="synthetic",
        request_status="failed",acquisition_status="failed")])
    write_json(tmp_path/"manifest.json",dict(runs=[dict(id="q",system="quest",config=str(CONFIG),
        pairs="pairs.jsonl",tolerance_ms=0)]))
    result=execute("e01_metadata_recovery",tmp_path/"manifest.json",tmp_path/"result",pilot=True,allow_synthetic=True)
    assert result["runs"][0]["prompts"] == 1
    assert result["runs"][0]["reference_failures"] == 1
    assert result["runs"][0]["scores"]["combined"]["trr_percent"] is None
    assert result["runs"][-1]["scores"]["combined"]["trr_percent"] is None


def test_e06_trains_intent_not_domain_labels(tmp_path):
    rows=[r for r in fixture_rows("train",4) if r["domain"] == "medical"]
    test=[r for r in fixture_rows("test",4) if r["domain"] == "medical"]
    write_rows(tmp_path/"train.jsonl",rows);write_rows(tmp_path/"test.jsonl",test)
    run=dict(id="binary",config=str(CONFIG),train="train.jsonl",test="test.jsonl",domain="medical",channel="meta",
             positive_label="intent-1",negative_label="intent-0",setting="templated_short",
             collection=dict(system="quest",block_tokens=16,residency_fraction=.2,setting="templated_short"))
    write_json(tmp_path/"manifest.json",dict(runs=[run]))
    result=execute("e06_within_domain_intent",tmp_path/"manifest.json",tmp_path/"result",pilot=True,allow_synthetic=True)
    metric=result["runs"][0]["metrics"]
    assert metric["confusion_labels"] == ["intent-0","intent-1"]
    assert "positive_class_accuracy_percent" in metric


def test_e06_failed_acquisition_without_observation_stays_in_denominator(tmp_path):
    train=[r for r in fixture_rows("train",4) if r["domain"] == "medical"]
    test=[r for r in fixture_rows("test",4) if r["domain"] == "medical"]
    test[1].pop("observation")
    test[1]["acquisition_status"]="failed"
    write_rows(tmp_path/"train.jsonl",train);write_rows(tmp_path/"test.jsonl",test)
    run=dict(id="binary",config=str(CONFIG),train="train.jsonl",test="test.jsonl",domain="medical",channel="meta",
             positive_label="intent-1",negative_label="intent-0",setting="templated_short",
             collection=dict(system="quest",block_tokens=16,residency_fraction=.2,setting="templated_short"))
    write_json(tmp_path/"manifest.json",dict(runs=[run]))
    result=execute("e06_within_domain_intent",tmp_path/"manifest.json",tmp_path/"result",pilot=True,allow_synthetic=True)
    assert result["runs"][0]["metrics"]["n_test"] == 4
    assert result["runs"][0]["metrics"]["failures"] == 1


def test_nested_training_sizes_keep_prefix_and_vocabulary():
    rows=fixture_rows("pool",6)
    small=nested_subset(rows,2,42,True);large=nested_subset(rows,4,42,True)
    assert {r["sample_id"] for r in small} <= {r["sample_id"] for r in large}
    assert all({r["labels"]["intent"] for r in small if r["domain"] == d} == {"intent-0","intent-1"} for d in DOMAINS)
    changed=copy.deepcopy(rows);changed[0]["labels"]["intent"]="changed"
    assert fingerprint(rows) != fingerprint(changed)


def test_sensitivity_config_cannot_be_only_a_label():
    c=load_config(CONFIG)
    run=dict(attack="MAA",setting="templated_short",factor="block_tokens",value=8,collection=dict(block_tokens=8))
    with pytest.raises(ValueError):validate_condition("e09_cache_sensitivity",run,c)
    c["cache"]["block_tokens"]=8
    assert validate_condition("e09_cache_sensitivity",run,c) == "block_tokens"


def test_asynchronous_schedule_checks_actual_overlap_and_arrivals(tmp_path):
    schedule=[dict(sample_id=str(i),planned_offset_ms=i,submission_ms=i,completion_ms=10+i,concurrency=4) for i in range(4)]
    write_rows(tmp_path/"schedule.jsonl",schedule)
    assert validate_schedule(tmp_path/"schedule.jsonl",4,["0"])["peak_active"] == 4
    with pytest.raises(ValueError):validate_schedule(tmp_path/"schedule.jsonl",1,["0"])
    for r in schedule:r["planned_offset_ms"]=0
    write_rows(tmp_path/"batches.jsonl",schedule)
    with pytest.raises(ValueError):validate_schedule(tmp_path/"batches.jsonl",4,["0"])


def test_schedule_uses_absolute_clock_when_sessions_reset_relative_zero(tmp_path):
    schedule=[]
    for session in range(2):
        for i in range(4):
            schedule.append(dict(sample_id=f"s{session}-{i}",planned_offset_ms=i,submission_ms=i,completion_ms=i+10,
                submission_monotonic_ms=1000+session*100+i,completion_monotonic_ms=1010+session*100+i,concurrency=4))
    write_rows(tmp_path/"schedule.jsonl",schedule)
    assert validate_schedule(tmp_path/"schedule.jsonl",4,["s0-0","s1-0"])["peak_active"] == 4


def test_cost_sums_measured_collection_time(tmp_path):
    rows=[dict(sample_id=d,domain=d,provenance="synthetic",collection_ms=2000) for d in DOMAINS]
    write_rows(tmp_path/"timings.jsonl",rows)
    write_json(tmp_path/"manifest.json",dict(runs=[dict(id="cost",system="quest",channel="meta",timings="timings.jsonl",timing_scope="observation_collection")]))
    result=execute("e11_profiling_cost",tmp_path/"manifest.json",tmp_path/"result",pilot=True,allow_synthetic=True)
    assert result["runs"][0]["mean_seconds_per_prompt"] == 2
    assert result["runs"][0]["total_observation_hours"] == 8/3600


def test_cost_rejects_service_latency_substitution(tmp_path):
    rows=[dict(sample_id=d,domain=d,provenance="synthetic",latency_ms=2000) for d in DOMAINS]
    write_rows(tmp_path/"timings.jsonl",rows)
    write_json(tmp_path/"manifest.json",dict(runs=[dict(id="cost",system="quest",channel="meta",timings="timings.jsonl",timing_scope="observation_collection")]))
    with pytest.raises(ValueError,match="collection_ms"):
        execute("e11_profiling_cost",tmp_path/"manifest.json",tmp_path/"result",pilot=True,allow_synthetic=True)


def test_complete_parameter_grids():
    sizes={"e03_attribute_inference":30,"e04_semantic_recovery":2,"e05_end_to_end":32,"e07_transfer_contribution":3,
           "e08_event_ablation":60,"e09_cache_sensitivity":30,"e10_concurrency":48,"e11_profiling_cost":6,"e12_profiling_scale":96}
    for name,count in sizes.items():
        params=yaml.safe_load((ROOT/"experiments"/name/"parameters.yaml").read_text(encoding="utf-8"))
        assert len(expected_grid(name,params)) == count
        assert params["random_forest"] == {"n_estimators":100,"max_depth":10}
        validate_spec(params)


def test_parameter_yaml_cannot_disagree_with_execution():
    params=canonical_spec("e12_profiling_scale")
    params["window_ms"]=20
    with pytest.raises(ValueError,match="window_ms"):validate_spec(params)
    params=canonical_spec("e07_transfer_contribution")
    params["test_total"]=4000
    with pytest.raises(ValueError,match="test_total"):validate_spec(params)
