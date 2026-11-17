"""Execute remaining Chapter 5 entry points on explicitly synthetic inputs."""
import json
from pathlib import Path
import pytest
import yaml
from experiments.chapter5.cli import execute

ROOT=Path(__file__).resolve().parents[1]
DOMAINS=("medical","financial","legal","code")


def save_json(path,value): path.write_text(json.dumps(value),encoding="utf-8")
def save_rows(path,rows): path.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")


def build(tmp_path,hardware=False,ntrain=2,config_change=None):
    config=yaml.safe_load((ROOT/"configs/quest.yaml").read_text())
    config["implementation_choices"]["metadata_spatial_bins"]=2
    if config_change: config_change(config)
    (tmp_path/"config.yaml").write_text(yaml.safe_dump(config),encoding="utf-8")
    for split,count in (("train",ntrain),("test",2)):
        rows=[]
        for d,domain in enumerate(DOMAINS):
            for i in range(count):
                sid=f"{split}-{domain}-{i}"
                if hardware:
                    obs=dict(submission_ms=0,capacity_slots=1,
                        h2d=[dict(time_ms=float(t),latency_us=230+d*10 if t<4+d else 150) for t in range(50)],
                        compute=[],page=[])
                else:
                    obs=dict(request_id=sid,start_ms=0,end_ms=20,snapshots=[],events=[
                        dict(time_ms=1,op="LOAD",layer=0,block_id=d),dict(time_ms=5,op="HIT",layer=0,block_id=d)])
                rows.append(dict(sample_id=sid,source_id=sid,provenance="synthetic",domain=domain,observation=obs))
        save_rows(tmp_path/f"{split}.jsonl",rows)
    save_json(tmp_path/"calibration.json",dict(baseline_h2d_us=150,compute_threshold_us=None,page_threshold_us=None))
    return dict(id="condition",config="config.yaml",train="train.jsonl",test="test.jsonl",
        attack="SAA" if hardware else "MAA",calibration="calibration.json" if hardware else None,
        setting="templated_short",collection=dict(system="quest",block_tokens=config["cache"]["block_tokens"],
        residency_fraction=.2,setting="templated_short",context_tokens=1000))


@pytest.mark.parametrize("experiment",["e05_end_to_end","e07_transfer_contribution","e08_event_ablation",
                                      "e09_cache_sensitivity","e10_concurrency"])
def test_experiment_runs_real_feature_rf_and_metric_pipeline(tmp_path,experiment):
    def change(c):
        if experiment=="e09_cache_sensitivity":
            c["cache"]["block_tokens"]=8
            c["cache"]["selected_token_budget"]=128
    run=build(tmp_path,hardware=experiment=="e07_transfer_contribution",config_change=change)
    if experiment=="e05_end_to_end":
        rows=[json.loads(l) for l in (tmp_path/"test.jsonl").read_text().splitlines()]
        rows[0].pop("observation")
        rows[0]["acquisition_status"]="failed"
        rows[0]["failure_reason"]="fixture acquisition failure"
        save_rows(tmp_path/"test.jsonl",rows)
    if experiment=="e07_transfer_contribution": run["ablation"]="h2d_only"
    if experiment=="e08_event_ablation": run["ablation"]="remove_load"
    if experiment=="e09_cache_sensitivity":
        run.update(factor="block_tokens",value=8)
        run["collection"].update(selected_token_budget=128)
    if experiment=="e10_concurrency":
        run.update(concurrent_workloads=4,schedule="schedule.jsonl")
        run["collection"]["concurrent_workloads"]=4
        ids=[json.loads(l)["sample_id"] for split in ("train","test")
             for l in (tmp_path/f"{split}.jsonl").read_text().splitlines()]
        save_rows(tmp_path/"schedule.jsonl",[dict(sample_id=sid,submission_ms=(i//4)*20+i%4,
            completion_ms=(i//4)*20+i%4+10,planned_offset_ms=i%4,concurrency=4) for i,sid in enumerate(ids)])
    save_json(tmp_path/"manifest.json",dict(runs=[run]))
    result=execute(experiment,tmp_path/"manifest.json",tmp_path/"output",pilot=True,allow_synthetic=True)
    assert result["runs"][0]["metrics"]["n_test"]==8
    assert (tmp_path/"output/condition/model.joblib").is_file()
    if experiment=="e05_end_to_end": assert result["runs"][0]["metrics"]["failures"]==1


def test_profiling_scale_executes_400_per_domain_training(tmp_path):
    run=build(tmp_path,ntrain=400)
    run["train_per_domain"]=400
    save_json(tmp_path/"manifest.json",dict(runs=[run]))
    result=execute("e12_profiling_scale",tmp_path/"manifest.json",tmp_path/"output",pilot=True,allow_synthetic=True)
    assert result["runs"][0]["train_count"]==1600
    assert result["runs"][0]["test_count"]==8
    assert (tmp_path/"output/condition/model.joblib").is_file()


def test_transfer_feature_comparison_rejects_calibration_changes(tmp_path):
    run=build(tmp_path,hardware=True)
    run["ablation"]="h2d_only"
    other=dict(run,id="page_condition",ablation="page_only",calibration="different.json")
    save_json(tmp_path/"different.json",dict(baseline_h2d_us=151,compute_threshold_us=None,page_threshold_us=None))
    save_json(tmp_path/"manifest.json",dict(runs=[run,other]))
    with pytest.raises(ValueError,match="same frozen calibration"):
        execute("e07_transfer_contribution",tmp_path/"manifest.json",tmp_path/"output",pilot=True,allow_synthetic=True)
