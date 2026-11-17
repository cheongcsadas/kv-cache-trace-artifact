import copy
import importlib
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import yaml
from experiments.plotting import plot_summary

ROOT=Path(__file__).resolve().parents[1]
runner=importlib.import_module("experiments.run")


def test_modified_classifier_parameter_fails_before_loading_data(tmp_path):
    config=yaml.safe_load((ROOT/"configs/quest.yaml").read_text())
    config["classifier"]["n_estimators"]=99
    path=tmp_path/"bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match="n_estimators"):
        runner.run(path,"absent-train.jsonl","absent-test.jsonl",tmp_path/"out","MAA")


def test_concurrency_plot_uses_all_declared_sweep_points(tmp_path):
    pytest.importorskip("matplotlib")
    records=[]
    for system in ("quest","infinigen","retrievalattention"):
        for attack in ("MAA","MSA","SAA","SSA"):
            for n in (1,4,8,16):
                key="sr_asr_percent" if attack in ("MSA","SSA") else "ai_asr_percent"
                records.append(dict(system=system,attack=attack,concurrent_workloads=n,metrics={key:50}))
    summary=dict(experiment="e10_concurrency",protocol="pilot",runs=records)
    names=plot_summary(summary,tmp_path/"figures")
    assert len(names)==2 and {Path(n).suffix for n in names} == {".png",".pdf"}
    bad=copy.deepcopy(summary)
    bad["runs"].pop()
    with pytest.raises(ValueError,match="every declared"):
        plot_summary(bad,tmp_path/"bad")
