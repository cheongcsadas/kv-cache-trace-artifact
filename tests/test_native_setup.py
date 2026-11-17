"""Deployment config and installed experiment entry-point invariants."""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from backends import SYSTEMS
from backends.catalog import SOURCES
from scripts.setup_native import checkout, executable_config
from scripts.build_native_project import build


def test_native_project_connects_prepared_prompts_to_each_configured_model(tmp_path):
    data = tmp_path / "prepared" / "project_data.json"
    data.parent.mkdir()
    data.write_text(json.dumps(dict(datasets={s:{"templated_short":dict(train="train.jsonl",test="test.jsonl")}
                                             for s in SYSTEMS})), encoding="utf-8")
    configs, interpreters = {}, {}
    for system in SYSTEMS:
        config = executable_config(system,tmp_path/"source"/system,tmp_path/"models"/system,128,512,"fixed-revision")
        configs[system] = tmp_path / f"{system}.yaml"
        configs[system].write_text(yaml.safe_dump(config), encoding="utf-8")
        interpreters[system] = tmp_path / system / "python"
    result = build(data,configs,interpreters,tolerance_ms=.5)
    for system in SYSTEMS:
        backend = result["backends"][system]
        config = yaml.safe_load(Path(backend["config"]).read_text())
        assert config["backend"]["revision"] == SOURCES[system]["revision"]
        assert config["cache"]["block_tokens"] == 16
        assert config["cache"]["residency_fraction"] == .2
        assert config["cache"]["selected_token_budget"] == 512
        assert config["inference"]["generation_tokens"] == 128
        assert config["model"]["revision"] == "fixed-revision"
        assert Path(result["datasets"][system]["templated_short"]["train"]) == data.parent / "train.jsonl"
        assert Path(backend["model_path"]) == tmp_path / "models" / system


def test_block_sweep_budget_rejects_incompatible_selected_token_counts(tmp_path):
    with pytest.raises(ValueError,match="divisible by 32"):
        executable_config("quest",tmp_path,tmp_path,128,500)


@pytest.mark.parametrize("selected", [32, True, 64.0])
def test_block_sweep_budget_reserves_historical_and_tail_pages(tmp_path, selected):
    with pytest.raises(ValueError, match=">=64"):
        executable_config("quest", tmp_path, tmp_path, 128, selected)


def test_existing_pinned_checkout_resumes_submodule_preparation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=SOURCES["quest"]["revision"]+"\n")
    monkeypatch.setattr("scripts.setup_native.subprocess.run", run)
    assert checkout("quest", tmp_path) == tmp_path.resolve()
    assert len(calls) == 2
    assert calls[1][-4:] == ["submodule", "update", "--init", "--recursive"]


def test_invalid_parameters_fail_before_source_or_checkpoint_download(tmp_path, monkeypatch):
    from scripts.setup_native import main
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid execution settings must not trigger downloads")
    monkeypatch.setattr("scripts.setup_native.checkout", unexpected)
    monkeypatch.setattr("scripts.setup_native.model_snapshot", unexpected)
    with pytest.raises(ValueError, match=">=64"):
        main(["--system", "quest", "--source-dir", str(tmp_path/"source"),
              "--model-dir", str(tmp_path/"model"), "--generation-tokens", "128",
              "--selected-token-budget", "32", "--output-config", str(tmp_path/"config.yaml")])


def test_every_experiment_has_a_package_command_entry_point():
    root=Path(__file__).resolve().parents[1]/"experiments"
    names=[p.name for p in root.glob("e[0-9][0-9]_*")]
    assert len(names) == 14
    for name in names:
        assert importlib.util.find_spec(f"experiments.{name}.__main__") is not None
