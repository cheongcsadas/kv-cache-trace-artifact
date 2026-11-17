"""A single native project keeps domain and semantic training inputs distinct."""
import json
import hashlib
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from backends import SYSTEMS
from scripts.build_native_project import build
from scripts.setup_native import executable_config
from experiments.chapter5.live import _paths


def test_project_merges_two_tasks_without_replacing_domain_training(tmp_path):
    configs, interpreters = {}, {}
    for system in SYSTEMS:
        config = executable_config(system, tmp_path / "source" / system,
                                   tmp_path / "models" / system, 128, 512)
        configs[system] = tmp_path / f"{system}.yaml"
        configs[system].write_text(yaml.safe_dump(config), encoding="utf-8")
        interpreters[system] = tmp_path / system / "python"
    vocabulary = {domain: dict(intent=[f"i{i}" for i in range(3)],
                               attribute=[f"a{i}" for i in range(60)], keyword=[f"k{i}" for i in range(60)])
                  for domain in ("medical", "financial", "legal", "code")}
    files = {}
    for task in ("domain", "semantic"):
        directory = tmp_path / task
        directory.mkdir()
        data = dict(datasets={system: {"templated_short": dict(train="train.jsonl", test="test.jsonl")}
                              for system in SYSTEMS},
                    profiling={system: {"templated_short": dict(train="profile.jsonl", test="test.jsonl")}
                               for system in SYSTEMS},
                    semantic_annotations=task == "semantic")
        if task == "semantic":
            data["label_vocabulary"] = vocabulary
        files[task] = directory / "project_data.json"
        files[task].write_text(json.dumps(data), encoding="utf-8")
    project = build(files["domain"], configs, interpreters, tolerance_ms=1,
                    semantic_data_path=files["semantic"])
    assert project["label_vocabulary"] == vocabulary  # Labels remain names, not paths.
    assert set(project["intent_pairs"]) == set(vocabulary)
    for system in SYSTEMS:
        for attack in ("MAA", "SAA", "MSA", "SSA"):
            task = "semantic" if attack in ("MSA", "SSA") else "domain"
            condition = dict(system=system, setting="templated_short", attack=attack)
            regular = _paths(project, condition, "e08_event_ablation")
            scale = _paths(project, condition, "e12_profiling_scale")
            assert Path(regular["train"]) == tmp_path / task / "train.jsonl"
            assert Path(scale["train"]) == tmp_path / task / "profile.jsonl"
    intent = _paths(project, dict(system="quest", setting="templated_short"), "e06_within_domain_intent")
    assert Path(intent["train"]) == tmp_path / "semantic" / "train.jsonl"
    data = json.loads(files["semantic"].read_text())
    data["semantic_annotations"] = False
    files["semantic"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="--semantic"):
        build(files["domain"], configs, interpreters, tolerance_ms=1,
              semantic_data_path=files["semantic"])


def _verified_project(tmp_path):
    configs, interpreters, descriptions = {}, {}, {}
    for system in SYSTEMS:
        directory = tmp_path / "runtime-models" / system
        directory.mkdir(parents=True)
        files = {"config.json": b'{"checkpoint":"v0.1"}',
                 "tokenizer.json": b'{"model":{"vocab":{"a":0,"b":1}}}',
                 "tokenizer_config.json": b'{"add_bos_token":true}',
                 "special_tokens_map.json": b'{"bos_token":"<s>"}'}
        for name, contents in files.items():
            (directory / name).write_bytes(contents)
        # The verifier must never read this or enumerate checkpoint shards.
        (directory / "model-00001-of-00002.safetensors").write_bytes(b"weight sentinel")
        descriptions[system] = dict(identity_verified=True, system=system,
            name_or_path="/different/preparation/tokenizer/directory",
            assets={name: dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
                    for name, value in files.items()})
        config = executable_config(system, tmp_path / "sources" / system, directory, 128, 512)
        config["backend"].update(source_dir=f"../sources/{system}", model_path=f"../runtime-models/{system}")
        configs[system] = tmp_path / "native-configs" / f"{system}.yaml"
        configs[system].parent.mkdir(exist_ok=True)
        configs[system].write_text(yaml.safe_dump(config), encoding="utf-8")
        interpreters[system] = tmp_path / system / "python"
    inputs = {}
    for name in ("domain", "semantic"):
        directory = tmp_path / "prepared" / name
        directory.mkdir(parents=True)
        data = dict(datasets={system: {"templated_short": dict(train="train.jsonl", test="test.jsonl")}
                              for system in SYSTEMS}, tokenizers=deepcopy(descriptions),
                    semantic_annotations=name == "semantic", conditions=[{"name": name}],
                    source_partition={"seed": 42, "task": name})
        inputs[name] = directory / "project_data.json"
        inputs[name].write_text(json.dumps(data), encoding="utf-8")
    return inputs, configs, interpreters


def _revise_data(path, mutate):
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_verified_assets_allow_relocated_identical_tokenizers_without_reading_weights(tmp_path, monkeypatch):
    inputs, configs, interpreters = _verified_project(tmp_path)
    opened = []
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert path.suffix not in {".safetensors", ".bin"}
        opened.append(path)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    # Exercise string config paths as well as paths relative to each config file.
    project = build(inputs["domain"], {key: str(value) for key, value in configs.items()},
                    interpreters, tolerance_ms=1, semantic_data_path=inputs["semantic"])
    for system in SYSTEMS:
        assert Path(project["backends"][system]["model_path"]) == tmp_path / "runtime-models" / system
        assert Path(project["datasets"][system]["templated_short"]["train"]) == inputs["domain"].parent / "train.jsonl"
        assert Path(project["semantic_datasets"][system]["templated_short"]["train"]) == inputs["semantic"].parent / "train.jsonl"
        assert tmp_path / "runtime-models" / system / "tokenizer.json" in opened
    assert project["conditions"] == [{"name": "domain"}]
    assert project["semantic_conditions"] == [{"name": "semantic"}]
    assert project["semantic_source_partition"] == {"seed": 42, "task": "semantic"}
    assert project["semantic_tokenizers"] == project["tokenizers"]


@pytest.mark.parametrize("data_section", ["domain", "semantic"])
def test_same_size_checkpoint_asset_mismatch_names_the_data_section(tmp_path, data_section):
    inputs, configs, interpreters = _verified_project(tmp_path)
    if data_section == "domain":
        # A v0.3 model directory can have the same file length or token counts.
        (tmp_path / "runtime-models" / "infinigen" / "config.json").write_bytes(b'{"checkpoint":"v0.3"}')
    else:
        # Domain assets still agree: the separate semantic preparation must also agree.
        _revise_data(inputs["semantic"], lambda data: data["tokenizers"]["infinigen"]["assets"]
                     ["config.json"].update(sha256=hashlib.sha256(b'{"checkpoint":"v0.3"}').hexdigest()))
    section = "tokenizers" if data_section == "domain" else "semantic_tokenizers"
    with pytest.raises(ValueError, match=f"infinigen: {section}.*SHA-256 mismatch for config.json"):
        build(inputs["domain"], configs, interpreters, tolerance_ms=1, semantic_data_path=inputs["semantic"])


@pytest.mark.parametrize("mutation,error", [("extra", "extra=.*added_tokens.json"),
                                           ("missing", "missing=.*tokenizer.json")])
def test_verified_asset_filename_sets_must_match(tmp_path, mutation, error):
    inputs, configs, interpreters = _verified_project(tmp_path)
    directory = tmp_path / "runtime-models" / "quest"
    if mutation == "extra":
        (directory / "added_tokens.json").write_text('{"unexpected":32000}', encoding="utf-8")
    else:
        (directory / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match=error):
        build(inputs["domain"], configs, interpreters, tolerance_ms=1)


@pytest.mark.parametrize("filename", ["../config.json", "subdir/tokenizer.json", "model.safetensors",
                                      "C:\\other\\config.json", "/other/config.json"])
def test_metadata_cannot_request_arbitrary_files_or_weights(tmp_path, filename):
    inputs, configs, interpreters = _verified_project(tmp_path)
    _revise_data(inputs["domain"], lambda data: data["tokenizers"]["quest"]["assets"].update(
        {filename: dict(bytes=1, sha256="0" * 64)}))
    with pytest.raises(ValueError, match="unsupported tokenizer asset filenames"):
        build(inputs["domain"], configs, interpreters, tolerance_ms=1)


@pytest.mark.parametrize("corruption", ["empty", "bad_digest", "too_large", "wrong_system"])
def test_verified_metadata_requires_valid_small_asset_descriptions(tmp_path, corruption):
    inputs, configs, interpreters = _verified_project(tmp_path)
    def change(data):
        description = data["tokenizers"]["quest"]
        if corruption == "empty":
            description["assets"] = {}
        elif corruption == "wrong_system":
            description["system"] = "infinigen"
        else:
            description["assets"]["tokenizer.json"].update(
                {"sha256": "not-a-sha"} if corruption == "bad_digest" else {"bytes": 50_000_001})
    _revise_data(inputs["domain"], change)
    with pytest.raises(ValueError, match="quest: tokenizers"):
        build(inputs["domain"], configs, interpreters, tolerance_ms=1)


def test_existing_embedded_semantic_tokenizers_are_also_checked(tmp_path):
    inputs, configs, interpreters = _verified_project(tmp_path)
    def change(data):
        data["semantic_tokenizers"] = deepcopy(data["tokenizers"])
        data["semantic_tokenizers"]["quest"]["assets"]["tokenizer.json"]["sha256"] = "0" * 64
    _revise_data(inputs["domain"], change)
    with pytest.raises(ValueError, match="quest: semantic_tokenizers.*SHA-256 mismatch"):
        build(inputs["domain"], configs, interpreters, tolerance_ms=1)


def test_replacing_semantic_data_does_not_reuse_previous_dataset_metadata(tmp_path):
    inputs, configs, interpreters = _verified_project(tmp_path)
    def stale(data):
        data["semantic_tokenizers"] = deepcopy(data["tokenizers"])
        data["semantic_tokenizers"]["quest"]["assets"]["tokenizer.json"]["sha256"] = "0" * 64
        data["semantic_conditions"] = [{"old": True}]
        data["semantic_source_partition"] = {"old": True}
    _revise_data(inputs["domain"], stale)
    _revise_data(inputs["semantic"], lambda data: [data.pop(key) for key in
                 ("tokenizers", "conditions", "source_partition")])
    project = build(inputs["domain"], configs, interpreters, tolerance_ms=1,
                    semantic_data_path=inputs["semantic"])
    assert not {"semantic_tokenizers", "semantic_conditions", "semantic_source_partition"} & set(project)
    assert project["semantic_datasets"]["quest"]["templated_short"]["train"] == str(inputs["semantic"].parent / "train.jsonl")
