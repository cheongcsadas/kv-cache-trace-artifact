"""Mock native child processes to check orchestration and failure publication."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts import prepare_matrix as runner


@pytest.fixture
def paths(tmp_path):
    inputs = []
    for name in ("medical", "financial", "legal", "code"):
        path = tmp_path / (name + " source.jsonl")
        path.write_text('{}\n')
        inputs.append(path)
    annotations = tmp_path / "annotations.json"
    annotations.write_text('{}')
    pythons, tokenizers = {}, {}
    for system in runner.SYSTEMS:
        executable = tmp_path / (system + " native python.exe")
        executable.write_text("mock executable, never run")
        pythons[system] = executable
        directory = tmp_path / (system + " tokenizer")
        directory.mkdir()
        tokenizers[system] = directory
    return dict(inputs=inputs, python_paths=pythons, tokenizer_paths=tokenizers,
                output=tmp_path / "prepared", annotations=annotations)


def metadata(system):
    model_id, model_type, revision = runner.MODEL_IDENTITIES[system]
    result = {"system": system, "model_id": model_id, "model_type": model_type, "revision": revision,
              "identity_verified": True, "backend_sha256": "synthetic-backend",
              "versions": {"transformers": runner.TRANSFORMERS_VERSIONS[system]},
              "assets": {name: {"sha256": sha} for name, sha in runner.IDENTITY_HASHES[system].items()}}
    result["fingerprint"] = runner._json_hash(result)
    return result


def make_child(command):
    output = Path(command[command.index("--output") + 1])
    system = command[command.index("--tokenizer") + 1].split("=", 1)[0]
    output.mkdir()
    child = {section: {system: {}} for section in runner.SECTIONS}
    child.update(semantic_annotations=True, conditions=[])
    native_metadata = metadata(system)
    child["tokenizers"][system] = native_metadata
    def data_file(name):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n')
        return str(path)
    for setting in runner.SETTINGS:
        pair = {split: data_file(f"{setting}/{split}.jsonl") for split in ("train", "test")}
        child["datasets"][system][setting] = pair
        child["profiling"][system][setting] = dict(train=data_file(f"{setting}/profiling_train.jsonl"), test=pair["test"])
        child["conditions"].append(dict(system=system, setting=setting, tokenizer_fingerprint=native_metadata["fingerprint"]))
    for target in (1000, 2000, 3000, 4000):
        child["contexts"][system][str(target)] = {split: data_file(f"context_{target}/{split}.jsonl") for split in ("train", "test")}
    for section in ("auxiliary", "transfer_test"):
        child[section][system] = data_file(section + ".jsonl")
    child["source_partition"] = dict(seed=int(command[command.index("--seed") + 1]), train_fraction=.75,
        domains={domain: {split: dict(source_count=1, sha256=hashlib.sha256(f"{domain}:{split}".encode()).hexdigest())
                         for split in ("train", "test", "auxiliary")} for domain in runner.DOMAINS})
    if "--label-vocabulary" in command:
        child["label_vocabulary"] = json.loads(Path(command[command.index("--label-vocabulary") + 1]).read_text())
    return output, child


def mock_children(monkeypatch, *, mutate=None, fail_at=None):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        if fail_at == len(calls):
            raise subprocess.CalledProcessError(2, command)
        output, child = make_child(command)
        if mutate:
            mutate(child, command, len(calls))
        (output / "project_data.json").write_text(json.dumps(child))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(runner.subprocess, "run", run)
    return calls


def mock_validator(monkeypatch):
    calls = []
    def validate(value):
        calls.append(deepcopy(value))
        return dict(systems=3, settings=5, domains=4, conditions=60)
    monkeypatch.setattr(runner, "validate_complete_matrix", validate)
    return calls


def test_native_commands_and_combined_manifest(monkeypatch, paths):
    calls = mock_children(monkeypatch)
    validations = mock_validator(monkeypatch)
    result = runner.prepare_matrix(**paths)
    assert len(calls) == 3 and len(validations) == 1
    assert set(result["tokenizers"]) == set(runner.SYSTEMS)
    assert len(result["conditions"]) == 15 and result["matrix"]["conditions"] == 60
    assert json.loads((paths["output"] / "project_data.json").read_text()) == result
    for system, (command, options) in zip(runner.SYSTEMS, calls):
        assert command[0] == str(paths["python_paths"][system].resolve())
        assert command[1:3] == ["-m", "datasets.preprocessing.prepare"]
        assert options["shell"] is False and options["check"] is True
        assert options["cwd"] == str(runner.ROOT)
        assert options["env"]["HF_HUB_OFFLINE"] == "1"
        if os.name == "nt":
            assert options["creationflags"] == subprocess.CREATE_NO_WINDOW
        assert "--semantic" in command and "--curate-lengths" in command
        assert "--require-complete-matrix" not in command
        assert str(paths["output"] / system) == command[command.index("--output") + 1]
        assert f"{system}={paths['tokenizer_paths'][system].resolve()}" in command
        assert all(str(value.resolve()) in command for value in paths["inputs"])
    assert result["preparation"]["parameters"]["semantic"] is True
    assert result["preparation"]["source_partition_sha256"] == runner._json_hash(result["source_partition"])
    assert all(item["sha256"] == hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
               for item in result["preparation"]["inputs"])


def test_optional_arguments_are_identical_in_all_native_calls(monkeypatch, paths):
    vocab = {domain: {slot: [f"{slot}{i}" for i in range(count)] for slot, count in
                     (("intent", 3), ("attribute", 60), ("keyword", 60))} for domain in runner.DOMAINS}
    vocabulary = paths["output"].parent / "labels.json"
    vocabulary.write_text(json.dumps(vocab))
    variants = paths["output"].parent / "variants.json"
    variants.write_text('{}')
    calls = mock_children(monkeypatch)
    mock_validator(monkeypatch)
    result = runner.prepare_matrix(**paths, label_vocabulary=vocabulary, prompt_variants=variants,
                                   seed=7, tolerance=.05, curate_lengths=False)
    assert result["label_vocabulary"] == vocab
    for command, _ in calls:
        assert "--curate-lengths" not in command
        assert command[command.index("--seed") + 1] == "7"
        assert command[command.index("--length-tolerance") + 1] == "0.05"
        assert command[command.index("--label-vocabulary") + 1] == str(vocabulary.resolve())
        assert command[command.index("--prompt-variants") + 1] == str(variants.resolve())


def test_failed_child_retains_partial_outputs_without_success_manifest(monkeypatch, paths):
    calls = mock_children(monkeypatch, fail_at=2)
    validations = mock_validator(monkeypatch)
    with pytest.raises(subprocess.CalledProcessError):
        runner.prepare_matrix(**paths)
    assert len(calls) == 2 and validations == []
    assert (paths["output"] / "quest" / "project_data.json").exists()
    assert not (paths["output"] / "project_data.json").exists()


@pytest.mark.parametrize("mutation,match", [
    (lambda child: child["tokenizers"]["quest"]["versions"].update(transformers="4.57.1"), "wrong native"),
    (lambda child: child["tokenizers"]["quest"].update(identity_verified=False), "official tokenizer"),
    (lambda child: child["tokenizers"]["quest"].update(fingerprint="wrong"), "fingerprint"),
    (lambda child: child["datasets"].update(infinigen={}), "exactly this system"),
    (lambda child: child["conditions"].pop(), "five prompt conditions"),
])
def test_child_metadata_is_not_assumed_native(monkeypatch, paths, mutation, match):
    mock_children(monkeypatch, mutate=lambda child, command, index: mutation(child) if index == 1 else None)
    validations = mock_validator(monkeypatch)
    with pytest.raises(ValueError, match=match):
        runner.prepare_matrix(**paths)
    assert validations == [] and not (paths["output"] / "project_data.json").exists()


def test_different_underlying_source_partitions_are_rejected(monkeypatch, paths):
    def mutate(child, command, index):
        if index == 2:
            child["source_partition"]["domains"]["medical"]["train"]["sha256"] = "0" * 64
    mock_children(monkeypatch, mutate=mutate)
    validations = mock_validator(monkeypatch)
    with pytest.raises(ValueError, match="different underlying source partitions"):
        runner.prepare_matrix(**paths)
    assert validations == [] and not (paths["output"] / "project_data.json").exists()


def test_changed_shared_input_aborts_before_publication(monkeypatch, paths):
    def mutate(child, command, index):
        if index == 1:
            paths["annotations"].write_text('{"changed":true}')
    calls = mock_children(monkeypatch, mutate=mutate)
    mock_validator(monkeypatch)
    with pytest.raises(ValueError, match="input changed"):
        runner.prepare_matrix(**paths)
    assert len(calls) == 1 and not (paths["output"] / "project_data.json").exists()


def test_final_matrix_rejection_never_publishes_success(monkeypatch, paths):
    mock_children(monkeypatch)
    def reject(summary):
        raise ValueError("Incomplete matrix")
    monkeypatch.setattr(runner, "validate_complete_matrix", reject)
    with pytest.raises(ValueError, match="Incomplete matrix"):
        runner.prepare_matrix(**paths)
    assert not (paths["output"] / "project_data.json").exists()
    assert not (paths["output"] / ".project_data.json.tmp").exists()


def test_existing_output_is_not_modified(monkeypatch, paths):
    paths["output"].mkdir()
    marker = paths["output"] / "user.txt"
    marker.write_text("preserve")
    calls = mock_children(monkeypatch)
    with pytest.raises(FileExistsError):
        runner.prepare_matrix(**paths)
    assert marker.read_text() == "preserve" and not calls


def test_child_data_paths_must_stay_inside_its_output(monkeypatch, paths):
    def mutate(child, command, index):
        if index == 1:
            child["auxiliary"]["quest"] = str(paths["annotations"].resolve())
    mock_children(monkeypatch, mutate=mutate)
    mock_validator(monkeypatch)
    with pytest.raises(ValueError, match="outside its native output"):
        runner.prepare_matrix(**paths)
    assert not (paths["output"] / "project_data.json").exists()


def test_cli_named_paths_and_default_curation(monkeypatch, paths, capsys):
    calls = []
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return {"matrix": {"conditions": 60}}
    monkeypatch.setattr(runner, "prepare_matrix", prepare)
    args = ["--input", *map(str, paths["inputs"]), "--annotations", str(paths["annotations"]),
            "--output", str(paths["output"])]
    for system in runner.SYSTEMS:
        args += ["--python", f"{system}={paths['python_paths'][system]}",
                 "--tokenizer", f"{system}={paths['tokenizer_paths'][system]}"]
    runner.main(args)
    assert calls[0][1]["curate_lengths"] is True and calls[0][1]["seed"] == 42
    assert json.loads(capsys.readouterr().out)["matrix"]["conditions"] == 60
    with pytest.raises(ValueError, match="exactly once"):
        runner._parse_named(["quest=first", "quest=second"])


def test_all_three_native_interpreters_are_required(monkeypatch, paths):
    paths["python_paths"].pop("infinigen")
    calls = mock_children(monkeypatch)
    with pytest.raises(ValueError, match="each of the three"):
        runner.prepare_matrix(**paths)
    assert not calls and not paths["output"].exists()


def test_native_python_paths_are_not_dereferenced_to_base_interpreters(monkeypatch, paths):
    def reject_resolve(self, *args, **kwargs):
        raise AssertionError("Resolving bin/python would discard a symlinked venv")
    monkeypatch.setattr(Path, "resolve", reject_resolve)
    selected = runner._named_paths(paths["python_paths"], directory=False)
    assert selected == {system: os.path.abspath(path) for system, path in paths["python_paths"].items()}
