"""Prepare the complete semantic prompt matrix in three native tokenizer environments."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.preprocessing.labels import DOMAINS, validate_label_vocabulary
from datasets.preprocessing.matrix import (IDENTITY_HASHES, MODEL_IDENTITIES, SETTINGS, SYSTEMS,
                                           TRANSFORMERS_VERSIONS, validate_complete_matrix)

SECTIONS = ("datasets", "contexts", "profiling", "auxiliary", "transfer_test", "tokenizers")


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _file_identity(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected an input file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _unchanged(files):
    if any(_file_identity(item["path"]) != item for item in files):
        raise ValueError("Shared preparation input changed during native matrix generation")


def _named_paths(values, *, directory):
    if not isinstance(values, dict) or set(values) != set(SYSTEMS):
        raise ValueError("Supply exactly one path for each of the three named systems")
    result = {}
    for system, value in values.items():
        path = Path(value).expanduser()
        # A venv's bin/python is commonly a symlink. Dereferencing it would
        # launch the base interpreter and discard that backend's dependencies.
        path = path.resolve(strict=True) if directory else Path(os.path.abspath(path))
        if not (path.is_dir() if directory else path.is_file()):
            raise ValueError(f"Invalid {'tokenizer directory' if directory else 'Python executable'} for {system}")
        result[system] = str(path)
    return result


def _native_metadata(system, description):
    if not isinstance(description, dict):
        raise ValueError(f"{system}: native tokenizer metadata is required")
    model_id, model_type, revision = MODEL_IDENTITIES[system]
    if (description.get("system") != system or description.get("identity_verified") is not True or
            description.get("model_id") != model_id or description.get("model_type") != model_type or
            description.get("revision") != revision or not description.get("backend_sha256") or
            not all(description.get("assets", {}).get(name, {}).get("sha256") == sha
                    for name, sha in IDENTITY_HASHES[system].items())):
        raise ValueError(f"{system}: child output does not identify the official tokenizer assets")
    if description.get("versions", {}).get("transformers") != TRANSFORMERS_VERSIONS[system]:
        raise ValueError(f"{system}: child output used the wrong native Transformers version")
    if description.get("fingerprint") != _json_hash({key: value for key, value in description.items()
                                                     if key != "fingerprint"}):
        raise ValueError(f"{system}: native tokenizer fingerprint mismatch")


def _source_partition(value, seed):
    if (not isinstance(value, dict) or value.get("seed") != seed or value.get("train_fraction") != .75
            or set(value.get("domains", {})) != set(DOMAINS)):
        raise ValueError("Native source partition must describe the same four-domain seeded split")
    for partitions in value["domains"].values():
        if not isinstance(partitions, dict) or set(partitions) != {"train", "test", "auxiliary"}:
            raise ValueError("Native source partition lacks train/test/auxiliary")
        for item in partitions.values():
            if (not isinstance(item, dict) or isinstance(item.get("source_count"), bool) or
                    not isinstance(item.get("source_count"), int) or item["source_count"] <= 0 or
                    not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
                raise ValueError("Invalid native source-partition digest")


def _child_file(value, root):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("Prepared data paths must be absolute")
    path = Path(value).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError("Prepared data path lies outside its native output directory")


def _check_data_paths(child, system, root):
    for section, conditions in (("datasets", set(SETTINGS)), ("profiling", set(SETTINGS)),
                                ("contexts", {"1000", "2000", "3000", "4000"})):
        mapping = child[section][system]
        if not isinstance(mapping, dict) or set(mapping) != conditions:
            raise ValueError(f"{system}: incomplete native {section} paths")
        for pair in mapping.values():
            if not isinstance(pair, dict) or set(pair) != {"train", "test"}:
                raise ValueError("Each prepared condition requires train/test paths")
            for value in pair.values():
                _child_file(value, root)
    for section in ("auxiliary", "transfer_test"):
        _child_file(child[section][system], root)


def prepare_matrix(inputs, python_paths, tokenizer_paths, output, *, annotations,
                   label_vocabulary=None, prompt_variants=None, seed=42,
                   tolerance=.1, curate_lengths=True):
    """Run fixed inputs in native interpreters and publish one validated manifest.

    Partial child outputs are retained on failure. The top-level project_data.json
    is created only after all child manifests, shared splits and matrix checks pass.
    """
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Seed must be an integer")
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or
            not math.isfinite(tolerance) or not 0 <= tolerance <= .1):
        raise ValueError("A complete matrix requires token-mean tolerance from zero to 0.1")
    if not isinstance(curate_lengths, bool):
        raise ValueError("curate_lengths must be boolean")
    python_paths = _named_paths(python_paths, directory=False)
    tokenizer_paths = _named_paths(tokenizer_paths, directory=True)
    if not inputs or isinstance(inputs, (str, Path)):
        raise ValueError("Supply the normalized corpus input paths")
    input_files = [_file_identity(path) for path in inputs]
    if len({item["path"] for item in input_files}) != len(input_files):
        raise ValueError("Normalized corpus input paths must be unique")
    annotation_file = _file_identity(annotations)
    optional_files = {name: _file_identity(path) for name, path in
                      (("label_vocabulary", label_vocabulary), ("prompt_variants", prompt_variants))
                      if path is not None}
    vocabulary = None
    if "label_vocabulary" in optional_files:
        vocabulary = validate_label_vocabulary(json.loads(Path(optional_files["label_vocabulary"]["path"])
                                                          .read_text(encoding="utf-8")))
    shared_files = input_files + [annotation_file] + list(optional_files.values())
    merged = {section: {} for section in SECTIONS}
    merged.update(semantic_annotations=True, conditions=[])
    if vocabulary is not None:
        merged["label_vocabulary"] = vocabulary
    output.mkdir(parents=True)
    for system in SYSTEMS:
        _unchanged(shared_files)
        child_output = output / system
        command = [python_paths[system], "-m", "datasets.preprocessing.prepare",
                   "--input", *[item["path"] for item in input_files],
                   "--annotations", annotation_file["path"], "--output", str(child_output),
                   "--tokenizer", f"{system}={tokenizer_paths[system]}", "--semantic",
                   "--seed", str(seed), "--length-tolerance", str(tolerance)]
        if curate_lengths:
            command.append("--curate-lengths")
        for key, flag in (("label_vocabulary", "--label-vocabulary"), ("prompt_variants", "--prompt-variants")):
            if key in optional_files:
                command.extend((flag, optional_files[key]["path"]))
        options = {"cwd": str(ROOT), "check": True, "shell": False}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        environment = os.environ.copy()
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        subprocess.run(command, env=environment, **options)
        _unchanged(shared_files)
        child = json.loads((child_output / "project_data.json").read_text(encoding="utf-8"))
        if not isinstance(child, dict) or child.get("semantic_annotations") is not True:
            raise ValueError(f"{system}: child output is not a semantic prepared-data manifest")
        for section in SECTIONS:
            if not isinstance(child.get(section), dict) or set(child[section]) != {system}:
                raise ValueError(f"{system}: native {section} must contain exactly this system")
        _native_metadata(system, child["tokenizers"][system])
        _check_data_paths(child, system, child_output)
        partition = child.get("source_partition")
        _source_partition(partition, seed)
        if "source_partition" in merged and partition != merged["source_partition"]:
            raise ValueError("Native systems used different underlying source partitions")
        merged["source_partition"] = partition
        child_vocabulary = child.get("label_vocabulary")
        if child_vocabulary is not None:
            child_vocabulary = validate_label_vocabulary(child_vocabulary)
        if child_vocabulary != vocabulary:
            raise ValueError("Native systems must use the same explicit label vocabulary")
        conditions = child.get("conditions")
        if (not isinstance(conditions, list) or len(conditions) != len(SETTINGS) or
                any(not isinstance(item, dict) or item.get("system") != system for item in conditions) or
                {item.get("setting") for item in conditions} != set(SETTINGS)):
            raise ValueError(f"{system}: child output must contain all five prompt conditions")
        merged["conditions"].extend(conditions)
        for section in SECTIONS:
            merged[section].update(child[section])
    _unchanged(shared_files)
    merged["matrix"] = validate_complete_matrix(merged)
    merged["preparation"] = {
        "schema_version": 1, "inputs": input_files, "annotations": annotation_file,
        **optional_files, "python_paths": python_paths, "tokenizer_paths": tokenizer_paths,
        "parameters": {"seed": seed, "length_tolerance": tolerance, "semantic": True,
                       "curate_lengths": curate_lengths, "train_per_domain": 3000,
                       "test_per_domain": 1000, "profiling_per_domain": 3200,
                       "auxiliary_per_domain": 250},
        "source_partition_sha256": _json_hash(merged["source_partition"]),
    }
    temporary = output / ".project_data.json.tmp"
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(merged, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    destination = output / "project_data.json"
    if destination.exists():
        raise FileExistsError(destination)
    temporary.replace(destination)
    return merged


def _parse_named(values):
    result = {}
    for value in values:
        system, separator, path = value.partition("=")
        if not separator or not path or system not in SYSTEMS or system in result:
            raise ValueError("Use each named system exactly once: system=path")
        result[system] = path
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--label-vocabulary", "--label-vocab")
    parser.add_argument("--prompt-variants")
    parser.add_argument("--python", action="append", required=True, help="system=/native/environment/bin/python")
    parser.add_argument("--tokenizer", action="append", required=True, help="system=/local/model-tokenizer-directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length-tolerance", type=float, default=.1)
    parser.add_argument("--curate-lengths", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    result = prepare_matrix(args.input, _parse_named(args.python), _parse_named(args.tokenizer), args.output,
        annotations=args.annotations, label_vocabulary=args.label_vocabulary, prompt_variants=args.prompt_variants,
        seed=args.seed, tolerance=args.length_tolerance, curate_lengths=args.curate_lengths)
    print(json.dumps({"project_data": str(Path(args.output).resolve() / "project_data.json"),
                      "matrix": result["matrix"]}))


if __name__ == "__main__":
    main()
