"""Collect actual undefended and defended native executions on identical prompts."""
from copy import deepcopy
from pathlib import Path
import sys

from experiments.chapter5.common import read_json, resolve, write_json
from experiments.chapter5.live import _configuration, _paths, _preflight_inputs, _collect_condition, _capture_key
from .specification import SYSTEMS, DEFENSES, EXPERIMENTS


def prepare_native(experiment, project_path, output, *, pilot=False):
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown Chapter 6 experiment")
    project_path = Path(project_path).resolve()
    project, base = read_json(project_path), project_path.parent
    for backend in project.get("backends", {}).values():
        executable = backend.get("python", sys.executable)
        if "/" in executable or "\\" in executable:
            backend["python"] = str(resolve(base, executable).resolve())
    plans = []
    for system in SYSTEMS:
        backend = project.get("backends", {}).get(system)
        if not isinstance(backend, dict) or not backend.get("source_dir") or not backend.get("config"):
            raise ValueError(f"Declare the native backend for {system}")
        condition = dict(system=system, attack="MAA", channel="meta", setting="templated_short")
        paths = {name: resolve(base, value).resolve() for name, value in
                 _paths(project, condition, "e03_attribute_inference").items()}
        # Validate the full held-out split even when only test latency is collected.
        _preflight_inputs(paths, condition, "e03_attribute_inference", pilot)
        original = _configuration(project, backend, base, condition)
        for defense in DEFENSES:
            config = deepcopy(original)
            config.setdefault("experiment", {})["defense"] = defense
            capture_paths = paths if experiment == EXPERIMENTS[0] else {"test": paths["test"]}
            # Defense must be part of the key: identical prompts with different
            # loading operations must never reuse undefended observations.
            key = f"{system}-{defense}-{_capture_key(condition, capture_paths, config)}"
            plans.append((dict(condition, defense=defense), capture_paths, config, key))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    captures = []
    for condition, paths, config, key in plans:
        actual = _collect_condition(project, base, condition, paths, config, output/key, pilot=pilot)
        captures.append(dict(system=condition["system"], defense=condition["defense"],
            setting="templated_short", channel="meta", config=actual["config"],
            collection=actual["condition"], **{split: actual[split] for split in ("train", "test") if split in actual}))
    manifest = output/"manifest.json"
    write_json(manifest, dict(experiment=experiment, captures=captures))
    return manifest
