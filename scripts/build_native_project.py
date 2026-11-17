"""Combine prepared corpus files and configured runtimes into a runnable project."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backends import SYSTEMS
from experiments.config import load_config


def entries(values):
    result = {}
    for value in values:
        system, path = value.split("=", 1)
        if system not in SYSTEMS or system in result:
            raise ValueError("Supply each known system once")
        result[system] = Path(path).resolve()
    return result


def _verify_tokenizer_assets(system, model_path, descriptions, section):
    """Bind runtime tokenizer files to prepared inputs without opening weights."""
    if descriptions is None:
        return
    if not isinstance(descriptions, dict):
        raise ValueError(f"{section} must be a mapping of systems to tokenizer metadata")
    description = descriptions.get(system)
    if description is None:
        return
    if not isinstance(description, dict):
        raise ValueError(f"{system}: invalid {section} metadata")
    if description.get("identity_verified") is not True:
        return
    from datasets.preprocessing.matrix import ASSET_NAMES
    context = f"{system}: {section}"
    if description.get("system", system) != system:
        raise ValueError(f"{context} identifies a different system")
    assets = description.get("assets")
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if not isinstance(assets, dict) or not required <= set(assets):
        raise ValueError(f"{context} verified tokenizer metadata requires configuration and tokenizer assets")
    if set(assets) - set(ASSET_NAMES):
        raise ValueError(f"{context} contains unsupported tokenizer asset filenames")
    for name, expected in assets.items():
        if not isinstance(expected, dict):
            raise ValueError(f"{context} has invalid asset metadata for {name}")
        digest, size = expected.get("sha256"), expected.get("bytes")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= 50_000_000):
            raise ValueError(f"{context} has invalid size/SHA-256 metadata for {name}")
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise ValueError(f"{context} runtime model_path is not a directory: {model_path}")
    # Probe only known small filenames; checkpoint shards are neither listed nor read.
    actual = {name for name in ASSET_NAMES if (model_path / name).is_file()}
    if actual != set(assets):
        raise ValueError(f"{context} tokenizer asset files differ at {model_path}; "
                         f"missing={sorted(set(assets)-actual)}, extra={sorted(actual-set(assets))}")
    for name, expected in assets.items():
        path = model_path / name
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"{context} tokenizer asset size mismatch for {name}: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"{context} tokenizer asset SHA-256 mismatch for {name}: {path}")


def build(data_path, configs, interpreters, *, tolerance_ms, intent_pairs=None, semantic_data_path=None):
    if set(configs) != set(SYSTEMS) or set(interpreters) != set(SYSTEMS):
        raise ValueError("Declare three system configs and three Python environments")
    data_path = Path(data_path).resolve()
    project = json.loads(data_path.read_text(encoding="utf-8"))
    def resolve_paths(value, directory=data_path.parent):
        if isinstance(value, dict):
            return {k: resolve_paths(v, directory) for k, v in value.items()}
        if isinstance(value, str):
            return str((directory / value).resolve())
        return value
    for section in ("datasets","contexts","profiling","auxiliary","transfer_test",
                    "semantic_datasets","semantic_profiling"):
        if section in project:
            project[section]=resolve_paths(project[section])
    if semantic_data_path is not None:
        semantic_data_path=Path(semantic_data_path).resolve()
        semantic=json.loads(semantic_data_path.read_text(encoding="utf-8"))
        if semantic.get("semantic_annotations") is not True or not semantic.get("datasets"):
            raise ValueError("Semantic project data must be prepared with --semantic")
        project["semantic_datasets"]=resolve_paths(semantic["datasets"],semantic_data_path.parent)
        project["semantic_profiling"]=resolve_paths(semantic.get("profiling",{}),semantic_data_path.parent)
        project["semantic_annotations"]=True
        for source, destination in (("tokenizers", "semantic_tokenizers"),
                                    ("conditions", "semantic_conditions"),
                                    ("source_partition", "semantic_source_partition")):
            if source in semantic:
                project[destination] = semantic[source]
            else:
                # Replacing semantic inputs must not retain another dataset's provenance.
                project.pop(destination, None)
        if semantic.get("label_vocabulary") is not None:
            from datasets.preprocessing.labels import validate_label_vocabulary
            project["label_vocabulary"]=validate_label_vocabulary(semantic["label_vocabulary"])
        else:
            # A strict paper-cardinality semantic input may omit a custom vocabulary.
            project.pop("label_vocabulary",None)
    project["backends"] = {}
    for system in SYSTEMS:
        config_path = Path(configs[system]).resolve()
        config = load_config(config_path)
        backend = config["backend"]
        if config["system"] != system:
            raise ValueError("System/config pairing differs")
        for field in ("source_dir", "model_path"):
            if not backend.get(field):
                raise ValueError("Use setup_native.py to prepare source and model paths first")
        model_path = (config_path.parent / backend["model_path"]).resolve()
        for section in ("tokenizers", "semantic_tokenizers"):
            _verify_tokenizer_assets(system, model_path, project.get(section), section)
        project["backends"][system] = dict(config=str(config_path),
            source_dir=str((config_path.parent / backend["source_dir"]).resolve()),
            model_path=str(model_path),
            python=str(interpreters[system]), device="cuda:0")
    if not math.isfinite(tolerance_ms) or tolerance_ms < 0:
        raise ValueError("Metadata matching tolerance must be finite and nonnegative")
    project["metadata_time_tolerance_ms"] = tolerance_ms
    if intent_pairs:
        project["intent_pairs"] = json.loads(Path(intent_pairs).read_text(encoding="utf-8"))
    elif project.get("label_vocabulary"):
        project["intent_pairs"]={domain:dict(positive=slots["intent"][0],negative=slots["intent"][1],channel="meta")
            for domain,slots in project["label_vocabulary"].items() if len(slots["intent"]) >= 2}
    return project


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared project_data.json")
    parser.add_argument("--semantic-data", help="Separate --semantic project_data.json for MSA/SSA and intent tasks")
    parser.add_argument("--config", action="append", required=True, help="system=local/config.yaml")
    parser.add_argument("--python", action="append", required=True, help="system=/path/to/environment/python")
    parser.add_argument("--metadata-time-tolerance-ms", type=float, required=True)
    parser.add_argument("--intent-pairs", help="JSON map domain -> {positive,negative,channel}")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    project = build(args.data, entries(args.config), entries(args.python),
                    tolerance_ms=args.metadata_time_tolerance_ms, intent_pairs=args.intent_pairs,
                    semantic_data_path=args.semantic_data)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(project, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
