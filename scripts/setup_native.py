"""Fetch pinned backend sources and model weights, then write an executable config."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backends.catalog import SOURCES
from experiments.config import load_config
from experiments.paper_protocol import validate_paper_config


def checkout(system, destination):
    spec = SOURCES[system]
    destination = Path(destination).resolve()
    if destination.exists():
        result = subprocess.run(["git", "-C", str(destination), "rev-parse", "HEAD"],
                                check=True, text=True, capture_output=True)
        if result.stdout.strip() != spec["revision"]:
            raise ValueError("Existing source checkout has a different revision; choose a new directory")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        spec["repository"], str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", spec["revision"]], check=True)
    # A pinned checkout may have been cloned without --recurse-submodules, or
    # preparation may have stopped during a dependency download. Resume both
    # cases before returning a source tree to the native build.
    subprocess.run(["git", "-C", str(destination), "submodule", "update", "--init", "--recursive"], check=True)
    return destination


def model_snapshot(identifier, destination, revision=None):
    from huggingface_hub import HfApi, snapshot_download
    # Standard Hub authentication and model access terms apply. No credentials
    # are written to the configuration or source package.
    actual = HfApi().model_info(identifier, revision=revision).sha
    snapshot_download(repo_id=identifier, revision=actual, local_dir=str(destination),
                      allow_patterns=["*.json", "*.safetensors", "tokenizer.model", "*.txt", "*.jinja"])
    return actual


def executable_config(system, source_dir, model_dir, generation_tokens, selected_token_budget,
                      model_revision=None):
    if (isinstance(generation_tokens, bool) or not isinstance(generation_tokens, int)
            or isinstance(selected_token_budget, bool) or not isinstance(selected_token_budget, int)
            or generation_tokens < 2 or selected_token_budget < 64 or selected_token_budget % 32):
        raise ValueError("Use at least two output tokens and a selection budget >=64 divisible by 32 for all block sweeps")
    config = copy.deepcopy(load_config(ROOT / "configs" / f"{system}.yaml"))
    config["backend"].update(SOURCES[system], source_dir=str(Path(source_dir).resolve()),
                              runtime_module=f"backends.{system}.runtime")
    config["model"]["revision"] = model_revision
    config["backend"]["model_path"] = str(Path(model_dir).resolve())
    config["cache"]["selected_token_budget"] = selected_token_budget
    config["inference"].update(generation_tokens=generation_tokens, dtype="float16",
                                sampling={"do_sample": False})
    validate_paper_config(config)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=SOURCES)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--local-model", action="store_true", help="Use an existing checkpoint without downloading weights")
    parser.add_argument("--generation-tokens", type=int, required=True)
    parser.add_argument("--selected-token-budget", type=int, required=True)
    parser.add_argument("--output-config", required=True)
    args = parser.parse_args(argv)
    destination = Path(args.output_config)
    if destination.exists():
        raise FileExistsError(destination)
    # Reject incompatible execution parameters before fetching native sources or
    # multi-gigabyte checkpoint files.
    result = executable_config(args.system, args.source_dir, args.model_dir,
                               args.generation_tokens, args.selected_token_budget,
                               args.model_revision)
    revision = args.model_revision
    if args.local_model:
        if not (Path(args.model_dir) / "config.json").is_file():
            raise ValueError("Local model directory must contain its model config.json")
    checkout(args.system, args.source_dir)
    if not args.local_model:
        revision = model_snapshot(result["model"]["id"], args.model_dir, revision)
    result["model"]["revision"] = revision
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(result, stream, sort_keys=False)
    print(f"Prepared {args.system}; configuration: {destination}")


if __name__ == "__main__":
    main()
