"""Native model runtimes and CPU/GPU KV management integrations."""
from importlib import import_module

SYSTEMS = ("quest", "infinigen", "retrievalattention")


def load_runtime(config, source_dir, model_path=None, device="cuda:0"):
    system = config["system"]
    if system not in SYSTEMS:
        raise ValueError(f"Unknown backend: {system}")
    module = import_module(f"backends.{system}.runtime")
    return module.Runtime(config, source_dir, model_path=model_path, device=device)
