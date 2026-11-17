"""Configuration loading; paper parameters and implementation choices stay separate."""
from pathlib import Path
import yaml


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["system"] not in {"quest", "infinigen", "retrievalattention"}:
        raise ValueError("Unknown system")
    if config["observation"]["window_ms"] <= 0 or config["observation"]["max_windows"] <= 0:
        raise ValueError("Invalid observation horizon")
    if not 0 < config["cache"]["residency_fraction"] <= 1:
        raise ValueError("Invalid residency fraction")
    return config
