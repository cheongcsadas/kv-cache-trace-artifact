"""Emit experiment manifests; each setting needs separately collected observations."""
import argparse
import json
from .config import load_config


def matrix(config, suite="main"):
    base = dict(system=config["system"], block_tokens=config["cache"]["block_tokens"],
                residency_fraction=config["cache"]["residency_fraction"])
    if suite == "main":
        for setting in config["data"]["settings"]:
            for attack in ["MAA", "SAA"]:
                yield dict(base, suite=suite, setting=setting, attack=attack)
        if config["system"] == "quest":
            for attack in ["MSA", "SSA"]:
                yield dict(base, suite=suite, setting="templated_short", attack=attack)
    elif suite == "sensitivity":
        for factor in ["residency_fraction", "block_tokens", "context_tokens"]:
            for value in config["sweeps"][factor]:
                defaults = dict(base, context_tokens=config["data"]["short_mean_tokens"],
                                setting="templated_short", suite=suite, attack="MAA", factor=factor)
                yield dict(defaults, **{factor: value})
    elif suite == "defenses":
        for defense in ["fixed_budget", "dynamic_timing"]:
            for mode in ["aware", "oblivious"]:
                yield dict(base, suite=suite, setting="templated_short", attack="MAA",
                           defense=defense, defense_mode=mode)
    elif suite == "ablations":
        for attack in ["MAA", "MSA", "SAA", "SSA"]:
            for ablation in ["full", "remove_hit", "remove_load", "remove_evict", "remove_all"]:
                yield dict(base, suite=suite, setting="templated_short", attack=attack, ablation=ablation)
        for ablation in ["h2d_only", "page_only", "fusion"]:
            yield dict(base, suite=suite, setting="templated_short", attack="SAA", ablation=ablation)
    elif suite in {"concurrency", "profiling"}:
        factor = "concurrent_workloads" if suite == "concurrency" else "training_per_domain"
        for value in config["sweeps"][factor]:
            for attack in ["MAA", "MSA", "SAA", "SSA"]:
                settings = ["templated_short", "templated_long"] if suite == "profiling" else ["templated_short"]
                for setting in settings:
                    yield dict(base, **{factor: value, "suite": suite, "attack": attack,
                                       "setting": setting})
    else: raise ValueError("Unknown suite")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--suite", choices=["main", "sensitivity", "defenses", "concurrency", "profiling", "ablations"], default="main")
    a = p.parse_args()
    for row in matrix(load_config(a.config), a.suite): print(json.dumps(row))


if __name__ == "__main__": main()
