"""Report backend prerequisites without downloading weights or contacting services."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.config import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    a = p.parse_args()
    c = load_config(a.config)
    report = dict(system=c["system"], platform=platform.system(), python=platform.python_version(),
                  nvcc=shutil.which("nvcc") is not None,
                  torch_installed=importlib.util.find_spec("torch") is not None,
                  required_backend_values={k:v for k,v in c["backend"].items() if v is None},
                  required_inference_values={k:v for k,v in c["inference"].items() if v is None})
    if report["torch_installed"]:
        import torch
        report.update(cuda_available=torch.cuda.is_available(), torch_version=torch.__version__)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
