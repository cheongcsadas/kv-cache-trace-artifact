"""Load the pinned, unchanged InfiniGen prediction primitives."""
from pathlib import Path
import hashlib
import importlib.util

REVISION = "3754f52d131c30ab7a310e3c4cf356e8e3326ac3"
FILES = {
    "kv_selection_controller.py": "a6cab73ade0253eb2dc828c8daffe61021636cf3",
    "partial_weight_generation_controller.py": "44d9feb014aaf0449dc9a73ff93caaad36ad5cc6",
}


def load_primitives(source_dir):
    root = Path(source_dir).resolve()
    modules = {}
    for name, expected in FILES.items():
        path = root / "speedup" / "infinigen" / "infinigen" / name
        data = path.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if digest != expected:
            raise ValueError(f"InfiniGen source fingerprint mismatch: {name}; use revision {REVISION}")
        spec = importlib.util.spec_from_file_location("kvtrail_infinigen_" + name[:-3], path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name[:-3]] = module
    return modules

