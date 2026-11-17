"""Build pinned Quest CUDA operators with the selected Python environment."""
import argparse
import hashlib
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backends.quest.runtime import verify_source


def build_quest(source_dir, cuda_architectures="native", jobs=4):
    import torch
    if platform.system() != "Linux" or torch.version.cuda is None:
        raise RuntimeError("Build Quest in its Linux CUDA PyTorch environment")
    if jobs < 1:
        raise ValueError("Build jobs must be positive")
    source = verify_source(source_dir)
    for executable in ("cmake", "ninja", "nvcc", "gcc-11", "g++-11"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Install the native build prerequisite: {executable}")
    dependencies = source / "kernels" / "3rdparty"
    for directory in ("raft", "flashinfer", "pybind"):
        if not any((dependencies / directory).iterdir()):
            raise RuntimeError("Initialize the pinned git submodules with setup_native.py")
    ops = source / "quest" / "ops"
    build = ops / "build"
    # Force the local git-submodule RAFT source rather than its mutable branch.
    command = ["cmake", "-S", str(ops), "-B", str(build), "-G", "Ninja",
               f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
               f"-DPython_EXECUTABLE={sys.executable}",
               f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}",
               f"-DCPM_raft_SOURCE={dependencies / 'raft'}"]
    subprocess.run(command, check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", str(jobs)], check=True)
    extensions = list(build.glob("_kernels*.so"))
    if len(extensions) != 1:
        raise RuntimeError("Quest build did not produce exactly one native operator module")
    target = source / "quest" / extensions[0].name
    shutil.copy2(extensions[0], target)
    print(f"Built Quest CUDA operators; SHA-256 {hashlib.sha256(target.read_bytes()).hexdigest()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--cuda-architectures", default="native", help="Use 89 for the paper's RTX 4090")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    build_quest(args.source_dir, args.cuda_architectures, args.jobs)


if __name__ == "__main__":
    main()
