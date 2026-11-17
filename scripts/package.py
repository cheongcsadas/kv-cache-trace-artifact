"""Package source files without local data, runtime caches, credentials or machine paths."""
import argparse
import hashlib
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = {"attacks", "configs", "datasets", "defenses", "experiments", "evaluation", "scripts", "tests", "docs", "backends"}
ROOT_FILES = {"README.md", "requirements.txt", "requirements-gpu.txt", "requirements-plots.txt", "pyproject.toml", ".gitignore"}
EXCLUDED = {"__pycache__", ".pytest_cache", "raw", "processed", "runs", ".git", ".venv"}
EXTENSIONS = {".py", ".md", ".txt", ".yaml", ".cu", ".toml", ".json", ".patch", ".sh", ".cpp", ".cuh"}


def source_files():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(p in EXCLUDED for p in relative.parts): continue
        if len(relative.parts) == 1:
            if relative.name not in ROOT_FILES: continue
        elif relative.parts[0] not in SOURCE_DIRS or path.suffix not in EXTENSIONS:
            continue
        yield path


def audit(files):
    findings = []
    # Content-only scan; the local root never appears in the archive.
    patterns = [re.compile(r"[A-Za-z]:[/\\]Users[/\\][^/\\\s]+", re.I),
                re.compile(r"/h[o]me/[^/\s]+/"),
                re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
                re.compile(r"\b(?:ghp|hf)_[A-Za-z0-9]{20,}\b")]
    for path in files:
        text = path.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in patterns):
            findings.append(path.relative_to(ROOT).as_posix())
    if findings: raise ValueError("Local identity/credential pattern detected: " + ", ".join(findings))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=str(ROOT.parent/"KvTrail-Artifact.zip"))
    a = p.parse_args()
    output = Path(a.output)
    files = list(source_files())
    audit(files)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo("KvTrail-Artifact/"+path.relative_to(ROOT).as_posix(), (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix+".sha256").write_text(checksum+"  "+output.name+"\n", encoding="ascii")
    print(f"Packaged {len(files)} source files; SHA-256 {checksum}")


if __name__ == "__main__": main()
