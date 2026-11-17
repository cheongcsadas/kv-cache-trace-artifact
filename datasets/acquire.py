"""Fetch pinned public corpus revisions and normalize actual source records."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import urllib.request
from urllib.parse import urlparse
import zipfile
import yaml
from datasets.preprocessing.core import normalize, write_jsonl


def safe_relative(name):
    path = PurePosixPath(name.replace("\\", "/"))
    if not path.parts or path.is_absolute() or any(p in {"..", ""} or ":" in p for p in path.parts):
        raise ValueError("Archive/file path must stay under its destination")
    return path


def extract_zip(archive, destination, max_uncompressed=2*1024**3):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if len(members) > 250000 or sum(m.file_size for m in members) > max_uncompressed:
            raise ValueError("Archive exceeds extraction limits")
        for member in members:
            relative = safe_relative(member.filename)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("Archive symbolic links are not accepted")
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError("Archive member escapes destination")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, target.open("xb") as out:
                    shutil.copyfileobj(source, out, 1024*1024)


def digest_file(path, git_blob=False):
    path = Path(path)
    digest = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        digest.update(b"blob " + str(path.stat().st_size).encode() + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination, git_blob=None, max_bytes=512*1024**2):
    if urlparse(url).scheme != "https":
        raise ValueError("Dataset downloads require HTTPS")
    destination = Path(destination)
    if destination.exists():
        if git_blob and digest_file(destination, True) == git_blob:
            return digest_file(destination)
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "KvTrail-Artifact/1.0"})
    total = 0
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("xb") as stream:
        if urlparse(response.url).scheme != "https":
            raise ValueError("Dataset redirect downgraded HTTPS")
        for chunk in iter(lambda: response.read(1024*1024), b""):
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("Download exceeds configured size limit")
            stream.write(chunk)
    if git_blob and digest_file(temporary, True) != git_blob:
        raise ValueError("Downloaded corpus file disagrees with its pinned Git blob")
    temporary.rename(destination)
    return digest_file(destination)


def _inputs(root, spec):
    paths = sorted({path for pattern in spec["inputs"] for path in root.glob(pattern) if path.is_file()})
    if spec["domain"] == "legal":
        complete = [p for p in paths if p.name == "CUADv1.json"]
        paths = complete or paths
    if not paths:
        raise ValueError("Pinned dataset has no matching corpus files")
    return paths


def normalize_download(name, root, spec, destination):
    rows = {}
    paths = _inputs(Path(root), spec)
    # Normalize MedQuAD as a corpus: some pinned XML documents have empty
    # QAPairs, while nonempty records in the remaining documents stay valid.
    if name == "medquad":
        paths = [Path(root)]
    for number, path in enumerate(paths):
        if name == "codexglue":
            # CONCODE files have .json names but are JSONL with nl/code fields.
            # The local reader accepts an explicit file regardless of suffix.
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        item = json.loads(line)
                        if not isinstance(item, dict) or not (item.get("code") or item.get("nl")):
                            raise ValueError("CONCODE record must contain source code or its original natural-language request")
        for row in normalize(name, path):
            previous = rows.setdefault(row["id"], row)
            if previous != row:
                raise ValueError("Conflicting corpus records share an identifier")
    write_jsonl(destination, rows.values())
    return len(rows)


def acquire(catalog, destination, selected=None):
    catalog = yaml.safe_load(Path(catalog).read_text(encoding="utf-8"))
    datasets = catalog["datasets"]
    selected = selected or list(datasets)
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=False)
    result = {"datasets": {}, "normalized": []}
    for name in selected:
        spec = datasets[name]
        if not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
            raise ValueError("Corpus revision must be a complete commit SHA")
        local = root/name
        local.mkdir()
        artifacts = []
        if "archive" in spec:
            archive = local/"repository.zip"
            artifacts.append(dict(path=archive.name, sha256=download(spec["archive"], archive)))
            extract_zip(archive, local/"source")
        else:
            (local/"source").mkdir()
        for entry in spec.get("files", []):
            relative = safe_relative(entry["path"])
            repository = spec["repository"].removeprefix("https://github.com/")
            if repository == spec["repository"] or len(repository.split("/")) != 2:
                raise ValueError("Catalog expects a public GitHub repository")
            url = f"https://raw.githubusercontent.com/{repository}/{spec['revision']}/{relative}"
            path = local/"source"/Path(*relative.parts)
            artifacts.append(dict(path=str(relative), sha256=download(url, path, entry.get("git_blob")),
                                  git_blob=entry.get("git_blob")))
        for archive in spec.get("extract", []):
            extract_zip(local/"source"/archive, local/"source"/"unpacked")
        output = local/"normalized.jsonl"
        count = normalize_download(name, local/"source", spec, output)
        result["datasets"][name] = dict(repository=spec["repository"], revision=spec["revision"],
            license=spec["license"], subset=spec.get("subset"), records=count, artifacts=artifacts)
        result["normalized"].append(str(output))
    (root/"sources.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="configs/datasets.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+", choices=["medquad", "finqa", "cuad", "codexglue"])
    args = parser.parse_args(argv)
    result = acquire(args.catalog, args.output, args.datasets)
    print(json.dumps({"normalized": result["normalized"]}))


if __name__ == "__main__":
    main()
