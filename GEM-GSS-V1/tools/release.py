"""Refresh hashes after reviewing source edits; does not run experiments or access APIs."""

import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "gem_gss/assets/release_manifest.json"
PATTERNS = (
    "*.md",
    "pyproject.toml",
    ".env.example",
    ".gitignore",
    "gem_gss/*.py",
    "gem_gss/assets/*.json",
    "gem_gss/assets/*.gz",
    "tests/*.py",
    "tools/*.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    record = json.loads(MANIFEST.read_text(encoding="utf8"))
    paths = sorted(
        {
            path
            for pattern in PATTERNS
            for path in ROOT.glob(pattern)
            if path.is_file() and path != MANIFEST
        }
    )
    record["version"] = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    record["source_file_count"] = len(paths) + 1
    record["source_sha256"] = {
        path.name: sha(path)
        for path in paths
        if path.parent == ROOT / "gem_gss" and path.suffix == ".py"
    }
    record["files"] = {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}
    MANIFEST.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(f"Updated {len(paths)} file hashes for {record['version']}")


if __name__ == "__main__":
    main()
