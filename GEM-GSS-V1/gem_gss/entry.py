"""Portable paths, source integrity and the single 0.4.9 command entry point."""

import hashlib
import json
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
MANIFEST = PACKAGE / "assets/release_manifest.json"
BACKENDS = ("reference", "compiled_chain_2d")


def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def active_source_hashes():
    return {path.name: digest(path) for path in sorted(PACKAGE.glob("*.py"))}


def configure(data_dir=None):
    from . import core

    core.WORKSPACE = (
        Path(data_dir or os.environ.get("GEM_GSS_DATA_DIR", "data")).expanduser().resolve()
    )
    return core


def cache_directory():
    root = Path(os.environ.get("GEM_GSS_CACHE_DIR", str(Path.cwd() / ".gem_gss_cache")))
    return root.expanduser().resolve() / "pde049"


def validate_backend(backend, case=None):
    if backend not in BACKENDS:
        raise ValueError("Unknown numerical backend: " + str(backend))
    if backend == "compiled_chain_2d" and case is not None:
        if len(case["coordinates"]) != 2 or case["fields"] != ["u"]:
            raise ValueError("compiled_chain_2d requires a scalar 2D field named u")


def check_release():
    record = manifest()
    source_tree = (PACKAGE.parent / "pyproject.toml").is_file()
    missing, modified = [], []
    checked = 0
    for relative, expected in record["files"].items():
        if not source_tree and not relative.startswith("gem_gss/"):
            continue
        path = PACKAGE.parent / relative
        checked += 1
        if not path.is_file():
            missing.append(relative)
        elif digest(path) != expected:
            modified.append(relative)
    report = dict(
        version=record["version"],
        source_file_count=record["source_file_count"],
        checked_hashes=checked,
        missing=missing,
        modified=modified,
        results_included=False,
        observation_data_included=False,
        API_calls=0,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing or modified:
        raise SystemExit(1)
    return report


def doctor(core, selected=None):
    rows = []
    for case in core.read(core.ASSETS / "cases.json"):
        if selected and case["id"] != selected:
            continue
        path = core.WORKSPACE / case["data_path"]
        row = dict(case=case["id"], data_available=path.is_file())
        if path.is_file():
            data = core.Data(case)
            graph = core.initial_graph(case)
            row.update(
                observation_hash_verified=True,
                shape=list(data.shape),
                candidates=sum(bool(n["candidate"]) for n in graph["nodes"]),
            )
        rows.append(row)
    if not rows:
        raise ValueError("Unknown case")
    print(
        json.dumps(
            dict(source_profile="pde049", cases=rows, API_calls=0, fits=0),
            ensure_ascii=False,
            indent=2,
        )
    )
    if any(not row["data_available"] for row in rows):
        raise SystemExit(2)


def main(argv=None):
    # A single parser owns both module and installed-script invocations.
    from .cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    main()
