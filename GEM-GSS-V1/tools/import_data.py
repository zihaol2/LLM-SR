"""Copy external observation files after verifying their archived hashes."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--workspace",
        type=Path,
        help="Local archive root containing the original 71_GEM_GSS folder",
    )
    source.add_argument("--source", type=Path, help="Directory containing T01.npz, T02.npz, etc.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="External data directory; not part of the code upload",
    )
    parser.add_argument("--cases", nargs="+", help="Optional case IDs; default: all eight")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    records = json.loads(
        (root / "gem_gss/assets/release_manifest.json").read_text(encoding="utf8")
    )["cases"]
    selected = set(args.cases or [row["id"] for row in records])
    if selected - {row["id"] for row in records}:
        parser.error("Unknown case ID")
    copies = []
    for row in records:
        if row["id"] not in selected:
            continue
        src = (
            args.workspace / row["archive_data_path"]
            if args.workspace
            else args.source / (row["id"] + ".npz")
        ).resolve()
        if not src.is_file() or sha(src) != row["data_sha256"]:
            raise ValueError("Missing or different observation file: " + str(src))
        dst = (args.output / (row["id"] + ".npz")).resolve()
        if dst.exists() and sha(dst) != row["data_sha256"]:
            raise FileExistsError("Refusing to replace different data: " + str(dst))
        copies.append((src, dst, row["id"]))
    for src, dst, cid in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        print(cid + ": observation hash verified")


if __name__ == "__main__":
    main()
