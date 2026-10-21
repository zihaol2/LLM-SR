"""Maintain immutable data/result hashes after an intentional release update."""

from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
selected = []
for folder in ("gems_pde/resources", "artifacts/published", "assets"):
    selected.extend(p for p in (ROOT / folder).rglob("*") if p.is_file())
selected.extend(
    ROOT / "artifacts" / name
    for name in (
        "references.json",
        "reported_comparison.json",
        "final_equations.txt",
        "source_provenance.json",
    )
)
sha = {
    p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(selected)
}
source = {
    p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted((ROOT / "gems_pde").rglob("*.py"))
}
output = ROOT / "artifacts" / "release_manifest.json"
output.write_text(
    json.dumps({"sha256": sha, "source_sha256": source}, indent=2) + "\n", encoding="utf-8"
)
print("Manifest:", len(sha), "fixed artifacts;", len(source), "source files")
