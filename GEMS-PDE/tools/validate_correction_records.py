"""Check complete recorded local neighborhoods without refitting or truth."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gems_pde import refine
from gems_pde.io import read, digest


def validate(root=ROOT):
    folder = root / "artifacts" / "published"
    count = 0
    accepted = 0
    for source in sorted((folder / "proposed42").glob("T*")):
        task = source.name
        archive = read(source / "equation_archive.json")
        state = read(folder / "proposed" / task / "checkpoint.json")
        for field, events in state["history"].items():
            current = next(
                v["model"]
                for v in archive["archives"][field]
                if v["id"] == archive["best_states"][field]
            )
            for step, event in enumerate(events, 1):
                path = folder / "proposed" / task / "candidates" / f"{field}_sweep{step:02d}.jsonl"
                with path.open(encoding="utf-8") as stream:
                    header = json.loads(next(stream))
                    records = [json.loads(line) for line in stream]
                if header["incumbent_sha256"] != digest(current):
                    raise RuntimeError("Wrong recorded incumbent")
                catalog = {mid: {} for mid in header["catalog_ids"]}
                expected = list(refine.neighbors(current, catalog, 8))
                if len(records) != header["candidate_count"] or len(records) != len(expected):
                    raise RuntimeError("Incomplete neighborhood")
                best = current
                finite = 0
                for index, (record, (ids, edit)) in enumerate(zip(records, expected)):
                    if (
                        record["index"] != index
                        or tuple(record["ids"]) != ids
                        or record["edit"] != edit
                    ):
                        raise RuntimeError(
                            "Recorded support/edit differs from complete enumeration"
                        )
                    model = record["model"]
                    finite += model is not None
                    if model and tuple(sorted(t["id"] for t in model["terms"])) != ids:
                        raise RuntimeError("Fitted model support differs")
                    if model and model["score"] < best["score"]:
                        best = model
                accept = best["score"] * 2 <= current["score"]
                if (
                    event["accepted"] != accept
                    or event["finite_candidates"] != finite
                    or event["candidates"] != len(expected)
                ):
                    raise RuntimeError("Acceptance or accounting mismatch")
                if accept:
                    current = best
                    accepted += 1
                if tuple(event["new_support"]) != refine.signature(current):
                    raise RuntimeError("Wrong selected support")
                count += len(records)
            if digest(current) != digest(state["models"][field]):
                raise RuntimeError("Final checkpoint differs from selected route")
    return {
        "candidate_fits_checked": count,
        "accepted_edits_checked": accepted,
        "truth_access": False,
        "API_calls": 0,
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
