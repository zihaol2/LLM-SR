"""Uniform continuation of experiment 42; no reference equations or API calls.

One rule: enumerate the complete one-add/delete/replace neighborhood, jointly
fit every candidate, and accept only a twofold improvement of the unchanged
validation/complexity score. Original search and original incumbents are kept.
"""

import argparse
import hashlib
import itertools
from pathlib import Path
import json
import time
from . import common as C, config
from .run_guard import guard_task

MAX_SWEEPS = 4
MIN_SCORE_FACTOR = 2.0


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def signature(model):
    return tuple(sorted(t["id"] for t in model["terms"]))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def valid_catalog(data, nodes):
    valid = {}
    rejected = []
    for node in nodes:
        count = len(node["params"])
        values = (
            [()]
            if not count
            else list(
                dict.fromkeys(
                    [(p,) * count for p in C.SETTINGS["coarse_parameters"]]
                    + [
                        tuple(p if i == j else 1.0 for i in range(count))
                        for j in range(count)
                        for p in C.SETTINGS["coarse_parameters"]
                    ]
                )
            )
        )
        for value in values:
            try:
                data.feature(node, value)
                valid[node["id"]] = node
                break
            except Exception:
                pass
        else:
            rejected.append(node["id"])
    return valid, rejected


def neighbors(model, catalog, max_terms):
    """Every distinct support exactly once; no correlation/parameter screening."""
    used = set(signature(model))
    available = sorted(set(catalog) - used)
    seen = set()
    for op, added, removed in itertools.chain(
        (("delete", None, mid) for mid in sorted(used) if len(used) > 1),
        (("replace", new, old) for old in sorted(used) for new in available),
        (("add", mid, None) for mid in available if len(used) < max_terms),
    ):
        ids = tuple(
            sorted((used - ({removed} if removed else set())) | ({added} if added else set()))
        )
        if ids in seen:
            continue
        seen.add(ids)
        yield ids, {"operation": op, "added": added, "removed": removed}


def sweep(data, field, incumbent, catalog, path):
    proposals = list(neighbors(incumbent, catalog, C.SETTINGS["max_terms"]))
    header = {
        "incumbent_sha256": digest(incumbent),
        "catalog_ids": sorted(catalog),
        "candidate_count": len(proposals),
        "joint_fit_budget": C.SETTINGS["max_nfev_full"],
    }
    done = {}
    best = incumbent
    best_edit = None
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0]) == header
        for line in lines[1:]:
            record = json.loads(line)
            done[record["index"]] = record
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(header, allow_nan=False) + "\n", encoding="utf-8")
    began = time.time()
    finite = 0
    with path.open("a", encoding="utf-8") as stream:
        for index, (ids, edit) in enumerate(proposals):
            if index in done:
                record = done[index]
            else:
                model = C.fit(
                    data, field, [catalog[mid] for mid in ids], incumbent["terms"], full_budget=True
                )
                record = {"index": index, "ids": ids, "edit": edit, "model": model}
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
            model = record["model"]
            finite += model is not None
            if model and model["score"] < best["score"]:
                best = model
                best_edit = edit
            if (index + 1) % 40 == 0 or index + 1 == len(proposals):
                print(
                    field,
                    path.stem,
                    index + 1,
                    "/",
                    len(proposals),
                    "best_score",
                    best["score"],
                    "seconds",
                    round(time.time() - began, 1),
                    flush=True,
                )
    accepted = best["score"] * MIN_SCORE_FACTOR <= incumbent["score"]
    return (best if accepted else incumbent), {
        "candidates": len(proposals),
        "finite_candidates": finite,
        "old_score": incumbent["score"],
        "best_candidate_score": best["score"],
        "accepted": accepted,
        "selected_edit": best_edit if accepted else None,
        "old_support": signature(incumbent),
        "new_support": signature(best) if accepted else signature(incumbent),
    }


def run(index, source_dir=None):
    source_dir = Path(source_dir) if source_dir is not None else C.ROOT / "proposed42"
    task = config.TASKS[index].id
    guard_task("proposed", index, source_dir / task)
    out = C.ROOT / "proposed" / task
    out.mkdir(parents=True, exist_ok=True)
    if (out / "frozen_result.json").exists():
        return read(out / "frozen_result.json")
    source = source_dir / task
    original = read(source / "frozen_result.json")
    if (
        original.get("reference_access") is not False
        or original.get("task") != C.NAMES[index]
        or original.get("anonymous_id") != task
    ):
        raise ValueError("Correction source must be a matching, reference-free frozen search")
    archive = read(source / "equation_archive.json")
    nodes = read(source / "constructible_graph.json")["nodes"]
    data = C.Data(C.NAMES[index])
    catalog, rejected = valid_catalog(data, nodes)
    statefile = out / "checkpoint.json"
    if statefile.exists():
        state = read(statefile)
    else:
        models = {
            field: next(st["model"] for st in values if st["id"] == archive["best_states"][field])
            for field, values in archive["archives"].items()
        }
        state = {
            "models": models,
            "history": {f: [] for f in data.fields},
            "completed_fields": [],
            "source_result_sha256": hashlib.sha256(
                (source / "frozen_result.json").read_bytes()
            ).hexdigest(),
        }
        C.dump(statefile, state)
    for field in data.fields:
        if field in state["completed_fields"]:
            continue
        if state["history"][field] and not state["history"][field][-1]["accepted"]:
            # A crash after the rejected sweep was saved must not start a new sweep.
            state["completed_fields"].append(field)
            C.dump(statefile, state)
            continue
        for step in range(len(state["history"][field]) + 1, MAX_SWEEPS + 1):
            model, event = sweep(
                data,
                field,
                state["models"][field],
                catalog,
                out / "candidates" / f"{field}_sweep{step:02d}.jsonl",
            )
            state["models"][field] = model
            state["history"][field].append(event)
            C.dump(statefile, state)
            print(task, field, "SWEEP", step, json.dumps(event), flush=True)
            if not event["accepted"]:
                break
        state["completed_fields"].append(field)
        C.dump(statefile, state)
    equations = {}
    for field in data.fields:
        changed = any(event["accepted"] for event in state["history"][field])
        equations[field] = (
            C.final_equation(data, field, state["models"][field])
            if changed
            else original["equations"][field]
        )
    result = {
        "task": C.NAMES[index],
        "anonymous_id": task,
        "method": "GEMS-PDE 42 + complete local joint-fit correction",
        "model": original.get("model", C.SETTINGS["model"]),
        "model_usage": "inherited search only; no new API call",
        "fresh_API_calls": 0,
        "inherited_initial_calls": 1,
        "inherited_search_calls": original.get("search_API_calls", 12),
        "continuation_not_independent_restart": True,
        "equations": equations,
        "MSE": float(sum(eq["MSE"] for eq in equations.values()) / len(equations)),
        "valid_catalog_nodes": len(catalog),
        "invalid_catalog_nodes": rejected,
        "source_result_sha256": state["source_result_sha256"],
        "reference_access": False,
        "frozen_at_unix": time.time(),
    }
    C.dump(out / "frozen_result.json", result)
    print("FROZEN", task, result["MSE"], flush=True)
    return result
