"""GEMS-PDE: persistent equation archive, local edits, fit-before-admission.

Shared rules for all tasks. No reference equation, benchmark-conditioned rule,
or reference-dependent termination. Every proposal gets a joint short fit.
"""

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import random
import time
import numpy as np
from . import common as C
from . import config, mechanisms as M
from .api import call_json
from .run_guard import guard_task


class Graph(M.Graph):
    def add(self, expr, group, parents=(), layer="DGM"):
        try:
            expr = M.normalize(expr, self.fields)
            normalized = C.normalize_node({"id": "pending", "expr": expr, "params": []})
            expr = normalized["expr"]
            if len(expr) > 300:
                raise ValueError("expression size cap")
            if expr in self.seen:
                mid = self.seen[expr]
                node = next(n for n in self.nodes if n["id"] == mid)
                node["parents"] = sorted((set(node["parents"]) | set(parents)) - {mid})
                return mid
            mid = f"M{len(self.nodes):04d}"
            node = {
                **normalized,
                "id": mid,
                "group": group,
                "parents": sorted(set(parents) - {mid}),
                "layer": layer,
            }
            self.nodes.append(node)
            self.seen[expr] = mid
            return mid
        except Exception as error:
            self.rejected.append({"expr": expr, "reason": str(error)})
            return None


def signature(model):
    return tuple(sorted(t["id"] for t in model["terms"]))


def distance(a, b):
    a, b = set(a), set(b)
    return 1 - len(a & b) / max(len(a | b), 1)


class Archive:
    def __init__(self, data, field):
        self.data = data
        self.field = field
        self.states = {}
        self.keys = {}
        self.edges = []
        self.add(C.fit(data, field, []), None, {"operation": "root"}, 0)

    def add(self, model, parent, edit, round_number):
        if model is None:
            return None
        key = signature(model)
        if key not in self.keys:
            sid = f"{self.field}_S{len(self.states):04d}"
            self.keys[key] = sid
            self.states[sid] = {
                "id": sid,
                "model": model,
                "visits": 0,
                "birth_round": round_number,
                "full_refinement_attempted": model["fully_refined"],
                "parents": [],
            }
        else:
            sid = self.keys[key]
            state = self.states[sid]
            if model["score"] < state["model"]["score"]:
                state["model"] = model
            state["full_refinement_attempted"] |= model["fully_refined"]
        if parent and parent != sid:
            state = self.states[sid]
            if parent not in state["parents"]:
                state["parents"].append(parent)
            self.edges.append({"parent": parent, "child": sid, "round": round_number, **edit})
        return sid

    def refine(self, sid, round_number):
        state = self.states[sid]
        m = state["model"]
        if not state["full_refinement_attempted"] and m["terms"]:
            fitted = C.fit(
                self.data, self.field, [t["node"] for t in m["terms"]], m["terms"], full_budget=True
            )
            if fitted:
                self.add(fitted, None, {"operation": "refine"}, round_number)
            state["full_refinement_attempted"] = True
        return state

    def view(self):
        values = list(self.states.values())
        selected = []

        def append(state):
            if state["id"] not in {v["id"] for v in selected}:
                selected.append(state)

        for state in sorted(values, key=lambda v: v["model"]["score"])[:8]:
            append(state)
        frontier = []
        for a in values:
            ma = a["model"]
            if not any(
                b["model"]["train_mse"] <= ma["train_mse"]
                and b["model"]["complexity"] <= ma["complexity"]
                and (
                    b["model"]["train_mse"] < ma["train_mse"]
                    or b["model"]["complexity"] < ma["complexity"]
                )
                for b in values
            ):
                frontier.append(a)
        for state in sorted(frontier, key=lambda v: (v["visits"], v["model"]["complexity"]))[:8]:
            append(state)
        remaining = [v for v in values if v["id"] not in {a["id"] for a in selected}]
        while remaining and len(selected) < C.SETTINGS["archive_prompt_states"]:
            state = min(
                remaining,
                key=lambda a: (
                    a["visits"],
                    -min(
                        (distance(signature(a["model"]), signature(b["model"])) for b in selected),
                        default=1.0,
                    ),
                    a["birth_round"],
                    a["model"]["complexity"],
                ),
            )
            append(state)
            remaining.remove(state)
        return selected

    def exploration_parent(self, focus, round_number):
        pool = [
            v
            for v in self.states.values()
            if v["id"] != focus and v["visits"] < 2 and v["model"]["terms"]
        ]
        if not pool:
            return None
        maximum = max(len(v["model"]["terms"]) for v in pool)
        target = 1 + round_number % maximum
        preferred = [v for v in pool if len(v["model"]["terms"]) == target]
        pool = preferred or pool
        f = signature(self.states[focus]["model"])
        return min(
            pool,
            key=lambda v: (
                v["visits"],
                -distance(signature(v["model"]), f),
                v["birth_round"],
                v["id"],
            ),
        )["id"]


def bank_features(data, graph):
    bank = {}
    for node in graph.nodes:
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
        records = []
        for value in values:
            try:
                vec = data.feature(node, value)
                train = vec[data.trainmask]
                records.append(
                    {
                        "params": value,
                        "vector": vec,
                        "normalized_train": train / max(np.linalg.norm(train), 1e-20),
                    }
                )
            except Exception:
                pass
        if records:
            bank[node["id"]] = {"node": node, "records": records}
    return bank


def edits_for(parent, bank, priorities, explicit, rng, data, field):
    terms = parent["model"]["terms"]
    used = {t["id"] for t in terms}
    pred = sum(
        (
            t["coefficient"]
            * data.feature(t["node"], [t["params"][p] for p in t["node"]["params"]])
            for t in terms
        ),
        np.zeros_like(data.y[field]),
    )
    residual = (data.y[field] - pred)[data.trainmask]
    available = [mid for mid in bank if mid not in used]
    cor = {
        mid: max(abs(v["normalized_train"] @ residual) for v in bank[mid]["records"])
        for mid in available
    }
    ranked = sorted(available, key=lambda mid: cor[mid], reverse=True)
    additions = []
    for mid in priorities[:4] + ranked[:2]:
        if mid in available and mid not in additions:
            additions.append(mid)
    groups = defaultdict(list)
    for mid in available:
        groups[bank[mid]["node"]["group"]].append(mid)
    group_names = sorted(groups)
    rng.shuffle(group_names)
    for group in group_names:
        mid = rng.choice(groups[group])
        if mid not in additions:
            additions.append(mid)
        if len(additions) >= 7:
            break
    others = [mid for mid in available if mid not in additions]
    if others:
        additions.append(rng.choice(others))
    additions = additions[:8]
    candidates = []

    def emit(operation, add=None, remove=None):
        ids = set(used)
        if operation in {"delete", "replace"}:
            if remove not in ids:
                return
            ids.remove(remove)
        if operation in {"add", "replace"}:
            if add not in bank or add in ids:
                return
            ids.add(add)
        if not ids or len(ids) > C.SETTINGS["max_terms"]:
            return
        key = tuple(sorted(ids))
        if key == tuple(sorted(used)) or key in {v[0] for v in candidates}:
            return
        candidates.append((key, {"operation": operation, "added": add, "removed": remove}))

    for e in explicit[:4]:
        if isinstance(e, dict):
            emit(e.get("operation"), e.get("add_node_id"), e.get("remove_node_id"))
    for mid in additions:
        emit("add", mid)
    if len(terms) > 1:
        for t in terms:
            emit("delete", remove=t["id"])
    removals = sorted(used)
    rng.shuffle(removals)
    for removed in removals[:2]:
        for added in additions[:2]:
            emit("replace", added, removed)
    return candidates[: C.SETTINGS["edits_per_parent_cap"]]


def run(index):
    task = config.TASKS[index].id
    guard_task("proposed42", index)
    out = C.ROOT / "proposed42" / task
    out.mkdir(parents=True, exist_ok=True)
    if (out / "frozen_result.json").exists():
        return json.loads((out / "frozen_result.json").read_text(encoding="utf-8"))
    data = C.Data(C.NAMES[index])
    prior = config.prior_for(index)
    rng = random.Random(C.SETTINGS["seed"] + index)
    metadata = {
        "fields": data.fields,
        "shape": list(next(iter(data.full.values())).shape),
        "x_range": [float(data.x.min()), float(data.x.max())],
        "field_ranges": {f: [float(v.min()), float(v.max())] for f, v in data.full.items()},
    }
    prompt = f"""Generate an initial L0 mechanism library and additional G0 mechanisms from weak observations only.
{prior}
Metadata: {json.dumps(metadata)}
Propose 8 simple L0 and 24 diverse additional G0 nodes. Do not name a benchmark or write a complete equation.
Use only {data.fields}, x, a,b,k; arithmetic; D(expression,order) through order4; np.sin/cos/tanh/exp.
Return strict JSON {{"L0":[{{"id":"...","expr":"...","parents":[]}}],"G0":[...]}}."""
    payload = call_json(prompt, out / "api/initial.json", kind="initial")
    graph = Graph(data.fields)
    mapping = {}
    for layer in ["L0", "G0"]:
        for n in payload.get(layer, []):
            mid = graph.add(str(n.get("expr", "")), "llm_proposal", layer=layer)
            if mid:
                mapping[n.get("id")] = mid
    for layer in ["L0", "G0"]:
        for n in payload.get(layer, []):
            if n.get("id") in mapping:
                graph.add(
                    n["expr"],
                    "llm_proposal",
                    [mapping[p] for p in n.get("parents", []) if p in mapping],
                    layer,
                )
    C.dump(
        out / "initial_graph.json",
        {"nodes": copy.deepcopy(graph.nodes), "prior": prior, "metadata": metadata},
    )
    M.grow_graph(graph)
    C.dump(out / "constructible_graph.json", {"nodes": graph.nodes, "rejected": graph.rejected})
    bank = bank_features(data, graph)
    archives = {f: Archive(data, f) for f in data.fields}
    history = []
    if (out / "checkpoint.json").exists():
        checkpoint = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
        history = checkpoint["history"]
        for field, archive in archives.items():
            archive.states = {st["id"]: st for st in checkpoint["archives"][field]}
            archive.keys = {signature(st["model"]): st["id"] for st in archive.states.values()}
            archive.edges = checkpoint["equation_graph"][field]

        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, list) else value

        rng.setstate(tuples(checkpoint["rng_state"]))
    for decision in range(len(history) + 1, C.SETTINGS["proposed_decisions"] + 1):
        views = {f: archive.view() for f, archive in archives.items()}
        summaries = {
            f: [
                {
                    "id": v["id"],
                    "terms": [
                        {k: t[k] for k in ["id", "expr", "params", "coefficient"]}
                        for t in v["model"]["terms"]
                    ],
                    "training_MSE": v["model"]["train_mse"],
                    "validation_MSE": v["model"]["validation_mse"],
                    "complexity": v["model"]["complexity"],
                    "visits": v["visits"],
                    "refinement_attempted": v["full_refinement_attempted"],
                }
                for v in values
            ]
            for f, values in views.items()
        }
        catalog = [
            {"id": mid, "expr": b["node"]["expr"], "group": b["node"]["group"]}
            for mid, b in bank.items()
        ]
        prompt = f"""Guide a persistent equation-state archive. You never see a reference equation.
{prior}
Metadata: {json.dumps(metadata)}
Persistent states: {json.dumps(summaries)}
Available mechanisms: {json.dumps(catalog)}
For each output choose an existing focus state. A temporarily high-error partial equation may have useful descendants.
Choose up to four priority mechanism IDs and up to four add/delete/replace edits relative to that focus state.
No new expressions. An edit uses existing add_node_id and/or an existing term's remove_node_id.
Prefer complementary mechanisms and simpler complete equations, while keeping distinct routes alive.
Return strict JSON {{"focus_states":{{"u":"u_S0000"}},"priority_nodes":{{"u":[...]}},
"edits":{{"u":[{{"operation":"add","add_node_id":"..."}}]}},"reason":"..."}}.
Provide entries for all output fields {data.fields}."""
        policy = call_json(prompt, out / "api" / f"search{decision:02d}.json", kind="policy")
        event = {"round": decision, "outputs": {}}
        for field, archive in archives.items():
            focus = policy.get("focus_states", {}).get(field)
            if focus not in archive.states:
                focus = min(archive.states, key=lambda k: archive.states[k]["model"]["score"])
            explore = archive.exploration_parent(focus, decision)
            parents = list(dict.fromkeys([focus] + ([explore] if explore else [])))
            created = []
            edit_log = []
            for parent_id in parents:
                parent = archive.refine(parent_id, decision)
                parent["visits"] += 1
                priorities = (
                    policy.get("priority_nodes", {}).get(field, []) if parent_id == focus else []
                )
                explicit = policy.get("edits", {}).get(field, []) if parent_id == focus else []
                proposals = edits_for(parent, bank, priorities, explicit, rng, data, field)
                for ids, edit in proposals:
                    model = C.fit(
                        data,
                        field,
                        [bank[mid]["node"] for mid in ids],
                        parent["model"]["terms"],
                        full_budget=False,
                    )
                    sid = archive.add(model, parent_id, edit, decision)
                    if sid:
                        created.append(sid)
                        edit_log.append({"parent": parent_id, "child": sid, **edit})
            created = list(dict.fromkeys(created))
            pending = [
                sid for sid in created if not archive.states[sid]["full_refinement_attempted"]
            ]
            selected = sorted(pending, key=lambda sid: archive.states[sid]["model"]["score"])[:3]
            while len(selected) < 6:
                others = [sid for sid in pending if sid not in selected]
                if not others:
                    break
                sid = min(
                    others,
                    key=lambda sid: (
                        -min(
                            (
                                distance(
                                    signature(archive.states[sid]["model"]),
                                    signature(archive.states[j]["model"]),
                                )
                                for j in selected
                            ),
                            default=1.0,
                        ),
                        archive.states[sid]["model"]["complexity"],
                    ),
                )
                selected.append(sid)
            for sid in selected:
                archive.refine(sid, decision)
            best = min(archive.states.values(), key=lambda st: st["model"]["score"])
            event["outputs"][field] = {
                "focus": focus,
                "exploration_parent": explore,
                "expanded_parents": parents,
                "proposals": len(edit_log),
                "full_refined_states": selected,
                "edits": edit_log,
                "archive_states": len(archive.states),
                "best_state": best["id"],
                "best_validation_MSE": best["model"]["validation_mse"],
            }
        history.append(event)
        C.dump(
            out / "checkpoint.json",
            {
                "history": history,
                "rng_state": rng.getstate(),
                "archives": {f: list(a.states.values()) for f, a in archives.items()},
                "equation_graph": {f: a.edges for f, a in archives.items()},
            },
        )
        print(
            task,
            "PROPOSED",
            decision,
            "archives",
            {f: len(a.states) for f, a in archives.items()},
            "bestval",
            {f: event["outputs"][f]["best_validation_MSE"] for f in data.fields},
            flush=True,
        )
    final = {}
    best_states = {}
    for field, archive in archives.items():
        pool = [st for st in archive.states.values() if st["model"]["terms"]]
        chosen = sorted(pool, key=lambda st: st["model"]["score"])[:4]
        for st in sorted(pool, key=lambda st: st["model"]["complexity"]):
            if st not in chosen:
                chosen.append(st)
            if len(chosen) >= 6:
                break
        while len(chosen) < 8:
            others = [st for st in pool if st not in chosen]
            if not others:
                break
            chosen.append(
                max(
                    others,
                    key=lambda st: min(
                        distance(signature(st["model"]), signature(v["model"])) for v in chosen
                    ),
                )
            )
        for state in chosen:
            archive.refine(state["id"], 13)
        eligible = [st for st in archive.states.values() if st["full_refinement_attempted"]]
        best = min(eligible, key=lambda st: st["model"]["score"])
        # Refit deletion neighbors jointly; do not keep a term solely due to path history.
        for _ in range(8):
            terms = best["model"]["terms"]
            neighbors = []
            if len(terms) <= 1:
                break
            for i in range(len(terms)):
                model = C.fit(
                    data,
                    field,
                    [t["node"] for j, t in enumerate(terms) if j != i],
                    terms,
                    full_budget=True,
                )
                sid = archive.add(
                    model, best["id"], {"operation": "final_delete", "removed": terms[i]["id"]}, 13
                )
                if sid:
                    neighbors.append(archive.states[sid])
            candidate = min([best] + neighbors, key=lambda st: st["model"]["score"])
            if candidate["id"] == best["id"]:
                break
            best = candidate
        final[field] = C.final_equation(data, field, best["model"])
        best_states[field] = best["id"]
    C.dump(
        out / "equation_archive.json",
        {
            "archives": {f: list(a.states.values()) for f, a in archives.items()},
            "edges": {f: a.edges for f, a in archives.items()},
            "best_states": best_states,
        },
    )
    result = {
        "task": C.NAMES[index],
        "anonymous_id": task,
        "method": "GEMS-PDE persistent archive",
        "model": C.SETTINGS["model"],
        "initial_library_calls": 1,
        "search_API_calls": C.SETTINGS["proposed_decisions"],
        "LLM_samples": 1 + C.SETTINGS["proposed_decisions"],
        "equations": final,
        "MSE": float(np.mean([v["MSE"] for v in final.values()])),
        "reference_access": False,
        "frozen_at_unix": time.time(),
    }
    C.dump(out / "frozen_result.json", result)
    print("FROZEN PROPOSED", task, result["MSE"], flush=True)
    return result
