"""Persistent partial solutions and local proposal queues, copied from SMS."""

from collections import OrderedDict, deque
import numpy as np
from . import core as C
from .fitting import fit, prediction
from .graph import signature, distance


class Archive:

    def __init__(self, data, field, registry):
        self.data, self.field, self.registry = (data, field, registry)
        self.states, self.keys, self.edges, self.attempted = ({}, {}, [], set())
        self.add(fit(data, field, [], registry), None, {"operation": "root"}, 0)

    def add(self, model, parent, edit, number):
        if model is None:
            return None
        key = signature(model)
        if key not in self.keys:
            sid = self.field + "_S" + str(len(self.states)).zfill(5)
            self.keys[key] = sid
            self.states[sid] = dict(id=sid, model=model, visits=0, birth_round=number)
        sid = self.keys[key]
        if model["score"] < self.states[sid]["model"]["score"]:
            self.states[sid]["model"] = model
        if parent and parent != sid:
            self.edges.append(dict(parent=parent, child=sid, round=number, **edit))
        return sid

    def best(self):
        return min(self.states.values(), key=lambda st: st["model"]["score"])

    def view(self):
        values = list(self.states.values())
        chosen = sorted(values, key=lambda st: st["model"]["score"])[:3]
        remaining = [st for st in values if st not in chosen]
        while remaining and len(chosen) < 6:
            st = min(
                remaining,
                key=lambda a: (
                    a["visits"],
                    -min((distance(signature(a["model"]), signature(b["model"])) for b in chosen)),
                    a["model"]["complexity"],
                    a["id"],
                ),
            )
            chosen.append(st)
            remaining.remove(st)
        return chosen

    def parents(self, focus):
        best = self.best()
        others = [st for st in self.view() if st["id"] != best["id"]]
        explore = (
            min(
                others,
                key=lambda st: (
                    st["visits"],
                    -distance(signature(st["model"]), signature(best["model"])),
                    st["id"],
                ),
            )
            if others
            else best
        )
        chosen = [best, explore] + [self.states[s] for s in focus if s in self.states]
        return list({st["id"]: st for st in chosen}.values())[:3]


class Probes:
    """One all-ones feature per node, for ordering only; failed nodes remain."""

    def __init__(self):
        self.cache = OrderedDict()
        self.bytes = 0

    def get(self, data, node):
        mid = node["id"]
        if mid in self.cache:
            self.cache.move_to_end(mid)
            return self.cache[mid]
        try:
            value = data.feature(node, [1.0] * len(node["params"]))
            value = value / max(float(np.linalg.norm(value)), 1e-30)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            value = None
        self.cache[mid] = value
        self.bytes += 0 if value is None else value.nbytes
        while self.bytes > 96 * 1024**2:
            _, old = self.cache.popitem(last=False)
            self.bytes -= 0 if old is None else old.nbytes
        return value


def rank_nodes(data, field, parent, pool, registry, probes):
    residual = data.y[field] - prediction(data, parent["model"])
    values = []
    used = set(signature(parent["model"]))
    for mid in sorted(pool - used):
        feature = probes.get(data, registry.nodes[mid])
        score = abs(float(feature @ residual)) if feature is not None else -1.0
        values.append((mid, score))
    return [mid for mid, _ in sorted(values, key=lambda pair: (-pair[1], pair[0]))]


def proposal_queues(archive, parents, memory, pool, rng, probes):
    queues = []
    for parent in parents:
        used = set(signature(parent["model"]))
        ranked = rank_nodes(archive.data, archive.field, parent, pool, archive.registry, probes)
        shuffled = ranked.copy()
        rng.shuffle(shuffled)
        groups = [[] for _ in range(6)]

        def emit(target, ids, op, donor=None):
            ids = set(ids)
            if ids == used or len(ids) > C.SETTINGS["max_terms"]:
                return
            edit = dict(operation=op, added=sorted(ids - used), removed=sorted(used - ids))
            if donor:
                edit["donor_state"] = donor
            groups[target].append((parent["id"], tuple(sorted(ids)), edit))

        for mid in ranked:
            emit(0, used | {mid}, "ranked_add")
        for mid in shuffled:
            emit(1, used | {mid}, "random_add")
        for mid in sorted(used):
            emit(2, used - {mid}, "delete")
        for new in ranked:
            for old in sorted(used):
                emit(3, used - {old} | {new}, "ranked_replace")
        for new in shuffled:
            for old in sorted(used):
                emit(4, used - {old} | {new}, "random_replace")
        for donor in memory:
            union = used | set(signature(donor["model"]))
            if len(union) > C.SETTINGS["max_terms"]:
                union = set(rng.sample(sorted(union), C.SETTINGS["max_terms"]))
            emit(5, union, "memory_recombine", donor["id"])
        queues.append(groups)
    return [deque(queues[r][g]) for g in range(6) for r in range(len(queues)) if queues[r][g]]
