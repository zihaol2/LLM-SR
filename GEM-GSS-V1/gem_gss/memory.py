"""Memory: retain valid hypotheses, deduplicate supports and record tested attempts.

This module proposes no equations and makes no LLM or fitting calls.
"""

import copy
import hashlib
import json
import math

from . import core as C
from .graph import signature


def fit_version(model):
    """Identify this exact fitted equation, including coefficients and parameters."""
    terms = sorted(model["terms"], key=lambda term: (term["id"], term["expr"]))
    return hashlib.sha256(
        json.dumps(terms, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")
    ).hexdigest()


class Archive:
    def __init__(self, data, field, registry):
        self.data, self.field, self.registry = data, field, registry
        self.states, self.keys, self.edges, self.attempted = {}, {}, [], set()
        self.active_ids = []

    def add(self, model, parent, edit, number):
        if model is None or not math.isfinite(model["score"]):
            return None
        key = signature(model)
        if key not in self.keys:
            sid = self.field + "_S" + str(len(self.states)).zfill(5)
            self.keys[key] = sid
            self.states[sid] = dict(
                id=sid, model=copy.deepcopy(model), visits=0, birth_round=number
            )
        sid = self.keys[key]
        if model["score"] < self.states[sid]["model"]["score"]:
            self.states[sid]["model"] = copy.deepcopy(model)
        self.states[sid]["model_version"] = fit_version(self.states[sid]["model"])
        if not self.active_ids and not model["terms"]:
            self.active_ids = [sid]
        if parent is not None and parent != sid:
            self.edges.append(dict(parent=parent, child=sid, round=number, **edit))
        return sid

    def record(self, model, proposal, number):
        support = tuple(proposal["support"])
        self.attempted.add(support)  # Failed fits count as attempts, never as fitted states.
        parent = proposal["parent"]
        self.states[parent]["visits"] += 1
        old = set(signature(self.states[parent]["model"]))
        edit = dict(
            operation=proposal["kind"],
            added=sorted(set(support) - old),
            removed=sorted(old - set(support)),
        )
        for key in ("donor_state", "reason", "rule"):
            if key in proposal:
                edit[key] = proposal[key]
        return self.add(model, parent, edit, number)

    def searchable_states(self):
        # Oversized intermediate fits remain in history, outside ordinary search
        # parents and final output until simplification restores the capacity.
        return [
            state
            for state in self.states.values()
            if len(state["model"]["terms"]) <= C.SETTINGS["max_terms"]
        ]

    def eligible(self, state):
        return len(state["model"]["terms"]) <= C.SETTINGS["max_terms"]

    def raw_best(self):
        return min(self.searchable_states(), key=lambda state: state["model"]["score"])

    def best(self):
        # Numerical helper only, never the per-round or final selection rule.
        return min(self.view(), key=lambda state: state["model"]["score"])

    def view(self):
        return [self.states[sid] for sid in self.active_ids]

    def tested_ids(self):
        return {mid for support in self.attempted for mid in support}

    def export(self):
        return dict(
            states=list(self.states.values()),
            edges=self.edges,
            attempted=sorted(self.attempted),
            active_ids=list(self.active_ids),
            selection_scope="bounded_frontier_v1",
        )

    def restore(self, snapshot):
        self.states = {state["id"]: state for state in snapshot["states"]}
        self.keys = {signature(state["model"]): sid for sid, state in self.states.items()}
        self.edges = snapshot["edges"]
        self.attempted = {tuple(support) for support in snapshot["attempted"]}
        if snapshot.get("selection_scope") != "bounded_frontier_v1":
            raise ValueError(
                "71 requires its own fresh working-set checkpoint; old experiment checkpoints are read-only diagnostics"
            )
        for state in self.states.values():
            state["model_version"] = fit_version(state["model"])
        self.active_ids = list(snapshot["active_ids"])
        if (
            not self.active_ids
            or len(self.active_ids) > 10
            or not set(self.active_ids) <= self.states.keys()
        ):
            raise ValueError("Invalid working set")
