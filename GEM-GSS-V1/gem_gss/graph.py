"""Persistent expression graph and validated online mechanism registration."""

import copy
import re
from collections import defaultdict

from . import core as C
from .core import Analyzer, Graph, canonical, operations, render
from .profile_rules import Profile


class Registry:

    def __init__(self, case, snapshot=None, empty=False):
        self.case = case
        self.graph = Graph(case["fields"], case["coordinates"])
        self.analyzer = Analyzer(case["fields"], case["coordinates"])
        self.infos = {}
        self.families = defaultdict(list)
        self.profile = Profile(case["fields"], case["coordinates"])
        self.probed_roots = set((snapshot or {}).get("probed_roots", []))
        self.exposed_roots = set((snapshot or {}).get("exposed_roots", []))
        self.provenance = []
        self.aliases = []
        if not empty:
            z = snapshot or C.initial_graph(case)
            self.graph.nodes = copy.deepcopy(z["nodes"])
            self.provenance = z.get("online_provenance", [])
            self.aliases = z.get("online_aliases", [])
            self.graph.edges = {(e["parent"], e["child"]): copy.deepcopy(e) for e in z["edges"]}
            for n in self.graph.nodes:
                e = self.graph.language.parse(n["expr"])
                self.graph.lookup[e] = n["id"]
                self.graph.expressions[n["id"]] = e
                if n["candidate"]:
                    self.graph.candidates[n["id"]] = e
        self.nodes = {n["id"]: n for n in self.graph.nodes}
        self.initial_nodes = len(self.nodes)
        self.initial_edges = len(self.graph.edges)
        for mid in self.graph.candidates:
            self.index(mid)
        self.birth = {n["id"]: n.get("birth_round", 0) for n in self.graph.nodes}

    def info(self, mid):
        if mid not in self.infos:
            self.infos[mid] = self.analyzer.analyze(
                self.graph.language.parse(self.nodes[mid]["expr"])
            )
        return self.infos[mid]

    def index(self, mid):
        info = self.info(mid)
        if info.value == 0:
            return
        f = self.analyzer.family(info.value, info.guards)
        self.families[f["body_key"]].append((mid, f["guards"]))

    def propose(self, text, round_number, origin=None, temporary=None):
        temporary = temporary or {}
        refs = {**{k: n["expr"] for k, n in self.nodes.items()}, **temporary}
        text = re.sub(
            "\\b(?:D\\d{6}|N\\d{6}|new\\d+)\\b",
            lambda m: ("(" + refs[m.group(0)] + ")" if m.group(0) in refs else m.group(0)),
            str(text),
        )
        e = canonical(self.graph.language.parse(text))
        self.profile.validate(e)
        expr = render(e)
        if operations(e) > 40 or len(expr) > 600 or e.order > 4:
            raise ValueError("Uniform expression/order budget")
        info = self.analyzer.analyze(e)
        if info.value == 0:
            raise ValueError("Identically zero")
        family = self.analyzer.family(info.value, info.guards)
        covering = next(
            (mid for mid, g in self.families.get(family["body_key"], []) if g <= family["guards"]),
            None,
        )
        if covering is None:
            factor, body = info.value.as_independent(*self.analyzer.physical, as_Add=False)
            if not factor.is_number and body != 1:
                core = self.analyzer.family(body, info.guards, False)
                for mid, _ in self.families.get(core["body_key"], []):
                    other = self.info(mid)
                    coef, _ = other.value.as_independent(*self.analyzer.physical, as_Add=False)
                    f = self.analyzer.family(other.value, other.guards, False)
                    if coef.is_number and coef != 0 and (f["guards"] <= core["guards"]):
                        covering = mid
                        break
        if covering:
            self.aliases.append(
                {
                    "round": round_number,
                    "expr": expr,
                    "target": covering,
                    "origin": origin,
                    "reason": "existing family coverage",
                }
            )
            return (covering, False)
        before = len(self.graph.nodes)
        self.graph.stage = "online"
        family = (
            origin.get("kind", "LLM_feedback_evolution")
            if isinstance(origin, dict)
            else "LLM_feedback_evolution"
        )
        mid = self.graph.add_expr(e, family)
        if mid is None:
            raise ValueError("No field/coordinate-dependent mechanism")
        for n in self.graph.nodes[before:]:
            n["birth_round"] = round_number
            self.nodes[n["id"]] = n
            self.birth[n["id"]] = round_number
        self.nodes[mid] = self.graph.nodes[int(mid[1:])]
        self.nodes[mid]["candidate"] = True
        self.index(mid)
        self.provenance.append(
            {
                "round": round_number,
                "root": mid,
                "origin": origin,
                "added_nodes": [n["id"] for n in self.graph.nodes[before:]],
            }
        )
        return (mid, True)

    def export(self):
        return {
            "nodes": self.graph.nodes,
            "edges": list(self.graph.edges.values()),
            "statistics": self.graph.stats(),
            "online_provenance": self.provenance,
            "online_aliases": self.aliases,
            "reference_access": False,
        }


def signature(model):
    return tuple(sorted((t["id"] for t in model["terms"])))
