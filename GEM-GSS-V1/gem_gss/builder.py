"""Finite, task-independent graph construction under the normative grammar.

No benchmark labels, reference equations, solution generator or fitting imports.
"""

import copy
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import networkx as nx

from .grammar import SPEC, Grammar, multi_indices
from .profile_rules import Profile

ROOT = Path(__file__).resolve().parent
RECIPE = json.loads((ROOT / "assets/recipe.json").read_text(encoding="utf8"))


def render(e):
    if e.op in {"Field", "Coordinate", "Parameter", "Constant"}:
        return str(e.value)
    if e.op == "PD":
        return "PD(" + render(e.args[0]) + "," + repr(tuple(e.value)) + ")"
    if e.op == "Negate":
        return "(-" + render(e.args[0]) + ")"
    signs = {"Add": "+", "Subtract": "-", "Multiply": "*", "Divide": "/", "Power": "**"}
    if e.op in signs:
        return "(" + render(e.args[0]) + signs[e.op] + render(e.args[1]) + ")"
    return e.op + "(" + render(e.args[0]) + ")"


def operations(e):
    return int(bool(e.args)) + sum((operations(a) for a in e.args))


def depends(e):
    if e.op in {"Field", "Coordinate"}:
        return True
    return any((depends(a) for a in e.args))


def meaningful_derivative(e, alpha, coordinates):

    def axes(z):
        if z.op == "Field":
            return set(range(len(coordinates)))
        if z.op == "Coordinate":
            return {coordinates.index(z.value)} if z.value in coordinates else set()
        return set().union(*(axes(a) for a in z.args)) if z.args else set()

    active = axes(e)
    return all((not count or j in active for j, count in enumerate(alpha)))


def canonical(e):
    args = tuple((canonical(a) for a in e.args))
    if e.op == "Power" and args[1].op == "Constant" and (float(args[1].value) == 1):
        return args[0]
    if e.op == "Multiply":
        for j in (0, 1):
            if args[j].op == "Constant" and float(args[j].value) == 1:
                return args[1 - j]
    if e.op == "Negate" and args[0].op == "Negate":
        return args[0].args[0]
    return replace(e, args=args)


class Graph:

    def __init__(self, fields, coordinates):
        self.fields = list(fields)
        self.coordinates = list(coordinates)
        self.language = Grammar(fields, coordinates, RECIPE["max_spatial_order"])
        self.nodes = []
        self.lookup = {}
        self.expressions = {}
        self.edges = {}
        self.events = []
        self.rejected = []
        self.stage = "initial"
        self.candidates = {}
        self.profile = Profile(fields, coordinates)

    def conditions(self, e):
        found = set()

        def visit(z):
            if z.op in SPEC["functions"]:
                found.add(z.op + ": " + SPEC["functions"][z.op]["real_domain"])
            if z.op == "Divide":
                found.add("Divide: denominator != 0")
            if z.op == "Power":
                found.add("Power: real-valued branch and required differentiability")
            for a in z.args:
                visit(a)

        visit(e)
        if e.order:
            found.add(
                "PD: operand must have the required spatial smoothness on the evaluated domain"
            )
        return sorted(found)

    def intern(self, e):
        if not depends(e) and (not (e.op == "Constant" and str(e.value) == "1")):
            return None
        if e in self.lookup:
            return self.lookup[e]
        children = [self.intern(a) for a in e.args]
        mid = "N" + str(len(self.nodes)).zfill(6)
        depth = max((self.nodes[int(c[1:])]["depth"] for c in children if c), default=-1) + 1
        parameters = set()

        def collect(z):
            if z.op == "Parameter":
                parameters.add(z.value)
            for a in z.args:
                collect(a)

        collect(e)
        entry = {
            "id": mid,
            "expr": render(e),
            "operator": e.op,
            "order": e.order,
            "operations": operations(e),
            "depth": depth,
            "birth_stage": self.stage,
            "candidate": False,
            "candidate_families": [],
            "params": sorted(parameters),
            "domain_conditions": self.conditions(e),
            "numeric_status": "not_checked",
            "operands": [
                {"node_id": c} if c else {"literal_expression": render(a)}
                for c, a in zip(children, e.args)
            ],
        }
        self.lookup[e] = mid
        self.expressions[mid] = e
        self.nodes.append(entry)
        for position, parent in enumerate(children):
            if parent:
                edge = self.edges.setdefault(
                    (parent, mid),
                    {"parent": parent, "child": mid, "operand_positions": []},
                )
                edge["operand_positions"].append(position)
        return mid

    def add_expr(self, e, family):
        reason = self.profile.rejection(e)
        if reason:
            self.rejected.append(
                dict(stage=self.stage, family=family, expr=render(e), reason=reason)
            )
            return None
        e = canonical(e)
        text = render(e)
        if (
            e.order > RECIPE["max_spatial_order"]
            or operations(e) > RECIPE["max_expression_operations"]
            or len(text) > RECIPE["max_expression_characters"]
        ):
            self.rejected.append(
                {
                    "stage": self.stage,
                    "family": family,
                    "expr": text,
                    "reason": "uniform expression budget",
                }
            )
            return None
        if not depends(e) and (not (e.op == "Constant" and str(e.value) == "1")):
            return None
        mid = self.intern(e)
        node = self.nodes[int(mid[1:])]
        node["candidate"] = True
        if family not in node["candidate_families"]:
            node["candidate_families"].append(family)
        self.candidates[mid] = e
        self.events.append({"node": mid, "family": family, "stage": self.stage})
        return mid

    def add(self, expr, group, parents=(), layer=None):
        try:
            return self.add_expr(self.language.parse(expr), group)
        except Exception as exc:
            self.rejected.append(
                {"stage": self.stage, "family": group, "expr": expr, "reason": str(exc)}
            )
            return None

    def stats(self):
        g = nx.DiGraph()
        g.add_nodes_from((n["id"] for n in self.nodes))
        g.add_edges_from(self.edges)
        sizes = sorted(map(len, nx.weakly_connected_components(g)), reverse=True)
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "candidate_roots": len(self.candidates),
            "support_only_nodes": len(self.nodes) - len(self.candidates),
            "construction_events": len(self.events),
            "weak_components": len(sizes),
            "isolates": len(list(nx.isolates(g))),
            "largest_component_nodes": sizes[0] if sizes else 0,
            "acyclic": nx.is_directed_acyclic_graph(g),
            "operator_counts": dict(Counter((n["operator"] for n in self.nodes))),
            "birth_stage_counts": dict(Counter((n["birth_stage"] for n in self.nodes))),
        }

    def export(self):
        return {
            "nodes": self.nodes,
            "edges": list(self.edges.values()),
            "construction_events": self.events,
            "rejected": self.rejected,
            "statistics": self.stats(),
            "numeric_validation": "not performed",
            "reference_access": False,
            "recipe_version": RECIPE["version"],
        }


def original_constructor(graph):
    from .constructors import grow_1d, grow_spatial

    if len(graph.coordinates) == 1:
        grow_1d(graph)
    elif len(graph.coordinates) in (2, 3):
        grow_spatial(graph)
    else:
        raise ValueError("Graph construction supports one, two or three spatial axes")


def build(payload, fields, coordinates):
    g = Graph(fields, coordinates)
    language = g.language
    stage_stats = []

    def snapshot():
        stage_stats.append({"stage": g.stage, **g.stats()})

    initial = []
    mapping = {}
    for layer in ("L0", "G0"):
        for item in payload.get(layer, []):
            mid = g.add(item["expr"], "LLM_" + layer)
            if mid:
                initial.append(mid)
                mapping[item.get("id")] = mid
    suggestions = [
        {
            "child": mapping.get(n.get("id")),
            "proposed_parents": [mapping[p] for p in n.get("parents", []) if p in mapping],
        }
        for layer in ("L0", "G0")
        for n in payload.get(layer, [])
        if n.get("id") in mapping
    ]
    snapshot()
    initial_graph = copy.deepcopy(g.export())
    initial_graph["accepted_seed_roots"] = len(set(initial))
    initial_graph["llm_parent_suggestions"] = suggestions
    g.stage = "base_and_compatibility"
    for text in [*fields, *coordinates, "t", "1"]:
        g.add(text, "declared_base")
    original_constructor(g)
    snapshot()
    atoms = [language.parse(text) for text in [*fields, *coordinates, "t"]]
    field_expr = [language.parse(f) for f in fields]
    field_powers = [language.parse(f + "**" + str(p)) for f in fields for p in (2, 3)]
    g.stage = "powers"
    for e in atoms:
        for p in RECIPE["power_exponents"]:
            g.add_expr(language.parse("(" + render(e) + ")**(" + str(p) + ")"), "power")
    snapshot()
    g.stage = "elementary"
    zero_order_seeds = [
        g.expressions[mid]
        for mid in set(initial)
        if g.expressions[mid].order == 0 and operations(g.expressions[mid]) <= 6
    ]
    elementary_base = sorted(set(atoms + field_powers + zero_order_seeds), key=render)
    for e in elementary_base:
        affine = language.parse("a*(" + render(e) + ")+b")
        for fun in SPEC["functions"]:
            g.add_expr(language.node(fun, e), "elementary_direct")
            g.add_expr(language.node(fun, affine), "elementary_affine")
    snapshot()
    g.stage = "binary"
    gradients = [language.axis(e, j, 1) for e in field_expr for j in range(len(coordinates))]
    binary_base = sorted(set(atoms + field_powers + gradients), key=render)
    for i, e in enumerate(binary_base):
        for j, f in enumerate(binary_base):
            if i <= j:
                for op in ("Add", "Multiply"):
                    g.add_expr(language.node(op, e, f), "binary_" + op)
            if i != j:
                for op in ("Subtract", "Divide"):
                    g.add_expr(language.node(op, e, f), "binary_" + op)
    snapshot()
    g.stage = "pde_macros"
    coefficients = (
        atoms + [language.parse("1")] + [language.parse(f + "/(a+" + f + ")") for f in fields]
    )
    for e in field_expr:
        for text in ["Lap(" + render(e) + ")", "Lap(Lap(" + render(e) + "))"]:
            g.add(text, "laplacian_macro")
        for h in coefficients:
            for j in range(len(coordinates)):
                transport = language.node("Multiply", h, language.axis(e, j, 1))
                g.add_expr(transport, "transport_component")
                for i in range(len(coordinates)):
                    g.add_expr(language.axis(transport, i, 1), "anisotropic_flux_component")
        if len(coordinates) == 2:
            text = (
                "PD("
                + render(e)
                + ",(2,0))*PD("
                + render(e)
                + ",(0,2))-PD("
                + render(e)
                + ",(1,1))**2"
            )
            g.add(text, "hessian_determinant")
        for f in field_expr:
            pieces = [
                language.node("Multiply", language.axis(e, j, 1), language.axis(f, j, 1))
                for j in range(len(coordinates))
            ]
            dot = pieces[0]
            for piece in pieces[1:]:
                dot = language.node("Add", dot, piece)
            g.add_expr(dot, "gradient_inner_product")
    snapshot()
    g.stage = "all_spatial_partials"
    derivative_inputs = list(g.candidates.values())
    for e in derivative_inputs:
        for alpha in multi_indices(len(coordinates), RECIPE["max_spatial_order"] - e.order):
            if meaningful_derivative(e, alpha, coordinates):
                g.add_expr(language.derivative(e, alpha), "all_partial_derivatives")
    snapshot()
    expanded = g.export()
    expanded["stage_statistics"] = stage_stats
    expanded["accepted_seed_roots"] = len(set(initial))
    expanded["llm_parent_suggestions"] = suggestions
    expanded["recipe_inputs"] = {
        "elementary_bases": len(elementary_base),
        "binary_bases": len(binary_base),
        "derivative_bases": len(derivative_inputs),
    }
    return (initial_graph, expanded)
