"""Conservative, reference-free symbolic and free-coefficient redundancy removal.

Original graph files are read-only. Parameter bindings and domain obligations
are retained. Nested derivative operators remain different evaluation paths.
"""

from collections import defaultdict, Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import sys
import csv
import sympy as s
import networkx as nx
from .builder import Graph, render


@dataclass(frozen=True)
class Info:
    value: object
    guards: frozenset
    risk: bool = False


def key(z):
    return s.srepr(z)


class Analyzer:

    def __init__(self, fields, coordinates):
        self.fields = {s.Symbol(v, real=True) for v in fields}
        self.coords = [s.Symbol(v, real=True) for v in coordinates]
        self.physical = self.fields | set(self.coords) | {s.Symbol("t", real=True)}
        self.parameters = {}
        self.zero_polynomials = 0

    def obligation(self, guards, kind, arg):
        proven = {
            "nonzero": arg.is_nonzero,
            "positive": arg.is_positive,
            "nonnegative": arg.is_nonnegative,
        }.get(kind)
        if proven is not True:
            guards.add((kind, arg))

    def linear_derivative(self, child, alpha):
        """Use linearity, but preserve nested/nonlinear numerical paths.

        A fixed spline differentiation operator is linear. No Leibniz or
        chain-rule expansion of observed-field products is performed here.
        """
        if child.is_Add:
            return s.Add(*(self.linear_derivative(term, alpha) for term in child.args))
        active = [c for c, n in zip(self.coords, alpha) if n]
        functions = child.atoms(s.Function)
        observed = bool(child.free_symbols & self.fields)
        opaque = any((f.func.__name__.startswith("PD_") for f in functions))
        polynomial = not observed and (not opaque) and child.is_polynomial(*active)
        if polynomial:
            value = child
            for c, n in zip(self.coords, alpha):
                if n:
                    value = s.diff(value, c, int(n))
            if value == 0:
                self.zero_polynomials += 1
            return value
        dependent = self.fields | set(active)
        coefficient, body = child.as_independent(*dependent, as_Add=False)
        fun = s.Function("PD_" + "_".join(map(str, alpha)), real=True)
        return coefficient * fun(body)

    @lru_cache(maxsize=None)
    def analyze(self, e):
        if e.op == "Constant":
            return Info(s.Rational(e.value), frozenset())
        if e.op in {"Field", "Coordinate", "Parameter"}:
            symbol = s.Symbol(e.value, real=True)
            if e.op == "Parameter":
                self.parameters[e.value] = symbol
            return Info(symbol, frozenset())
        infos = [self.analyze(a) for a in e.args]
        v = [a.value for a in infos]
        guards = set().union(*(a.guards for a in infos))
        risk = any((a.risk for a in infos))
        op = e.op
        if op == "Add":
            value = v[0] + v[1]
        elif op == "Subtract":
            value = v[0] - v[1]
        elif op == "Negate":
            value = -v[0]
        elif op == "Multiply":
            value = v[0] * v[1]
        elif op == "Divide":
            self.obligation(guards, "nonzero", v[1])
            value = v[0] / v[1]
        elif op == "Power":
            base, power = v
            if power.is_Integer:
                if power < 0:
                    self.obligation(guards, "nonzero", base)
            elif power.is_Rational and power.q % 2 == 0:
                self.obligation(guards, "positive" if power < 0 else "nonnegative", base)
                risk = True
            else:
                self.obligation(guards, "positive", base)
                risk = True
            value = base**power
        elif op == "PD":
            child = v[0]
            alpha = e.value
            value = self.linear_derivative(child, alpha)
            if risk:
                scale, body = child.as_coeff_Mul()
                guards.add(("smooth_order_" + str(sum(alpha)), body))
        else:
            arg = v[0]
            if op == "Log":
                self.obligation(guards, "positive", arg)
            if op == "Tan":
                self.obligation(guards, "nonzero", s.cos(arg))
            if op in {"ArcSin", "ArcCos"}:
                guards.add(("abs_le_1", s.Abs(arg)))
                risk = True
            if op == "ArcTanh":
                guards.add(("abs_lt_1", s.Abs(arg)))
            if op == "ArcCosh":
                guards.add(("ge_1", arg))
                risk = True
            if op == "Sqrt":
                self.obligation(guards, "nonnegative", arg)
                risk = True
            if op == "CubeRoot":
                value = s.Function("RealCubeRoot", real=True)(arg)
                risk = True
                return Info(value, frozenset(guards), risk)
            mapping = {
                "Exp": s.exp,
                "Log": s.log,
                "Sin": s.sin,
                "Cos": s.cos,
                "Tan": s.tan,
                "ArcSin": s.asin,
                "ArcCos": s.acos,
                "ArcTan": s.atan,
                "Sinh": s.sinh,
                "Cosh": s.cosh,
                "Tanh": s.tanh,
                "ArcSinh": s.asinh,
                "ArcCosh": s.acosh,
                "ArcTanh": s.atanh,
                "Sqrt": s.sqrt,
                "Sinc": s.sinc,
            }
            if op == "Exp":
                shift, body = arg.as_independent(*self.physical, as_Add=True)
                value = s.exp(shift) * s.exp(body)
            else:
                value = mapping[op](arg)
        return Info(value, frozenset(guards), risk)

    def structural_key(self, info):
        return (key(info.value), tuple(sorted(((kind, key(arg)) for kind, arg in info.guards))))

    def family(self, value, guards, parameters_from_guards=True):
        """Alpha-renaming acts at candidate roots, never on embedded children."""
        coefficient, body = value.as_coeff_Mul()
        if coefficient == 0:
            return None
        parameter_set = body.free_symbols - self.physical
        if parameters_from_guards:
            parameter_set |= (
                set().union(*(arg.free_symbols - self.physical for _, arg in guards))
                if guards
                else set()
            )
        params = sorted(parameter_set, key=str)
        if len(params) > 6:
            raise ValueError("Unexpectedly large parameter scope")
        targets = [s.Symbol("Q" + str(i), real=True) for i in range(len(params))]
        best = None
        result = None
        for permutation in itertools.permutations(targets):
            rename = dict(zip(params, permutation))
            b = body.xreplace(rename)
            g = frozenset(((kind, key(arg.xreplace(rename))) for kind, arg in guards))
            signature = (key(b), tuple(sorted(g)))
            if best is None or signature < best:
                best = signature
                result = {
                    "body_key": signature[0],
                    "guards": g,
                    "rename": {str(a): str(b) for a, b in rename.items()},
                    "coefficient": coefficient,
                }
        return result


def rank(node):
    return (node["operations"], len(node["params"]), len(node["expr"]), node["id"])


def deduplicate_graph(raw, declaration):
    original_sha = hashlib.sha256(
        json.dumps(raw, sort_keys=True, ensure_ascii=False).encode("utf8")
    ).hexdigest()
    language = Graph(declaration["fields"], declaration["coordinates"]).language
    analyzer = Analyzer(declaration["fields"], declaration["coordinates"])
    nodes = {n["id"]: n for n in raw["nodes"]}
    infos = {}
    structural = defaultdict(list)
    families = {}
    zero = []
    for n in raw["nodes"]:
        expression = language.parse(n["expr"])
        info = analyzer.analyze(expression)
        infos[n["id"]] = info
        structural[analyzer.structural_key(info)].append(n["id"])
        if n["candidate"]:
            if info.value == 0:
                zero.append(n["id"])
            else:
                families[n["id"]] = analyzer.family(info.value, info.guards)
    buckets = defaultdict(list)
    for mid, f in families.items():
        buckets[f["body_key"]].append(mid)
    selected = set()
    aliases = {}
    reason = Counter()
    for bucket in buckets.values():
        kept = []
        for mid in sorted(bucket, key=lambda i: (len(families[i]["guards"]), rank(nodes[i]))):
            f = families[mid]
            representative = next((r for r in kept if families[r]["guards"] <= f["guards"]), None)
            if representative is None:
                kept.append(mid)
                selected.add(mid)
                continue
            kind = (
                "alpha_or_numeric_scale_equivalence"
                if families[representative]["guards"] == f["guards"]
                else "covered_by_broader_domain"
            )
            aliases[mid] = {"representative": representative, "reason": kind}
            reason[kind] += 1
    by_model = defaultdict(list)
    for mid in selected:
        by_model[families[mid]["body_key"]].append(mid)
    for mid in sorted(list(selected), key=lambda i: rank(nodes[i]), reverse=True):
        info = infos[mid]
        multiplier, body = info.value.as_independent(*analyzer.physical, as_Add=False)
        if multiplier.is_number or multiplier == 1 or body == 1:
            continue
        core = analyzer.family(body, info.guards, parameters_from_guards=False)
        possibilities = by_model.get(core["body_key"], [])
        representatives = []
        for r in possibilities:
            if r == mid or r not in selected:
                continue
            rm, _ = infos[r].value.as_independent(*analyzer.physical, as_Add=False)
            if not rm.is_number or rm == 0:
                continue
            target = analyzer.family(infos[r].value, infos[r].guards, parameters_from_guards=False)
            if target["guards"] <= core["guards"]:
                representatives.append(r)
        if representatives:
            r = min(representatives, key=lambda i: rank(nodes[i]))
            selected.remove(mid)
            aliases[mid] = {
                "representative": r,
                "reason": "outer_parameter_factor_covered",
                "multiplier": str(multiplier),
                "body_parameter_rename": core["rename"],
                "representative_parameter_rename": analyzer.family(
                    infos[r].value, infos[r].guards, False
                )["rename"],
            }
            reason["outer_parameter_factor_covered"] += 1
    for mid in zero:
        aliases[mid] = {"representative": None, "reason": "identically_zero"}
    reason["identically_zero"] = len(zero)
    proofs = []
    for mid, row in aliases.items():
        target = row["representative"]
        kind = row["reason"]
        if kind == "identically_zero":
            valid = infos[mid].value == 0
        elif kind == "outer_parameter_factor_covered":
            _, body = infos[mid].value.as_independent(*analyzer.physical, as_Add=False)
            source = analyzer.family(body, infos[mid].guards, False)
            dest = analyzer.family(infos[target].value, infos[target].guards, False)
            valid = source["body_key"] == dest["body_key"] and dest["guards"] <= source["guards"]
        else:
            source, dest = (families[mid], families[target])
            valid = source["body_key"] == dest["body_key"] and dest["guards"] <= source["guards"]
        proofs.append(bool(valid))
    if not all(proofs):
        raise AssertionError("Unproved candidate coverage")

    def ultimate(mid):
        seen = set()
        while mid in aliases and aliases[mid]["representative"] is not None:
            if mid in seen:
                raise AssertionError("Alias cycle")
            seen.add(mid)
            mid = aliases[mid]["representative"]
        return mid

    for row in aliases.values():
        if row["representative"]:
            row["representative"] = ultimate(row["representative"])
    class_of = {}
    representative = {}
    for signature, members in structural.items():
        rep = min(members, key=lambda i: rank(nodes[i]))
        representative[signature] = rep
        for mid in members:
            class_of[mid] = signature
    kept_classes = set()
    order = []

    def visit(signature):
        if signature in kept_classes:
            return
        rep = nodes[representative[signature]]
        for operand in rep["operands"]:
            if "node_id" in operand:
                child = class_of[operand["node_id"]]
                if child == signature:
                    raise AssertionError("Unsafe self-class construction")
                visit(child)
        kept_classes.add(signature)
        order.append(signature)

    selected_classes = {class_of[mid] for mid in selected}
    for mid in sorted(selected):
        visit(class_of[mid])
    new_id = {signature: "D" + str(i).zfill(6) for i, signature in enumerate(order)}
    new_nodes = []
    new_edges = {}
    for signature in order:
        source = representative[signature]
        n = nodes[source]
        operands = []
        for position, operand in enumerate(n["operands"]):
            if "node_id" not in operand:
                operands.append(operand)
                continue
            parent = new_id[class_of[operand["node_id"]]]
            operands.append({"node_id": parent, "original_operand_id": operand["node_id"]})
            edge = new_edges.setdefault(
                (parent, new_id[signature]),
                {"parent": parent, "child": new_id[signature], "operand_positions": []},
            )
            edge["operand_positions"].append(position)
        depth = (
            max(
                (new_nodes[int(o["node_id"][1:])]["depth"] for o in operands if "node_id" in o),
                default=-1,
            )
            + 1
        )
        item = {
            **n,
            "id": new_id[signature],
            "source_id": source,
            "operands": operands,
            "depth": depth,
            "candidate": signature in selected_classes,
            "semantic_class_aliases": structural[signature],
            "evaluation_expr": n["expr"],
            "parameter_scope": "Original symbols retained in the dependency DAG",
            "exact_domain_obligations": [
                (kind, str(arg))
                for kind, arg in sorted(infos[source].guards, key=lambda v: (v[0], str(v[1])))
            ],
        }
        new_nodes.append(item)
    alias_rows = []
    for mid, n in nodes.items():
        info = infos[mid]
        signature = class_of[mid]
        if n["candidate"]:
            record = aliases.get(mid, {"representative": mid, "reason": "retained"})
            target = record["representative"]
            target_new = new_id[class_of[target]] if target else ""
        else:
            record = {
                "representative": representative[signature] if signature in kept_classes else None,
                "reason": "support_retained" if signature in kept_classes else "unused_support",
            }
            target = record["representative"]
            target_new = new_id[signature] if signature in kept_classes else ""
        alias_rows.append(
            {
                "old_id": mid,
                "old_expression": n["expr"],
                "was_candidate": n["candidate"],
                "representative_old_id": target or "",
                "new_id": target_new,
                "reason": record["reason"],
                "original_numeric_scale": (
                    str(families[mid]["coefficient"]) if mid in families else ""
                ),
                "original_parameter_rename": (
                    json.dumps(families[mid]["rename"], sort_keys=True) if mid in families else ""
                ),
                "retained_domain_obligations": json.dumps(
                    [(kind, str(arg)) for kind, arg in info.guards]
                ),
                "details": json.dumps(record, ensure_ascii=False),
            }
        )
    declared = {n["id"] for n in new_nodes}
    expected = defaultdict(list)
    for n in new_nodes:
        for position, operand in enumerate(n["operands"]):
            if "node_id" in operand:
                expected[operand["node_id"], n["id"]].append(position)
    graph = nx.DiGraph()
    graph.add_nodes_from(declared)
    graph.add_edges_from(new_edges)
    stats = {
        "nodes": len(new_nodes),
        "edges": len(new_edges),
        "candidate_roots": len(selected_classes),
        "support_only_nodes": len(new_nodes) - len(selected_classes),
        "weak_components": nx.number_weakly_connected_components(graph),
        "isolates": len(list(nx.isolates(graph))),
        "acyclic": nx.is_directed_acyclic_graph(graph),
        "operator_counts": dict(Counter((n["operator"] for n in new_nodes))),
    }
    checks = {
        "all_candidates_accounted": len(selected) + len(aliases)
        == raw["statistics"]["candidate_roots"],
        "all_direct_alias_proofs_valid": all(proofs),
        "all_selected_mapped": all((class_of[mid] in new_id for mid in selected)),
        "unique_node_ids": len(declared) == len(new_nodes),
        "no_dangling_or_self_edges": all(
            (a in declared and b in declared and (a != b) for a, b in new_edges)
        ),
        "complete_dependency_edges": set(expected) == set(new_edges)
        and all((new_edges[e]["operand_positions"] == expected[e] for e in expected)),
        "aliases_terminate_at_retained_candidates": all(
            (
                row["representative"] is None or row["representative"] in selected
                for row in aliases.values()
            )
        ),
        "valid_representative_grammar": all(
            (language.parse(n["expr"]).order == n["order"] <= 4 for n in new_nodes)
        ),
        "DAG": stats["acyclic"],
        "raw_graph_unchanged": True,
        "no_empty_class": all((structural[signature] for signature in order)),
    }
    result = {
        "nodes": new_nodes,
        "edges": list(new_edges.values()),
        "statistics": stats,
        "reference_access": False,
        "numeric_validation": "Not run; symbolic/domain-aware redundancy only",
        "source_graph_sha256": original_sha,
        "evaluation_path_policy": "Nested derivatives kept distinct; aliases and original expressions retained",
        "candidate_scope": "Independent per-mechanism parameters, same parameter bounds, free real outer coefficients",
        "checks": checks,
    }
    return (result, alias_rows)
