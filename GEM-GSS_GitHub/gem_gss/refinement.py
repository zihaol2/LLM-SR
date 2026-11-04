"""Terminal-only deletion, polynomial blocks and existing graph-core comparisons."""

import copy
from dataclasses import replace
import numpy as np
import sympy as s
from . import core as C
from .graph import Registry
from .fitting import fit


class TargetView:

    def __init__(self, data, field):
        self.y = {"u": data.y[field]}
        self.scale = {"u": data.scale[field]}
        self.feature = data.feature
        self.fit_calls = 0
        self.fit_nfev = 0


def additive_deletions(expression, language):
    if expression.op in {"Add", "Subtract"}:
        yield expression.args[0]
        yield (
            language.node("Negate", expression.args[1])
            if expression.op == "Subtract"
            else expression.args[1]
        )
    positions = {
        "Add": (0, 1),
        "Subtract": (0, 1),
        "Multiply": (0, 1),
        "Divide": (0,),
        "PD": (0,),
        "Negate": (0,),
    }
    for index in positions.get(expression.op, ()):
        for reduced in additive_deletions(expression.args[index], language):
            args = list(expression.args)
            args[index] = reduced
            yield replace(expression, args=tuple(args))


def deletion_proposals(model, registry):
    from .core import render

    terms = model["terms"]
    seen = set()
    for index, term in enumerate(terms):
        rest = [t for i, t in enumerate(terms) if i != index]
        options = [(None, "delete_mechanism")]
        expression = registry.graph.language.parse(term["expr"])
        options.extend(
            (
                (render(e), "delete_additive_component")
                for e in additive_deletions(expression, registry.graph.language)
            )
        )
        for text, kind in options:
            new_id = None
            if text is not None:
                try:
                    new_id, _ = registry.propose(text, 21, dict(kind="post_selection_prune"))
                except (ValueError, KeyError, SyntaxError):
                    continue
            ids = sorted({t["id"] for t in rest} | ({new_id} if new_id else set()))
            signature = tuple(ids)
            if signature in seen or signature == tuple(sorted((t["id"] for t in terms))):
                continue
            seen.add(signature)
            nodes = [registry.nodes[mid] for mid in ids]
            if C.complexity(nodes, registry) >= model["complexity"]:
                continue
            initial = copy.deepcopy(model)
            if new_id:
                initial["terms"].append(dict(id=new_id, params=term["params"]))
            yield dict(
                kind=kind,
                removed_or_simplified=term["id"],
                before_expression=term["expr"],
                after_expression=registry.nodes[new_id]["expr"] if new_id else None,
                nodes=nodes,
                initial=initial,
            )


def reduce_model(data, model, registry):
    initial = copy.deepcopy(model)
    baseline = fit(
        data, "u", [registry.nodes[t["id"]] for t in initial["terms"]], registry, initial=initial
    )
    current = baseline if baseline and baseline["score"] < initial["score"] else initial
    trials = []
    accepted = []
    iteration = 0
    while True:
        iteration += 1
        round_trials = []
        for proposal in deletion_proposals(current, registry):
            fitted = fit(data, "u", proposal["nodes"], registry, initial=proposal["initial"])
            row = {k: v for k, v in proposal.items() if k not in {"nodes", "initial"}}
            row.update(
                iteration=iteration,
                parent_MSE=current["fit_mse"],
                parent_complexity=current["complexity"],
                parent_score=current["score"],
                valid=fitted is not None,
            )
            if fitted:
                row.update(
                    MSE=fitted["fit_mse"],
                    MSE_ratio=fitted["fit_mse"] / current["fit_mse"],
                    complexity=fitted["complexity"],
                    score=fitted["score"],
                    model=fitted,
                )
                round_trials.append(row)
            trials.append(row)
        better = [r for r in round_trials if r["score"] < current["score"]]
        if not better:
            break
        chosen = min(better, key=lambda r: (r["score"], r["complexity"], r["MSE"]))
        accepted.append(chosen)
        current = chosen["model"]
    return dict(
        original=initial,
        baseline_refit=baseline,
        pruned=current,
        trials=trials,
        accepted=accepted,
        full_fits=data.fit_calls,
        reference_access_for_decisions=False,
    )


def reaction_block(model, registry, field):
    variable = s.Symbol(field, real=True)
    block = []
    has_nonpolynomial = False
    vanishes = True
    for term in model["terms"]:
        node = registry.nodes[term["id"]]
        value = registry.info(term["id"]).value
        if node["order"] != 0 or value.free_symbols & registry.analyzer.physical != {variable}:
            continue
        block.append(term)
        has_nonpolynomial |= not bool(value.is_polynomial(variable))
        vanishes &= value.subs(variable, 0) == 0
    return (block, has_nonpolynomial, vanishes)


def graph_powers(registry, field, include_constant):
    variable = s.Symbol(field, real=True)
    powers = {}
    for mid in sorted(registry.graph.candidates):
        node = registry.nodes[mid]
        if node["order"] != 0 or node["params"]:
            continue
        value = registry.info(mid).value
        if value.free_symbols - registry.analyzer.physical or value.free_symbols - {variable}:
            continue
        if not value.is_polynomial(variable):
            continue
        polynomial = s.Poly(value, variable)
        if len(polynomial.terms()) != 1:
            continue
        (exponent,), coefficient = polynomial.terms()[0]
        if not coefficient or (exponent == 0 and (not include_constant)):
            continue
        key = (C.complexity([node], registry), len(node["expr"]), mid)
        if exponent not in powers or key < powers[exponent][0]:
            powers[exponent] = (key, node)
    return [(exponent, item[1]) for exponent, item in sorted(powers.items())]


ZERO_WRAPPERS = {"ArcSinh", "ArcTanh", "Sinh", "Tanh", "ArcTan", "ArcSin", "Sin"}


def fixed_core(value, physical):
    factor, body = value.as_independent(*physical, as_Add=False)
    if body == 1 or body.free_symbols - physical:
        return None
    return body


class GraphCores:

    def __init__(self, registry):
        self.registry = registry
        self.physical = set(registry.analyzer.physical)
        self.lookup = {}
        for mid in sorted(registry.graph.candidates):
            node = registry.nodes[mid]
            if node["params"]:
                continue
            value = registry.info(mid).value
            core = fixed_core(value, self.physical)
            if core is None:
                continue
            key = s.srepr(s.expand(core))
            rank = (C.complexity([node], registry), len(node["expr"]), mid)
            if key not in self.lookup or rank < self.lookup[key][0]:
                self.lookup[key] = (rank, mid)

    def find(self, value):
        core = fixed_core(value, self.physical)
        if core is None:
            return None
        entry = self.lookup.get(s.srepr(s.expand(core)))
        return entry[1] if entry else None


def core_proposals(term, registry, index):
    expression = registry.graph.language.parse(term["expr"])
    physical = index.physical
    if expression.op in ZERO_WRAPPERS:
        inside = registry.analyzer.analyze(expression.args[0]).value
        mid = index.find(inside)
        if mid:
            yield (mid, "nonlinear_wrapper_to_existing_core")
    elif expression.op == "Divide":
        numerator = registry.analyzer.analyze(expression.args[0]).value
        denominator = registry.analyzer.analyze(expression.args[1]).value
        if not numerator.free_symbols - physical and denominator.free_symbols - physical:
            mid = index.find(numerator)
            if mid:
                yield (mid, "parameter_modulated_quotient_to_existing_core")
    elif expression.op == "Multiply":
        value = registry.info(term["id"]).value
        factors = s.Mul.make_args(value)
        fixed = [v for v in factors if not v.free_symbols - physical]
        variable = [v for v in factors if v.free_symbols - physical]
        if fixed and variable:
            mid = index.find(s.Mul(*fixed))
            if mid:
                yield (mid, "parameter_modulated_product_to_existing_core")


def refine_core(data, registry, field, model, index=None):
    index = index or GraphCores(registry)
    view = TargetView(data, field)
    original = copy.deepcopy(model)
    current = original
    trials = []
    accepted = []
    iteration = 0
    while True:
        iteration += 1
        seen = set()
        candidates = []
        for i, term in enumerate(current["terms"]):
            for mid, rule in core_proposals(term, registry, index):
                support = tuple(
                    sorted({t["id"] for j, t in enumerate(current["terms"]) if j != i} | {mid})
                )
                if support in seen or support == tuple(sorted((t["id"] for t in current["terms"]))):
                    continue
                seen.add(support)
                nodes = [registry.nodes[x] for x in support]
                if C.complexity(nodes, registry) >= current["complexity"]:
                    continue
                fitted = fit(view, "u", nodes, registry, initial=current)
                row = dict(
                    iteration=iteration,
                    kind="replace_existing_subexpression",
                    rule=rule,
                    removed_or_simplified=term["id"],
                    replacement_id=mid,
                    before_expression=term["expr"],
                    after_expression=registry.nodes[mid]["expr"],
                    parent_MSE=current["fit_mse"],
                    parent_complexity=current["complexity"],
                    parent_score=current["score"],
                    valid=fitted is not None,
                )
                if fitted:
                    row.update(
                        model=fitted,
                        MSE=fitted["fit_mse"],
                        complexity=fitted["complexity"],
                        score=fitted["score"],
                        MSE_ratio=fitted["fit_mse"] / max(current["fit_mse"], 1e-300),
                    )
                    candidates.append(row)
                trials.append(row)
        better = [row for row in candidates if row["score"] < current["score"]]
        if not better:
            break
        selected = min(better, key=lambda r: (r["score"], r["complexity"], r["MSE"]))
        accepted.append(selected)
        current = selected["model"]
    return dict(
        original=original,
        pruned=current,
        trials=trials,
        accepted=accepted,
        full_fits=view.fit_calls,
        reference_access=False,
        rule="Compare existing graph cores; jointly refit unchanged fitter; lower complexity and original Score; originals remain eligible",
    )
