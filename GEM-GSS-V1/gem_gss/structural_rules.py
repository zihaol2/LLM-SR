"""Reference-free structural proposal rules; never fit or select a final model."""

from dataclasses import replace

import sympy as s

from . import core as C

ZERO_WRAPPERS = {"ArcSinh", "ArcTanh", "Sinh", "Tanh", "ArcTan", "ArcSin", "Sin"}


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
