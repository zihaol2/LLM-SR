"""Symbolic support helpers only; no reference files or evaluation entry point."""

import sympy as s


def canonical(z):
    if z.is_Number:
        return ("constant",)
    if z.is_Symbol:
        return ("variable", str(z))
    if z.is_Mul:
        parts = tuple(sorted((canonical(t) for t in z.args if not t.is_Number), key=repr))
        return parts[0] if len(parts) == 1 else ("product", *parts)
    if z.is_Add:
        return ("sum", *tuple(sorted(set((canonical(t) for t in z.args)), key=repr)))
    if z.is_Pow:
        base, power = z.args
        return ("power", canonical(base), str(power) if power.is_Integer else "free_scale")
    return (z.func.__name__, *tuple((canonical(t) for t in z.args)))


def support(expression, language):
    coords = tuple((s.Symbol(c, real=True) for c in language.coordinates)) + (
        s.Symbol("t", real=True),
    )
    fields = {f: s.Function(f)(*coords) for f in language.fields}
    expression = s.expand(expression.doit())
    jets = {u: s.Symbol(f) for f, u in fields.items()}
    for derivative in expression.atoms(s.Derivative):
        field = next((f for f, u in fields.items() if derivative.expr == u), None)
        if field is None:
            continue
        suffix = "".join((str(v) * int(n) for v, n in derivative.variable_count))
        jets[derivative] = s.Symbol(field + "_" + suffix)
    expression = s.expand(expression.xreplace(jets))
    out = set()
    for term in s.Add.make_args(expression):
        coef, body = term.as_coeff_Mul()
        if abs(float(coef)) > 1e-10:
            out.add(repr(canonical(body)))
    return out


def symbolic_term(term, language):
    z = language.symbolic(language.parse(term["expr"]))
    z = z.subs({s.Symbol(p): s.Float(v, 15) for p, v in term["params"].items()})
    return s.Float(term["coefficient"], 15) * z
