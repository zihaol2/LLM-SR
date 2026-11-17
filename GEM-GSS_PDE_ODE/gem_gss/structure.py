"""Symbolic support helpers only; no reference files or evaluation entry point."""

import sympy as s


def canonical(z):
    if isinstance(z, s.Derivative):
        return ('PD', canonical(z.expr), tuple((str(v), int(n)) for v,n in z.variable_count))
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


def support(expression, language, threshold=1e-10):
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
        if abs(float(coef)) > threshold:
            out.add(repr(canonical(body)))
    return out


def symbolic_term(term, language):
    z = language.symbolic(language.parse(term["expr"]))
    z = z.subs({s.Symbol(p): s.Float(v, 15) for p, v in term["params"].items()})
    return s.Float(term["coefficient"], 15) * z


def normalize_operator(z):
    """Exact real-observation identities only. Never expand a chain rule."""
    if isinstance(z, s.Derivative):
        inside = normalize_operator(z.expr)
        variables = list(z.variable_count)
        if isinstance(inside, s.Derivative):
            variables = list(inside.variable_count)+variables
            inside = inside.expr
        if inside.func == s.log:
            coefficient, body = inside.args[0].as_coeff_Mul()
            if coefficient.is_positive and body.func == s.exp:
                inside = body.args[0]  # log(c exp(f)) = log(c)+f for real f,c>0.
        if inside.is_Add:
            wrt = {v for v,_ in variables}
            inside = s.Add(*(part for part in inside.args if part.free_symbols & wrt))
        if not any(v in inside.free_symbols for v,_ in variables):
            return s.Integer(0)
        return s.Derivative(inside, *variables, evaluate=False)
    if not z.args:
        return z
    result = z.func(*(normalize_operator(a) for a in z.args))
    if result.func == s.atanh and result.args[0].func == s.tanh:
        return result.args[0].args[0]  # Real observed fields; exact factor 1 only.
    return result


def operator_support(terms, language, threshold=0.):
    out = set()
    for term in terms:
        if abs(term['coefficient']) <= threshold:
            continue
        z = language.symbolic(language.parse(term['expr']))
        z = z.subs({s.Symbol(p):s.Float(v,15) for p,v in term['params'].items()})
        z = normalize_operator(z)
        if z != 0:
            out.add(repr(canonical(z)))
    return out


def reference_operator_support(text, language):
    # The top-level RHS sum defines reference mechanisms; derivative interiors
    # remain intact, even if nonlinear or internally additive.
    expression = language.symbolic(language.parse(text))
    return {repr(canonical(normalize_operator(body)))
            for term in s.Add.make_args(expression)
            for coefficient,body in [term.as_coeff_Mul()] if coefficient != 0 and body != 0}
