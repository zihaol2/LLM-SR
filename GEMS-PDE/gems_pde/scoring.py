"""Post-freeze reference scoring only. Never imported by discovery modules."""

import ast
import sympy as s
from sympy.printing.latex import LatexPrinter

X = s.Symbol("x", real=True)
U = {f: s.Function(f)(X) for f in ("u", "v")}
EPS = 1e-10


class EquationPrinter(LatexPrinter):
    def _print_Derivative(self, expr):
        order = len(expr.variables)
        op = r"\partial_x" if order == 1 else r"\partial_x^{" + str(order) + "}"
        return op + r"\!\left[" + self._print(expr.expr) + r"\right]"

    def _print_Function(self, expr, exp=None):
        if expr.func.__name__ in {"u", "v"} and expr.args == (X,):
            name = expr.func.__name__
            return name if exp is None else name + "^{" + str(exp) + "}"
        return super()._print_Function(expr, exp=exp)


def parse(expr, params=None, expand_derivatives=True):
    params = params or {}

    def walk(n):
        if isinstance(n, ast.Expression):
            return walk(n.body)
        if isinstance(n, ast.Name):
            if n.id == "x":
                return X
            if n.id in U:
                return U[n.id]
            if n.id in params:
                return s.Float(params[n.id], 15)
            raise ValueError("Unknown identifier " + n.id)
        if isinstance(n, ast.Constant):
            return s.Rational(str(n.value))
        if isinstance(n, ast.UnaryOp):
            return -walk(n.operand) if isinstance(n.op, ast.USub) else walk(n.operand)
        if isinstance(n, ast.BinOp):
            l, r = walk(n.left), walk(n.right)
            return {
                ast.Add: lambda: l + r,
                ast.Sub: lambda: l - r,
                ast.Mult: lambda: l * r,
                ast.Div: lambda: l / r,
                ast.Pow: lambda: l**r,
            }[type(n.op)]()
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name) and n.func.id in {"D", "d_dx", "d_axis"}:
                order = int(n.args[-1].value)
                return (
                    s.diff(walk(n.args[0]), X, order)
                    if expand_derivatives
                    else s.Derivative(walk(n.args[0]), (X, order), evaluate=False)
                )
            fun = n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
            return {"sin": s.sin, "cos": s.cos, "tanh": s.tanh, "exp": s.exp}[fun](walk(n.args[0]))
        raise ValueError(type(n).__name__)

    return walk(ast.parse(expr, mode="eval"))


def canonical(z):
    """Coefficient-independent shape; integers used as powers stay exact."""
    if z.is_Number:
        return ("constant",)
    if z.is_Symbol:
        return ("variable", str(z))
    if z.is_Mul:
        parts = tuple(sorted((canonical(t) for t in z.args if not t.is_Number), key=repr))
        return parts[0] if len(parts) == 1 else ("product", *parts)
    if z.is_Add:
        return ("sum", *tuple(sorted(set(canonical(t) for t in z.args), key=repr)))
    if z.is_Pow:
        base, power = z.args
        exponent = str(power) if power.is_Integer else "free_scale"
        return ("power", canonical(base), exponent)
    return (z.func.__name__, *tuple(canonical(t) for t in z.args))


def support(expr):
    jets = {}
    for f, u in U.items():
        for n in range(1, 13):
            jets[s.diff(u, X, n)] = s.Symbol(f + "_" + "x" * n)
        jets[u] = s.Symbol(f)
    expression = s.expand(expr).xreplace(jets)
    terms = s.Add.make_args(s.expand(expression))
    pairs = [t.as_coeff_Mul() for t in terms]
    result = set()
    for coef, body in pairs:
        # Absolute arithmetic tolerance, not a threshold relative to the largest
        # coefficient: unrelated, badly scaled atoms must not erase valid terms.
        if abs(float(coef)) <= EPS:
            continue
        result.add(repr(canonical(body)))
    return result
