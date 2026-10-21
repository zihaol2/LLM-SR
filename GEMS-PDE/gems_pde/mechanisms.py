"""Shared finite expression grammar and pre-search DGM-inspired constructors."""

import ast
import re


def normalize(expr, fields):
    expr = expr.replace("^", "**").replace("d_dx(", "D(")
    for f in fields:
        expr = re.sub(r"\b" + f + r"_(x{1,4})\b", lambda m: f"D({f},{len(m.group(1))})", expr)
    expr = re.sub(r"d_axis\(([^,]+),\s*0,\s*([1-4])\)", r"D(\1,\2)", expr)
    tree = ast.parse(expr, mode="eval")
    names = set(fields) | {"x", "a", "b", "k", "np", "D"}
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Attribute,
        ast.Constant,
    )
    for n in ast.walk(tree):
        if not isinstance(n, allowed):
            raise ValueError("unsupported AST")
        if isinstance(n, ast.Name) and n.id not in names:
            raise ValueError("unknown symbol")
        if isinstance(n, ast.Attribute) and not (
            isinstance(n.value, ast.Name)
            and n.value.id == "np"
            and n.attr in {"sin", "cos", "tanh", "exp"}
        ):
            raise ValueError("unknown function")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            if (
                n.func.id != "D"
                or len(n.args) != 2
                or not isinstance(n.args[1], ast.Constant)
                or n.args[1].value not in {1, 2, 3, 4}
            ):
                raise ValueError("derivative signature")
    return ast.unparse(tree)


def derivative_order(expr, fields):
    def order(n):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "D":
            return order(n.args[0]) + int(n.args[1].value)
        return max([order(c) for c in ast.iter_child_nodes(n)] or [0])

    return order(ast.parse(expr, mode="eval"))


class Graph:
    def __init__(self, fields):
        self.fields = fields
        self.nodes = []
        self.seen = {}
        self.rejected = []

    def add(self, expr, group, parents=(), layer="DGM"):
        try:
            expr = normalize(expr, self.fields)
            if len(expr) > 240 or derivative_order(expr, self.fields) > 4:
                raise ValueError("complexity/order cap")
            if expr in self.seen:
                return self.seen[expr]
            names = {
                n.id for n in ast.walk(ast.parse(expr, mode="eval")) if isinstance(n, ast.Name)
            }
            params = sorted(names & {"a", "b", "k"})
            mid = f"M{len(self.nodes):04d}"
            node = {
                "id": mid,
                "expr": expr,
                "group": group,
                "parents": list(parents),
                "layer": layer,
                "params": params,
                "complexity": sum(
                    isinstance(n, (ast.BinOp, ast.Call))
                    for n in ast.walk(ast.parse(expr, mode="eval"))
                ),
            }
            self.nodes.append(node)
            self.seen[expr] = mid
            return mid
        except Exception as e:
            self.rejected.append({"expr": expr, "reason": str(e)})
            return None


def grow_graph(graph):
    """Uniform small constructor closure, independent of task identity and truth."""
    fields = graph.fields
    roots = {f: graph.add(f, "power", layer="L0") for f in fields}
    for f in fields:
        for p in range(1, 6):
            base = graph.add(f"{f}**{p}", "power", [roots[f]])
            if p <= 3:
                for order in range(1, 5):
                    graph.add(f"D({f}**{p},{order})", "differentiate", [base])
        for fun in ("sin", "cos", "tanh", "exp"):
            graph.add(f"np.{fun}(a*{f})", "local_transform", [roots[f]])
        for denominator in (f"a+{f}", f"a+{f}**2"):
            graph.add(f"{f}/({denominator})", "local_transform", [roots[f]])
        for other in fields:
            product = graph.add(f"{f}*{other}", "power", [roots[f], roots[other]])
            gradient = graph.add(f"D({f},1)", "differentiate", [roots[f]])
            for denominator in (other, f"{other}**2", f"a+{other}", f"1+a*{other}**2"):
                div = graph.add(f"D({f},1)/({denominator})", "quotient", [gradient, roots[other]])
                graph.add(f"D(D({f},1)/({denominator}),1)", "quotient", [div])
            graph.add(f"D({f}*D({other},1),1)", "product_flux", [roots[f], roots[other]])
    for f in ["1"] + fields:
        powers = [1] if f == "1" else [1, 2, 3]
        for p in powers:
            base = f"({f})**{p}"
            for modulation in [
                "np.sin(k*x)",
                "np.cos(k*x)",
                "np.tanh(k*x)",
                "np.exp(-k*x**2)",
                "np.exp(k*x)",
            ]:
                mid = graph.add(f"({base})*({modulation})", "modulate")
                if f != "1":
                    graph.add(f"D(({base})*({modulation}),1)", "modulate", [mid])
    # General factored state responses and affine spatial responses. The same
    # rule applies to every field, power, function and task; no reference test.
    for f in fields:
        for p in range(1, 4):
            response = f"{f}*(1-a*{f}**{p})"
            rid = graph.add(response, "power", [roots[f]])
            graph.add(f"D({response},1)", "differentiate", [rid])
            for modulation in [
                "np.sin(k*x)",
                "np.cos(k*x)",
                "np.tanh(k*x)",
                "np.exp(-k*x**2)",
                "np.exp(k*x)",
            ]:
                for factor in [modulation, f"(1-b*({modulation}))"]:
                    mid = graph.add(f"({response})*({factor})", "modulate", [rid])
                    graph.add(f"D(({response})*({factor}),1)", "modulate", [mid])
