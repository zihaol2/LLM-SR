"""AST-only conversion of LLM equation functions to fitted basis groups.

Generated functions are parsed, never executed as Python programs.
"""

import ast
import copy
import json
import re
import textwrap
from . import mechanisms
from .vendor.llmpdesr import buffer, code_manipulation
from .vendor.llmpdesr.config import ExperienceBufferConfig


def param_index(node):
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "params"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, int)
    ):
        if 0 <= node.slice.value < 64:
            return node.slice.value
    return None


class Inline(ast.NodeTransformer):
    def __init__(self, locals):
        self.locals = locals

    def visit_Name(self, n):
        return copy.deepcopy(self.locals.get(n.id, n))

    def visit_Call(self, n):
        n = self.generic_visit(n)
        if isinstance(n.func, ast.Name) and n.func.id in {"d_dx", "D", "d_axis"}:
            if not n.args:
                raise ValueError("missing derivative operand")
            order = n.args[-1] if len(n.args) > 1 else ast.Constant(1)
            for kw in n.keywords:
                if kw.arg == "order":
                    order = kw.value
            return ast.Call(ast.Name("D", ast.Load()), [n.args[0], order], [])
        return n


def extract(raw, fields):
    text = raw.strip()
    if "```" in text:
        blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
        if blocks:
            text = max(blocks, key=len)
        else:
            text = "\n".join(
                line for line in text.splitlines() if not line.strip().startswith("```")
            )
    text = textwrap.dedent(text).strip()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = ast.parse(
            "def equation():\n" + "\n".join("    " + line for line in text.splitlines())
        )
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if funcs:
        func = next((n for n in funcs if n.name.startswith("equation")), funcs[0])
        statements = func.body
    else:
        statements = tree.body
    locals = {}
    returned = None
    for statement in statements:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            name = statement.targets[0].id
            if name in {"u", "v", "x", "params", "D", "d_dx", "np"}:
                raise ValueError("cannot overwrite input")
            locals[name] = Inline(locals).visit(copy.deepcopy(statement.value))
        elif isinstance(statement, ast.Return):
            returned = Inline(locals).visit(copy.deepcopy(statement.value))
            break
        else:
            raise ValueError("Only straight-line mathematical assignments and return are allowed")
    if returned is None:
        values = [locals.get(f + "_t_pred") for f in fields]
        if any(v is None for v in values):
            raise ValueError("missing return")
    else:
        values = list(returned.elts) if isinstance(returned, (ast.Tuple, ast.List)) else [returned]
    if len(values) != len(fields):
        raise ValueError("wrong number of outputs")
    equations = {}
    shared_outputs = set()
    for field, value in zip(fields, values):
        groups = {}

        def additive(node, sign=1):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                return additive(node.left, sign) + additive(node.right, sign)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
                return additive(node.left, sign) + additive(node.right, -sign)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return additive(node.operand, -sign)
            return [(node, sign)]

        def factors(node):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                return factors(node.left) + factors(node.right)
            return [node]

        for term, sign in additive(value):
            fs = factors(term)
            outside = [j for j, n in enumerate(fs) if param_index(n) is not None]
            position = outside[0] if outside else None
            outer = param_index(fs[position]) if position is not None else "fixed"
            rest = [v for j, v in enumerate(fs) if j != position]
            base = ast.Constant(sign)
            for factor in rest:
                base = ast.BinOp(base, ast.Mult(), factor)
            groups.setdefault(outer, []).append(base)
        if len(groups) > 8:
            raise ValueError("more than eight outer-coefficient groups")
        all_internal = set()
        nodes = []
        for outer, bases in groups.items():
            expression = bases[0]
            for b in bases[1:]:
                expression = ast.BinOp(expression, ast.Add(), b)
            internal = {param_index(n) for n in ast.walk(expression) if param_index(n) is not None}
            if internal & set(groups):
                raise ValueError("same parameter used internally and as an outer coefficient")
            all_internal |= internal

            class Params(ast.NodeTransformer):
                def visit_Subscript(self, n):
                    index = param_index(n)
                    if index is None:
                        raise ValueError("Unsupported indexing")
                    return ast.Name("p" + str(index), ast.Load())

            expression = Params().visit(expression)
            ast.fix_missing_locations(expression)
            allowed = (
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
            for n in ast.walk(expression):
                if not isinstance(n, allowed):
                    raise ValueError("Unsupported expression AST")
                if isinstance(n, ast.Name) and n.id not in set(fields) | {"x", "D", "np"} | {
                    "p" + str(i) for i in internal
                }:
                    raise ValueError("Unknown name " + n.id)
                if isinstance(n, ast.Attribute) and not (
                    isinstance(n.value, ast.Name)
                    and n.value.id == "np"
                    and n.attr in {"sin", "cos", "tanh", "exp"}
                ):
                    raise ValueError("Unknown function")
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                    if (
                        n.func.id != "D"
                        or len(n.args) != 2
                        or not isinstance(n.args[1], ast.Constant)
                        or n.args[1].value not in {1, 2, 3, 4}
                    ):
                        raise ValueError("Invalid derivative")
            expr = ast.unparse(expression)
            if mechanisms.derivative_order(expr, fields) > 4:
                raise ValueError("Total derivative order >4")
            nodes.append(
                {
                    "id": f"outer{outer}",
                    "outer_parameter": outer,
                    "fixed_coefficient": outer == "fixed",
                    "expr": expr,
                    "params": sorted("p" + str(i) for i in internal),
                    "complexity": sum(
                        isinstance(n, (ast.BinOp, ast.Call)) for n in ast.walk(expression)
                    ),
                }
            )
        indices = all_internal | {v for v in groups if isinstance(v, int)}
        if len(indices) > 32:
            raise ValueError("more than 32 parameters per output")
        if shared_outputs & indices:
            raise ValueError("Parameters must be independent across output equations")
        shared_outputs |= indices
        equations[field] = nodes
    body = "\n".join("    " + ast.unparse(n) for n in statements)
    return equations, body


def template(prior, metadata, fields):
    prefix = (
        prior + "\n\n[OBSERVATION METADATA]\n" + json.dumps(metadata) + "\n\nimport numpy as np\n"
    )
    args = "x, " + ", ".join(fields) + ", dx, params, d_dx"
    doc = """Discover complete mathematical RHS expressions. Do not name or assume a benchmark.
Use only the supplied state fields, x, ordinary arithmetic, np.sin/cos/tanh/exp,
and d_dx(expression, order) with total derivative order <=4. No imports, I/O,
numerical algorithms, branches, or loops in the equation body.
Use straight-line local assignments and return the output arrays.
Each output may contain at most eight fitted basis expressions. Prefer an outer
params[i] multiplying each basis; fixed mathematical terms are also allowed.
Internal nonlinear parameters may also use params[i].
Use distinct indices for internal and outer parameters; do not share parameters
across output equations. At most 32 distinct fitted constants per output.
All coefficients are fitted from the same observational training subset;
validation data select the final candidate. Do not encode a known equation.
The following Pareto frontier shows already evaluated mathematical candidates.
{PARETO_FRONTIER_PLACEHOLDER}
"""
    if len(fields) == 1:
        body = "    u_t_pred = params[0] * d_dx(u, order=2)\n    return u_t_pred"
    else:
        body = "    u_t_pred = params[0] * d_dx(u, order=2)\n    v_t_pred = params[32] * d_dx(v, order=2)\n    return u_t_pred, v_t_pred"
    fn = code_manipulation.Function("equation", args, body, docstring=doc)
    return code_manipulation.Program(prefix, [fn])
