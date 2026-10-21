"""Shared canonical complexity, variable projection, and frozen-result refit.

Numerical arithmetic is retained from experiment 42/44. No reference imports.
"""

from collections import OrderedDict
import ast
import re
import warnings
import numpy as np
import sympy as s
from scipy.interpolate import make_interp_spline
from scipy.optimize import least_squares
from . import config, mechanisms
from .observations import Data as ObservationData
from .io import dump

SETTINGS = config.SETTINGS
ROOT = config.RUN_DIR
NAMES = config.names()

D = s.Function("D")
PHYSICAL = {s.Symbol(k) for k in ["u", "v", "x"]}


def symbolic(expr):
    def walk(n):
        if isinstance(n, ast.Name):
            if not re.fullmatch(r"u|v|x|a|b|k|p\d+", n.id):
                raise ValueError("unsupported identifier")
            return s.Symbol(n.id)
        if isinstance(n, ast.Constant) and isinstance(n.value, (float, int)):
            return s.Rational(str(n.value))
        if isinstance(n, ast.UnaryOp):
            return -walk(n.operand) if isinstance(n.op, ast.USub) else walk(n.operand)
        if isinstance(n, ast.BinOp):
            a, b = walk(n.left), walk(n.right)
            return {
                ast.Add: lambda: a + b,
                ast.Sub: lambda: a - b,
                ast.Mult: lambda: a * b,
                ast.Div: lambda: a / b,
                ast.Pow: lambda: a**b,
            }[type(n.op)]()
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name) and n.func.id == "D":
                z = walk(n.args[0])
                order = int(n.args[1].value)
                if not 1 <= order <= 4:
                    raise ValueError("invalid derivative order")
                if not z.free_symbols & PHYSICAL:
                    return s.Integer(0)
                scalar, base = z.as_independent(*PHYSICAL, as_Add=False)
                if base.func == D:
                    return scalar * D(base.args[0], s.Integer(int(base.args[1]) + order))
                return scalar * D(base, s.Integer(order))
            name = n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
            return {"sin": s.sin, "cos": s.cos, "tanh": s.tanh, "exp": s.exp}[name](walk(n.args[0]))
        raise ValueError("unsupported expression")

    return walk(ast.parse(expr, mode="eval").body)


def python_expr(z):
    if z.is_Symbol:
        return str(z)
    if z.is_Integer:
        return str(z)
    if z.is_Rational:
        return "(" + str(z.p) + "/" + str(z.q) + ")"
    if z.is_Number:
        return repr(float(z))
    if z.is_Add:
        return "(" + " + ".join(python_expr(a) for a in z.args) + ")"
    if z.is_Mul:
        return "(" + " * ".join(python_expr(a) for a in z.args) + ")"
    if z.is_Pow:
        return "(" + python_expr(z.base) + " ** " + python_expr(z.exp) + ")"
    if z.func == D:
        return "D(" + python_expr(z.args[0]) + ", " + str(int(z.args[1])) + ")"
    return "np." + z.func.__name__ + "(" + python_expr(z.args[0]) + ")"


def operation_cost(z, in_add=False):
    if z.is_Atom:
        return 0
    if z.func == D:
        return 2 + operation_cost(z.args[0])
    if z.is_Add:
        return len(z.args) - 1 + sum(operation_cost(a, True) for a in z.args)
    if z.is_Mul:
        # A subtraction already counts at the parent Add; do not count -1*u again.
        operations = len(z.args) - 1 - int(in_add and z.args[0] == -1)
        return operations + sum(operation_cost(a) for a in z.args)
    if z.is_Pow:
        return 2 + operation_cost(z.base) + operation_cost(z.exp)
    return 1 + sum(operation_cost(a) for a in z.args)


def normalize_node(node):
    node = dict(node)
    z = symbolic(node["expr"])
    if not node.get("fixed_coefficient", False):
        scalar, z = z.as_independent(*PHYSICAL, as_Add=False)
        if scalar == 0:
            raise ValueError("zero mechanism")
    if z == 0:
        raise ValueError("zero mechanism")
    expr = python_expr(z)
    if mechanisms.derivative_order(expr, ["u", "v"]) > 4:
        raise ValueError("derivative order cap")
    node.update(
        expr=expr,
        params=sorted(str(p) for p in z.free_symbols - PHYSICAL),
        complexity=operation_cost(z),
    )
    return node


def complexity(terms, shared=False):
    cost = 0
    internals = set()
    outer = 0
    for i, t in enumerate(terms):
        n = normalize_node(t.get("node", t))
        z = symbolic(n["expr"])
        cost += operation_cost(z)
        if not n.get("fixed_coefficient", False):
            outer += 1
            if z != 1:
                cost += 1
        for p in n["params"]:
            internals.add(p if shared else (i, p))
    return max(1, cost + outer + len(internals) + max(0, len(terms) - 1))


class Data(ObservationData):
    def __init__(self, name):
        super().__init__(name)
        self.feature_cache = OrderedDict()
        self.derivatives = {}

    def feature(self, node, params, full=False):
        cachekey = (node["expr"], tuple(map(float, params)), full)
        if cachekey in self.feature_cache:
            self.feature_cache.move_to_end(cachekey)
            return self.feature_cache[cachekey]
        vals = self.full if full else self.values

        def derivative(a, n):
            measured = next((f for f, v in vals.items() if a is v), None)
            k = (measured, n, full)
            if measured is not None and k in self.derivatives:
                return self.derivatives[k]
            a = np.broadcast_to(a, next(iter(vals.values())).shape)
            out = make_interp_spline(self.x, a, k=5, axis=0)(self.x, nu=n)
            if measured is not None:
                self.derivatives[k] = out
            return out

        ns = {
            "np": np,
            "x": self.x[:, None],
            "D": derivative,
            **vals,
            **dict(zip(node["params"], params)),
        }
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            a = np.asarray(eval(node["expr"], {"__builtins__": {}}, ns), dtype=float)
        a = np.broadcast_to(a, next(iter(vals.values())).shape)
        if not np.all(np.isfinite(a)) or np.max(abs(a)) > 1e10:
            raise ValueError("invalid feature")
        v = self.window(a).ravel()
        if np.linalg.norm(v) < 1e-13:
            raise ValueError("zero feature")
        self.feature_cache[cachekey] = v
        if len(self.feature_cache) > 4096:
            self.feature_cache.popitem(last=False)
        return v


def fit(data, field, nodes, initial_terms=None, shared=False, full_budget=True):
    nodes = [normalize_node(n) for n in nodes]
    tr = data.trainmask
    va = data.valmask
    y = data.y[field]
    if not nodes:
        v = float(np.mean(y[va] ** 2))
        return {
            "terms": [],
            "train_mse": float(np.mean(y[tr] ** 2)),
            "validation_mse": v,
            "complexity": 0,
            "score": v / data.scale[field],
            "fit_nfev": 0,
            "fully_refined": True,
        }
    slots = {}
    indices = []
    for i, n in enumerate(nodes):
        ix = []
        for p in n["params"]:
            key = p if shared else (i, p)
            if key not in slots:
                slots[key] = len(slots)
            ix.append(slots[key])
        indices.append(ix)
    count = len(slots)
    ones = np.ones(count)
    inherited = ones.copy()
    old = {t["expr"]: t for t in initial_terms or []}
    for n, ix in zip(nodes, indices):
        if n["expr"] in old:
            for p, j in zip(n["params"], ix):
                inherited[j] = old[n["expr"]]["params"].get(p, 1.0)
    starts = [inherited] + [np.full(count, p) for p in SETTINGS["coarse_parameters"]]
    # Vary only newly added parameters while preserving fitted parent parameters.
    newslots = {j for n, ix in zip(nodes, indices) if n["expr"] not in old for j in ix}
    for value in SETTINGS["coarse_parameters"]:
        p = inherited.copy()
        for j in newslots:
            p[j] = value
        starts.append(p)
    unique = {tuple(map(float, p)): p for p in starts}
    counts = {"loss": 0, "nfev": 0}
    fixed = np.array([n.get("fixed_coefficient", False) for n in nodes])
    free = ~fixed

    def compute(p):
        A = np.column_stack([data.feature(n, p[ix]) for n, ix in zip(nodes, indices)])
        c = np.ones(len(nodes))
        target = y[tr] - A[tr][:, fixed] @ c[fixed]
        if np.any(free):
            norm = np.maximum(np.linalg.norm(A[tr][:, free], axis=0), 1e-20)
            c[free] = np.linalg.lstsq(A[tr][:, free] / norm, target, rcond=1e-9)[0] / norm
        return A, c

    def loss(p):
        counts["loss"] += 1
        try:
            A, c = compute(p)
            return (A[tr] @ c - y[tr]) / np.sqrt(data.scale[field])
        except Exception:
            return np.full(int(tr.sum()), 1e6)

    starts = sorted(unique.values(), key=lambda p: float(np.mean(loss(p) ** 2)))[
        : 2 if count else 1
    ]
    models = []
    for p0 in starts:
        p = p0.copy()
        if count:
            try:
                opt = least_squares(
                    loss,
                    p,
                    bounds=(-8, 8),
                    max_nfev=SETTINGS["max_nfev_full" if full_budget else "max_nfev_screen"],
                    ftol=1e-8,
                    xtol=1e-8,
                    gtol=1e-8,
                    diff_step=1e-4,
                )
                p = opt.x
                counts["nfev"] += opt.nfev
            except Exception:
                pass
        try:
            A, c = compute(p)
            prediction = A @ c
            terms = [
                {
                    "id": n["id"],
                    "expr": n["expr"],
                    "params": dict(zip(n["params"], map(float, p[ix]))),
                    "coefficient": float(coef),
                    "node": n,
                }
                for n, ix, coef in zip(nodes, indices, c)
            ]
            train = float(np.mean((prediction[tr] - y[tr]) ** 2))
            valid = float(np.mean((prediction[va] - y[va]) ** 2))
            C = complexity(terms, shared)
            score = float((valid / data.scale[field] + 1e-14) * np.exp(SETTINGS["lambda"] * C))
            if np.isfinite(score):
                models.append(
                    {
                        "terms": terms,
                        "train_mse": train,
                        "validation_mse": valid,
                        "complexity": C,
                        "score": score,
                        "fit_nfev": counts["nfev"],
                        "fit_loss_calls": counts["loss"],
                        "fully_refined": full_budget or not count,
                    }
                )
        except Exception:
            pass
    if not models:
        return None
    return min(models, key=lambda m: m["score"])


def final_equation(data, field, model):
    terms = model["terms"]
    if not terms:
        return {
            "terms": [],
            "MSE": float(np.mean(data.yfull[field] ** 2)),
            "training_MSE": model["train_mse"],
            "validation_MSE": model["validation_mse"],
        }
    A = np.column_stack(
        [
            data.feature(t["node"], [t["params"][p] for p in t["node"]["params"]], True)
            for t in terms
        ]
    )
    fixed = np.array([t["node"].get("fixed_coefficient", False) for t in terms])
    free = ~fixed
    c = np.ones(len(terms))
    target = data.yfull[field] - A[:, fixed] @ c[fixed]
    if np.any(free):
        norm = np.maximum(np.linalg.norm(A[:, free], axis=0), 1e-30)
        c[free] = np.linalg.lstsq(A[:, free] / norm, target, rcond=1e-10)[0] / norm
    result = [
        dict({k: v for k, v in t.items() if k != "node"}, coefficient=float(coef))
        for t, coef in zip(terms, c)
    ]
    return {
        "terms": result,
        "MSE": float(np.mean((A @ c - data.yfull[field]) ** 2)),
        "training_MSE": model["train_mse"],
        "validation_MSE": model["validation_mse"],
        "complexity": model["complexity"],
    }
