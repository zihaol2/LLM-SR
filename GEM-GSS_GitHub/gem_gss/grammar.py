"""Offline normative scalar-component grammar. No data, references, API or search imports.

This validates syntax and conservative spatial order, not numerical domains.
Vector/tensor operators lower to finite collections of these scalar expressions.
"""

import ast
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sympy as sp

SPEC = json.loads((Path(__file__).parent / "assets" / "standard.json").read_text(encoding="utf-8"))
FUNCTIONS = {
    alias: name for name, entry in SPEC["functions"].items() for alias in [name, *entry["aliases"]]
}


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple = ()
    value: object = None
    order: int = 0


class Grammar:

    def __init__(self, fields, coordinates, max_order=4, parameters=("a", "b", "k")):
        self.fields = tuple(fields)
        self.coordinates = tuple(coordinates)
        self.max_order = max_order
        self.parameters = set(parameters)
        if not self.coordinates or len(set(self.coordinates)) != len(self.coordinates):
            raise ValueError("Distinct spatial coordinates required")
        if (
            set(self.fields) & set(self.coordinates)
            or "t" in self.fields
            or "t" in self.coordinates
            or self.parameters & (set(self.fields) | set(self.coordinates) | {"t"})
        ):
            raise ValueError("Symbol declarations must be disjoint")

    def node(self, op, *args, value=None):
        order = max((a.order for a in args), default=0)
        return Expr(op, tuple(args), value, order)

    def derivative(self, child, alpha):
        alpha = tuple(alpha)
        if len(alpha) != len(self.coordinates) or any((type(n) is not int or n < 0 for n in alpha)):
            raise ValueError("Nonnegative integer multi-index must match spatial dimension")
        if not any(alpha):
            return child
        order = child.order + sum(alpha)
        if self.max_order is not None and order > self.max_order:
            raise ValueError("Spatial derivative order exceeds execution profile")
        return Expr("PD", (child,), alpha, order)

    def axis(self, child, index, order):
        if type(index) is not int or not 0 <= index < len(self.coordinates):
            raise ValueError("Invalid spatial axis")
        if type(order) is not int or order < 1:
            raise ValueError("Positive integer derivative order required")
        alpha = [0] * len(self.coordinates)
        alpha[index] = order
        return self.derivative(child, alpha)

    def parse(self, text):
        return self.walk(ast.parse(text.replace("^", "**"), mode="eval").body)

    @staticmethod
    def integer(node):
        if not isinstance(node, ast.Constant) or type(node.value) is not int:
            raise ValueError("Integer literal required")
        return node.value

    def walk(self, n):
        if isinstance(n, ast.Constant):
            if type(n.value) not in (int, float) or not math.isfinite(n.value):
                raise ValueError("Finite real numeric literal required")
            return Expr("Constant", value=str(n.value))
        if isinstance(n, ast.Name):
            if n.id in self.fields:
                return Expr("Field", value=n.id)
            if n.id in {*self.coordinates, "t"}:
                return Expr("Coordinate", value=n.id)
            if n.id in self.parameters or re.fullmatch("p\\d+|theta_\\d+", n.id):
                return Expr("Parameter", value=n.id)
            raise ValueError("Undeclared symbol " + n.id)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            child = self.walk(n.operand)
            return child if isinstance(n.op, ast.UAdd) else self.node("Negate", child)
        if isinstance(n, ast.BinOp):
            names = {
                ast.Add: "Add",
                ast.Sub: "Subtract",
                ast.Mult: "Multiply",
                ast.Div: "Divide",
                ast.Pow: "Power",
            }
            if type(n.op) not in names:
                raise ValueError("Unsupported arithmetic")
            return self.node(names[type(n.op)], self.walk(n.left), self.walk(n.right))
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                name = n.func.id
            elif (
                isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and (n.func.value.id == "np")
            ):
                name = n.func.attr
            else:
                raise ValueError("Unsupported call target")
            if n.keywords:
                raise ValueError("Use positional canonical arguments")
            if name in FUNCTIONS:
                if len(n.args) != 1:
                    raise ValueError("Unary function signature")
                return self.node(FUNCTIONS[name], self.walk(n.args[0]))
            if name == "PD":
                if len(n.args) != 2 or not isinstance(n.args[1], (ast.Tuple, ast.List)):
                    raise ValueError("PD(E,(alpha_1,...,alpha_d)) required")
                return self.derivative(
                    self.walk(n.args[0]), [self.integer(v) for v in n.args[1].elts]
                )
            if name == "d_axis":
                if len(n.args) != 3:
                    raise ValueError("d_axis signature")
                return self.axis(
                    self.walk(n.args[0]), self.integer(n.args[1]), self.integer(n.args[2])
                )
            if name in {"D", "Dx", "Dy", "Dz", "d_dx", "d_dy", "d_dz"}:
                if len(n.args) != 2:
                    raise ValueError("Derivative signature")
                index = {"D": 0, "Dx": 0, "Dy": 1, "Dz": 2, "d_dx": 0, "d_dy": 1, "d_dz": 2}[name]
                return self.axis(self.walk(n.args[0]), index, self.integer(n.args[1]))
            if name in {"Lap", "lap", "laplace", "laplacian", "Laplacian"}:
                if len(n.args) != 1:
                    raise ValueError("Laplacian signature")
                child = self.walk(n.args[0])
                terms = [self.axis(child, j, 2) for j in range(len(self.coordinates))]
                result = terms[0]
                for term in terms[1:]:
                    result = self.node("Add", result, term)
                return result
            raise ValueError("Unregistered operator " + name)
        raise ValueError("Unsupported syntax " + type(n).__name__)

    def symbolic(self, node, time_in_field=True):
        coords = tuple((sp.Symbol(c, real=True) for c in self.coordinates))
        variables = coords + ((sp.Symbol("t", real=True),) if time_in_field else ())

        def convert(n):
            if n.op == "Constant":
                return sp.Rational(n.value)
            if n.op == "Field":
                return sp.Function(n.value)(*variables)
            if n.op in {"Coordinate", "Parameter"}:
                return sp.Symbol(n.value, real=True) if n.op == "Coordinate" else sp.Symbol(n.value)
            args = [convert(v) for v in n.args]
            if n.op == "PD":
                return sp.Derivative(
                    args[0], *[(c, v) for c, v in zip(coords, n.value) if v], evaluate=False
                )
            if n.op == "Negate":
                return -args[0]
            if n.op == "Add":
                return args[0] + args[1]
            if n.op == "Subtract":
                return args[0] - args[1]
            if n.op == "Multiply":
                return args[0] * args[1]
            if n.op == "Divide":
                return args[0] / args[1]
            if n.op == "Power":
                return args[0] ** args[1]
            if n.op == "CubeRoot":
                return sp.real_root(args[0], 3)
            return getattr(sp, SPEC["functions"][n.op]["sympy"])(args[0])

        return convert(node)

    def from_symbolic(self, z):
        """Import expanded scalar jet expressions for offline compatibility tests."""
        if z.is_Number:
            if z.is_Rational and (not z.is_Integer):
                return self.node(
                    "Divide", Expr("Constant", value=str(z.p)), Expr("Constant", value=str(z.q))
                )
            return Expr("Constant", value=str(z))
        if z.is_Symbol:
            return self.parse(str(z))
        if isinstance(z, sp.Derivative):
            alpha = [0] * len(self.coordinates)
            for coord, n in z.variable_count:
                if str(coord) not in self.coordinates:
                    raise ValueError("Only spatial derivatives on RHS")
                alpha[self.coordinates.index(str(coord))] += int(n)
            return self.derivative(self.from_symbolic(z.expr), alpha)
        if z.func.__name__ in self.fields:
            expected = tuple(self.coordinates)
            actual = tuple(map(str, z.args))
            if actual not in (expected, expected + ("t",)):
                raise ValueError("Unexpected field arguments")
            return Expr("Field", value=z.func.__name__)
        if z.is_Add or z.is_Mul:
            args = [self.from_symbolic(a) for a in z.args]
            result = args[0]
            for arg in args[1:]:
                result = self.node("Add" if z.is_Add else "Multiply", result, arg)
            return result
        if z.is_Pow:
            return self.node("Power", *[self.from_symbolic(a) for a in z.args])
        names = {entry["sympy"]: name for name, entry in SPEC["functions"].items()}
        if z.func.__name__ in names and len(z.args) == 1:
            return self.node(names[z.func.__name__], self.from_symbolic(z.args[0]))
        raise ValueError("Unregistered symbolic expression " + str(z))


def multi_indices(d, m):
    """All, rather than a selected list of, nonzero partials of total order <=m."""
    import itertools

    return [a for a in itertools.product(range(m + 1), repeat=d) if 1 <= sum(a) <= m]
