"""Real execution profile: exclude Sinc and Hessian determinants only."""

import itertools
from functools import lru_cache

import sympy as s


class JetProfile:

    def __init__(self, fields, coordinates):
        self.fields = tuple(fields)
        self.coordinates = tuple(coordinates)
        self.d = len(coordinates)
        self.axes = [s.Symbol(c, real=True) for c in coordinates]
        self.jets = {}
        for f in fields:
            for alpha in itertools.product(range(3), repeat=self.d):
                if sum(alpha) <= 2:
                    self.jets[f, alpha] = s.Symbol(f + "__" + "".join(map(str, alpha)), real=True)
        self.hessian = [v for (f, a), v in self.jets.items() if sum(a) == 2]

    def derivative(self, z, axis):
        out = s.diff(z, self.axes[axis])
        for (field, alpha), jet in self.jets.items():
            if sum(alpha) >= 2 or not z.has(jet):
                continue
            next_alpha = list(alpha)
            next_alpha[axis] += 1
            out += s.diff(z, jet) * self.jets[field, tuple(next_alpha)]
        return out

    @lru_cache(maxsize=None)
    def symbolic_jet(self, e):
        op = e.op
        if op == "Field":
            return self.jets[e.value, (0,) * self.d]
        if op == "Constant":
            return s.Rational(e.value)
        if op in {"Coordinate", "Parameter"}:
            return s.Symbol(e.value, real=True)
        a = [self.symbolic_jet(v) for v in e.args]
        if op == "Add":
            return a[0] + a[1]
        if op == "Subtract":
            return a[0] - a[1]
        if op == "Multiply":
            return a[0] * a[1]
        if op == "Divide":
            return a[0] / a[1]
        if op == "Power":
            return a[0] ** a[1]
        if op == "Negate":
            return -a[0]
        if op == "PD":
            value = a[0]
            for axis, n in enumerate(e.value):
                for _ in range(n):
                    value = self.derivative(value, axis)
            return value
        funcs = {
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
            "CubeRoot": lambda z: s.real_root(z, 3),
        }
        return funcs[op](a[0])

    @lru_cache(maxsize=None)
    def rejection(self, e):
        if e.op == "Sinc":
            return "Sinc excluded by execution profile"
        for child in e.args:
            reason = self.rejection(child)
            if reason:
                return reason
        if e.order == 2:
            z = self.symbolic_jet(e)
            if any((z.has(v) for v in self.hessian)):
                try:
                    if s.Poly(z, *self.hessian, domain="EX").total_degree() > 1:
                        return "Fully nonlinear order-2 dependence excluded"
                except s.PolynomialError:
                    return "Non-affine function of order-2 jets excluded"
        return None

    def validate(self, e):
        reason = self.rejection(e)
        if reason:
            raise ValueError(reason)
        return e


class Profile(JetProfile):

    def __init__(self, fields, coordinates):
        super().__init__(fields, coordinates)
        self.determinants = []
        if self.d >= 2:
            for field in fields:
                matrix = []
                for i in range(self.d):
                    row = []
                    for j in range(self.d):
                        alpha = [0] * self.d
                        alpha[i] += 1
                        alpha[j] += 1
                        row.append(self.jets[field, tuple(alpha)])
                    matrix.append(row)
                self.determinants.append(
                    s.Poly(s.det(s.Matrix(matrix)), *self.hessian, domain="EX")
                )

    @lru_cache(maxsize=None)
    def rejection(self, e):
        if e.op == "Sinc":
            return "Sinc excluded by execution profile"
        for child in e.args:
            reason = self.rejection(child)
            if reason:
                return reason
        if (
            self.determinants
            and e.order == 2
            and (e.op in {"Add", "Subtract", "Multiply", "Divide", "Negate"})
        ):
            try:
                z = s.Poly(self.symbolic_jet(e), *self.hessian, domain="EX")
            except s.PolynomialError:
                return None
            for determinant in self.determinants:
                ratios = [z.coeff_monomial(m) / c for m, c in determinant.terms()]
                if ratios[0] != 0 and all((s.simplify(v - ratios[0]) == 0 for v in ratios[1:])):
                    return "Hessian determinant combination excluded by execution profile"
        return None
