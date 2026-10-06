"""The mechanism language: expression trees and the operator grammar.

`expr_ast` is the structured form the design step writes (JSON trees that we
compile); `pde_grammar` is the callable surface the sandbox binds and an
evaluated candidate imports as `G`.
"""
from __future__ import annotations
import ast
import json
import os

import numpy as np


# ==========================================================================
# structured expressions (JSON trees)
# ==========================================================================

"""Structured mechanism definitions: a JSON expression tree, compiled by us.

The LLM no longer writes operator strings. It writes a small JSON tree whose nodes
are named operations from a FIXED, PROBLEM-INDEPENDENT vocabulary. This module owns
the syntax completely -- arity, argument kinds, differential order and the generated
numpy code -- so everything the old string interface used to get wrong (arity,
function names, float literals, axis naming, brackets, `ADD`/`TANH`) cannot even be
expressed.

A definition is a list `[op, arg, ...]`. Leaves are

    "f1"            a measured field (by name)
    "phi"           a formal declared in the signature
    1  /  0.5       a numeric literal

and nodes are built from the vocabulary in `OPS`, for example

    ["d", ["mul", "f1", ["x", "axis"]], "axis", 1]      d/dx [ f1 * x1 ]
    ["flux_div", ["mul", ["1 - f1"], ["coord", "tanh", 1, 0, 0]], "axis"]
"""

import numpy as np  # noqa: F401  (made available to the compiled function)
import keyword


# --------------------------------------------------------------------- vocabulary
# each entry: arity, extra differential order, code template
#   {i} refers to the i-th argument of the node
_OPS: dict[str, tuple[int, object, str]] = {
    # ---- arithmetic -------------------------------------------------------
    # arity -1 means "two or more": a sum/product written as a flat list is the
    # natural way to write a multi-term polynomial, and refusing it only costs a
    # rejected mechanism (see `poly` for the polynomial closure itself).
    "add":  (-1, 0, ""),
    "sub":  (2, 0, "({0} - {1})"),
    "mul":  (-1, 0, ""),
    "div":  (2, 0, "({0} / {1})"),
    "neg":  (1, 0, "(-{0})"),
    "pow":  (2, 0, "({0} ** {1})"),
    # a polynomial closure in one expression: poly(f, c1, p1, c2, p2, ...)
    "poly": (-1, 0, ""),
    "abs":  (1, 0, "ABS({0})"),
    "sqrt": (1, 0, "np.sqrt({0})"),
    "min":  (2, 0, "np.minimum({0}, {1})"),
    "max":  (2, 0, "np.maximum({0}, {1})"),
    # ---- scalar functions (elementwise) -----------------------------------
    "sin":      (1, 0, "np.sin({0})"),
    "cos":      (1, 0, "np.cos({0})"),
    "tanh":     (1, 0, "np.tanh({0})"),
    "exp":      (1, 0, "np.exp({0})"),
    "log":      (1, 0, "np.log({0})"),
    "log1p":    (1, 0, "np.log1p({0})"),
    "sigmoid":  (1, 0, "(1.0 / (1.0 + np.exp(-{0})))"),
    "softplus": (1, 0, "np.log1p(np.exp({0}))"),
    "inv":      (1, 0, "(1.0 / {0})"),
    # ---- coordinates ------------------------------------------------------
    "x":     (1, 0, "X({0})"),
    "coord": (4, 0, "coord({0!r}, {1}, {2}, {3})"),
    # `coord` wraps a LINEAR argument, so a Gaussian envelope could only be written as
    # a nest (exp of mul of pow of x). A localised bell is common enough as a forcing
    # envelope or a beam profile that it gets its own idiom rather than being lost.
    "bump":  (1, 0, "np.exp(-{0} * (X(0) ** 2))"),
    # ---- differential operators -------------------------------------------
    "d":       (3, None, "D({0}, {1}, {2})"),          # order from the 3rd arg
    "lap":     (1, 2, "dif({0})"),
    "biharm":  (1, 4, "hdif({0}, 0)"),
    "disp":    (2, 3, "disp({0}, {1})"),
    # ---- composite flux idioms (still pure mathematics) --------------------
    "flux_div":          (2, 1, "D({0}, {1}, 1)"),                              # div(flux)
    "pair_flux_div":     (3, 1, "D(MUL({0}, {1}), {2}, 1)"),                    # div(a*b)
    "gradient_flux":     (3, 2, "D(MUL({0}, D({1}, {2}, 1)), {2}, 1)"),         # div(k*grad p)
    "log_gradient_flux": (2, 2, "D(DIV(D({0}, {1}, 1), {0}), {1}, 1)"),         # div(grad p / p)
    "power_gradient_flux": (3, 2, "D(MUL(POW({0}, {2}), D({0}, {1}, 1)), {1}, 1)"),
}

_MAX_ORDER = 4


# Which argument of each op is an AXIS slot (1-based position in the list form).
_AXIS_SLOT = {
    "d": 2, "disp": 2, "x": 1,
    "flux_div": 2, "pair_flux_div": 3, "gradient_flux": 3,
    "log_gradient_flux": 2, "power_gradient_flux": 2,
}


def canonicalize_axes(expr):
    """Rename every name used in an axis slot to the formal `axis`.

    A model may write `"x1"` (the coordinate's own name) or invent a formal where an
    axis index belongs. The atom builder knows only the name `axis`, so anything else
    in that slot would be bound as a scalar coefficient and the mechanism would die on
    a type error for reasons that have nothing to do with its physics.
    """
    if not isinstance(expr, list) or not expr:
        return _safe_formal(expr)
    op = expr[0]
    slot = _AXIS_SLOT.get(op)
    out = [op]
    for position, arg in enumerate(expr[1:], start=1):
        if slot is not None and position == slot and isinstance(arg, str):
            out.append("axis")
        elif op == "coord" and position == 1:
            out.append(arg)                      # the scalar-function name
        else:
            out.append(canonicalize_axes(arg))
    return out


def _safe_formal(value):
    """A leaf name that is a Python keyword (`lambda`, `class`, ...) would compile to
    `def m(lambda, ...)` and die on a SyntaxError, losing the mechanism for a reason
    that has nothing to do with its physics. The signature is derived from the tree
    and the search passes arguments positionally, so renaming the formal is safe."""
    if isinstance(value, str) and keyword.iskeyword(value):
        return value + "_"
    return value


def _is_axis_token(value) -> bool:
    return isinstance(value, (int, str))


def order_of(expr) -> int:
    """Differential order of an expression tree."""
    if not isinstance(expr, list):
        return 0
    op = expr[0]
    spec = _OPS.get(op)
    if spec is None:
        return 0
    children = [order_of(a) for a in expr[1:]]
    inner = max(children) if children else 0
    if op == "d":
        k = expr[3] if len(expr) > 3 and isinstance(expr[3], int) else 1
        return int(k) + order_of(expr[1])
    extra = spec[1]
    return int(extra or 0) + inner


def validate(expr, formals, fields=("f1",)) -> str | None:
    """Return None when the tree is admissible, else a one-line reason."""
    if isinstance(expr, (int, float)):
        return None
    if isinstance(expr, str):
        if expr in formals or expr in fields:
            return None
        return f"`{expr}` is neither a declared formal nor a measured field"
    if not isinstance(expr, list) or not expr:
        return f"not a node: {expr!r}"
    op = expr[0]
    if not isinstance(op, str) or op not in _OPS:
        return f"`{op}` is not in the operator vocabulary"
    arity = _OPS[op][0]
    n_args = len(expr) - 1
    if arity == -1:
        # variadic: `mul`/`add` take two or more, `poly` a base plus (coeff, power) pairs
        if op == "poly":
            if n_args < 3 or (n_args - 1) % 2:
                return (f"`poly(f, c1, p1, c2, p2, ...)` needs a base plus "
                        f"coefficient/power pairs, got {n_args} argument(s)")
            for k in range(3, len(expr), 2):
                power = expr[k]
                if not isinstance(power, int) or isinstance(power, bool) \
                        or not (1 <= power <= 8):
                    return "`poly` powers must be integer literals in 1..8"
        elif n_args < 2:
            return f"`{op}` takes 2 or more arguments, got {n_args}"
    elif n_args != arity:
        return f"`{op}` takes {arity} argument(s), got {n_args}"
    if op == "d":
        if not _is_axis_token(expr[2]):
            return "`d` needs an axis (an integer or the formal `axis`)"
        if not isinstance(expr[3], int) or not (1 <= expr[3] <= _MAX_ORDER):
            return f"`d` needs an integer order in 1..{_MAX_ORDER}"
    if op == "pow" and not isinstance(expr[2], (int, float)):
        return "`pow` needs a numeric exponent"
    if op == "power_gradient_flux" and not isinstance(expr[3], (int, float)):
        return ("`power_gradient_flux(phi, axis, n)` needs a numeric exponent as its "
                "third argument")
    if op == "coord":
        if not isinstance(expr[1], str):
            return "`coord` takes a function name first (id/sin/cos/exp/tanh)"
        if expr[1] not in ("id", "sin", "cos", "exp", "tanh"):
            return f"`coord` does not know the function `{expr[1]}`"
    # `coord`'s first argument is a function name already checked above.
    for arg in expr[(2 if op == "coord" else 1):]:
        reason = validate(arg, formals, fields)
        if reason:
            return reason
    return None


def compile_expr(expr, formals, fields=("f1",)) -> tuple[str, int]:
    """(python source for the expression, differential order). Validates first."""
    reason = validate(expr, formals, fields)
    if reason:
        raise ValueError(reason)
    return _source(expr), order_of(expr)


def _source(expr) -> str:
    if isinstance(expr, (int, float)):
        return repr(expr)
    if isinstance(expr, str):
        return expr
    op, args = expr[0], expr[1:]
    if _OPS[op][0] == -1:
        parts = [_source(a) for a in args]
        if op == "add":
            return "(" + " + ".join(parts) + ")"
        if op == "mul":
            return "(" + " * ".join(parts) + ")"
        base = parts[0]
        terms = []
        for i in range(1, len(args), 2):
            coeff, power = parts[i], args[i + 1]
            powered = base if power == 1 else f"({base} ** {power})"
            terms.append(f"({coeff} * {powered})")
        return "(" + " + ".join(terms) + ")"
    code = _OPS[op][2]
    return code.format(*[_source(a) for a in args])


def signature_of(expr, wanted):
    """Collect the formals a tree actually uses (plus anything the caller wants)."""
    out, seen = [], set()
    for name in list(wanted) + list(_leaves(expr)):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _leaves(expr):
    if isinstance(expr, str):
        yield expr
    elif isinstance(expr, list):
        # `coord`'s first argument is a function NAME, not a value: skip it.
        start = 2 if expr and expr[0] == "coord" else 1
        for a in expr[start:]:
            yield from _leaves(a)


def vocabulary_text() -> str:
    """The op list as prompt text: one line per operator.

    The order rule is stated once, in the header, and repeated per operator only
    when it differs from the default (the max over the operator's arguments), so
    the block stays short enough to read in one pass.
    """
    lines = ["  (order of every operator = MAX over its arguments unless noted)"]
    for name, (arity, extra, _code) in _OPS.items():
        fallback = ", ".join("a%d" % i for i in range(arity)) if arity > 0 else "a, b, ..."
        args, doc = _DOC.get(name, (fallback, ""))
        if name == "d":
            note = "order += order(expr)"
        elif extra:
            note = f"order = {extra} + order(inside)"
        else:
            note = ""
        line = f"  {name}({args})"
        if doc:
            line += f"   {doc}"
        if note:
            line += f"   [{note}]"
        lines.append(line)
    return "\n".join(lines)


_DOC = {
    "add": ("a, b, ...", "sum of two or more, any number of terms"),
    "sub": ("a, b", "pointwise difference a - b"),
    "mul": ("a, b, ...", "product of two or more factors (e.g. f*f*f = f**3)"),
    "div": ("a, b", "pointwise ratio a / b"),
    "neg": ("a", "negation"),
    "pow": ("a, n", "a ** n; n is a numeric literal"),
    "poly": ("f, c1, p1, c2, p2, ...",
             "c1*f**p1 + c2*f**p2 + ...; a polynomial closure in the expression f, "
             "powers are integer literals, coefficients may be formals"),
    "abs": ("a", "absolute value"),
    "sqrt": ("a", "square root"),
    "min": ("a, b", "pointwise minimum"),
    "max": ("a, b", "pointwise maximum"),
    "sin": ("a", "elementwise sine"),
    "cos": ("a", "elementwise cosine"),
    "tanh": ("a", "elementwise tanh"),
    "exp": ("a", "elementwise exponential"),
    "log": ("a", "elementwise natural logarithm"),
    "log1p": ("a", "elementwise log(1 + a)"),
    "sigmoid": ("a", "elementwise 1 / (1 + exp(-a))"),
    "softplus": ("a", "elementwise log(1 + exp(a))"),
    "inv": ("a", "elementwise 1 / a"),
    "x": ("axis", "the coordinate grid of an axis, e.g. x1"),
    "coord": ("func, k1, k2, k3",
              "func(k1*x1 + k2*x2 + k3*x3); func is a quoted name "
              "(id/sin/cos/exp/tanh)"),
    "bump": ("k", "a smooth bell centred at x1 = 0, exp(-k*x1**2); the shape of a "
                  "localised envelope or beam profile, width set by k"),
    "d": ("expr, axis, order", "order-th derivative of expr along an axis; "
                               "order is an integer 1..4"),
    "lap": ("expr", "sum of second derivatives over all axes"),
    "biharm": ("expr", "fourth derivative along x1"),
    "disp": ("expr, axis", "third derivative along an axis"),
    "flux_div": ("flux, axis", "OUTER divergence of a flux expression"),
    "pair_flux_div": ("a, b, axis", "OUTER divergence of the product a*b"),
    "gradient_flux": ("coeff, phi, axis",
                      "OUTER divergence of coeff * d(phi, axis, 1)"),
    "log_gradient_flux": ("phi, axis",
                          "OUTER divergence of d(phi, axis, 1) / phi"),
    "power_gradient_flux": ("phi, axis, n",
                            "OUTER divergence of phi**n * d(phi, axis, 1); "
                            "n is a numeric literal"),
}


# ==========================================================================
# the operator grammar the search imports as G
# ==========================================================================

"""Operator grammar for LLM-PISR1.

Every public constructor maps numpy grids to numpy grids. Names are abbreviated and
deliberately unexplained: this module defines *what can be built*, not *what any
construction means* for a particular physical system.

Two layers are exposed:

    primitives   D, S, MUL          -- always available
    mechanisms   adv, dif, disp, hdif, rxn, prs, frc, cpl

Every mechanism is one *fixed* differential operator, not a family of them: the
derivative order is part of the definition rather than an argument. Transport is
first order, dispersion third, hyper-dissipation fourth, diffusion second. That
collapse is deliberate -- letting the order float would merge distinct physical
operators (transport vs dispersion) into a single slot.

Nothing here privileges one mechanism over another; the search has to decide.
"""


# ------------------------------------------------------------------ binding
# The problem supplies its coordinate axes and a derivative operator once per
# evaluation; the constructors below are then pure functions of the grid fields.
_AXES = ()
_DERIV = None
_VECTOR = ()          # grid arrays ordered so that entry k belongs to axis k
_PRESSURE = None      # the one field the gradient operator belongs to
_FORCING = ()         # fields that exist only as bare source terms
_COORDS = ()          # coordinate grids, in axis order, for `wave()`


def bind(axes, derivative, vector=(), pressure=None, forcing=(), coords=()):
    """Attach the current problem's axes and derivative operator.

    derivative(field, axis, order) -> grid

    `vector` is the declared vector field as a tuple of grids in axis order, so
    entry k is the axis-k component. It is empty when the problem does not come
    with such a structure.

    `pressure` / `forcing` declare the physical roles of the remaining fields: a
    restricted field is a role, not a free variable. When they are declared the
    role is enforced *here*, at run time, on object identity -- so no candidate can
    route a restricted field into a constructor its role does not allow, whatever
    the submitted text looks like.
    """
    global _AXES, _DERIV, _VECTOR, _PRESSURE, _FORCING, _COORDS
    _AXES = tuple(axes)
    _DERIV = derivative
    _VECTOR = tuple(vector)
    _PRESSURE = pressure
    _FORCING = tuple(forcing)
    _COORDS = tuple(coords)


def _forbid_restricted(*arrays):
    """Raise if any argument is a field whose role forbids this construction."""
    for name, restricted in (("the pressure field", (_PRESSURE,) if _PRESSURE is not None else ()),
                             ("a forcing component", _FORCING)):
        for field in restricted:
            for array in arrays:
                if array is field:
                    raise ValueError(
                        f"{name} may not enter this construction: the pressure exists "
                        f"only through `prs`, a forcing component only through `frc`")
    return arrays


def vector_size() -> int:
    return len(_VECTOR)


_FUNCS = {
    "id": lambda a: a,
    "sin": np.sin,
    "cos": np.cos,
    "exp": np.exp,
    "tanh": np.tanh,
}


# --------------------------------------------------------------- primitives
def D(f, axis, order):
    return _DERIV(f, axis, order)


def S(f, func):
    return _FUNCS[func](f)


def MUL(a, b):
    return a * b


def POW(a, n):
    """Pointwise power `a ** n` (n a positive integer)."""
    return a ** int(n)


def ABS(a):
    """Pointwise absolute value."""
    return np.abs(a)


def X(axis):
    """The coordinate grid of an axis, so a definition can build polynomials in x."""
    if not _COORDS:
        raise ValueError("no coordinate grids have been declared for this problem")
    return _COORDS[int(axis)]


def DIV(a, b):
    """Pointwise division -- the primitive a ratio-valued (saturating) term needs."""
    return a / b


def T(f):
    """Pointwise tanh of one field (alias of `S(f, 'tanh')`)."""
    return np.tanh(f)


def coord(func, k1, k2=0.0, k3=0.0):
    """A coordinate function of an arbitrary wavevector: `func(k1*x1 + k2*x2 + k3*x3)`.

    `func` may be any of {'id', 'sin', 'cos', 'exp', 'tanh'}; the wavevector is
    fitted. This is the general form of the external/terrain pattern.
    """
    if not _COORDS:
        raise ValueError("no coordinate grids have been declared for this problem")
    arg = k1 * _COORDS[0]
    if len(_COORDS) > 1:
        arg = arg + k2 * _COORDS[1]
    if len(_COORDS) > 2:
        arg = arg + k3 * _COORDS[2]
    return _FUNCS[func](arg)


# --------------------------------------------------------------- mechanisms
# Each name denotes exactly one operator, with its differential order fixed.
# Transport is first order by definition; higher orders are separate mechanisms.
def adv(phi):
    """The transport operator as a whole: sum_i v_i * d_i phi.

    This is one physical mechanism, not a family of directional pieces: the sum
    over the contraction index is what makes it `(u . grad)`. Writing only one of
    its addends is not transport at all, so the whole operator -- and its index
    contraction -- is offered as a single indivisible construction.
    """
    if not _VECTOR:
        raise ValueError("no vector field has been declared for this problem")
    _forbid_restricted(phi)
    return sum(MUL(_VECTOR[i], D(phi, i, 1)) for i in range(len(_VECTOR)))


def dif(f):
    _forbid_restricted(f)
    return sum(D(f, ax, 2) for ax in _AXES)


def disp(f, axis):
    _forbid_restricted(f)
    return D(f, axis, 3)


def hdif(f, axis):
    _forbid_restricted(f)
    return D(f, axis, 4)


def rxn(f, func="id"):
    _forbid_restricted(f)
    return S(f, func)


def prs(f, axis):
    if _PRESSURE is not None and f is not _PRESSURE:
        raise ValueError(
            "prs() is the pressure gradient: it may only be applied to the "
            "declared pressure field")
    return D(f, axis, 1)


def frc(f):
    if _FORCING and not any(f is field for field in _FORCING):
        raise ValueError(
            "frc() is the bare source term: it may only be applied to a declared "
            "forcing component")
    return f


def cpl(fa, fb):
    _forbid_restricted(fa, fb)
    return MUL(fa, fb)


def wave(k1, k2, func="sin", k3=0.0):
    """A coordinate sinusoid: `func(k1*x1 + k2*x2 [+ k3*x3])`.

    This is the *mechanism-level* form of an external/source pattern. Candidates kept
    reaching for it by name (`G.coord_sin`, `G.coord_cos`) and being rejected for it;
    exposing it makes that whole family of forcing terms legal, cheap and comparable
    with the other mechanisms. Nothing about which wavevector is right is assumed --
    `k1`, `k2` and the phase are ordinary numbers the optimiser fits.
    """
    if not _COORDS:
        raise ValueError("no coordinate grids have been declared for this problem")
    arg = k1 * _COORDS[0] + k2 * _COORDS[1]
    if len(_COORDS) > 2:
        arg = arg + k3 * _COORDS[2]
    return _FUNCS[func](arg)


def coord_sin(k1, k2, k3=0.0):
    """`sin(k1*x1 + k2*x2 [+ k3*x3])` -- an alias of `wave(..., 'sin')`.

    Kept as its own name because it is the spelling candidates actually reach for.
    A draw that says `coord_sin` is making exactly the same hypothesis as one that
    says `wave`, so the two have to be judged the same way rather than one of them
    dying before it is ever fitted.
    """
    return wave(k1, k2, "sin", k3)


def coord_cos(k1, k2, k3=0.0):
    """`cos(k1*x1 + k2*x2 [+ k3*x3])` -- an alias of `wave(..., 'cos')`."""
    return wave(k1, k2, "cos", k3)


__all__ = ["D", "S", "MUL", "DIV", "POW", "ABS", "X", "T", "coord", "adv", "dif",
           "disp", "hdif", "rxn", "prs", "frc", "cpl", "wave", "coord_sin", "coord_cos"]

MECHANISM_NAMES = set({"adv", "dif", "disp", "hdif", "rxn", "prs", "frc", "cpl",
                       "wave", "coord_sin", "coord_cos"})
PRIMITIVE_NAMES = frozenset({"D", "S", "MUL", "DIV", "POW", "ABS", "X", "T", "coord"})

# How many derivative orders each constructor may consume, used by the checker to
# read the highest order a candidate relies on.
ORDER_ARGS = {
    "D": 2,        # D(f, axis, order)
}

# Differential order implied by each constructor's name; no constructor takes an
# order argument any more, so a mechanism cannot be re-purposed as another one.
FIXED_ORDER = {
    "adv": 1, "prs": 1, "dif": 2, "disp": 3, "hdif": 4,
    "rxn": 0, "frc": 0, "cpl": 0, "S": 0, "MUL": 0, "wave": 0,
    "coord_sin": 0, "coord_cos": 0,
    "DIV": 0, "T": 0, "coord": 0, "POW": 0, "ABS": 0, "X": 0,
}


# ------------------------------------------------------- growable mechanisms
# The library above is the fixed base. `register_mechanism` lets the expert loop
# add a *new* mechanism at run time, provided it is a fixed differential operator
# written only from the primitives (D / S / MUL) and the mechanisms that already
# exist. It never accepts a free-form expression: the definition is a closed
# algebra, so what gets promoted is a named physical operator, not a fitted term.
_EXTRA: dict[str, dict] = {}


def _definition_order(node) -> int:
    """True differential order of a definition expression (additive through D)."""
    if isinstance(node, ast.Call):
        name = node.func.id
        args = list(node.args) + [kw.value for kw in node.keywords]
        if name == "D":
            order = 1
            if len(node.args) > 2 and isinstance(node.args[2], ast.Constant):
                order = int(node.args[2].value)
            for kw in node.keywords:
                if kw.arg == "order" and isinstance(kw.value, ast.Constant):
                    order = int(kw.value.value)
            return order + (_definition_order(node.args[0]) if node.args else 0)
        base = FIXED_ORDER.get(name, 0)
        inner = max([_definition_order(a) for a in args] or [0])
        return base + inner
    if isinstance(node, ast.Name) or isinstance(node, ast.Constant):
        return 0
    return max([_definition_order(c) for c in ast.iter_child_nodes(node)] or [0])


def validate_definition(signature, definition: str, order: int) -> str | None:
    """Return an error string when a mechanism definition is not admissible."""
    if not definition or not definition.strip():
        return "empty definition"
    try:
        tree = ast.parse(definition, mode="eval")
    except SyntaxError as error:
        return f"not a valid expression: {error}"
    allowed_calls = ({"D", "S", "MUL", "DIV", "POW", "ABS", "X", "T", "coord"}
                     | set(MECHANISM_NAMES))
    allowed_names = set(signature)

    def walk(node):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("only plain calls are allowed")
            fn = node.func.id
            if fn not in allowed_calls:
                raise ValueError(f"`{fn}` is outside the mechanism algebra")
            for arg in node.args:
                walk(arg)
            for kw in node.keywords:
                walk(kw.value)
            return
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ValueError(
                    f"`{node.id}` is not a formal of the signature {signature}")
            return
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                raise ValueError("only integer constants and scalar-function names")
            if not isinstance(node.value, int):
                if not (isinstance(node.value, str) and node.value in _FUNCS):
                    raise ValueError("only integer constants and scalar-function "
                                     "names (id/sin/cos/exp/tanh) may appear")
            return
        # `a * b` / `a / b` / `a + b` are the same fixed operators as MUL/DIV and are
        # accepted as sugar, so a natural definition is not thrown away over spelling.
        if isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Mult, ast.Div, ast.Add, ast.Sub)):
            walk(node.left)
            walk(node.right)
            return
        raise ValueError(f"`{type(node).__name__}` is not allowed here")

    try:
        walk(tree.body)
    except ValueError as error:
        return str(error)
    computed = _definition_order(tree.body)
    if computed != int(order):
        return f"declared order {order} but the definition computes order {computed}"
    return None


def register_mechanism(spec: dict) -> str:
    """Compile and install one mechanism. Raises ValueError when inadmissible."""
    name = str(spec["name"]).strip()
    if not name.isidentifier() or name in PRIMITIVE_NAMES or name in MECHANISM_NAMES:
        raise ValueError(f"`{name}` is not a fresh identifier")
    signature = [str(s) for s in spec["signature"]]
    for formal in signature:
        if not formal.isidentifier():
            raise ValueError(f"`{formal}` is not a valid formal name")
    if spec.get("expr") is not None:
        # Structured tree: our own compiler owns the syntax, so the string-level
        # checks (arity, names, brackets) are not needed and cannot fail here.
        expr = canonicalize_axes(spec["expr"])
        # The signature is DERIVED from the tree, never trusted from the file: an
        # older registry may carry a signature that an earlier version mis-derived.
        signature = signature_of(expr, [])
        try:
            body, order = compile_expr(expr, signature)
        except ValueError as error:
            raise ValueError(str(error)) from None
        if order > 4:
            raise ValueError(f"computed order {order} exceeds 4")
        spec = dict(spec)
        spec["expr"] = expr
        spec["order"] = order
        spec["definition"] = body
    else:
        order = int(spec["order"])
        error = validate_definition(signature, spec["definition"], order)
        if error:
            raise ValueError(error)
        body = spec["definition"]
    fields = [s for s in signature if s != "axis"]
    guard = f"    _forbid_restricted({', '.join(fields)})\n" if fields else ""
    source = (f"def {name}({', '.join(signature)}):\n{guard}"
              f"    return {body}\n")
    namespace: dict = {}
    try:
        exec(source, globals(), namespace)
    except SyntaxError as error:
        # A formal that is a Python keyword, or any other name the generated `def`
        # cannot carry, must cost this one mechanism rather than the whole run.
        raise ValueError(f"`{name}` does not compile: {error}") from None
    function = namespace[name]
    function.__doc__ = (f"[grown mechanism, order {order}] "
                        f"{spec.get('physics', '')}".strip())
    globals()[name] = function
    MECHANISM_NAMES.add(name)
    FIXED_ORDER[name] = order
    _EXTRA[name] = dict(spec)
    return name


def unregister_mechanism(name: str) -> None:
    if name in _EXTRA:
        _EXTRA.pop(name, None)
        MECHANISM_NAMES.discard(name)
        FIXED_ORDER.pop(name, None)
        globals().pop(name, None)


def load_extra_from_env() -> None:
    """Subprocesses re-import this module, so the grown library is re-read here.

    `main.py` exports `PDE_GRAMMAR_REGISTRY` to the child processes it spawns;
    re-registering at import keeps `G.<new mechanism>` resolvable inside the
    sandbox (where the candidate code actually calls it).
    """
    path = os.environ.get("PDE_GRAMMAR_REGISTRY", "")
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        for spec in payload.get("mechanisms", []):
            if spec.get("builtin"):
                continue
            try:
                register_mechanism(spec)
            except Exception:
                continue
    except Exception:
        return


load_extra_from_env()
