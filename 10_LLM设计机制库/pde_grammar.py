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
import ast
import json
import os

import numpy as np

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
    order = int(spec["order"])
    error = validate_definition(signature, spec["definition"], order)
    if error:
        raise ValueError(error)
    fields = [s for s in signature if s != "axis"]
    guard = f"    _forbid_restricted({', '.join(fields)})\n" if fields else ""
    source = (f"def {name}({', '.join(signature)}):\n{guard}"
              f"    return {spec['definition']}\n")
    namespace: dict = {}
    exec(source, globals(), namespace)
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

