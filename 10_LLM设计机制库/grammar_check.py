"""Minimal AST layer for the physics-guided PDESR.

Only three jobs, nothing else:

  * `check_conformance` -- the operator algebra is closed. A candidate may build
    from the primitives (D, S, MUL, DIV, T, coord) and the mechanisms in
    `pde_grammar`, plus any mechanism the LLM expert has grown; anything else is
    rejected before it costs sandbox time.
  * `annotate` / `effective_complexity` -- price a candidate so that a mechanism
    call costs less than the same term hand-expanded from primitives, and terms
    above the declared order are penalised.
  * `structural_key` / `mechanism_summary` -- identity and a short reading.
"""
from __future__ import annotations

import ast
import dataclasses
import re
import textwrap

import pde_grammar

_FIELD_NAMES = frozenset(f"f{i}" for i in range(1, 10))


@dataclasses.dataclass(frozen=True)
class GrammarContext:
    n_max: int = 4
    penalty: int = 3
    axis_count: int = 1


@dataclasses.dataclass
class TermInfo:
    n_terms: int = 0
    n_mechanism_terms: int = 0
    n_primitive_terms: int = 0
    max_order: int = 0
    mechanisms: tuple = ()
    violations: int = 0
    parse_ok: bool = True


def _is_params(node: ast.AST) -> bool:
    return (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id == "params")


def _flatten_additive(node, sign=1, out=None):
    if out is None:
        out = []
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        _flatten_additive(node.left, sign, out)
        _flatten_additive(node.right, -sign if isinstance(node.op, ast.Sub) else sign, out)
    else:
        out.append((sign, node))
    return out


def _strip_scaling(node):
    while True:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            node = node.operand
            continue
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            if _is_params(node.left):
                node = node.right
                continue
            if _is_params(node.right):
                node = node.left
                continue
        break
    return node


def _symbol_table(tree):
    table = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            table.setdefault(node.targets[0].id, node.value)
    return table


def _resolve_root(node, table, depth=0):
    if depth < 8 and isinstance(node, ast.Name) and node.id in table:
        return _resolve_root(table[node.id], table, depth + 1)
    return node


def _walk_expanded(node, table, depth=0):
    if depth < 8 and isinstance(node, ast.Name) and node.id in table:
        yield from _walk_expanded(table[node.id], table, depth + 1)
        return
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _walk_expanded(child, table, depth + 1)


def _call_name(node):
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
    return None


def _norm_arg(node):
    text = ast.unparse(node)
    if text.startswith("G."):
        text = text[2:]
    return text.replace(" ", "").replace('"', "'")


def term_signature(node, table=None):
    root = _resolve_root(_strip_scaling(node), table or {})
    name = _call_name(root)
    if name not in pde_grammar.MECHANISM_NAMES:
        return None
    parts = [_norm_arg(a) for a in root.args]
    parts += [f"{kw.arg}={_norm_arg(kw.value)}" for kw in root.keywords]
    return f"{name}({','.join(parts)})"


def _call_order(name, call):
    if name in pde_grammar.FIXED_ORDER:
        return pde_grammar.FIXED_ORDER[name]
    if name == "D":
        for kw in call.keywords:
            if kw.arg == "order" and isinstance(kw.value, ast.Constant):
                return int(kw.value.value)
        if len(call.args) > 2 and isinstance(call.args[2], ast.Constant):
            return int(call.args[2].value)
        return 1
    return None


def _target_assignments(tree):
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id
                if name.endswith("_t_pred") or (name.startswith("f") and name.endswith("_t")):
                    found.append((name, node))
                    break
    return found


def _contains_field(node):
    return any(isinstance(sub, ast.Name) and sub.id in _FIELD_NAMES for sub in ast.walk(node))


def _is_field_product(node):
    return (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
            and _contains_field(node.left) and _contains_field(node.right))


def _mechanism_of(node, table):
    root = _resolve_root(_strip_scaling(node), table)
    name = _call_name(root)
    if name is None and _is_field_product(root):
        return "cpl"
    return name if name in pde_grammar.MECHANISM_NAMES else None


# ----------------------------------------------------------------- conformance
_SCALAR_ALIAS = re.compile(r"\bnp\.(sin|cos|exp|tanh)\(([^()]*(?:\([^()]*\)[^()]*)*)\)")


def normalise(body: str) -> str:
    """Rewrite `np.tanh(x)` etc. into the algebra's own spelling."""
    text = body or ""
    for _ in range(3):
        rewritten = _SCALAR_ALIAS.sub(
            lambda match: f"G.S({match.group(2)}, '{match.group(1)}')", text)
        if rewritten == text:
            break
        text = rewritten
    return text


_ALLOWED_NODES = (
    ast.Module, ast.Assign, ast.Name, ast.Load, ast.Store, ast.Attribute, ast.Call,
    ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.USub, ast.UAdd, ast.Pow, ast.Subscript, ast.Tuple, ast.Return, ast.Expr,
    ast.AugAssign,
)
_SAFETY_MARKERS = ("disallowed syntax", "outside the grammar", "not a grammar constructor",
                   "subscripting")
_DIFFERENTIAL = frozenset({"D", "adv", "dif", "disp", "hdif", "prs"})
_PRODUCT = frozenset({"MUL", "cpl"})
_SCALAR_FUNC = frozenset({"S", "rxn"})
_EXTRA_PRIMITIVES = frozenset({"DIV", "POW", "ABS", "X", "T", "coord", "frc", "wave",
                               "coord_sin", "coord_cos"})
_BASE_ALLOWED = (_DIFFERENTIAL | _PRODUCT | _SCALAR_FUNC | _EXTRA_PRIMITIVES) - {"D"}


def _allowed_calls():
    return set(_BASE_ALLOWED) | set(pde_grammar.MECHANISM_NAMES)


def is_safety_violation(why: str) -> bool:
    return any(marker in (why or "") for marker in _SAFETY_MARKERS)


def check_conformance(body: str, ctx: "GrammarContext | None" = None):
    try:
        tree = ast.parse(textwrap.dedent(body or ""))
    except SyntaxError:
        return True, ""
    allowed = _allowed_calls() | {"D"}
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return False, f"disallowed syntax: {type(node).__name__}"
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "G" and node.attr not in allowed:
            return False, f"`G.{node.attr}` is not a grammar constructor"
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if isinstance(node.func, ast.Attribute):
                base = node.func.value.id if isinstance(node.func.value, ast.Name) else None
                if base != "G":
                    return False, f"call to `{base}.{name}` is outside the grammar"
                if name not in allowed:
                    return False, f"`G.{name}` is not a grammar constructor"
            elif name not in allowed:
                return False, f"call to `{name}` is outside the grammar"
        if isinstance(node, ast.Subscript) and not _is_params(node):
            return False, "subscripting is outside the grammar"
    return True, ""


# ------------------------------------------------------------------- pricing
def annotate(body: str, ctx: GrammarContext) -> TermInfo:
    info = TermInfo()
    source = textwrap.dedent(body or "")
    if not source.strip():
        info.parse_ok = False
        return info
    try:
        tree = ast.parse(source)
    except SyntaxError:
        info.parse_ok = False
        return info
    mechanisms = set()
    table = _symbol_table(tree)
    for _name, assign in _target_assignments(tree):
        for _sign, term in _flatten_additive(assign.value):
            info.n_terms += 1
            mech = _mechanism_of(term, table)
            if mech:
                info.n_mechanism_terms += 1
                mechanisms.add(mech)
            else:
                info.n_primitive_terms += 1
            term_order = 0
            for sub in _walk_expanded(term, table):
                cname = _call_name(sub)
                if cname not in pde_grammar.MECHANISM_NAMES | pde_grammar.PRIMITIVE_NAMES:
                    continue
                order = _call_order(cname, sub)
                if order is not None:
                    term_order = max(term_order, order)
                if cname in pde_grammar.MECHANISM_NAMES:
                    mechanisms.add(cname)
            info.max_order = max(info.max_order, term_order)
            if term_order > ctx.n_max:
                info.violations += 1
    info.mechanisms = tuple(sorted(mechanisms))
    return info


def effective_complexity(raw_complexity: int, info: TermInfo, ctx: GrammarContext) -> int:
    value = raw_complexity - info.n_mechanism_terms + ctx.penalty * info.violations
    return max(1, int(value))


def structural_complexity(body: str) -> int:
    try:
        tree = ast.parse(textwrap.dedent(body or ""))
    except SyntaxError:
        return 1
    cost = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in _DIFFERENTIAL:
                cost += 2
            elif name in _PRODUCT or name in _EXTRA_PRIMITIVES or name in _SCALAR_FUNC:
                cost += 1
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                cost += 1
        elif _is_params(node):
            cost += 1
    return max(1, cost)


# -------------------------------------------------------------- identity / read
def structural_key(body: str) -> str:
    try:
        tree = ast.parse(textwrap.dedent(body or ""))
    except SyntaxError:
        return ""
    table = _symbol_table(tree)
    per_target = []
    for name, assign in _target_assignments(tree):
        terms = []
        for _sign, term in _flatten_additive(assign.value):
            mech = _mechanism_of(term, table)
            sig = term_signature(term, table) if mech else None
            terms.append(sig if sig else f"raw:{_norm_arg(term)[:60]}")
        per_target.append(f"{name}:{','.join(sorted(terms))}")
    return "|".join(sorted(per_target))


def mechanism_summary(body: str) -> str:
    try:
        tree = ast.parse(textwrap.dedent(body or ""))
    except SyntaxError:
        return ""
    table = _symbol_table(tree)
    counts = {}
    for _name, assign in _target_assignments(tree):
        for _sign, term in _flatten_additive(assign.value):
            mech = _mechanism_of(term, table)
            if mech:
                counts[mech] = counts.get(mech, 0) + 1
    return ", ".join(f"{k} x{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))

