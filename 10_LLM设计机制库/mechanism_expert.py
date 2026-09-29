"""The LLM-as-physics-expert loop that grows the mechanism library.

One round: show the current library + the measured data evidence, let the model
propose named fixed-order operators (it may use the new primitives DIV / T / coord),
validate them, and promote only the ones that measurably enlarge what a short
equation can express on a held-out split.

The gate is self-contained: a small greedy forward selection over the built-in
atoms, compared with the same selection once the proposal's atoms are added, on
rows the coefficients never saw. No screening/coverage machinery involved.
"""
from __future__ import annotations

import ast
import json
import os
import re

import numpy as np
from scipy.interpolate import make_interp_spline

import pde_grammar
import mechanism_registry


SYSTEM = (
    "You are a physicist who studies partial differential equations. You are looking "
    "at an anonymous dataset (fields f1..fN and their measured time derivatives) and "
    "at the operator library a machine may search. Your job is to extend that library "
    "with new physical mechanisms it cannot express, so the search can build better "
    "equations.\n"
    "A mechanism is ONE named operator with a FIXED differential order. Write it as a "
    "closed algebraic definition over the primitives D(f, axis, order), S(f, func), "
    "MUL(a, b), DIV(a, b), T(f), coord(func, k1, k2, k3) and the mechanisms that "
    "already exist. No numpy, no arithmetic operators, no attributes, no free-form "
    "expressions, no fitted coefficients -- only the fixed structure.\n"
)


# ------------------------------------------------------------------ binding
def _axes(data):
    return sorted((k for k in data if re.fullmatch(r"x\d+", k)), key=lambda s: int(s[1:]))


def _fields(data):
    return sorted((k for k in data if re.fullmatch(r"f\d+", k)), key=lambda s: int(s[1:]))


def _targets(data):
    return sorted((k for k in data if k.endswith("_t_true")),
                  key=lambda s: int(s[1:].split("_")[0]))


def bind_data_grammar(data):
    """Give `pde_grammar` a derivative operator in *this* process."""
    axes = _axes(data)
    fields = _fields(data)
    coords = [np.asarray(data[a], dtype=float) for a in axes]
    field_shape = np.shape(data[fields[0]])

    def deriv(field, axis, order):
        if order == 0:
            return field
        coord = coords[axis]
        return make_interp_spline(coord, field, k=5, axis=axis)(coord, nu=order)

    grid = []
    for axis, coord in enumerate(coords):
        shape = [1] * len(field_shape)
        shape[axis] = coord.size
        grid.append(coord.reshape(shape))
    if len(grid) == 1:
        grid.append(np.zeros_like(grid[0]))
    vector = tuple(np.asarray(data[f], dtype=float) for f in fields[:len(axes)])
    pde_grammar.bind(tuple(range(len(axes))), deriv, vector=vector, coords=tuple(grid))
    return deriv


# ------------------------------------------------------------------ the gate
_FUNCS = {"id": lambda a: a, "sin": np.sin, "cos": np.cos,
          "exp": np.exp, "tanh": np.tanh}


def base_atoms(data):
    """A compact library of built-in mechanism atoms, for the held-out comparison."""
    axes = _axes(data)
    fields = _fields(data)
    coords = [np.asarray(data[a], dtype=float) for a in axes]
    field_shape = np.shape(data[fields[0]])
    grid = []
    for axis, coord in enumerate(coords):
        shape = [1] * len(field_shape)
        shape[axis] = coord.size
        grid.append(coord.reshape(shape))
    if len(grid) == 1:
        grid.append(np.zeros_like(grid[0]))

    def deriv(field, axis, order):
        return make_interp_spline(coords[axis], field, k=5, axis=axis)(coords[axis], nu=order)

    atoms = []
    for f in fields:
        arr = np.asarray(data[f], dtype=float)
        atoms.append((f"frc({f})", arr))
        for axis in range(len(axes)):
            atoms.append((f"prs({f},{axis})", deriv(arr, axis, 1)))
        atoms.append((f"dif({f})", sum(deriv(arr, a, 2) for a in range(len(axes)))))
        for func in ("id", "sin", "cos", "exp", "tanh"):
            atoms.append((f"rxn({f},{func})", _FUNCS[func](arr)))
    for i, f in enumerate(fields):
        for g in fields[i:]:
            atoms.append((f"cpl({f},{g})", np.asarray(data[f]) * np.asarray(data[g])))
    for func in ("sin", "cos", "exp", "tanh"):
        for k in (-2, -1, 1, 2):
            arg = k * grid[0]
            atoms.append((f"coord({func},{k})",
                          np.broadcast_to(_FUNCS[func](arg), field_shape)))
    return atoms


def mechanism_atoms(spec, data):
    function = getattr(pde_grammar, spec["name"], None)
    if function is None:
        return []
    signature = list(spec["signature"])
    fields = _fields(data)
    axes = range(len(_axes(data)))
    field_like = {"phi", "psi", "f", "u", "v", "field", "scalar"}
    slots = [s for s in signature if s != "axis" and (s in field_like or s in fields)]
    # what to try for each non-axis formal: the measured fields, or a sign for a
    # scalar coefficient (the search fits the real value later)
    options = []
    for formal in signature:
        if formal == "axis":
            continue
        options.append([(f, np.asarray(data[f], dtype=float)) for f in fields]
                       if formal in slots else [(formal, 1.0), (formal, -1.0), (formal, 0.0)])
    field_shape = np.shape(data[fields[0]])
    out = []
    import itertools
    for choice in itertools.product(*options):
        for axis in (axes if "axis" in signature else [None]):
            args, labels = [], []
            k = 0
            for formal in signature:
                if formal == "axis":
                    args.append(axis)
                    labels.append(str(axis))
                else:
                    label, value = choice[k]
                    args.append(value)
                    labels.append(label if isinstance(label, str) and label in fields
                                  else f"{formal}={value:+.0f}")
                    k += 1
            try:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    array = np.asarray(function(*args), dtype=float)
                if not np.all(np.isfinite(array)):
                    continue
                if array.shape != field_shape:
                    array = np.broadcast_to(array, field_shape)
                out.append((f"{spec['name']}({','.join(labels)})", array))
            except Exception:
                continue
            if len(out) >= 60:
                return out
    return out


def _split_residual(atoms, data, train, test, budget):
    """Held-out residual of a greedy `budget`-atom fit (summed over targets)."""
    targets = _targets(data)
    columns = {}
    for name, arr in atoms:
        values = np.asarray(arr, dtype=float).ravel()
        if np.all(np.isfinite(values)):
            columns[name] = values
    names = list(columns)
    y = {t: np.asarray(data[t], dtype=float).ravel() for t in targets}
    chosen, base = [], 0.0
    for t in targets:
        base += float(np.mean((y[t][test] - y[t][train].mean()) ** 2))
    for _ in range(budget):
        best, best_name = None, None
        for name in names:
            if name in chosen:
                continue
            cols = [columns[c] for c in chosen + [name]]
            A = np.stack(cols, axis=1)
            total = 0.0
            for t in targets:
                coef, *_ = np.linalg.lstsq(A[train], y[t][train], rcond=None)
                total += float(np.mean((y[t][test] - A[test] @ coef) ** 2))
            if best is None or total < best:
                best, best_name = total, name
        if best is None or best > base * 0.999:
            break
        chosen.append(best_name)
        base = best
    return base, chosen


def gate(spec, data, threshold=3.0, budget=3, seed=0, test_frac=0.3):
    detail = {"name": spec["name"], "order": spec["order"]}
    bind_data_grammar(data)
    try:
        pde_grammar.register_mechanism(spec)
    except Exception as error:
        detail["reason"] = f"could not be compiled: {error}"
        return False, detail
    new_atoms = mechanism_atoms(spec, data)
    detail["n_atoms"] = len(new_atoms)
    if not new_atoms:
        pde_grammar.unregister_mechanism(spec["name"])
        detail["reason"] = "produced no usable atom on this problem"
        return False, detail

    base = base_atoms(data)
    n = np.asarray(data[_targets(data)[0]], dtype=float).size
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = max(5, int(test_frac * n))
    test, train = order[:n_test], order[n_test:]
    r0, _ = _split_residual(base, data, train, test, budget)
    r1, picked = _split_residual(base + new_atoms, data, train, test, budget)
    detail["base_residual"] = r0
    detail["with_proposal_residual"] = r1
    detail["ratio"] = (r1 / r0) if r0 > 0 else 1.0
    detail["used_atom"] = picked[-1] if picked else ""
    if r0 > 0 and r1 <= r0 / threshold:
        return True, detail
    pde_grammar.unregister_mechanism(spec["name"])
    detail["reason"] = (f"held-out residual ratio {detail['ratio']:.3f} did not beat "
                        f"1/{threshold}")
    return False, detail


# ------------------------------------------------------------- prompt / parse
def expert_prompt(registry, data, n_proposals=3):
    targets = _targets(data)
    lines = [
        SYSTEM,
        "### CURRENT MECHANISM LIBRARY",
        registry.render_prompt(),
        "",
        "### DATASET",
        f"axes: {', '.join(_axes(data))}   fields: {', '.join(_fields(data))}   "
        f"targets: {', '.join(targets)}",
        f"field statistics: "
        + "; ".join(f"{f} in [{float(np.min(data[f])):+.3f}, "
                    f"{float(np.max(data[f])):+.3f}], "
                    f"mean {float(np.mean(data[f])):+.3f}" for f in _fields(data)),
    ]
    rejected = registry.rejected_block()
    if rejected:
        lines += ["", rejected]
    lines += [
        "",
        f"### TASK: propose at most {n_proposals} NEW mechanisms",
        "Name an operator the library lacks, state its true differential order, and "
        "say which physical process it represents. You may use DIV, T and coord inside "
        "the definition. Answer with ONE JSON array, nothing else:",
        "[",
        '  {"name": "lowercase_identifier",',
        '   "signature": ["phi", "psi", "axis"],',
        '   "order": 1,',
        '   "definition": "D(MUL(phi, psi), axis, 1)",',
        '   "physics": "what physical process this operator represents",',
        '   "rationale": "which measured regularity made you propose it"}',
        "]",
        "Rules: reference only the signature names and integer constants; call only "
        "D, S, MUL, DIV, T, coord or an existing mechanism; the declared order must "
        "equal the true order (orders add through D). If nothing is missing, return [].",
    ]
    return "\n".join(lines)


def parse_proposals(text: str):
    if not text:
        return []
    body = text.strip()
    if "```" in body:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", body, re.S)
        if blocks:
            body = max(blocks, key=len)
    try:
        payload = json.loads(body)
    except Exception:
        start, end = body.find("["), body.rfind("]")
        if start < 0 or end <= start:
            return []
        try:
            payload = json.loads(body[start:end + 1])
        except Exception:
            return []
    if isinstance(payload, dict):
        payload = [payload]
    return [p for p in payload if isinstance(p, dict)]


def validate_proposal(proposal: dict, registry):
    name = str(proposal.get("name", "")).strip()
    if not name or not name.isidentifier():
        return None, f"bad name `{name}`"
    if registry.has(name) or name in pde_grammar.MECHANISM_NAMES:
        return None, f"`{name}` already exists"
    signature = proposal.get("signature")
    if not isinstance(signature, list) or not signature:
        return None, "signature must be a non-empty list"
    signature = [str(s) for s in signature]
    if any(not s.isidentifier() for s in signature):
        return None, "signature entries must be identifier names"
    try:
        order = int(proposal.get("order"))
    except Exception:
        return None, "order must be an integer"
    definition = str(proposal.get("definition", "")).strip()
    # A definition that uses `axis` but forgot to declare it is a spelling slip, not
    # a wrong mechanism: declare it for the model instead of discarding the proposal.
    try:
        used = {n.id for n in ast.walk(ast.parse(definition, mode="eval"))
                if isinstance(n, ast.Name)}
    except Exception:
        used = set()
    if "axis" in used and "axis" not in signature:
        signature.append("axis")
    error = pde_grammar.validate_definition(signature, definition, order)
    if error:
        return None, error
    return {"name": name, "signature": signature, "order": order,
            "definition": definition,
            "physics": str(proposal.get("physics", "")).strip(),
            "rationale": str(proposal.get("rationale", "")).strip(),
            "builtin": False}, ""


def grow(registry, data, llm_call, rounds=1, proposals=3, threshold=3.0,
         seed=0, verbose=True):
    bind_data_grammar(data)
    promoted, log = [], []
    for round_index in range(max(1, rounds)):
        text = llm_call(expert_prompt(registry, data, proposals))
        parsed = parse_proposals(text)
        if verbose:
            print(f"[expert] round {round_index + 1}: {len(parsed)} proposal(s)",
                  flush=True)
            if not parsed:
                print(f"[expert]   raw reply: {' '.join((text or '').split())[:300]!r}",
                      flush=True)
        for proposal in parsed:
            spec, reason = validate_proposal(proposal, registry)
            if spec is None:
                if verbose:
                    print(f"[expert]   invalid: {reason}", flush=True)
                continue
            ok, detail = gate(spec, data, threshold=threshold, seed=seed + round_index)
            if ok:
                spec["provenance"] = {"by": "llm-expert", "round": round_index + 1,
                                      "ratio": detail.get("ratio"),
                                      "used_atom": detail.get("used_atom")}
                registry.add(spec)
                promoted.append(spec)
                log.append(f"PROMOTED {spec['name']} ratio {detail.get('ratio')}")
                if verbose:
                    print(f"[expert]   PROMOTED {spec['name']} order={spec['order']} "
                          f"ratio={detail.get('ratio'):.3f}", flush=True)
            else:
                registry.add({**spec, "status": "rejected",
                              "rejection": detail.get("reason", "no improvement")})
                log.append(f"rejected {spec['name']}: {detail.get('reason')}")
                if verbose:
                    print(f"[expert]   rejected {spec['name']}: "
                          f"{detail.get('reason')}", flush=True)
    return promoted, log

