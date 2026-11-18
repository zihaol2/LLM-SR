"""GEM55 variable projection, with warm starts and auditable optimization status."""

import numpy as np
from scipy.optimize import least_squares

from . import core as C
from .core import SETTINGS, complexity


def fit(data, field, nodes, registry, initial=None, full=True, shared=False):
    data.fit_calls += 1
    y = data.y[field]
    if not nodes:
        v = float(np.mean(y**2))
        return {
            "terms": [],
            "fit_mse": v,
            "complexity": 0,
            "score": v / data.scale[field],
            "fully_refined": True,
            "fit_nfev": 0,
            "optimizer": {
                "method": "variable_projection",
                "success": True,
                "nonlinear_parameters": 0,
                "linear_coefficients": 0,
            },
        }
    names = [n["params"] for n in nodes]
    slots = {}
    indices = []
    for i, ps in enumerate(names):
        ix = []
        for p in ps:
            key = p if shared else (i, p)
            if key not in slots:
                slots[key] = len(slots)
            ix.append(slots[key])
        indices.append(ix)
    count = len(slots)
    old = {t["id"]: t for t in (initial or {}).get("terms", [])}
    inherited = np.ones(count)
    for i, n in enumerate(nodes):
        if n["id"] in old:
            inherited[indices[i]] = [old[n["id"]]["params"].get(p, 1.0) for p in names[i]]
    starts = [inherited] + [np.full(count, p) for p in SETTINGS["coarse_parameters"]]
    newslots = [j for i, n in enumerate(nodes) if n["id"] not in old for j in indices[i]]
    for p in SETTINGS["coarse_parameters"]:
        q = inherited.copy()
        q[newslots] = p
        starts.append(q)
    starts = list({tuple(v): v for v in starts}.values())
    C = complexity(nodes, registry, shared)
    fixed = np.array([n.get("fixed_coefficient", False) for n in nodes])
    free = ~fixed

    def compute(p):
        A = np.column_stack([data.feature(n, p[indices[i]]) for i, n in enumerate(nodes)])
        c = np.ones(len(nodes))
        target = y - A[:, fixed] @ c[fixed]
        if np.any(free):
            norm = np.maximum(np.linalg.norm(A[:, free], axis=0), 1e-20)
            c[free] = np.linalg.lstsq(A[:, free] / norm, target, rcond=1e-09)[0] / norm
        return (A, c)

    evaluations = 0

    def loss(p):
        nonlocal evaluations
        evaluations += 1
        try:
            A, c = compute(p)
            return (A @ c - y) / np.sqrt(data.scale[field])
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            return np.full(len(y), 1000000.0)

    starts = sorted(starts, key=lambda p: float(np.mean(loss(p) ** 2)))[: 2 if count else 1]
    models = []
    nfev = 0
    statuses = []
    for p in starts:
        p = p.copy()
        if count:
            try:
                opt = least_squares(
                    loss,
                    p,
                    bounds=(-8, 8),
                    max_nfev=45 if full else 12,
                    ftol=1e-08,
                    xtol=1e-08,
                    gtol=1e-08,
                    diff_step=0.0001,
                )
                p = opt.x
                nfev += int(opt.nfev)
                statuses.append(
                    {
                        "success": bool(opt.success),
                        "status": int(opt.status),
                        "nfev": int(opt.nfev),
                        "message": str(opt.message),
                    }
                )
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as err:
                statuses.append({"success": False, "error": type(err).__name__})
        try:
            A, c = compute(p)
            pred = A @ c
            valid = float(np.mean((pred - y) ** 2))
            score = float((valid / data.scale[field] + 1e-14) * np.exp(0.1 * C))
            if not np.isfinite(score):
                continue
            terms = [
                {
                    "id": n["id"],
                    "expr": n["expr"],
                    "params": dict(zip(names[i], map(float, p[indices[i]]))),
                    "coefficient": float(c[i]),
                    "fixed_coefficient": bool(fixed[i]),
                }
                for i, n in enumerate(nodes)
            ]
            models.append(
                {
                    "terms": terms,
                    "fit_mse": valid,
                    "complexity": C,
                    "score": score,
                    "fully_refined": bool(full or not count),
                    "fit_nfev": nfev,
                    "loss_evaluations": evaluations,
                    "optimizer": {
                        "method": "variable_projection",
                        "nonlinear_parameters": count,
                        "linear_coefficients": int(np.sum(free)),
                        "starts": statuses,
                        "success": bool(not count or (statuses and statuses[-1]["success"])),
                    },
                }
            )
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            pass
    data.fit_nfev += evaluations
    return min(models, key=lambda m: m["score"]) if models else None


def prediction(data, model):
    if not model["terms"]:
        return np.zeros_like(next(iter(data.y.values())))
    columns = []
    for term in model["terms"]:
        names = sorted(term["params"])
        columns.append(
            data.feature(
                {"expr": term["expr"], "params": names},
                [term["params"][p] for p in names],
            )
        )
    return np.column_stack(columns) @ np.array([t["coefficient"] for t in model["terms"]])


def final_equation(data, field, model):
    mse = float(np.mean((prediction(data, model) - data.y[field]) ** 2))
    if not np.isclose(mse, model["fit_mse"], rtol=1e-10, atol=1e-25):
        raise RuntimeError("Final MSE differs from the selected projected model")
    return dict(
        terms=model["terms"],
        MSE=mse,
        score=model["score"],
        complexity=model["complexity"],
        optimizer=model["optimizer"],
        MSE_definition="All observed times, GEM spatial-window residual; not forecast error",
        final_refit=False,
    )


C.prediction = prediction


def make_data(case, backend=None):
    """Both backends share this module's variable-projection fitter."""
    backend = backend or C.SETTINGS["numerical_backend"]
    if backend == "reference":
        return C.Data(case)
    if backend == "compiled_chain_2d":
        from .chain_numerics import ChainData

        return ChainData(case)
    raise ValueError("Unknown numerical backend: " + str(backend))


def restore_models(data, archives, stats=None):
    if stats and hasattr(data, "stats"):
        data.stats.update(stats)


def statistics(data):
    if hasattr(data, "stats"):
        return dict(backend="compiled_chain_2d", counts=dict(data.stats))
    return dict(backend="reference")
