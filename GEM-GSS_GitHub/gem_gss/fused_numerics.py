"""Guarded 2D-only discrete derivative/window fusion; original methods intact.

Accelerate parameter-dependent evaluations. Every returned model is evaluated
and linearly projected with the original operators. Rank/accuracy-sensitive
models and incumbent contenders receive a full original fit with the SAME input.
No graph, sampling, solver, starts, budget, reference equations or API changes.
"""

from collections import Counter, OrderedDict
from types import SimpleNamespace
import copy
import time
import numpy as np
from scipy.interpolate import make_interp_spline
from . import core as C
from . import fitting

BaseData = C.Data
original_fit = fitting.fit
original_final_equation = fitting.final_equation
GUARDS = dict(
    feature_relative=1e-09,
    prediction_relative_to_target=1e-09,
    MSE_relative=1e-06,
    coefficient_relative=1e-06,
    condition_limit=100000000.0,
)
_INSTANCES = []
_INSTALLED = False
_REFERENCE_ONLY = set()


class FusedData(BaseData):

    def __init__(self, case):
        super().__init__(case)
        if self.d != 2:
            raise ValueError("This numerical experiment supports 2D only")
        self.reference = BaseData.__new__(BaseData)
        self.reference.__dict__ = self.__dict__.copy()
        self.reference.rawcache = OrderedDict()
        self.reference.rawbytes = 0
        self.reference.feature_cache = OrderedDict()
        self.reference.featurebytes = 0
        self.operators = {}
        self.weights = {}
        self.reference_keys = set()
        self.checked = set()
        self.reference_only = set(_REFERENCE_ONLY)
        self.stats = Counter()
        self.best_verified = {}
        _INSTANCES.append(self)

    def register(self, model):
        for term in model["terms"]:
            self.reference_keys.add(
                (
                    term["expr"],
                    tuple((float(term["params"][p]) for p in sorted(term["params"]))),
                    False,
                )
            )

    def feature(self, node, params, full=False):
        key = (node["expr"], tuple(map(float, params)), full)
        if key in self.reference_keys or node["expr"] in self.reference_only:
            self.feature_calls += 1
            self.stats["reference_feature_calls"] += 1
            return self.reference.feature(node, params, full)
        expression = self.parsed(node["expr"])
        chain = []
        while expression.op == "PD":
            chain.append(tuple(expression.value))
            expression = expression.args[0]
        if not chain:
            return super().feature(node, params, full)
        self.feature_calls += 1
        if key in self.feature_cache:
            self.feature_cache.move_to_end(key)
            self.stats["fused_cache_hits"] += 1
            return self.feature_cache[key]
        started = time.perf_counter()
        chain = tuple(chain)
        if chain not in self.weights:
            qs = []
            bound = 1.0
            for axis, phi in enumerate(self.phis):
                q = phi.T.copy()
                for alpha in chain:
                    order = alpha[axis]
                    if not order:
                        continue
                    opkey = (axis, order)
                    if opkey not in self.operators:
                        x = self.axes[axis]
                        self.operators[opkey] = make_interp_spline(x, np.eye(len(x)), k=5, axis=0)(
                            x, nu=order
                        )
                    operator = self.operators[opkey]
                    q = q @ operator
                    bound *= max(float(np.linalg.norm(operator, ord=np.inf)), 1.0)
                qs.append(q)
            self.weights[chain] = (qs, bound)
        weights, bound = self.weights[chain]
        with np.errstate(all="ignore"):
            value = np.broadcast_to(
                self.raw(expression, dict(zip(node["params"], params)), full), self.shape
            )
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite feature input")
        if float(np.max(np.abs(value))) > np.sqrt(np.finfo(float).max) / max(bound, 1.0):
            self.stats["large_dynamic_range_fallback"] += 1
            return self.reference.feature(node, params, full)
        for axis, q in enumerate(weights):
            value = np.moveaxis(np.tensordot(q, value, axes=(1, axis)), 0, axis)
        vec = value.ravel()
        if not np.isfinite(vec).all():
            raise ValueError("Nonfinite fused feature")
        if node["expr"] not in self.checked:
            reference = self.reference.feature(node, params, full)
            error = float(
                np.linalg.norm(vec - reference)
                / max(np.linalg.norm(reference), np.finfo(float).eps * np.sqrt(len(reference)))
            )
            self.stats["feature_reference_checks"] += 1
            self.checked.add(node["expr"])
            if error > GUARDS["feature_relative"]:
                self.reference_only.add(node["expr"])
                self.stats["feature_guard_fallback"] += 1
                return reference
        self.feature_cache[key] = vec
        self.featurebytes += vec.nbytes
        while self.featurebytes > 32 * 1024**2 and self.feature_cache:
            _, old = self.feature_cache.popitem(last=False)
            self.featurebytes -= old.nbytes
        self.stats["fused_feature_evaluations"] += 1
        self.stats["fused_feature_seconds"] += time.perf_counter() - started
        return vec


def owner(data):
    return (
        data
        if isinstance(data, FusedData)
        else getattr(getattr(data, "feature", None), "__self__", None)
    )


def reference_view(data, backend):
    return SimpleNamespace(
        y=data.y, scale=data.scale, feature=backend.reference.feature, fit_calls=0, fit_nfev=0
    )


def rank_condition(A, free):
    if not np.any(free):
        return (0, 1.0)
    matrix = A[:, free]
    norm = np.maximum(np.linalg.norm(matrix, axis=0), 1e-20)
    singular = np.linalg.svd(matrix / norm, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-09))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    return (rank, condition)


def fit(data, field, nodes, registry, initial=None, full=True, shared=False):
    backend = owner(data)
    if not isinstance(backend, FusedData):
        return original_fit(data, field, nodes, registry, initial, full, shared)
    started = time.perf_counter()
    backend.stats["logical_fit_calls"] += 1

    def baseline(reason, already_counted=False):
        view = reference_view(data, backend)
        t = time.perf_counter()
        result = original_fit(view, field, nodes, registry, initial, full, shared)
        if not already_counted:
            data.fit_calls += 1
        data.fit_nfev += view.fit_nfev
        backend.stats["baseline_fit_seconds"] += time.perf_counter() - t
        backend.stats["baseline_" + reason] += 1
        if result:
            backend.register(result)
            backend.best_verified[field] = min(
                backend.best_verified.get(field, float("inf")), result["score"]
            )
        backend.stats["total_fit_seconds"] += time.perf_counter() - started
        return result

    if not nodes or not any((n["params"] for n in nodes)):
        return baseline("no_internal_parameters")
    if not any((n["params"] and backend.parsed(n["expr"]).op == "PD" for n in nodes)):
        return baseline("no_fusable_parameterized_derivative")
    if any((n["expr"] in backend.reference_only for n in nodes)):
        return baseline("preflight_or_feature_guard")
    trial_start = time.perf_counter()
    model = original_fit(data, field, nodes, registry, initial, full, shared)
    backend.stats["accelerated_optimizer_seconds"] += time.perf_counter() - trial_start
    if model is None:
        return baseline("invalid_fast_result", True)
    try:
        verification_start = time.perf_counter()
        terms = model["terms"]
        fast_columns = []
        reference_columns = []
        for term in terms:
            params = [term["params"][p] for p in sorted(term["params"])]
            node = dict(expr=term["expr"], params=sorted(term["params"]))
            fast_columns.append(data.feature(node, params))
            reference_columns.append(backend.reference.feature(node, params))
        A = np.column_stack(fast_columns)
        B = np.column_stack(reference_columns)
        coefficients = np.array([t["coefficient"] for t in terms])
        fixed = np.array([t.get("fixed_coefficient", False) for t in terms])
        free = ~fixed
        rank_fast, cond_fast = rank_condition(A, free)
        rank_reference, cond_reference = rank_condition(B, free)
        if rank_fast != rank_reference:
            return baseline("rank_change", True)
        if max(cond_fast, cond_reference) > GUARDS["condition_limit"]:
            return baseline("ill_conditioned", True)
        refcoef = np.ones(len(terms))
        target = data.y[field] - B[:, fixed] @ refcoef[fixed]
        if np.any(free):
            norm = np.maximum(np.linalg.norm(B[:, free], axis=0), 1e-20)
            refcoef[free] = np.linalg.lstsq(B[:, free] / norm, target, rcond=1e-09)[0] / norm
        fast_pred = A @ coefficients
        ref_pred = B @ refcoef
        y = data.y[field]
        mse = float(np.mean((ref_pred - y) ** 2))
        pred_difference = float(
            np.linalg.norm(fast_pred - ref_pred) / max(np.linalg.norm(y), 1e-30)
        )
        mse_difference = abs(mse - model["fit_mse"]) / max(
            mse, np.finfo(float).eps ** 2 * data.scale[field]
        )
        coef_difference = float(
            np.linalg.norm(refcoef - coefficients) / max(np.linalg.norm(refcoef), 1e-30)
        )
        backend.stats["reference_verification_seconds"] += time.perf_counter() - verification_start
        backend.stats["reference_verified_fits"] += 1
        if pred_difference > GUARDS["prediction_relative_to_target"]:
            return baseline("prediction_guard", True)
        if mse_difference > GUARDS["MSE_relative"]:
            return baseline("MSE_guard", True)
        if coef_difference > GUARDS["coefficient_relative"]:
            return baseline("coefficient_guard", True)
        canonical = copy.deepcopy(model)
        for term, c in zip(canonical["terms"], refcoef):
            term["coefficient"] = float(c)
        canonical["fit_mse"] = mse
        canonical["score"] = float(
            (mse / data.scale[field] + 1e-14) * np.exp(0.1 * model["complexity"])
        )
        if canonical["score"] <= backend.best_verified.get(field, 1.0) * (
            1 + GUARDS["MSE_relative"]
        ):
            return baseline("incumbent_contender", True)
        backend.register(canonical)
        backend.stats["accepted_accelerated_fits"] += 1
        backend.stats["total_fit_seconds"] += time.perf_counter() - started
        return canonical
    except (ValueError, ArithmeticError, np.linalg.LinAlgError):
        return baseline("verification_exception", True)


def final_equation(data, field, model):
    backend = owner(data)
    if isinstance(backend, FusedData):
        return original_final_equation(reference_view(data, backend), field, model)
    return original_final_equation(data, field, model)


def install(reference_only_expressions=()):
    global _INSTALLED
    _REFERENCE_ONLY.update(reference_only_expressions)
    if not _INSTALLED:
        C.Data = FusedData
        fitting.fit = fit
        fitting.final_equation = final_equation
        _INSTALLED = True


def statistics():
    result = Counter()
    for data in _INSTANCES:
        result.update(data.stats)
    return dict(result)
