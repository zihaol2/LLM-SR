"""A bounded statistics executor: the LLM chooses WHAT to measure, we compute it.

No physical identity is given to the model. It receives the structural description
of the anonymous arrays and a catalogue of statistics it may request, then returns a
JSON plan. This module validates that plan (known op, known array, bounded sizes) and
executes it, returning a readable report.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import make_interp_spline


OPS = """
Descriptive statistics
  summary(field)                       min, max, mean, std, rms, negative fraction, quartiles
  space_spectrum(field, at_time)       strongest spatial wavenumbers of one time slice
  time_spectrum(field, at_x)           strongest temporal frequencies at one position
  drift(field)                         rms of the second half over the first half, and the two means
  cross_correlation(field_a, field_b)  Pearson correlation over all points
  derivative_rms(field, orders)        rms of the spatial derivatives of the given orders
  correlate_with_target(field, order)  correlation between the spatial derivative of a field
                                       and that field's measured time derivative
  sample_profile(field, times, points) a few sample rows at a few times

Discriminative statistics. Each one separates competing functional explanations of the
same data instead of merely describing it:
  balance_test(target)                 is the domain average of the target constant in time
                                       (consistent with a pure boundary flux) or does it vary
                                       (consistent with a distributed source), and does the
                                       domain average of the field itself drift
  nested_fit(field)                    R2 of nested regressions of the target on the first
                                       derivative, then plus a state-times-derivative term,
                                       then plus a quadratic state term, and with a
                                       second-order derivative added, so a state-independent
                                       response can be told from a state-dependent one
  segmented_response(field, segments)  the same fitted response computed separately in
                                       spatial segments, so a response that varies in space
                                       can be told from one that does not
  state_conditioned_response(field, bins)
                                       the same local response computed in bins of the field
                                       value, so the response can be seen to weaken,
                                       strengthen or change sign with the state

`at_time` / `at_x` accept "first", "mid" or "last". `orders` is a list from {1,2}.
A plan is a JSON array of objects, each with an "op" and the arguments above, for
example {"op": "space_spectrum", "field": "f1", "at_time": "mid"}.
"""


MAX_OPS = 24
MAX_TOP = 6


class Data:
    def __init__(self, data: dict):
        self.data = data
        self.axes = sorted((k for k in data if k.startswith("x") and k[1:].isdigit()),
                           key=lambda s: int(s[1:]))
        self.fields = sorted((k for k in data if k.startswith("f") and k[1:].isdigit()
                              and not k.endswith("_t_true")), key=lambda s: int(s[1:]))
        self.x = np.asarray(data[self.axes[0]], dtype=float)

    def field(self, name):
        return np.asarray(self.data[name], dtype=float)

    def deriv(self, field, order=1):
        if order == 0:
            return field
        return make_interp_spline(self.x, field, k=5, axis=0)(self.x, nu=order)

    def slice_index(self, key, n):
        if key in ("mid", None):
            return n // 2
        if key == "first":
            return 0
        if key == "last":
            return n - 1
        return int(key) % n


def _peaks(power, axis, top):
    order = np.argsort(power)[::-1][:top]
    total = max(float(power.sum()), 1e-300)
    return [(int(i), float(power[i] / total)) for i in order]


def execute(plan, data: dict):
    d = Data(data)
    report = []
    if not isinstance(plan, list):
        return ["plan was not a list"], []
    kept = []
    for request in plan[:MAX_OPS]:
        if not isinstance(request, dict):
            continue
        op = str(request.get("op", "")).strip()
        try:
            if op == "summary":
                name = str(request.get("field"))
                f = d.field(name)
                q = np.percentile(f, [25, 50, 75])
                report.append(
                    f"summary {name}: min {f.min():+.4f} max {f.max():+.4f} "
                    f"mean {f.mean():+.4f} std {f.std():.4f} rms {np.sqrt(np.mean(f**2)):.4f} "
                    f"negative fraction {np.mean(f < 0):.3f} "
                    f"quartiles {q[0]:+.4f}/{q[1]:+.4f}/{q[2]:+.4f}")
            elif op == "space_spectrum":
                name = str(request.get("field"))
                f = d.field(name)
                j = d.slice_index(request.get("at_time"), f.shape[1])
                row = f[:, j] - f[:, j].mean()
                power = np.abs(np.fft.rfft(row)) ** 2
                top = min(int(request.get("top", 4)), MAX_TOP)
                peaks = _peaks(power[1:], None, top)
                txt = ", ".join(f"k={k + 1} ({s * 100:.0f}%)" for k, s in peaks)
                report.append(f"space_spectrum {name} at_time={request.get('at_time', 'mid')}: {txt}")
            elif op == "time_spectrum":
                name = str(request.get("field"))
                f = d.field(name)
                i = d.slice_index(request.get("at_x"), f.shape[0])
                series = f[i, :] - f[i, :].mean()
                power = np.abs(np.fft.rfft(series)) ** 2
                top = min(int(request.get("top", 4)), MAX_TOP)
                peaks = _peaks(power[1:], None, top)
                txt = ", ".join(f"mode {k + 1} ({s * 100:.0f}%)" for k, s in peaks)
                report.append(f"time_spectrum {name} at_x={request.get('at_x', 'mid')}: {txt}")
            elif op == "drift":
                name = str(request.get("field"))
                f = d.field(name)
                half = f.shape[1] // 2
                a = float(np.sqrt(np.mean(f[:, :half] ** 2)))
                b = float(np.sqrt(np.mean(f[:, half:] ** 2)))
                report.append(f"drift {name}: rms second/first {b / max(a, 1e-30):.3f} "
                              f"(first {a:.4f}, second {b:.4f}), "
                              f"mean first {f[:, :half].mean():+.4f}, "
                              f"mean second {f[:, half:].mean():+.4f}")
            elif op == "cross_correlation":
                a_name, b_name = str(request.get("field_a")), str(request.get("field_b"))
                a, b = d.field(a_name).ravel(), d.field(b_name).ravel()
                r = float(np.corrcoef(a, b)[0, 1])
                report.append(f"cross_correlation {a_name},{b_name}: {r:+.3f}")
            elif op == "derivative_rms":
                name = str(request.get("field"))
                f = d.field(name)
                orders = request.get("orders") or [1, 2]
                parts = []
                for order in orders:
                    order = int(order)
                    if order not in (1, 2):
                        continue
                    g = d.deriv(f, order)
                    parts.append(f"order {order}: rms {np.sqrt(np.mean(g**2)):.4f}")
                report.append(f"derivative_rms {name}: " + ", ".join(parts))
            elif op == "correlate_with_target":
                name = str(request.get("field"))
                order = int(request.get("order", 0))
                if order not in (0, 1, 2):
                    order = 0
                target = f"{name}_t_true"
                if target not in data:
                    report.append(f"correlate_with_target {name}: target {target} not present")
                    continue
                g = d.deriv(d.field(name), order)
                r = float(np.corrcoef(g.ravel(), d.field(target).ravel())[0, 1])
                report.append(f"correlate_with_target {name} order {order}: {r:+.3f}")
            elif op == "sample_profile":
                name = str(request.get("field"))
                f = d.field(name)
                times = request.get("times") or ["first", "mid", "last"]
                points = min(int(request.get("points", 8)), 12)
                parts = []
                for key in times:
                    j = d.slice_index(key, f.shape[1])
                    row = f[::max(1, f.shape[0] // points), j][:points]
                    parts.append(f"{key}: " + " ".join(f"{v:+.3f}" for v in row))
                report.append(f"sample_profile {name}: " + " | ".join(parts))
            elif op == "balance_test":
                target = str(request.get("target"))
                field_name = target[:-len("_t_true")]
                y = d.field(target)
                f = d.field(field_name)
                dom_y = y.mean(axis=0)
                dom_f = f.mean(axis=0)
                rel = abs(float(dom_y.mean())) / max(float(np.sqrt(np.mean(y ** 2))), 1e-30)
                drift = (float(dom_f[-1]) - float(dom_f[0])) / max(
                    float(np.sqrt(np.mean(f ** 2))), 1e-30)
                report.append(
                    f"balance_test {field_name}: domain average of {target} is "
                    f"{float(dom_y.mean()):+.6f} (std {float(dom_y.std()):.6f}), "
                    f"|domain average| / rms(target) = {rel:.5f}; the domain average of "
                    f"{field_name} changes by {drift:+.2%} of its rms over the record")
            elif op == "nested_fit":
                name = str(request.get("field"))
                target = f"{name}_t_true"
                f = d.field(name)
                y = d.field(target).ravel()
                d1 = d.deriv(f, 1)
                d2 = d.deriv(f, 2)
                base = float(np.mean((y - y.mean()) ** 2))

                def r2(columns):
                    A = np.stack([c.ravel() for c in columns], axis=1)
                    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                    resid = y - A @ coef
                    return 1.0 - float(np.mean(resid ** 2)) / max(base, 1e-30)

                parts = []
                parts.append(f"[d1] R2 {r2([d1]):.4f}")
                parts.append(f"[d1,f*d1] R2 {r2([d1, f * d1]):.4f}")
                parts.append(f"[d1,f*d1,f^2*d1] R2 {r2([d1, f * d1, (f ** 2) * d1]):.4f}")
                parts.append(f"[d1,d2] R2 {r2([d1, d2]):.4f}")
                parts.append(f"[d1,f*d1,d2] R2 {r2([d1, f * d1, d2]):.4f}")
                report.append(f"nested_fit {name}: " + ", ".join(parts))
            elif op == "segmented_response":
                name = str(request.get("field"))
                target = f"{name}_t_true"
                f = d.field(name)
                y = d.field(target)
                d1 = d.deriv(f, 1)
                n = f.shape[0]
                edges = np.linspace(0, n, 4).astype(int)
                parts = []
                for s in range(3):
                    lo, hi = edges[s], edges[s + 1]
                    A = d1[lo:hi].ravel()
                    B = y[lo:hi].ravel()
                    coef = float(np.dot(A, B) / max(np.dot(A, A), 1e-30))
                    parts.append(f"segment {s + 1} coefficient {coef:+.3f}")
                report.append(f"segmented_response {name}: " + ", ".join(parts))
            elif op == "state_conditioned_response":
                name = str(request.get("field"))
                target = f"{name}_t_true"
                f = d.field(name)
                y = d.field(target)
                d1 = d.deriv(f, 1)
                bins = min(int(request.get("bins", 5)), 8)
                lo_v, hi_v = float(f.min()), float(f.max())
                edges = np.linspace(lo_v, hi_v + 1e-12, bins + 1)
                parts = []
                for b in range(bins):
                    lo, hi = edges[b], edges[b + 1]
                    mask = (f >= lo) & (f < hi) if b < bins - 1 else (f >= lo) & (f <= hi)
                    A = d1[mask]
                    B = y[mask]
                    if A.size < 50:
                        continue
                    coef = float(np.dot(A, B) / max(np.dot(A, A), 1e-30))
                    parts.append(f"state {float(f[mask].mean()):+.3f} "
                                 f"(n={int(mask.sum())}) -> response {coef:+.3f}")
                report.append(f"state_conditioned_response {name}: " + ", ".join(parts))
            else:
                continue
            kept.append(request)
        except Exception as error:                                   # noqa: BLE001
            report.append(f"{op}: failed ({type(error).__name__})")
    return report, kept
