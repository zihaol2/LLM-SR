"""Score the probe answers with the same weak-form loss the paper's specs use."""
import inspect
import json
import os
import re
import sys
import textwrap

import numpy as np
import scipy.io
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PAPER_MSE = {
    "Topography_Chemotaxis": 7.6e-5,
    "Morphogenesis": 1.2e-10,
    "Forced_Swift_Hohenberg": 2.2e-4,
    "Traffic_Flow_Bottleneck": 1.6e-4,
    "Predator_Prey": 4.4e-9,
}
N_PARAMS = {
    "Topography_Chemotaxis": 5, "Morphogenesis": 5, "Forced_Swift_Hohenberg": 6,
    "Traffic_Flow_Bottleneck": 5, "Predator_Prey": 15,
}


def load(name: str):
    mat = scipy.io.loadmat(os.path.join(ROOT, "01_原版复现_LLM-PDESR", "data", name, "data.mat"))
    u = mat["u"]
    x = mat["x"].flatten()
    t = mat["t"].flatten()
    dx = float(x[1] - x[0])
    u_t = CubicSpline(t, u, axis=1)(t, 1)
    if "v" in mat:
        v = mat["v"]
        v_t = CubicSpline(t, v, axis=1)(t, 1)
    else:
        v = np.zeros_like(u)
        v_t = np.zeros_like(u)
    return x, u, v, u_t, v_t, dx


def weak_loss(name, rhs, wants_v: bool) -> float:
    """Mirror the spec: K=10 sin^4 windows, trapezoid in x, BFGS on params."""
    x, u, v, u_t, v_t, dx = load(name)
    x_b = x.reshape(-1, 1)
    n_x = len(x)
    length = x[-1] - x[0]
    k_windows = 10
    width = 2.0 * (length / k_windows)
    phi = np.zeros((n_x, k_windows))
    for i in range(k_windows):
        center = x[0] + (i + 0.5) * (length / k_windows)
        mask = np.abs(x - center) < width / 2.0
        phi[mask, i] = np.sin(np.pi * (x[mask] - (center - width / 2.0)) / width) ** 4

    def d_dx(array, order=1):
        if order > 4:
            raise ValueError("Quintic spline allowed up to 4th order only.")
        return make_interp_spline(x, array, k=5, axis=0)(x, nu=order)

    def project(field):
        return np.stack([np.trapz(field * phi[:, [i]], dx=dx, axis=0)
                         for i in range(k_windows)], axis=0)

    target_u, target_v = project(u_t), project(v_t)
    n = N_PARAMS[name]

    def loss(params):
        if wants_v:
            pred_u, pred_v = rhs(x_b, u, v, params, d_dx)
        else:
            pred_u = rhs(x_b, u, params, d_dx)
            pred_v = None
        total = np.mean((project(pred_u) - target_u) ** 2)
        if pred_v is not None:
            total += np.mean((project(pred_v) - target_v) ** 2)
        return float(total)

    result = minimize(loss, [0.5] * n, method="BFGS")
    return float(result.fun)


def parse(text: str):
    roles = ""
    match = re.search(r"FIELD_ROLES\s*:(.*?)(?:\n\s*EQUATION|\Z)", text, re.S)
    if match:
        roles = " ".join(match.group(1).split())
    code = None
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
    for block in blocks:
        if "def rhs" in block:
            code = textwrap.dedent(block).strip()
            break
    return roles, code


def main() -> int:
    raw = json.load(open(os.path.join(HERE, "role_probe_raw.json"), encoding="utf-8"))
    rows = []
    for key, text in sorted(raw.items()):
        name, condition, model = key.split("|")
        roles, code = parse(text)
        if not code:
            rows.append((name, condition, None, "no code block", roles))
            continue
        namespace = {"np": np}
        try:
            exec(code, namespace)               # same trust level as the project sandbox
            rhs = namespace["rhs"]
            wants_v = "v" in inspect.signature(rhs).parameters
            mse = weak_loss(name, rhs, wants_v)
            note = "ok"
        except Exception as error:
            mse, note = None, f"{type(error).__name__}: {error}"[:70]
        rows.append((name, condition, mse, note, roles))

    print(f"{'dataset':<24} {'cond':<10} {'my MSE':>10} {'paper':>10}  roles")
    for name, condition, mse, note, roles in rows:
        mine = "n/a" if mse is None else f"{mse:.3e}"
        print(f"{name:<24} {condition:<10} {mine:>10} {PAPER_MSE[name]:>10.1e}  "
              f"{roles[:78]}")
        if note != "ok":
            print(f"{'':<24} {'':<10} {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
