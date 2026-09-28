"""Score whatever the traffic role ablation already produced. No API calls."""
import json
import os
import re

import numpy as np
import scipy.io
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NAME = "Traffic_Flow_Bottleneck"


def parse(text: str) -> str | None:
    for block in re.findall(r"```(?:python)?\s*(.*?)```", text, re.S):
        if "def rhs" in block:
            return block.strip()
    return None


def flags(code: str) -> dict:
    return {
        "tanh": "np.tanh" in code,
        "flux": bool(re.search(r"u\s*\*\s*\(\s*1\s*-\s*u", code)),
        "flux_deriv": bool(re.search(r"d_dx\([^)]*u\s*\*\s*\(\s*1\s*-\s*u", code)),
        "rel_grad": bool(re.search(r"d_dx\([^)]*/\s*u", code)),
        "nd_dx": len(re.findall(r"d_dx\(", code)),
    }


def weak_loss(rhs) -> float:
    mat = scipy.io.loadmat(os.path.join(ROOT, "01_原版复现_LLM-PDESR", "data", NAME, "data.mat"))
    x = mat["x"].flatten()
    t = mat["t"].flatten()
    u = mat["u"]
    u_t = CubicSpline(t, u, axis=1)(t, 1)
    x_b = x.reshape(-1, 1)
    dx = float(x[1] - x[0])
    length = x[-1] - x[0]
    windows = 10
    width = 2.0 * (length / windows)
    phi = np.zeros((len(x), windows))
    for i in range(windows):
        center = x[0] + (i + 0.5) * (length / windows)
        mask = np.abs(x - center) < width / 2.0
        phi[mask, i] = np.sin(np.pi * (x[mask] - (center - width / 2.0)) / width) ** 4

    def d_dx(array, order=1):
        return make_interp_spline(x, array, k=5, axis=0)(x, nu=order)

    def project(field):
        return np.stack([np.trapz(field * phi[:, [i]], dx=dx, axis=0)
                         for i in range(windows)], axis=0)

    target = project(u_t)

    def loss(params):
        return float(np.mean((project(rhs(x_b, u, params, d_dx)) - target) ** 2))

    return float(minimize(loss, [0.5] * 8, method="BFGS", options={"maxiter": 80}).fun)


raw = json.load(open(os.path.join(HERE, "traffic_role_ablation.json"), encoding="utf-8"))
rows = []
for key, text in sorted(raw.items()):
    arm, index = key.split("|")
    code = parse(text)
    if not code:
        rows.append((arm, index, None, None, "no code block"))
        continue
    fl = flags(code)
    try:
        namespace = {"np": np}
        exec(code, namespace)
        mse = weak_loss(namespace["rhs"])
        note = ""
    except Exception as error:                                    # noqa: BLE001
        mse, note = None, f"{type(error).__name__}"
    rows.append((arm, index, mse, fl, note))

print(f"{'arm':<8} {'#':>3} {'MSE':>11}  tanh  u(1-u)  flux-deriv  rel-grad  n(d_dx)  note")
for arm, index, mse, fl, note in rows:
    if fl is None:
        print(f"{arm:<8} {index:>3} {'-':>11}  {note}")
        continue
    value = "n/a" if mse is None else f"{mse:.3e}"
    print(f"{arm:<8} {index:>3} {value:>11}  {str(fl['tanh']):<5} {str(fl['flux']):<7} "
          f"{str(fl['flux_deriv']):<11} {str(fl['rel_grad']):<9} {fl['nd_dx']:>7}  {note}")

print()
for arm in ("wrong", "correct", "none"):
    subset = [r for r in rows if r[0] == arm]
    scored = [r[2] for r in subset if r[2] is not None]
    with_flags = [r for r in subset if r[3] is not None]
    judged = sum(1 for r in with_flags if r[3]["tanh"] and r[3]["flux"])
    print(f"{arm:<8} n={len(subset):<3} scored={len(scored):<3} "
          f"tanh+flux={judged:<3} "
          f"median MSE={'n/a' if not scored else format(float(np.median(scored)), '.3e')}  "
          f"best={'n/a' if not scored else format(float(np.min(scored)), '.3e')}")
