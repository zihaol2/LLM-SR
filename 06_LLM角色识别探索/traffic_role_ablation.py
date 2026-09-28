"""Does a wrong variable role still let the model find the equation?

Traffic_Flow_Bottleneck, 20 samples per arm:

    wrong    the model is told f1 is a passive scalar / tracer (a wrong but very
             plausible identity, and the one the model itself proposed most often)
    correct  the model is told f1 is traffic density on a road with a bottleneck
    none     no role statement at all, only the measured data card

Every returned equation is scored in the same weak-form loss the paper's spec uses,
and checked for the two structural fingerprints of the true equation: a tanh
bottleneck modulation and a conserved flux of the form d_dx(u(1-u)*...).
"""
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import scipy.io
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NAME = "Traffic_Flow_Bottleneck"
SAMPLES = {"wrong": 20, "correct": 10, "none": 10}
WORKERS = 8

sys.path.insert(0, HERE)
from probe_roles_prompt import data_card, prompt_for          # noqa: E402  (shared card)

ROLE_WRONG = ("f1 is a passive scalar concentration / tracer field, advected by a "
              "background flow")
ROLE_CORRECT = ("f1 is traffic density on a road with a spatial bottleneck (the road "
                "narrows along x), so the flux depends on a space-varying capacity")


def prompt(arm: str) -> str:
    # `described=False` keeps the mechanism out; we add the role statement ourselves
    text = prompt_for(NAME, 1, False)
    text = text.replace(
        "from a numerical simulation and is noise free.",
        "from a numerical simulation and is noise free.\n"
        "Measured evidence about the data:\n" + data_card(NAME, 1))
    if arm == "wrong":
        text = text.replace("FIELD_ROLES: f1 =",
                            f"You are TOLD that {ROLE_WRONG}.\nFIELD_ROLES: f1 =")
    elif arm == "correct":
        text = text.replace("FIELD_ROLES: f1 =",
                            f"You are TOLD that {ROLE_CORRECT}.\nFIELD_ROLES: f1 =")
    return text


def call(prompt_text: str, api_key: str) -> str:
    body = json.dumps({
        "model": "deepseek-v4-flash", "max_tokens": 2000, "temperature": 0.9,
        "messages": [{"role": "user", "content": prompt_text}],
    }).encode("utf-8")
    last = ""
    for _ in range(4):
        try:
            request = urllib.request.Request(
                "https://api.teamorouter.cn/v1/chat/completions", data=body,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = payload["choices"][0]["message"]["content"]
            if text and text.strip():
                return text
            last = "empty"
        except Exception as error:                                # noqa: BLE001
            last = f"{type(error).__name__}"
    return f"[FAILED] {last}"


def parse(text: str):
    code = None
    for block in re.findall(r"```(?:python)?\s*(.*?)```", text, re.S):
        if "def rhs" in block:
            code = block.strip()
            break
    return code


def flags(code: str) -> dict:
    body = code.replace(" ", "")
    return {
        "tanh": "np.tanh" in code,
        "flux_u(1-u)": bool(re.search(r"u\s*\*\s*\(1\s*-\s*u", code)),
        "flux_d_dx": bool(re.search(r"d_dx\(\s*[^,]*u\s*\*\s*\(1\s*-\s*u", code)),
        "relative_grad": bool(re.search(r"d_dx\([^)]*/\s*u", code)),
        "terms": body.count("d_dx"),
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
        if order > 4:
            raise ValueError("order too high")
        return make_interp_spline(x, array, k=5, axis=0)(x, nu=order)

    def project(field):
        return np.stack([np.trapz(field * phi[:, [i]], dx=dx, axis=0)
                         for i in range(windows)], axis=0)

    target = project(u_t)

    def loss(params):
        return float(np.mean((project(rhs(x_b, u, params, d_dx)) - target) ** 2))

    result = minimize(loss, [0.5] * 8, method="BFGS")
    return float(result.fun)


def main() -> int:
    env = {}
    for line in open(os.path.join(ROOT, ".env"), encoding="utf-8"):
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S.*)$", line)
        if match:
            env[match.group(1)] = match.group(2).strip()
    api_key = env.get("TEAMOROUTER_API_KEY") or env.get("API_KEY")

    out_path = os.path.join(HERE, "traffic_role_ablation.json")
    raw = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    jobs = [(arm, i) for arm in ("wrong", "correct", "none")
            for i in range(SAMPLES[arm])
            if f"{arm}|{i}" not in raw]
    print(f"running {len(jobs)} calls")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(call, prompt(arm), api_key): (arm, i) for arm, i in jobs}
        done = 0
        for future in as_completed(futures):      # save as they finish, not in order
            arm, index = futures[future]
            raw[f"{arm}|{index}"] = future.result()
            json.dump(raw, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
            done += 1
            if done % 5 == 0:
                print(f"  {done}/{len(jobs)} done", flush=True)

    summary = {}
    for arm in ("wrong", "correct", "none"):
        scores, hits, flag_rows = [], 0, []
        for i in range(SAMPLES[arm]):
            text = raw.get(f"{arm}|{i}", "")
            code = parse(text)
            if not code:
                continue
            fl = flags(code)
            flag_rows.append(fl)
            if fl["tanh"] and fl["flux_u(1-u)"]:
                hits += 1
            namespace = {"np": np}
            try:
                exec(code, namespace)
                scores.append(weak_loss(namespace["rhs"]))
            except Exception:                                     # noqa: BLE001
                pass
        summary[arm] = {
            "n_code": len(flag_rows), "n_scored": len(scores),
            "tanh_and_flux": hits,
            "median_mse": float(np.median(scores)) if scores else None,
            "best_mse": float(np.min(scores)) if scores else None,
            "tanh_rate": sum(f["tanh"] for f in flag_rows),
            "flux_rate": sum(f["flux_u(1-u)"] for f in flag_rows),
        }
        print(f"\n### arm={arm}")
        for key, value in summary[arm].items():
            print(f"   {key}: {value}")
    json.dump(summary, open(os.path.join(HERE, "traffic_role_ablation_summary.json"),
                            "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
