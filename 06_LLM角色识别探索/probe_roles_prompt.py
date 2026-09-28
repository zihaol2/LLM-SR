"""Ask a model to identify the physical fields and the equation, from anonymous data.

Two conditions per dataset:

    anon       the fields are f1, f2 ... and nothing about their physical meaning is
               given, so the model has to identify the quantities itself
    described  the paper's own task sentence is added (it names the mechanism)

The point is to separate "can it name the physics" from "can it write the equation",
on the five structurally novel equations the paper uses.
"""
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.io
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DATASETS = [
    ("Topography_Chemotaxis", 1),
    ("Morphogenesis", 1),
    ("Forced_Swift_Hohenberg", 1),
    ("Traffic_Flow_Bottleneck", 1),
    ("Predator_Prey", 2),
]

DESCRIBED = {
    "Topography_Chemotaxis":
        "in a biological population migrating along a periodic terrain gradient with "
        "crowding constraints",
    "Morphogenesis":
        "in a morphogen diffusion from a localized Gaussian source with "
        "Michaelis-Menten absorption",
    "Forced_Swift_Hohenberg":
        "in a subcritical optical medium driven by a Gaussian laser and stabilized by "
        "hyper-viscosity",
    "Traffic_Flow_Bottleneck":
        "in a traffic flow with a spatial bottleneck and anticipation-driven relative "
        "gradient diffusion",
    "Predator_Prey":
        "in a coupled system where a diffusing prey population is suppressed by local "
        "predator encounters, and a diffusing predator population migrates toward prey "
        "density gradients",
}


def prompt_for(name: str, n_fields: int, described: bool) -> str:
    fields = ", ".join(f"f{i+1}" for i in range(n_fields))
    targets = ", ".join(f"f{i+1}_t" for i in range(n_fields))
    systems = "a single field" if n_fields == 1 else f"{n_fields} fields"
    lines = [
        f"You are given numerical data from a one-dimensional PDE system with {systems} "
        f"({fields}) on a spatial grid x.",
        f"Targets: the time derivatives ({targets}).",
        "",
        "Nothing about the physical meaning of the fields is provided on purpose: the "
        "symbols carry no interpretation beyond their numerical values. The data comes "
        "from a numerical simulation and is noise free.",
    ]
    if described:
        lines.append(f"Context from the experiment: the fields come from {DESCRIBED[name]}.")
    lines += [
        "",
        "Do two things, in this order.",
        "1. Identify the physical quantity each field represents. Give the concrete "
        "evidence you used from the data (range, sign, spatial spectrum, time behaviour, "
        "correlations). If the data does not pin the identification down, say so and give "
        "the best alternatives.",
        "2. Write the right-hand side of the governing equation(s).",
        "",
        "Allowed: d_dx(field, order) with order <= 4 for spatial derivatives; "
        "np.sin, np.cos, np.exp, np.tanh. Every constant must come from params[i] "
        f"(params has {6 if n_fields == 1 else 15} entries). Do not use any other np.* "
        "function and do not hard-code numbers.",
        "",
        "Answer in exactly this format:",
        "FIELD_ROLES: " + "; ".join(f"f{i+1} = <quantity> (evidence: <one short line>)"
                                    for i in range(n_fields)),
        "EQUATION:",
        "```python",
    ]
    args = "x, " + ", ".join(f"f{i+1}" for i in range(n_fields)) + ", params, d_dx"
    lines.append(f"def rhs({args}):")
    for i in range(n_fields):
        lines.append(f"    f{i+1}_t = ...")
    returned = ", ".join(f"f{i+1}_t" for i in range(n_fields))
    lines.append(f"    return {returned}")
    lines += ["```", "", "Only output those two blocks, no extra commentary."]
    return "\n".join(lines)


def data_card(name: str, n_fields: int) -> str:
    """The evidence a model would need to name the physics: statistics plus a sample."""
    mat = scipy.io.loadmat(os.path.join(ROOT, "01_原版复现_LLM-PDESR", "data", name, "data.mat"))
    x = mat["x"].flatten()
    t = mat["t"].flatten()
    u = mat["u"]
    var = mat["v"] if "v" in mat else None
    u_t = CubicSpline(t, u, axis=1)(t, 1)
    lines = [f"grid: nx={x.size}, nt={t.size}, dx={x[1]-x[0]:.4f}, domain "
             f"[{x[0]:.2f}, {x[-1]:.2f}]"]
    stack = [("f1", u)] + ([("f2", var)] if n_fields > 1 and var is not None else [])
    for label, field in stack:
        mid = field[:, field.shape[1] // 2]
        power = np.abs(np.fft.rfft(mid - mid.mean())) ** 2
        top = np.argsort(power)[::-1][:3]
        share = power[top] / max(power.sum(), 1e-30)
        half = field.shape[1] // 2
        drift = (np.sqrt(np.mean(field[:, half:] ** 2))
                 / max(np.sqrt(np.mean(field[:, :half] ** 2)), 1e-30))
        lines.append(
            f"{label}: min {field.min():+.3f} max {field.max():+.3f} "
            f"mean {field.mean():+.3f} rms {np.sqrt(np.mean(field**2)):.3f} "
            f"negative fraction {np.mean(field < 0):.2f}")
        lines.append(
            f"   spatial spectrum at mid-time: k=" +
            ", ".join(f"{int(k)} ({float(s) * 100:.0f}%)" for k, s in zip(top, share)) +
            f"; rms drift second half / first half = {drift:.2f}")
        lines.append(
            f"   sample rows (x = {', '.join(f'{v:.2f}' for v in x[::max(1, x.size // 7)][:8])})")
        for slice_index in (0, field.shape[1] // 2, -1):
            row = field[::max(1, field.shape[0] // 7), slice_index][:8]
            lines.append(f"     t={t[slice_index]:.2f}: " +
                         ", ".join(f"{v:+.3f}" for v in row))
    if n_fields > 1 and var is not None:
        corr = float(np.corrcoef(u.ravel(), var.ravel())[0, 1])
        lines.append(f"correlation between f1 and f2 over all points: {corr:+.2f}")
    lines.append("targets: " + ", ".join(f"f{i+1}_t" for i in range(n_fields)) +
                 " (exact time derivatives of the fields above)")
    return "\n".join(lines)


def call(prompt: str, api_key: str, base_url: str, model: str, temperature: float = 0.3) -> str:
    body = json.dumps({
        "model": model,
        "max_tokens": 2000,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    # the gateway drops long generations now and then; retry a few times
    last = ""
    for attempt in range(4):
        try:
            request = urllib.request.Request(
                base_url.rstrip("/") + "/chat/completions", data=body,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            if content and content.strip():
                return content
            last = "empty response"
        except Exception as error:                     # noqa: BLE001 - report and retry
            last = f"{type(error).__name__}: {error}"
        print(f"    retry {attempt + 1} ({last[:60]})", flush=True)
    return f"[FAILED] {last}"


def main() -> int:
    env_path = os.path.join(ROOT, ".env")
    env = {}
    for line in open(env_path, encoding="utf-8"):
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S.*)$", line)
        if match:
            env[match.group(1)] = match.group(2).strip()
    api_key = env.get("TEAMOROUTER_API_KEY") or env.get("API_KEY")
    if not api_key:
        print("no api key in .env")
        return 1
    base_url = "https://api.teamorouter.cn/v1"
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek-v4-flash"

    out = {}
    out_path = os.path.join(HERE, "role_probe_raw.json")
    if os.path.exists(out_path):
        out = json.load(open(out_path, encoding="utf-8"))
    out = {k: v for k, v in out.items() if v and v.strip()}   # retry earlier empties
    conditions = sys.argv[2].split(",") if len(sys.argv) > 2 else ["anon", "described"]
    jobs = []
    for name, n_fields in DATASETS:
        for condition in conditions:
            key = f"{name}|{condition}|{model}"
            if key in out:
                print("cached", key)
                continue
            base = condition.split("#")[0]
            described = base == "described"
            prompt = prompt_for(name, n_fields, described)
            if base == "anon_with_data":
                prompt = prompt.replace(
                    "from a numerical simulation and is noise free.",
                    "from a numerical simulation and is noise free.\n"
                    "Measured evidence about the data:\n" + data_card(name, n_fields))
            jobs.append((key, prompt, 0.9 if "#" in condition else 0.3))

    def run(job):
        key, prompt, temperature = job
        return key, call(prompt, api_key, base_url, model, temperature)

    with ThreadPoolExecutor(max_workers=5) as pool:
        for key, text in pool.map(run, jobs):
            out[key] = text
            json.dump(out, open(out_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            print(f"--- {key} ({len(text)} chars)")
            print(text[:500])
    print("saved", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
