"""Data-only role identification: give the LLM statistics, ask for the roles.

No physics description and no equation are given or requested. For each of the five
problems the model gets one measured statistics card (ranges, sign, spectra, drift,
sample rows, cross-correlation) and, as a physics expert, states what physical
quantity each anonymous field represents. Each problem is asked `--repeats` times
(default 5, like seeds) so the stability of the answer can be seen.

    python roles_probe.py                       # 5 problems x 5 repeats
    python roles_probe.py deepseek-v4-flash
    python roles_probe.py deepseek-v4-flash 3   # 3 repeats instead of 5
"""
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import scipy.io

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_ROOT = os.path.join(ROOT, "01_原版复现_LLM-PDESR", "data")

DATASETS = [
    ("Topography_Chemotaxis", 1),
    ("Morphogenesis", 1),
    ("Forced_Swift_Hohenberg", 1),
    ("Traffic_Flow_Bottleneck", 1),
    ("Predator_Prey", 2),
]

# Only used to label the summary; the model never sees these.
TRUTH = {
    "Topography_Chemotaxis": ["population density"],
    "Morphogenesis": ["morphogen concentration"],
    "Forced_Swift_Hohenberg": ["order parameter"],
    "Traffic_Flow_Bottleneck": ["traffic density"],
    "Predator_Prey": ["prey density", "predator density"],
}

BASE_URL = "https://api.teamorouter.cn/v1"


def prompt_for(n_fields: int) -> str:
    fields = ", ".join(f"f{i + 1}" for i in range(n_fields))
    systems = "a single field" if n_fields == 1 else f"{n_fields} fields"
    lines = [
        "You are a physics expert.",
        "",
        f"You are given numerical measurements from an anonymous one-dimensional PDE "
        f"experiment with {systems} ({fields}) on a spatial grid x, together with their "
        "measured time derivatives. Nothing about the physical meaning of the fields "
        "is provided on purpose: the symbols carry no interpretation beyond their "
        "numerical values, and the data comes from a numerical simulation and is "
        "noise free.",
        "",
        "Analyse the statistical regularities in the measured evidence below and "
        "identify the physical quantity each field most likely represents. Give the "
        "concrete evidence you used (range, sign, spatial spectrum, time behaviour, "
        "correlations). If the data does not pin the identification down, say so and "
        "give the best alternatives.",
        "",
        "Answer in exactly this format and nothing else:",
        "FIELD_ROLES:",
    ]
    for i in range(n_fields):
        lines.append(f"f{i + 1} = <physical quantity> (evidence: <one short line>)")
    lines.append("")
    lines.append("MEASURED EVIDENCE:")
    return "\n".join(lines)


def data_card(name: str, n_fields: int) -> str:
    mat = scipy.io.loadmat(os.path.join(DATA_ROOT, name, "data.mat"))
    x = mat["x"].flatten()
    t = mat["t"].flatten()
    u = mat["u"]
    var = mat["v"] if "v" in mat else None

    lines = [f"grid: nx={x.size}, nt={t.size}, dx={x[1] - x[0]:.4f}, "
             f"domain [{x[0]:.2f}, {x[-1]:.2f}]"]
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
            f"mean {field.mean():+.3f} rms {np.sqrt(np.mean(field ** 2)):.3f} "
            f"negative fraction {np.mean(field < 0):.2f}")
        lines.append(
            "   spatial spectrum at mid-time: k="
            + ", ".join(f"{int(k)} ({float(s) * 100:.0f}%)" for k, s in zip(top, share))
            + f"; rms drift second half / first half = {drift:.2f}")
        lines.append("   sample rows (x = "
                     + ", ".join(f"{v:.2f}" for v in x[::max(1, x.size // 7)][:8]) + ")")
        for slice_index in (0, field.shape[1] // 2, -1):
            row = field[::max(1, field.shape[0] // 7), slice_index][:8]
            lines.append(f"     t={t[slice_index]:.2f}: "
                         + ", ".join(f"{v:+.3f}" for v in row))
    if n_fields > 1 and var is not None:
        corr = float(np.corrcoef(u.ravel(), var.ravel())[0, 1])
        lines.append(f"correlation between f1 and f2 over all points: {corr:+.2f}")
    lines.append("targets: " + ", ".join(f"f{i + 1}_t" for i in range(n_fields))
                 + " (the measured time derivatives of the fields above)")
    return "\n".join(lines)


def call(prompt: str, api_key: str, model: str, temperature: float = 1.0) -> str:
    payload = {
        "model": model,
        "max_tokens": 1500,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    # DeepSeek defaults to a long chain of thought here, which makes a short role
    # answer take minutes; the framework runs it disabled and so do we.
    if "deepseek" in model.lower():
        payload["thinking"] = {"type": "disabled"}
    body = json.dumps(payload).encode("utf-8")
    last = ""
    for attempt in range(4):
        try:
            request = urllib.request.Request(
                BASE_URL + "/chat/completions", data=body,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            if content and content.strip():
                return content
            last = "empty response"
        except Exception as error:                       # noqa: BLE001
            last = f"{type(error).__name__}: {error}"
        print(f"    retry {attempt + 1} ({last[:70]})", flush=True)
    return f"[FAILED] {last}"


def answer_of(text: str, field: str) -> str:
    """The phrase the model gave for one field, for the summary."""
    match = re.search(rf"^\s*{field}\s*=\s*(.+)$", text, re.M | re.I)
    if match:
        return " ".join(match.group(1).split())
    match = re.search(rf"{field}\s*=\s*([^;\n]+)", text, re.I)
    if match:
        return " ".join(match.group(1).split())
    return "(no FIELD_ROLES line) " + " ".join(text.split())[:100]


def write_summary(out: dict, model: str, repeats: int) -> str:
    lines = [f"# Data-only role identification ({model}, {repeats} repeats each)", "",
             "Only measured statistics were given; no physics description and no "
             "equation was requested.", ""]
    for name, n_fields in DATASETS:
        truth = TRUTH.get(name, [""] * n_fields)
        lines.append(f"## {name}")
        for index in range(n_fields):
            field = f"f{index + 1}"
            truth_note = truth[index] if index < len(truth) else ""
            lines.append(f"**{field}**  (reference truth: {truth_note})")
            for seed in range(1, repeats + 1):
                key = f"{name}|seed{seed}|{model}"
                text = out.get(key, "")
                lines.append(f"- seed{seed}: {answer_of(text, field)}")
            lines.append("")
    path = os.path.join(HERE, "roles_summary.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek-v4-flash"
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    env = {}
    for line in open(os.path.join(ROOT, ".env"), encoding="utf-8"):
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S.*)$", line)
        if match:
            env[match.group(1)] = match.group(2).strip()
    api_key = env.get("TEAMOROUTER_API_KEY") or env.get("API_KEY")
    if not api_key:
        print("no api key in .env")
        return 1

    out_path = os.path.join(HERE, "roles_raw.json")
    out = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    out = {k: v for k, v in out.items() if v and v.strip()}

    jobs = []
    for name, n_fields in DATASETS:
        card = data_card(name, n_fields)
        base = prompt_for(n_fields) + "\n" + card
        for seed in range(1, repeats + 1):
            key = f"{name}|seed{seed}|{model}"
            if key in out:
                continue
            jobs.append((key, base + "\n\nNow give the FIELD_ROLES for this dataset."))

    print(f"running {len(jobs)} call(s): {len(DATASETS)} problems x {repeats} repeats",
          flush=True)

    def run(job):
        key, prompt = job
        return key, call(prompt, api_key, model)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run, job): job[0] for job in jobs}
        for future in as_completed(futures):
            key, text = future.result()
            out[key] = text
            json.dump(out, open(out_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            print(f"--- {key} ({len(text)} chars)", flush=True)

    summary = write_summary(out, model, repeats)
    print("saved", out_path, flush=True)
    print("saved", summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
