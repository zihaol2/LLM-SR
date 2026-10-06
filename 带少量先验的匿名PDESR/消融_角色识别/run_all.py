"""Role-identification ablation across problems.

Same three conditions for every problem, same shared prompt template:

    cond1  measured statistics only
    cond2  + the (assumed) true equation
    cond3  + the classical parent PDEs          <- the full pipeline

For each problem the equation is taken to be the ground truth (we assume the search
already found it), so this isolates the role-identification step.

    python run_all.py evidence      # build the evidence blocks (needs the API for
                                    # the statistics reading + the parent match)
    python run_all.py run           # run the three conditions
    python run_all.py all
"""
import argparse
import io
import json
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # .../带少量先验的LLMPDESR
CODE = os.path.join(ROOT, "code")
sys.path.insert(0, CODE)

import library  # noqa: E402
import roles  # noqa: E402

# ---------------------------------------------------------------- problems

MAT = r"C:\Users\Li ZiHao\Desktop\AI4MATH\LLM-PDESR\01_原版复现_LLM-PDESR\data"

PROBLEMS = {
    "traffic": {
        "reuse_from": os.path.join(ROOT, "results_main"),
        "truth_role": "vehicle density on a road with a bottleneck",
        "truth_keywords": ["traffic", "vehicle", "road", "car", "congestion"],
    },
    "swift_hohenberg": {
        "mat": os.path.join(MAT, "Forced_Swift_Hohenberg", "data.mat"),
        "truth_role": "order parameter (pattern-forming amplitude)",
        "truth_keywords": ["order parameter", "swift", "pattern", "amplitude",
                           "optical", "laser", "mode", "instability"],
        "equation_body": """# Gaussian pump, localised along x1
pump = 1.5 * np.exp(-0.1 * x[:, 0:1] ** 2)

# Second-order term
second = -2.0 * d_axis(u, 0, 2)

# Fourth-order stabilisation
fourth = -1.0 * d_axis(u, 0, 4)

# Cubic term with a quintic saturation
nonlinear = -1.0 * u ** 3 + 1.0 * u ** 5

u_t_pred = pump * u + second + fourth + nonlinear""",
        "coefficients": "pump 1.5, pump width 0.1, u_xx -2.0, u_xxxx -1.0, u^3 -1.0, u^5 +1.0",
    },
    "chemotaxis": {
        "mat": os.path.join(MAT, "Topography_Chemotaxis", "data.mat"),
        "truth_role": "population density",
        "truth_keywords": ["population", "species", "biological", "cell", "chemotax",
                           "organism", "ecolog", "bacteria"],
        "equation_body": """# Second derivative
diffusion = 0.1 * d_axis(u, 0, 2)

# Divergence of a crowding-limited flux along a periodic spatial modulation
flux = -0.5 * (u * (1.0 - u) * np.cos(x[:, 0:1]))
transport = d_axis(flux, 0, 1)

# Logistic local term
local = 1.0 * u * (1.0 - u)

u_t_pred = diffusion + transport + local""",
        "coefficients": "u_xx +0.1, flux -0.5, logistic +1.0",
    },
}

# ------------------------------------------------- shared prompt template

HEADER = """You are a physicist doing inverse identification on ONE anonymous 1D dataset.
The dataset holds a single field, f1, sampled on one spatial axis x1, together with
its measured time derivative. Nothing here names the dataset: whatever naming comes
out has to be forced by the evidence below.

"""

TASK = """
### YOUR TASK
1. NAME IT. State the concrete physical quantity f1 most likely represents, and the
   discipline and sub-field it belongs to. Point to the specific piece of evidence
   that pins the naming down.
2. PARENT FAMILY. If classical PDE families were supplied above, say which one the
   evidence points to and why. If none were supplied, leave "parent_family" empty.
3. CONFIDENCE. Give high | medium | low, and the single measurement or term that
   would most change your answer.

Answer with ONE JSON object, nothing else:
{
  "variable": {"symbol": "f1", "quantity": "...", "parent_family": "...", "reasoning": "..."},
  "discipline": {"field": "...", "subfield": "...", "reasoning": "..."},
  "confidence": "high|medium|low",
  "key_evidence": ["...", "..."],
  "would_change_if": "..."
}
"""


def evidence_dir(name):
    return os.path.join(HERE, "evidence", name)


def write_evidence(name, stats_report, macro, equation_text, coefficients, parents, meta):
    out = evidence_dir(name)
    os.makedirs(out, exist_ok=True)
    io.open(os.path.join(out, "statistics_report.txt"), "w", encoding="utf-8",
            newline="\n").write("\n".join(stats_report))
    io.open(os.path.join(out, "macro_reading.json"), "w", encoding="utf-8",
            newline="\n").write(json.dumps(macro, ensure_ascii=False, indent=1))
    io.open(os.path.join(out, "equation.txt"), "w", encoding="utf-8",
            newline="\n").write(equation_text)
    io.open(os.path.join(out, "parents.json"), "w", encoding="utf-8",
            newline="\n").write(json.dumps(
                {"coefficients": coefficients, "parents": parents},
                ensure_ascii=False, indent=1))
    io.open(os.path.join(out, "meta.json"), "w", encoding="utf-8",
            newline="\n").write(json.dumps(meta, ensure_ascii=False, indent=1))
    print("  evidence written to", out)


def load_stats_plan():
    plan = json.load(io.open(os.path.join(ROOT, "results_design", "statistics_plan.json"),
                             encoding="utf-8"))["plan"]
    return plan


def load_matrix(name):
    """The two benchmark problems ship as .mat; u_t is the spline derivative in time."""
    import scipy.io as sio
    m = sio.loadmat(PROBLEMS[name]["mat"])
    x = np.asarray(m["x"], dtype=float).ravel()
    u = np.asarray(m["u"], dtype=float)
    t = np.asarray(m["t"], dtype=float).ravel()
    u_t = CubicSpline(t, u, axis=1)(t, 1)
    return {"x1": x, "f1": u, "f1_t_true": u_t, "h1": float(x[1] - x[0])}


def dataset_description(name, data):
    x = np.asarray(data["x1"], dtype=float)
    u = np.asarray(data["f1"], dtype=float)
    lines = [
        "structure: 1 spatial coordinate(s): x1; 1 time coordinate: t (there is no x2 and no x3)",
        "a field is indexed as (space, time)",
        "arrays:",
        f"  x1: shape {np.shape(x)}, range [{x.min():+.4f}, {x.max():+.4f}], "
        f"spacing {float(x[1] - x[0]):.6f}",
        f"  f1: shape {np.shape(u)}, range [{u.min():+.4f}, {u.max():+.4f}], mean {u.mean():+.4f}",
        f"  f1_t_true: shape {np.shape(data['f1_t_true'])}, described only as the measured "
        f"time derivative of f1",
        "No physical meaning is attached to the arrays themselves.",
        "Every statement about the structure of the data must agree with the dimensions and "
        "shapes listed above. Do not assume any axis that is not listed.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------- evidence


def build_evidence(call):
    plan = load_stats_plan()
    for name, spec in PROBLEMS.items():
        print("=" * 70)
        print("evidence:", name)
        out = evidence_dir(name)
        if spec.get("reuse_from"):
            lib = json.load(io.open(os.path.join(spec["reuse_from"], "designed_library.json"),
                                    encoding="utf-8"))
            rec = json.load(io.open(os.path.join(spec["reuse_from"], "role_identification.json"),
                                    encoding="utf-8"))
            write_evidence(
                name,
                lib["statistics_report"],
                (lib.get("inferred_roles") or {}).get("f1") or {},
                rec["equation_with_values"],
                ", ".join(f"params[{i}] = {v}" for i, v in enumerate(rec["best_equation"]["params"])),
                (rec.get("parents") or {}).get("parents", []),
                {"truth_role": spec["truth_role"], "truth_keywords": spec["truth_keywords"],
                 "source": "results_main (the 100-sample run)"})
            continue

        data = load_matrix(name)
        report, _ = library.execute(plan, data)
        print("  statistics:", len(report), "lines")
        macro_payload = roles.parse_json(call(library.role_only_prompt("\n".join(report))))
        macro = (macro_payload.get("inferred_roles") or {}).get("f1") or {}
        print("  macro reading:", (macro.get("identity") or "")[:90])
        equation_text = (spec["equation_body"] + "\n\n### FITTED COEFFICIENTS\n"
                         f"{spec['coefficients']}\n")
        parents_raw = call(roles.parents_prompt(equation_text, dataset_description(name, data), 6))
        parents = (roles.parse_json(parents_raw).get("parents") or [])
        print("  parents:", [p.get("name") for p in parents if isinstance(p, dict)])
        write_evidence(name, report, macro, equation_text, spec["coefficients"], parents,
                       {"truth_role": spec["truth_role"], "truth_keywords": spec["truth_keywords"],
                        "parents_raw": parents_raw})


# -------------------------------------------------------------------- run


def load_evidence(name):
    out = evidence_dir(name)
    report = io.open(os.path.join(out, "statistics_report.txt"), encoding="utf-8").read()
    macro = json.load(io.open(os.path.join(out, "macro_reading.json"), encoding="utf-8"))
    equation = io.open(os.path.join(out, "equation.txt"), encoding="utf-8").read()
    parents = json.load(io.open(os.path.join(out, "parents.json"), encoding="utf-8"))
    return report, macro, equation, parents


def stats_block(report, macro):
    out = ["### EVIDENCE 1 -- MEASURED STATISTICS",
           "These statistics were computed on the data itself. They describe the field f1,",
           "its measured time derivative, and how the two relate. Nothing about the subject",
           "matter was given to whoever asked for them.", ""]
    out += ["  " + line for line in report.splitlines() if line.strip()]
    if macro.get("identity"):
        out += ["", "Read off those measurements alone, before any equation existed:",
                macro["identity"]]
    if macro.get("evidence"):
        out += ["", "The measurements that reading was based on:", macro["evidence"]]
    return "\n".join(out)


def equation_block(equation):
    return "\n".join(
        ["### EVIDENCE 2 -- THE DISCOVERED EQUATION",
         "An automated search found the equation below for the measured time derivative.",
         "It reproduces the data down to the loss floor, so this is the equation the data",
         "supports. The coefficients the search fitted are written in place.", "",
         equation.strip()])


def parents_block(parents):
    return "\n".join(
        ["### EVIDENCE 3 -- CLASSICAL PARENT PDES",
         "The same equation, matched against the classical PDE families it could be a",
         "modification of. Each entry gives the family, its canonical form, how the",
         "discovered terms map onto it, and the standard physical role of its symbols.", "",
         json.dumps(parents["parents"], ensure_ascii=False, indent=1)])


def build_prompts(name):
    report, macro, equation, parents = load_evidence(name)
    e1, e2, e3 = stats_block(report, macro), equation_block(equation), parents_block(parents)
    return {
        "cond1_statistics_only": HEADER + e1 + "\n" + TASK,
        "cond2_statistics_equation": HEADER + e1 + "\n\n" + e2 + "\n" + TASK,
        "cond3_statistics_equation_parents": HEADER + e1 + "\n\n" + e2 + "\n\n" + e3 + "\n" + TASK,
    }


def check_fairness(prompts, e1):
    problems = []
    for name, text in prompts.items():
        if not text.startswith(HEADER):
            problems.append(f"{name}: shared header differs")
        if not text.endswith(TASK):
            problems.append(f"{name}: shared task block differs")
        if e1 not in text:
            problems.append(f"{name}: statistics block is not byte-identical")
    return problems


def run(call, repeats):
    summary = {}
    for name, spec in PROBLEMS.items():
        prompts = build_prompts(name)
        e1 = prompts["cond1_statistics_only"][len(HEADER):-len(TASK)]
        problems = check_fairness(prompts, e1)
        if problems:
            raise SystemExit(f"{name}: fairness check failed: {problems}")

        pdir = os.path.join(HERE, "prompts", name)
        os.makedirs(pdir, exist_ok=True)
        for prompt_name, text in prompts.items():
            io.open(os.path.join(pdir, prompt_name + ".txt"), "w", encoding="utf-8",
                    newline="\n").write(text)
        print("=" * 78)
        print("%s:  fairness OK | prompts %s"
              % (name, {k: len(v) for k, v in prompts.items()}))

        results = {}
        for prompt_name, text in prompts.items():
            runs = []
            for rep in range(1, repeats + 1):
                payload = roles.parse_json(call(text))
                quantity = ((payload.get("variable") or {}).get("quantity") or "")
                hit = any(k in quantity.lower() for k in spec["truth_keywords"])
                runs.append({"raw_response": payload, "quantity": quantity,
                             "hit": bool(hit)})
                print("  [%-34s rep%d] %-6s | hit=%-5s | %s"
                      % (prompt_name, rep, payload.get("confidence"), hit, quantity[:62]),
                      flush=True)
            results[prompt_name] = runs
        summary[name] = {"truth_role": spec["truth_role"],
                         "truth_keywords": spec["truth_keywords"], "conditions": results}

    out = os.path.join(HERE, "ablation_all.json")
    with io.open(out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=1)
    print("\nwrote", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["evidence", "run", "all"], default="all", nargs="?")
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("ABLATION_REPEATS", "3")))
    args = parser.parse_args()

    call = roles.make_call()
    if args.stage in ("evidence", "all"):
        build_evidence(call)
    if args.stage in ("run", "all"):
        run(call, args.repeats)


if __name__ == "__main__":
    main()
