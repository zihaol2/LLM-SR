"""A5: observation-driven operator library, grown by the search itself.

Copied here from 19_观测驱动算子库/code/lib_exp.py so that folder 22 carries the code
that produced its headline number. Paths are resolved from this file's location, so it
runs unchanged from here (start it from the repository root so that .env is found).

The supervisor's question: can fine-grained feedback be wired to an operator LIBRARY
so that the LLM produces LARGE and DIVERSE hypotheses from the observations, instead
of reciting physics?

Loop, 10 rounds x 10 equations = 100 samples:

  1. PROPOSE 6 structurally distinct operators from (data description + current library
     + per-operator gains + last round's fine-grained feedback).
  2. VALIDATE each on the data: single-term weak-form R^2 against the target, and a
     collinearity check against the library already admitted. Only mechanisms that pass
     enter the library, so the library is grown BY THE DATA, not by a human prior.
  3. WRITE 10 candidate equations, with the library offered as usable mechanisms.
  4. EVALUATE with the same weak-form loss as every other arm; update the Pareto front.
  5. FEED BACK the residual figure + numbers + the per-operator gain table. The gain
     table is the one signal that does NOT depend on the current equation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(EXP)
sys.path.insert(0, os.path.join(REPO, "18_零先验_细粒度反馈", "code"))
import exp as E  # noqa: E402  (reuse the evaluated, identical weak-form problem)

MAX_LIBRARY = 40
K_OPERATORS = 6
R2_ADMIT = 0.02
MAX_COLLINEARITY = 0.95


# --------------------------------------------------------------------- signatures


def signatures(expr: str) -> list:
    """Which of the four structural facts the human spec states does an operator show?"""
    t = expr.replace(" ", "").lower()
    out = []
    if "d_axis(" in t and ("*" in t.split("d_axis(")[1][:40]):
        out.append("divergence_of_a_product")
    if "1-f1" in t or "1 -f1" in t:
        out.append("self_limiting_factor")
    if "/f1" in t or "/f1)**" in t:
        out.append("ratio_relative")
    if ("tanh(" in t and "x" in t) or ("exp(-" in t and "x" in t):
        out.append("coordinate_modulation")
    return out


# ----------------------------------------------------------------------- prompts


def data_card(prob) -> str:
    f, ft, x = prob.f, prob.ft, prob.x
    return (f"  x1 : {len(x)} points spanning [{x[0]:.3g}, {x[-1]:.3g}]\n"
            f"  f1 : shape {f.shape}, range [{f.min():.4g}, {f.max():.4g}], "
            f"mean {f.mean():.4g}\n"
            f"  f1_t: the measured time derivative, rms {np.sqrt(np.mean(ft**2)):.4g}, "
            f"range [{ft.min():.4g}, {ft.max():.4g}]")


def library_text(lib: list, with_gain: bool = True) -> str:
    if not lib:
        return "(the library is still empty)"
    lines = []
    for i, op in enumerate(lib):
        g = f"   [explains {op['r2']:.3f} of the target alone]" if with_gain else ""
        lines.append(f"  op{i + 1}: {op['expr']}{g}")
    return "\n".join(lines)


def propose_prompt(prob, lib, feedback, notes, k=K_OPERATORS) -> str:
    return f"""You are designing MECHANISMS for an anonymous one-dimensional system.
Nothing says what the field is; do not guess a subject area.

### THE DATA
{data_card(prob)}

### THE MECHANISM LIBRARY SO FAR (each entry is one candidate term; the bracket says
### how much of the measured time derivative that single term explains on its own)
{library_text(lib)}

### WHAT THE LAST ROUND OF FITTING SHOWED
{feedback if feedback.strip() else "(nothing yet)"}

{notes}

### YOUR TASK
Propose exactly {k} NEW mechanisms. Requirements, all mandatory:
1. Each must be a SINGLE closed-form expression in x, f1 and spatial derivatives
   d_axis(<expr>, 0, order) with order <= 4. Only np.sin, np.cos, np.tanh, np.exp.
2. They must be STRUCTURALLY DISTINCT from each other and from every library entry
   above -- different operator content, not the same shape with different numbers.
   Spread them out: cover different differential orders and different kinds of
   dependence (on the field alone; on the field times its gradient; on the spatial
   coordinate; on a ratio or a product; on a saturating function). Do not send
   near-duplicates; they will be measured and discarded.
3. Write NO free parameters: a mechanism is a fixed expression. Any number you are
   unsure about is simply not allowed -- the search fits coefficients later.

Answer with ONE JSON object, nothing else:
{{
 "operators": [
   {{"name": "short_name", "expression": "d_axis(f1 * (1 - f1), 0, 1)",
     "why": "what dependence this adds that the library lacks"}}
 ]
}}
"""


SEARCH_PROMPT_LIB = """
### THE MECHANISM LIBRARY (discovered from this data by the search itself)
Each line is a mechanism that survived a data test; the bracket is how much of the
target that single mechanism explains on its own. You may use, modify or ignore them.
{library}
"""


def feedback_prompt(prob, lib, diag, equation, params, loss) -> str:
    return f"""You are analysing an automated search on an anonymous one-dimensional
dataset. Nothing says what the field is; do not guess a subject area.

### THE MECHANISM LIBRARY, WITH ITS MEASURED WORTH
{library_text(lib)}

### THE CURRENT BEST EQUATION
{equation}

### ITS FITTED COEFFICIENTS
{params}

### WEAK-FORM LOSS
{loss:.3e}      (lower is better)

### WHERE THE FIT IS WRONG, MEASURED
{json.dumps(diag, ensure_ascii=False, indent=1)}

### THE FIGURE
Left to right: (a) the residual field over the whole (x, t) domain; (b) the error per
test window along x; (c) the leftover vs the field value; (d) the spatial spectrum.

### YOUR TASK
Answer with ONE JSON object, nothing else:
{{
 "where_it_fits_badly": [
   {{"region": "...", "evidence": "...", "what_seems_missing": "..."}}
 ],
 "library_verdict": "which library entries look useful and which look useless, and the
   measurement that says so",
 "new_mechanism_directions": ["anonymous, testable descriptions of the KIND of term the
   library still lacks", "..."],
 "phenomenon": "the UPDATED description of the phenomenon, at most 700 characters;
   rewrite from scratch, keep only what the evidence supports, and NEVER use a
   subject-area noun.",
 "confidence": "low|medium|high"
}}
"""


# ------------------------------------------------------------------ operator side


def compile_operator(expr: str, prob):
    fn = eval("lambda x, f1, d_axis: (" + expr + ")", {"np": np})   # noqa: S307
    v = np.asarray(fn(prob.x_b, prob.f, prob.d_axis), float)
    if v.shape == (prob.f.shape[0], 1):
        v = v * np.ones_like(prob.f)
    if v.shape != prob.f.shape:
        raise ValueError(f"shape {v.shape} does not broadcast to {prob.f.shape}")
    if not np.all(np.isfinite(v)) or np.max(np.abs(v)) > 1e6:
        raise ValueError("non-finite or absurdly large")
    return fn, v


def score_operator(vec: np.ndarray, prob, lib_vecs: list) -> tuple:
    w = prob.window(vec).ravel()
    y = prob.target().ravel()
    A = np.stack([np.ones_like(w), w], axis=1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1.0 - float(np.mean((y - A @ c) ** 2)) / (float(np.var(y)) + 1e-300)
    coll = 0.0
    for v in lib_vecs:
        wv = prob.window(v).ravel()
        if np.std(w) > 0 and np.std(wv) > 0:
            coll = max(coll, abs(float(np.corrcoef(w, wv)[0, 1])))
    return r2, coll


def grow_library(prob, lib, proposed) -> tuple:
    """Admit only mechanisms the data likes and that are not collinear with the rest."""
    admitted, rejected = [], []
    lib_vecs = [op["vec"] for op in lib]
    for cand in proposed:
        name = str(cand.get("name") or f"op{len(lib) + 1}")[:24]
        expr = str(cand.get("expression") or "").strip()
        if not expr:
            continue
        try:
            _, vec = compile_operator(expr, prob)
            r2, coll = score_operator(vec, prob, lib_vecs)
        except Exception as exc:                                   # noqa: BLE001
            rejected.append({"name": name, "expr": expr,
                             "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if r2 < R2_ADMIT:
            rejected.append({"name": name, "expr": expr,
                             "reason": f"explains too little (R2 {r2:.3f})"})
            continue
        if coll > MAX_COLLINEARITY:
            rejected.append({"name": name, "expr": expr,
                             "reason": f"collinear with the library ({coll:.3f})"})
            continue
        admitted.append({"name": name, "expr": expr, "r2": round(r2, 4),
                         "collinearity": round(coll, 3), "vec": vec,
                         "why": str(cand.get("why") or "")[:200],
                         "signatures": signatures(expr)})
        lib_vecs.append(vec)
    lib = lib + admitted
    lib.sort(key=lambda o: -o["r2"])
    lib = lib[:MAX_LIBRARY]
    return lib, admitted, rejected


# --------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--image_model", default="deepseek-v4-flash-vision-exp")
    ap.add_argument("--base_url", default="https://api.teamorouter.cn/v1")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--per_round", type=int, default=10)
    ap.add_argument("--concurrent", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(EXP, "results_A5"))
    ap.add_argument("--tag", default="lib")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "figs"), exist_ok=True)

    llm = E.LLM(E.read_key(), args.base_url, args.model, args.image_model)
    prob = E.Problem(E.DATA)
    n_rounds = args.samples // args.per_round

    lib, history, front = [], [], []
    phenomenon, feedback, directions = "", "", ""
    best = {"mse": float("inf"), "code": "", "params": None, "round": None}

    for rnd in range(1, n_rounds + 1):
        print(f"\n=== round {rnd}/{n_rounds}  |library| {len(lib)} ===", flush=True)
        t0 = time.time()

        # ---- 1. propose mechanisms from the observations
        prop = llm.chat(propose_prompt(prob, lib, feedback, directions),
                        tag=f"r{rnd}propose")
        proposed = (E.parse_json(prop) or {}).get("operators") or []
        lib, admitted, rejected = grow_library(prob, lib, proposed)
        print(f"  proposed {len(proposed)}, admitted {len(admitted)}, "
              f"rejected {len(rejected)}, library {len(lib)}", flush=True)
        for op in admitted:
            print(f"    + op: {op['expr'][:70]}  (R2 {op['r2']:.3f}, "
                  f"coll {op['collinearity']:.2f})", flush=True)

        # ---- 2. write candidate equations, library offered as usable mechanisms
        block = (SEARCH_PROMPT_LIB.format(library=library_text(lib))
                 if lib else "")
        prompt = E.build_prompt(phenomenon, feedback, directions, block)
        with ThreadPoolExecutor(max_workers=args.concurrent) as pool:
            replies = list(pool.map(lambda i: llm.chat(prompt, tag=f"r{rnd}c{i}"),
                                    range(args.per_round)))

        # ---- 3. evaluate
        for i, reply in enumerate(replies):
            rec = {"round": rnd, "idx": i, "kind": "candidate", "prompt": prompt,
                   "reply": reply}
            try:
                fn = E.compile_candidate(reply)
                res = prob.evaluate(fn)
                body = E.extract_body(reply)
                used = [op["name"] for op in lib if op["name"] + "(" in body]
                rec.update(res, code=body, complexity=E.complexity_of(body),
                           library_calls=used)
                front = E.update_front(front, res["mse"], rec["complexity"], body)
                if res["mse"] < best["mse"]:
                    best = {"mse": res["mse"], "code": body,
                            "params": res["params"], "round": rnd}
                print(f"  candidate {i}: mse {res['mse']:.3e}  cx {rec['complexity']}"
                      f"  calls={used}", flush=True)
            except Exception as exc:                               # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  candidate {i}: rejected ({rec['error']})", flush=True)
            history.append(rec)

        # ---- 4. fine-grained feedback, including the per-operator gain table
        fig, diag = None, None
        if best["params"] is not None:
            fn = E.compile_candidate(best["code"])
            R = prob.residual(fn, best["params"])
            diag = E.diagnose(prob, R, rhs=fn, params=best["params"])
            fig = os.path.join(args.out, "figs", f"{args.tag}_r{rnd:02d}.png")
            E.render(prob, R, diag, best["mse"], fig)
        if fig:
            fb = feedback_prompt(prob, lib, diag, best["code"], best["params"],
                                 best["mse"])
            reply = llm.chat(fb, image_png=fig, tag=f"r{rnd}feedback")
            payload = E.parse_json(reply) or {}
            phenomenon = str(payload.get("phenomenon") or phenomenon)
            feedback = "\n".join(
                f"- {w.get('region', '?')}: {w.get('what_seems_missing', '?')} "
                f"(evidence: {w.get('evidence', '?')})"
                for w in (payload.get("where_it_fits_badly") or [])
                if isinstance(w, dict))
            directions = "\n".join(f"- {d}" for d in
                                   (payload.get("new_mechanism_directions") or []))
            history.append({"round": rnd, "kind": "feedback", "diagnostics": diag,
                            "best_mse": best["mse"], "library_size": len(lib),
                            "library": [{"name": o["name"], "expr": o["expr"],
                                         "r2": o["r2"], "signatures": o["signatures"]}
                                        for o in lib],
                            "parsed": payload, "phenomenon_new": phenomenon,
                            "noun_hits": E.noun_hits(phenomenon)})
            print(f"  feedback: best {best['mse']:.3e}; phenomenon "
                  f"{len(phenomenon)} chars; directions "
                  f"{len(payload.get('new_mechanism_directions') or [])}", flush=True)

        with open(os.path.join(args.out, "history.jsonl"), "w",
                  encoding="utf-8") as fh:
            for rec in history:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(os.path.join(args.out, "library.json"), "w",
                  encoding="utf-8") as fh:
            json.dump([{k: v for k, v in o.items() if k != "vec"} for o in lib],
                      fh, ensure_ascii=False, indent=1)
        with open(os.path.join(args.out, "state.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"round": rnd, "best": best, "library_size": len(lib),
                       "phenomenon": phenomenon,
                       "elapsed_s": round(time.time() - t0, 1)},
                      fh, ensure_ascii=False, indent=1)

    sig = sorted({s for o in lib for s in o["signatures"]})
    print("\n=== done ===")
    print("best loss:", best["mse"])
    print("library:", len(lib), "entries; structural signatures present:",
          sig or "none")
    print("best equation:\n" + best["code"])
    print("\nphenomenon:\n" + phenomenon)


if __name__ == "__main__":
    try:
        main()
    except Exception:                                          # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
