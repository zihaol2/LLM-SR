"""Name the anonymous field from a prior reading plus the posterior equation evidence.

Two pieces of evidence are put in front of the model, and only their combination is
allowed to produce the answer:

  1. PRIOR -- the macro role the design step read off the measured statistics before
     any equation existed. It is a structural constraint, not ground truth.
  2. POSTERIOR -- the best equation the search found, plus the classical PDE families
     it looks like a modification of. The model names three candidate parents itself
     and states the standard physical role of every symbol inside each family, then
     carries those roles over to the anonymous symbols f1 / x1.

The equation is handed over TOGETHER WITH the coefficient values the search fitted
for it. The body alone is symbolic in `params[i]`, and `1 - params[2]*f1` with
params[2] = 1 is the statement `1 - f1` -- the term that separates a classical
family from a generic one. Dropping the values would hand over a weaker finding
than the one the search actually made.

Nothing in either prompt names the dataset. The naming has to be forced by the match.
"""
from __future__ import annotations

import json
import os
import re


# --------------------------------------------------------------------- prompts


def _fmt_param(value) -> str:
    """One fitted coefficient, parenthesised when negative so it reads in place."""
    text = f"{float(value):g}"
    return text if float(value) >= 0 else f"({text})"


def annotate_equation(equation_text: str, params) -> str:
    """Append the fitted coefficient values, and the equation with them filled in.

    Without this, an equation written as `1 - params[2]*f1` reaches the parent match
    with params[2] unknown, and a crowding factor that is exactly `1 - f1` cannot be
    told apart from a generic density dependence. The values are the search's own
    fit, so this states the same discovery more completely -- it adds no information
    that came from outside the run.
    """
    if not params:
        return equation_text

    def substitute(match: "re.Match") -> str:
        index = int(match.group(1))
        return _fmt_param(params[index]) if 0 <= index < len(params) else match.group(0)

    filled = re.sub(r"params\[(\d+)\]", substitute, equation_text)
    listed = ", ".join(f"params[{i}] = {float(v):g}" for i, v in enumerate(params))
    return (f"{filled.rstrip()}\n\n"
            "### FITTED COEFFICIENTS\n"
            f"The search fitted {listed}, and those values are written in place above.\n")


def parents_prompt(equation_text: str, dataset_description: str,
                   n_parents: int = 3) -> str:
    return f"""You are a mathematical physicist. An automated search has discovered ONE
equation for an anonymous one-dimensional dataset. Nothing here says what the field
is, and you are not asked to guess that yet.

Your job in THIS step is only the parent match: name the {n_parents} classical PDE
families this equation is most likely a modification of, and say exactly how it was
modified. Give, for each candidate parent:
  - "name": the standard family name. A "family" may be either a purely mathematical
    equation class, or an application-level model with a conventional name in some
    discipline. Give a MIX of both kinds, and include at least one application-level
    candidate. NO family name is supplied in this instruction, on purpose: any example
    here would bias the match and contaminate the comparison. Derive every candidate
    from the equation itself.
  - "canonical_form": the parent written term by term;
  - "term_mapping": which term of the discovered equation corresponds to which parent
    term, and which terms are ADDED, DROPPED or MODIFIED relative to the parent;
  - "symbol_roles": what the dependent variable and every coefficient usually stand
    for in that family, in physics terms;
  - "fit": "high" | "medium" | "low".

### THE DISCOVERED EQUATION
{equation_text}

### THE STRUCTURAL DESCRIPTION OF THE DATA
{dataset_description}

Answer with ONE JSON object, nothing else:
{{"parents": [{{"name": "...", "canonical_form": "...", "term_mapping": "...",
"symbol_roles": "...", "fit": "high"}}]}}
"""


def final_role_prompt(initial_role_text: str, equation_text: str,
                      parents_text: str, dataset_description: str) -> str:
    return f"""You are a physicist doing inverse identification. Two pieces of evidence
must be combined into ONE answer. Neither piece is allowed to name the dataset on its
own authority: the naming has to be forced by the evidence.

### EVIDENCE 1 -- prior: the macro role read off the measured statistics
This was produced before any equation was known, from measured statistics only. Treat
it as a structural constraint, not as ground truth; if it is vague or wrong, say so.
{initial_role_text}

### EVIDENCE 2 -- posterior: the discovered equation and its classical parents
The best equation the search found:
{equation_text}

The classical PDE families it appears to be a modification of, with the standard
physical role of every symbol in those families:
{parents_text}

### THE STRUCTURAL DESCRIPTION OF THE DATA
{dataset_description}

### YOUR TASK
1. PARENT DECISION. Say which parent family (or combination) the equation really
   belongs to, and why the modifications point there rather than to the runner-ups.
2. ROLE TRANSFER. Carry the symbol roles from that parent over to f1, x1 and f1_t, and
   reconcile them with Evidence 1. Where the two pieces of evidence disagree, say which
   one you trust and why.
3. NAME IT. State the concrete physical quantity f1 most likely represents, and the
   discipline and sub-field the equation belongs to. Point to the term or coefficient
   that pins the naming down.
4. CONFIDENCE. Give high | medium | low, and the single measurement or term that
   would most change your answer.

Answer with ONE JSON object, nothing else:
{{
  "variable": {{"symbol": "f1", "quantity": "...", "parent_family": "...", "reasoning": "..."}},
  "discipline": {{"field": "...", "subfield": "...", "reasoning": "..."}},
  "parent_ranking": [{{"name": "...", "fit": "high|medium|low", "why": "..."}}],
  "role_transfer": "...",
  "confidence": "high|medium|low",
  "key_evidence": ["...", "..."],
  "would_change_if": "..."
}}
"""


# ------------------------------------------------------------------ machinery


def parse_json(text: str) -> dict:
    """Parse the model's JSON, tolerating fences and surrounding prose."""
    body = (text or "").strip()
    if "```" in body:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", body, re.S)
        if blocks:
            body = max(blocks, key=len)
    try:
        return json.loads(body)
    except Exception:
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            return json.loads(body[start:end + 1])
        except Exception:
            return {}


def ask_json(call, prompt: str, label: str, attempts: int = 3, verbose: bool = True):
    """Ask until the reply parses, then give up loudly."""
    last = ""
    for attempt in range(1, attempts + 1):
        last = call(prompt)
        payload = parse_json(last)
        if payload:
            return payload, last
        if verbose:
            print(f"  [{label}] attempt {attempt}/{attempts}: reply was not JSON, "
                  f"retrying", flush=True)
            if attempt == 1:
                print(f"  [{label}] raw reply started: "
                      f"{' '.join((last or '').split())[:200]!r}", flush=True)
    return {}, last


def identify(call, initial_role_text: str, equation_text: str,
             dataset_description: str = "", n_parents: int = 3,
             verbose: bool = True, params=None) -> dict:
    """Run the two-step identification and return the full record."""
    resolved = annotate_equation(equation_text, params)
    parents_p = parents_prompt(resolved, dataset_description, n_parents)
    if verbose:
        print("role identification: matching the equation to its classical parents ...",
              flush=True)
    parents, parents_raw = ask_json(call, parents_p, "parents", verbose=verbose)
    parents_text = json.dumps(parents, ensure_ascii=False, indent=1)

    if verbose:
        names = [p.get("name") for p in parents.get("parents", [])
                 if isinstance(p, dict)]
        print(f"  parent candidates: {names}", flush=True)
        print("role identification: naming the variable from prior + posterior ...",
              flush=True)
    final_p = final_role_prompt(initial_role_text, resolved, parents_text,
                                dataset_description)
    final, final_raw = ask_json(call, final_p, "role", verbose=verbose)
    return {"parents_prompt": parents_p, "parents_raw": parents_raw,
            "parents": parents, "final_prompt": final_p,
            "equation_with_values": resolved,
            "final_raw": final_raw, "final": final}


def run_on_results(results_dir: str, call, n_parents: int = 3,
                   verbose: bool = True) -> dict:
    """Read a finished search directory, name the field, write the record back."""
    front_path = os.path.join(results_dir, "final_pareto_front.json")
    if not os.path.exists(front_path):
        raise SystemExit(f"no final_pareto_front.json in {results_dir}")
    with open(front_path, encoding="utf-8") as handle:
        front = json.load(handle)
    if not front:
        raise SystemExit(f"the Pareto front in {results_dir} is empty; nothing to name")
    best = front[0]

    dataset_description, initial_role_text = "", ""
    design_path = os.path.join(results_dir, "designed_library.json")
    if os.path.exists(design_path):
        with open(design_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        dataset_description = payload.get("dataset_description", "")
        roles = payload.get("inferred_roles") or {}
        claim = roles.get("f1") if isinstance(roles, dict) else None
        if isinstance(claim, dict):
            initial_role_text = str(claim.get("identity", ""))
        elif claim:
            initial_role_text = str(claim)
        if not initial_role_text.strip():
            # A prior-only design step produces no statistics-based role reading, so
            # fall back to the structural prior itself and say so in the prompt.
            prior = payload.get("prior_text", "")
            if prior:
                initial_role_text = ("[no measured-statistics reading was made for this "
                                     "run; this is the experimenter's structural prior] "
                                     + prior.strip())

    record = identify(call, initial_role_text, best.get("equation", ""),
                      dataset_description, n_parents, verbose,
                      params=best.get("params"))
    record["best_equation"] = {"mse": best.get("mse"),
                               "complexity": best.get("complexity"),
                               "params": best.get("params"),
                               "equation": best.get("equation")}
    record["initial_role"] = initial_role_text

    out_path = os.path.join(results_dir, "role_identification.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=1)
    if verbose:
        final = record.get("final") or {}
        print(f"role identification written to {out_path}", flush=True)
        print(f"  variable  : {final.get('variable')}", flush=True)
        print(f"  discipline: {final.get('discipline')}", flush=True)
        print(f"  confidence: {final.get('confidence')}", flush=True)
    return record


def make_call(api_model: str = "deepseek-v4-flash",
              api_base_url: str = "https://api.teamorouter.cn/v1",
              api_key_env: str = "TEAMOROUTER_API_KEY",
              max_tokens: int = 8192, thinking: str = "disabled"):
    """Build the single-request LLM call used by the standalone CLI."""
    import sampler
    import core as config_lib

    cfg = config_lib.Config(use_api=True, api_model=api_model, max_tokens=max_tokens,
                            thinking=thinking, api_base_url=api_base_url,
                            api_key_env=api_key_env, concurrent_requests=1)
    llm = sampler.LocalLLM(1, trim=False)
    llm._instruction_prompt = ""

    def call(prompt_text: str) -> str:
        out = llm._draw_samples_api(prompt_text, cfg)
        return out[0] if out else ""

    return call


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Name the anonymous field from a finished run")
    parser.add_argument("--results_dir", default="",
                        help="directory holding final_pareto_front.json")
    parser.add_argument("--equation_file", default="",
                        help="identify from an equation given directly, instead of a "
                             "finished run (for example the ground-truth skeleton)")
    parser.add_argument("--initial_role", default="",
                        help="Evidence 1 for --equation_file (a macro role reading)")
    parser.add_argument("--dataset_description", default="",
                        help="structural description used with --equation_file")
    parser.add_argument("--api_model", default="deepseek-v4-flash")
    parser.add_argument("--api_base_url", default="https://api.teamorouter.cn/v1")
    parser.add_argument("--api_key_env", default="TEAMOROUTER_API_KEY")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--thinking", default="disabled")
    parser.add_argument("--n_parents", type=int, default=6,
                        help="how many candidate parents to ask for; 6 rather than 3 "
                             "because a purely mathematical framing tends to crowd out "
                             "the application-level family that actually fits")
    args = parser.parse_args()

    call = make_call(args.api_model, args.api_base_url, args.api_key_env,
                     args.max_tokens, args.thinking)
    if args.equation_file:
        with open(args.equation_file, encoding="utf-8") as handle:
            equation = handle.read()
        record = identify(call, args.initial_role, equation,
                          args.dataset_description, args.n_parents)
        out = args.equation_file + ".role.json"
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=1)
        final = record.get("final") or {}
        print(f"written to {out}")
        print("  variable  :", final.get("variable"))
        print("  discipline:", final.get("discipline"))
        print("  confidence:", final.get("confidence"))
    else:
        if not args.results_dir:
            raise SystemExit("give either --results_dir or --equation_file")
        run_on_results(args.results_dir, call, args.n_parents)
