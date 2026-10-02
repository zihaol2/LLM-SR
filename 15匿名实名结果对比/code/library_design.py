"""Let the LLM design this problem's mechanism library (configuration B).

Human-imposed rules, not human-imposed physics:

  * it must state the SCOPE of the problem and what is out of scope;
  * it must organise the mechanisms into LEVELS with an explicit priority;
  * it must give each mechanism as a fixed-order operator over the primitives;
  * it must state a SEARCH PLAN (which level first, what to add or drop, how to judge);
  * it must list patterns that are BANNED here as unphysical.

The library it produces is then validated (closed algebra, honest order), measured
on the data (kept only if it produces usable atoms), frozen and written next to the
run so the prior is reportable. The search still has to find the equation in the
data -- the design only fixes the language.
"""
import json
import os
import re

import numpy as np

import pde_grammar
import mechanism_expert


PRIMITIVES = """
  D(f, axis, order)        derivative along an axis
  S(f, func)               func in {id, sin, cos, exp, tanh}
  MUL(a, b), DIV(a, b)     pointwise product / ratio
  POW(a, n)                pointwise power (n a positive integer)
  ABS(a)                   pointwise absolute value
  X(axis)                  the coordinate grid, e.g. X(0) is x1
  T(f)                     pointwise tanh
  coord(func, k1, k2, k3)  func(k1*x1 + k2*x2 + k3*x3)

Inside a definition you may also write + - * / directly. Because sums are allowed,
`exp` of an ARBITRARY polynomial is expressible, e.g.
  S(X(0) + a*POW(X(0),2) + b*POW(X(0),3), 'exp')      (free degree and #terms)
Reference mechanisms already implemented (you may use them inside a definition):
  adv/dif/disp/hdif/prs/frc/rxn/cpl/wave/coord_sin/coord_cos
"""


def design_prompt(variables_text: str, domain_text: str, n_max: int,
                  n_mechanisms: int = 10) -> str:
    return f"""You are the physicist in charge of designing the search language for ONE
problem. You are given the physical identity of the measured variables (nothing about
the answer). Your job is NOT to write the equation: it is to design the LIBRARY of
physical mechanisms in which the equation will be searched, and the plan for using it.

### THE VARIABLES (given, correct)
{variables_text}

### DOMAIN
{domain_text}

### THE META-GRAMMAR (fixed; every mechanism you define must be a fixed-order operator
### built from these primitives, with no fitted coefficients inside the definition)
{PRIMITIVES}

### THE RULES YOU MUST FOLLOW (these are the framework; the physics is yours)
1. SCOPE: state in one or two sentences what physics this problem can contain and
   what is explicitly OUT of scope here.
2. LEVELS: organise your mechanisms into 2-4 LEVELS (e.g. level 1 = the conservation
   or balance law, level 2 = constitutive closures, level 3 = higher-order/singular
   corrections), each with a priority (1 = explore first).
3. MECHANISMS: propose up to {n_mechanisms} mechanisms. Each is a NAMED operator with
   a signature, its true differential order (<= {n_max}), a definition over the
   primitives, which level it belongs to, and one line of physics.
4. SEARCH PLAN: say how the search should move through the levels, what to add when
   the residual stalls, and what to remove when complexity grows without accuracy.
5. BANNED: list patterns that would be unphysical for THIS problem.
6. Be broad inside your scope: an operator the library lacks cannot be found by the
   search, so do not leave an obvious physical mechanism out.

Answer with ONE JSON object, nothing else:
{{
  "scope": "...",
  "out_of_scope": "...",
  "levels": [{{"level": 1, "name": "...", "priority": 1, "idea": "..."}}],
  "mechanisms": [
    {{"name": "flux_div", "signature": ["phi", "psi", "axis"], "order": 1,
      "definition": "D(MUL(phi, psi), axis, 1)",
      "physics": "divergence of a pairwise flux", "level": 1, "priority": 1}}
  ],
  "search_plan": "...",
  "banned": ["..."]
}}
"""


def parse_design(text: str) -> dict:
    if not text:
        return {}
    body = text.strip()
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


def _split(atoms, data, train, test, budget):
    return mechanism_expert._split_residual(atoms, data, train, test, budget)


def measure_ratio(spec, data, budget=3, seed=0, test_frac=0.3):
    """How much the mechanism enlarges a short held-out fit (kept registered)."""
    mechanism_expert.bind_data_grammar(data)      # X(0) / coord need the grids
    pde_grammar.register_mechanism(spec)
    new_atoms = mechanism_expert.mechanism_atoms(spec, data)
    if not new_atoms:
        return None, 0
    base = mechanism_expert.base_atoms(data)
    n = np.asarray(data[mechanism_expert._targets(data)[0]], dtype=float).size
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = max(5, int(test_frac * n))
    test, train = order[:n_test], order[n_test:]
    r0, _ = _split(base, data, train, test, budget)
    r1, _ = _split(base + new_atoms, data, train, test, budget)
    return (r1 / r0 if r0 > 0 else 1.0), len(new_atoms)


def design(llm_call, variables_text: str, domain_text: str, data, n_max: int,
           n_mechanisms: int = 10, verbose: bool = True):
    """One design round. Returns (design_dict, accepted_specs, log)."""
    return design_from_prompt(
        llm_call,
        design_prompt(variables_text, domain_text, n_max, n_mechanisms),
        data, n_max, verbose)


def design_from_prompt(llm_call, prompt: str, data, n_max: int, verbose: bool = True):
    """Same as `design`, but the caller supplies the prompt."""
    text = llm_call(prompt)
    design = parse_design(text)
    if not design:
        if verbose:
            print(f"[design] no parsable design; raw: "
                  f"{' '.join((text or '').split())[:300]!r}", flush=True)
        return {}, [], ["no design"]

    accepted, log = [], []
    seen = set()
    for proposal in design.get("mechanisms", []):
        if not isinstance(proposal, dict):
            continue
        name = str(proposal.get("name", "")).strip()
        if not name or name in seen or name in pde_grammar.MECHANISM_NAMES:
            continue
        seen.add(name)
        signature = proposal.get("signature")
        if not isinstance(signature, list) or not signature:
            log.append(f"invalid {name}: no signature")
            continue
        try:
            order = int(proposal.get("order"))
        except Exception:
            log.append(f"invalid {name}: no order")
            continue
        definition = str(proposal.get("definition", "")).strip()
        used = set()
        try:
            import ast
            used = {n.id for n in ast.walk(ast.parse(definition, mode="eval"))
                    if isinstance(n, ast.Name)}
        except Exception:
            pass
        signature = [str(s) for s in signature]
        if "axis" in used and "axis" not in signature:
            signature.append("axis")
        error = pde_grammar.validate_definition(signature, definition, order)
        if error:
            log.append(f"invalid {name}: {error}")
            continue
        if order > n_max:
            log.append(f"invalid {name}: order {order} > {n_max}")
            continue
        spec = {"name": name, "signature": signature, "order": order,
                "definition": definition, "builtin": False,
                "physics": str(proposal.get("physics", "")).strip(),
                "level": proposal.get("level"), "priority": proposal.get("priority")}
        ratio, n_atoms = measure_ratio(spec, data)
        if ratio is None:
            pde_grammar.unregister_mechanism(name)
            log.append(f"invalid {name}: no usable atom")
            continue
        spec["evidence_ratio"] = float(ratio)
        spec["n_atoms"] = int(n_atoms)
        accepted.append(spec)
        log.append(f"accepted {name} (order {order}, ratio {ratio:.3f})")
    if verbose:
        for line in log:
            print(f"[design] {line}", flush=True)
    return design, accepted, log


def render(design: dict, specs) -> str:
    lines = ["### DESIGNED MECHANISM LIBRARY (built for this problem, from the variable"
             " identities)"]
    if design.get("scope"):
        lines.append(f"Scope: {design['scope']}")
    if design.get("out_of_scope"):
        lines.append(f"Out of scope: {design['out_of_scope']}")
    for level in design.get("levels", []):
        lines.append(f"  level {level.get('level')} (priority {level.get('priority')}): "
                     f"{level.get('name', '')} -- {level.get('idea', '')}")
    by_name = {s["name"]: s for s in specs}
    ordered = sorted(specs, key=lambda s: (s.get("level") or 99, s.get("priority") or 99))
    lines.append("Mechanisms (use these as `G.<name>(...)`):")
    for spec in ordered:
        lines.append(f"  {spec['name']}({', '.join(spec['signature'])})   "
                     f"order {spec['order']}   level {spec.get('level')}   "
                     f"{spec.get('physics', '')}")
        lines.append(f"      definition: {spec['definition']}")
    if design.get("search_plan"):
        lines.append(f"Search plan: {design['search_plan']}")
    if design.get("banned"):
        lines.append("BANNED here (unphysical): " + "; ".join(
            str(b) for b in design["banned"]))
    return "\n".join(lines)


# ------------------------------------------------------- anonymous (config C)
# The physical identity of the variables is NOT given. The model first decides what
# to measure, we compute it, and only then does it infer the roles and design the
# library. "What to measure" is therefore the model's choice, not ours.

import stat_engine  # noqa: E402  (kept next to this module)


PLANNER_SYSTEM = (
    "You are an expert in statistics and data analysis. You are given only a structural "
    "description of an anonymous dataset, with no subject-matter information at all. "
    "Your job is to design the measurement plan: decide which statistics should be "
    "computed so that the data is described AND so that competing functional "
    "explanations of it can be told apart. Write the plan generally, so that it would "
    "be sensible for any spatiotemporal dataset of this shape, and assume nothing about "
    "the subject matter. You may only ask for statistics from the catalogue below.\n"
)


def plan_prompt(dataset_description: str, n_ops: int = 14) -> str:
    return f"""{PLANNER_SYSTEM}
### THE ANONYMOUS DATASET (structure only, no physical meaning attached)
{dataset_description}

### THE STATISTICS YOU MAY REQUEST
{stat_engine.OPS}

### RULES
1. Return between 8 and {n_ops} requests.
2. The plan has two required parts. The first part describes the data: each field on
   its own, its spectra and its basic statistics. The second part is discriminative:
   measurements whose only purpose is to separate competing functional explanations
   of the same data.
3. In the discriminative part, at least one request must test whether the target
   balances over the domain (balance_test), at least one must test whether the
   response of the target depends on the state of the field (nested_fit,
   state_conditioned_response), and at least one must test whether that response
   varies in space (segmented_response). These three are required, not optional.
4. Cover the relation between each field and the measured target as well as the fields
   on their own.
5. Keep the plan general. Do not assume a particular application, a particular number
   of fields, or any specific mechanism, and do not name a subject area.
6. Every structural claim you make must agree with the dimensions and shapes in the
   dataset description. Do not assume axes that are not listed there.
7. Say nothing about what you expect to find. Only list what to measure.

Answer with ONE JSON array, nothing else:
[
  {{"op": "summary", "field": "f1"}},
  {{"op": "space_spectrum", "field": "f1", "at_time": "mid"}},
  {{"op": "correlate_with_target", "field": "f1", "order": 1}}
]
"""


def parse_plan(text: str):
    if not text:
        return []
    body = text.strip()
    if "```" in body:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", body, re.S)
        if blocks:
            body = max(blocks, key=len)
    try:
        payload = json.loads(body)
    except Exception:
        start, end = body.find("["), body.rfind("]")
        if start < 0 or end <= start:
            return []
        try:
            payload = json.loads(body[start:end + 1])
        except Exception:
            return []
    if isinstance(payload, dict):
        payload = [payload]
    return [p for p in payload if isinstance(p, dict)]


def design_prompt_from_stats(stats_report: str, n_max: int,
                             n_mechanisms: int = 10) -> str:
    return f"""You are a physicist. The dataset is anonymous: no variable carries a
name, and nothing about its physical meaning was provided. You chose the statistics
below yourself in the previous step, and they have now been measured.

### MEASURED EVIDENCE (the statistics you asked for)
{stats_report}

### THE META-GRAMMAR (fixed; every mechanism you define must be a fixed-order operator
### built from these primitives, with no fitted coefficients inside the definition)
{PRIMITIVES}

### YOUR TASK, in this order
0. STRUCTURAL SELF-CHECK. Before anything else, restate to yourself the number of
   spatial coordinates and the shape of each array exactly as given above, and make
   every later statement agree with them. Do not assume axes that are not present.
1. INFER THE PHYSICAL CHARACTER OF EACH FIELD. For every field, state what physical
   quantity it most likely represents, how confident you are, and exactly which
   measured numbers support that reading. If several scenarios stay plausible, rank
   them and say what measurement would separate them.
2. DESIGN THE MECHANISM LIBRARY for the scenario you favour, under the same rules as
   before: state scope and out-of-scope, organise the mechanisms into 2 to 4 levels
   with priorities, give a search plan, list patterns that are banned as unphysical,
   and give each mechanism a name, a signature, its true differential order
   (at most {n_max}) and a definition over the primitives.

Answer with ONE JSON object, nothing else:
{{
  "inferred_roles": {{
    "f1": {{"identity": "...", "confidence": "high|medium|low", "evidence": "..."}}
  }},
  "scope": "...",
  "out_of_scope": "...",
  "levels": [{{"level": 1, "name": "...", "priority": 1, "idea": "..."}}],
  "mechanisms": [
    {{"name": "flux_div", "signature": ["phi", "psi", "axis"], "order": 1,
      "definition": "D(MUL(phi, psi), axis, 1)",
      "physics": "divergence of a pairwise flux", "level": 1, "priority": 1}}
  ],
  "search_plan": "...",
  "banned": ["..."]
}}
"""
