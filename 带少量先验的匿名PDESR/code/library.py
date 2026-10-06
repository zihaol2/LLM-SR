"""Designing this problem's mechanism library.

The design prompt (the prior is authoritative, the measurements are secondary),
the JSON-tree validation that turns a proposal into a mechanism, the bounded
statistics executor the model may ask for, and the atom check that decides
whether a proposed mechanism can be instantiated on the data at all.
"""
from __future__ import annotations
import ast
import json
import os
import re

import numpy as np
from scipy.interpolate import make_interp_spline

import grammar


# ==========================================================================
# the design prompt and its validation
# ==========================================================================

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




# Rendered once: the structured-expression vocabulary the LLM writes against.
EXPR_VOCAB = grammar.vocabulary_text()


REPAIR_REMINDER = (
    "\n\n### RETRY\n"
    "Your previous reply was not the JSON value requested (it was prose, or it was cut "
    "off before the JSON closed). Reply again with ONLY that JSON value: no "
    "introduction, no restatement, no headings, no commentary and no markdown fences. "
    "Start with the opening brace and end with the closing brace, and keep each "
    "mechanism's \"expr\" short.")


PRIMITIVES = """
  D(f, axis, order)        derivative of order `order` along an axis. `axis` is an
                           INTEGER index (0 means x1) -- never a grid or a coordinate
  S(f, func)               func in {id, sin, cos, exp, tanh}
  MUL(a, b), DIV(a, b)     pointwise product / ratio
  POW(a, n)                pointwise power (n a positive integer)
  ABS(a)                   pointwise absolute value
  X(axis)                  the coordinate GRID of an axis, e.g. X(0) is x1; it is a
                           grid to be used inside functions, never as the `axis`
                           argument of D
  T(f)                     pointwise tanh
  coord(func, k1, k2, k3)  func(k1*x1 + k2*x2 + k3*x3)

Inside a definition you may also write + - * / directly. Because sums are allowed,
`exp` of an ARBITRARY polynomial is expressible, e.g.
  S(X(0) + a*POW(X(0),2) + b*POW(X(0),3), 'exp')      (free degree and #terms)
There is NO `ADD` / `SUB` / `MUL3` helper: write sums and differences with infix
`+` and `-`, and nest `MUL(a, b)` when you need three factors.
Reference mechanisms already implemented (you may use them inside a definition):
  adv/dif/disp/hdif/prs/frc/rxn/cpl/wave/coord_sin/coord_cos
"""


def design_prompt(variables_text: str, domain_text: str, n_max: int,
                  n_mechanisms: int = 10) -> str:
    return f"""You design the search language for ONE problem. You are told what is
known about the measured variables and nothing about the answer. Your job is NOT to
write the equation: it is to design the LIBRARY of physical mechanisms the equation
will be searched in, and the plan for using it.

### WHAT IS KNOWN ABOUT THE VARIABLES (given, correct)
{variables_text}

### DOMAIN
{domain_text}

### THE EXPRESSION FORMAT (fixed; you write a JSON tree, never operator strings)
A mechanism definition is a JSON list `[op, arg, ...]`. Leaves are
  "f1"           a measured field, by name
  "phi"          a formal you declare -- a scalar slot the search fills in
  1  /  0.5      a numeric literal
Every node is ONE operation from this vocabulary; arity, argument kinds and order
rules are enforced by the compiler, so the syntax cannot be got wrong.

{EXPR_VOCAB}

Examples (note how a transport mechanism ends in its outer divergence):
  ["d", ["mul", "f1", ["x", "axis"]], "axis", 1]                  d/dx[ f1 * x1 ]
  ["flux_div", ["mul", ["sub", 1, "f1"], ["coord", "tanh", 1, 0, 0]], "axis"]
  ["gradient_flux", ["inv", ["add", 1, "f1"]], "f1", "axis"]

### THE RULES YOU MUST FOLLOW (these are the framework; the physics is yours)
1. SCOPE: state in one or two sentences what physics this problem can contain and
   what is explicitly OUT of scope here.
2. LEVELS: organise your mechanisms into 2-4 LEVELS with a priority (1 = explore
   first). LEVEL 1 MUST CARRY THE PRIOR'S OWN CLAIMS -- the balance law AND every
   effect the text above names (a geometry-dependent modulation, an anticipation /
   relative-gradient response, ...). Levels 2-3 are only for refinements the prior
   does NOT name.
3. MECHANISMS: propose up to {n_mechanisms}. Each is a NAMED operator with "expr" (the
   tree above), its level and one line of physics; signature and differential order
   come from the tree -- do not write them. A TRANSPORT mechanism is the DIVERGENCE of
   its flux, i.e. its outermost node is `flux_div`, `pair_flux_div`, `gradient_flux`,
   `log_gradient_flux`, `power_gradient_flux`, or `d` of order 1; a bare flux is an
   algebraic source, not transport. Propose DISTINCT mechanisms -- one that differs
   from another only by an extra factor or another saturation wastes the budget.
   EXPOSE EVERY OPEN NUMBER. Whatever the text above does not fix -- a scale (how fast
   a shape varies), a depth (how far a profile dips), a fraction -- must be a formal
   the search fills in, never a literal baked into the definition. Declare it in the
   signature and put its name where the number would go:
     ["flux_div", ["mul", "f1", ["mul", ["sub", 1,
        ["mul", "a", ["coord", "tanh", "k", 0, 0]]], ["sub", 1, "f1"]]], "axis"]
   `k` sets how fast the profile varies, `a` how deeply it dips; the search passes both
   at the call, which a definition with `1` baked in cannot accept.
4. SEARCH PLAN: say how the search should move through the levels, what to add when
   the residual stalls, and what to remove when complexity grows without accuracy.
5. BANNED: list patterns that would be unphysical for THIS problem.
6. COVER EVERY CLAIM. An operator the library lacks cannot be found by the search, so
   walk the variables text claim by claim and make every structure it asserts
   expressible by at least one mechanism. Anything you leave out goes in "excluded"
   with the reason.

Answer with ONE JSON object, nothing else:
{{
  "scope": "...",
  "out_of_scope": "...",
  "levels": [{{"level": 1, "name": "...", "priority": 1, "idea": "..."}}],
  "mechanisms": [
      {{"name": "pair_flux", "expr": ["pair_flux_div", "f1", "f1", "axis"],
        "physics": "divergence of a pairwise flux", "level": 1, "priority": 1}}
  ],
  "search_plan": "...",
  "banned": ["..."],
  "excluded": ["a structure the variables text asserts that you chose NOT to cover, and why"]
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
    payload = None
    try:
        payload = json.loads(body)
    except Exception:
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            payload = json.loads(body[start:end + 1])
        except Exception:
            return {}
    if isinstance(payload, list):
        # The model sometimes answers with the mechanisms array itself.
        payload = {"mechanisms": payload}
    if not isinstance(payload, dict):
        return {}
    return payload


def canonicalize_axis(signature, definition):
    """Rename whichever formal the definition uses as `D`'s axis argument to `axis`.

    The atom builder recognizes the axis slot by the name `axis` (an integer index).
    A model that calls that formal `x_axis`, `ax` or `dir` would otherwise have it
    treated as a scalar coefficient, and the mechanism would be rejected for a reason
    that has nothing to do with its physics. Pure alpha-conversion of signature and
    definition.
    """
    try:
        tree = ast.parse(definition, mode="eval")
    except SyntaxError:
        return signature, definition
    used_as_axis = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "D"
                and len(node.args) >= 2 and isinstance(node.args[1], ast.Name)
                and node.args[1].id in signature):
            used_as_axis.add(node.args[1].id)
    rename = {name: "axis" for name in used_as_axis if name != "axis"}
    if not rename:
        return signature, definition
    new_signature, seen = [], set()
    for name in signature:
        canonical = rename.get(name, name)
        if canonical not in seen:
            seen.add(canonical)
            new_signature.append(canonical)
    new_definition = definition
    for old, new in rename.items():
        new_definition = re.sub(rf"\b{re.escape(old)}\b", new, new_definition)
    return new_signature, new_definition


def atom_usable(spec, data) -> bool:
    """The admission gate: USABILITY, not usefulness.

    A mechanism is kept when it compiles and produces at least one finite, correctly
    shaped atom on this problem. Nothing else is measured here -- the earlier
    held-out "evidence ratio" was recorded but never read, and it read like a quality
    filter while accepting six of ten entries at its no-information value.
    """
    bind_data_grammar(data)      # X(0) / coord need the grids
    grammar.register_mechanism(spec)
    atoms = mechanism_atoms(spec, data)
    if not atoms:
        grammar.unregister_mechanism(spec["name"])
        return False
    return True


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
    for attempt in range(2, 4):
        if design:
            break
        if verbose:
            print(f"[design] attempt {attempt}: reply was not parsable JSON "
                  f"(truncated or prose); retrying", flush=True)
        text = llm_call(prompt + REPAIR_REMINDER)
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
        if not name or name in seen or name in grammar.MECHANISM_NAMES:
            continue
        seen.add(name)
        if proposal.get("expr") is not None:
            # Structured path: we own the syntax, the signature is DERIVED from the
            # tree's leaves, and the differential order is computed, not trusted.
            expr = grammar.canonicalize_axes(proposal["expr"])
            signature = grammar.signature_of(expr, [])
            reason = grammar.validate(expr, signature)
            if reason:
                log.append(f"invalid {name}: {reason}")
                continue
            order = grammar.order_of(expr)
            if order > n_max:
                log.append(f"invalid {name}: order {order} > {n_max}")
                continue
            spec = {"name": name, "signature": signature, "order": order,
                    "expr": expr, "builtin": False,
                    "physics": str(proposal.get("physics", "")).strip(),
                    "level": proposal.get("level"),
                    "priority": proposal.get("priority")}
            # keep the compiled form as well, so the registry can render it
            spec["definition"] = grammar.compile_expr(expr, signature)[0]
            ok = atom_usable(spec, data)
            if not ok:
                why = LAST_ATOM_ERROR
                log.append(f"invalid {name}: no usable atom"
                           + (f" ({why})" if why else ""))
                continue
            accepted.append(spec)
            log.append(f"accepted {name} (order {order})")
            continue
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
        tree = None
        try:
            tree = ast.parse(definition, mode="eval")
            used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        except Exception:
            pass
        # The declared differential order is bookkeeping; the definition is the truth.
        # When the two disagree (a common slip: `D(MUL(f1, D(f1,axis,1)), axis, 1)` is
        # order 2 but was signed as 1) trust the definition instead of throwing away an
        # otherwise usable mechanism. It is still rejected above if it exceeds n_max.
        if tree is not None:
            try:
                computed_order = int(grammar._definition_order(tree.body))
                if computed_order > order:
                    log.append(f"repaired {name}: declared order {order} -> {computed_order}")
                    order = computed_order
                elif computed_order < order:
                    # The definition is SHORT of the order it claims: the classic
                    # "wrote the flux but forgot the outer divergence" slip (exactly
                    # how folder 09's search stalled). Do not silently demote it to an
                    # algebraic source term -- reject and say why.
                    log.append(f"invalid {name}: declares order {order} but the definition "
                               f"computes order {computed_order} -- a transport mechanism "
                               f"must be the DIVERGENCE of its flux")
                    continue
            except Exception:
                pass
        signature = [str(s) for s in signature]
        # Repair the common mismatch: the definition names a symbol the author forgot
        # to list as a formal (e.g. `D(MUL(c, f1), axis, 1)` signed only ['c','axis']).
        # `axis` was already handled this way; extend it to every identifier that is a
        # variable rather than a grammar call, so a usable mechanism is not thrown away
        # over a bookkeeping slip.
        known_calls = (set(grammar.PRIMITIVE_NAMES)
                       | set(grammar.MECHANISM_NAMES)
                       | set(grammar._FUNCS))
        for used_name in sorted(used - set(signature) - known_calls):
            if used_name.isidentifier():
                signature.append(used_name)
        # A formal that the definition uses as D's axis argument must be called
        # `axis`, or the atom builder will treat it as a scalar coefficient.
        signature, definition = canonicalize_axis(signature, definition)
        error = grammar.validate_definition(signature, definition, order)
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
        if not atom_usable(spec, data):
            why = LAST_ATOM_ERROR
            log.append(f"invalid {name}: no usable atom"
                       + (f" ({why})" if why else ""))
            continue
        accepted.append(spec)
        log.append(f"accepted {name} (order {order})")
    if verbose:
        for line in log:
            print(f"[design] {line}", flush=True)
    return design, accepted, log


def render_guidance(design: dict, specs) -> str:
    """The library as MACRO GUIDANCE for a free-numpy search.

    Nothing here is an API: the search writes the equation itself, in numpy, and the
    library only says which structures to aim for and in what order. Each entry is
    shown by its mathematical definition instead of by a call signature, so the model
    is not nudged into calling something that does not exist.
    """
    lines = ["### LIBRARY GUIDANCE (directions for this problem, ordered by priority)",
             "",
             "These are NOT functions to call. They are the shapes this problem is expected",
             "to be built from, written out so you can see what is meant. Write the equation",
             "yourself with the allowed operators; anything that follows this guidance and",
             "reproduces the measurements is acceptable."]
    if design.get("scope"):
        lines += ["", f"Scope: {design['scope']}"]
    if design.get("out_of_scope"):
        lines.append(f"Out of scope: {design['out_of_scope']}")
    for level in design.get("levels", []):
        lines.append(f"  level {level.get('level')} (priority {level.get('priority')}): "
                     f"{level.get('name', '')} -- {level.get('idea', '')}")
    ordered = sorted(specs, key=lambda s: (s.get("level") or 99, s.get("priority") or 99))
    lines += ["", "Structures to aim for, in the order they should be tried:"]
    for spec in ordered:
        free = [s for s in spec.get("signature", []) if s not in ("axis", "f1")]
        note = (f"   (free scale(s) {', '.join(free)} -- choose the value, do not fix it)"
                if free else "")
        lines.append(f"  - {spec['name']}: {spec.get('definition', '')}{note}")
        if spec.get("physics"):
            lines.append(f"      {spec['physics']}")
    if design.get("search_plan"):
        lines += ["", f"Search plan: {design['search_plan']}"]
    if design.get("banned"):
        lines += ["", "BANNED here (unphysical): "
                  + "; ".join(str(b) for b in design["banned"])]
    return "\n".join(lines)


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
        # Show the CALL syntax, not a bare name: a bare rendering gets copied
        # verbatim into the equation body and dies with a NameError.
        lines.append(f"  G.{spec['name']}({', '.join(spec['signature'])})   "
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

PLANNER_SYSTEM = (
    "You are an expert in statistics and data analysis. You are given only a structural "
    "description of an anonymous dataset, with no subject-matter information at all. "
    "Your job is to design the measurement plan: decide which statistics should be "
    "computed so that the data is described AND so that competing functional "
    "explanations of it can be told apart. Write the plan generally, so that it would "
    "be sensible for any spatiotemporal dataset of this shape, and assume nothing about "
    "the subject matter. You may only ask for statistics from the catalogue below.\n"
)


def role_only_prompt(stats_report: str) -> str:
    """The measurements -> a macro role reading, and nothing else.

    This is Evidence 1 of the role identification. It deliberately does NOT see the
    provided prior: the whole point is that the two pieces of evidence are independent
    (one from measurements, one from the discovered equation and its parents).
    """
    return f"""You are a physicist. The dataset is anonymous: no variable carries a name,
and nothing about its physical meaning is provided. You are given only measurements,
made on the data. Your ONLY job is to say what the field most likely is and how its
evolution is structured, as accurately as those numbers allow. Do NOT design
mechanisms and do NOT propose an equation.

### MEASURED EVIDENCE
{stats_report}

Be concise and precise. The statement is used as one side of a later identification,
so it must contain no claim the numbers do not support; if the numbers cannot separate
two readings, say which one they favour and what would separate them.

Answer with ONE JSON object, nothing else:
{{
  "inferred_roles": {{
    "f1": {{"identity": "...", "confidence": "high|medium|low", "evidence": "..."}}
  }}
}}
"""


def plan_prompt(dataset_description: str, n_ops: int = 14,
                prior_text: str = "") -> str:
    if prior_text:
        system = (
            "You are an expert in statistics and data analysis. You are given an "
            "anonymous dataset together with a SHORT STRUCTURAL PRIOR from the "
            "experimenter. The subject-area nouns are withheld, but the structural "
            "claims in the prior are given as correct and come FIRST in authority. "
            "Your job is to design the measurement plan: decide which statistics "
            "should be computed so that the data is described AND so that competing "
            "functional explanations can be told apart WITHIN the frame the prior "
            "fixes. Read the prior before anything else and let it decide what "
            "matters; the dataset description only fixes dimensions. You may only ask "
            "for statistics from the catalogue below.\n")
        body = ("### 1. PROVIDED PRIOR -- AUTHORITATIVE, READ FIRST\n"
                + prior_text.strip()
                + "\n\n### 2. DATASET STRUCTURE (dimensions and ranges only)\n"
                + dataset_description)
    else:
        system = PLANNER_SYSTEM
        body = ("### THE ANONYMOUS DATASET (structure only, no physical meaning "
                "attached)\n" + dataset_description)
    return f"""{system}
{body}

### THE STATISTICS YOU MAY REQUEST
{OPS}

### RULES
0. The plan must be able to CHECK the prior's structural claims (when a prior is
   given), not merely describe the data. Spend requests on the parts of the prior
   that the data could contradict or confirm.
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
5. Do not invent a subject area: never name the domain, and never assume fields or
   mechanisms beyond what the structure (and, when given, the prior) states.
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
                             n_mechanisms: int = 10,
                             prior_text: str = "") -> str:
    if prior_text:
        opening = (
            "You are a physicist. The dataset is anonymous: no variable carries a name.\n"
            "You are given TWO sources of information, in this order of authority.\n\n"
            "### 1. PROVIDED PRIOR -- AUTHORITATIVE. READ THIS FIRST.\n"
            "Supplied by the experimenter and given as correct. Only the subject-area\n"
            "nouns were withheld. It fixes the physical frame: what kind of system this\n"
            "is, which structures exist, and which mechanisms must be expressible.\n\n"
            + prior_text.strip() + "\n\n"
            "### 2. MEASURED EVIDENCE -- SECONDARY. Used only inside that frame.\n"
            "The statistics you chose yourself in the previous step, now measured:\n\n"
            + stats_report + "\n\n"
            "PRECEDENCE RULE: the prior fixes the scope; the measured numbers choose\n"
            "among the mechanisms that live inside it and set their levels and\n"
            "priorities. Every structural claim the prior makes MUST end up expressible\n"
            "by at least one mechanism. If a measurement seems to point outside the\n"
            "prior's frame, do NOT silently override the prior -- record it under\n"
            "\"tension_with_prior\" and keep the mechanism the prior requires.\n")
    else:
        opening = (
            "You are a physicist. The dataset is anonymous: no variable carries a\n"
            "name, and nothing about its physical meaning was provided. You chose the\n"
            "statistics below yourself in the previous step, and they have now been\n"
            "measured.\n\n### MEASURED EVIDENCE (the statistics you asked for)\n"
            + stats_report + "\n")
    return f"""{opening}
### THE META-GRAMMAR (fixed; every mechanism you define must be a fixed-order operator
### built from these primitives, with no fitted coefficients inside the definition)
{PRIMITIVES}

### YOUR TASK, in this order
0. STRUCTURAL SELF-CHECK. Before anything else, restate to yourself the number of
   spatial coordinates and the shape of each array exactly as given above, and make
   every later statement agree with them. Do not assume axes that are not present.
0b. LIST THE PRIOR'S DEMANDS (when a prior is given). Name every structure the prior
    says must exist, and check that the library you are about to design can express
    each of them. A prior's claim with no mechanism behind it is an unfinished design.
1. INFER THE PHYSICAL CHARACTER OF EACH FIELD. For every field, state what physical
   quantity it most likely represents, how confident you are, and exactly which
   measured numbers support that reading. Anchor the reading in the prior FIRST and
   then let the measured numbers confirm or refine it; if the two disagree, follow the
   prior and say so. If several scenarios stay plausible, rank them and say what
   measurement would separate them.
2. DESIGN THE MECHANISM LIBRARY for the scenario you favour, under the same rules as
   before: state scope and out-of-scope, organise the mechanisms into 2 to 4 levels
   with priorities, give a search plan, list patterns that are banned as unphysical,
   and give each mechanism a name, a signature, its true differential order
   (at most {n_max}) and a definition over the primitives.
   START FROM THE PRIOR: every structure the prior demands must have at least one
   mechanism that can express it. Only then add the mechanisms the measurements call
   for, and use the measurements to set levels and priorities.
   BE BROAD, AND COVER EVERY CLAIM: walk through the prior claim by claim; an operator
   the library lacks cannot be found by the search, so do not leave an obvious
   mechanism out. If you deliberately leave one out, say so in "excluded" and why.
   SIGNATURE CONVENTION: every identifier that appears in the definition must be one
   of the names listed in the signature. The measured field is written `f1`; if the
    mechanism acts on the field, put `f1` in the signature too. Using formal names
    instead is equally fine, e.g.
      {{"name": "pair_flux", "signature": ["phi", "psi", "axis"],
        "definition": "D(MUL(phi, psi), axis, 1)"}}
   A definition that mentions `f1` (or any other symbol) without listing it in the
   signature is rejected before it reaches the data.
   AXIS CONVENTION: every differentiation is written `D(<expression>, axis, k)` with
   `axis` the INTEGER axis formal (0 means x1), and `axis` listed in the signature.
   `X(0)` is the coordinate grid, to be used inside a function -- never as that axis.
   PRACTICAL NOTES:
     * Absolute value is the primitive `ABS(a)`; `S(a, func)` accepts only
       id / sin / cos / exp / tanh, so `S(a, 'abs')` is rejected.
       Likewise there is no `TANH` / `EXP` / `SIN` / `COS`: tanh is `T(a)` or
       `S(a, 'tanh')`, exponential is `S(a, 'exp')`.
     * Write sums with INFIX `+` (e.g. `DIV(1, 1 + f1)`), never `ADD(...)` / `SUB(...)`.
     * Each definition must be SELF-CONTAINED over the primitives: a definition may
       not call a mechanism proposed in the same answer (they are validated one at a
       time, so a reference to a sibling is rejected -- inline what you need).
     * Propose DISTINCT mechanisms; near-duplicates waste the budget.
     * EXPOSE EVERY OPEN NUMBER. Whatever the prior does not fix -- a scale (how fast
       a shape varies), a depth (how far a profile dips), a fraction -- must be a
       formal the search fills in, never a literal baked into the definition. Declare
       it in the signature and put its name where the number would go:
         ["flux_div", ["mul", "f1", ["mul", ["sub", 1,
            ["mul", "a", ["coord", "tanh", "k", 0, 0]]], ["sub", 1, "f1"]]], "axis"]
     * EVERY transport mechanism must be the DIVERGENCE of its flux: write the outer
       derivative explicitly, `D( <flux> , axis, 1)`. A bare flux with no outer
       derivative is an algebraic source, not transport, and is rejected.
     * Only integer numeric literals are allowed; write a small offset as `DIV(1, 10)`.

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
  "banned": ["..."],
  "tension_with_prior": "a measurement that points outside the prior's frame, or empty",
  "excluded": ["a prior claim you chose NOT to cover, and why"]
}}
"""


# ==========================================================================
# the bounded statistics executor
# ==========================================================================

"""A bounded statistics executor: the LLM chooses WHAT to measure, we compute it.

No physical identity is given to the model. It receives the structural description
of the anonymous arrays and a catalogue of statistics it may request, then returns a
JSON plan. This module validates that plan (known op, known array, bounded sizes) and
executes it, returning a readable report.
"""



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


# ==========================================================================
# instantiating a proposed mechanism on the data (atom check)
# ==========================================================================

"""The LLM-as-physics-expert loop that grows the mechanism library.

One round: show the current library + the measured data evidence, let the model
propose named fixed-order operators (it may use the new primitives DIV / T / coord),
validate them, and promote only the ones that measurably enlarge what a short
equation can express on a held-out split.

The gate is self-contained: a small greedy forward selection over the built-in
atoms, compared with the same selection once the proposal's atoms are added, on
rows the coefficients never saw. No screening/coverage machinery involved.
"""





# Last exception raised while instantiating a proposal's atoms; "no usable atom" is
# otherwise undiagnosable. Read by the design step when it logs a rejection.
LAST_ATOM_ERROR = ""


SYSTEM = (
    "You are a physicist who studies partial differential equations. You are looking "
    "at an anonymous dataset (fields f1..fN and their measured time derivatives) and "
    "at the operator library a machine may search. Your job is to extend that library "
    "with new physical mechanisms it cannot express, so the search can build better "
    "equations.\n"
    "A mechanism is ONE named operator with a FIXED differential order. Write it as a "
    "closed algebraic definition over the primitives D(f, axis, order), S(f, func), "
    "MUL(a, b), DIV(a, b), T(f), coord(func, k1, k2, k3) and the mechanisms that "
    "already exist. No numpy, no arithmetic operators, no attributes, no free-form "
    "expressions, no fitted coefficients -- only the fixed structure.\n"
)


# ------------------------------------------------------------------ binding
def _axes(data):
    return sorted((k for k in data if re.fullmatch(r"x\d+", k)), key=lambda s: int(s[1:]))


def _fields(data):
    return sorted((k for k in data if re.fullmatch(r"f\d+", k)), key=lambda s: int(s[1:]))


def _targets(data):
    return sorted((k for k in data if k.endswith("_t_true")),
                  key=lambda s: int(s[1:].split("_")[0]))


def bind_data_grammar(data):
    """Give `grammar` a derivative operator in *this* process."""
    axes = _axes(data)
    fields = _fields(data)
    coords = [np.asarray(data[a], dtype=float) for a in axes]
    field_shape = np.shape(data[fields[0]])

    def deriv(field, axis, order):
        if order == 0:
            return field
        coord = coords[axis]
        return make_interp_spline(coord, field, k=5, axis=axis)(coord, nu=order)

    grid = []
    for axis, coord in enumerate(coords):
        shape = [1] * len(field_shape)
        shape[axis] = coord.size
        grid.append(coord.reshape(shape))
    if len(grid) == 1:
        grid.append(np.zeros_like(grid[0]))
    vector = tuple(np.asarray(data[f], dtype=float) for f in fields[:len(axes)])
    grammar.bind(tuple(range(len(axes))), deriv, vector=vector, coords=tuple(grid))
    return deriv


# ------------------------------------------------------------------ the gate
_FUNCS = {"id": lambda a: a, "sin": np.sin, "cos": np.cos,
          "exp": np.exp, "tanh": np.tanh}


def base_atoms(data):
    """A compact library of built-in mechanism atoms, for the held-out comparison."""
    axes = _axes(data)
    fields = _fields(data)
    coords = [np.asarray(data[a], dtype=float) for a in axes]
    field_shape = np.shape(data[fields[0]])
    grid = []
    for axis, coord in enumerate(coords):
        shape = [1] * len(field_shape)
        shape[axis] = coord.size
        grid.append(coord.reshape(shape))
    if len(grid) == 1:
        grid.append(np.zeros_like(grid[0]))

    def deriv(field, axis, order):
        return make_interp_spline(coords[axis], field, k=5, axis=axis)(coords[axis], nu=order)

    atoms = []
    for f in fields:
        arr = np.asarray(data[f], dtype=float)
        atoms.append((f"frc({f})", arr))
        for axis in range(len(axes)):
            atoms.append((f"prs({f},{axis})", deriv(arr, axis, 1)))
        atoms.append((f"dif({f})", sum(deriv(arr, a, 2) for a in range(len(axes)))))
        for func in ("id", "sin", "cos", "exp", "tanh"):
            atoms.append((f"rxn({f},{func})", _FUNCS[func](arr)))
    for i, f in enumerate(fields):
        for g in fields[i:]:
            atoms.append((f"cpl({f},{g})", np.asarray(data[f]) * np.asarray(data[g])))
    for func in ("sin", "cos", "exp", "tanh"):
        for k in (-2, -1, 1, 2):
            arg = k * grid[0]
            atoms.append((f"coord({func},{k})",
                          np.broadcast_to(_FUNCS[func](arg), field_shape)))
    return atoms


def mechanism_atoms(spec, data):
    function = getattr(grammar, spec["name"], None)
    if function is None:
        return []
    signature = list(spec["signature"])
    fields = _fields(data)
    axes = range(len(_axes(data)))
    field_like = {"phi", "psi", "f", "u", "v", "field", "scalar"}
    slots = [s for s in signature if s != "axis" and (s in field_like or s in fields)]
    # what to try for each non-axis formal: the measured fields, or a sign for a
    # scalar coefficient (the search fits the real value later)
    options = []
    for formal in signature:
        if formal == "axis":
            continue
        options.append([(f, np.asarray(data[f], dtype=float)) for f in fields]
                       if formal in slots else [(formal, 1.0), (formal, -1.0), (formal, 0.0)])
    field_shape = np.shape(data[fields[0]])
    out = []
    import itertools
    for choice in itertools.product(*options):
        for axis in (axes if "axis" in signature else [None]):
            args, labels = [], []
            k = 0
            for formal in signature:
                if formal == "axis":
                    args.append(axis)
                    labels.append(str(axis))
                else:
                    label, value = choice[k]
                    args.append(value)
                    labels.append(label if isinstance(label, str) and label in fields
                                  else f"{formal}={value:+.0f}")
                    k += 1
            try:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    array = np.asarray(function(*args), dtype=float)
                if not np.all(np.isfinite(array)):
                    continue
                if array.shape != field_shape:
                    array = np.broadcast_to(array, field_shape)
                out.append((f"{spec['name']}({','.join(labels)})", array))
            except Exception as error:
                # Keep the reason: "no usable atom" is otherwise impossible to
                # diagnose, and the usual causes are an axis passed as a grid or a
                # shape mismatch.
                global LAST_ATOM_ERROR
                LAST_ATOM_ERROR = f"{type(error).__name__}: {error}"
                continue
            if len(out) >= 60:
                return out
    return out
