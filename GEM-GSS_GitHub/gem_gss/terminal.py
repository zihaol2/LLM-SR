"""Latest terminal rule, extended to complete multi-field systems and 2D.

Five original routes; simplify each target field using the unchanged scalar
fitter; retain one system anchor and its least-cost single-field deletion per
route; <=10 systems; one final LLM response with field-scoped artifact bans.
"""

import copy
import itertools
from pathlib import Path
import time
import numpy as np
import sympy as s
from . import core as C
from . import compression
from .graph import Registry
from .fitting import fit, final_equation
from .refinement import (
    reaction_block,
    graph_powers,
    reduce_model,
    TargetView,
    GraphCores,
    refine_core,
)
from .api import once
from sympy.printing.str import StrPrinter


def same_search(hashes):
    return all((C.sha(Path(path)) == digest for path, digest in hashes.items()))


def terminal_siblings(trace):
    trials = trace["trials"]
    if not trials:
        return []
    last = max((t["iteration"] for t in trials))
    return [t for t in trials if t["iteration"] == last and t.get("valid")]


class EquationPrinter(StrPrinter):

    def _print_Float(self, value):
        return format(float(value), ".9g")

    def _print_Function(self, value):
        if value.func.__name__ in ("u", "v"):
            return value.func.__name__
        return super()._print_Function(value)

    def _print_Derivative(self, value):
        suffix = ",".join((str(x) + (f"^{n}" if n != 1 else "") for x, n in value.variable_count))
        return "D[" + suffix + "](" + self._print(value.expr) + ")"


def prepare_routes(case, folder):
    """Same protected frontier selection, shared by scalar and coupled systems."""
    from . import structure

    cp = C.read(folder / "checkpoint.json")
    data = C.Data(case)
    choices = {}
    champions = {}
    simplest = {}
    support_cache = {}
    for field in case["fields"]:
        states = [st for st in cp["archives"][field]["states"] if st["model"]["terms"]]
        if not states:
            raise RuntimeError("No fitted nonempty field model")
        champion = min(
            states,
            key=lambda st: (
                st["model"]["score"],
                st["model"]["complexity"],
                st["model"]["fit_mse"],
                st["id"],
            ),
        )
        champions[field] = champion
        frontier = []
        error = float("inf")
        for st in sorted(
            states, key=lambda st: (st["model"]["complexity"], st["model"]["fit_mse"], st["id"])
        ):
            if st["model"]["fit_mse"] < error:
                frontier.append(st)
                error = st["model"]["fit_mse"]
        if champion not in frontier:
            frontier.append(champion)
        groups = {}
        for st in frontier:
            expr = sum(
                (structure.symbolic_term(t, data.language) for t in st["model"]["terms"]),
                s.Integer(0),
            )
            key = tuple(sorted(structure.support(expr, data.language)))
            support_cache[field, st["id"]] = key
            prior = groups.get(key)
            if prior is None or (st["model"]["complexity"], st["model"]["fit_mse"]) < (
                prior["model"]["complexity"],
                prior["model"]["fit_mse"],
            ):
                groups[key] = st
        choices[field] = list(groups.values())
        simplest[field] = groups[support_cache[field, champion["id"]]]

    def row(states):
        return dict(
            states={f: st["id"] for f, st in states.items()},
            models={f: st["model"] for f, st in states.items()},
            first_round=max(st["birth_round"] for st in states.values()),
            MSE=float(np.mean([st["model"]["fit_mse"] for st in states.values()])),
            relative_MSE=float(
                np.mean([st["model"]["fit_mse"] / data.scale[f] for f, st in states.items()])
            ),
            complexity=sum(st["model"]["complexity"] for st in states.values()),
            search_score=float(np.mean([st["model"]["score"] for st in states.values()])),
        )

    values = [
        row(dict(zip(case["fields"], states)))
        for states in itertools.product(*(choices[f] for f in case["fields"]))
    ]
    protected = row(champions)
    representative = row(simplest)
    if protected not in values:
        values.append(protected)
    selected, front = route_shortlist(
        values, protected=protected, simpler_equivalent=representative
    )
    C.dump(
        folder / "final_selection/provenance.json",
        dict(
            source="Protected full-archive numerical frontiers and symbolic representatives",
            candidate_states={label: r["states"] for label, r in zip("ABCDE", selected)},
            protected_search_winner=protected["states"],
            protected_winner_in_candidates=protected in selected,
            protected_simplest_equivalent=representative["states"],
            simplest_equivalent_in_candidates=representative in selected,
            system_frontier_size=len(front),
            reference_read=False,
        ),
    )


def simplify_field(data, reg, field, original):
    view = TargetView(data, field)
    before = view.fit_calls
    result = dict(
        original=copy.deepcopy(original), deletion=reduce_model(view, original, reg), polynomial=[]
    )
    block, nonpoly, vanishes = reaction_block(original, reg, field)
    if nonpoly:
        removed = {t["id"] for t in block}
        retained = [t for t in original["terms"] if t["id"] not in removed]
        powers = graph_powers(reg, field, include_constant=not vanishes)
        result["available_powers"] = [p for p, _ in powers]
        for size in range(1, len(powers) + 1):
            nodes = [reg.nodes[t["id"]] for t in retained] + [n for _, n in powers[:size]]
            print(
                data.case["id"],
                "TERMINAL",
                field,
                "powers",
                [p for p, _ in powers[:size]],
                "temporary terms",
                len(nodes),
                flush=True,
            )
            joint = fit(view, "u", nodes, reg, initial=original)
            if joint is not None:
                result["polynomial"].append(
                    dict(
                        exponents=[p for p, _ in powers[:size]],
                        joint=joint,
                        deletion=reduce_model(view, joint, reg),
                    )
                )
    original_fits = view.fit_calls - before
    _, anchor, _ = choose_anchor(result)
    trace = refine_core(data, reg, field, copy.deepcopy(anchor), GraphCores(reg))
    deletion_view = TargetView(data, field)
    deletion = reduce_model(deletion_view, trace["pruned"], reg)
    result["composite_traces"] = dict(
        existing_subexpression=trace, existing_subexpression_then_deletion=deletion
    )
    result["composite_fit_calls"] = trace["full_fits"] + deletion_view.fit_calls
    result["full_fits"] = original_fits + result["composite_fit_calls"]
    return result


def simplify_families(case, folder):
    cp = C.read(folder / "checkpoint.json")
    if len(cp["history"]) != 20 or cp.get("pending") is not None:
        raise RuntimeError("Freeze all 20 rounds first")
    if not (folder / "frozen_result.json").exists():
        raise RuntimeError("Search is not frozen")
    if not (folder / "final_selection/provenance.json").exists():
        prepare_routes(case, folder)
    mapping = C.read(folder / "final_selection/provenance.json")["candidate_states"]
    states = {f: {st["id"]: st for st in cp["archives"][f]["states"]} for f in case["fields"]}
    data = C.Data(case)
    dest = folder / "terminal_families"
    dest.mkdir(exist_ok=True)
    families = []
    for label, ids in mapping.items():
        path = dest / f"family_{label}.json"
        family = C.read(path) if path.exists() else dict(label=label, states=ids, fields={})
        if family["states"] != ids:
            raise RuntimeError("Cached terminal family differs from original route")
        for field in case["fields"]:
            if field in family["fields"]:
                continue
            print(case["id"], "TERMINAL FAMILY", label, field, flush=True)
            reg = Registry(case, cp["graph"])
            model = copy.deepcopy(states[field][ids[field]]["model"])
            family["fields"][field] = simplify_field(data, reg, field, model)
            family["full_fits"] = sum((x["full_fits"] for x in family["fields"].values()))
            C.dump(path, family)
        families.append(family)
    return (data, families)


def aggregate(models, data):
    return dict(
        MSE=float(np.mean([m["fit_mse"] for m in models.values()])),
        relative_MSE=float(np.mean([m["fit_mse"] / data.scale[f] for f, m in models.items()])),
        complexity=sum((m["complexity"] for m in models.values())),
        original_Score=float(np.mean([m["score"] for m in models.values()])),
    )


def choose_anchor(part):
    branches = [("deletion", part["deletion"])] + [
        ("powers_" + ",".join(map(str, p["exponents"])), p["deletion"]) for p in part["polynomial"]
    ]
    branches.extend(((name, trace) for name, trace in part.get("composite_traces", {}).items()))
    options = [("original", part["original"], None)] + [
        (name, t["pruned"], t)
        for name, t in branches
        if t["pruned"]["terms"] and len(t["pruned"]["terms"]) <= C.SETTINGS["max_terms"]
    ]
    return min(options, key=lambda v: (v[1]["score"], v[1]["complexity"], v[1]["fit_mse"]))


def shortlist(families, data):
    pool = []
    for family in families:
        chosen = {f: choose_anchor(part) for f, part in family["fields"].items()}
        models = {f: c[1] for f, c in chosen.items()}
        summary = aggregate(models, data)
        pool.append(
            dict(
                models=models,
                origins=[
                    dict(
                        family=family["label"],
                        role="best_original_Score",
                        branches={f: c[0] for f, c in chosen.items()},
                    )
                ],
            )
        )
        deletes = []
        for field, (_, model, trace) in chosen.items():
            if trace is None:
                continue
            for t in terminal_siblings(trace):
                if (
                    not t["model"]["terms"]
                    or len(t["model"]["terms"]) > 8
                    or t["complexity"] >= model["complexity"]
                ):
                    continue
                candidate = {**models, field: t["model"]}
                info = aggregate(candidate, data)
                deletes.append(
                    dict(
                        models=candidate,
                        summary=info,
                        origins=[
                            dict(
                                family=family["label"],
                                role="least_MSE_cost_deletion",
                                changed_field=field,
                                removed=t["removed_or_simplified"],
                                parent_relative_MSE=summary["relative_MSE"],
                                parent_complexity=summary["complexity"],
                                relative_MSE_ratio=info["relative_MSE"] / summary["relative_MSE"],
                                field_MSE_ratio=t["MSE"] / model["fit_mse"],
                                accepted_by_original_Score=t["score"] < model["score"],
                            )
                        ],
                    )
                )
        if deletes:
            pool.append(
                min(
                    deletes,
                    key=lambda c: (c["summary"]["relative_MSE"], c["summary"]["complexity"]),
                )
            )
    assert len(pool) <= 10
    unique = {}
    for row in pool:
        key = tuple(
            (
                (f, tuple(sorted(((t["expr"], tuple(sorted(t["params"]))) for t in m["terms"]))))
                for f, m in sorted(row["models"].items())
            )
        )
        if key not in unique:
            unique[key] = dict(models=row["models"], origins=list(row["origins"]))
        else:
            prior = unique[key]
            prior["origins"].extend(row["origins"])
            if (
                aggregate(row["models"], data)["relative_MSE"]
                < aggregate(prior["models"], data)["relative_MSE"]
            ):
                prior["models"] = row["models"]
    rows = sorted(
        unique.values(),
        key=lambda r: (
            aggregate(r["models"], data)["relative_MSE"],
            aggregate(r["models"], data)["complexity"],
        ),
    )
    for i, row in enumerate(rows, 1):
        row.update(id=f"K{i:02}", summary=aggregate(row["models"], data))
    assert 1 <= len(rows) <= 10
    return rows


def make_cards(data, rows):
    from .structure import symbolic_term

    printer = EquationPrinter()
    cards = []

    def num(x):
        return float(format(float(x), ".9g"))

    for row in rows:
        equations = {}
        for field, model in row["models"].items():
            expression = sum(
                (symbolic_term(t, data.language) for t in model["terms"]), s.Integer(0)
            )
            terms = []
            vectors = []
            for t in model["terms"]:
                v = data.feature(
                    {"expr": t["expr"], "params": sorted(t["params"])},
                    [t["params"][p] for p in sorted(t["params"])],
                )
                vectors.append(v)
                terms.append(
                    dict(
                        expression=t["expr"],
                        coefficient=num(t["coefficient"]),
                        parameters={p: num(v) for p, v in t["params"].items()},
                        contribution_norm_relative_to_target=num(
                            np.linalg.norm(t["coefficient"] * v)
                            / (np.linalg.norm(data.y[field]) + 1e-30)
                        ),
                    )
                )
            cosine = 0.0
            if len(vectors) > 1:
                A = np.column_stack(vectors)
                A /= np.maximum(np.linalg.norm(A, axis=0), 1e-30)
                g = np.abs(A.T @ A)
                np.fill_diagonal(g, 0)
                cosine = float(g.max())
            equations[field] = dict(
                equation=field + "_t = " + printer.doprint(expression),
                MSE=num(model["fit_mse"]),
                relative_MSE=num(model["fit_mse"] / data.scale[field]),
                complexity=model["complexity"],
                mechanism_count=len(model["terms"]),
                fit_optimizer_success=bool(model["optimizer"]["success"]),
                max_feature_cosine=num(cosine),
                fitted_terms=terms,
            )
        cards.append(
            dict(
                id=row["id"],
                **{k: num(v) if isinstance(v, float) else v for k, v in row["summary"].items()},
                relative_MSE_multiple_of_best=num(
                    row["summary"]["relative_MSE"] / rows[0]["summary"]["relative_MSE"]
                ),
                equations=equations,
                route_comparisons=row["origins"],
            )
        )
    return cards


def make_prompt(case, cards):
    return f"""Choose ONE governing PDE SYSTEM from the {len(cards)} frozen candidates below.\nThe 20-round search and all fits are finished. This is one terminal decision only.\nFields: {case['fields']}. Spatial coordinates IN ORDER: {case['coordinates']}.\nOne candidate contains ALL target equations. Select ONE system ID, not separate IDs per field.\nCandidate rule fixed in advance: simplify five original routes per target; each route contributes a complete best-original-Score system and its least-cost lower-complexity single-field deletion, then deduplicate; <=10 systems. Every final equation has <=8 mechanisms; the system may have more than 8 in total.\nALL existing graph-polynomial prefixes were considered for eligible reaction blocks before the final capacity check. No reference equation, previous LLM choice, correctness label or Jaccard is supplied.\nWeak prior: {case['prior']}\nObserved metadata: {C.json.dumps(case['metadata'], separators=(',', ':'))}\n\nExisting-core comparisons retain the original polynomial refinement and compare zero-at-origin wrappers and parameter-modulated products/quotients with existing simpler graph cores. All alternatives are jointly fitted before this decision.\n\nDecision priorities:\n1. Prefer LOW MSE strongly. For systems, relative_MSE is the MEAN of each equation's MSE divided by its own target mean square; inspect EACH equation too, rather than allowing one field to hide another's error. Candidates are ordered by this normalized error. A modest complexity saving alone does not justify orders of magnitude larger residual. Name and quantify the alternative when accepting worse fit.\n2. Seek governing mechanisms, not a residual interpolator. Among accurate models, decide whether added complexity explains a credible mechanism or merely absorbs numerical derivative errors. Compare ABSOLUTE/target-normalized residual levels as well as deletion ratios; a many-fold ratio between already tiny residuals need not establish a physical mechanism. Do not automatically choose minimum MSE.\n3. You may explicitly BAN an implausible numerical correction using its actual contribution, coefficient/parameter scales, functional form over the observed domain, and deletion evidence together. No universal coefficient or residual threshold is imposed. Tiny raw coefficients alone are insufficient because features may be huge. Conversely, a numerically useful correction need not be a governing mechanism.\n4. Be consistent: if rejecting a numerical artifact, list it in banned_terms with its TARGET FIELD and exact fitted_terms.expression, and select a provided system whose equation for that field contains none of those bans. A term can be valid in another equation, so bans are field-scoped. If rejecting multiple expression variants, list each. Leave bans empty if no rejection is supported. Do not call a term a meaningless artifact to exclude but retain it solely for lower MSE.\n5. Read FULL equations. Opposite signs on different powers or derivative orders do NOT prove cancellation. Use correlation/contribution evidence. Symbolically identical nested derivatives are merged for display where possible; different numerical paths can give unstable individual coefficients but a meaningful sum.\n6. Derivative notation: PD(E,alpha) follows the DECLARED coordinate order. In 1D PD(E,(i,))=d_x^i E. In 2D PD(E,(i,j))=d_x^i d_y^j E, so (2,0) and (0,2) are distinct directions. D[x^i,y^j](E) in displayed equations uses the same convention. Do not conflate x and y or erase anisotropy without evidence.\n7. Prefer coherent concise structures with plausible fitted coefficients while preserving the main quantitative accuracy. Near-simple numbers support interpretation but do not prove truth. Optimizer success is not an uncertainty interval. Parameters are independent across mechanisms AND target equations.\n\nMSE is the spatial-window residual on all observed times using estimated derivatives, not held-out predictive error. Raw system MSE is also provided but normalized error is the cross-field comparison. Complexity is summed across equations. Original Score is informative, not a veto. Do not modify, combine or refit candidates.\n{C.json.dumps(cards, separators=(',', ':'))}\nReturn JSON only: {{"selected":"K01","banned_terms":[{{"field":"{case['fields'][0]}","expression":"EXACT raw expression to exclude","reason":"Evidence for rejecting this correction"}}],"reason":"Quantitative comparison of fit, mechanism structure and complexity for all fields","comparison_to_best_MSE":"Name the best candidate and justify any accuracy sacrifice","coefficient_assessment":"Credibility and uncertainty","limitations":"What the data do not establish"}}"""


def valid_choice(policy, rows):
    chosen = next((r for r in rows if r["id"] == policy.get("selected")), None)
    bans = policy.get("banned_terms")
    if chosen is None or not isinstance(bans, list):
        return None
    for ban in bans:
        if not isinstance(ban, dict):
            return None
        field = ban.get("field")
        expr = ban.get("expression")
        if (
            field not in chosen["models"]
            or not isinstance(ban.get("reason"), str)
            or (not ban["reason"].strip())
        ):
            return None
        if not any((expr == t["expr"] for row in rows for t in row["models"][field]["terms"])):
            return None
        if any((expr == t["expr"] for t in chosen["models"][field]["terms"])):
            return None
    return chosen


def run(case, folder):
    dest = folder / "final_pipeline"
    dest.mkdir(exist_ok=True)
    if (dest / "result.json").exists():
        return C.read(dest / "result.json")
    frozen = {
        str(p.resolve()): C.sha(p)
        for p in [
            folder / "checkpoint.json",
            folder / "frozen_result.json",
            folder / "protocol.json",
        ]
    }
    protocol = dict(
        frozen_inputs=frozen,
        implementation_sha256=C.sha(Path(__file__)),
        reference_access=False,
        API_budget=1,
        candidate_limit=10,
        final_mechanism_limit_per_field=8,
        system_rule="Five complete system routes; one per-field best-Score anchor plus least normalized-MSE-cost single-field deletion",
        bans="Explicit and field-scoped in same final response",
        intermediate_polynomial_capacity_truncation=False,
    )
    if (dest / "protocol.json").exists():
        assert C.read(dest / "protocol.json") == protocol
    else:
        C.dump(dest / "protocol.json", protocol)
    data, families = simplify_families(case, folder)
    rows = shortlist(families, data)
    cards = make_cards(data, rows)
    prompt = make_prompt(case, cards)
    tokens = compression.tokens(prompt)
    if tokens > 20000:
        raise RuntimeError(f"Final input {tokens}>20000; no API sent")
    C.dump(dest / "candidates.json", rows)
    C.dump(dest / "llm_cards.json", cards)
    full_fits = sum((f["full_fits"] for f in families))
    C.dump(
        dest / "pre_dispatch.json",
        dict(
            input_tokens=tokens,
            candidate_count=len(rows),
            full_terminal_fits=full_fits,
            reference_access=False,
        ),
    )
    final = dest / "final_selection"
    final.mkdir(exist_ok=True)
    (final / "prompt.txt").write_text(prompt, encoding="utf8")
    assert same_search(frozen)
    policy, record = once(prompt, final / "api", 1)
    chosen = valid_choice(policy, rows)
    if chosen is None:
        C.dump(
            dest / "failure.json",
            dict(
                reason="Invalid system choice or field-scoped ban inconsistency; no automatic extra call",
                policy=policy,
                record=record,
            ),
        )
        raise RuntimeError("Final system choice invalid or inconsistent with bans")
    equations = {f: final_equation(data, f, m) for f, m in chosen["models"].items()}
    assert all((len(eq["terms"]) <= 8 for eq in equations.values())) and same_search(frozen)
    result = dict(
        case=case["id"],
        selected=chosen["id"],
        equations=equations,
        **chosen["summary"],
        origins=chosen["origins"],
        candidate_count=len(rows),
        policy=policy,
        usage=record.get("usage"),
        additional_API_calls=record["http_attempts"],
        full_terminal_fits=full_fits,
        reference_access=False,
        search_unchanged=True,
        ban_consistency_passed=True,
        frozen_unix=time.time(),
    )
    C.dump(dest / "result.json", result)
    C.dump(folder / "final_result.json", result)
    return result


def route_shortlist(values, limit=5, protected=None, simpler_equivalent=None):
    if limit < 1:
        raise ValueError("At least one candidate is required")

    def dominates(a, b):
        return (
            a["complexity"] <= b["complexity"]
            and a["relative_MSE"] <= b["relative_MSE"]
            and (a["complexity"] < b["complexity"] or a["relative_MSE"] < b["relative_MSE"])
        )

    frontier = sorted(
        [a for a in values if not any((dominates(b, a) for b in values))],
        key=lambda a: (a["complexity"], a["relative_MSE"]),
    )
    if protected is not None and protected not in values:
        raise ValueError("Protected search winner is absent from the archive rows")
    if simpler_equivalent is not None and simpler_equivalent not in values:
        raise ValueError("Simpler equivalent is absent from the archive rows")
    selected = []
    for row in [protected, simpler_equivalent, *frontier[:1], *frontier[-1:]]:
        if row is not None and row not in selected and (len(selected) < limit):
            selected.append(row)
    while len(selected) < min(limit, len(frontier)):
        chosen = [i for i, row in enumerate(frontier) if row in selected]
        remaining = [i for i, row in enumerate(frontier) if row not in selected]
        if not remaining:
            break
        index = max(
            remaining, key=lambda i: (min((abs(i - j) for j in chosen), default=len(frontier)), -i)
        )
        selected.append(frontier[index])
    if len(selected) < limit:
        rest = sorted(
            [a for a in values if a not in frontier],
            key=lambda a: (a["complexity"], a["relative_MSE"]),
        )
        selected.extend((a for a in rest if a not in selected))
        selected = selected[:limit]
    if protected is not None and protected not in selected:
        raise AssertionError("Search winner lost")
    return (sorted(selected, key=lambda a: (a["complexity"], a["relative_MSE"])), frontier)
