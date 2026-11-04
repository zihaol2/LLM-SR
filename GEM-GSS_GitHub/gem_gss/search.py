"""One graph-informed LLM selection, mandatory testing, then local search."""

from collections import Counter, defaultdict, deque
import math
import random
import time
import numpy as np
from . import core as C
from .graph import Registry, signature, feedback
from .fitting import fit, final_equation
from .memory import Archive, Probes, proposal_queues
from .api import once
from . import compression
from . import rotation
from . import recombine
from .policy_adapter import normalize_policy


def draw(registry, rng):
    return [
        mid
        for mid in sorted(registry.graph.candidates)
        if rng.random() < C.SETTINGS["sample_probability"]
    ]


def graph_packet(registry, sampled, archives, stats):
    selected = set(sampled)
    active = {t["id"] for a in archives.values() for s in a.view() for t in s["model"]["terms"]}
    incident, neighbors = ([], set())
    incoming, outgoing = (Counter(), Counter())
    for edge in registry.graph.edges.values():
        parent, child = (edge["parent"], edge["child"])
        outgoing[parent] += 1
        incoming[child] += 1
        if parent in selected or child in selected:
            incident.append([parent, child])
            neighbors.update((parent, child))
    ids = selected | neighbors | active
    nodes = {}
    for mid in sorted(ids):
        n = registry.nodes[mid]
        nodes[mid] = dict(
            expr=n["expr"],
            op=n["operator"],
            order=n["order"],
            depth=n["depth"],
            operations=n["operations"],
            params=n["params"],
            families=n.get("candidate_families", []),
            operands=n.get("operands", []),
            birth_stage=n.get("birth_stage", "unknown"),
            birth_round=getattr(registry, "birth", {}).get(mid, 0),
            indegree=incoming[mid],
            outdegree=outgoing[mid],
            candidate=bool(n["candidate"]),
        )
        if mid in selected:
            nodes[mid]["history"] = stats.get(mid, dict(sampled=0, tested=0))
    return dict(
        sampled_ids=sampled,
        historical_active_ids=sorted(active),
        nodes=nodes,
        dependency_edges=sorted(incident),
        edge_meaning="operand -> expression; operand positions are in each child record",
        topology_meaning="depth is expression construction depth, order is spatial derivative order; neither is a fit score",
        construction_depth_counts=dict(Counter((n["depth"] for n in registry.nodes.values()))),
        graph_node_count=len(registry.nodes),
        candidate_count=len(registry.graph.candidates),
    )


def make_verbose_prompt(case, registry, archives, sampled, stats, history):
    packet = graph_packet(registry, sampled, archives, stats)
    required = math.ceil(len(sampled) * C.SETTINGS["minimum_test_fraction"])
    summaries, diagnoses = ({}, {})
    for field, a in archives.items():
        summaries[field] = [
            dict(
                id=s["id"],
                terms=s["model"]["terms"],
                score=s["model"]["score"],
                MSE=s["model"]["fit_mse"],
                C=s["model"]["complexity"],
                visits=s["visits"],
                optimization=s["model"]["optimizer"],
            )
            for s in a.view()
        ]
        diagnoses[field] = [feedback(a.data, field, s, a) for s in a.view()]
    text = f'Guide anonymous observation-only PDE discovery. Read the sampled mechanisms AND their graph relationships.\n{case['prior']}\nMetadata: {C.json.dumps(case['metadata'], separators=(',', ':'))}\nNo reference equation, benchmark name, analytic solution, historical winner or external assessment is provided.\nThis run starts fresh from the cached initial graph. All observed times are used, without a validation split.\nObjective=(spatial-window MSE/target_scale+1e-14)*exp(0.1*C). Lower is better.\nFitting is variable projection: for every nonlinear parameter vector, solve all outer coefficients by linear least squares;\nthen optimize only nonlinear parameters. Existing terms inherit parent parameters; coarse alternatives are also checked.\nEach mechanism has a free outer coefficient and independent internal parameters. Max 8 terms, spatial order<=4,\nexpression<=40 operations and <=600 characters. Sinc and Hessian determinant excluded by the inherited grammar.\nGrammar: fields {case['fields']}, coordinates {case['coordinates']}, t; parameters a,b,k,p0,p1,...;\narithmetic +,-,*,/,**; Exp,Log,Sin,Cos,Tan,ArcSin,ArcCos,ArcTan,Sinh,Cosh,Tanh,ArcSinh,ArcCosh,ArcTanh,Sqrt,CubeRoot;\nPD(E,alpha) is a spatial partial derivative with multi-index alpha. CubeRoot is real cube root.\nMechanism meanings must be inferred from expressions; do not invent known physical labels.\nGraph dependencies are construction relationships, not evidence that nearby nodes form a good equation.\nComplete sampled-root and one-hop graph packet: {C.json.dumps(packet, separators=(',', ':'))}\nHistorical partial solutions: {C.json.dumps(summaries, separators=(',', ':'))}\nPer-route residual, contribution and edit feedback: {C.json.dumps(diagnoses, separators=(',', ':'))}\nRecent completed rounds: {C.json.dumps(history[-3:], separators=(',', ':'))}\nThere are {len(sampled)} sampled roots. For EACH target field independently, select EXACTLY {required} DIFFERENT sampled root IDs for immediate fitting.\nAll selected sampled mechanisms WILL be jointly fitted, with no second residual-ranking gate.\nChoose by mathematical meaning, derivative role, construction families, topology, relation to existing solutions,\nresidual evidence and whether a candidate has actually been tested. Consider different hypotheses and complementary terms.\nExpress selections compactly as groups with a shared parent and kept-term list. Do not write per-node reasons. For joint=false, EACH candidate is separately\nadded to the explicit kept-term set; this makes one fit per candidate. For joint=true, candidates are added together\nin one fit on top of that kept-term set. Use joint=true only for a reasoned complementary block, not to inflate coverage. A group must stay <=8 terms.\nChoose parent_state from displayed states. The keep list must be a subset of that parent support. For joint=false, len(keep plus EACH candidate) must be <=8; a full 8-term parent cannot keep all terms while adding a new one. For joint=true, len(keep union candidates) must be <=8. Return the kept set, not a verbal deletion instruction.\nCurrent low scores of partial routes do not prove they cannot lead to a better solution.\nAlso suggest up to 8 local multi-term edits TOTAL across fields, shared evenly across the declared fields,\nand up to 4 optional new expressions TOTAL. The separate 32-fit local budget is TOTAL per round,\nshared evenly across target fields. For two fields, give at most 4 edits per field and each gets at most 16 local fits.\nnew_nodes may use earlier new1/new2 IDs. Reuse existing mechanisms when possible.\nReturn a JSON object only. Use the actual fields {case['fields']}:\n{{"tests":{{"u":[{{"parent_state":"u_S00000","candidates":["D000001","D000002"],"keep":[],"joint":false}}]}},\n"focus_states":{{"u":["u_S00000"]}},\n"edits":{{"u":[{{"parent_state":"u_S00000","keep":[],"add":["D000003"]}}]}},\n"new_nodes":[{{"id":"new1","expr":"...","field":"u","parent_state":"u_S00000","reason":"..."}}],\n"reason":"One short overall plan; no per-node explanations"}}\nDo not omit sampled IDs from consideration merely because their initial residual correlation might be low.'
    return (text, packet)


def make_prompt(case, registry, archives, sampled, stats, history, number=1, preview=None):
    raw, packet = make_verbose_prompt(case, registry, archives, sampled, stats, history)
    extra = {
        mid
        for info in (preview or {}).values()
        for example in info.get("examples", [])
        for mid in example["support"]
    }
    for mid in sorted(extra - packet["nodes"].keys()):
        n = registry.nodes[mid]
        packet["nodes"][mid] = dict(
            expr=n["expr"],
            order=n["order"],
            depth=n["depth"],
            candidate=bool(n["candidate"]),
            operands=n["operands"],
        )
    if extra:
        p0 = raw.index(compression.PREFIX) + len(compression.PREFIX)
        p1 = raw.index("\n" + compression.HIST, p0)
        raw = raw[:p0] + C.json.dumps(packet, separators=(",", ":")) + raw[p1:]
    forced = {f: rotation.forced_for(a, sampled, number) for f, a in archives.items()}
    appendix = (
        "\nThis experiment uses ten frozen random groups, repeated twice. Current round: "
        + str(number)
        + ".\nOn the second sweep the following sampled IDs MUST be tested for each field, because they have never entered fitting:\n"
        + C.json.dumps(forced, separators=(",", ":"))
        + "\nInclude these IDs in tests, then fill remaining quota. Coverage is an attempted fit, not proof of numeric validity.\nHistorical recombination uses the entire tested archive, not only the states displayed above.\nThe 32 full local fits are split into 12 original local fits (including your edits) and 20 historical recombinations across four round-robin proposal types, including group crossover. Terminal simplification happens after all 20 rounds, BEFORE the one final LLM choice.\nFresh mechanisms may combine with historical mechanisms after discarding their incorrect old context.\nJoint feature probes use fixed internal parameters only to propose supports. Every chosen support is fully refitted.\nHistorical joint-probe previews: "
        + C.json.dumps(preview or {}, separators=(",", ":"))
    )
    raw += appendix
    raw += "\nParent capacities (state: [term_count, free_slots]): " + C.json.dumps(
        {
            f: {
                st["id"]: [
                    len(st["model"]["terms"]),
                    C.SETTINGS["max_terms"] - len(st["model"]["terms"]),
                ]
                for st in a.view()
            }
            for f, a in archives.items()
        },
        separators=(",", ":"),
    )
    text, audit = compression.encode(raw, packet)
    packet["compression_audit"] = audit
    return (text, packet)


def build_tests(policy, archive, sampled, rng, number=1):
    """Execute the declared half quota, reserving second-sweep uncovered IDs."""
    required = math.ceil(len(sampled) * C.SETTINGS["minimum_test_fraction"])
    sample_set = set(sampled)
    chosen, proposals, repairs = (set(), [], [])
    forced = rotation.forced_for(archive, sampled, number)
    shown = {s["id"] for s in archive.view()}
    mapping = policy.get("tests", {})
    groups = mapping.get(archive.field, []) if isinstance(mapping, dict) else []
    groups = groups if isinstance(groups, list) else []

    def support(parent_id, candidates, remove):
        st = archive.states[parent_id]
        old = set(signature(st["model"]))
        if not set(remove) <= old:
            return None
        ids = old - set(remove) | set(candidates)
        return tuple(sorted(ids)) if len(ids) <= C.SETTINGS["max_terms"] else None

    for group in groups:
        if not isinstance(group, dict):
            continue
        parent = group.get("parent_state")
        if not isinstance(parent, str) or parent not in shown:
            repairs.append("invalid_parent")
            continue
        candidates = group.get("candidates", [])
        remove = group.get("remove", [])
        if not isinstance(candidates, list) or not isinstance(remove, list):
            repairs.append("invalid_list")
            continue
        candidates = list(
            dict.fromkeys(
                (
                    x
                    for x in candidates
                    if isinstance(x, str) and x in sample_set and (x not in chosen)
                )
            )
        )
        slots = max(0, required - len(forced) - len(chosen - set(forced)))
        if len([x for x in candidates if x not in forced]) > slots:
            repairs.append("over_quota_selection")
        candidates = [x for x in candidates if x in forced] + [
            x for x in candidates if x not in forced
        ][:slots]
        remove = [x for x in remove if isinstance(x, str)]
        if not candidates:
            continue
        batches = [candidates] if group.get("joint") is True else [[x] for x in candidates]
        for batch in batches:
            ids = support(parent, batch, remove)
            if ids is None:
                repairs.append("oversized_support_or_invalid_removal")
                continue
            proposals.append(
                dict(
                    parent=parent,
                    support=ids,
                    selected=batch,
                    source="LLM",
                    reason=str(group.get("reason", "")),
                )
            )
            chosen.update(batch)
    missing = list(sample_set - chosen)
    rng.shuffle(missing)
    parents = [s for s in archive.view() if len(signature(s["model"])) < C.SETTINGS["max_terms"]]
    if not parents:
        parents = [archive.states[archive.field + "_S00000"]]
    for i, mid in enumerate((m for m in forced if m not in chosen)):
        parent = parents[i % len(parents)]["id"]
        proposals.append(
            dict(
                parent=parent,
                support=support(parent, [mid], []),
                selected=[mid],
                source="coverage_revisit",
                reason="Second-sweep previously untested root",
            )
        )
        chosen.add(mid)
    missing = [m for m in missing if m not in chosen]
    for i, mid in enumerate(missing[: max(0, required - len(chosen))]):
        parent = parents[i % len(parents)]["id"]
        ids = support(parent, [mid], [])
        proposals.append(
            dict(
                parent=parent,
                support=ids,
                selected=[mid],
                source="quota_repair",
                reason="LLM plan underfilled or invalid",
            )
        )
        chosen.add(mid)
    assert len(chosen) == required
    return (
        proposals,
        dict(
            required=required,
            selected=sorted(chosen),
            LLM_selected=sorted(
                {m for p in proposals if p["source"] == "LLM" for m in p["selected"]}
            ),
            repair_selected=sorted(
                {m for p in proposals if p["source"] == "quota_repair" for m in p["selected"]}
            ),
            forced_untested=forced,
            validation_repairs=repairs,
        ),
    )


def evaluate(archive, proposal, number, kind):
    parent = archive.states[proposal["parent"]]
    support = tuple(proposal["support"])
    started = time.time()
    initial = parent["model"]
    if proposal.get("donor_state"):
        donor = archive.states[proposal["donor_state"]]
        inherited = {t["id"]: t for t in donor["model"]["terms"]}
        inherited.update({t["id"]: t for t in parent["model"]["terms"]})
        initial = dict(terms=list(inherited.values()))
    model = fit(
        archive.data,
        archive.field,
        [archive.registry.nodes[m] for m in support],
        archive.registry,
        initial=initial,
        full=True,
    )
    edit = dict(
        operation=kind,
        added=sorted(set(support) - set(signature(parent["model"]))),
        removed=sorted(set(signature(parent["model"])) - set(support)),
    )
    if proposal.get("donor_state"):
        edit["donor_state"] = proposal["donor_state"]
    archive.attempted.add(support)
    archive.states[parent["id"]]["visits"] += 1
    sid = archive.add(model, parent["id"], edit, number)
    return dict(
        field=archive.field,
        parent=parent["id"],
        state=sid,
        support=list(support),
        kind=kind,
        selected=proposal.get("selected", []),
        source=proposal.get("source", "local"),
        valid=model is not None,
        score=model["score"] if model else None,
        optimizer=model["optimizer"] if model else None,
        delta_score=model["score"] - parent["model"]["score"] if model else None,
        seconds=time.time() - started,
    )


def local_search(archive, sampled, policy, registry, number, rng, probes, shared_new):
    fields = registry.case.get("fields", [archive.field])
    index = fields.index(archive.field)
    count = len(fields)
    half = (
        0
        if C.SETTINGS["local_fits_per_round"] == 0
        else C.SETTINGS["local_budget_split"]["original"]
    )
    fit_budget = half // count + int(index < half % count)
    edit_budget = 8 // count + int(index < 8 % count)
    shown = {s["id"] for s in archive.view()}
    fm = policy.get("focus_states", {})
    focus = fm.get(archive.field, []) if isinstance(fm, dict) else []
    focus = (
        [s for s in focus if isinstance(s, str) and s in shown] if isinstance(focus, list) else []
    )
    parents = archive.parents(focus)
    pool = set(sampled) | set(shared_new["ids"])
    memory = archive.view()
    for st in parents + memory:
        pool.update(signature(st["model"]))
    priority = deque()
    edits_map = policy.get("edits", {})
    edits = edits_map.get(archive.field, []) if isinstance(edits_map, dict) else []
    edits = edits[:edit_budget] if isinstance(edits, list) else []
    edits += [
        dict(parent_state=archive.best()["id"], add=[m], remove=[]) for m in shared_new["ids"]
    ]
    for e in edits:
        if not isinstance(e, dict):
            continue
        parent = e.get("parent_state")
        if not isinstance(parent, str) or parent not in archive.states:
            continue
        if not isinstance(e.get("add", []), list) or not isinstance(e.get("remove", []), list):
            continue
        add = {shared_new["mapping"].get(m, m) for m in e.get("add", []) if isinstance(m, str)}
        remove = {m for m in e.get("remove", []) if isinstance(m, str)}
        old = set(signature(archive.states[parent]["model"]))
        support = old - remove | add
        if (
            remove <= old
            and support <= registry.graph.candidates.keys()
            and (len(support) <= 8)
            and (support != old)
        ):
            pool.update(support)
            priority.append(dict(parent=parent, support=tuple(sorted(support))))
    queues = proposal_queues(archive, parents, memory, pool, rng, probes)
    rows, tick = ([], 0)
    while (priority or any(queues)) and len(rows) < fit_budget:
        if priority:
            p = priority.popleft()
            kind = "LLM_local_edit"
        else:
            queue = queues[tick % len(queues)]
            tick += 1
            if not queue:
                continue
            parent, support, edit = queue.popleft()
            p = dict(parent=parent, support=support)
            kind = edit["operation"]
        if tuple(p["support"]) in archive.attempted or tuple(p["support"]) in archive.keys:
            continue
        rows.append(evaluate(archive, p, number, kind))
    return rows


def historical_search(archive, sampled, registry, number, probes, shared_new):
    queues, info = recombine.queues(archive, sampled, shared_new["ids"], probes)
    fields = registry.case["fields"]
    index = fields.index(archive.field)
    half = (
        0
        if C.SETTINGS["local_fits_per_round"] == 0
        else C.SETTINGS["local_budget_split"]["whole_history_recombination"]
    )
    budget = half // len(fields) + int(index < half % len(fields))
    rows = []
    tick = 0
    while any(queues) and len(rows) < budget:
        queue = queues[tick % len(queues)]
        tick += 1
        if not queue:
            continue
        p = queue.popleft()
        if p["support"] in archive.attempted or p["support"] in archive.keys:
            continue
        row = evaluate(archive, p, number, "historical_" + p["origin"])
        row["joint_probe_relative_MSE"] = p["probe_relative_MSE"]
        row["history_anchor"] = p["history_anchor"]
        if p.get("donor_state"):
            row["donor_state"] = p["donor_state"]
            row["crossover_capped"] = p.get("crossover_capped", False)
        rows.append(row)
    return (rows, info)


def new_nodes(policy, registry, number):
    proposals = policy.get("new_nodes", [])
    mapping, temporary, ids, events = ({}, {}, [], [])
    for p in (proposals if isinstance(proposals, list) else [])[:4]:
        try:
            mid, novel = registry.propose(p["expr"], number, p, temporary)
            alias = p.get("id", "")
            if isinstance(alias, str):
                mapping[alias] = mid
                temporary[alias] = registry.nodes[mid]["expr"]
            ids.append(mid)
            events.append(dict(node=mid, novel=novel, proposal=p))
        except (ValueError, TypeError, KeyError, SyntaxError, AttributeError) as err:
            events.append(dict(rejected=type(err).__name__))
    return dict(mapping=mapping, ids=list(dict.fromkeys(ids)), events=events)


def run(case, out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "frozen_result.json").exists():
        return C.read(out / "frozen_result.json")
    cp = C.read(out / "checkpoint.json") if (out / "checkpoint.json").exists() else None
    started = cp["started_unix"] if cp else time.time()
    data = C.Data(case)
    registry = Registry(case, cp["graph"] if cp else None)
    archives = {f: Archive(data, f, registry) for f in case["fields"]}
    rng, probes = (random.Random(C.SETTINGS["seed"]), Probes())
    history, stats, pending = ([], {}, None)
    groups = cp["rotation_groups"] if cp else rotation.partition(registry.graph.candidates, rng)
    if cp:
        history, stats, pending = (cp["history"], cp["stats"], cp["pending"])
        for f, a in archives.items():
            saved = cp["archives"][f]
            a.states = {s["id"]: s for s in saved["states"]}
            a.keys = {signature(s["model"]): s["id"] for s in a.states.values()}
            a.edges = saved["edges"]
            a.attempted = {tuple(ids) for ids in saved["attempted"]}

        def tuples(x):
            return tuple((tuples(v) for v in x)) if isinstance(x, list) else x

        rng.setstate(tuples(cp["rng_state"]))
        for k, v in cp["counters"].items():
            setattr(data, k, v)
    else:
        C.dump(out / "initial_graph.json", registry.export())

    def save():
        C.dump(
            out / "checkpoint.json",
            dict(
                started_unix=started,
                graph=registry.export(),
                history=history,
                stats=stats,
                pending=pending,
                rng_state=rng.getstate(),
                rotation_groups=groups,
                counters={k: getattr(data, k) for k in ("fit_calls", "fit_nfev", "feature_calls")},
                archives={
                    f: dict(
                        states=list(a.states.values()), edges=a.edges, attempted=sorted(a.attempted)
                    )
                    for f, a in archives.items()
                },
            ),
        )

    for number in range(len(history) + 1, C.SETTINGS["rounds"] + 1):
        begin = time.time()
        if pending is None:
            sampled = rotation.group_for(groups, number)
            for mid in sampled:
                st = stats.setdefault(mid, dict(sampled=0, tested=0))
                st["sampled"] += 1
                st["last_sample_round"] = number
            preview = {}
            for f, a in archives.items():
                qs, info = recombine.queues(a, sampled, [], probes)
                preview[f] = dict(
                    history_mechanisms=info["history_mechanisms"],
                    examples=[
                        dict(support=list(p["support"]), relative_probe_MSE=p["probe_relative_MSE"])
                        for q in qs
                        for p in list(q)[:2]
                    ],
                )
            text, packet = make_prompt(
                case, registry, archives, sampled, stats, history, number, preview
            )
            pending = dict(round=number, sampled=sampled, prompt=text)
            C.dump(out / "inputs" / f"{number:02d}_graph.json", packet)
            save()
        else:
            assert pending["round"] == number
            sampled, text = (pending["sampled"], pending["prompt"])
        print(
            case["id"],
            "ROUND",
            number,
            "sampled",
            len(sampled),
            "minimum tested",
            math.ceil(len(sampled) * C.SETTINGS["minimum_test_fraction"]),
            "prompt chars",
            len(text),
            flush=True,
        )
        raw_policy, record = once(text, out / "api", number)
        policy = normalize_policy(raw_policy, archives)
        mandatory, selection, local = ({}, {}, {})
        shared_new = new_nodes(policy, registry, number)
        recombination_info = {}
        for f, a in archives.items():
            proposals, selection[f] = build_tests(policy, a, sampled, rng, number)
            mandatory[f] = []
            for i, p in enumerate(proposals):
                mandatory[f].append(evaluate(a, p, number, "mandatory_" + p["source"]))
                if (i + 1) % 16 == 0:
                    print(
                        case["id"],
                        number,
                        f,
                        "mandatory fits",
                        i + 1,
                        "/",
                        len(proposals),
                        "best",
                        a.best()["model"]["score"],
                        flush=True,
                    )
            tested = {m for row in mandatory[f] for m in row["support"]} & set(sampled)
            assert set(selection[f]["selected"]) <= tested
            assert len(tested) >= selection[f]["required"]
            selection[f]["actually_tested_sampled"] = sorted(tested)
            local[f] = local_search(a, sampled, policy, registry, number, rng, probes, shared_new)
            extra, recombination_info[f] = historical_search(
                a, sampled, registry, number, probes, shared_new
            )
            local[f].extend(extra)
        rows = [r for group in (mandatory, local) for batch in group.values() for r in batch]
        for mid in {m for row in rows for m in row["support"]}:
            st = stats.setdefault(mid, dict(sampled=0, tested=0))
            st["tested"] += 1
            st["last_test_round"] = number
        best = {
            f: dict(
                state=a.best()["id"],
                MSE=a.best()["model"]["fit_mse"],
                score=a.best()["model"]["score"],
                complexity=a.best()["model"]["complexity"],
            )
            for f, a in archives.items()
        }
        event = dict(
            round=number,
            sampled_count=len(sampled),
            best=best,
            mandatory_fits=sum(map(len, mandatory.values())),
            local_fits=sum(map(len, local.values())),
            sampled_tested={f: len(s["actually_tested_sampled"]) for f, s in selection.items()},
            required_sampled_tests=math.ceil(len(sampled) * C.SETTINGS["minimum_test_fraction"]),
            LLM_selected={f: len(s["LLM_selected"]) for f, s in selection.items()},
            repair_selected={f: len(s["repair_selected"]) for f, s in selection.items()},
            valid_fits=sum((r["valid"] for r in rows)),
            API_status=record["status"],
            graph_candidates=len(registry.graph.candidates),
            new_mechanisms=shared_new["events"],
            historical_recombination=recombination_info,
            reason=policy.get("reason", ""),
            policy_normalization_errors=policy.get("_normalization_errors", []),
            seconds=time.time() - begin,
        )
        C.dump(
            out / "rounds" / f"{number:02d}.json",
            dict(
                event=event, sampled=sampled, selection=selection, mandatory=mandatory, local=local
            ),
        )
        history.append(event)
        pending = None
        save()
        print(
            case["id"],
            "COMPLETE",
            number,
            "best",
            best,
            "coverage",
            event["sampled_tested"],
            "required",
            event["required_sampled_tests"],
            "repairs",
            event["repair_selected"],
            flush=True,
        )
    records = [C.read(p) for p in sorted((out / "api").glob("*_record.json"))]
    result = dict(
        method="GEM-GSS: graph-guided sampled tests with variable projection",
        anonymous_id=case["id"],
        completed_rounds=len(history),
        search_API_calls=sum((r["http_attempts"] for r in records)),
        accepted_LLM_decisions=sum((r["status"] == "accepted" for r in records)),
        new_initialization_calls=0,
        cached_initialization_calls=1,
        equations={f: final_equation(data, f, a.best()["model"]) for f, a in archives.items()},
        MSE=float(np.mean([a.best()["model"]["fit_mse"] for a in archives.values()])),
        all_rounds_half_sample_tested=all(
            (
                all((v >= e["required_sampled_tests"] for v in e["sampled_tested"].values()))
                for e in history
            )
        ),
        fit_calls=data.fit_calls,
        loss_evaluations=data.fit_nfev,
        seconds=time.time() - started,
        reference_access=False,
        frozen_unix=time.time(),
    )
    initial_roots = set((m for group in groups for m in group))
    result["initial_root_attempt_coverage"] = {
        f: len(initial_roots & {m for support in a.attempted for m in support}) / len(initial_roots)
        for f, a in archives.items()
    }
    if C.SETTINGS["rounds"] == 20 and C.SETTINGS["minimum_test_fraction"] == 0.5:
        assert all((v == 1.0 for v in result["initial_root_attempt_coverage"].values()))
    C.dump(out / "final_graph.json", registry.export())
    C.dump(out / "frozen_result.json", result)
    return result
