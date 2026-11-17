"""Hypothesis generation: coverage proposals, history recombination and online revisions."""
import math
from collections import deque
from . import core as C, rotation, recombine
from .graph import signature
from .structural_rules import additive_deletions, reaction_block, graph_powers, GraphCores, core_proposals


def normalize_policy(raw, archives):
    """Translate explicit kept sets; invalid plans never trigger invented deletions."""
    import copy
    policy = copy.deepcopy(raw) if isinstance(raw, dict) else {}
    errors = []
    # Preserve the old `edits` interface as an alias. Never discard a valid old
    # plan merely because an optional new `revisions` section was also present.
    revisions = policy.get("revisions", {})
    if not isinstance(revisions, dict):
        errors.append(dict(section="revisions", error="expected target-field mapping"))
    revisions = revisions if isinstance(revisions, dict) else {}
    legacy = policy.get("edits", {})
    if not isinstance(legacy, dict):
        errors.append(dict(section="edits", error="expected target-field mapping"))
    if isinstance(legacy, dict):
        for field in archives:
            old_items = legacy.get(field, [])
            if isinstance(old_items, list):
                current = revisions.get(field, [])
                current = current if isinstance(current, list) else []
                revisions[field] = current+[item for item in old_items if item not in current]
    policy["revisions"] = revisions
    for key in ("tests", "revisions"):
        mapping = policy.get(key, {})
        if not isinstance(mapping, dict):
            continue
        for field, archive in archives.items():
            groups = mapping.get(field, [])
            if not isinstance(groups, list):
                continue
            for index, group in enumerate(groups):
                if not isinstance(group, dict):
                    continue
                if key == "revisions" and "parent_state" not in group and isinstance(group.get("base"), str) and group["base"] in archive.states:
                    group["parent_state"] = group["base"]
                if "keep" not in group:
                    continue
                parent_id = group.get("parent_state", "")
                parent = archive.states.get(parent_id) if isinstance(parent_id, str) else None
                keep = group["keep"]
                old = set(signature(parent["model"])) if parent else set()
                valid = isinstance(keep, list) and all(isinstance(mid, str) for mid in keep)
                if parent is None or not valid or not set(keep) <= old:
                    group["remove"] = ["__invalid_kept_set__"]
                    errors.append(dict(section=key, field=field, index=index, error="invalid kept set"))
                else:
                    group["remove"] = sorted(old-set(keep))
    policy["errors"] = errors
    return policy


def register_mechanisms(policy, registry, number):
    mapping, temporary, ids, events = {}, {}, [], []
    proposals = policy.get("new_nodes", [])
    if not isinstance(proposals, list):
        events.append(dict(rejected="new_nodes must be a list", proposal=proposals))
        proposals = []
    for index, item in enumerate(proposals):
        if index >= C.SETTINGS["new_roots_per_call"]:
            events.append(dict(index=index, rejected="declared new-node limit", proposal=item))
            continue
        try:
            mid, novel = registry.propose(item["expr"], number, item, temporary)
            alias = item.get("id", "")
            if isinstance(alias, str):
                mapping[alias] = mid
                temporary[alias] = registry.nodes[mid]["expr"]
            ids.append(mid)
            events.append(dict(index=index, node=mid, novel=novel, proposal=item))
        except (ValueError, TypeError, KeyError, SyntaxError, AttributeError) as error:
            events.append(dict(index=index, rejected=type(error).__name__, reason=str(error), proposal=item))
    return dict(mapping=mapping, ids=list(dict.fromkeys(ids)), events=events)


def history_candidates(archive, sampled, new_ids, probes):
    queues, info = recombine.queues(archive, sampled, new_ids, probes)
    return queues, info


def next_unseen(queues, archive):
    """Round-robin proposal streams; fitted/failed supports are both deduplicated."""
    while any(queues):
        for queue in queues:
            while queue:
                proposal = queue.popleft()
                support = tuple(proposal["support"])
                if support not in archive.attempted and support not in archive.keys:
                    yield proposal
                    break


def llm_revisions(archive, policy, shared_new, audit=None, number=1):
    """Compile every explicit old/new edit, with an outcome for every item.

    `base/new` supplies a complete equation expression and is evaluated exactly
    as written. A mechanism base is provenance, not an invented parent equation.
    It therefore uses the existing empty parent; a known state base is retained.
    """
    audit = audit if audit is not None else []
    mapping = policy.get("revisions", {})
    requests = mapping.get(archive.field, []) if isinstance(mapping, dict) else []
    if not isinstance(requests, list):
        audit.append(dict(status="rejected", reason="revisions must be a list", proposal=requests))
        return
    for index, item in enumerate(requests):
        entry = dict(index=index, proposal=item, status="rejected")
        audit.append(entry)
        try:
            if not isinstance(item, dict):
                raise ValueError("edit must be an object")
            parent_id = item.get("parent_state")
            expression = item.get("new", item.get("expr"))
            if expression is not None:
                if not isinstance(expression, str):
                    raise ValueError("complete replacement expression must be a string")
                base = item.get("base")
                if parent_id is None and isinstance(base, str) and base in archive.states:
                    parent_id = base
                if parent_id is None:
                    if base is not None and (not isinstance(base, str) or base not in archive.registry.nodes):
                        raise ValueError("unknown base mechanism or state")
                    parent_id = archive.field+"_S00000"
                if not isinstance(parent_id, str) or parent_id not in archive.states:
                    raise ValueError("unknown parent state")
                if parent_id not in archive.active_ids:
                    raise ValueError('Complete edits must use supplied working-set parents')
                temporary = {alias: archive.registry.nodes[node]["expr"] for alias, node in shared_new["mapping"].items()}
                mid, novel = archive.registry.propose(expression, number,
                    dict(kind="LLM_complete_equation_revision", source_base=base, parent_state=parent_id), temporary)
                ids = {mid}
                entry.update(format="complete_expression", registered_node=mid, novel=novel)
            else:
                if not isinstance(parent_id, str) or parent_id not in archive.states:
                    raise ValueError("parent_state required for kept-set edit")
                if parent_id not in archive.active_ids:
                    raise ValueError('LLM edits must use the supplied working-set parents')
                parent = archive.states[parent_id]
                add, remove = item.get("add", []), item.get("remove", [])
                if not isinstance(add, list) or not isinstance(remove, list) or not all(isinstance(mid, str) for mid in add+remove):
                    raise ValueError("add/remove must be mechanism-ID lists")
                old = set(signature(parent["model"]))
                if not set(remove) <= old:
                    raise ValueError("kept/removal set is not a subset of parent")
                ids = old-set(remove) | {shared_new["mapping"].get(mid, mid) for mid in add}
                if not ids <= archive.registry.graph.candidates.keys():
                    raise ValueError("unknown added mechanism or new-node alias")
                entry["format"] = "kept_set"
            if tuple(sorted(ids)) == signature(archive.states[parent_id]["model"]):
                raise ValueError("edit does not change the parent support")
            entry.update(status="scheduled", parent=parent_id, support=sorted(ids))
            yield dict(parent=parent_id, support=tuple(sorted(ids)), kind="online_LLM_revision",
                       reason=str(item.get("reason", "")), edit_audit_index=len(audit)-1)
        except (ValueError, TypeError, KeyError, SyntaxError, AttributeError) as error:
            entry.update(status="rejected", reason=str(error), error_type=type(error).__name__)


def build_tests(policy, archive, sampled, rng, number=1):
    """Execute the declared half quota, reserving second-sweep uncovered IDs."""
    required = rotation.quota(sampled)
    sample_set = rotation.eligible(archive,sampled)
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
    parents = archive.view()
    if not parents:
        raise RuntimeError('Missing working-set parents')
    fallback_drop = {}
    def fallback_support(parent, mid):
        old = set(signature(parent['model']))
        # If a parent already has eight terms, replace its smallest actual
        # RMS-contribution term. Never switch to an undisplayed empty parent.
        if mid not in old and len(old) >= C.SETTINGS['max_terms']:
            if parent['id'] not in fallback_drop:
                ranked = []
                import numpy as np
                for term in parent['model']['terms']:
                    names = sorted(term['params'])
                    feature = archive.data.feature(dict(expr=term['expr'], params=names),
                        [term['params'][name] for name in names])
                    ranked.append((float(np.linalg.norm(term['coefficient']*feature)), term['id']))
                fallback_drop[parent['id']] = min(ranked)[1]
            old.remove(fallback_drop[parent['id']])
        return tuple(sorted(old | {mid}))
    for i, mid in enumerate((m for m in forced if m not in chosen)):
        parent = parents[i % len(parents)]["id"]
        proposals.append(
            dict(
                parent=parent,
                support=fallback_support(archive.states[parent], mid),
                selected=[mid],
                source="coverage_revisit",
                reason="Second-sweep previously untested root",
            )
        )
        chosen.add(mid)
    missing = [m for m in missing if m not in chosen]
    for i, mid in enumerate(missing[: max(0, required - len(chosen))]):
        parent = parents[i % len(parents)]["id"]
        ids = fallback_support(archive.states[parent], mid)
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
