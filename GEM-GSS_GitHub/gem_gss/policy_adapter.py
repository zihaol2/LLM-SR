"""Translate explicit kept sets to the existing add/remove executor.

No numeric ranking, automatic deletion choice, new fit, or reference access.
Legacy remove-format outputs still work. Invalid kept sets remain invalid.
"""

import copy


def normalize_policy(policy, archives):
    policy = copy.deepcopy(policy) if isinstance(policy, dict) else {}
    errors = []
    for kind in ["tests", "edits"]:
        mapping = policy.get(kind, {})
        if not isinstance(mapping, dict):
            continue
        for field, archive in archives.items():
            groups = mapping.get(field, [])
            if not isinstance(groups, list):
                continue
            for number, group in enumerate(groups):
                if not isinstance(group, dict) or "keep" not in group:
                    continue
                parent = archive.states.get(group.get("parent_state", ""))
                keep = group["keep"]
                old = {t["id"] for t in parent["model"]["terms"]} if parent else set()
                valid = (
                    parent is not None
                    and isinstance(keep, list)
                    and all((isinstance(x, str) for x in keep))
                )
                valid = valid and set(keep) <= old
                if not valid:
                    group["remove"] = ["__invalid_kept_set__"]
                    errors.append(
                        dict(
                            kind=kind,
                            field=field,
                            group=number,
                            error="keep must be a subset of the parent support",
                        )
                    )
                else:
                    group["remove"] = sorted(old - set(keep))
    policy["_normalization_errors"] = errors
    return policy
