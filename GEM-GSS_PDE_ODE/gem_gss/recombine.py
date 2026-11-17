"""Feature-level joint proposals from the whole tested archive, not the 6 displayed states."""

from collections import deque
import numpy as np
from . import core as C
from .graph import signature


def queues(archive, sampled, new_ids, probes):
    reg = archive.registry
    data = archive.data
    states = archive.searchable_states()
    historical = {m for st in states for m in signature(st["model"])}
    fresh = set(sampled) | set(new_ids)
    vectors = {}
    for mid in sorted(historical | fresh):
        v = probes.get(data, reg.nodes[mid])
        if v is not None and np.isfinite(v).all() and (np.linalg.norm(v) > 1e-15):
            vectors[mid] = v
    old = sorted(historical & vectors.keys())
    new = sorted(fresh & vectors.keys())
    if not old or not new:
        return (
            [],
            dict(
                history_mechanisms=len(historical),
                history_states=len(states),
                probe_valid=len(vectors),
            ),
        )
    allids = sorted(vectors)
    index = {m: i for i, m in enumerate(allids)}
    X = np.column_stack([vectors[m] for m in allids])
    xn = X[:, [index[m] for m in new]]
    y = data.y[archive.field]
    y = y / max(float(np.linalg.norm(y)), 1e-30)
    dots = X.T @ y
    cross = xn.T @ X
    newdots = dots[[index[m] for m in new]]
    pair = []
    grow = []
    exchange = []
    oldcol = np.array([index[m] for m in old])
    g = cross[:, oldcol]
    a = newdots[:, None]
    b = dots[oldcol][None, :]
    den = 1 - g * g
    with np.errstate(all="ignore"):
        errors = 1 - (a * a - 2 * g * a * b + b * b) / np.maximum(den, 1e-14)
    errors = np.where(den > 1e-12, np.maximum(errors, 0), np.inf)
    parents = {}
    for st in sorted(states, key=lambda st: st["model"]["score"]):
        for mid in signature(st["model"]):
            parents.setdefault(mid, st["id"])
    for j, oldid in enumerate(old):
        for i in np.argsort(errors[:, j]):
            support = tuple(sorted({new[i], oldid}))
            if not np.isfinite(errors[i, j]):
                break
            if len(support) != 2 or support in archive.attempted or support in archive.keys:
                continue
            pair.append(
                (
                    float(errors[i, j]),
                    dict(
                        parent=parents[oldid],
                        support=support,
                        origin="fresh_history_pair",
                        history_anchor=oldid,
                        probe_relative_MSE=float(errors[i, j]),
                    ),
                )
            )
            break
    gram_cache = {}

    def extend(st, used, target):
        if not used or len(used) >= C.SETTINGS["max_terms"]:
            return
        columns = [index[m] for m in used]
        base = X[:, columns]
        key = tuple(used)
        if key not in gram_cache:
            G = base.T @ base
            pinv = np.linalg.pinv(G, rcond=1e-10)
            beta = pinv @ dots[columns]
            gram_cache[key] = (pinv, beta, max(0.0, 1 - float(dots[columns] @ beta)))
        pinv, beta, err = gram_cache[key]
        gn = cross[:, columns]
        corr = newdots - gn @ beta
        variance = 1 - np.einsum("ij,jk,ik->i", gn, pinv, gn)
        score = np.full(len(new), np.inf)
        valid = variance > 1e-12
        score[valid] = np.maximum(0.0, err - corr[valid] ** 2 / variance[valid])
        for i in np.argsort(score):
            mid = new[i]
            support = tuple(sorted(set(used) | {mid}))
            if (
                mid in used
                or not np.isfinite(score[i])
                or support in archive.attempted
                or (support in archive.keys)
            ):
                continue
            target.append(
                (
                    int(st["visits"] > 0),
                    float(score[i]),
                    dict(
                        parent=st["id"],
                        support=support,
                        origin="history_expand" if target is grow else "history_remove_then_add",
                        history_anchor=st["id"],
                        probe_relative_MSE=float(score[i]),
                    ),
                )
            )
            break

    for st in states:
        used = tuple(sorted(signature(st["model"])))
        if not used or not set(used) <= vectors.keys():
            continue
        extend(st, used, grow)
        if len(used) > 1:
            cols = [index[m] for m in used]
            beta = np.linalg.lstsq(X[:, cols], y, rcond=1e-09)[0]
            drop = used[int(np.argmin(np.abs(beta)))]
            extend(st, tuple((m for m in used if m != drop)), exchange)
    pair.sort(key=lambda z: z[0])
    grow.sort(key=lambda z: z[:2])
    exchange.sort(key=lambda z: z[:2])

    def unique(items):
        seen = set()
        anchors = set()
        out = []
        for item in items:
            p = item[-1]
            support = p["support"]
            anchor = p["history_anchor"]
            if support in seen or anchor in anchors:
                continue
            seen.add(support)
            anchors.add(anchor)
            out.append(p)
        return deque(out)

    grouped = crossover_proposals(archive, X, index, dots, gram_cache)
    return (
        [unique(pair), unique(grow), unique(exchange), unique(grouped)],
        dict(
            history_mechanisms=len(historical),
            history_states=len(states),
            probe_valid=len(vectors),
            fresh_mechanisms=len(new),
            pair_candidates=len(pair),
            grow_candidates=len(grow),
            exchange_candidates=len(exchange),
            group_candidates=len(grouped),
            fixed_internal_probe="all ones, same probes as baseline; proposal only, never a node exclusion rule",
        ),
    )


def crossover_proposals(archive, X, index, dots, gram_cache):
    states = list(archive.states.values())
    out = []
    seen = set()
    limit = C.SETTINGS["max_terms"]
    grams = {}
    for recipient in archive.view():
        S = tuple(sorted(signature(recipient["model"])))
        if len(S) < 2 or not set(S) <= index.keys():
            continue
        scols = [index[m] for m in S]
        cross = X[:, scols].T @ X
        where = {m: i for i, m in enumerate(S)}
        GS = cross[:, scols]
        for donor in states:
            H = tuple(sorted(signature(donor["model"])))
            if len(H) < 2 or len(H) >= limit or (not set(H) <= index.keys()):
                continue
            extras = tuple((m for m in S if m not in H))
            if not extras or set(H) <= set(S):
                continue
            hcols = [index[m] for m in H]
            if H not in grams:
                hbase = X[:, hcols]
                grams[H] = hbase.T @ hbase
            GH = grams[H]
            positions = [where[m] for m in extras]
            GE = GS[np.ix_(positions, positions)]
            EH = cross[np.ix_(positions, hcols)]
            G = np.block([[GH, EH.T], [EH, GE]])
            ids = H + extras
            b = dots[[index[m] for m in ids]]
            beta = np.linalg.lstsq(G, b, rcond=1e-09)[0]
            capped = len(ids) > limit
            if capped:
                keep = sorted(
                    range(len(extras)), key=lambda j: (-abs(beta[len(H) + j]), extras[j])
                )[: limit - len(H)]
                positions = list(range(len(H))) + [len(H) + j for j in keep]
                ids = tuple((ids[j] for j in positions))
                G = G[np.ix_(positions, positions)]
                b = b[positions]
            support = tuple(sorted(ids))
            if support in seen or support in archive.attempted or support in archive.keys:
                continue
            error = max(0.0, 1 - float(b @ np.linalg.pinv(G, rcond=1e-10) @ b))
            out.append(
                (
                    error,
                    dict(
                        parent=recipient["id"],
                        donor_state=donor["id"],
                        support=support,
                        origin="group_crossover",
                        history_anchor=donor["id"],
                        probe_relative_MSE=error,
                        crossover_capped=capped,
                    ),
                )
            )
            seen.add(support)
    out.sort(key=lambda row: (row[0], row[1]["donor_state"], row[1]["support"]))
    return out
