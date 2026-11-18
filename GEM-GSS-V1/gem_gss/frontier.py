"""Bounded MSE/complexity working set. Never searches unsubmitted history."""

import math


def representatives(states, limit=10):
    if limit < 2:
        raise ValueError("Frontier limit must be at least two")
    unique = {row["id"]: row for row in states}
    values = [
        row
        for row in unique.values()
        if math.isfinite(row["model"]["fit_mse"]) and row["model"]["fit_mse"] >= 0
    ]
    values.sort(key=lambda row: (row["model"]["complexity"], row["model"]["fit_mse"], row["id"]))
    front, error = [], float("inf")
    for row in values:
        if row["model"]["fit_mse"] < error:
            front.append(row)
            error = row["model"]["fit_mse"]
    if len(front) <= limit:
        return front, dict(input_count=len(unique), frontier_count=len(front), compressed=False)
    # Preserve endpoints, then the largest omitted bend of each selected chord.
    # Normalize complexity and log(MSE); no Score or benchmark labels are used.
    cs = [row["model"]["complexity"] for row in front]
    es = [math.log(max(row["model"]["fit_mse"], 1e-300)) for row in front]
    cr, er = max(cs) - min(cs), max(es) - min(es)
    points = [
        ((c - min(cs)) / max(cr, 1e-30), (e - min(es)) / max(er, 1e-30)) for c, e in zip(cs, es)
    ]
    selected = {0, len(front) - 1}
    while len(selected) < limit:
        options = []
        ordered = sorted(selected)
        for left, right in zip(ordered, ordered[1:]):
            ax, ay = points[left]
            bx, by = points[right]
            dx, dy = bx - ax, by - ay
            denominator = max(dx * dx + dy * dy, 1e-30)
            for i in range(left + 1, right):
                px, py = points[i]
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denominator))
                distance = (px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2
                options.append((distance, -i, i))
        if not options:
            break
        selected.add(max(options)[2])
    return [front[i] for i in sorted(selected)], dict(
        input_count=len(unique),
        frontier_count=len(front),
        compressed=True,
        compression="normalized complexity/log-MSE chord deviation; endpoints protected",
    )


def update(archive, previous_ids, current_ids, limit=10):
    pool = list(dict.fromkeys(list(previous_ids) + list(current_ids)))
    states = [
        archive.states[sid]
        for sid in pool
        if sid is not None and len(archive.states[sid]["model"]["terms"]) <= 8
    ]
    chosen, audit = representatives(states, limit)
    archive.active_ids = [state["id"] for state in chosen]
    return dict(
        **audit,
        input_ids=pool,
        selected_ids=list(archive.active_ids),
        candidate_scope="previous working set union current-round valid fits",
    )
