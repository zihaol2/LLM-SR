"""Numerical feedback. No reference access, regional residual tables or vetoes."""

import copy
import time

import numpy as np

from .fitting import fit


def evaluate(archive, proposal):
    started = time.time()
    parent = archive.states[proposal["parent"]]["model"]
    initial = copy.deepcopy(proposal.get("initial", parent))
    if proposal.get("donor_state"):
        donor = archive.states[proposal["donor_state"]]["model"]
        inherited = {term["id"]: term for term in donor["terms"]}
        inherited.update({term["id"]: term for term in initial["terms"]})
        initial = dict(terms=list(inherited.values()))
    before = archive.data.fit_calls
    model = fit(
        archive.data,
        archive.field,
        [archive.registry.nodes[mid] for mid in proposal["support"]],
        archive.registry,
        initial=initial,
        full=True,
    )
    if archive.data.fit_calls != before + 1:
        raise RuntimeError("Every proposal must consume exactly one counted fit")
    row = {key: value for key, value in proposal.items() if key != "initial"}
    row.update(
        valid=model is not None,
        score=model["score"] if model else None,
        MSE=model["fit_mse"] if model else None,
        complexity=model["complexity"] if model else None,
        optimizer=model["optimizer"] if model else None,
        delta_score=model["score"] - parent["score"] if model else None,
        seconds=time.time() - started,
    )
    return model, row


def diagnose(data, field, state):
    model = state["model"]
    target_norm = max(float(np.linalg.norm(data.y[field])), 1e-30)
    contributions = []
    for term in model["terms"]:
        names = sorted(term["params"])
        vector = data.feature(
            dict(expr=term["expr"], params=names),
            [term["params"][name] for name in names],
        )
        contributions.append(
            dict(
                id=term["id"],
                coefficient=term["coefficient"],
                relative_contribution=float(
                    np.linalg.norm(term["coefficient"] * vector) / target_norm
                ),
            )
        )
    return dict(
        state=state["id"],
        MSE=model["fit_mse"],
        complexity=model["complexity"],
        normalized_MSE=model["fit_mse"] / data.scale[field],
        contributions=contributions,
    )
