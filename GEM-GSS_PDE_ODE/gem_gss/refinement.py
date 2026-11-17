"""Full online migration of the old post-search numerical refinement.

Generation/path selection is here; every fit is delegated to Feedback and every
valid result is stored by Memory via the supplied evaluate callback. No fit cap,
no polynomial-prefix truncation, and no requirement for intermediate improvements
before trying higher polynomial degrees. References and API clients are absent.
"""
import copy
import sympy as s
from . import core as C, structure
from .graph import signature
from .frontier import representatives
from .structural_rules import additive_deletions, reaction_block, graph_powers, GraphCores, core_proposals


def select_routes(archive):
    """Repair the supplied working set, never rescan complete history."""
    return [dict(label=f'R{i+1:02d}',state=state['id'],model=copy.deepcopy(state['model']))
            for i,state in enumerate(archive.view()) if state['model']['terms']]


class FullRefiner:
    def __init__(self, archive, number, evaluate):
        self.archive, self.registry, self.number = archive, archive.registry, number
        self.evaluate = evaluate

    def trial(self, parent, ids, kind, initial=None, **extra):
        # Deliberately re-fit previously tested supports: warm starts and parent
        # context matter, and the old terminal procedure did not skip these fits.
        return self.evaluate(dict(parent=parent["state"], support=tuple(sorted(set(ids))),
                                  kind=kind, initial=copy.deepcopy(initial or parent["model"]), **extra))

    def deletions(self, parent):
        model, seen = parent["model"], set()
        for i, term in enumerate(model["terms"]):
            rest = [other for j, other in enumerate(model["terms"]) if j != i]
            expression = self.registry.graph.language.parse(term["expr"])
            options = [(None, "online_delete_mechanism")]
            options.extend((C.render(reduced), "online_delete_additive_component")
                           for reduced in additive_deletions(expression, self.registry.graph.language))
            for text, kind in options:
                mid = None
                if text is not None:
                    try:
                        mid, _ = self.registry.propose(text, self.number, dict(kind=kind))
                    except (ValueError, KeyError, SyntaxError):
                        continue
                ids = tuple(sorted({other["id"] for other in rest} | ({mid} if mid else set())))
                if ids in seen or ids == signature(model):
                    continue
                seen.add(ids)
                if C.complexity([self.registry.nodes[x] for x in ids], self.registry) >= model["complexity"]:
                    continue
                initial = copy.deepcopy(model)
                if mid:
                    initial["terms"].append(dict(id=mid, params=term["params"]))
                yield ids, kind, initial, dict(removed_or_simplified=term["id"],
                                              before_expression=term["expr"], after_expression=text)

    def reduce(self, original):
        baseline = self.trial(original, signature(original["model"]), "online_baseline_refit")
        current = baseline if baseline["model"] and baseline["model"]["score"] < original["model"]["score"] else original
        trials, accepted, iteration = [], [], 0
        while True:
            iteration += 1
            fitted = []
            for ids, kind, initial, extra in self.deletions(current):
                row = self.trial(current, ids, kind, initial=initial, iteration=iteration, **extra)
                trials.append(row)
                if row["model"]:
                    fitted.append(row)
            better = [row for row in fitted if row["model"]["score"] < current["model"]["score"]]
            if not better:
                break
            current = min(better, key=lambda row: (row["model"]["score"], row["model"]["complexity"], row["model"]["fit_mse"]))
            accepted.append(current)
        return dict(original=original, baseline=baseline, pruned=current, trials=trials, accepted=accepted)

    def core_reduce(self, original):
        current, trials, accepted = original, [], []
        index = GraphCores(self.registry)
        iteration = 0
        while True:
            iteration += 1
            seen, fitted = set(), []
            for i, term in enumerate(current["model"]["terms"]):
                rest = {other["id"] for j, other in enumerate(current["model"]["terms"]) if j != i}
                for mid, rule in core_proposals(term, self.registry, index):
                    ids = tuple(sorted(rest | {mid}))
                    if ids in seen or ids == signature(current["model"]):
                        continue
                    seen.add(ids)
                    if C.complexity([self.registry.nodes[x] for x in ids], self.registry) >= current["model"]["complexity"]:
                        continue
                    row = self.trial(current, ids, "online_replace_existing_core", iteration=iteration, rule=rule)
                    trials.append(row)
                    if row["model"]:
                        fitted.append(row)
            better = [row for row in fitted if row["model"]["score"] < current["model"]["score"]]
            if not better:
                break
            current = min(better, key=lambda row: (row["model"]["score"], row["model"]["complexity"], row["model"]["fit_mse"]))
            accepted.append(current)
        return dict(original=original, pruned=current, trials=trials, accepted=accepted)

    @staticmethod
    def branches(family):
        branches = [("deletion", family["deletion"])]
        branches.extend(("powers_"+",".join(map(str, item["exponents"])), item["deletion"])
                        for item in family["polynomial"])
        branches.extend(family.get("core_traces", {}).items())
        return branches

    def anchor(self, family):
        options = [("original", family["original"], None)]
        for label, trace in self.branches(family):
            row = trace["pruned"]
            if row["model"]["terms"] and len(row["model"]["terms"]) <= C.SETTINGS["max_terms"]:
                options.append((label, row, trace))
        return min(options, key=lambda item: (item[1]["model"]["score"], item[1]["model"]["complexity"], item[1]["model"]["fit_mse"]))

    def family(self, route):
        original = dict(state=route["state"], model=copy.deepcopy(route["model"]))
        family = dict(label=route["label"], original=original, deletion=self.reduce(original), polynomial=[])
        block, nonpoly, vanishes = reaction_block(original["model"], self.registry, self.archive.field)
        if nonpoly:
            removed = {term["id"] for term in block}
            retained = [term["id"] for term in original["model"]["terms"] if term["id"] not in removed]
            powers = graph_powers(self.registry, self.archive.field, include_constant=not vanishes)
            family["available_powers"] = [power for power, _ in powers]
            for size in range(1, len(powers)+1):
                ids = retained+[node["id"] for _, node in powers[:size]]
                exponents = [power for power, _ in powers[:size]]
                print("FULL REVISION", self.number, route["label"], self.archive.field,
                      "powers", exponents, "temporary terms", len(set(ids)), flush=True)
                joint = self.trial(original, ids, "online_polynomial_replacement", exponents=exponents,
                                   route=route["label"], temporary_term_count=len(set(ids)))
                if joint["model"]:
                    family["polynomial"].append(dict(exponents=exponents, joint=joint, deletion=self.reduce(joint)))
        # Original T03 numerical pipeline above, plus the existing-core rule
        # already present in the newer method. Neither truncates prefix trials.
        _, anchor, _ = self.anchor(family)
        core = self.core_reduce(anchor)
        family["core_traces"] = dict(existing_core=core, existing_core_then_deletion=self.reduce(core["pruned"]))
        return family

    def run(self):
        routes = select_routes(self.archive)  # Freeze all routes before any refinement.
        families = [self.family(route) for route in routes]
        # Every valid intermediate was already recorded by the feedback callback.
        # All capacity-eligible intermediates compete in the round-output frontier;
        # no Score anchor/deletion-sibling shortlist hides them from that update.
        return dict(routes=routes, families=families)
