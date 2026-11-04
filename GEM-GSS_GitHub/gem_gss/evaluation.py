"""Explicit post-selection evaluation. Never imported by discovery or fitting."""

from pathlib import Path
import sympy as s
from . import core as C, structure


def evaluate_result(case_id, path):
    if path.name != "final_result.json" or not path.is_file():
        raise ValueError("Evaluate a frozen final_result.json, not a pending search state")
    result = C.read(path)
    references = C.read(C.ASSETS / "evaluation_references.json")
    case = next(c for c in C.read(C.ASSETS / "cases.json") if c["id"] == case_id)
    language = C.Graph(case["fields"], case["coordinates"]).language
    fields = {}
    found = set()
    true = set()
    for field, equation in result["equations"].items():
        expression = sum(
            (structure.symbolic_term(t, language) for t in equation["terms"]), s.Integer(0)
        )
        a = structure.support(expression, language)
        b = structure.support(
            language.symbolic(language.parse(references[case_id][field])), language
        )
        found.update((field, item) for item in a)
        true.update((field, item) for item in b)
        fields[field] = dict(
            jaccard=len(a & b) / len(a | b) if a | b else 1.0,
            matched=len(a & b),
            true_terms=len(b),
            found_terms=len(a),
            missing=sorted(b - a),
            extra=sorted(a - b),
        )
    matched = len(found & true)
    return dict(
        case=case_id,
        jaccard=matched / len(found | true) if found | true else 1.0,
        recovery=matched / len(true) if true else 1.0,
        fields=fields,
        post_selection_only=True,
    )
