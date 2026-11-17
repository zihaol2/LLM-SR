"""One declared protocol; coverage, recombination and revision budgets are separate."""
import os

SETTINGS = {
    "rounds": 20,
    "minimum_test_fraction": 0.5,
    "coverage_schedule": "legacy_half",
    "history_fits_per_round": 25,
    "revision_fits_per_round": None,
    "frontier_limit": 10,
    "compact_prompt": False,
    "numerical_backend": "reference",
    "fused_reference_only_expressions": ["PD(Exp(a*u),(1,0))"],
    "full_revision_scope": "At most 10 preliminary round-pool frontier routes; full prefixes/deletions/existing-core refinement; all valid fits enter round pool",
    "feedback_rule": "Working-set equations and aggregate numerical feedback only; no per-round choice or veto",
    "max_revision_requests": 8,
    "new_roots_per_call": 4,
    "coarse_parameters": [0.25, 0.5, 1.0, 2.0],
    "seed": 4204,
    "lambda": 0.1,
    "max_terms": 8,
    "model": os.environ.get("GEM_GSS_MODEL", ""),
    "base_url": os.environ.get("GEM_GSS_BASE_URL", ""),
    "temperature": 0.45,
    "max_tokens": 20000,
    "max_input_tokens": 100000,
    "reference_access": False,
    "fitter": "Unchanged variable projection; normalized lstsq; nonlinear bounds [-8,8]; full budget 45",
    "prompt_encoding": "A-style mechanism dictionary; no one-hop neighbors or explicit edges",
    "sampling_schedule": "Ten frozen initial-root groups, two sweeps, half quota and second-sweep coverage",
    "final_rule": "One LLM selects an existing final-working-set fit using the explicit MSE/complexity marginal-benefit prompt; no terminal fits",
    "frontier_scope": "previous working set union current-round valid fits; full archive used only by history recombination",
    "frontier_compression": "Preserve endpoints and largest normalized complexity/log-MSE chord deviations; no fixed fill count",
    "invalid_feedback_policy": "Log invalid planning proposals; repair coverage using only the same supplied working set",
    "unavailable_response_policy": "For malformed, failed or uncertain responses, preserve prior reviews and use the existing coverage fallback; log unavailable LLM planning; no repeat request",
}


def field_budget(total, fields, field):
    """A budget is TOTAL across target fields, not a per-field multiplier."""
    index = fields.index(field)
    return total // len(fields) + int(index < total % len(fields))
