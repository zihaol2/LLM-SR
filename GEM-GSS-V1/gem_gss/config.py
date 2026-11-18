"""One declared protocol; coverage, recombination and revision budgets are separate."""

import os

SETTINGS = {
    "rounds": 20,
    "minimum_test_fraction": 0.1,
    "coverage_schedule": "global_1pct",
    "history_fits_per_round": 25,
    "frontier_limit": 10,
    "compact_prompt": True,
    "numerical_backend": "reference",
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
}


def field_budget(total, fields, field):
    """A budget is TOTAL across target fields, not a per-field multiplier."""
    index = fields.index(field)
    return total // len(fields) + int(index < total % len(fields))
