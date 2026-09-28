"""Print the exact role statements recorded for every probe cell, for explanation."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
raw = json.load(open(os.path.join(HERE, "role_probe_raw.json"), encoding="utf-8"))

TRUTH = {
    "Topography_Chemotaxis": "population density (non-negative, crowding-limited)",
    "Morphogenesis": "morphogen concentration (diffusion + source + Michaelis-Menten)",
    "Forced_Swift_Hohenberg": "order parameter of a pattern-forming (SH-type) system",
    "Traffic_Flow_Bottleneck": "traffic density on a road with a spatial bottleneck",
    "Predator_Prey": "f1 prey/resource density, f2 predator/consumer density",
}

for problem, truth in TRUTH.items():
    print("=" * 78)
    print(f"{problem}\n  truth: {truth}")
    for key in sorted(k for k in raw if k.startswith(problem) and "anon_with_data" in k):
        label = key.split("|")[1].replace("anon_with_data", "seed-1")
        text = raw[key]
        match = re.search(r"FIELD_ROLES\s*:(.*?)(?:\n\s*EQUATION|\Z)", text, re.S)
        roles = " ".join(match.group(1).split()) if match else "(no FIELD_ROLES line)"
        print(f"  {label:<8} [{len(text)} chars] {roles[:230]}")
