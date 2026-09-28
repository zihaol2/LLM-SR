"""Show which role-probe cells are filled, and summarise the role lines."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
raw = json.load(open(os.path.join(HERE, "role_probe_raw.json"), encoding="utf-8"))

for condition in ("anon", "described", "anon_with_data"):
    keys = sorted(k for k in raw if f"|{condition}" in k or f"|{condition}#" in k)
    print(f"\n=== {condition}: {len(keys)} cells")
    for key in keys:
        text = raw[key]
        label = key.split("|deepseek")[0]
        match = re.search(r"FIELD_ROLES\s*:(.*?)(?:\n\s*EQUATION|\Z)", text, re.S)
        roles = " ".join(match.group(1).split())[:110] if match else "(no FIELD_ROLES line)"
        print(f"  {label:<48} {len(text):>5} chars  {roles}")
