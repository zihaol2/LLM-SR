"""Leak audit for the A5 run: print its exact round-1 prompts, scan for leakage.

Copied here from 19_观测驱动算子库/code/audit_prompts.py so that the disclosure in
README section 4 can be re-checked from this folder. Reads the A5 run in place.
"""
import json
import os
import re

import a5_lib_exp as L

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(EXP)
OUT = os.path.join(REPO, "19_观测驱动算子库", "results")   # the A5 run, read in place

SUBJECT = ["traffic", "road", "vehicle", "car", "cars", "driver", "density",
           "population", "chemotaxis", "bacteria", "species", "fluid", "laser",
           "optical", "swift", "hohenberg", "bottleneck", "highway", "lane",
           "jam", "burgers", "kuramoto", "sivashinsky", "navier", "stokes",
           "fisher", "kpp", "reaction-diffusion"]


def main():
    prob = L.E.Problem(L.E.DATA)
    rows = [json.loads(l) for l in open(os.path.join(OUT, "history.jsonl"),
                                        encoding="utf-8")]

    print("#" * 78)
    print("# PROMPT 1 (round 1) — operator proposal, verbatim")
    print("#" * 78)
    print(L.propose_prompt(prob, [], "", ""))

    print("#" * 78)
    print("# PROMPT 2 (round 1) — equation search, verbatim")
    print("#" * 78)
    print(L.E.build_prompt("", "", "", ""))

    print("#" * 78)
    print("# PROMPT 3 (round 1) — feedback, template")
    print("#" * 78)
    print(L.feedback_prompt(prob, [], {"example": "diagnostics go here"},
                            "<best code>", "<params>", 1.0))

    # ---- scan every model output for subject-area vocabulary
    print("#" * 78)
    print("# LEAKAGE SCAN of model outputs (candidate code + feedback replies)")
    print("#" * 78)
    hits = {}
    for r in rows:
        text = (r.get("reply") or "") + (json.dumps(r.get("parsed") or {},
                                                    ensure_ascii=False))
        ts = " ".join(text.split())
        for w in SUBJECT:
            n = len(re.findall(rf"\b{re.escape(w)}\w*", ts, re.I))
            if n:
                hits.setdefault(w, []).append((r.get("round"), n))
    if not hits:
        print("  no subject-area vocabulary found in any model output")
    for w, occ in hits.items():
        print(f"  {w:16s} appears in rounds {sorted({o[0] for o in occ})}")
        for r in rows:
            text = " ".join(((r.get("reply") or "")
                             + json.dumps(r.get("parsed") or {},
                                          ensure_ascii=False)).split())
            for m in re.finditer(rf"\b{re.escape(w)}\w*", text, re.I):
                print(f"      round {r.get('round')}: "
                      f"...{text[max(0, m.start() - 60):m.end() + 40]}...")

    print()
    print("phenomenon-description noun hits, by round:",
          [(r["round"], r.get("noun_hits")) for r in rows
           if r.get("kind") == "feedback"])

    # ---- how much does the round-1 proposal copy the prompt's own example?
    prop = next(r for r in rows if r.get("kind") == "candidate")
    first_prompt = prop["prompt"]
    print()
    print("round-1 search prompt contains the phrase 'd_axis(f1 * (1 - f1)':",
          "d_axis(f1 * (1 - f1)" in first_prompt)


if __name__ == "__main__":
    main()
