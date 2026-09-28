"""Loose structural check plus a few representative equation bodies, no API calls."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
raw = json.load(open(os.path.join(HERE, "traffic_role_ablation.json"), encoding="utf-8"))


def code_of(text: str) -> str | None:
    for block in re.findall(r"```(?:python)?\s*(.*?)```", text, re.S):
        if "def rhs" in block:
            return block.strip()
    return None


for arm in ("wrong", "correct", "none"):
    bodies = [code_of(raw[k]) for k in raw if k.startswith(f"{arm}|")]
    bodies = [b for b in bodies if b]
    loose = sum(1 for b in bodies
                if re.search(r"d_dx\(\s*[^)]*u\s*\*", b) or re.search(r"d_dx\(\s*flux", b))
    tanh = sum(1 for b in bodies if "tanh" in b)
    print(f"{arm:<8} codes={len(bodies):<3} with np.tanh={tanh:<3} "
          f"derivative-of-a-u-flux={loose}")

print()
for key in ("wrong|14", "wrong|16", "correct|5", "none|0"):
    body = code_of(raw.get(key, ""))
    if body:
        single = " ".join(body.split())
        print(f"### {key}\n   {single[:340]}\n")
