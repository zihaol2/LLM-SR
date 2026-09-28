"""Judge the three models' free-text role answers with an explicit keyword rubric.

The rubric is stated in full so the counting is reproducible and arguable, unlike a
hand reading. HIT = the answer names the true quantity; PARTIAL = right class, wrong
specific quantity; MISS = something else.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
raw = json.load(open(os.path.join(HERE, "role_probe_raw.json"), encoding="utf-8"))


def judge_problem(name: str, text: str) -> str:
    body = text.lower()
    if name == "Topography_Chemotaxis":
        if re.search(r"population|species|cell", body) and re.search(r"density|concentration", body):
            return "HIT"
        return "PARTIAL" if re.search(r"density|concentration", body) else "MISS"
    if name == "Morphogenesis":
        if "morphogen" in body or ("concentration" in body and re.search(r"diffus|source", body)):
            return "HIT"
        return "PARTIAL" if re.search(r"density|concentration|temperature", body) else "MISS"
    if name == "Forced_Swift_Hohenberg":
        if "order parameter" in body or "swift-hohenberg" in body:
            return "HIT"
        return "PARTIAL" if re.search(r"pattern|amplitude", body) else "MISS"
    if name == "Traffic_Flow_Bottleneck":
        if "traffic" in body or ("vehicle" in body and "density" in body):
            return "HIT"
        return "PARTIAL" if re.search(r"tracer|scalar|concentration|density", body) else "MISS"
    # Predator_Prey: f1 then f2
    if re.search(r"prey|resource|autocatalytic", body):
        field1 = "HIT"
    else:
        field1 = "PARTIAL" if re.search(r"population|density|concentration", body) else "MISS"
    if re.search(r"predator|consumer|grazer|hunter", body):
        field2 = "HIT"
    else:
        field2 = "PARTIAL" if re.search(r"chemical|density|concentration|allele", body) else "MISS"
    return f"{field1}/{field2}"


def head_claims(name: str, line: str) -> str:
    """The model's *primary* claim per field: the noun phrase right after `fN =`.

    A long answer that lists alternatives would otherwise be scored on every candidate
    it mentions, which rewards hedging instead of identification.
    """
    claims = []
    fields = ["f1", "f2"] if name == "Predator_Prey" else ["f1"]
    for field in fields:
        match = re.search(rf"{field}\s*=\s*([^;(\n]+)", line, re.I)
        claims.append(match.group(1).strip() if match else "")
    return " ; ".join(f"{f}={c}" for f, c in zip(fields, claims))


models = ["deepseek-v4-flash", "claude-haiku-4-5", "gemini-3.5-flash-lite"]
problems = ["Topography_Chemotaxis", "Morphogenesis", "Forced_Swift_Hohenberg",
            "Traffic_Flow_Bottleneck", "Predator_Prey"]

print("=== 判定表（HIT 命中 / PARTIAL 类对但身份不对 / MISS 错）===")
print(f"{'equation':<24} " + " ".join(f"{m.split('-')[0][:9]:>12}" for m in models))
score = {m: {"HIT": 0, "PARTIAL": 0, "MISS": 0} for m in models}
uncertain = {m: 0 for m in models}
total = {m: 0 for m in models}
for name in problems:
    cells = []
    for model in models:
        verdicts = []
        for seed in ("", "#2", "#3"):
            key = f"{name}|anon_with_data{seed}|{model}"
            text = raw.get(key, "")
            match = re.search(r"FIELD_ROLES\s*:(.*?)(?:\n\s*EQUATION|\Z)", text, re.S)
            if not match:
                verdicts.append("n/a")
                continue
            line = match.group(1)
            verdict = judge_problem(name, line)
            verdicts.append(verdict)
            total[model] += 1
            for part in verdict.split("/"):
                score[model][part] = score[model].get(part, 0) + 1
            if re.search(r"not uniquely|not identifiable|undetermined|not determined|"
                         r"cannot be (?:uniquely )?determined|alternatives?|not pinned", line, re.I):
                uncertain[model] += 1
        hit = sum(1 for v in verdicts if v.startswith("HIT"))
        cells.append(f"{hit}/3")
    print(f"{name:<24} " + " ".join(f"{c:>12}" for c in cells))

print("\n=== 汇总（按场计：Predator-Prey 算两个场）===")
for model in models:
    s = score[model]
    n = sum(s.values())
    print(f"{model:<22} HIT {s['HIT']:>2}/{n:<3} PARTIAL {s['PARTIAL']:>2} MISS {s['MISS']:>2}  "
          f"| 自述'不能唯一确定' {uncertain[model]}/{total[model]} 次调用")

print("\n=== 严格模式：只判 `fN =` 后面的主张本体 ===")
strict = {m: {"HIT": 0, "PARTIAL": 0, "MISS": 0} for m in models}
strict_total = {m: 0 for m in models}
for name in problems:
    cells = []
    for model in models:
        hits = 0
        for seed in ("", "#2", "#3"):
            key = f"{name}|anon_with_data{seed}|{model}"
            text = raw.get(key, "")
            match = re.search(r"FIELD_ROLES\s*:(.*?)(?:\n\s*EQUATION|\Z)", text, re.S)
            if not match:
                continue
            verdict = judge_problem(name, head_claims(name, match.group(1)))
            strict_total[model] += 1
            for part in verdict.split("/"):
                strict[model][part] = strict[model].get(part, 0) + 1
            hits += verdict.startswith("HIT")
        cells.append(f"{hits}/3")
    print(f"{name:<24} " + " ".join(f"{c:>12}" for c in cells))
for model in models:
    s = strict[model]
    n = sum(s.values())
    print(f"{model:<22} HIT {s['HIT']:>2}/{n:<3} PARTIAL {s['PARTIAL']:>2} MISS {s['MISS']:>2}")
