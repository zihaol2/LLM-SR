"""Score the three conditions and draw the summary figure.

Two levels are scored, both by reading the 27 responses next to the ground truth:

  naming  the response COMMITS to the true quantity ("traffic density",
          "order parameter", "population density")
  family  the true parent family is named even if the quantity stays hedged
          ("a scalar field (e.g. concentration, temperature or amplitude)")
  none    neither

The labels below are the hand-assigned verdicts, so the judgement is auditable
against `ablation_all.json` rather than hidden in a keyword rule.
"""
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# (problem, condition, rep) -> verdict
VERDICT = {
    ("traffic", "cond1", 1): "none", ("traffic", "cond1", 2): "none",
    ("traffic", "cond1", 3): "none",
    ("traffic", "cond2", 1): "none", ("traffic", "cond2", 2): "none",
    ("traffic", "cond2", 3): "none",
    ("traffic", "cond3", 1): "naming", ("traffic", "cond3", 2): "naming",
    ("traffic", "cond3", 3): "naming",
    ("swift_hohenberg", "cond1", 1): "none", ("swift_hohenberg", "cond1", 2): "none",
    ("swift_hohenberg", "cond1", 3): "none",
    # cond2 names the family (Swift-Hohenberg / Ginzburg-Landau) but the quantity
    # stays a hedged list: "concentration, temperature anomaly, or amplitude"
    ("swift_hohenberg", "cond2", 1): "family", ("swift_hohenberg", "cond2", 2): "family",
    ("swift_hohenberg", "cond2", 3): "family",
    ("swift_hohenberg", "cond3", 1): "naming", ("swift_hohenberg", "cond3", 2): "naming",
    ("swift_hohenberg", "cond3", 3): "naming",
    ("chemotaxis", "cond1", 1): "none", ("chemotaxis", "cond1", 2): "none",
    ("chemotaxis", "cond1", 3): "none",
    ("chemotaxis", "cond2", 1): "naming", ("chemotaxis", "cond2", 2): "naming",
    ("chemotaxis", "cond2", 3): "naming",
    ("chemotaxis", "cond3", 1): "naming", ("chemotaxis", "cond3", 2): "naming",
    ("chemotaxis", "cond3", 3): "naming",
}

PROBLEM_LABEL = {
    "traffic": "交通瓶颈\n(真值: 车流密度)",
    "swift_hohenberg": "Forced Swift–Hohenberg\n(真值: 序参量/图样振幅)",
    "chemotaxis": "Topography Chemotaxis\n(真值: 种群密度)",
}
COND_LABEL = ["cond1\n只给统计", "cond2\n+真值方程", "cond3\n+方程+母体"]

COND_KEY = {"cond1": "cond1_statistics_only",
            "cond2": "cond2_statistics_equation",
            "cond3": "cond3_statistics_equation_parents"}

with io.open(os.path.join(HERE, "ablation_all.json"), encoding="utf-8") as handle:
    data = json.load(handle)

problems = ["traffic", "swift_hohenberg", "chemotaxis"]
colour = {"naming": "#2e8b57", "family": "#e08a1e", "none": "#c0392b"}
text = {"naming": "命名正确", "family": "只对族", "none": "未命中"}

fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.0), dpi=150,
                             gridspec_kw={"width_ratios": [2.5, 1]})

tally = {"cond1": [0, 0, 0], "cond2": [0, 0, 0], "cond3": [0, 0, 0]}  # naming, family, none
for row, prob in enumerate(problems):
    for col, cond in enumerate(("cond1", "cond2", "cond3")):
        for rep in range(1, 4):
            verdict = VERDICT[(prob, cond, rep)]
            runs = data[prob]["conditions"][COND_KEY[cond]]
            confidence = (runs[rep - 1]["raw_response"] or {}).get("confidence") or "?"
            x0 = col * 1.15 + (rep - 1) * 0.36
            y0 = len(problems) - 1 - row
            ax.add_patch(Rectangle((x0, y0 + 0.12), 0.33, 0.76,
                                   facecolor=colour[verdict], alpha=0.88,
                                   edgecolor="white", lw=1.4))
            ax.text(x0 + 0.165, y0 + 0.60, text[verdict], ha="center", va="center",
                    color="white", fontsize=7.6)
            ax.text(x0 + 0.165, y0 + 0.30, confidence, ha="center", va="center",
                    color="white", fontsize=6.8)
        tally[cond][0 if verdict == "naming" else (1 if verdict == "family" else 2)] += 0

for cond in ("cond1", "cond2", "cond3"):
    for row, prob in enumerate(problems):
        pass

ax.set_xlim(0, 3 * 1.15)
ax.set_ylim(0, 3)
ax.set_xticks([i * 1.15 + 0.52 for i in range(3)], COND_LABEL, fontsize=10)
ax.set_yticks([2.5, 1.5, 0.5],
              [PROBLEM_LABEL[p] for p in problems], fontsize=9)
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("三种证据条件下的角色命名（每格一次独立回答，灰字为置信度）", fontsize=11)

counts = {cond: {"naming": 0, "family": 0, "none": 0} for cond in tally}
for (prob, cond, rep), verdict in VERDICT.items():
    counts[cond][verdict] += 1

bottoms = [0, 0, 0]
for kind in ("naming", "family", "none"):
    vals = [counts[c][kind] for c in ("cond1", "cond2", "cond3")]
    bx.bar([0, 1, 2], vals, bottom=bottoms, color=colour[kind], width=0.6,
           label={"naming": "命名正确", "family": "只对族", "none": "未命中"}[kind])
    for i, (v, b) in enumerate(zip(vals, bottoms)):
        if v:
            bx.text(i, b + v / 2, "%d" % v, ha="center", va="center", color="white",
                    fontsize=10)
    bottoms = [b + v for b, v in zip(bottoms, vals)]
bx.set_xticks([0, 1, 2], ["cond1", "cond2", "cond3"])
bx.set_ylabel("回答数（3 问题 × 3 次 = 9）")
bx.set_title("三条件汇总", fontsize=11)
bx.legend(fontsize=8, loc="upper left")
bx.grid(axis="y", alpha=0.25)

fig.tight_layout()
out = os.path.join(HERE, "对比_角色识别_三问题.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
for cond in ("cond1", "cond2", "cond3"):
    c = counts[cond]
    print("%-6s naming %d/9 | family-only %d/9 | none %d/9" % (cond, c["naming"], c["family"], c["none"]))
