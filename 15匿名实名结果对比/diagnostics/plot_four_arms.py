"""TRAFFIC: four arms, best-so-far MSE against the number of evaluated samples.

  baseline NAMED       original spec, variables disclosed, no library   (03, 500 samples)
  baseline ANONYMOUS   original spec, variables hidden,   no library   (01, 100 samples)
  library + NAMED      designed mechanism library, variables disclosed (10, 100 samples)
  library + ANONYMOUS  designed mechanism library, variables hidden    (11, 100 samples)

Left panel is the first 100 samples of every arm (the comparable window); the right panel
shows the full runs, where only the named baseline goes further.

The two baselines use the original search language (`d_dx(u, order)` plus free numpy);
the two library arms use the mechanism grammar. That difference is part of the picture.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "results"

ARMS = [
    ("baseline NAMED  (no library)", RUN / "baseline_named", "#1f77b4"),
    ("baseline ANONYMOUS  (no library)", RUN / "baseline_anonymous", "#ff7f0e"),
    ("library + NAMED", RUN / "library_named", "#2ca02c"),
    ("library + ANONYMOUS  (folder 11)", RUN / "library_anonymous_11", "#d62728"),
]


def curve(run_dir: Path):
    path = run_dir / "all_samples_history.jsonl"
    if not path.exists():
        path = run_dir / "all_samples.jsonl"
    xs, ms = [], []
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        if row.get("mse") is None:
            continue
        xs.append(float(row.get("sample_order", len(xs))))
        ms.append(float(row["mse"]))
    xs, ms = np.asarray(xs), np.asarray(ms)
    order = np.argsort(xs)
    return xs[order], np.minimum.accumulate(ms[order])


def main():
    curves = []
    print(f"{'arm':<36} {'n':>4} {'best@100':>11} {'best all':>11}")
    for label, run_dir, colour in ARMS:
        xs, best = curve(run_dir)
        cut = best[xs <= 100]
        curves.append((label, xs, best, colour))
        print(f"{label:<36} {len(xs):>4} {cut[-1]:>11.3e} {best[-1]:>11.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, (xmax, title) in zip(axes, [(100, "first 100 samples"), (510, "full runs")]):
        for label, xs, best, colour in curves:
            keep = xs <= xmax
            ax.plot(xs[keep], best[keep], color=colour, linewidth=1.9,
                    label=label if xmax == 100 else None)
            ax.annotate(f"{best[keep][-1]:.2e}", (xs[keep][-1], best[keep][-1]),
                        textcoords="offset points", xytext=(6, 2), fontsize=8.5,
                        color=colour)
        ax.set_yscale("log")
        ax.set_xlabel("evaluated samples")
        ax.set_ylabel("best MSE so far  (weak-form)")
        ax.set_title(f"TRAFFIC - {title}")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    out = HERE / "traffic_four_arms.png"
    fig.savefig(out, dpi=160)
    print("saved:", out)


if __name__ == "__main__":
    main()
