"""One clean summary of the fully-anonymous arms. Reads existing runs, writes nothing
back into them. No new experiments.

The ladder, all of it fully anonymous (no structural prior, no subject nouns):
   A3  frontier feedback only                (no library, no figure diagnosis)
   +A4 figure-based fine-grained feedback    (no library)
   +A5 observation-driven operator library   (no library before it; library grown from data)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(EXP)

ARMS = [
    ("A3", "前沿反馈（无库、无图）", "18_零先验_细粒度反馈/results_pareto", "#7f8c8d"),
    ("A4", "+ 图的细粒度反馈", "18_零先验_细粒度反馈/results_both", "#2e7d32"),
    ("A5", "+ 观测驱动的算子库", "19_观测驱动算子库/results", "#0b5394"),
]
BASELINE_ANON = 1.336e-4        # 完全匿名基线（无库、无结构先验）
FLOOR = 3.128e-11              # 半匿名 + 人写机制库（只作参照）


def best_traj(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    ms = []
    for r in rows:
        is_cand = (r.get("kind") == "candidate" if r.get("kind")
                   else ("reply" in r and "feedback_reply" not in r))
        if is_cand and isinstance(r.get("mse"), float) and r["mse"] < 1e6:
            ms.append(r["mse"])
    return (np.minimum.accumulate(ms) if ms else np.array([])), ms


def main():
    table = []
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    for tag, label, rel, color in ARMS:
        t, ms = best_traj(os.path.join(REPO, rel, "history.jsonl"))
        distinct = len({round(m, 10) for m in ms})
        lib_path = os.path.join(REPO, rel, "library.json")
        nlib = len(json.load(open(lib_path, encoding="utf-8"))) if os.path.exists(
            lib_path) else 0
        table.append({"tag": tag, "setting": label, "run": rel,
                      "best_mse": float(t[-1]) if t.size else None,
                      "evaluated": len(ms), "distinct_mse": distinct,
                      "collapsed": bool(ms) and distinct < 0.5 * len(ms),
                      "library_entries": nlib,
                      "gain_vs_anonymous_baseline":
                          round(BASELINE_ANON / float(t[-1]), 2) if t.size else None})
        ax.semilogy(range(1, len(t) + 1), t, lw=2.4, color=color,
                    label=f"{tag}  {label}")

    ax.axhline(BASELINE_ANON, ls="--", lw=1.3, color="#2c3e50",
               label=f"完全匿名基线 {BASELINE_ANON:.2e}")
    ax.axhline(FLOOR, ls="-", lw=1.4, color="#e67e22",
               label=f"半匿名 + 人写机制库（参照）{FLOOR:.2e}")
    ax.set_xlabel("候选方程序号")
    ax.set_ylabel("best weak-form MSE (log)")
    ax.set_title("完全匿名设定下的三级阶梯：前沿 → 图的细粒度反馈 → 观测驱动算子库",
                 fontsize=11)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_fig = os.path.join(EXP, "阶梯.png")
    fig.savefig(out_fig, dpi=150)

    with open(os.path.join(EXP, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"arms": table,
                   "anonymous_baseline": BASELINE_ANON,
                   "semi_anonymous_reference": FLOOR}, fh,
                  ensure_ascii=False, indent=1)

    print(f"{'arm':4s} {'setting':26s} {'best MSE':>10s} {'eval':>5s} "
          f"{'distinct':>8s} {'collapse':>8s} {'lib':>4s} {'vs anon':>8s}")
    for r in table:
        print(f"{r['tag']:4s} {r['setting']:26s} {r['best_mse']:10.3e} "
              f"{r['evaluated']:5d} {r['distinct_mse']:8d} "
              f"{str(r['collapsed']):>8s} {r['library_entries']:4d} "
              f"{r['gain_vs_anonymous_baseline']:8.2f}")
    print("\nwrote", out_fig)
    print("wrote", os.path.join(EXP, "summary.json"))


if __name__ == "__main__":
    main()
