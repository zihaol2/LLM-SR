"""Evaluation tools, imported only by explicit post-search CLI commands."""

from pathlib import Path
import csv
import json
import time
import numpy as np
import sympy as s
from . import config
from .io import dump, read, sha


def evaluate(run_dir, references, output_dir, indices=None):
    """Check all selected freezes before parsing any reference equations."""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    indices = list(range(len(config.TASKS))) if indices is None else list(indices)
    stages = ["proposed", "baseline_100"]
    if all(
        (run_dir / "proposed42" / config.TASKS[i].id / "frozen_result.json").is_file()
        for i in indices
    ):
        stages.insert(0, "proposed42")
    paths = {}
    for stage in stages:
        for i in indices:
            task = config.TASKS[i]
            path = run_dir / stage / task.id / "frozen_result.json"
            if not path.is_file():
                raise RuntimeError(
                    f"Freeze every selected result before evaluation: missing {stage}/{task.id}"
                )
            paths[(stage, i)] = path
    frozen = {key: read(p) for key, p in paths.items()}
    for (stage, i), value in frozen.items():
        task = config.TASKS[i]
        if (
            value.get("reference_access") is not False
            or value.get("task") != task.name
            or value.get("anonymous_id") != task.id
        ):
            raise RuntimeError("Invalid frozen-result identity/reference-access metadata")
    hashes = {p.relative_to(run_dir).as_posix(): sha(p) for p in paths.values()}
    # References are deliberately absent from all discovery imports and inputs.
    from . import scoring as E

    reference = read(references)
    rows = []
    details = []
    totals = {stage: {"intersection": 0, "union": 0, "jaccards": []} for stage in stages}
    for (stage, i), value in frozen.items():
        task = config.TASKS[i]
        found = set()
        truth = set()
        outputs = {}
        for field, equation in value["equations"].items():
            expression = sum(
                s.Float(t["coefficient"], 15) * E.parse(t["expr"], t["params"])
                for t in equation["terms"]
            )
            a = E.support(expression)
            b = E.support(E.parse(reference[task.name][field]))
            found |= {(field, t) for t in a}
            truth |= {(field, t) for t in b}
            union = len(a | b)
            outputs[field] = {
                "intersection": len(a & b),
                "union": union,
                "jaccard": len(a & b) / union if union else 1.0,
                "discovered": sorted(a),
                "reference": sorted(b),
                "MSE": equation["MSE"],
            }
        intersection = len(found & truth)
        union = len(found | truth)
        j = intersection / union if union else 1.0
        row = {
            "method": stage,
            "task_id": task.id,
            "task": task.name,
            "MSE": value["MSE"],
            "jaccard": j,
            "intersection": intersection,
            "union": union,
        }
        rows.append(row)
        details.append({**row, "outputs": outputs, "equations": value["equations"]})
        total = totals[stage]
        total["intersection"] += intersection
        total["union"] += union
        total["jaccards"].append(j)
    summary = {
        stage: {
            "mean_jaccard": float(np.mean(v["jaccards"])),
            "pooled_jaccard": v["intersection"] / v["union"] if v["union"] else 1.0,
            "exact_matches": sum(j == 1 for j in v["jaccards"]),
        }
        for stage, v in totals.items()
    }
    # A concurrent mutation invalidates this evaluation rather than mixing versions.
    if any(sha(run_dir / rel) != digest for rel, digest in hashes.items()):
        raise RuntimeError("Frozen results changed during evaluation")
    result = {
        "evaluation_unix": time.time(),
        "frozen_sha256": hashes,
        "reference_sha256": sha(references),
        "all_selected_results_frozen_before_reference_parse": True,
        "rows": rows,
        "details": details,
        "summary": summary,
        "MSE_definition": "All-observed-time spatial window integral residual after final outer-coefficient refit; not held-out forecasting MSE",
        "jaccard_definition": "Unchanged expanded additive differential structural support; EPS=1e-10; numeric coefficient/scales ignored, powers/functions/variables/output labels retained",
    }
    dump(output_dir / "comparison.json", result)
    with (output_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 冻结后结果比较",
        "",
        "MSE 为全部观测时刻的空间窗口积分残差，不是独立预测误差。",
        "",
        "| 方法 | 任务 | MSE | Jaccard |",
        "|---|---|---:|---:|",
    ]
    lines += [
        f'| {r["method"]} | {r["task"]} | {r["MSE"]:.8e} | {r["jaccard"]:.6f} |' for r in rows
    ]
    lines += ["", "## 汇总", ""] + [
        f'- {stage}: macro={v["mean_jaccard"]:.9f}, pooled={v["pooled_jaccard"]:.9f}'
        for stage, v in summary.items()
    ]
    (output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def check_recorded_mse(run_dir, indices=None):
    """Recalculate residuals with saved coefficients, without fitting or truth."""
    from . import common as C

    indices = range(len(config.TASKS)) if indices is None else indices
    checks = []
    for i in indices:
        task = config.TASKS[i]
        data = C.Data(task.name)
        for stage in ("proposed42", "proposed", "baseline_100"):
            path = Path(run_dir) / stage / task.id / "frozen_result.json"
            if not path.exists():
                continue
            result = read(path)
            for field, eq in result["equations"].items():
                prediction = np.zeros_like(data.yfull[field])
                for term in eq["terms"]:
                    node = {"expr": term["expr"], "params": list(term["params"])}
                    prediction += term["coefficient"] * data.feature(
                        node, list(term["params"].values()), True
                    )
                mse = float(np.mean((prediction - data.yfull[field]) ** 2))
                if not np.isclose(mse, eq["MSE"], rtol=5e-4, atol=1e-18):
                    raise RuntimeError(
                        f'Recorded MSE differs for {stage}/{task.id}/{field}: {mse} vs {eq["MSE"]}'
                    )
                checks.append(
                    {
                        "method": stage,
                        "task": task.id,
                        "field": field,
                        "recorded": eq["MSE"],
                        "recomputed": mse,
                    }
                )
    return checks


def usage_summary(run_dir):
    result = {}
    for stage in ("proposed42", "baseline_100"):
        accepted = 0
        prompt = 0
        completion = 0
        total = 0
        attempts = 0
        attempt_total = 0
        for path in (Path(run_dir) / stage).glob("*/api/*_usage.json"):
            value = read(path)
            usage = value.get("usage") or {}
            accepted += 1
            prompt += usage.get("prompt_tokens", 0)
            completion += usage.get("completion_tokens", 0)
            total += usage.get("total_tokens", 0)
        for path in (Path(run_dir) / stage).glob("*/api/*_attempts.json"):
            values = read(path)
            attempts += len(values)
            attempt_total += sum((v.get("usage") or {}).get("total_tokens", 0) for v in values)
        result[stage] = {
            "accepted_returns_with_usage_files": accepted,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "logged_HTTP_attempts": attempts,
            "all_logged_attempts_total_tokens": attempt_total,
            "attempt_accounting_available": attempts > 0,
        }
    return result
