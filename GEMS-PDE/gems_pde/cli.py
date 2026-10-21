"""One portable CLI for fresh search, saved-search correction and evaluation."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
from . import config
from .io import dump, read, sha


def selected(values):
    if values == ["all"]:
        return list(range(len(config.TASKS)))
    ids = [t.id for t in config.TASKS]
    if not values or any(v not in ids for v in values):
        raise ValueError("Unknown task ID; use list-tasks")
    return list(dict.fromkeys(ids.index(v) for v in values))


def parser():
    p = argparse.ArgumentParser(
        description="GEMS-PDE 42 + complete local correction; adapted LLM-PDESR baseline"
    )
    sub = p.add_subparsers(dest="command", required=True)
    for name, helptext in [
        ("search", "LLM initialization + persistent archive search"),
        ("baseline", "100 LLM equation returns per task"),
        ("refine", "Zero-API complete local joint-fit correction"),
        ("run", "Search, correction and baseline; performs paid API calls unless --offline"),
        ("evaluate", "Post-freeze structural comparison"),
        ("doctor", "Check local dependencies/data without API calls"),
        ("audit-published", "Verify packaged results and re-evaluate without API calls"),
        ("usage", "Summarize saved API usage"),
    ]:
        q = sub.add_parser(name, help=helptext)
        q.add_argument("--run-dir", type=Path, default=Path("runs/main"))
        q.add_argument("--tasks-file", type=Path, default=None)
        q.add_argument("--tasks", nargs="+", default=["all"], help="Task IDs, e.g. T01 T04, or all")
        q.add_argument(
            "--offline",
            action="store_true",
            help="Never issue HTTP requests; require exact cached requests",
        )
        if name == "refine":
            q.add_argument(
                "--source-dir",
                type=Path,
                help="Directory containing Txx archives/graphs; default RUN/proposed42",
            )
        if name == "evaluate":
            q.add_argument("--references", type=Path, required=True)
            q.add_argument("--output-dir", type=Path, default=Path("runs/evaluation"))
        if name == "audit-published":
            q.add_argument("--output-dir", type=Path, default=Path("runs/published_audit"))
            q.add_argument(
                "--check-mse",
                action="store_true",
                help="Recalculate saved-equation residuals without refitting",
            )
    sub.add_parser("list-tasks", help="Show task IDs, data and prior file names")
    return p


def main(argv=None):
    opts = parser().parse_args(argv)
    if opts.command == "list-tasks":
        for t in config.TASKS:
            print(t.id, t.name)
        return
    configure_run = (
        opts.output_dir if opts.command in ("evaluate", "audit-published") else opts.run_dir
    )
    config.configure(configure_run, opts.tasks_file, opts.offline)
    indices = selected(opts.tasks)
    if opts.command == "doctor":
        from .observations import Data

        details = []
        for i in indices:
            t = config.TASKS[i]
            config.prior_for(i)
            d = Data(t.name)
            details.append(
                {
                    "task": t.id,
                    "fields": d.fields,
                    "shape": list(d.full["u"].shape),
                    "train_residuals_per_output": int(d.trainmask.sum()),
                    "validation_residuals_per_output": int(d.valmask.sum()),
                }
            )
        print(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "versions": {
                        v: importlib.metadata.version(v)
                        for v in ("numpy", "scipy", "sympy", "requests")
                    },
                    "API_calls": 0,
                    "tasks": details,
                },
                indent=2,
            )
        )
        return
    if opts.command in ("evaluate", "audit-published", "usage"):
        from .evaluation import evaluate, check_recorded_mse, usage_summary

        if opts.command == "usage":
            print(json.dumps(usage_summary(opts.run_dir), indent=2))
            return
        if opts.command == "audit-published":
            root = config.PROJECT
            manifest = read(root / "artifacts" / "release_manifest.json")
            checksums = {**manifest["sha256"], **manifest["source_sha256"]}
            for rel, expected in checksums.items():
                if sha(root / rel) != expected:
                    raise RuntimeError("Packaged artifact changed: " + rel)
            run_dir = root / "artifacts" / "published"
            references = root / "artifacts" / "references.json"
            checks = check_recorded_mse(run_dir, indices) if opts.check_mse else []
            dump(opts.output_dir / "mse_checks.json", checks)
        else:
            run_dir = opts.run_dir
            references = opts.references
        result = evaluate(run_dir, references, opts.output_dir, indices)
        dump(opts.output_dir / "usage.json", usage_summary(run_dir))
        print(json.dumps(result["summary"], indent=2))
        return
    if opts.command in ("search", "run"):
        from . import search

        for i in indices:
            search.run(i)
    if opts.command in ("refine", "run"):
        from . import refine

        source = getattr(opts, "source_dir", None)
        for i in indices:
            refine.run(i, source)
    if opts.command in ("baseline", "run"):
        from . import baseline

        for i in indices:
            baseline.run(i)


if __name__ == "__main__":
    main()
