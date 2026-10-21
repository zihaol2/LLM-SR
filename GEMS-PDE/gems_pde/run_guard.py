"""Bind resumable stages to their actual settings, data, priors, and source."""

import platform
import importlib.metadata
from . import config
from .io import dump, read, sha


def guard_task(stage, index, source=None):
    task = config.TASKS[index]
    fixed = {
        "schema": 1,
        "stage": stage,
        "task_id": task.id,
        "task_name": task.name,
        "settings": config.SETTINGS.copy(),
        "api_endpoint": config.endpoint(),
        "data_sha256": sha(task.data_path),
        "prior_sha256": sha(task.prior_path),
        "code_sha256": {
            p.relative_to(config.PACKAGE).as_posix(): sha(p)
            for p in sorted(config.PACKAGE.rglob("*.py"))
        },
        "runtime": {
            "python": platform.python_version(),
            **{k: importlib.metadata.version(k) for k in ("numpy", "scipy", "sympy", "requests")},
        },
    }
    if source is not None:
        fixed["source_sha256"] = {
            name: sha(source / name)
            for name in ("frozen_result.json", "constructible_graph.json", "equation_archive.json")
        }
    folder = config.RUN_DIR / stage / task.id
    path = folder / "run_protocol.json"
    if path.exists():
        if read(path) != fixed:
            raise RuntimeError(
                f"Run inputs/settings/source changed for {stage}/{task.id}; use a new --run-dir"
            )
    else:
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError(
                f"Refusing to adopt unguarded existing output {stage}/{task.id}; use a new --run-dir"
            )
        dump(path, fixed)
    return fixed
