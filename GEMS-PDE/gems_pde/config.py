"""Shared historical settings and task paths. No reference-equation access."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent
RUN_DIR = Path.cwd() / "runs" / "main"
OFFLINE = False
TASK_FILE = PACKAGE / "resources" / "tasks.json"
SETTINGS = {
    "model": "deepseek-v4-flash",
    "temperature": 0.45,
    "max_tokens": 6500,
    "thinking": "disabled",
    "train_times": 80,
    "validation_times": 32,
    "windows": 10,
    "max_terms": 8,
    "max_derivative_order": 4,
    "max_nfev_full": 45,
    "max_nfev_screen": 12,
    "multistart_finalists": 2,
    "lambda": 0.10,
    "coarse_parameters": [0.25, 0.5, 1.0, 2.0],
    "baseline_samples_per_task": 100,
    "proposed_decisions": 12,
    "archive_prompt_states": 24,
    "parent_expansions_per_round": 2,
    "edits_per_parent_cap": 20,
    "full_refinements_per_round": 6,
    "final_refinements": 8,
    "seed": 4200,
    "complexity": "2*D calls + arithmetic (power=2) + unary function calls + distinct free parameters; outer multiplications and between-basis additions counted after normalization",
    "reference_access_during_search": False,
}


@dataclass(frozen=True)
class Task:
    id: str
    name: str
    data_path: Path
    prior_path: Path


def load_tasks(path):
    path = Path(path).resolve()
    entries = json.loads(path.read_text(encoding="utf-8"))
    tasks = []
    for entry in entries:
        tid = entry["id"]
        if (
            not isinstance(tid, str)
            or not tid
            or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in tid
            )
        ):
            raise ValueError("Task IDs must be simple directory names")
        tasks.append(
            Task(
                tid,
                entry["name"],
                (path.parent / entry["data"]).resolve(),
                (path.parent / entry["prior"]).resolve(),
            )
        )
    if (
        not tasks
        or len({t.id for t in tasks}) != len(tasks)
        or len({t.name for t in tasks}) != len(tasks)
    ):
        raise ValueError("Task IDs and names must be nonempty and unique")
    return tasks


TASKS = load_tasks(TASK_FILE)


def endpoint():
    base = os.environ.get("PDE_API_BASE_URL", "https://api.teamorouter.cn/v1").rstrip("/")
    url = urlsplit(base)
    if (
        url.scheme not in ("https", "http")
        or not url.netloc
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("PDE_API_BASE_URL must be a plain HTTP(S) URL without credentials/query")
    return base + "/chat/completions"


def names():
    return [t.name for t in TASKS]


def task_by_name(name):
    return next(t for t in TASKS if t.name == name)


def prior_for(index):
    text = TASKS[index].prior_path.read_text(encoding="utf-8")
    if "[DATA]" in text:
        raise ValueError(
            "Task prior must contain only the D1b/V1 prior, not a historical full specification"
        )
    return text.strip()


def load_environment():
    """Read only this release's optional .env; never read the historical parent."""
    path = PROJECT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip().startswith("PDE_"):
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def configure(run_dir=None, tasks_file=None, offline=False):
    global RUN_DIR, TASKS, TASK_FILE, OFFLINE
    load_environment()
    if run_dir is not None:
        RUN_DIR = Path(run_dir).resolve()
    if tasks_file is not None:
        TASK_FILE = Path(tasks_file).resolve()
        TASKS = load_tasks(TASK_FILE)
    OFFLINE = offline
    SETTINGS["model"] = os.environ.get("PDE_MODEL", "deepseek-v4-flash")
    SETTINGS["thinking"] = os.environ.get("PDE_THINKING", "disabled")
    if (
        RUN_DIR == PROJECT
        or RUN_DIR.is_relative_to(PACKAGE)
        or RUN_DIR.is_relative_to(PROJECT / "artifacts")
    ):
        raise ValueError("Choose a run directory outside the source/resources/published artifacts")
    # The numerical module may already have been imported by an embedding caller.
    import sys

    common = sys.modules.get("gems_pde.common")
    if common is not None:
        common.ROOT = RUN_DIR
        common.NAMES = names()
