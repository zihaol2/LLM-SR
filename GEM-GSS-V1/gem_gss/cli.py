"""Run a declared online search, or inspect bundled data without an API request."""

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy

from . import __version__
from . import core as C


def cases():
    return {case["id"]: case for case in C.read(C.ASSETS / "cases.json")}


def declaration(args):
    if args.case_json:
        path = Path(args.case_json).resolve()
        case = C.read(path)
        case["data_path"] = str((path.parent / case["data_path"]).resolve())
        case.setdefault("data_sha256", C.sha(case["data_path"]))
        if case.get("graph_path"):
            case["graph_path"] = str((path.parent / case["graph_path"]).resolve())
    else:
        case = copy.deepcopy(cases()[args.case])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", case["id"]):
        raise ValueError("Unsafe case ID")
    if len(case["coordinates"]) not in (1, 2, 3):
        raise ValueError("This implementation supports 1D, 2D and 3D observations")
    if args.graph:
        case["graph_path"] = str(Path(args.graph).resolve())
    return case


def run(args):
    case = declaration(args)
    if args.model:
        C.SETTINGS["model"] = args.model
    if args.base_url:
        C.SETTINGS["base_url"] = args.base_url.rstrip("/")
    if args.history_fits < 0:
        raise ValueError("Fit budgets must be nonnegative")
    C.SETTINGS["history_fits_per_round"] = args.history_fits
    C.SETTINGS["compact_prompt"] = args.compact_prompt
    C.SETTINGS["numerical_backend"] = args.numerical_backend
    C.SETTINGS["coverage_schedule"] = args.coverage_schedule
    if args.coverage_schedule == "global_1pct":
        C.SETTINGS["minimum_test_fraction"] = 0.1
    else:
        C.SETTINGS["minimum_test_fraction"] = 0.5
    from .entry import active_source_hashes, validate_backend

    validate_backend(args.numerical_backend, case)
    from .api import credentials

    credentials()
    from .search import run as search

    folder = Path(args.output).resolve() / case["id"]
    folder.mkdir(parents=True, exist_ok=True)
    graph = C.initial_graph(case)
    protocol = dict(
        version=__version__,
        case=case,
        settings=copy.deepcopy(C.SETTINGS),
        source_sha256=active_source_hashes(),
        source_profile="pde049",
        graph_sha256=hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest(),
        tokenizer_sha256=C.sha(C.ASSETS / "tokenizer.json.gz"),
        numerics=args.numerical_backend
        + (
            "; fixed original features; compiled chain rules on shared observed spline jets; singular original fallback; unchanged optimizer"
            if args.numerical_backend == "compiled_chain_2d"
            else "; original quintic-spline operators and optimizer settings"
        ),
        rounds=C.SETTINGS["rounds"],
        new_search_API_budget=C.SETTINGS["rounds"],
        bootstrap_API_budget=1,
        round_planning_API_budget=C.SETTINGS["rounds"] - 1,
        new_final_selection_API_budget=1,
        total_API_budget=C.SETTINGS["rounds"] + 1,
        initialization="Fresh empty memory; original graph/observations/prior; no historical models or reference results",
        reference_access=False,
        API_retry=False,
        versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
    )
    path = folder / "protocol.json"
    if path.exists() and C.read(path) != protocol:
        raise RuntimeError("Run protocol changed; use a new output directory")
    C.dump(path, protocol)
    lock = folder / "run.lock"
    with lock.open("x", encoding="utf8") as file:
        json.dump(dict(pid=os.getpid(), started_unix=time.time()), file)
    try:
        result = search(case, folder)
        C.dump(
            folder / "execution_status.json",
            dict(
                status="complete",
                rounds=result["completed_rounds"],
                final_API_calls=result["final_API_calls"],
                terminal_fit_calls=0,
                reference_access=False,
            ),
        )
    except Exception as error:
        C.dump(
            folder / "execution_status.json",
            dict(
                status="failed",
                error_type=type(error).__name__,
                message=str(error),
                automatic_retry=False,
            ),
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gem-gss-pde", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data-dir",
        default=argparse.SUPPRESS,
        help="Observation directory (or GEM_GSS_DATA_DIR)",
    )
    parser.add_argument("--data-dir", default=os.environ.get("GEM_GSS_DATA_DIR", "data"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", parents=[common])
    checker = commands.add_parser("doctor", parents=[common])
    checker.add_argument("--case", choices=list(cases()))
    commands.add_parser("check")
    runner = commands.add_parser("run", parents=[common])
    group = runner.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", choices=list(cases()))
    group.add_argument("--case-json")
    runner.add_argument("--graph")
    runner.add_argument("--output", default="runs/online")
    runner.add_argument("--model")
    runner.add_argument("--base-url")
    runner.add_argument("--history-fits", type=int, default=25)
    runner.add_argument(
        "--compact-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lossless prompt compression",
    )
    runner.add_argument(
        "--numerical-backend",
        choices=["reference", "compiled_chain_2d"],
        default="reference",
    )
    runner.add_argument(
        "--coverage-schedule",
        choices=["legacy_half", "global_1pct"],
        default="global_1pct",
    )
    args = parser.parse_args(argv)
    from .entry import check_release, configure, doctor

    configure(args.data_dir)
    if args.command == "check":
        check_release()
    elif args.command == "run":
        run(args)
    elif args.command == "doctor":
        doctor(C, args.case)
    else:
        print(
            json.dumps(
                [
                    {key: case[key] for key in ("id", "fields", "coordinates")}
                    for case in cases().values()
                ],
                indent=2,
            )
        )
