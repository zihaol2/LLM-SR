"""Run a declared online search, or inspect bundled data without an API request."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import numpy as np
import scipy
from . import __version__, core as C


def cases():
    return {case["id"]: case for case in C.read(C.ASSETS/"cases.json")}


def declaration(args):
    if args.case_json:
        path = Path(args.case_json).resolve()
        case = C.read(path)
        case["data_path"] = str((path.parent/case["data_path"]).resolve())
        case.setdefault("data_sha256", C.sha(case["data_path"]))
        if case.get("graph_path"):
            case["graph_path"] = str((path.parent/case["graph_path"]).resolve())
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
    C.SETTINGS['compact_prompt'] = args.compact_prompt
    C.SETTINGS['numerical_backend'] = args.numerical_backend
    C.SETTINGS['coverage_schedule'] = args.coverage_schedule
    if args.coverage_schedule=='global_1pct':
        C.SETTINGS['minimum_test_fraction']=0.1
        C.SETTINGS['sampling_schedule']='Original ten groups and two display passes; LLM selects 10% of each displayed group (~1% global) per round; prefer untested; no full-coverage catch-up'
    from .entry import validate_backend, active_source_hashes, profile_name
    validate_backend(args.numerical_backend)
    from .api import credentials
    credentials()
    from .search import run as search
    folder = Path(args.output).resolve()/case["id"]
    folder.mkdir(parents=True, exist_ok=True)
    graph = C.initial_graph(case)
    protocol = dict(version=__version__, case=case, settings=copy.deepcopy(C.SETTINGS),
        source_sha256=active_source_hashes(), source_profile=profile_name(),
        graph_sha256=hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest(),
        tokenizer_sha256=C.sha(C.ASSETS/"tokenizer.json.gz"),
        numerics=args.numerical_backend+('; fixed original features; compiled chain rules on shared observed spline jets; singular original fallback; unchanged optimizer' if args.numerical_backend=='compiled_chain_2d' else '; original quintic-spline operators and optimizer settings'), rounds=20,
        new_search_API_budget=20, bootstrap_API_budget=1, round_planning_API_budget=19, new_final_selection_API_budget=1,
        total_API_budget=21,
        initialization="Fresh empty memory; original graph/observations/prior; no historical models or reference results",
        reference_access=False, API_retry=False,
        versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__))
    path = folder/"protocol.json"
    if path.exists():
        previous=C.read(path)
        comparable={key:value for key,value in previous.items() if key!='backend_transition'}
        if comparable!=protocol:
            if not args.resume_backend_transition or case['id']!='T2D' or args.numerical_backend!='guarded_fused_2d':
                raise RuntimeError('Protocol changed; explicit audited 2D backend transition required')
            cp=C.read(folder/'checkpoint.json')
            if cp['pending'] is not None or previous['case']!=case:
                raise RuntimeError('Backend transition requires a frozen round boundary and identical observations/prior')
            for key,value in previous['settings'].items():
                if key!='numerical_backend' and value!=protocol['settings'].get(key):
                    raise RuntimeError('Unrequested search-setting change: '+key)
            for key in ['graph_sha256','tokenizer_sha256','versions']:
                if previous[key]!=protocol[key]:raise RuntimeError('Graph/tokenizer/numeric environment changed')
            for name in ['core.py','fitting.py','builder.py','grammar.py','graph.py','deduplicate.py',
                         'profile_rules.py','recombine.py','rotation.py','frontier.py','memory.py',
                         'hypothesis.py','structural_rules.py','wire.py']:
                if previous['source_sha256'].get(name)!=protocol['source_sha256'].get(name):
                    raise RuntimeError('Unrequested search/data/grammar change at backend transition: '+name)
            transition=dict(previous_version=previous['version'],new_version=__version__,
                completed_rounds=len(cp['history']),checkpoint_fit_calls=cp['counters']['fit_calls'],
                scope='Guarded 2D numerical backend; current JSON field contract; search budget and observed data preserved',
                previous_source_sha256=previous['source_sha256'],transition_unix=time.time(),
                cached_planning_reused=True)
            backup=folder/'protocol.before_guarded_fusion.json'
            if backup.exists():raise RuntimeError('Backend transition already recorded')
            C.dump(backup,previous);protocol['backend_transition']=transition
        elif 'backend_transition' in previous:
            protocol['backend_transition']=previous['backend_transition']
    C.dump(path, protocol)
    lock = folder/"run.lock"
    with lock.open("x", encoding="utf8") as file:
        json.dump(dict(pid=os.getpid(), started_unix=time.time()), file)
    try:
        result = search(case, folder)
        C.dump(folder/"execution_status.json", dict(status="complete", rounds=result["completed_rounds"],
               final_API_calls=result['final_API_calls'], terminal_fit_calls=0, reference_access=False))
    except Exception as error:
        C.dump(folder/"execution_status.json", dict(status="failed", error_type=type(error).__name__,
               message=str(error), automatic_retry=False))
        raise
    finally:
        lock.unlink(missing_ok=True)


def doctor():
    from .compression import tokens
    rows = []
    for key, case in cases().items():
        data = C.Data(case)
        graph = C.initial_graph(case)
        rows.append(dict(case=key, shape=list(data.shape), nodes=len(graph["nodes"]),
                         candidates=sum(node["candidate"] for node in graph["nodes"]), observation_hash_verified=True))
    print(json.dumps(dict(cases=rows, tokenizer_check=tokens("GEM-GSS online"), API_calls=0, fits=0), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gem-gss-online", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    commands.add_parser("doctor")
    runner = commands.add_parser("run")
    group = runner.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", choices=list(cases()))
    group.add_argument("--case-json")
    runner.add_argument("--graph")
    runner.add_argument("--output", default="runs/online")
    runner.add_argument("--model")
    runner.add_argument("--base-url")
    runner.add_argument("--history-fits", type=int, default=25)
    runner.add_argument('--compact-prompt',action='store_true',help='Moderate reversible search-prompt deduplication; final selector unchanged')
    runner.add_argument('--numerical-backend',choices=['reference','compiled_chain_2d'],default='reference')
    runner.add_argument('--resume-backend-transition',action='store_true')
    runner.add_argument('--coverage-schedule',choices=['legacy_half','global_1pct'],default='legacy_half')
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args)
    elif args.command == "doctor":
        doctor()
    else:
        print(json.dumps([{key: case[key] for key in ("id", "fields", "coordinates")} for case in cases().values()], indent=2))
