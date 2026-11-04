"""One public entry point. No API requests are made by importing this module."""

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
    return {case["id"]: case for case in C.read(C.ASSETS / "cases.json")}


def declaration(args):
    if getattr(args, "case_json", None):
        path = Path(args.case_json).resolve()
        case = C.read(path)
        data_path = Path(case["data_path"])
        case["data_path"] = str((path.parent / data_path).resolve())
        case.setdefault("data_sha256", C.sha(case["data_path"]))
        if case.get("graph_path"):
            case["graph_path"] = str((path.parent / case["graph_path"]).resolve())
    else:
        case = copy.deepcopy(cases()[args.case])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", case["id"]):
        raise ValueError("Case IDs must contain letters, digits, underscore or hyphen")
    if len(case["coordinates"]) not in (1, 2):
        raise ValueError("This release validates 1D and 2D only; 3D is not claimed")
    if getattr(args, "graph", None):
        case["graph_path"] = str(Path(args.graph).resolve())
    return case


def configure(args):
    if getattr(args, "model", None):
        C.SETTINGS["model"] = args.model
    if getattr(args, "base_url", None):
        C.SETTINGS["base_url"] = args.base_url.rstrip("/")
    from .api import credentials

    credentials()  # Fail before creating an outbound-attempt marker.


def enable_ablation(name):
    """Same sampling/input factors; all variants share the latest terminal rule."""
    if not name:
        return
    from . import compression, rotation, search

    C.SETTINGS["ablation"] = name
    drop = name in ("A", "AB")
    quarter = name in ("B", "AB")
    if quarter:
        C.SETTINGS["minimum_test_fraction"] = 0.25

        def forced_for(archive, sampled, number):
            if number <= 10:
                return []
            tested = {mid for support in archive.attempted for mid in support}
            required = int(np.ceil(len(sampled) * 0.25))
            return [mid for mid in sampled if mid not in tested][:required]

        rotation.forced_for = forced_for
    original_packet = search.graph_packet
    original_verbose = search.make_verbose_prompt
    prefix = compression.PREFIX
    if drop:
        compression.PREFIX = (
            "Sampled and active historical mechanism dictionary (neighbor rows and edges omitted): "
        )

        def packet(registry, sampled, archives, stats):
            result = original_packet(registry, sampled, archives, stats)
            keep = set(sampled) | set(result["historical_active_ids"])
            result["nodes"] = {mid: n for mid, n in result["nodes"].items() if mid in keep}
            result["dependency_edges"] = []
            result["edge_meaning"] = "No edges supplied in this ablation"
            return result

        search.graph_packet = packet

    def verbose(case, registry, archives, sampled, stats, history):
        text, packet = original_verbose(case, registry, archives, sampled, stats, history)
        if drop:
            text = text.replace(prefix, compression.PREFIX)
            text = text.replace(
                "Read the sampled mechanisms AND their graph relationships.",
                "Read the sampled mechanisms and numerical search feedback.",
            )
            text = text.replace(
                "Graph dependencies are construction relationships, not evidence that nearby nodes form a good equation.",
                "One-hop neighbors and edges are omitted. Infer meanings from the provided expressions.",
            )
            text = text.replace("construction families, topology,", "construction families,")
        if quarter:
            text += "\nBudget ablation: test one quarter of sampled roots per field. Sweep two prioritizes untested IDs only up to this quota; full coverage is not guaranteed.\n"
        return text, packet

    search.make_verbose_prompt = verbose


def run(args):
    case = declaration(args)
    configure(args)
    numerics = (
        "guarded-fused-2d"
        if len(case["coordinates"]) == 2 and not args.reference_numerics
        else "reference"
    )
    if numerics == "guarded-fused-2d":
        from .fused_numerics import install

        install()
    enable_ablation(args.ablation)
    from . import search

    folder = Path(args.output).resolve() / case["id"]
    folder.mkdir(parents=True, exist_ok=True)
    graph = C.initial_graph(case)
    protocol = dict(
        release_version=__version__,
        case=case,
        settings=copy.deepcopy(C.SETTINGS),
        source_sha256={p.name: C.sha(p) for p in sorted(C.ROOT.glob("*.py"))},
        graph_sha256=hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest(),
        rounds=20,
        new_search_API_budget=20,
        new_final_selection_API_budget=1,
        initialization="Fresh empty archives; supplied/cached graph; no published model loaded",
        numerics=numerics,
        terminal_rule="Five protected routes, deletion/polynomials/existing cores, <=10 complete systems, one final call",
        reference_access=False,
        API_retry=False,
        versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
    )
    path = folder / "protocol.json"
    if path.exists() and C.read(path) != protocol:
        raise RuntimeError("Run protocol changed. Use a separate output directory.")
    C.dump(path, protocol)
    lock = folder / "run.lock"
    with lock.open("x", encoding="utf8") as file:
        json.dump(dict(pid=os.getpid(), started_unix=time.time()), file)
    try:
        search.run(case, folder)
        if not args.search_only:
            from .terminal import run as finish

            finish(case, folder)
        C.dump(
            folder / "execution_status.json",
            dict(status="search_only" if args.search_only else "complete", numerics=numerics),
        )
    except Exception as exc:
        C.dump(
            folder / "execution_status.json",
            dict(
                status="failed",
                error_type=type(exc).__name__,
                message=str(exc),
                automatic_retry=False,
            ),
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


def build_graph(args):
    from .builder import build
    from .deduplicate import deduplicate_graph

    case = declaration(args)
    payload = (
        C.read(args.seeds) if args.seeds else C.read(C.ASSETS / "seed_proposals.json")[case["id"]]
    )
    _, raw = build(payload, case["fields"], case["coordinates"])
    graph, _ = deduplicate_graph(raw, case)
    if not all(graph["checks"].values()):
        raise RuntimeError("Graph validation failed")
    C.dump(Path(args.output), graph)
    print(json.dumps(graph["statistics"], indent=2))


def initialize(args):
    """Optional fresh seed generation: exactly one separately logged request."""
    case = declaration(args)
    configure(args)
    from .api import once

    prompt = f"""Propose a small initial mathematical mechanism library for observation-only PDE discovery.
{case['prior']}
Fields: {case['fields']}; spatial coordinates IN ORDER: {case['coordinates']}; t is time.
Metadata: {json.dumps(case['metadata'],separators=(',',':'))}
Return up to 8 basic L0 expressions and 24 composed G0 expressions. Use only declared fields/coordinates, constants, a,b,k, arithmetic, elementary functions and PD(E,alpha). Spatial order <=4. No analytic solution, unobserved field, benchmark identity or reference equation is available. A G0 expression may combine several L0 expressions. Use explicit mathematical expressions, not node IDs inside expressions. Return JSON only:
{{"L0":[{{"id":"l0_1","expr":"..."}}],"G0":[{{"id":"g0_1","expr":"...","parents":["l0_1"]}}]}}
"""
    output = Path(args.output).resolve()
    payload, record = once(prompt, output / "api", 1, phase="initialization")
    if record["status"] != "accepted" or not isinstance(payload.get("L0"), list):
        raise RuntimeError("No usable initial library; no extra API request")
    C.dump(output / "seed_proposals.json", {name: payload.get(name, []) for name in ["L0", "G0"]})


def doctor(args):
    from . import compression
    from .fitting import prediction

    published = (
        C.read(C.ASSETS / "published_records.json")["standard"] if args.verify_results else None
    )
    output = []
    for key, case in cases().items():
        data = C.Data(case)
        graph = C.initial_graph(case)
        row = dict(
            case=key,
            shape=list(data.shape),
            nodes=len(graph["nodes"]),
            candidates=sum(n["candidate"] for n in graph["nodes"]),
            observations_sha256_verified=True,
        )
        if published:
            mses = {
                f: float(np.mean((prediction(data, eq) - data.y[f]) ** 2))
                for f, eq in published[key]["equations"].items()
            }
            reference = {f: eq["MSE"] for f, eq in published[key]["equations"].items()}
            assert all(np.isclose(mses[f], reference[f], rtol=1e-6, atol=1e-24) for f in mses), (
                key,
                mses,
                reference,
            )
            row["frozen_prediction_MSE_verified"] = mses
        output.append(row)
    print(
        json.dumps(
            dict(
                cases=output,
                tokenizer_tokens=compression.tokens("GEM-GSS offline check"),
                API_calls=0,
                fits=0,
            ),
            indent=2,
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gem-gss", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List bundled cases, no network")
    check = commands.add_parser(
        "doctor", help="Check all packaged observations/graphs/tokenizer, no API"
    )
    check.add_argument(
        "--verify-results",
        action="store_true",
        help="Also recompute frozen predictions, no parameter fitting",
    )
    for name in ["run", "build-graph", "initialize"]:
        sub = commands.add_parser(name)
        group = sub.add_mutually_exclusive_group(required=True)
        group.add_argument("--case", choices=list(cases()))
        group.add_argument(
            "--case-json", help="Custom case declaration; paths are relative to the JSON file"
        )
        sub.add_argument(
            "--output",
            default=(
                "runs"
                if name == "run"
                else ("graph.json" if name == "build-graph" else "initialization")
            ),
        )
        if name in ["run", "initialize"]:
            sub.add_argument("--model")
            sub.add_argument("--base-url")
        if name == "run":
            sub.add_argument("--graph", help="Use a separately generated graph snapshot")
            sub.add_argument("--search-only", action="store_true")
            sub.add_argument(
                "--reference-numerics",
                action="store_true",
                help="Use the original 2D operators; 1D always uses them",
            )
            sub.add_argument(
                "--ablation",
                choices=["A", "B", "AB"],
                help="A drops neighbor/edge inputs; B quarters the mandatory quota",
            )
        if name == "build-graph":
            sub.add_argument(
                "--seeds", help="L0/G0 proposal JSON; default uses the original cached proposals"
            )
    evaluate = commands.add_parser(
        "evaluate", help="Post-hoc structure evaluation ONLY after final_result.json exists"
    )
    evaluate.add_argument("--case", choices=list(cases()), required=True)
    evaluate.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    if args.command == "list":
        print(
            json.dumps(
                [{k: c[k] for k in ["id", "fields", "coordinates"]} for c in cases().values()],
                indent=2,
            )
        )
    elif args.command == "doctor":
        doctor(args)
    elif args.command == "run":
        run(args)
    elif args.command == "build-graph":
        build_graph(args)
    elif args.command == "initialize":
        initialize(args)
    else:
        from .evaluation import evaluate_result

        print(
            json.dumps(evaluate_result(args.case, Path(args.result)), ensure_ascii=False, indent=2)
        )
