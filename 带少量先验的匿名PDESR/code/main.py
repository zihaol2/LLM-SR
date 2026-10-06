"""Semi-anonymous physics-guided PDESR, end to end.

The structural prior in the spec docstring is global: the library design and
the search see exactly the same text, and only the subject-area nouns are
withheld. The designed library is GUIDANCE for a free-numpy search (not an
API), and after the search the field is named from the statistics-based prior
reading plus the best equation with its fitted coefficients and its classical
parent PDEs.
"""
from __future__ import annotations
import json
import os
from argparse import ArgumentParser, ArgumentTypeError
from typing import Any, Sequence, Tuple

import numpy as np

import core
import evaluator
import library
import roles
import sampler


# ==========================================================================
# the search loop
# ==========================================================================



# from collections.abc import Sequence



def _extract_function_names(specification: str) -> Tuple[str, str]:

    run_functions = list(core.yield_decorated(specification, 'evaluate', 'run'))
    if len(run_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@evaluate.run`.')
    evolve_functions = list(core.yield_decorated(specification, 'equation', 'evolve'))
    
    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@equation.evolve`.')
    
    return evolve_functions[0], run_functions[0]



def run_search(
        specification: str,
        inputs: Sequence[Any],
        config: core.Config,
        max_sample_nums: int | None,
        class_config: core.ClassConfig,
        mechanism_text: str = "",
        log_dir: str | None = None,
):


    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = core.text_to_program(specification)
    database = core.ExperienceBuffer(config.experience_buffer, template, function_to_evolve,
                                     mechanism_text=mechanism_text,
                                     prompt_dump_path=(os.path.join(log_dir, "search_prompt.txt")
                                                       if log_dir else None))

    if log_dir is None:
        profiler = None
    else:
        profiler = core.Profiler(log_dir)


    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(evaluator.Evaluator(
            database,
            template,
            function_to_evolve,
            function_to_run,
            inputs,
            timeout_seconds=config.evaluate_timeout_seconds,
            sandbox_class=class_config.sandbox_class,
        ))

    initial = template.get_function(function_to_evolve).body
    evaluators[0].analyse(initial, island_id=None, version_generated=None, profiler=profiler)


    samplers = [sampler.Sampler(database, evaluators, 
                                config.samples_per_prompt, 
                                max_sample_nums=max_sample_nums, 
                                llm_class=class_config.llm_class,
                                config = config) 
                                for _ in range(config.num_samplers)]

    for s in samplers:
        s.sample(profiler=profiler)

    print("\n" + "=" * 50)
    print("🎉 Evolution Finished! Final Pareto Frontier:")

    final_front_data = []

    for idx, p in enumerate(database.pareto_front):
        print(f"[{idx + 1}] MSE: {p.mse:.2e} | Complexity: {p.complexity}")
        print(f"{p.body.strip()}\n")

        final_front_data.append({
            "rank": idx + 1,
            "mse": p.mse,
            "complexity": p.complexity,
            "params": getattr(p, "params", None),
            "equation": p.body.strip()
        })
    print("=" * 50)

    if log_dir:
        pareto_path = os.path.join(log_dir, "final_pareto_front.json")
        with open(pareto_path, 'w', encoding='utf-8') as f:
            json.dump(final_front_data, f, indent=4)
        print(f"✅ Final Pareto front successfully saved to: {pareto_path}")



# ==========================================================================
# the command line and the design / role phases
# ==========================================================================

def _bool_arg(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ArgumentTypeError(f"invalid boolean value: {value!r}")


def ask_json(call, prompt: str, parse, label: str, attempts: int = 3):
    """Ask until the reply parses, or give up loudly with the raw reply printed."""
    for attempt in range(1, attempts + 1):
        text = call(prompt)
        payload = parse(text)
        if payload:
            return payload
        print(f'  [{label}] attempt {attempt}/{attempts}: reply was not the requested '
              f'JSON, retrying', flush=True)
        if attempt == 1:
            print(f'  [{label}] raw reply started: '
                  f"{' '.join((text or '').split())[:200]!r}", flush=True)
        prompt = prompt + (
            "\n\n### RETRY\nYour previous reply was not the JSON requested. Reply "
            "again with ONLY that JSON value: no introduction, no commentary, no "
            "markdown fences.")
    return {}


parser = ArgumentParser()
parser.add_argument('--use_api', type=_bool_arg, default=False)
parser.add_argument('--api_model', type=str, default="deepseek-v4-flash")
parser.add_argument('--api_base_url', type=str, default="https://api.teamorouter.cn/v1")
parser.add_argument('--api_key_env', type=str, default="TEAMOROUTER_API_KEY")
parser.add_argument('--thinking', type=str, default="disabled")
parser.add_argument('--reasoning_effort', type=str, default="")
parser.add_argument('--max_tokens', type=int, default=16384)
parser.add_argument('--concurrent', type=int, default=4)
parser.add_argument('--spec_path', type=str, required=True)
parser.add_argument('--data_path', type=str, default="./traffic_flow_bottleneck.npz",
                    help="the measured dataset (.npz with x1/f1/f1_t_true/h1)")
parser.add_argument('--log_path', type=str, required=True)
parser.add_argument('--problem_name', type=str, default="Traffic_Flow_Bottleneck")
parser.add_argument('--max_samples', type=int, default=100)
parser.add_argument('--evaluate_timeout', type=int, default=60)
parser.add_argument('--max_order', type=int, default=2,
                    help="highest differential order the guidance allows the search")
# the LLM designs this problem's mechanism library first
parser.add_argument('--design_library', type=int, default=0)
parser.add_argument('--design_mechanisms', type=int, default=10)
parser.add_argument('--design_prior', type=int, default=1,
                    help="configuration B for the semi-anonymous arm: the short prior "
                         "in the spec docstring is GLOBAL, so the statistics plan and "
                         "the library design see it too (0 = statistics-only, as 15)")
parser.add_argument('--design_only', type=int, default=0,
                    help="stop after the measurement/role/library design phase, without "
                         "spending samples on the search")
parser.add_argument('--design_stats', type=int, default=0,
                    help="0 = the library is designed from the PROVIDED PRIOR alone "
                         "(the default here); 1 = the design also sees the measured "
                         "statistics")
parser.add_argument('--role_stats', type=int, default=1,
                    help="when the library is prior-only, still collect the statistics "
                         "and read Evidence 1 off them (they are never shown to the "
                         "library-design step)")
parser.add_argument('--design_from', type=str, default="",
                    help="reuse a saved designed_library.json instead of designing the "
                         "library again; the search then runs on exactly that library")
parser.add_argument('--stats_ops', type=int, default=14,
                    help="upper bound on how many statistics the model may request")
# role identification: name the anonymous field from prior + posterior evidence
parser.add_argument('--role_identify', type=int, default=1,
                    help="after the search, name the field from the statistics-based "
                         "prior reading plus the best equation and its classical parents")
parser.add_argument('--n_parents', type=int, default=6,
                    help="how many candidate classical parent PDEs to match against; "
                         "6 rather than 3 because a purely mathematical framing tends "
                         "to crowd out the application-level family that fits")
args = parser.parse_args()


if __name__ == '__main__':
    class_config = core.ClassConfig(llm_class=sampler.LocalLLM,
                                    sandbox_class=evaluator.LocalSandbox)
    cfg = core.Config(use_api=args.use_api,
                      api_model=args.api_model,
                      max_tokens=args.max_tokens,
                      reasoning_effort=args.reasoning_effort,
                      thinking=args.thinking,
                      api_base_url=args.api_base_url,
                      api_key_env=args.api_key_env,
                      concurrent_requests=args.concurrent,
                      evaluate_timeout_seconds=args.evaluate_timeout)

    with open(args.spec_path, encoding="utf-8") as handle:
        specification = handle.read()

    # The semi-anonymous prior lives in the spec docstring. Under configuration B it
    # is GLOBAL: the statistics plan, the library design and the search prompt all see
    # exactly the same text. --design_prior 0 reproduces the statistics-only flow.
    prior_text = ""
    if args.design_prior:
        import re as _re
        _docstring = _re.search(r'"""(.*?)"""', specification, _re.S)
        if _docstring:
            prior_text = _docstring.group(1).strip()
        if prior_text:
            print(f'configuration B: the spec prior is global ({len(prior_text)} chars) '
                  f'-> plan + library design + search', flush=True)
    with np.load(args.data_path) as archive:
        data = {key: archive[key] for key in archive.files}

    n_max = args.max_order
    mechanism_text = ""
    os.makedirs(args.log_path, exist_ok=True)

    design_text = ""
    design = {}
    specs = []
    dataset_description = ""
    if args.design_from:
        with open(args.design_from, encoding="utf-8") as handle:
            payload = json.load(handle)
        # Keep a copy next to this run: role identification reads the prior reading
        # and the mechanism list from <log_path>/designed_library.json, so a reused
        # design must land there too or Evidence 1 comes out empty.
        with open(os.path.join(args.log_path, "designed_library.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
        design = payload.get("design", {})
        specs = payload.get("mechanisms", [])
        for spec in specs:
            spec.setdefault("provenance", {"by": "llm-design"})
        design_text = library.render(design, specs)
        dataset_description = payload.get("dataset_description", "")
        prior_text = payload.get("prior_text", prior_text)
        print(f'reused design: {len(specs)} mechanism(s) from {args.design_from}',
              flush=True)
    elif args.design_library:
        axes = sorted([k for k in data if k.startswith('x') and k[1:].isdigit()],
                      key=lambda s: int(s[1:]))
        fields = sorted([k for k in data if k.startswith('f') and k[1:].isdigit()
                         and not k.endswith('_t_true')],
                        key=lambda s: int(s[1:]))
        targets = [k for k in data if k.endswith('_t_true')]
        n_axis = len(axes)
        lines = [
            f"structure: {n_axis} spatial coordinate(s): {', '.join(axes)}; "
            f"1 time coordinate: t"
            + ("" if n_axis != 1 else " (there is no x2 and no x3)"),
            "a field is indexed as (space, time)",
            "arrays:",
        ]
        for axis in axes:
            g = np.asarray(data[axis], dtype=float)
            lines.append(f"  {axis}: shape {np.shape(g)}, range "
                         f"[{float(np.min(g)):+.4f}, {float(np.max(g)):+.4f}], "
                         f"spacing {float(g[1] - g[0]):.6f}")
        if 't' in data:
            tt = np.asarray(data['t'], dtype=float)
            lines.append(f"  t: shape {np.shape(tt)}, range "
                         f"[{float(np.min(tt)):+.4f}, {float(np.max(tt)):+.4f}], "
                         f"spacing {float(tt[1] - tt[0]):.6f}")
        for f in fields:
            v = np.asarray(data[f], dtype=float)
            lines.append(f"  {f}: shape {np.shape(v)}, range "
                         f"[{float(np.min(v)):+.4f}, {float(np.max(v)):+.4f}], "
                         f"mean {float(np.mean(v)):+.4f}")
        for target in targets:
            lines.append(f"  {target}: shape {np.shape(data[target])}, "
                         f"described only as the measured time derivative of "
                         f"{target[:-len('_t_true')]}")
        if prior_text:
            lines.append("No physical meaning is attached to the arrays themselves; "
                         "the experimenter supplies one short structural prior "
                         "separately (see PROVIDED PRIOR).")
        else:
            lines.append("No physical meaning is attached to any of these arrays.")
        lines.append("Every statement about the structure of the data must agree with "
                     "the dimensions and shapes listed above. Do not assume any axis "
                     "that is not listed.")
        dataset_description = "\n".join(lines)
        domain_text = (
            f"axes: {', '.join(axes)}   fields: {', '.join(fields)}   "
            f"targets: {', '.join(targets)}\n"
            + "\n".join(
                f"  {f}: shape {np.shape(data[f])}, range "
                f"[{float(np.min(data[f])):+.3f}, {float(np.max(data[f])):+.3f}]"
                for f in fields))
        design_llm = sampler.LocalLLM(1, trim=False)
        design_llm._instruction_prompt = ""
        raw_log = {}

        def _design_call(prompt_text: str) -> str:
            out = design_llm._draw_samples_api(prompt_text, cfg)
            text = out[0] if out else ""
            raw_log["last"] = text
            return text

        plan, report, used, stats_report = [], [], [], ""
        stats_roles = {}
        if args.design_stats:
            print('step 1: the model decides which statistics to compute ...', flush=True)
            plan = library.parse_plan(
                _design_call(library.plan_prompt(dataset_description, args.stats_ops,
                                                        prior_text)))
            print(f'  planned {len(plan)} statistic(s)', flush=True)
            report, used = library.execute(plan, data)
            stats_report = "\n".join(report)
            with open(os.path.join(args.log_path, "statistics_plan.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"plan": plan, "used": used, "report": report}, handle,
                          ensure_ascii=False, indent=1)
            for line in report[:8]:
                print(f'  {line}', flush=True)
            print('step 2: infer the roles and design the library (prior first, then the '
                  'measured statistics) ...', flush=True)
            design_prompt_text = library.design_prompt_from_stats(
                stats_report, n_max, args.design_mechanisms, prior_text)
        else:
            if not prior_text:
                raise SystemExit(
                    '--design_stats 0 means "design the library from the prior alone", '
                    'but no prior was found in the spec docstring')
            if args.role_stats:
                # The library stays prior-only, but Evidence 1 of the role
                # identification is a reading of the MEASUREMENTS, so the statistics
                # are still collected here -- they are simply never shown to the
                # library-design step.
                print('step 1: measuring the statistics that describe the data '
                      '(they feed the ROLE reading only, not the library) ...',
                      flush=True)
                plan = library.parse_plan(
                    _design_call(library.plan_prompt(
                        dataset_description, args.stats_ops, prior_text)))
                print(f'  planned {len(plan)} statistic(s)', flush=True)
                report, used = library.execute(plan, data)
                stats_report = "\n".join(report)
                with open(os.path.join(args.log_path, "statistics_plan.json"), "w",
                          encoding="utf-8") as handle:
                    json.dump({"plan": plan, "used": used, "report": report}, handle,
                              ensure_ascii=False, indent=1)
                for line in report[:8]:
                    print(f'  {line}', flush=True)
                print('step 1b: reading the macro role off those measurements ...',
                      flush=True)
                stats_roles = ask_json(
                    _design_call, library.role_only_prompt(stats_report),
                    library.parse_design, "role")
                claim = ((stats_roles.get("inferred_roles") or {}).get("f1") or {})
                if isinstance(claim, dict) and claim.get("identity"):
                    print(f'  role of f1 (from measurements only): {claim["identity"]} '
                          f'({claim.get("confidence")})', flush=True)
            else:
                print('step 1: skipped -- prior-only library design, no statistic is '
                      'measured', flush=True)
            print('step 2: design the library from the PROVIDED PRIOR alone ...',
                  flush=True)
            design_prompt_text = library.design_prompt(
                prior_text, domain_text, n_max, args.design_mechanisms)

        design, specs, _log = library.design_from_prompt(
            _design_call, design_prompt_text, data, n_max)
        # Keep the model's verbatim reply next to the run: without it the prompt
        # cannot be debugged against what the model actually chose.
        with open(os.path.join(args.log_path, "design_prompt.txt"), "w",
                  encoding="utf-8") as handle:
            handle.write(design_prompt_text)
        with open(os.path.join(args.log_path, "design_reply_raw.txt"), "w",
                  encoding="utf-8") as handle:
            handle.write(raw_log.get("last", ""))
        for field_name, claim in (design.get("inferred_roles") or {}).items():
            if isinstance(claim, dict):
                print(f'  inferred {field_name}: {claim.get("identity")} '
                      f'({claim.get("confidence")})', flush=True)
        for spec in specs:
            spec["provenance"] = {"by": "llm-design"}
        with open(os.path.join(args.log_path, "designed_library.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"dataset_description": dataset_description,
                       "prior_text": prior_text,
                       "prior_visible_everywhere": bool(args.design_prior and prior_text),
                       "design_mode": ("prior_first_then_stats" if args.design_stats
                                       else "prior_only"),
                       "statistics_used_for_library": bool(args.design_stats),
                       "role_reading_from": ("design" if design.get("inferred_roles")
                                             else ("measurements"
                                                   if stats_roles.get("inferred_roles")
                                                   else "none")),
                       "statistics_report": report,
                       "inferred_roles": (design.get("inferred_roles")
                                          or stats_roles.get("inferred_roles")),
                       "design": design, "mechanisms": specs}, handle,
                      ensure_ascii=False, indent=1)
        design_text = library.render(design, specs)
        print(f'designed library: {len(specs)} mechanism(s) accepted', flush=True)

        if args.design_only:
            print('\n=== design phase only: stopping before the search ===', flush=True)
            print(f"prior_visible_everywhere: {bool(args.design_prior and prior_text)}")
            print(f"role      : {json.dumps(design.get('inferred_roles'), ensure_ascii=False)[:400]}")
            print(f"library   : {[s['name'] for s in specs]}")
            print(f"written   : {os.path.join(args.log_path, 'designed_library.json')}")
            raise SystemExit(0)

    if design_text:
        # The library is GUIDANCE, not an API: the search writes numpy itself. Each
        # entry is shown by its definition, in priority order, so the model can see
        # which structures this problem is expected to be built from.
        mechanism_text = library.render_guidance(design, specs)
        mechanism_text += (
            f"\n\nHighest differential order allowed: {n_max}."
            f"\nWrite the equation yourself in numpy: d_axis(field, axis, order) for "
            f"derivatives (axis=0 is x1), np.sin / np.cos / np.tanh / np.exp for "
            f"mathematical functions, and params[0..4] for the coefficients. The "
            f"guidance above says which structures to try and in what order; it is "
            f"not an API and you are free to combine or modify it.")
        print(f'library guidance: {len(specs)} mechanism(s)', flush=True)

    run_search(
        specification=specification,
        inputs={'data': data},
        config=cfg,
        max_sample_nums=args.max_samples,
        class_config=class_config,
        log_dir=args.log_path,
        mechanism_text=mechanism_text,
    )

    if args.role_identify:
        role_llm = sampler.LocalLLM(1, trim=False)
        role_llm._instruction_prompt = ""

        def _role_call(prompt_text: str) -> str:
            out = role_llm._draw_samples_api(prompt_text, cfg)
            return out[0] if out else ""

        print('step 6: identifying the variable role (prior reading + best equation '
              '+ classical parents) ...', flush=True)
        try:
            roles.run_on_results(args.log_path, _role_call, args.n_parents)
        except SystemExit as error:
            print(f'  role identification skipped: {error}', flush=True)
