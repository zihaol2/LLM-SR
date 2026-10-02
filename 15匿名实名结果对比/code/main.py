"""Physics-guided PDESR: the original search plus a growing mechanism library.

The original PDESR pipeline (sampler / evaluator / Pareto buffer) is unchanged. On
top of it there is exactly one addition: the operator grammar and the LLM expert
loop that can add new mechanisms to it. Nothing else.

    python main.py --problem_name Traffic_Flow_Bottleneck \
        --spec_path specs/specification_Traffic_Flow_Bottleneck_numpy.txt \
        --log_path results/Traffic_Flow_Bottleneck --use_api True \
        --api_model deepseek-v4-flash --api_base_url https://api.teamorouter.cn/v1 \
        --api_key_env TEAMOROUTER_API_KEY --max_samples 100 \
        --grow_mechanisms 1
"""
import json
import os
from argparse import ArgumentParser, ArgumentTypeError

import numpy as np

import pipeline
import config
import sampler
import evaluator
import grammar_check


def _bool_arg(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ArgumentTypeError(f"invalid boolean value: {value!r}")


parser = ArgumentParser()
parser.add_argument('--use_api', type=_bool_arg, default=False)
parser.add_argument('--api_model', type=str, default="deepseek-v4-flash")
parser.add_argument('--api_base_url', type=str, default="https://api.teamorouter.cn/v1")
parser.add_argument('--api_key_env', type=str, default="TEAMOROUTER_API_KEY")
parser.add_argument('--thinking', type=str, default="disabled")
parser.add_argument('--reasoning_effort', type=str, default="")
parser.add_argument('--max_tokens', type=int, default=8192)
parser.add_argument('--concurrent', type=int, default=4)
parser.add_argument('--spec_path', type=str, required=True)
parser.add_argument('--data_path', type=str, default="./traffic_flow_bottleneck.npz",
                    help="the measured dataset (.npz with x1/f1/f1_t_true/h1)")
parser.add_argument('--log_path', type=str, required=True)
parser.add_argument('--problem_name', type=str, default="Traffic_Flow_Bottleneck")
parser.add_argument('--max_samples', type=int, default=100)
parser.add_argument('--evaluate_timeout', type=int, default=60)
parser.add_argument('--max_order', type=int, default=2,
                    help="highest differential order the grammar prices without penalty")
parser.add_argument('--penalty', type=int, default=3)
# the one addition: the LLM expert that grows the mechanism library
parser.add_argument('--grow_mechanisms', type=int, default=0)
parser.add_argument('--grow_rounds', type=int, default=1)
parser.add_argument('--grow_proposals', type=int, default=3)
parser.add_argument('--registry_path', type=str, default="")
parser.add_argument('--promote_threshold', type=float, default=3.0)
# configuration B: the LLM designs this problem's mechanism library first
parser.add_argument('--design_library', type=int, default=0)
parser.add_argument('--design_mechanisms', type=int, default=10)
parser.add_argument('--stats_ops', type=int, default=14,
                    help="upper bound on how many statistics the model may request")
args = parser.parse_args()


if __name__ == '__main__':
    class_config = config.ClassConfig(llm_class=sampler.LocalLLM,
                                      sandbox_class=evaluator.LocalSandbox)
    cfg = config.Config(use_api=args.use_api,
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
    with np.load(args.data_path) as archive:
        data = {key: archive[key] for key in archive.files}

    axis_count = len([k for k in data
                      if k.startswith('x') and k[1:].isdigit()])
    ctx = grammar_check.GrammarContext(n_max=args.max_order, penalty=args.penalty,
                                       axis_count=max(1, axis_count))

    mechanism_text = ""
    registry_path = os.path.abspath(
        args.registry_path or os.path.join(args.log_path, "mechanism_registry.json"))
    os.environ["PDE_GRAMMAR_REGISTRY"] = registry_path
    os.makedirs(args.log_path, exist_ok=True)

    import mechanism_registry
    registry = mechanism_registry.Registry.load(registry_path)
    reinstated = mechanism_registry.register_into_grammar(registry)
    if reinstated:
        print(f'registry: reinstated {len(reinstated)} mechanism(s)', flush=True)

    design_text = ""
    if args.design_library:
        import library_design
        import stat_engine
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
        lines.append("No physical meaning is attached to any of these arrays.")
        lines.append("Every statement about the structure of the data must agree with "
                     "the dimensions and shapes listed above. Do not assume any axis "
                     "that is not listed.")
        dataset_description = "\n".join(lines)
        design_llm = sampler.LocalLLM(1, trim=False)
        design_llm._instruction_prompt = ""

        def _design_call(prompt_text: str) -> str:
            out = design_llm._draw_samples_api(prompt_text, cfg)
            return out[0] if out else ""

        print('step 1: the model decides which statistics to compute ...', flush=True)
        plan = library_design.parse_plan(
            _design_call(library_design.plan_prompt(dataset_description, args.stats_ops)))
        print(f'  planned {len(plan)} statistic(s)', flush=True)
        report, used = stat_engine.execute(plan, data)
        stats_report = "\n".join(report)
        with open(os.path.join(args.log_path, "statistics_plan.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"plan": plan, "used": used, "report": report}, handle,
                      ensure_ascii=False, indent=1)
        for line in report[:8]:
            print(f'  {line}', flush=True)

        print('step 2: infer the roles and design the library ...', flush=True)
        design, specs, _log = library_design.design_from_prompt(
            _design_call,
            library_design.design_prompt_from_stats(stats_report, ctx.n_max,
                                                    args.design_mechanisms),
            data, ctx.n_max)
        for field_name, claim in (design.get("inferred_roles") or {}).items():
            if isinstance(claim, dict):
                print(f'  inferred {field_name}: {claim.get("identity")} '
                      f'({claim.get("confidence")})', flush=True)
        for spec in specs:
            spec["provenance"] = {"by": "llm-design"}
            registry.add(spec)
        with open(os.path.join(args.log_path, "designed_library.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"dataset_description": dataset_description,
                       "statistics_report": report,
                       "inferred_roles": design.get("inferred_roles"),
                       "design": design, "mechanisms": specs}, handle,
                      ensure_ascii=False, indent=1)
        registry.save(registry_path)
        design_text = library_design.render(design, specs)
        print(f'designed library: {len(specs)} mechanism(s) accepted', flush=True)

    if args.grow_mechanisms:
        import mechanism_expert
        expert_llm = sampler.LocalLLM(1, trim=False)
        expert_llm._instruction_prompt = ""

        def _expert_call(prompt_text: str) -> str:
            out = expert_llm._draw_samples_api(prompt_text, cfg)
            return out[0] if out else ""

        print('running the mechanism-expert loop ...', flush=True)
        promoted, _log = mechanism_expert.grow(
            registry, data, _expert_call, rounds=args.grow_rounds,
            proposals=args.grow_proposals, threshold=args.promote_threshold)
        registry.save(registry_path)

    if design_text or registry.grown():
        mechanism_text = (design_text + "\n\n" + registry.render_prompt()).strip()
        mechanism_text += f"\n\nHighest differential order allowed: {ctx.n_max}."
        print(f'mechanism library: {len(registry.mechanisms)} entries, '
              f'{len(registry.grown())} grown -> {registry_path}', flush=True)

    pipeline.main(
        specification=specification,
        inputs={'data': data},
        config=cfg,
        max_sample_nums=args.max_samples,
        class_config=class_config,
        log_dir=args.log_path,
        mechanism_text=mechanism_text,
        grammar_context=ctx,
    )

