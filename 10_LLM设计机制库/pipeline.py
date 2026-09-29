
from __future__ import annotations

# from collections.abc import Sequence
from typing import Any, Tuple, Sequence

import code_manipulation
import config as config_lib
import evaluator
import buffer
import sampler
import profile


def _extract_function_names(specification: str) -> Tuple[str, str]:

    run_functions = list(code_manipulation.yield_decorated(specification, 'evaluate', 'run'))
    if len(run_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@evaluate.run`.')
    evolve_functions = list(code_manipulation.yield_decorated(specification, 'equation', 'evolve'))
    
    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@equation.evolve`.')
    
    return evolve_functions[0], run_functions[0]



def main(
        specification: str,
        inputs: Sequence[Any],
        config: config_lib.Config,
        max_sample_nums: int | None,
        class_config: config_lib.ClassConfig,
        **kwargs
):


    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)
    database = buffer.ExperienceBuffer(config.experience_buffer, template, function_to_evolve,
                                       mechanism_text=kwargs.get('mechanism_text', ''))

    log_dir = kwargs.get('log_dir', None)
    if log_dir is None:
        profiler = None
    else:
        profiler = profile.Profiler(log_dir)


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
            grammar_context=kwargs.get('grammar_context')
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

    import json
    import os

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
            "equation": p.body.strip()
        })
    print("=" * 50)

    if log_dir:
        pareto_path = os.path.join(log_dir, "final_pareto_front.json")
        with open(pareto_path, 'w', encoding='utf-8') as f:
            json.dump(final_front_data, f, indent=4)
        print(f"✅ Final Pareto front successfully saved to: {pareto_path}")

