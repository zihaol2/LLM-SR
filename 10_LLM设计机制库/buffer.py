from __future__ import annotations
import profile
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from typing import Any, Tuple, Mapping
from absl import logging
import numpy as np
import scipy
import code_manipulation
import config as config_lib
Signature = Tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1:])
    return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    test_scores = [scores_per_test[k] for k in scores_per_test.keys()]
    return sum(test_scores) / len(test_scores)


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class Prompt:

    code: str
    version_generated: int
    island_id: int

class ExperienceBuffer:
    def __init__(
            self,
            config: config_lib.ExperienceBufferConfig,
            template: code_manipulation.Program,
            function_to_evolve: str,
            mechanism_text: str = "",
    ) -> None:
        self._config = config
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._mechanism_text = mechanism_text

        self.pareto_front: list[code_manipulation.Function] = []
        self._last_reset_time: float = time.time()

    def _register_program_in_island(
            self,
            program: code_manipulation.Function,
            island_id: int,
            scores_per_test: ScoresPerTest,
            **kwargs
    ) -> None:
        self._islands[island_id].register_program(program, scores_per_test)
        score = _reduce_score(scores_per_test)
        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
            logging.info('Best score of island %d increased to %s', island_id, score)

        profiler: profile.Profiler = kwargs.get('profiler', None)
        if profiler:
            global_sample_nums = kwargs.get('global_sample_nums', None)
            sample_time = kwargs.get('sample_time', None)
            evaluate_time = kwargs.get('evaluate_time', None)
            program.score = score
            program.global_sample_nums = global_sample_nums
            program.sample_time = sample_time
            program.evaluate_time = evaluate_time
            profiler.register_function(program)

    def register_program(
            self,
            program: code_manipulation.Function,
            island_id: int | None,
            scores_per_test: ScoresPerTest,
            **kwargs
    ) -> None:
        if not hasattr(program, 'mse') or program.mse is None or program.complexity is None:
            return
        new_front = []
        is_dominated = False

        for p in self.pareto_front:
            if (p.mse <= program.mse and p.complexity <= program.complexity) and \
                    (p.mse < program.mse or p.complexity < program.complexity):
                is_dominated = True
                break

        if not is_dominated:
            for p in self.pareto_front:
                if not ((program.mse <= p.mse and program.complexity <= p.complexity) and \
                        (program.mse < p.mse or program.complexity < p.complexity)):
                    new_front.append(p)

            new_front.append(program)
            new_front.sort(key=lambda x: x.mse)
            self.pareto_front = new_front
            logging.info(f'Pareto Front Updated! Current size: {len(self.pareto_front)}')

        profiler = kwargs.get('profiler', None)
        if profiler:
            program.score = -program.mse
            program.global_sample_nums = kwargs.get('global_sample_nums', None)
            program.sample_time = kwargs.get('sample_time', None)
            program.evaluate_time = kwargs.get('evaluate_time', None)
            profiler.register_function(program)

    def get_prompt(self) -> Prompt:
        if not self.pareto_front:
            pareto_text = "No valid equations found yet. Please explore basic combinations."
        else:
            pareto_text = "### Current Pareto Frontier (Trade-off between Accuracy and Complexity):\n"
            pareto_text += "Observe these equations. If complexity increases without a significant MSE drop, simplify the structure.\n\n"

            selected = []
            if len(self.pareto_front) <= 3:
                selected = self.pareto_front
            else:
                selected = [self.pareto_front[0], self.pareto_front[len(self.pareto_front) // 2], self.pareto_front[-1]]

            for idx, p in enumerate(selected):
                pareto_text += f"[{idx + 1}] MSE: {p.mse:.2e} | Complexity Score: {p.complexity}\n"
                pareto_text += f"{p.body.strip()}\n\n"

        if self._mechanism_text:
            pareto_text = self._mechanism_text + "\n\n" + pareto_text

        spec_str = str(self._template)
        if "{PARETO_FRONTIER_PLACEHOLDER}" in spec_str:
            spec_str = spec_str.replace("{PARETO_FRONTIER_PLACEHOLDER}", pareto_text)
        else:
            spec_str += "\n\n" + pareto_text

        return Prompt(code=spec_str, version_generated=1, island_id=0)

    def reset_islands(self) -> None:
        pass


class Island:
    def __init__(
            self,
            template: code_manipulation.Program,
            function_to_evolve: str,
            functions_per_prompt: int,
            cluster_sampling_temperature_init: float,
            cluster_sampling_temperature_period: int,
    ) -> None:
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve
        self._functions_per_prompt: int = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = (
            cluster_sampling_temperature_period)

        self._clusters: dict[Signature, Cluster] = {}
        self._num_programs: int = 0


    def register_program(
            self,
            program: code_manipulation.Function,
            scores_per_test: ScoresPerTest,
    ) -> None:
        signature = _get_signature(scores_per_test)
        if signature not in self._clusters:
            score = _reduce_score(scores_per_test)
            self._clusters[signature] = Cluster(score, program)
        else:
            self._clusters[signature].register_program(program)
        self._num_programs += 1


    def get_prompt(self) -> tuple[str, int]:
        signatures = list(self._clusters.keys())
        cluster_scores = np.array(
            [self._clusters[signature].score for signature in signatures])
        
        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (
                1 - (self._num_programs % period) / period)
        probabilities = _softmax(cluster_scores, temperature)

        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        idx = np.random.choice(
            len(signatures), size=functions_per_prompt, p=probabilities)
        chosen_signatures = [signatures[i] for i in idx]
        implementations = []
        scores = []
        for signature in chosen_signatures:
            cluster = self._clusters[signature]
            implementations.append(cluster.sample_program())
            scores.append(cluster.score)

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1
        return self._generate_prompt(sorted_implementations), version_generated


    def _generate_prompt(
            self,
            implementations: Sequence[code_manipulation.Function]) -> str:
        implementations = copy.deepcopy(implementations)

        versioned_functions: list[code_manipulation.Function] = []
        for i, implementation in enumerate(implementations):
            new_function_name = f'{self._function_to_evolve}_v{i}'
            implementation.name = new_function_name
            if i >= 1:
                implementation.docstring = (
                    f'Improved version of `{self._function_to_evolve}_v{i - 1}`.')
            implementation = code_manipulation.rename_function_calls(
                str(implementation), self._function_to_evolve, new_function_name)
            versioned_functions.append(
                code_manipulation.text_to_function(implementation))

        next_version = len(implementations)
        new_function_name = f'{self._function_to_evolve}_v{next_version}'
        header = dataclasses.replace(
            implementations[-1],
            name=new_function_name,
            body='',
            docstring=('Improved version of '
                       f'`{self._function_to_evolve}_v{next_version - 1}`.'),
        )
        versioned_functions.append(header)

        prompt = dataclasses.replace(self._template, functions=versioned_functions)
        
        return str(prompt)


class Cluster:

    def __init__(self, score: float, implementation: code_manipulation.Function):
        self._score = score
        self._programs: list[code_manipulation.Function] = [implementation]
        self._lengths: list[int] = [len(str(implementation))]

    @property
    def score(self) -> float:
        return self._score

    def register_program(self, program: code_manipulation.Function) -> None:
        self._programs.append(program)
        self._lengths.append(len(str(program)))

    def sample_program(self) -> code_manipulation.Function:
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
                max(self._lengths) + 1e-6)
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        return np.random.choice(self._programs, p=probabilities)

