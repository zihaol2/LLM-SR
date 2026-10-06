"""Everything the search loop needs that is not an LLM call.

Source-text handling (`Function`, `Program`, call renaming), the run
configuration, the experience buffer that holds the islands / Pareto front and
assembles the prompts, and the profiler that writes `all_samples_history.jsonl`
beside each run. These were four modules; nothing outside the search loop ever
imported them separately, so they are one module here.
"""
from __future__ import annotations
import ast
import copy
import dataclasses
import json
import logging
import os.path
import time
import tokenize
from collections.abc import Iterator, Mapping, MutableSet, Sequence
from typing import Any, Dict, List, Tuple, Type

import numpy as np
from absl import logging as absl_logging


# ==========================================================================
# source text: Function / Program
# ==========================================================================


from collections.abc import Iterator, MutableSet, Sequence
import io

@dataclasses.dataclass
class Function:

    name: str
    args: str
    body: str
    return_type: str | None = None
    docstring: str | None = None
    score: int | None = None  
    global_sample_nums: int | None = None
    mse: float | None = None
    complexity: int | None = None
    sample_time: float | None = None  
    evaluate_time: float | None = None  
    # The coefficients the evaluator fitted for this candidate. They are part of
    # what was discovered: `1 - params[2]*f1` with params[2] = 1 is the statement
    # `1 - f1`, and nothing downstream can tell those apart without the value.
    params: list | None = None

    def __str__(self) -> str:
        return_type = f' -> {self.return_type}' if self.return_type else ''

        function = f'def {self.name}({self.args}){return_type}:\n'
        if self.docstring:
            new_line = '\n' if self.body else ''
            function += f'    """{self.docstring}"""{new_line}'

        function += self.body + '\n\n'
        
        return function

    def __setattr__(self, name: str, value: str) -> None:
        if name == 'body':
            value = value.strip('\n')

        if name == 'docstring' and value is not None:
            if '"""' in value:
                value = value.strip()
                value = value.replace('"""', '')
        super().__setattr__(name, value)


@dataclasses.dataclass(frozen=True)
class Program:

    preface: str
    functions: list[Function]


    def __str__(self) -> str:
        program = f'{self.preface}\n' if self.preface else ''
        program += '\n'.join([str(f) for f in self.functions])
        
        return program


    def find_function_index(self, function_name: str) -> int:
        function_names = [f.name for f in self.functions]
        count = function_names.count(function_name)
        if count == 0:
            raise ValueError(
                f'function {function_name} does not exist in program:\n{str(self)}'
            )
        if count > 1:
            raise ValueError(
                f'function {function_name} exists more than once in program:\n'
                f'{str(self)}'
            )
        index = function_names.index(function_name)
        
        return index


    def get_function(self, function_name: str) -> Function:
        index = self.find_function_index(function_name)
        
        return self.functions[index]


class ProgramVisitor(ast.NodeVisitor):

    def __init__(self, sourcecode: str):
        self._codelines: list[str] = sourcecode.splitlines()
        self._preface: str = ''
        self._functions: list[Function] = []
        self._current_function: str | None = None


    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.col_offset == 0:  
            self._current_function = node.name

            if not self._functions:
                has_decorators = bool(node.decorator_list)
                if has_decorators:
                    decorator_start_line = min(decorator.lineno for decorator in node.decorator_list)
                    self._preface = '\n'.join(self._codelines[:decorator_start_line - 1])
                else:
                    self._preface = '\n'.join(self._codelines[:node.lineno - 1])

            function_end_line = node.end_lineno
            body_start_line = node.body[0].lineno - 1

            docstring = None
            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                docstring = f'  """{ast.literal_eval(ast.unparse(node.body[0]))}"""'
                if len(node.body) > 1:
                    body_start_line = node.body[1].lineno - 1
                else:
                    body_start_line = function_end_line

            self._functions.append(Function(
                name=node.name,
                args=ast.unparse(node.args), 
                return_type=ast.unparse(node.returns) if node.returns else None,
                docstring=docstring,
                body='\n'.join(self._codelines[body_start_line:function_end_line]),
            ))
            
        self.generic_visit(node)

    def return_program(self) -> Program:
        return Program(preface=self._preface, functions=self._functions)


def text_to_program(text: str) -> Program:
    try:
        tree = ast.parse(text)
        visitor = ProgramVisitor(text)
        visitor.visit(tree)
        return visitor.return_program()
    
    except Exception as e:
        absl_logging.warning('Failed parsing %s', text)
        raise e


def text_to_function(text: str) -> Function:
    program = text_to_program(text)
    
    if len(program.functions) != 1:
        raise ValueError(f'Only one function expected, got {len(program.functions)}'
                         f':\n{program.functions}')
    
    return program.functions[0]


def _tokenize(code: str) -> Iterator[tokenize.TokenInfo]:
    code_bytes = code.encode()
    code_io = io.BytesIO(code_bytes)
    
    return tokenize.tokenize(code_io.readline)


def _untokenize(tokens: Sequence[tokenize.TokenInfo]) -> str:
    code_bytes = tokenize.untokenize(tokens)
    
    return code_bytes.decode()


def _yield_token_and_is_call(code: str) -> Iterator[tuple[tokenize.TokenInfo, bool]]:
    try:
        tokens = _tokenize(code)
        prev_token = None
        is_attribute_access = False
        
        for token in tokens:
            if (prev_token and  
                    prev_token.type == tokenize.NAME and  
                    token.type == tokenize.OP and  
                    token.string == '('):  
                yield prev_token, not is_attribute_access
                is_attribute_access = False
            else:
                if prev_token:
                    is_attribute_access = (
                            prev_token.type == tokenize.OP and prev_token.string == '.'
                    )
                    yield prev_token, False
            prev_token = token
        
        if prev_token:
            yield prev_token, False
    
    except Exception as e:
        absl_logging.warning('Failed parsing %s', code)
        raise e


def rename_function_calls(code: str, source_name: str, target_name: str) -> str:
    if source_name not in code:
        return code
    modified_tokens = []
    
    for token, is_call in _yield_token_and_is_call(code):
        if is_call and token.string == source_name:
            modified_token = tokenize.TokenInfo(
                type=token.type,
                string=target_name,
                start=token.start,
                end=token.end,
                line=token.line
            )
            modified_tokens.append(modified_token)
        else:
            modified_tokens.append(token)
    
    return _untokenize(modified_tokens)


def get_functions_called(code: str) -> MutableSet[str]:
    return set(token.string for token, is_call in
               _yield_token_and_is_call(code) if is_call)


def yield_decorated(code: str, module: str, name: str) -> Iterator[str]:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                attribute = None
                if isinstance(decorator, ast.Attribute):
                    attribute = decorator
                elif isinstance(decorator, ast.Call):
                    attribute = decorator.func
                if (attribute is not None
                        and attribute.value.id == module
                        and attribute.attr == name):
                    yield node.name



# ==========================================================================
# run configuration
# ==========================================================================



from typing import Type



@dataclasses.dataclass(frozen=True)
class ExperienceBufferConfig:
    functions_per_prompt: int = 3
    num_islands: int = 10
    reset_period: int = 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000


@dataclasses.dataclass(frozen=True)
class Config:
    experience_buffer: ExperienceBufferConfig = dataclasses.field(default_factory=ExperienceBufferConfig)
    num_samplers: int = 1
    num_evaluators: int = 1
    samples_per_prompt: int = 4
    # Wall-clock budget for a single sandbox evaluation. Coupled systems
    # (e.g. Predator_Prey: 2 fields, MAX_NPARAMS=15) legitimately need 30-45 s,
    # so the original 30 s silently killed most of their candidates.
    evaluate_timeout_seconds: int = 60
    use_api: bool = False
    api_model: str = "gpt-3.5-turbo"
    # Reasoning models (e.g. deepseek-flash) spend part of the budget on the
    # chain of thought, so a small max_tokens yields a truncated/empty answer.
    max_tokens: int = 16384
    # Empty string keeps the provider default (DeepSeek: high).
    reasoning_effort: str = ""
    # Empty string keeps the provider default (DeepSeek: enabled).
    # Set to "disabled" to skip the chain of thought (much faster and cheaper).
    thinking: str = ""
    # Any OpenAI-compatible base URL, e.g. https://api.teamorouter.cn/v1 .
    # Empty string falls back to the built-in DeepSeek / Zhipu endpoints.
    api_base_url: str = ""
    # Name of the environment variable holding the API key.
    api_key_env: str = "API_KEY"
    # How many of the samples_per_prompt requests to issue in parallel.
    concurrent_requests: int = 4


@dataclasses.dataclass()
class ClassConfig:
    # main.py fills these with the concrete classes; typed as `Any`
    # so this module need not import the LLM client or the sandbox.
    llm_class: Type[Any]
    sandbox_class: Type[Any]



# ==========================================================================
# experience buffer: islands, Pareto front, prompts
# ==========================================================================

from collections.abc import Mapping, Sequence
from typing import Any, Tuple, Mapping
import scipy
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
            config: ExperienceBufferConfig,
            template: Program,
            function_to_evolve: str,
            mechanism_text: str = "",
            prompt_dump_path: str | None = None,
    ) -> None:
        self._config = config
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._mechanism_text = mechanism_text
        # The first prompt of a run is what defines the arm, so it is written next
        # to the results and can be diffed against another arm's.
        self._prompt_dump_path = prompt_dump_path
        self._prompt_dumped = False

        self.pareto_front: list[Function] = []
        self._last_reset_time: float = time.time()

    def _register_program_in_island(
            self,
            program: Function,
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
            absl_logging.info('Best score of island %d increased to %s', island_id, score)

        profiler: Profiler = kwargs.get('profiler', None)
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
            program: Function,
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
        absl_logging.info(f'Pareto Front Updated! Current size: {len(self.pareto_front)}')

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

        if self._prompt_dump_path and not self._prompt_dumped:
            self._prompt_dumped = True
            try:
                with open(self._prompt_dump_path, "w", encoding="utf-8") as handle:
                    handle.write(spec_str)
            except OSError:
                pass

        return Prompt(code=spec_str, version_generated=1, island_id=0)

    def reset_islands(self) -> None:
        pass


class Island:
    def __init__(
            self,
            template: Program,
            function_to_evolve: str,
            functions_per_prompt: int,
            cluster_sampling_temperature_init: float,
            cluster_sampling_temperature_period: int,
    ) -> None:
        self._template: Program = template
        self._function_to_evolve: str = function_to_evolve
        self._functions_per_prompt: int = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = (
            cluster_sampling_temperature_period)

        self._clusters: dict[Signature, Cluster] = {}
        self._num_programs: int = 0


    def register_program(
            self,
            program: Function,
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
            implementations: Sequence[Function]) -> str:
        implementations = copy.deepcopy(implementations)

        versioned_functions: list[Function] = []
        for i, implementation in enumerate(implementations):
            new_function_name = f'{self._function_to_evolve}_v{i}'
            implementation.name = new_function_name
            if i >= 1:
                implementation.docstring = (
                    f'Improved version of `{self._function_to_evolve}_v{i - 1}`.')
            implementation = rename_function_calls(
                str(implementation), self._function_to_evolve, new_function_name)
            versioned_functions.append(
                text_to_function(implementation))

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

    def __init__(self, score: float, implementation: Function):
        self._score = score
        self._programs: list[Function] = [implementation]
        self._lengths: list[int] = [len(str(implementation))]

    @property
    def score(self) -> float:
        return self._score

    def register_program(self, program: Function) -> None:
        self._programs.append(program)
        self._lengths.append(len(str(program)))

    def sample_program(self) -> Function:
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
                max(self._lengths) + 1e-6)
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        return np.random.choice(self._programs, p=probabilities)



# ==========================================================================
# profiler: per-sample log + tensorboard
# ==========================================================================


from typing import List, Dict
from torch.utils.tensorboard import SummaryWriter


class Profiler:
    def __init__(
            self,
            log_dir: str | None = None,
            pkl_dir: str | None = None,
            max_log_nums: int | None = None,
    ):

        logging.getLogger().setLevel(logging.INFO)
        self._log_dir = log_dir
        if log_dir:
            self._writer = SummaryWriter(log_dir=log_dir)
            self._json_file_path = os.path.join(log_dir, 'all_samples_history.jsonl')
            with open(self._json_file_path, 'w', encoding='utf-8') as f:
                pass
        else:
            self._json_file_path = None
        # ======================================================================

        self._max_log_nums = max_log_nums
        self._num_samples = 0
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = -99999999
        self._cur_best_program_str = None
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        self._all_sampled_functions: Dict[int, Function] = {}

        if log_dir:
            self._writer = SummaryWriter(log_dir=log_dir)

        self._each_sample_best_program_score = []
        self._each_sample_evaluate_success_program_num = []
        self._each_sample_evaluate_failed_program_num = []
        self._each_sample_tot_sample_time = []
        self._each_sample_tot_evaluate_time = []

    def _write_tensorboard(self):
        if not self._log_dir:
            return

        self._writer.add_scalar(
            'Best Score of Function',
            self._cur_best_program_score,
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Legal/Illegal Function',
            {
                'legal function num': self._evaluate_success_program_num,
                'illegal function num': self._evaluate_failed_program_num
            },
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Total Sample/Evaluate Time',
            {'sample time': self._tot_sample_time, 'evaluate time': self._tot_evaluate_time},
            global_step=self._num_samples
        )
        if self._cur_best_program_str is not None:
            self._writer.add_text(
                'Best Function String',
                self._cur_best_program_str,
                global_step=self._num_samples
            )
        if self._cur_best_program_str is not None:
            self._writer.add_text(
                'Best Function String',
                self._cur_best_program_str,
                global_step=self._num_samples
            )
        # # Log the function_str
        # self._writer.add_text(
        #     'Best Function String',
        #     self._cur_best_program_str,
        #     global_step=self._num_samples
        # )

    def _write_json(self, programs: Function):
        if not self._json_file_path:
            return

        sample_order = programs.global_sample_nums
        sample_order = sample_order if sample_order is not None else 0

        mse = getattr(programs, 'mse', None)
        complexity = getattr(programs, 'complexity', None)

        content = {
            'sample_order': sample_order,
            'mse': mse,
            'complexity': complexity,
            'score': programs.score,
            'params': getattr(programs, 'params', None),
            'function': str(programs).strip()
        }

        with open(self._json_file_path, 'a', encoding='utf-8') as json_file:
            json.dump(content, json_file, ensure_ascii=False)
            json_file.write('\n')

    def register_function(self, programs: Function):
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return

        sample_orders: int = programs.global_sample_nums
        if sample_orders not in self._all_sampled_functions:
            self._num_samples += 1
            self._all_sampled_functions[sample_orders] = programs
            self._record_and_verbose(sample_orders)
            self._write_tensorboard()
            self._write_json(programs)

    def _record_and_verbose(self, sample_orders: int):
        function = self._all_sampled_functions[sample_orders]
        function_str = str(function).strip('\n')
        mse_str = f"{function.mse:.2e}" if hasattr(function, 'mse') and function.mse is not None else "N/A"
        comp_str = str(function.complexity) if hasattr(function,
                                                       'complexity') and function.complexity is not None else "N/A"
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score
        print(f'================= Evaluated Function =================')
        print(f'{function_str}')
        print(f'------------------------------------------------------')
        print(f'MSE          : {mse_str}')
        print(f'Complexity   : {comp_str}')
        print(f'Sample time  : {str(sample_time)}')
        print(f'Evaluate time: {str(evaluate_time)}')
        print(f'Sample orders: {str(sample_orders)}')
        print(f'======================================================\n\n')

        if function.score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = sample_orders
            self._cur_best_program_str = function_str

        if score:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time
