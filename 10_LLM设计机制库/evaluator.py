from __future__ import annotations

from abc import abstractmethod, ABC
import ast
import time
from collections.abc import Sequence
import copy
from typing import Any, Type
import profile
import multiprocessing
import numpy as np
import code_manipulation
import buffer
import evaluator_accelerate
import leak_guard
import grammar_check


class _FunctionLineVisitor(ast.NodeVisitor):

    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None: 
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        assert self._function_end_line is not None
        return self._function_end_line


def _calculate_complexity(code_str: str) -> int:
    if '"""' in code_str:
        clean_code = code_str.split('"""')[-1]
    else:
        clean_code = code_str

    complexity = 0
    complexity += clean_code.count('d_dx') * 2
    complexity += clean_code.count('params')
    complexity += clean_code.count('+') + clean_code.count('-') + clean_code.count('*') + clean_code.count('/')
    complexity += clean_code.count('sin') + clean_code.count('cos') + clean_code.count('exp')

    return complexity if complexity > 0 else 1


def _trim_function_body(generated_code: str) -> str:
    if not generated_code:
        return ''

    code = f'def fake_function_header():\n{generated_code}'

    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)
        
        except SyntaxError as e:
            if e.lineno is None:
                return ''
            code = '\n'.join(code.splitlines()[:e.lineno - 1])

    if not code:
        return ''

    visitor = _FunctionLineVisitor('fake_function_header')
    visitor.visit(tree)
    body_lines = code.splitlines()[1:visitor.function_end_line]
    return '\n'.join(body_lines) + '\n\n'


def _sample_to_program(
        generated_code: str,
        version_generated: int | None,
        template: code_manipulation.Program,
        function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
    body = _trim_function_body(generated_code)
    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            code=body,
            source_name=f'{function_to_evolve}_v{version_generated}',
            target_name=function_to_evolve
        )

    program = copy.deepcopy(template)
    evolved_function = program.get_function(function_to_evolve)
    evolved_function.body = body
    
    return evolved_function, str(program)


class Sandbox(ABC):

    @abstractmethod
    def run(
            self,
            program: str,
            function_to_run: str,
            function_to_evolve: str,
            inputs: Any,  
            test_input: str, 
            timeout_seconds: int,
            **kwargs
    ) -> tuple[Any, bool]:
        raise NotImplementedError(
            'Must provide a sandbox for executing untrusted code.')


class LocalSandbox(Sandbox):
    def __init__(self, verbose=False, numba_accelerate=False):
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate


    def run(self, program: str, function_to_run: str, function_to_evolve: str, 
        inputs: Any, test_input: str, timeout_seconds: int, **kwargs) -> tuple[Any, bool]:

        dataset = inputs[test_input]
        result_queue = multiprocessing.Queue()
        
        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, function_to_run, function_to_evolve, dataset, self._numba_accelerate, result_queue)
        )
        process.start()
        process.join(timeout=timeout_seconds)

        if process.is_alive():
            process.terminate()
            process.join()
            results = None, False
        else:
            results = self._get_results(result_queue)
        
        if self._verbose:
            self._print_evaluation_details(program, results, **kwargs)

        return results


    def _get_results(self, queue):
        for _ in range(5):
            if not queue.empty():
                return queue.get_nowait()
            time.sleep(0.1)
        return None, False


    def _print_evaluation_details(self, program, results, **kwargs):
        print('================= Evaluated Program =================')
        function = code_manipulation.text_to_program(program).get_function(kwargs.get('func_to_evolve', 'equation'))
        print(f'{str(function).strip()}\n-----------------------------------------------------')
        print(f'Score: {results}\n=====================================================\n\n')



    def _compile_and_run_function(self, program, function_to_run, function_to_evolve, 
                                  dataset, numba_accelerate, result_queue):
        try:
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program=program,
                    function_to_evolve=function_to_evolve
                )
            
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]
            results = function_to_run(dataset)
            if not isinstance(results, (int, float, np.floating)):
                result_queue.put((None, False))
                return
            result_queue.put((float(results), True))

        except Exception as e:
            print(f"Execution Error: {e}")
            result_queue.put((None, False))



def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    for name in code_manipulation.get_functions_called(program):
        if name.startswith(f'{function_to_evolve}_v'):
            return True
    return False



class Evaluator:

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            template: code_manipulation.Program,
            function_to_evolve: str, 
            function_to_run: str, 
            inputs: Sequence[Any], 
            timeout_seconds: int = 60,
            sandbox_class: Type[Sandbox] = Sandbox,
            grammar_context=None
    ):
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = sandbox_class()
        self._grammar_context = grammar_context

    def analyse(
            self,
            sample: str,
            island_id: int | None,
            version_generated: int | None,
            **kwargs 
    ) -> None:
        if self._grammar_context is not None:
            normalised = grammar_check.normalise(sample)
            if normalised != sample:
                print("✏️  [NORMALISED] numpy spelling -> grammar call")
                sample = normalised

        new_function, program = _sample_to_program(
            sample, version_generated, self._template, self._function_to_evolve)

        if self._grammar_context is not None:
            conforms, why = grammar_check.check_conformance(sample, self._grammar_context)
            if not conforms:
                print(f"🚫 [ILLEGAL] {why}")
                return

        # Leak guard: a candidate is a right-hand side built from the measured
        # fields. Anything that reaches the filesystem, an import or a socket --
        # e.g. `np.load` of the generator's closed-form force -- is not a
        # hypothesis about the data and is dropped before it costs sandbox time.
        verdict = leak_guard.check(new_function.body)
        if not verdict.allowed:
            print(f"🚫 [LEAK GUARD] {verdict.reason}")
            profiler: profile.Profiler = kwargs.get('profiler', None)
            if profiler:
                new_function.global_sample_nums = kwargs.get('global_sample_nums', None)
                new_function.score = None
                new_function.sample_time = kwargs.get('sample_time', None)
                new_function.evaluate_time = 0.0
                profiler.register_function(new_function)
            return

        scores_per_test = {}

        time_reset = time.time()

        for current_input in self._inputs:
            test_output, runs_ok = self._sandbox.run(
                program, self._function_to_run, self._function_to_evolve, self._inputs, current_input,
                self._timeout_seconds
            )

            if runs_ok and not _calls_ancestor(program, self._function_to_evolve) and test_output is not None:


                if not isinstance(test_output, (int, float, np.floating)):
                    print(f"❌ [Error]: {test_output}")
                    mse = float('inf')
                else:
                    mse = float(test_output)

                try:
                    if self._grammar_context is not None:
                        raw = grammar_check.structural_complexity(new_function.body)
                        info = grammar_check.annotate(new_function.body, self._grammar_context)
                        complexity = grammar_check.effective_complexity(
                            raw, info, self._grammar_context)
                    else:
                        complexity = _calculate_complexity(new_function.body)
                except Exception:
                    complexity = 999
                    mse = float('inf')


                scores_per_test[current_input] = -mse
                new_function.mse = mse
                new_function.complexity = complexity

        evaluate_time = time.time() - time_reset

        if scores_per_test:
            self._database.register_program(
                new_function,
                island_id,
                scores_per_test,
                **kwargs,
                evaluate_time=evaluate_time
            )
        
        else:
            # A draw that reaches the sandbox and comes back with nothing used to
            # vanish silently. The evaluation timeout falls hardest on the many-term
            # candidates -- the ones that could carry the true skeleton -- so a
            # timed-out draw has to be visible in the log instead of showing up as
            # an unexplained missing score.
            print(f"⏱️  [NO SCORE] the sandbox returned nothing for this draw "
                  f"(timeout {self._timeout_seconds}s, or a runtime error); "
                  f"it gets no MSE and no frontier slot")
            profiler: profile.Profiler = kwargs.get('profiler', None)
            if profiler:
                global_sample_nums = kwargs.get('global_sample_nums', None)
                sample_time = kwargs.get('sample_time', None)
                new_function.global_sample_nums = global_sample_nums
                new_function.score = None
                new_function.sample_time = sample_time
                new_function.evaluate_time = evaluate_time
                profiler.register_function(new_function)

