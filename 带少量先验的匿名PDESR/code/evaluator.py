"""Evaluating one candidate: sandbox, scoring and guards.

Merges the evaluator, the optional numba decorator and the leak guard (the AST
scan that stops a candidate from reaching the filesystem). `repair_body_layout`
restores a reply whose indentation the model mangled.
"""
from __future__ import annotations
import ast
import contextlib
import copy
import dataclasses
import io
import multiprocessing
import re
import sys
import textwrap
import time
from abc import abstractmethod, ABC
from collections.abc import Sequence
from typing import Any, Type

import numpy as np

import core


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
    # The derivative call costs the most; the baseline spells it `d_dx`, this harness
    # spells it `d_axis`, so count both.
    complexity += clean_code.count('d_dx') * 2
    complexity += clean_code.count('d_axis') * 2
    complexity += clean_code.count('params')
    complexity += clean_code.count('+') + clean_code.count('-') + clean_code.count('*') + clean_code.count('/')
    complexity += clean_code.count('sin') + clean_code.count('cos') + clean_code.count('exp')

    return complexity if complexity > 0 else 1


def _ensure_accumulator_initialised(body: str) -> str:
    """Prepend `f1_t_pred = 0.0` when a body starts by accumulating into it."""
    for line in (body or "").splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if re.match(r"\s*f1_t_pred\s*[+\-*/]=", line):
            return "    f1_t_pred = 0.0\n" + body
        return body
    return body


_CODE_LINE = re.compile(
    r"^\s*(#.*"
    r"|f1_t_pred\b.*"
    r"|return\b.*"
    r"|[A-Za-z_]\w*\s*=[^=].*"
    r"|(if|for|while|with|try|else|elif|except|finally)\b.*)\s*$")


def _indent_one_level(text: str) -> str:
    return '\n'.join(('    ' + ln) if ln.strip() else ln
                     for ln in text.splitlines()) + '\n'


def _candidate_blocks(text: str):
    """Runs of consecutive lines that look like code, longest first."""
    runs, current = [], []
    for line in text.splitlines():
        if not line.strip() or _CODE_LINE.match(line):
            current.append(line)
        else:
            if any(part.strip() for part in current):
                runs.append('\n'.join(current))
            current = []
    if any(part.strip() for part in current):
        runs.append('\n'.join(current))
    runs.sort(key=len, reverse=True)
    return runs


def repair_body_layout(text: str):
    """Recover an executable body from a reply whose layout is not the expected one.

    The reply may be flat, may have a first line at column 0 with the rest indented,
    may carry prose before, between or after the code, or may open by accumulating
    into `f1_t_pred`. Each candidate layout is normalised and then PARSED, and only a
    candidate that parses and that both writes and returns `f1_t_pred` is kept.
    """
    for block in _candidate_blocks(text):
        if 'f1_t_pred' not in block or 'return' not in block:
            continue
        for layout in (textwrap.dedent(block),
                       '\n'.join(ln.strip() for ln in block.splitlines())):
            indented = _indent_one_level(layout)
            try:
                ast.parse(f'def _body():\n{indented}')
            except SyntaxError:
                continue
            return _ensure_accumulator_initialised(indented)
    return None


def _trim_function_body(generated_code: str) -> str:
    if not generated_code:
        return ''

    # The reply often carries the whole `def equation(...)` header back, sometimes
    # behind a sentence of prose or inside a code fence.
    stripped = generated_code.strip()
    if "```" in stripped:
        fenced = re.findall(r"```[a-zA-Z]*\s*(.*?)```", stripped, re.S)
        if fenced:
            stripped = max(fenced, key=len).strip()
    lines = stripped.splitlines()
    for start, line in enumerate(lines):
        if line.startswith("def "):
            tail = "\n".join(lines[start:])
            try:
                parsed = ast.parse(tail)
            except SyntaxError:
                break
            if parsed.body and isinstance(parsed.body[0], ast.FunctionDef):
                body_lines = tail.splitlines()[1:]
                if any(part.strip() for part in body_lines):
                    return _ensure_accumulator_initialised(
                        "\n".join(body_lines) + "\n\n")
            break

    repaired = repair_body_layout(generated_code)
    if repaired:
        return repaired

    code = f'def fake_function_header():\n{generated_code}'

    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)

        except SyntaxError as e:
            # `lineno` can point at the first line, in which case truncating to the
            # lines above it yields an empty string that fails to parse again -- the
            # loop would never terminate. Bail out instead.
            if e.lineno is None or e.lineno <= 1:
                return ''
            code = '\n'.join(code.splitlines()[:e.lineno - 1])

    if not code:
        return ''

    visitor = _FunctionLineVisitor('fake_function_header')
    visitor.visit(tree)
    body_lines = code.splitlines()[1:visitor.function_end_line]
    body = '\n'.join(body_lines)
    if body.strip():
        return _ensure_accumulator_initialised(body + '\n\n')

    # Last resort: a body that lost its indentation entirely.
    for layout in (textwrap.dedent(generated_code),
                   '\n'.join(ln.strip() for ln in generated_code.splitlines())):
        reindented = '\n'.join(('    ' + ln) if ln.strip() else ln
                               for ln in layout.splitlines())
        if 'f1_t_pred' not in reindented or 'return' not in reindented:
            continue
        try:
            ast.parse(f'def fake_function_header():\n{reindented}')
        except SyntaxError:
            continue
        return _ensure_accumulator_initialised(reindented + '\n\n')
    return ''


def _sample_to_program(
        generated_code: str,
        version_generated: int | None,
        template: core.Program,
        function_to_evolve: str,
) -> tuple[core.Function, str]:
    body = _trim_function_body(generated_code)
    if version_generated is not None:
        body = core.rename_function_calls(
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
    ) -> tuple[Any, bool, list | None]:
        """(test_output, runs_ok, fitted_params)."""
        raise NotImplementedError(
            'Must provide a sandbox for executing untrusted code.')


# The evaluator prints the coefficients it fitted on its own stdout. Reading them
# back out of that line keeps the fitted values in the record without adding a
# second output channel to the spec every problem has to implement.
_PARAMS_LINE = re.compile(r"Params:\s*\[([^\]]*)\]")


def _fitted_params(text: str) -> list | None:
    """The fitted coefficients the evaluator printed, or None if it printed none."""
    match = _PARAMS_LINE.search(text or "")
    if not match:
        return None
    values = []
    # numpy renders an array as `[ 0.5   1.    -0.05]` -- no commas -- so split on
    # commas and on whitespace.
    for piece in re.split(r"[,\s]+", match.group(1).strip()):
        if not piece:
            continue
        try:
            values.append(float(piece))
        except ValueError:
            return None
    return values or None


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
                return self._as_result(queue.get_nowait())
            time.sleep(0.1)
        return None, False, None

    @staticmethod
    def _as_result(value):
        """(test_output, runs_ok, fitted_params), tolerating a bare (output, ok)."""
        if isinstance(value, tuple) and len(value) == 3:
            return value
        if isinstance(value, tuple) and len(value) == 2:
            return value[0], value[1], None
        return None, False, None


    def _print_evaluation_details(self, program, results, **kwargs):
        print('================= Evaluated Program =================')
        function = core.text_to_program(program).get_function(kwargs.get('func_to_evolve', 'equation'))
        print(f'{str(function).strip()}\n-----------------------------------------------------')
        print(f'Score: {results}\n=====================================================\n\n')



    def _compile_and_run_function(self, program, function_to_run, function_to_evolve, 
                                  dataset, numba_accelerate, result_queue):
        try:
            if numba_accelerate:
                program = add_numba_decorator(
                    program=program,
                    function_to_evolve=function_to_evolve
                )
            
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]

            # Everything the evaluator prints is captured and re-emitted, so the
            # run log looks exactly as before, and the `Params: [...]` line is read
            # back out of it.
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                results = function_to_run(dataset)
            transcript = captured.getvalue()
            if transcript:
                sys.stdout.write(transcript)

            if not isinstance(results, (int, float, np.floating)):
                result_queue.put((None, False, None))
                return
            result_queue.put((float(results), True, _fitted_params(transcript)))

        except Exception as e:
            print(f"Execution Error: {e}")
            result_queue.put((None, False, None))



def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    for name in core.get_functions_called(program):
        if name.startswith(f'{function_to_evolve}_v'):
            return True
    return False



class Evaluator:

    def __init__(
            self,
            database: core.ExperienceBuffer,
            template: core.Program,
            function_to_evolve: str, 
            function_to_run: str, 
            inputs: Sequence[Any], 
            timeout_seconds: int = 60,
            sandbox_class: Type[Sandbox] = Sandbox,
    ):
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = sandbox_class()

    def analyse(
            self,
            sample: str,
            island_id: int | None,
            version_generated: int | None,
            **kwargs 
    ) -> None:
        new_function, program = _sample_to_program(
            sample, version_generated, self._template, self._function_to_evolve)

        if not (new_function.body or '').strip():
            # The reply carried no executable body. Running the empty function would
            # fail later with a confusing `NoneType * float` type error, so reject the
            # draw here and say why.
            print("🚫 [UNPARSABLE] the reply carried no executable body; draw dropped",
                  flush=True)
            return

        # Leak guard: a candidate is a right-hand side built from the measured
        # fields. Anything that reaches the filesystem, an import or a socket --
        # e.g. `np.load` of the generator's closed-form force -- is not a
        # hypothesis about the data and is dropped before it costs sandbox time.
        verdict = check(new_function.body)
        if not verdict.allowed:
            print(f"🚫 [LEAK GUARD] {verdict.reason}")
            profiler: core.Profiler = kwargs.get('profiler', None)
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
            test_output, runs_ok, fitted_params = self._sandbox.run(
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
                    complexity = _calculate_complexity(new_function.body)
                except Exception:
                    complexity = 999
                    mse = float('inf')


                scores_per_test[current_input] = -mse
                new_function.mse = mse
                new_function.complexity = complexity
                new_function.params = fitted_params

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
            profiler: core.Profiler = kwargs.get('profiler', None)
            if profiler:
                global_sample_nums = kwargs.get('global_sample_nums', None)
                sample_time = kwargs.get('sample_time', None)
                new_function.global_sample_nums = global_sample_nums
                new_function.score = None
                new_function.sample_time = sample_time
                new_function.evaluate_time = evaluate_time
                profiler.register_function(new_function)


# ==========================================================================
# leak guard: no candidate may reach outside the process
# ==========================================================================

# Names that reach the filesystem, the process or the interpreter itself.
_BLOCKED_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "input",
    "globals", "locals", "vars", "breakpoint",
})
# Module roots that exist only to reach outside the process.
_BLOCKED_ROOTS = frozenset({
    "os", "sys", "io", "pathlib", "socket", "urllib", "http", "requests",
    "subprocess", "shutil", "glob", "pickle", "importlib", "ctypes", "inspect",
    "tempfile", "zipfile", "tarfile", "sqlite3", "ftplib", "smtplib",
})
# numpy entry points that read or write files. None of them can appear in a
# right-hand side built from the measured grids.
_BLOCKED_NUMPY = frozenset({
    "load", "loadtxt", "genfromtxt", "fromfile", "memmap", "save", "savez",
    "savez_compressed", "savetxt", "tofile", "fromregex", "DataSource",
})

_DEF = re.compile(r"^\s*def\s+\w+\s*\(", re.M)


@dataclasses.dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str = ""


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def check(body: str) -> Verdict:
    """Scan a candidate's function body for out-of-process access."""
    try:
        tree = ast.parse(textwrap.dedent(body or ""))
    except SyntaxError:
        # Malformed code is the sandbox's problem, not this one.
        return Verdict(True)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            root = (node.names[0].name.split(".")[0] if isinstance(node, ast.Import)
                    else (node.module or "").split(".")[0])
            return Verdict(False, f"importing `{root}` is outside the measured fields")
        if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            return Verdict(False, f"`{node.id}` is outside the measured fields")
        if isinstance(node, ast.Attribute):
            chain = _dotted(node)
            root = chain.split(".")[0]
            if root in _BLOCKED_ROOTS:
                return Verdict(False, f"`{chain}` is outside the measured fields")
            if root == "np" and chain.split(".")[-1] in _BLOCKED_NUMPY:
                return Verdict(False, f"`{chain}` reads or writes a file")
    return Verdict(True)


def body_of(code: str) -> str:
    """Recover the body from a history entry that stores the whole function."""
    lines = (code or "").splitlines()
    for index, line in enumerate(lines):
        if re.match(r"^\s*def\s+\w+\s*\(", line):
            for end in range(index, len(lines)):
                if lines[end].rstrip().endswith(":"):
                    return textwrap.dedent("\n".join(lines[end + 1:]))
            break
    return textwrap.dedent(code or "")


def audit_histories(paths) -> tuple[int, int, list[str]]:
    """(candidates, rejected, examples) over frozen `all_samples_history.jsonl`."""
    import json

    total = 0
    rejected = 0
    examples: list[str] = []
    for path in paths:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            body = body_of(row.get("function", ""))
            total += 1
            verdict = check(body)
            if not verdict.allowed:
                rejected += 1
                if len(examples) < 5:
                    examples.append(f"{path}: {verdict.reason}")
    return total, rejected, examples


# ==========================================================================
# optional numba decorator for the evolved function
# ==========================================================================



def add_numba_decorator(
        program: str,
        function_to_evolve: str,
) -> str:

    tree = ast.parse(program)

    numba_imported = False
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == 'numba' for alias in node.names):
            numba_imported = True
            break

    if not numba_imported:
        import_node = ast.Import(names=[ast.alias(name='numba', asname=None)])
        tree.body.insert(0, import_node)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_to_evolve:
            decorator = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='numba', ctx=ast.Load()),
                    attr='jit',
                    ctx=ast.Load()
                ),
                args=[],  
                keywords=[ast.keyword(arg='nopython', value=ast.Constant(value=True))]
            )
            node.decorator_list.append(decorator)

    modified_program = ast.unparse(tree)
    return modified_program
