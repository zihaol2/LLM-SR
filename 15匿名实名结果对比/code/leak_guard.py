"""Keep the baseline search from reading the answer instead of fitting it.

The comparison baseline writes free-form numpy, so nothing in its algebra stops a
candidate from calling `np.load('../data/2DFORCED_v4/truth_force.npz')` -- the
closed-form forcing the generator writes out for verification -- or from opening a
file, an import or a socket. Every claim that the baseline "only saw the measured
fields" rests on that never happening.

This module makes it a checked property: an AST scan over the submitted body that
rejects file, import and network access before the candidate costs sandbox time.
It is a *deny* list, not an algebra: an ordinary equation written in numpy is
untouched. `audit_histories()` re-runs it over frozen candidate sets, which is how
the false-positive rate can be quoted instead of assumed.
"""
from __future__ import annotations

import ast
import dataclasses
import re
import textwrap

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

