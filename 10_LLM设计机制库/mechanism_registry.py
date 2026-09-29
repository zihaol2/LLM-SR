"""The growing mechanism library: the fixed base plus what the expert loop added.

Only a record and a loader. A promoted mechanism is a named, fixed differential
operator written from the primitives (D, S, MUL, DIV, T, coord) and the mechanisms
that already exist -- never a free-form expression.
"""
from __future__ import annotations

import json
import os

import pde_grammar


BUILTINS = (
    {"name": "adv", "signature": ["phi"], "order": 1,
     "doc": "transport (u.grad)phi, contraction built in"},
    {"name": "dif", "signature": ["phi"], "order": 2,
     "doc": "isotropic second-order (diffusion) operator"},
    {"name": "disp", "signature": ["phi", "axis"], "order": 3,
     "doc": "third-order derivative along one axis"},
    {"name": "hdif", "signature": ["phi", "axis"], "order": 4,
     "doc": "fourth-order derivative along one axis"},
    {"name": "prs", "signature": ["phi", "axis"], "order": 1,
     "doc": "first-order derivative along one axis"},
    {"name": "frc", "signature": ["phi"], "order": 0,
     "doc": "the field itself (bare source term)"},
    {"name": "rxn", "signature": ["phi", "func"], "order": 0,
     "doc": "scalar function of one field, func in {id,sin,cos,exp,tanh}"},
    {"name": "cpl", "signature": ["phi", "psi"], "order": 0,
     "doc": "pointwise product of two fields"},
    {"name": "wave", "signature": ["k1", "k2", "func", "k3"], "order": 0,
     "doc": "coordinate function sin/cos/exp/tanh(k1*x1 + k2*x2 + k3*x3)"},
    {"name": "coord_sin", "signature": ["k1", "k2", "k3"], "order": 0,
     "doc": "alias of wave(...,'sin')"},
    {"name": "coord_cos", "signature": ["k1", "k2", "k3"], "order": 0,
     "doc": "alias of wave(...,'cos')"},
)


class Registry:
    def __init__(self, mechanisms=None, path: str = ""):
        self.path = path
        self.mechanisms = list(mechanisms) if mechanisms else [
            dict(b, builtin=True) for b in BUILTINS]

    @classmethod
    def load(cls, path: str) -> "Registry":
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            recorded = {m["name"] for m in payload.get("mechanisms", [])}
            mechanisms = list(payload.get("mechanisms", []))
            for builtin in BUILTINS:
                if builtin["name"] not in recorded:
                    mechanisms.append(dict(builtin, builtin=True))
            return cls(mechanisms, path=path)
        return cls(path=path)

    def save(self, path: str | None = None) -> str:
        target = path or self.path
        if not target:
            raise ValueError("no registry path")
        os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "mechanisms": self.mechanisms}, handle,
                      ensure_ascii=False, indent=1)
        self.path = target
        return target

    def has(self, name: str) -> bool:
        return any(m["name"] == name for m in self.mechanisms)

    def grown(self):
        return [m for m in self.mechanisms
                if not m.get("builtin") and m.get("status") != "rejected"]

    def add(self, spec: dict) -> None:
        self.mechanisms.append(dict(spec))

    def signature_of(self, name: str) -> str:
        for mechanism in self.mechanisms:
            if mechanism["name"] == name:
                return f"{name}({', '.join(mechanism['signature'])})"
        return name

    def render_prompt(self) -> str:
        lines = ["### MECHANISM LIBRARY (the knowledge base you may build from)",
                 "Fixed base (always available):"]
        for mechanism in self.mechanisms:
            if mechanism.get("builtin"):
                lines.append(f"  {self.signature_of(mechanism['name']):<26}"
                             f" order {mechanism['order']}  {mechanism.get('doc', '')}")
        grown = self.grown()
        if grown:
            lines += ["", "Grown by the expert loop and VALIDATED (available too):"]
            for mechanism in grown:
                lines.append(f"  {self.signature_of(mechanism['name']):<26}"
                             f" order {mechanism['order']}  "
                             f"{mechanism.get('physics', '')}")
        lines += ["", "New primitives you may use inside a definition:",
                  "  DIV(a, b)            pointwise ratio",
                  "  T(f)                 pointwise tanh",
                  "  coord(func, k1, k2, k3)   func(k1*x1 + k2*x2 + k3*x3), "
                  "func in {id,sin,cos,exp,tanh}"]
        return "\n".join(lines)

    def rejected_block(self) -> str:
        rejected = [m for m in self.mechanisms if m.get("status") == "rejected"]
        if not rejected:
            return ""
        lines = ["### MECHANISMS ALREADY TRIED AND REJECTED (do not re-propose)"]
        for mechanism in rejected:
            lines.append(f"  {self.signature_of(mechanism['name'])}: "
                         f"{mechanism.get('rejection', 'no held-out improvement')}")
        return "\n".join(lines)


def register_into_grammar(registry: Registry):
    installed = []
    for mechanism in registry.grown():
        try:
            pde_grammar.register_mechanism(mechanism)
            installed.append(mechanism["name"])
        except Exception:
            continue
    return installed

