"""Observation validation, spline features and expression complexity."""

# BLAS limits must be set before importing NumPy.
# ruff: noqa: E402
import os

for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import warnings
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT
ASSETS = ROOT / "assets"
from .builder import Graph as Graph
from .builder import canonical as canonical
from .builder import operations as operations
from .builder import render as render
from .config import SETTINGS as SETTINGS
from .deduplicate import Analyzer as Analyzer
from .profile_rules import Profile


def read(path):
    path = Path(path)
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt", encoding="utf8") as file:
            return json.load(file)
    return json.loads(path.read_text(encoding="utf8"))


@lru_cache(maxsize=1)
def bundled_graphs():
    return read(ASSETS / "initial_graphs.json.gz")


def initial_graph(case):
    return read(case["graph_path"]) if case.get("graph_path") else bundled_graphs()[case["id"]]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")

    def convert(z):
        if isinstance(z, np.generic):
            return z.item()
        raise TypeError(type(z).__name__)

    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=convert),
        encoding="utf-8",
    )
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Data:

    def __init__(self, case):
        self.case = case
        self.fields = case["fields"]
        self.coordinates = case["coordinates"]
        self.d = len(self.coordinates)
        if self.d not in (1, 2, 3) or not self.fields:
            raise ValueError("Declare fields on one, two or three spatial axes")
        path = WORKSPACE / case["data_path"]
        if sha(path) != case["data_sha256"]:
            raise ValueError("Observation hash changed")
        if path.suffix == ".mat":
            z = loadmat(path, variable_names=[*self.coordinates, "t", *self.fields])
        else:
            with np.load(path) as file:
                if set(file.files) != {*self.coordinates, "t", *self.fields}:
                    raise ValueError("Observation-only file required")
                z = {k: file[k].copy() for k in file.files}
        self.axes = [np.asarray(z[c]).ravel() for c in self.coordinates]
        self.t = np.asarray(z["t"]).ravel()
        self.full = {f: np.asarray(z[f], dtype=float) for f in self.fields}
        self.shape = next(iter(self.full.values())).shape
        expected = tuple(len(a) for a in self.axes) + (len(self.t),)
        for field, values in self.full.items():
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(f"Field {field} must be finite with shape {expected}")
        for name, axis in zip(self.coordinates, self.axes):
            spacing = np.diff(axis)
            if (
                len(axis) < 6
                or not np.isfinite(axis).all()
                or not np.all(spacing > 0)
                or not np.allclose(spacing, spacing[0])
            ):
                raise ValueError(
                    f"Spatial axis {name} needs >=6 finite, increasing, uniform points"
                )
        if len(self.t) < 2 or not np.isfinite(self.t).all() or not np.all(np.diff(self.t) > 0):
            raise ValueError("Time needs >=2 finite, strictly increasing points")
        self.idx = np.arange(len(self.t))
        self.values = self.full
        counts = [10] if self.d == 1 else [4] * self.d
        self.phis = []
        for coord, count in zip(self.axes, counts):
            width = 2 * (coord[-1] - coord[0]) / count
            phi = np.zeros((len(coord), count))
            for j in range(count):
                center = coord[0] + (j + 0.5) * (coord[-1] - coord[0]) / count
                mask = np.abs(coord - center) < width / 2
                phi[mask, j] = np.sin(np.pi * (coord[mask] - center + width / 2) / width) ** 4
            self.phis.append(phi * (coord[1] - coord[0]))
        self.window_shape = tuple(counts)
        ft = {f: CubicSpline(self.t, v, axis=self.d)(self.t, 1) for f, v in self.full.items()}
        self.yfull = {f: self.window(v).ravel() for f, v in ft.items()}
        self.y = self.yfull
        self.scale = {f: float(np.mean(v * v)) + 1e-30 for f, v in self.y.items()}
        self.language = Graph(self.fields, self.coordinates).language
        self.profile = Profile(self.fields, self.coordinates)
        self.rawcache = OrderedDict()
        self.rawbytes = 0
        self.feature_cache = OrderedDict()
        self.featurebytes = 0
        self.fit_calls = 0
        self.feature_calls = 0
        self.fit_nfev = 0

    def window(self, a):
        value = a
        for axis, phi in enumerate(self.phis):
            value = np.moveaxis(np.tensordot(phi.T, value, axes=(1, axis)), 0, axis)
        return value

    @lru_cache(maxsize=65536)
    def parsed(self, text):
        return self.profile.validate(self.language.parse(text))

    @lru_cache(maxsize=65536)
    def params(self, e):
        if e.op == "Parameter":
            return frozenset([e.value])
        return frozenset().union(*(self.params(a) for a in e.args))

    def raw(self, e, p, full=False, memo=None):
        if memo is None:
            memo = {}
        if e in memo:
            return memo[e]
        independent = not self.params(e)
        cachekey = (e, full)
        if independent and cachekey in self.rawcache:
            self.rawcache.move_to_end(cachekey)
            return self.rawcache[cachekey]
        shape = self.shape
        op = e.op
        if op == "Field":
            out = (self.full if full else self.values)[e.value]
        elif op == "Coordinate":
            axis = self.d if e.value == "t" else self.coordinates.index(e.value)
            arr = (self.t if full else self.t[self.idx]) if axis == self.d else self.axes[axis]
            dims = [1] * (self.d + 1)
            dims[axis] = len(arr)
            out = arr.reshape(dims)
        elif op == "Parameter":
            out = p[e.value]
        elif op == "Constant":
            out = float(e.value)
        else:
            a = [self.raw(c, p, full, memo) for c in e.args]
            if op == "Add":
                out = a[0] + a[1]
            elif op == "Subtract":
                out = a[0] - a[1]
            elif op == "Multiply":
                out = a[0] * a[1]
            elif op == "Divide":
                out = a[0] / a[1]
            elif op == "Power":
                out = np.power(a[0], a[1])
            elif op == "Negate":
                out = -a[0]
            elif op == "PD":
                out = np.broadcast_to(a[0], shape)
                for axis, n in enumerate(e.value):
                    if n:
                        out = make_interp_spline(self.axes[axis], out, k=5, axis=axis)(
                            self.axes[axis], nu=n
                        )
            else:
                fn = {
                    "Exp": np.exp,
                    "Log": np.log,
                    "Sin": np.sin,
                    "Cos": np.cos,
                    "Tan": np.tan,
                    "ArcSin": np.arcsin,
                    "ArcCos": np.arccos,
                    "ArcTan": np.arctan,
                    "Sinh": np.sinh,
                    "Cosh": np.cosh,
                    "Tanh": np.tanh,
                    "ArcSinh": np.arcsinh,
                    "ArcCosh": np.arccosh,
                    "ArcTanh": np.arctanh,
                    "Sqrt": np.sqrt,
                    "CubeRoot": np.cbrt,
                    "Sinc": lambda z: np.sinc(z / np.pi),
                }[op]
                out = fn(a[0])
        memo[e] = out
        if (
            independent
            and isinstance(out, np.ndarray)
            and (out.size > 1)
            and (op not in {"Field", "Coordinate"})
        ):
            self.rawcache[cachekey] = out
            self.rawbytes += out.nbytes
            while self.rawbytes > 64 * 1024**2 and self.rawcache:
                _, old = self.rawcache.popitem(last=False)
                self.rawbytes -= old.nbytes
        return out

    def feature(self, node, params, full=False):
        self.feature_calls += 1
        key = (node["expr"], tuple(map(float, params)), full)
        if key in self.feature_cache:
            self.feature_cache.move_to_end(key)
            return self.feature_cache[key]
        e = self.parsed(node["expr"])
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            out = np.broadcast_to(self.raw(e, dict(zip(node["params"], params)), full), self.shape)
        if not np.isfinite(out).all():
            raise ValueError("Feature outside finite execution domain")
        vec = self.window(out).ravel()
        if not np.isfinite(vec).all():
            raise ValueError("Nonfinite window feature")
        self.feature_cache[key] = vec
        self.featurebytes += vec.nbytes
        while self.featurebytes > 32 * 1024**2 and self.feature_cache:
            _, old = self.feature_cache.popitem(last=False)
            self.featurebytes -= old.nbytes
        return vec


def opcost(z, in_add=False):
    if z.is_Atom:
        return 0
    if z.func.__name__.startswith("PD_"):
        directions = sum((int(n) > 0 for n in z.func.__name__[3:].split("_")))
        return 2 * directions + opcost(z.args[0])
    if z.is_Add:
        return len(z.args) - 1 + sum((opcost(a, True) for a in z.args))
    if z.is_Mul:
        return len(z.args) - 1 - int(in_add and z.args[0] == -1) + sum((opcost(a) for a in z.args))
    if z.is_Pow:
        return 2 + sum((opcost(a) for a in z.args))
    return 1 + sum((opcost(a) for a in z.args))


def complexity(nodes, registry, shared=False):
    cost = 0
    parameters = set()
    for i, n in enumerate(nodes):
        z = registry.info(n["id"]).value
        if not n.get("fixed_coefficient", False):
            _, z = z.as_independent(*registry.analyzer.physical, as_Add=False)
            cost += 1 + int(z != 1)
        cost += opcost(z)
        for p in z.free_symbols - registry.analyzer.physical:
            parameters.add(str(p) if shared else (i, str(p)))
    return max(1, cost + len(parameters) + max(0, len(nodes) - 1))
