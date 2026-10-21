"""Observation-only spline derivatives and spatial window projection."""

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import CubicSpline
from . import config


def validate_observations(z, x, t):
    if x.ndim != 1 or t.ndim != 1 or len(x) < 6 or len(t) < 6:
        raise ValueError("Need one-dimensional x/t arrays and at least six samples per axis")
    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(t))
        or not np.all(np.diff(x) > 0)
        or not np.all(np.diff(t) > 0)
    ):
        raise ValueError("x and t must be finite and strictly increasing")
    if not np.allclose(np.diff(x), np.diff(x)[0], rtol=1e-6, atol=1e-12):
        raise ValueError("The historical spatial quadrature requires a uniform x grid")
    if "u" not in z:
        raise ValueError("Missing state field u")
    for f in ("u", "v"):
        if f in z and (z[f].shape != (len(x), len(t)) or not np.all(np.isfinite(z[f]))):
            raise ValueError(f"{f} must be a finite array with shape (nx,nt)")


class Data:
    def __init__(self, name):
        path = config.task_by_name(name).data_path
        z = loadmat(path)
        self.x = z["x"].ravel()
        self.t = z["t"].ravel()
        validate_observations(z, self.x, self.t)
        self.fields = [f for f in ["u", "v"] if f in z]
        self.full = {f: z[f] for f in self.fields}
        self.ft = {f: CubicSpline(self.t, z[f], axis=1)(self.t, 1) for f in self.fields}
        nx = len(self.x)
        width = 2 * (self.x[-1] - self.x[0]) / config.SETTINGS["windows"]
        self.phi = np.zeros((nx, config.SETTINGS["windows"]))
        for i in range(config.SETTINGS["windows"]):
            center = self.x[0] + (i + 0.5) * (self.x[-1] - self.x[0]) / config.SETTINGS["windows"]
            mask = abs(self.x - center) < width / 2
            self.phi[mask, i] = np.sin(np.pi * (self.x[mask] - (center - width / 2)) / width) ** 4
        self.phi *= self.x[1] - self.x[0]
        common = np.unique(
            np.linspace(
                2,
                len(self.t) - 3,
                config.SETTINGS["train_times"] + config.SETTINGS["validation_times"],
            ).astype(int)
        )
        valpos = set(
            np.linspace(0, len(common) - 1, config.SETTINGS["validation_times"]).astype(int)
        )
        self.trainidx = np.array([v for i, v in enumerate(common) if i not in valpos])
        self.valididx = np.array([v for i, v in enumerate(common) if i in valpos])
        self.idx = np.r_[self.trainidx, self.valididx]
        self.trainmask = np.tile(
            np.arange(len(self.idx)) < len(self.trainidx), config.SETTINGS["windows"]
        )
        self.valmask = ~self.trainmask
        if (
            len(self.trainidx) != config.SETTINGS["train_times"]
            or len(self.valididx) != config.SETTINGS["validation_times"]
        ):
            raise ValueError(
                "Not enough distinct interior times for the requested train/validation split"
            )
        self.values = {f: v[:, self.idx] for f, v in self.full.items()}
        self.y = {f: self.window(self.ft[f][:, self.idx]).ravel() for f in self.fields}
        self.yfull = {f: self.window(self.ft[f]).ravel() for f in self.fields}
        self.scale = {f: float(np.mean(v**2)) + 1e-30 for f, v in self.y.items()}
        self.cached_D = {}

    def window(self, a):
        return self.phi.T @ a
