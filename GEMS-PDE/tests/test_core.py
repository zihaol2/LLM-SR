"""Offline checks of numerical, search, parsing and resume invariants."""

import ast
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from gems_pde import config
from gems_pde import common as C
from gems_pde import refine, search
from gems_pde.baseline_adapter import extract, template
from gems_pde.vendor.llmpdesr.buffer import ExperienceBuffer
from gems_pde.vendor.llmpdesr.config import ExperienceBufferConfig
from gems_pde.io import read


class CoreTests(unittest.TestCase):
    def test_observation_split_and_projection(self):
        for task in config.TASKS:
            d = C.Data(task.name)
            self.assertEqual(int(d.trainmask.sum()), 800)
            self.assertEqual(int(d.valmask.sum()), 320)
            self.assertFalse(set(d.trainidx) & set(d.valididx))
            n = {"expr": "u", "params": []}
            np.testing.assert_allclose(d.feature(n, []), d.window(d.values["u"]).ravel())

    def test_canonical_complexity_and_derivative_cap(self):
        for a, b in [("u", "u**1"), ("a*u", "u"), ("D(D(u,1),1)", "D(u,2)")]:
            left = C.normalize_node({"expr": a})
            right = C.normalize_node({"expr": b})
            self.assertEqual(left["expr"], right["expr"])
            self.assertEqual(C.complexity([left]), C.complexity([right]))
        with self.assertRaises(ValueError):
            C.normalize_node({"expr": "D(D(u,4),1)"})

    def test_variable_projection_recovers_linear_and_nonlinear_parameters(self):
        x = np.linspace(-1.2, 1.2, 80)
        y = 2.4 * np.sin(1.3 * x)
        tr = np.arange(80) % 4 != 0
        va = ~tr
        d = SimpleNamespace(
            y={"u": y}, scale={"u": float(np.mean(y * y))}, trainmask=tr, valmask=va
        )
        d.feature = lambda node, p: np.sin(float(p[0]) * x)
        fitted = C.fit(d, "u", [{"id": "sin", "expr": "np.sin(a*u)", "params": ["a"]}])
        self.assertIsNotNone(fitted)
        self.assertAlmostEqual(fitted["terms"][0]["params"]["a"], 1.3, places=5)
        self.assertAlmostEqual(fitted["terms"][0]["coefficient"], 2.4, places=5)
        self.assertLess(fitted["validation_mse"], 1e-12)

    def test_archive_keeps_high_error_partial_routes(self):
        with patch.object(
            C, "fit", return_value={"terms": [], "score": 1.0, "fully_refined": True}
        ):
            archive = search.Archive(None, "u")
        root = next(iter(archive.states))
        good = archive.add(
            {"terms": [{"id": "A"}], "score": 0.1, "fully_refined": True}, root, {}, 1
        )
        poor = archive.add(
            {"terms": [{"id": "B"}], "score": 10.0, "fully_refined": True}, root, {}, 1
        )
        self.assertIn(good, archive.states)
        self.assertIn(poor, archive.states)
        self.assertEqual(len(archive.states), 3)

    def test_complete_local_neighborhood(self):
        model = {"terms": [{"id": f"M{i}"} for i in range(3)]}
        candidates = list(refine.neighbors(model, {f"M{i}": {} for i in range(150)}, 8))
        self.assertEqual(len(candidates), 591)
        self.assertEqual(len({ids for ids, edit in candidates}), 591)
        self.assertEqual(sum(edit["operation"] == "replace" for ids, edit in candidates), 441)

    def test_every_neighbor_is_jointly_fit_and_twofold_gate_is_enforced(self):
        incumbent = {"terms": [{"id": "A"}, {"id": "B"}], "score": 10.0}
        catalog = {k: {"id": k} for k in "ABCD"}
        with tempfile.TemporaryDirectory() as td:
            with patch.object(C, "fit", return_value={"terms": [], "score": 6.0}) as fit:
                model, event = refine.sweep(
                    None, "u", incumbent, catalog, Path(td) / "reject.jsonl"
                )
                self.assertEqual(fit.call_count, 8)
                self.assertTrue(all(call.kwargs["full_budget"] for call in fit.call_args_list))
                self.assertTrue(
                    all(call.args[3] is incumbent["terms"] for call in fit.call_args_list)
                )
                self.assertFalse(event["accepted"])
                self.assertIs(model, incumbent)
            with patch.object(C, "fit", return_value={"terms": [], "score": 4.0}):
                model, event = refine.sweep(
                    None, "u", incumbent, catalog, Path(td) / "accept.jsonl"
                )
                self.assertTrue(event["accepted"])
                self.assertEqual(model["score"], 4.0)

    def test_baseline_parser_shares_internal_parameter_without_running_code(self):
        raw = """def equation(x,u,dx,params,d_dx):
    a=params[0]*np.sin(params[2]*u)
    b=params[1]*np.cos(params[2]*u)
    return a+b
"""
        nodes, body = extract(raw, ["u"])
        self.assertEqual(len(nodes["u"]), 2)
        self.assertEqual([n["params"] for n in nodes["u"]], [["p2"], ["p2"]])
        with self.assertRaises(ValueError):
            extract("def equation():\n    import os\n    return u", ["u"])

    def test_original_pareto_core_prompt(self):
        program = template("[D1] generic\n[V1] u,x,t", {}, ["u"])
        db = ExperienceBuffer(ExperienceBufferConfig(), program, "equation")
        from gems_pde.vendor.llmpdesr.code_manipulation import Function

        for mse, complexity in [(1.0, 2), (2.0, 3), (0.5, 4)]:
            db.register_program(
                Function("equation", "u", "    return u", mse=mse, complexity=complexity),
                0,
                {"observations": -mse},
            )
        self.assertEqual(len(db.pareto_front), 2)
        text = db.get_prompt().code
        self.assertIn("Current Pareto Frontier", text)
        self.assertNotIn("{PARETO_FRONTIER_PLACEHOLDER}", text)

    def test_discovery_modules_do_not_import_reference_scoring(self):
        for name in (
            "search",
            "baseline",
            "refine",
            "common",
            "observations",
            "mechanisms",
            "config",
        ):
            tree = ast.parse((config.PACKAGE / (name + ".py")).read_text(encoding="utf-8"))
            for n in ast.walk(tree):
                if isinstance(n, ast.ImportFrom):
                    imports = [n.module or ""] + [v.name for v in n.names]
                    self.assertFalse(
                        any("scoring" in v or "evaluation" in v for v in imports), name
                    )

    def test_resume_rejects_changed_configuration(self):
        from gems_pde.run_guard import guard_task

        old_dir = config.RUN_DIR
        old_seed = config.SETTINGS["seed"]
        try:
            with tempfile.TemporaryDirectory() as td:
                config.RUN_DIR = Path(td)
                guard_task("test", 0)
                guard_task("test", 0)
                config.SETTINGS["seed"] += 1
                with self.assertRaises(RuntimeError):
                    guard_task("test", 0)
        finally:
            config.RUN_DIR = old_dir
            config.SETTINGS["seed"] = old_seed

    def test_evaluation_checks_freezes_before_reference_file_access(self):
        from gems_pde.evaluation import evaluate

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError, "Freeze every selected result"):
                evaluate(Path(td), Path(td) / "absent-references.json", Path(td) / "out", [0])


class ApiTests(unittest.TestCase):
    def test_invalid_json_usage_is_logged_and_cache_is_offline(self):
        from gems_pde.api import call_json

        old_offline = config.OFFLINE
        try:
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "initial.json"
                bodies = [
                    {
                        "model": "test",
                        "usage": {"total_tokens": 7},
                        "choices": [{"message": {"content": "not JSON"}}],
                    },
                    {
                        "model": "test",
                        "usage": {"total_tokens": 9},
                        "choices": [{"message": {"content": '{"L0":[],"G0":[]}'}}],
                    },
                ]
                responses = [SimpleNamespace(status_code=200, json=lambda b=b: b) for b in bodies]
                with patch.dict(
                    os.environ, {"PDE_API_KEY": "unit-test-placeholder"}, clear=False
                ), patch("gems_pde.api.requests.post", side_effect=responses) as post, patch(
                    "gems_pde.api.time.sleep"
                ):
                    config.OFFLINE = False
                    call_json("test", path, "initial")
                    self.assertEqual(post.call_count, 2)
                attempts = read(path.with_name("initial_attempts.json"))
                self.assertEqual(sum(v["usage"]["total_tokens"] for v in attempts), 16)
                self.assertFalse(attempts[0]["accepted"])
                self.assertTrue(attempts[1]["accepted"])
                for file in Path(td).iterdir():
                    self.assertNotIn("unit-test-placeholder", file.read_text())
                config.OFFLINE = True
                with patch(
                    "gems_pde.api.requests.post", side_effect=AssertionError("Network forbidden")
                ):
                    self.assertEqual(call_json("test", path, "initial"), {"L0": [], "G0": []})
                    with self.assertRaises(RuntimeError):
                        call_json("changed prompt", path, "initial")
                    with self.assertRaises(RuntimeError):
                        call_json("test", Path(td) / "missing.json", "initial")
        finally:
            config.OFFLINE = old_offline


if __name__ == "__main__":
    unittest.main()
