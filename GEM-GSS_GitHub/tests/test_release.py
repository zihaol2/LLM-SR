"""Offline release checks: no API key, no external network, no original folders."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gem_gss import api, core as C, structure, rotation
from gem_gss.builder import Graph
from gem_gss.cli import cases
from gem_gss.fitting import fit, prediction
from gem_gss.graph import Registry
from gem_gss.refinement import GraphCores, core_proposals
from gem_gss.terminal import valid_choice


def toy_case(folder):
    x = np.linspace(-1, 1, 20)
    t = np.linspace(0, 0.3, 12)
    u = 2 + np.exp(-t)[None, :] * np.sin(x[:, None])
    path = folder / "observations.npz"
    np.savez_compressed(path, x=x, t=t, u=u)
    return dict(
        id="TOY",
        fields=["u"],
        coordinates=["x"],
        prior="An observed smooth scalar field evolves.",
        metadata=dict(shape=list(u.shape)),
        data_path=str(path),
        data_sha256=C.sha(path),
    )


class ReleaseTests(unittest.TestCase):
    def test_all_bundled_frozen_predictions(self):
        published = C.read(C.ASSETS / "published_records.json")["standard"]
        expected_roots = dict(T01=2170, T02=3932, T03=3278, T04=1894, T05=5650, T2D=12216)
        for key, case in cases().items():
            with self.subTest(case=key):
                data = C.Data(case)
                self.assertEqual(
                    sum(n["candidate"] for n in C.initial_graph(case)["nodes"]), expected_roots[key]
                )
                for field, equation in published[key]["equations"].items():
                    mse = float(np.mean((prediction(data, equation) - data.y[field]) ** 2))
                    self.assertTrue(
                        np.isclose(mse, equation["MSE"], rtol=1e-6, atol=1e-24), (key, field, mse)
                    )

    def test_domain_aware_symbolic_guards_and_online_growth(self):
        case = cases()["T01"]
        reg = Registry(case)
        g = reg.graph.language
        a = reg.analyzer
        self.assertNotEqual(
            a.structural_key(a.analyze(g.parse("u**2/u"))),
            a.structural_key(a.analyze(g.parse("u"))),
        )
        before = len(reg.nodes)
        mid, novel = reg.propose("u*(1-u)", 1)
        self.assertTrue(novel)
        self.assertGreater(len(reg.nodes), before)
        self.assertTrue(reg.nodes[mid]["candidate"])
        same, novel = reg.propose("u*(1-u)", 2)
        self.assertEqual(mid, same)
        self.assertFalse(novel)

    def test_existing_multifield_core_is_available(self):
        graph = Graph(["u", "v"], ["x"])
        core = graph.add("u*v", "test")
        wrapped = graph.add("ArcSinh(a*(u*v))", "test")
        reg = Registry(dict(fields=["u", "v"], coordinates=["x"]), graph.export())
        proposals = list(
            core_proposals(dict(id=wrapped, expr=reg.nodes[wrapped]["expr"]), reg, GraphCores(reg))
        )
        self.assertIn((core, "nonlinear_wrapper_to_existing_core"), proposals)

    def test_second_sweep_tests_initial_omissions(self):
        groups = rotation.partition([str(n) for n in range(103)], random.Random(4204))
        from types import SimpleNamespace

        archive = SimpleNamespace(attempted=set())
        for round_number in range(1, 21):
            group = rotation.group_for(groups, round_number)
            quota = int(np.ceil(len(group) / 2))
            forced = rotation.forced_for(archive, group, round_number)
            selected = forced + [m for m in group if m not in forced][: quota - len(forced)]
            archive.attempted.update((m,) for m in selected)
        self.assertEqual(len({m for s in archive.attempted for m in s}), 103)

    def test_linear_projection_is_solved_without_nonlinear_iteration(self):
        with tempfile.TemporaryDirectory() as folder:
            case = toy_case(Path(folder))
            data = C.Data(case)
            graph = Graph(["u"], ["x"])
            mid = graph.add("u", "test")
            reg = Registry(case, graph.export())
            column = data.feature(reg.nodes[mid], [])
            data.y["u"] = 2.75 * column
            data.scale["u"] = float(np.mean(data.y["u"] ** 2))
            model = fit(data, "u", [reg.nodes[mid]], reg)
            self.assertAlmostEqual(model["terms"][0]["coefficient"], 2.75, places=10)
            self.assertEqual(model["optimizer"]["nonlinear_parameters"], 0)
            self.assertLess(model["fit_mse"], 1e-25)

    def test_api_cache_does_not_repeat_request_or_serialize_secret(self):
        class Response:
            status_code = 200

            def json(self):
                return dict(
                    model="mock",
                    choices=[dict(message=dict(content='{"tests":{},"new_nodes":[]}'))],
                    usage={},
                )

        calls = []

        def transport(*args, **kwargs):
            calls.append(1)
            return Response()

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            api.once("mock", root, 1, transport=transport)
            api.once("mock", root, 1, transport=transport)
            self.assertEqual(len(calls), 1)
            self.assertNotIn("Authorization", (root / "round01_request.json").read_text())
        self.assertEqual(api.parse('{"tests": {}}'), {"tests": {}})
        with self.assertRaises(ValueError):
            api.parse('{"tests": []]}')

    def test_contradictory_final_choice_is_rejected(self):
        rows = [dict(id="K01", models={"u": dict(terms=[dict(expr="u")])})]
        policy = dict(
            selected="K01", banned_terms=[dict(field="u", expression="u", reason="remove")]
        )
        self.assertIsNone(valid_choice(policy, rows))

    def test_posthoc_metrics_on_all_recorded_equations(self):
        from gem_gss.evaluation import evaluate_result

        published = C.read(C.ASSETS / "published_records.json")["standard"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "final_result.json"
            for key, result in published.items():
                with self.subTest(case=key):
                    C.dump(path, result)
                    metric = evaluate_result(key, path)
                    self.assertEqual(metric["jaccard"], 1.0)
                    self.assertEqual(metric["recovery"], 1.0)

    def test_twenty_round_mock_search_resume_and_terminal(self):
        from gem_gss import search, terminal

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            case = toy_case(folder)
            graph = Graph(["u"], ["x"])
            # Enough dictionary content to exercise the real >=50% compression
            # gate without weakening production checks for a tiny toy prompt.
            for expr in ["u", "PD(u,(1,))", "PD(u,(2,))"] + [
                f"Sin({n}*x)*u" for n in range(1, 601)
            ]:
                graph.add(expr, "test")
            snapshot = graph.export()
            C.dump(folder / "protocol.json", dict(case=case, test_only=True))
            calls = []

            def mock_once(prompt, output, number, **kwargs):
                calls.append((str(output), number))
                if output.parent.name == "final_selection":
                    rows = C.read(output.parents[1] / "candidates.json")
                    policy = dict(selected=rows[0]["id"], banned_terms=[])
                else:
                    policy = dict(
                        tests={},
                        edits={},
                        new_nodes=(
                            [dict(id="new1", expr="u*(1-u)", field="u")] if number == 1 else []
                        ),
                    )
                record = dict(
                    http_attempts=1,
                    status="accepted",
                    usage={},
                    policy=policy,
                    simulated_no_network=True,
                )
                C.dump(output / f"round{number:02d}_record.json", record)
                return policy, record

            read = C.read

            def no_truth(path):
                self.assertNotIn("evaluation_references", str(path))
                self.assertNotIn("published_records", str(path))
                return read(path)

            with patch.object(C, "initial_graph", return_value=snapshot), patch.object(
                C, "read", side_effect=no_truth
            ), patch.object(search, "once", side_effect=mock_once), patch.object(
                terminal, "once", side_effect=mock_once
            ), patch.object(
                api.requests, "post", side_effect=AssertionError("Network forbidden")
            ), redirect_stdout(
                io.StringIO()
            ):
                result = search.run(case, folder)
                self.assertEqual(result["completed_rounds"], 20)
                self.assertEqual(result["initial_root_attempt_coverage"], {"u": 1.0})
                before = len(calls)
                search.run(case, folder)
                self.assertEqual(len(calls), before)
                final = terminal.run(case, folder)
                self.assertLessEqual(final["candidate_count"], 10)
                self.assertTrue(final["ban_consistency_passed"])
                self.assertEqual(len(calls), 21)
                terminal.run(case, folder)
                self.assertEqual(len(calls), 21)


if __name__ == "__main__":
    unittest.main()
