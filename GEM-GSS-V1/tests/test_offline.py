"""Release integrity, numerical regressions and offline search/resume contracts."""

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gem_gss import core as C
from gem_gss import search
from gem_gss.builder import Graph, build, original_constructor
from gem_gss.chain_numerics import ChainData
from gem_gss.compression import make_prompt, search_packet
from gem_gss.entry import (
    active_source_hashes,
    check_release,
    manifest,
    validate_backend,
)
from gem_gss.final_selection import ASSESSMENT_KEYS, selection_prompt
from gem_gss.fitting import fit
from gem_gss.graph import Registry
from gem_gss.memory import Archive
from gem_gss.wire import (
    decode_blocks,
    decode_search,
    decode_templates,
    encode_blocks,
    encode_search,
    encode_templates,
    expand_response_expressions,
)

ROOT = Path(__file__).resolve().parents[1]


def observation(folder, dimensions=1, fields=("u",), **updates):
    coordinates = list("xyz"[:dimensions])
    axes = [np.linspace(-1, 1, 9) for _ in coordinates]
    t = np.linspace(0, 1, 9)
    mesh = np.meshgrid(*axes, t, indexing="ij")
    values = 0.1 * mesh[0] + 0.03 * mesh[1] ** 2 if dimensions > 1 else 1 + 0.25 * mesh[0]
    values = values + 2 * mesh[-1]
    arrays = dict(zip(coordinates, axes), t=t, **{f: values.copy() for f in fields})
    arrays.update(updates)
    path = Path(folder) / "observations.npz"
    np.savez(path, **arrays)
    graph = Graph(fields, coordinates)
    for expression in [*fields, "1"]:
        graph.add(expression, "offline_test")
    graph_path = Path(folder) / "graph.json"
    C.dump(graph_path, graph.export())
    return dict(
        id="OFFLINE",
        fields=list(fields),
        coordinates=coordinates,
        prior="Discover the observed temporal rate.",
        metadata={},
        data_path=str(path),
        data_sha256=C.sha(path),
        graph_path=str(graph_path),
    )


def final_policy(folder):
    cards = C.read(folder.parent / "checkpoint.json")["pending"]["cards"]
    return dict(
        selected={
            f: min(rows, key=lambda row: row["normalized_MSE"])["candidate"]
            for f, rows in cards.items()
        },
        assessment={f: {key: "Offline mock assessment." for key in ASSESSMENT_KEYS} for f in cards},
    )


class OfflineRelease(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.original_settings = C.SETTINGS.copy()
        self.addCleanup(lambda: (C.SETTINGS.clear(), C.SETTINGS.update(self.original_settings)))
        C.SETTINGS.update(
            rounds=20,
            history_fits_per_round=1,
            coverage_schedule="global_1pct",
            minimum_test_fraction=0.1,
            compact_prompt=True,
            numerical_backend="reference",
        )
        self.network = patch(
            "requests.sessions.Session.request",
            side_effect=AssertionError("No network in offline tests"),
        )
        self.network.start()
        self.addCleanup(self.network.stop)

    def test_release_integrity_and_single_source(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(check_release()["modified"])
        import gem_gss

        self.assertTrue(gem_gss.__version__.startswith("0.4.9"))
        self.assertFalse((ROOT / "gem_gss/versions").exists())
        self.assertEqual(active_source_hashes(), manifest()["source_sha256"])
        self.assertFalse(manifest()["results_included"])
        self.assertFalse(manifest()["observation_data_included"])

    def test_all_eight_frozen_graphs(self):
        graphs = C.bundled_graphs()
        self.assertEqual(len(graphs), 8)
        for case in manifest()["cases"]:
            actual = hashlib.sha256(
                json.dumps(graphs[case["id"]], sort_keys=True).encode()
            ).hexdigest()
            self.assertEqual(actual, case["graph_sha256"])

    def test_cli_has_one_profile_and_consistent_defaults(self):
        for arguments in [
            ["list"],
            ["--data-dir", str(self.folder), "list"],
            ["list", "--data-dir", str(self.folder)],
            ["run", "--help"],
        ]:
            result = subprocess.run(
                [sys.executable, "-B", "-m", "gem_gss", *arguments],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertNotIn("--profile", result.stdout)
            self.assertNotIn("--resume-backend-transition", result.stdout)
        bad = subprocess.run(
            [sys.executable, "-B", "-m", "gem_gss", "list", "--profile", "pde047"],
            cwd=ROOT,
            capture_output=True,
        )
        self.assertNotEqual(bad.returncode, 0)
        validate_backend("compiled_chain_2d", dict(fields=["u"], coordinates=["x", "y"]))
        with self.assertRaises(ValueError):
            validate_backend("compiled_chain_2d", dict(fields=["u"], coordinates=["x"]))

    def test_twenty_round_loop_and_completed_resume(self):
        case = observation(self.folder)
        calls = []

        def fake_once(prompt, folder, number):
            calls.append(number)
            policy = final_policy(folder) if number == 20 else {}
            record = dict(status="accepted", round=number, http_attempts=0, policy=policy)
            C.dump(folder / f"round{number:02d}_record.json", record)
            return policy, record

        with (
            patch.object(search, "once", side_effect=fake_once),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = search.run(case, self.folder / "run")
            self.assertEqual(result["completed_rounds"], 20)
            self.assertEqual(result["total_API_calls"], 0)
            self.assertLess(result["equations"]["u"]["MSE"], 1e-20)
            self.assertEqual(result["terminal_fit_calls"], 0)
            self.assertEqual(sum(result["fit_breakdown"].values()), result["fit_calls"])
            self.assertEqual(search.run(case, self.folder / "run"), result)
            self.assertEqual(calls, list(range(21)))

    def test_resume_pending_rounds_without_repeating_fits_or_requests(self):
        from gem_gss.api import once as cached_once

        C.SETTINGS["rounds"] = 3
        case = observation(self.folder)
        out = self.folder / "resume"
        http_calls, interrupted = [], set()

        def fake_once(prompt, folder, number):
            # Exercise the real persistent HTTP gate, with a local transport.
            def transport(*args, **kwargs):
                http_calls.append(number)
                policy = final_policy(folder) if number == 3 else {}
                from types import SimpleNamespace

                return SimpleNamespace(
                    status_code=200,
                    json=lambda: dict(
                        choices=[dict(message=dict(content=json.dumps(policy)))],
                        model="offline",
                    ),
                )

            result = cached_once(prompt, folder, number, transport=transport)
            if number not in interrupted:
                interrupted.add(number)
                raise InterruptedError("Simulated interruption after paid response was persisted")
            return result

        fit_counts = {}
        with (
            patch.object(search, "once", side_effect=fake_once),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            for number in range(4):
                with self.assertRaises(InterruptedError):
                    search.run(case, out)
                cp = C.read(out / "checkpoint.json")
                self.assertEqual(cp["pending"]["round"], number)
                fit_counts[number] = cp["counters"]["fit_calls"]
                restored = search.Search(case, out)
                self.assertEqual(restored.data.fit_calls, fit_counts[number])
                if number:
                    with patch.object(
                        restored,
                        "fit_round",
                        side_effect=AssertionError("Pending fits must not repeat"),
                    ):
                        restored.plan_round(number)
            result = search.run(case, out)
        self.assertEqual(http_calls, [0, 1, 2, 3])
        self.assertEqual(result["total_API_calls"], 4)
        self.assertEqual(result["fit_calls"], fit_counts[3])

    def test_failed_final_selection_keeps_frozen_working_set(self):
        C.SETTINGS["rounds"] = 1
        case = observation(self.folder)
        out = self.folder / "unavailable"

        def fake_once(prompt, folder, number):
            return {}, dict(status="failed_no_retry" if number else "accepted", http_attempts=0)

        with patch.object(search, "once", side_effect=fake_once):
            with self.assertRaisesRegex(RuntimeError, "Final LLM unavailable"):
                search.run(case, out)
        cp = C.read(out / "checkpoint.json")
        self.assertEqual(cp["pending"]["phase"], "final_selection")
        self.assertIsNone(cp["final_selection"])
        self.assertFalse((out / "final_result.json").exists())

    def test_compiled_polynomial_derivative_matches_reference(self):
        case = observation(self.folder, dimensions=2)
        data = ChainData(case, prepare=False)
        node = dict(expr="PD((a*(u*u)),(1,0))", params=["a"])
        actual, expected = data.feature(node, [0.7]), data.reference.feature(node, [0.7])
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
        self.assertEqual(data.stats["singular_or_nonfinite_original_fallbacks"], 0)

    def test_cross_zero_cuberoot_and_singular_derivative(self):
        x, y, t = np.meshgrid(
            np.linspace(-1, 1, 9),
            np.linspace(-1, 1, 9),
            np.linspace(0, 1, 9),
            indexing="ij",
        )
        case = observation(self.folder, dimensions=2, u=x + 0.1 * y + 0.01 * t)
        data = ChainData(case, prepare=False)
        node = dict(expr="CubeRoot(a*u)", params=["a"])
        np.testing.assert_allclose(data.feature(node, [0.7]), data.reference.feature(node, [0.7]))
        self.assertEqual(data.stats["singular_or_nonfinite_original_fallbacks"], 0)
        node = dict(expr="PD(CubeRoot(a*u),(1,0))", params=["a"])
        np.testing.assert_allclose(data.feature(node, [0.7]), data.reference.feature(node, [0.7]))
        self.assertEqual(data.stats["singular_or_nonfinite_original_fallbacks"], 1)
        node = dict(expr="a/u", params=["a"])
        with self.assertRaises(ValueError):
            data.feature(node, [0.7])
        self.assertEqual(data.stats["singular_or_nonfinite_original_fallbacks"], 2)

    def test_mesh_validation_survives_python_optimization(self):
        case = observation(self.folder, fields=("u", "v"), v=np.zeros((8, 9)))
        code = (
            "from gem_gss.core import Data; import json; Data(json.loads("
            + repr(json.dumps(case))
            + "))"
        )
        for flags in [[], ["-O"]]:
            result = subprocess.run(
                [sys.executable, *flags, "-B", "-c", code],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ValueError", result.stderr)
            self.assertIn("Field v", result.stderr)
        for values in [
            dict(x=np.linspace(1, -1, 9)),
            dict(x=np.array([0, 1, 2, 3, 4, 5, 6, 7, 9])),
            dict(t=np.zeros(9)),
            dict(u=np.full((9, 9), np.nan)),
        ]:
            with self.subTest(values=list(values)):
                with self.assertRaises(ValueError):
                    C.Data(observation(self.folder, **values))

    def test_three_dimensional_online_construction(self):
        graph = Graph(["u"], ["x", "y", "z"])
        original_constructor(graph)
        expressions = {node["expr"] for node in graph.nodes if node["candidate"]}
        self.assertIn("PD((u**2),(0, 0, 1))", expressions)
        self.assertIn("PD((u**2),(1, 1, 1))", expressions)
        self.assertTrue(graph.stats()["acyclic"])
        # Exercise every stage with the production graph recipe.
        _, expanded = build({}, ["u"], ["x", "y", "z"])
        self.assertTrue(expanded["statistics"]["acyclic"])
        self.assertGreater(expanded["statistics"]["candidate_roots"], 0)
        self.assertFalse(expanded["reference_access"])

    def test_lossless_prompts_and_final_selection_unchanged(self):
        for dimensions in (1, 2, 3):
            with self.subTest(dimensions=dimensions):
                case = observation(self.folder, dimensions=dimensions)
                data, registry = C.Data(case), Registry(case)
                archive = Archive(data, "u", registry)
                archive.add(fit(data, "u", [], registry), None, dict(operation="root"), 0)
                archives = {"u": archive}
                sampled = sorted(registry.graph.candidates)
                packet = search_packet(registry, archives, sampled, {}, [], 0, {})
                self.assertEqual(decode_search(encode_search(packet)), packet)
                self.assertEqual(decode_templates(encode_templates(packet)), packet)
                self.assertEqual(decode_blocks(encode_blocks(packet)), packet)
                text, output = make_prompt(case, registry, archives, sampled, {}, [], 0, {})
                audit = output["compression_audit"]
                self.assertLessEqual(audit["compressed_tokens"], audit["uncompressed_tokens"])
                self.assertEqual(
                    hashlib.sha256(text.encode()).hexdigest(),
                    audit["sent_prompt_sha256"],
                )
                final_text, final_packet = make_prompt(case, registry, archives, [], {}, [], 20, {})
                self.assertEqual(final_text, selection_prompt(final_packet["candidates"]))

    def test_search_matches_unified_049_reference_fixture(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = search_regression()
        self.assertEqual(result["fit_calls"], 147)
        self.assertEqual(
            result["fit_breakdown"],
            dict(
                initial_empty=2,
                coverage=8,
                history=12,
                online_revision=125,
                ordinary_local=0,
                terminal=0,
            ),
        )
        expected = {
            "u": (
                ["1", "(u*v)", "PD(u,(1,))"],
                [0.7789537642004215, 0.2680352448539523, -0.22781698318875257],
                6.820373575495723e-05,
            ),
            "v": (
                ["1", "(u*v)", "PD(u,(1,))", "(u**3)"],
                [
                    0.167364514250441,
                    0.13110472232494258,
                    0.2930747003470287,
                    -0.004011902547375954,
                ],
                1.0870473994164118e-05,
            ),
        }
        for field, (expressions, coefficients, mse) in expected.items():
            equation = result["equations"][field]
            self.assertEqual([term["expr"] for term in equation["terms"]], expressions)
            np.testing.assert_allclose(
                [term["coefficient"] for term in equation["terms"]],
                coefficients,
                rtol=1e-7,
                atol=1e-10,
            )
            np.testing.assert_allclose(equation["MSE"], mse, rtol=1e-7)

    def test_bundled_prompt_payloads_roundtrip_and_cost(self):
        import random
        from types import SimpleNamespace

        baselines = dict(
            T01=7594, T02=13371, T03=11034, T04=6740, T05=18487, T2D=41643, T3D=77245, TSC1=18295
        )
        decoders = {
            "moderate-reversible-v1": decode_search,
            "lossless-expression-templates-v2": decode_templates,
            "lossless-node-blocks-v1": decode_blocks,
        }
        for case in C.read(C.ASSETS / "cases.json"):
            with self.subTest(case=case["id"]):
                nodes = {n["id"]: n for n in C.bundled_graphs()[case["id"]]["nodes"]}
                ids = sorted(mid for mid, node in nodes.items() if node["candidate"])
                random.Random(C.SETTINGS["seed"]).shuffle(ids)
                sampled = ids[: len(ids) // 10]
                state = dict(id="S000000", model=dict(terms=[], fit_mse=1.0, complexity=0))
                archives = {
                    f: SimpleNamespace(
                        field=f,
                        data=SimpleNamespace(scale={f: 1.0}, y={f: np.ones(1)}),
                        view=lambda: [state],
                    )
                    for f in case["fields"]
                }
                _, packet = make_prompt(
                    case, SimpleNamespace(nodes=nodes), archives, sampled, {}, [], 0, {}
                )
                audit = packet.pop("compression_audit")
                wire = packet.pop("wire_packet")
                self.assertEqual(decoders[audit["codec"]](wire), packet)
                self.assertLess(audit["compressed_tokens"], baselines[case["id"]])

    def test_api_token_gate_and_failed_request_are_not_retried(self):
        from gem_gss.api import once
        from gem_gss.compression import tokens

        prompt = "A self-contained offline prompt"
        folder = self.folder / "api"
        packet = dict(
            compression_audit=dict(
                sent_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                compressed_tokens=tokens(prompt),
                max_input_tokens=tokens(prompt),
            )
        )
        C.dump(self.folder / "inputs/00_graph.json", packet)
        with (
            patch("gem_gss.api.credentials", return_value="offline"),
            patch("requests.post", side_effect=TimeoutError) as post,
        ):
            _, first = once(prompt, folder, 0)
            _, second = once(prompt, folder, 0)
            self.assertEqual(first["status"], "failed_no_retry")
            self.assertEqual(first, second)
            self.assertEqual(post.call_count, 1)
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                once(prompt + " changed", folder, 0)
            self.assertEqual(post.call_count, 1)
        packet["compression_audit"]["max_input_tokens"] -= 1
        C.dump(self.folder / "inputs/00_graph.json", packet)
        with self.assertRaisesRegex(RuntimeError, "token gate"):
            once(prompt, folder, 0)

    def test_reply_template_expansion_preserves_multindices(self):
        raw = dict(
            new_nodes=[dict(id="new1", expr="PD(@E1,(0,0,2))")],
            revisions={"u": [dict(new="PD(@E1,(1,0,0))")]},
        )
        expanded, audit = expand_response_expressions(
            raw, dict(expression_templates={"@E1": "Exp(a*u)"})
        )
        self.assertEqual(expanded["new_nodes"][0]["expr"], "PD(Exp(a*u),(0,0,2))")
        self.assertEqual(expanded["revisions"]["u"][0]["new"], "PD(Exp(a*u),(1,0,0))")
        self.assertEqual(len(audit), 2)
        self.assertIn("@E1", raw["new_nodes"][0]["expr"])


def search_regression():
    C.SETTINGS.update(
        rounds=4,
        history_fits_per_round=4,
        coverage_schedule="global_1pct",
        minimum_test_fraction=0.1,
        compact_prompt=True,
        numerical_backend="reference",
    )
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        x = np.linspace(-1, 1, 9)
        t = np.linspace(0, 1, 9)
        u = (
            1
            + 0.3 * x[:, None]
            + 0.2 * x[:, None] ** 2
            + (0.8 + 0.1 * x[:, None]) * t[None, :]
            + 0.2 * t[None, :] ** 2
        )
        v = 0.5 + 0.2 * x[:, None] + (0.3 + 0.2 * x[:, None]) * t[None, :] + 0.1 * t[None, :] ** 2
        path = root / "data.npz"
        np.savez(path, x=x, t=t, u=u, v=v)
        graph = Graph(["u", "v"], ["x"])
        for text in [
            "1",
            "u",
            "v",
            "u*u",
            "u*v",
            "x",
            "t",
            "PD(u,(1,))",
            "PD(v,(1,))",
            "PD(u,(2,))",
            "Sin(a*x)",
            "Exp(a*u)",
        ]:
            graph.add(text, "offline")
        gp = root / "graph.json"
        C.dump(gp, graph.export())
        case = dict(
            id="COMPARE",
            fields=["u", "v"],
            coordinates=["x"],
            prior="Discover rates.",
            metadata={},
            data_path=str(path),
            data_sha256=C.sha(path),
            graph_path=str(gp),
        )

        def fake_once(prompt, folder, number):
            packet = C.read(folder.parent / "inputs" / f"{number:02d}_graph.json")
            policy = {}
            if number == 4:
                cards = packet["candidates"]
                policy = dict(
                    selected={
                        f: min(rows, key=lambda r: r["normalized_MSE"])["candidate"]
                        for f, rows in cards.items()
                    },
                    assessment={f: {k: "offline" for k in ASSESSMENT_KEYS} for f in cards},
                )
            else:
                policy["tests"] = {
                    f: [
                        dict(
                            parent_state=rows[0]["state"],
                            keep=[],
                            candidates=packet["sampled_ids"][:1],
                            joint=False,
                        )
                    ]
                    for f, rows in packet["working_set"].items()
                }
                if number == 1:
                    policy["new_nodes"] = [dict(id="new1", expr="u**3", reason="test")]
                    policy["revisions"] = {
                        "u": [
                            dict(
                                parent_state=packet["working_set"]["u"][0]["state"],
                                keep=[],
                                add=["new1"],
                                reason="test edit",
                            )
                        ]
                    }
            record = dict(status="accepted", round=number, http_attempts=0, policy=policy)
            C.dump(folder / f"round{number:02d}_record.json", record)
            return policy, record

        with (
            patch.object(search, "once", side_effect=fake_once),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = search.run(case, root / "run")
        return result


if __name__ == "__main__":
    unittest.main()
