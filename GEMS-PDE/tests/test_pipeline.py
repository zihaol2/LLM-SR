"""Small synthetic end-to-end run; mock LLM only, real numerical fitting."""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.io import savemat
from gems_pde import config, common as C
from gems_pde.cli import main
from gems_pde.io import read
from gems_pde.io import dump


class PipelineTests(unittest.TestCase):
    def test_fresh_search_correction_baseline_resume_and_postfreeze_evaluation(self):
        original_settings = copy.deepcopy(config.SETTINGS)
        original_tasks = config.TASKS
        original_file = config.TASK_FILE
        original_dir = config.RUN_DIR
        original_offline = config.OFFLINE
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                x = np.linspace(-np.pi, np.pi, 18)
                t = np.linspace(0, 2, 31)
                savemat(
                    root / "data.mat",
                    {"x": x, "t": t, "u": 2 + np.cos(x[:, None]) * np.exp(-0.2 * t[None, :])},
                )
                (root / "prior.txt").write_text(
                    "[D1] A smooth scalar profile evolves.\n[V1] u is state, x is space, t is time.\n"
                )
                (root / "tasks.json").write_text(
                    json.dumps(
                        [
                            {
                                "id": "S01",
                                "name": "synthetic",
                                "data": "data.mat",
                                "prior": "prior.txt",
                            }
                        ]
                    )
                )
                config.SETTINGS.update(
                    train_times=12,
                    validation_times=6,
                    proposed_decisions=2,
                    baseline_samples_per_task=2,
                    max_nfev_full=4,
                    max_nfev_screen=2,
                    max_terms=3,
                )
                payload = {
                    "L0": [{"id": "u", "expr": "u"}, {"id": "diff", "expr": "D(u,2)"}],
                    "G0": [],
                }

                def policy(prompt, path, kind=None):
                    return (
                        payload
                        if kind == "initial"
                        else {"focus_states": {}, "priority_nodes": {}, "edits": {}}
                    )

                raw = "def equation(x,u,dx,params,d_dx):\n    return params[0]*d_dx(u,order=2)\n"
                args = [
                    "--tasks-file",
                    str(root / "tasks.json"),
                    "--run-dir",
                    str(root / "run"),
                    "--offline",
                ]
                with patch("gems_pde.search.call_json", side_effect=policy) as json_call, patch(
                    "gems_pde.baseline.call_text", return_value=raw
                ) as text_call, patch("gems_pde.refine.MAX_SWEEPS", 1), patch(
                    "gems_pde.api.requests.post", side_effect=AssertionError("No network")
                ), redirect_stdout(
                    io.StringIO()
                ):
                    main(["run"] + args)
                    self.assertEqual(json_call.call_count, 3)
                    self.assertEqual(text_call.call_count, 2)
                    # A frozen resume performs neither new LLM returns nor numerical search.
                    main(["run"] + args)
                    self.assertEqual(json_call.call_count, 3)
                    self.assertEqual(text_call.call_count, 2)
                for stage in ("proposed42", "proposed", "baseline_100"):
                    result = read(root / "run" / stage / "S01" / "frozen_result.json")
                    self.assertIs(result["reference_access"], False)
                    self.assertTrue(np.isfinite(result["MSE"]))
                # Simulate interruption after rejection but before marking the field complete.
                from gems_pde import refine

                checkpoint = root / "run" / "proposed" / "S01" / "checkpoint.json"
                state = read(checkpoint)
                source_archive = read(root / "run" / "proposed42" / "S01" / "equation_archive.json")
                state["models"]["u"] = next(
                    s["model"]
                    for s in source_archive["archives"]["u"]
                    if s["id"] == source_archive["best_states"]["u"]
                )
                state["history"]["u"] = [{"accepted": False}]
                state["completed_fields"] = []
                dump(checkpoint, state)
                (checkpoint.parent / "frozen_result.json").unlink()
                with patch(
                    "gems_pde.refine.sweep", side_effect=AssertionError("Rejected route must stop")
                ), redirect_stdout(io.StringIO()):
                    refine.run(0)
                references = root / "references.json"
                references.write_text(json.dumps({"synthetic": {"u": "0.2*D(u,2)"}}))
                from gems_pde.evaluation import evaluate

                result = evaluate(root / "run", references, root / "evaluation", [0])
                self.assertEqual(len(result["rows"]), 3)
                self.assertEqual(result["summary"]["baseline_100"]["mean_jaccard"], 1.0)
        finally:
            config.SETTINGS.clear()
            config.SETTINGS.update(original_settings)
            config.TASKS = original_tasks
            config.TASK_FILE = original_file
            config.RUN_DIR = original_dir
            config.OFFLINE = original_offline
            C.ROOT = original_dir
            C.NAMES = config.names()


if __name__ == "__main__":
    unittest.main()
