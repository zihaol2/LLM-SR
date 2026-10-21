"""D1b/V1 LLM-PDESR Pareto core with the shared numerical adapter."""

import json
import time
import numpy as np
from . import common as C, config
from . import baseline_adapter as B
from .api import call_text
from .run_guard import guard_task


def run(index):
    task = config.TASKS[index].id
    guard_task("baseline_100", index)
    out = C.ROOT / "baseline_100" / task
    out.mkdir(parents=True, exist_ok=True)
    if (out / "frozen_result.json").exists():
        return json.loads((out / "frozen_result.json").read_text(encoding="utf-8"))
    data = C.Data(C.NAMES[index])
    prior = config.prior_for(index)
    meta = {
        "fields": data.fields,
        "shape": list(next(iter(data.full.values())).shape),
        "x_range": [float(data.x.min()), float(data.x.max())],
        "field_ranges": {f: [float(v.min()), float(v.max())] for f, v in data.full.items()},
    }
    program = B.template(prior, meta, data.fields)
    (out / "weak_prior_template.py").write_text(str(program), encoding="utf-8")
    db = B.buffer.ExperienceBuffer(B.ExperienceBufferConfig(), program, "equation")
    archive = (
        json.loads((out / "candidate_archive.json").read_text(encoding="utf-8"))
        if (out / "candidate_archive.json").exists()
        else []
    )
    history = (
        json.loads((out / "history.json").read_text(encoding="utf-8"))
        if (out / "history.json").exists()
        else []
    )

    def register(item):
        fn = B.code_manipulation.Function(
            "equation",
            program.functions[0].args,
            item["body"],
            mse=item["training_MSE"],
            complexity=item["complexity"],
        )
        db.register_program(fn, 0, {"observations": -item["training_MSE"]})

    for item in archive:
        register(item)

    def evaluate(raw, iteration):
        started = time.time()
        try:
            proposed, body = B.extract(raw, data.fields)
            fitted = {
                f: C.fit(data, f, nodes, shared=True, full_budget=True)
                for f, nodes in proposed.items()
            }
            if any(v is None for v in fitted.values()):
                raise ValueError("no valid numerical fit")
            train = float(np.mean([m["train_mse"] for m in fitted.values()]))
            score = float(np.mean([m["score"] for m in fitted.values()]))
            complexity = sum(m["complexity"] for m in fitted.values())
            item = {
                "iteration": iteration,
                "body": body,
                "equations": fitted,
                "training_MSE": train,
                "score": score,
                "complexity": complexity,
            }
            archive.append(item)
            register(item)
            event = {
                "iteration": iteration,
                "ok": True,
                "validation_MSE": {f: m["validation_mse"] for f, m in fitted.items()},
            }
        except Exception as error:
            event = {
                "iteration": iteration,
                "ok": False,
                "error": type(error).__name__ + ": " + str(error),
            }
        event["seconds"] = time.time() - started
        history.append(event)
        C.dump(out / "history.json", history)
        C.dump(out / "candidate_archive.json", archive)
        print(
            task,
            "BASELINE",
            iteration,
            "/",
            C.SETTINGS["baseline_samples_per_task"],
            event.get("validation_MSE", event.get("error")),
            flush=True,
        )

    if not history:
        evaluate(str(program.functions[0]), 0)
    start = max(v["iteration"] for v in history) + 1
    for iteration in range(start, C.SETTINGS["baseline_samples_per_task"] + 1):
        prompt = (
            "Discover a mathematical equation from observations. Return only the complete Python equation function.\n\n"
            + db.get_prompt().code
        )
        raw = call_text(prompt, out / "api" / f"call{iteration:03d}.txt")
        evaluate(raw, iteration)
    best = min(archive, key=lambda m: m["score"])
    final = {f: C.final_equation(data, f, m) for f, m in best["equations"].items()}
    result = {
        "task": C.NAMES[index],
        "anonymous_id": task,
        "method": "LLM-PDESR Pareto core; D1/V1; shared canonical complexity and fitter",
        "model": C.SETTINGS["model"],
        "LLM_samples": C.SETTINGS["baseline_samples_per_task"],
        "valid_generated_candidates": sum(h["ok"] for h in history if h["iteration"] > 0),
        "seed_equations_excluded_from_sample_count": 1,
        "equations": final,
        "selected_iteration": best["iteration"],
        "MSE": float(np.mean([v["MSE"] for v in final.values()])),
        "complexity": best["complexity"],
        "reference_access": False,
        "frozen_at_unix": time.time(),
    }
    C.dump(out / "frozen_result.json", result)
    print("FROZEN BASELINE", task, result["MSE"], flush=True)
    return result
